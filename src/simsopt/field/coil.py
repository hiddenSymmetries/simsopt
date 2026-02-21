from math import pi
import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import Derivative
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve
import simsoptpp as sopp

__all__ = ['Coil', 'RegularizedCoil', 'CircularRegularizedCoil', 'RectangularRegularizedCoil',
           'Current', 'coils_via_symmetries',
           'load_coils_from_makegrid_file',
           'apply_symmetries_to_currents', 'apply_symmetries_to_curves',
           'coils_to_makegrid', 'coils_to_focus', 'coils_to_vtk'
           ]


class Coil(sopp.Coil, Optimizable):
    """
    Represents a magnetic coil as a combination of a geometric curve and an electric current.

    This class combines a :class:`~simsopt.geo.curve.Curve` and a :class:`Current` object, and 
    is used as input for :class:`~simsopt.field.biotsavart.BiotSavart` field calculations. 

    Args:
        curve (simsopt.geo.curve.Curve) : The geometric curve describing the coil shape.
        current (Current) : The current object describing the electric current in the coil.
    """

    def __init__(self, curve, current):
        self._curve = curve
        self._current = current
        sopp.Coil.__init__(self, curve, current)
        Optimizable.__init__(self, depends_on=[curve, current])

    def vjp(self, v_gamma, v_gammadash, v_current):
        r"""
        Compute the vector-Jacobian product,

        .. math::
            \frac{\partial \mathbf{\gamma}}{\partial \mathbf{x}}^T \mathbf{v}_\gamma + \frac{\partial \mathbf{\gamma'}}{\partial \mathbf{x}}^T \mathbf{v}_{\gamma'} + \frac{\partial \mathbf{I}}{\partial \mathbf{x}}^T \mathbf{v}_I

        where :math:`\mathbf{x}` are the degrees of freedom of the coil.

        Args:
            v_gamma (array, shape (n, 3)) : Vector w.r.t. :math:`\gamma`; same shape as curve gamma.
            v_gammadash (array, shape (n, 3)) : Vector w.r.t. :math:`\gamma'`; same shape as curve gammadash.
            v_current (array, shape (1,)) : Vector w.r.t. coil current (scalar).

        Returns:
            The vector-Jacobian product of the coil.
        """
        return self.curve.dgamma_by_dcoeff_vjp(v_gamma) \
            + self.curve.dgammadash_by_dcoeff_vjp(v_gammadash) \
            + self.current.vjp(v_current)

    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.curve.Curve.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.curve.Curve.plot()`

        Args:
            kwargs (dictionary): Additional keyword arguments.
        """
        return self.curve.plot(**kwargs)
    
class RegularizedCoil(Coil):
    """
    A coil with a model for its cross section. This cross section is used to compute the
    forces and torques on the coil.
    
    Args:
        curve (simsopt.geo.curve.Curve) : The geometric curve describing the coil shape.
        current (Current) : The current object describing the electric current in the coil.
        regularization (float) : The regularization parameter for the coil cross section.
    """
    def __init__(self, curve, current, regularization):
        self.regularization = regularization
        Coil.__init__(self, curve, current)
    
    @staticmethod
    def _coil_force_pure(B, I, t):
        r"""
        Compute the pointwise Lorentz force per unit length on a coil with n quadrature points, in Newtons/meter. 

        .. math::
            dF/d\ell = I \vec{t} \times \vec{B}

        where :math:`\vec{t}` is the tangent vector to the coil curve,
        :math:`\vec{B}` is the magnetic field at the quadrature points,
        :math:`I` is the coil current.

        Args:
            B (array, shape (n,3)): Array of magnetic field.
            I (float): Coil current.
            t (array, shape (n,3)): Array of coil tangent vectors.
        Returns:
            array (shape (n,3)): Array of force per unit length.
        """
        import jax.numpy as jnp
        return jnp.cross(I * t, B)
    
    def B_regularized(self):
        """Calculate the regularized field on this coil following the Landreman and Hurwitz method.
        
        Returns:
            array (shape (n,3)): The regularized field on the coil.
        """
        from .selffield import B_regularized_pure
        return B_regularized_pure(
            self.curve.gamma(),
            self.curve.gammadash(),
            self.curve.gammadashdash(),
            self.curve.quadpoints,
            self._current.get_value(),
            self.regularization,
        )
    
    def self_force(self):
        """
        Compute the self-force per unit length of this coil, in Newtons/meter.
        
        Returns:
            array (shape (n,3)): Array of self-force per unit length.
        """
        I = self.current.get_value()
        gammadash = self.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        tangent = gammadash / gammadash_norm
        B = self.B_regularized()
        return self._coil_force_pure(B, I, tangent)
    
    def force(self, source_coils):
        r"""
        Compute the force per unit length on this coil from other coils, in Newtons/meter.
        
        .. math::
            dF_i/d\ell = I_i \vec{t_i} \times (\vec{B_{self}} + \vec{B_{mutual}})

        where :math:`\vec{t_i}` is the tangent vector to the ith coil curve,
        :math:`\vec{B_{self}}` is the self-field of the ith coil,
        :math:`\vec{B_{mutual}}` is the mutual field from the other coils.

        Args:
            source_coils (list of Coil or RegularizedCoil, shape (m,)): 
                List of coils contributing forces on this coil. 
                Can be a mix of Coil and RegularizedCoil objects.
        Returns:
            array (shape (n,3)): Array of forces per unit length along the coil curve.
        """
        from .biotsavart import BiotSavart
        gammadash = self.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        tangent = gammadash / gammadash_norm
        mutual_coils = [c for c in source_coils if c is not self]
        mutual_field = BiotSavart(mutual_coils).set_points(self.curve.gamma())
        B_mutual = mutual_field.B()
        mutualforce = np.cross(self.current.get_value() * tangent, B_mutual)
        selfforce = self.self_force()
        return (selfforce + mutualforce)
    
    def net_force(self, source_coils):
        r"""
        Compute the net forces on this coil from other coils, in Newtons. This is
        the integrated pointwise force per unit length dF_i/d\ell on the coil curve.

        .. math::
            F_net = \int (dF_i/d\ell) d\ell

        Args:
            source_coils (list of Coil or RegularizedCoil, shape (m,)): 
                List of coils contributing forces on this coil. 
                Can be a mix of Coil and RegularizedCoil objects.
        Returns:
            np.array (shape (3,)): Array of net forces.
        """
        Fi = self.force(source_coils)
        gammadash = self.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        net_force = np.sum(gammadash_norm * Fi, axis=0) / gammadash.shape[0]
        return net_force
    
    def torque(self, source_coils):
        r"""
        Compute the torques per unit length on this coil from other coils in Newtons 
        (note that the force is per unit length, so the force has units of Newtons/meter 
        and the torques per unit length have units of Newtons).

        .. math::
            dT_i/d\ell = (\gamma_i - c_i) \times (dF_i/d\ell)

        where :math:`\gamma_i` is the position vector of the ith coil curve, 
        :math:`c_i` is the centroid of the ith coil curve,
        :math:`dF_i/d\ell` is the pointwise force per unit length on the ith coil curve.

        Args:
            source_coils (list of Coil or RegularizedCoil, shape (m,)): 
                List of coils contributing torques on this coil. 
                Can be a mix of Coil and RegularizedCoil objects.
        Returns:
            np.array (shape (n,3)): Array of torques per unit length along the coil curve.
        """
        gamma = self.curve.gamma()
        center = self.curve.centroid()
        return np.cross(gamma - center, self.force(source_coils))
    
    def net_torque(self, source_coils):
        r"""
        Compute the net torques on this coil from other coils, in Newton-meters. This is
        the integrated pointwise torque per unit length on the coil curve.

        .. math::
            T_net = \int dT_i/d\ell d\ell

        Args:
            source_coils (list of Coil or RegularizedCoil, shape (m,)): 
                List of coils contributing torques on this coil. 
                Can be a mix of Coil and RegularizedCoil objects.
        Returns:
            np.array (shape (3,)): Array of net torques.
        """
        Ti = self.torque(source_coils)
        gammadash = self.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        net_torque = np.sum(gammadash_norm * Ti, axis=0) / gammadash.shape[0]
        return net_torque


class CircularRegularizedCoil(RegularizedCoil):
    """
    A coil with a circular cross section. The regularization parameter is computed
    from the radius during initialization.
    
    Args:
        curve (simsopt.geo.curve.Curve) : The geometric curve describing the coil shape.
        current (Current) : The current object describing the electric current in the coil.
        a (float) : The radius of the circular cross-section.
    """
    def __init__(self, curve, current, a):
        from .selffield import regularization_circ
        regularization = regularization_circ(a)
        self.a = a
        RegularizedCoil.__init__(self, curve, current, regularization)


class RectangularRegularizedCoil(RegularizedCoil):
    """
    A coil with a rectangular cross section. The regularization parameter is computed
    from the width and height during initialization.
    
    Args:
        curve (simsopt.geo.curve.Curve) : The geometric curve describing the coil shape.
        current (Current) : The current object describing the electric current in the coil.
        a (float) : The width of the rectangular cross-section.
        b (float) : The height of the rectangular cross-section.
    """
    def __init__(self, curve, current, a, b):
        from .selffield import regularization_rect
        regularization = regularization_rect(a, b)
        self.a = a
        self.b = b
        RegularizedCoil.__init__(self, curve, current, regularization)    

class CurrentBase(Optimizable):
    """
    Abstract base class for current objects that are optimizable.

    Args:
        kwargs (dictionary): Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        Optimizable.__init__(self, **kwargs)

    def __mul__(self, other):
        """
        Multiply the current object by a scalar.

        Args:
            other: The scalar to multiply the current object by.

        Returns:
            A new current object that is the product of the current object and the scalar.
        """
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __rmul__(self, other):
        """
        Multiply the current object by a scalar.

        Args:
            other: The scalar to multiply the current object by.

        Returns:
            A new current object that is the product of the current object and the scalar.
        """
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __truediv__(self, other):
        """
        Divide the current object by a scalar.

        Args:
            other: The scalar to divide the current object by.

        Returns:
            A new current object that is the quotient of the current object and the scalar.
        """
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, 1.0/other)

    def __neg__(self):
        """
        Negate the current value in the current object.

        Returns:
            A new current object that has the opposite sign of current.
        """
        return ScaledCurrent(self, -1.)

    def __add__(self, other):
        """
        Add two current objects.

        Returns:
            A new current object that is the sum of the current object and the other current object.
        """
        return CurrentSum(self, other)

    def __sub__(self, other):
        """
        Subtract two current objects.

        Returns:
            A new current object that is the difference of the current object and the other current object.
        """
        return CurrentSum(self, -other)

    # https://stackoverflow.com/questions/11624955/avoiding-python-sum-default-start-arg-behavior
    def __radd__(self, other):
        """
        Add two current objects. This allows sum() to work (the default start value is zero).

        Returns:
            A new current object that is the sum of the current object and the other current object.
        """
        if other == 0:
            return self
        return self.__add__(other)


class Current(sopp.Current, CurrentBase):
    """
    An optimizable object that wraps around a single scalar degree of freedom representing 
    an electric current.

    This class is used for the current in a coil, or in a set of coils constrained 
    to use the same current.

    Args:
        current (float) : Initial value of the current.
        dofs (array-like or None, optional) : Degrees of freedom for optimization. If None, uses the current value.
        kwargs (dictionary): Additional keyword arguments.
    """

    def __init__(self, current, dofs=None, **kwargs):
        sopp.Current.__init__(self, current)
        if dofs is None:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 x0=self.get_dofs(), **kwargs)
        else:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 dofs=dofs, **kwargs)

    def vjp(self, v_current):
        """
        Compute the Jacobian-vector product of the current function.

        Args:
            v_current (array, shape (1,)) : The vector to multiply the Jacobian with.

        Returns:
            The Jacobian-vector product of the current function.
        """
        return Derivative({self: v_current})

    @property
    def current(self):
        """
        Get the current value of the current object.

        Returns:
            The current value of the current object.
        """
        return self.get_value()


class ScaledCurrent(sopp.CurrentBase, CurrentBase):
    """
    Represents a current that is a scaled version of another current object 
    (Scales :mod:`Current` by a factor.). The 'scale' is not treated as a dof, so it is not optimized.
    The scaled current has value I = scale * I_0 where `I_0` is the 'current_to_scale'.

    Used, for example, to flip currents for stellarator symmetric coils.

    Args:
        current_to_scale (CurrentBase) : The current object to scale.
        scale (float) : The scaling factor.
        kwargs (dictionary): Additional keyword arguments.
    """

    def __init__(self, current_to_scale, scale, **kwargs):
        self.current_to_scale = current_to_scale
        self.scale = scale
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_to_scale], **kwargs)

    def vjp(self, v_current):
        """
        Compute the Jacobian-vector product of the current function.

        Args:
            v_current (array, shape (1,)) : The vector to multiply the Jacobian with.

        Returns:
            The Jacobian-vector product of the current function.
        """
        return self.scale * self.current_to_scale.vjp(v_current)

    def get_value(self):
        """
        Get the current value of the current object.

        Returns:
            The current value of the current object.
        """
        return self.scale * self.current_to_scale.get_value()

class CurrentSum(sopp.CurrentBase, CurrentBase):
    """
    Represents the sum of two :mod:`Current` objects.

    Used to enforce current constraints or combine currents in optimization.

    Args:
        current_a (CurrentBase) : First current object.
        current_b (CurrentBase) : Second current object.
    """

    def __init__(self, current_a, current_b):
        self.current_a = current_a
        self.current_b = current_b
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_a, current_b])

    def vjp(self, v_current):
        """
        Compute the Jacobian-vector product of the current function.

        Args:
            v_current (array, shape (1,)) : The vector to multiply the Jacobian with.

        Returns:
            The Jacobian-vector product of the current function.
        """
        return self.current_a.vjp(v_current) + self.current_b.vjp(v_current)

    def get_value(self):
        """
        Get the current value of the current object.

        Returns:
            The current value of the current object.
        """
        return self.current_a.get_value() + self.current_b.get_value()


def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    """
    Generate a list of curves by applying rotational and (optionally) stellarator symmetries.

    Take a list of ``n`` :mod:`simsopt.geo.curve.Curve`s and return ``n * nfp *
    (1+int(stellsym))`` :mod:`simsopt.geo.curve.Curve` objects obtained by
    applying rotations and flipping corresponding to ``nfp`` fold rotational
    symmetry and optionally stellarator symmetry.

    Args:
        base_curves (list) : List of base curves to replicate.
        nfp (int) : Number of field periods (rotational symmetry).
        stellsym (bool) : Whether to apply stellarator symmetry (flipping).

    Returns:
        curves (list) : List of curves with symmetries applied.
    """
    flip_list = [False, True] if stellsym else [False]
    curves = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_curves)):
                if k == 0 and not flip:
                    curves.append(base_curves[i])
                else:
                    rotcurve = RotatedCurve(base_curves[i], 2*pi*k/nfp, flip)
                    curves.append(rotcurve)
    return curves


def apply_symmetries_to_currents(base_currents, nfp, stellsym):
    """
    Generate a list of currents by applying rotational and (optionally) stellarator symmetries.
        
    Take a list of ``n`` :mod:`Current`s and return ``n * nfp * (1+int(stellsym))``
    :mod:`Current` objects obtained by copying (for ``nfp`` rotations) and
    sign-flipping (optionally for stellarator symmetry).

    Args:
        base_currents (list of Current) : List of base current objects to replicate.
        nfp (int) : Number of field periods (rotational symmetry).
        stellsym (bool) : Whether to apply stellarator symmetry (sign flip).

    Returns:
        currents (list of Current) : List of current objects with symmetries applied.
    """
    flip_list = [False, True] if stellsym else [False]
    currents = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_currents)):
                current = ScaledCurrent(base_currents[i], -1.) if flip else base_currents[i]
                currents.append(current)
    return currents

def coils_to_vtk(coils, filename, close=False, extra_data=None):
    """
    Export a list of Coil objects in VTK format, so they can be
    viewed using Paraview. This function requires the python package ``pyevtk``,
    which can be installed using ``pip install pyevtk``.

    Saves coil currents, net forces, net torques, and pointwise forces and torques.

    Args:
        coils (list): A python list of Coil objects.
        filename (str): Name of the file to write.
        close (bool): Whether to draw the segment from the last quadrature point back to the first.
        extra_data (dict): Additional data to save to the VTK file.
    """
    from simsopt.geo.curve import curves_to_vtk

    # get the curves and currents
    curves = [coil.curve for coil in coils]
    currents = [coil.current.get_value() for coil in coils]

    # get the number of points per curve
    if close:
        ppl = np.asarray([c.gamma().shape[0]+1 for c in curves])
    else:
        ppl = np.asarray([c.gamma().shape[0] for c in curves])
    ppl_cumsum = np.concatenate([[0], np.cumsum(ppl)])

    # get the current data, which is the same at every point on a given coil
    contig = np.ascontiguousarray
    pointData = {}
    data = np.concatenate([i*np.ones((ppl[i], )) for i in range(len(curves))])
    coil_data = np.zeros(data.shape)
    for i in range(len(currents)):
        coil_data[ppl_cumsum[i]:ppl_cumsum[i+1]] = currents[i]
    coil_data = np.ascontiguousarray(coil_data)
    pointData['I'] = coil_data
    pointData['I_mag'] = contig(np.abs(coil_data))

    if not isinstance(coils[0], RegularizedCoil):
        print("Warning: coils_to_vtk will not save forces and torques for coils that "
              "do not have a model for their cross section. Please use the RegularizedCoil class.")
    else:    
        net_forces = np.zeros((len(coils), 3))
        net_torques = np.zeros((len(coils), 3))
        coil_forces = np.zeros((data.shape[0], 3))
        coil_torques = np.zeros((data.shape[0], 3))
        for i, c in enumerate(coils):
            # get the pointwise forces and torques for the current coil
            coil_force_temp = c.force(coils)
            coil_torque_temp = c.torque(coils)

            # get the net forces and torques for the current coil, 
            # which is the same at every point on the coil
            net_forces[i, :] = c.net_force(coils)
            net_torques[i, :] = c.net_torque(coils)

            # if the curve is closed, add the first point to the end
            if close:
                coil_force_temp = np.vstack((coil_force_temp, coil_force_temp[0, :]))
                coil_torque_temp = np.vstack((coil_torque_temp, coil_torque_temp[0, :]))
            coil_forces[ppl_cumsum[i]:ppl_cumsum[i+1], :] = coil_force_temp
            coil_torques[ppl_cumsum[i]:ppl_cumsum[i+1], :] = coil_torque_temp

        # copy force and torque data over to pointwise data on a coil curve
        coil_data = np.zeros((data.shape[0], 3))
        for i in range(len(coils)):
            coil_data[ppl_cumsum[i]:ppl_cumsum[i+1], :] = net_forces[i, :]
        coil_data = np.ascontiguousarray(coil_data)
        pointData['NetForces'] = (contig(coil_data[:, 0]),
                                    contig(coil_data[:, 1]),
                                    contig(coil_data[:, 2]))
        coil_data = np.zeros((data.shape[0], 3))
        for i in range(len(coils)):
            coil_data[ppl_cumsum[i]:ppl_cumsum[i+1], :] = net_torques[i, :]
        coil_data = np.ascontiguousarray(coil_data)

        # Add pointwise force and torque data to the dictionary
        pointData['NetTorques'] = (contig(coil_data[:, 0]),
                                    contig(coil_data[:, 1]),
                                    contig(coil_data[:, 2]))
        pointData["Pointwise_Forces"] = (contig(coil_forces[:, 0]), contig(coil_forces[:, 1]), contig(coil_forces[:, 2]))
        pointData["Pointwise_Torques"] = (contig(coil_torques[:, 0]), contig(coil_torques[:, 1]), contig(coil_torques[:, 2]))
    
    # If extra data is provided, add it to the dictionary
    if extra_data is not None:
        pointData = {**pointData, **extra_data}

    # Call curves_to_vtk to save the curves and extra dictionary data 
    curves_to_vtk(curves, filename, close=close, extra_data=pointData)

def coils_via_symmetries(curves, currents, nfp, stellsym, regularizations=None):
    """
    Generate a list of Coil objects by applying rotational and (optionally) stellarator symmetries.

    Take a list of ``n`` curves and return ``n * nfp * (1+int(stellsym))``
    ``Coil`` objects obtained by applying rotations and flipping corresponding
    to ``nfp`` fold rotational symmetry and optionally stellarator symmetry.

    If regularizations are provided for the base curves, then RegularizedCoil objects are returned. This
    is a coil class that carries around a regularization for a finite cross section, which is used 
    for computing e.g. forces and torques on the coil. Format is e.g.
    regularizations = [regularization_circ(0.05) for _ in range(ncoils)]

    Args:
        curves (list, shape (n_coils,)) : List of base curves.
        currents (list, shape (n_coils,)) : List of base current objects.
        nfp (int) : Number of field periods (rotational symmetry).
        stellsym (bool) : Whether to apply stellarator symmetry.
        regularizations (np.array, shape (n_coils,), optional): The regularization objects for the coils representing the finite coil cross section.

    Returns:
        coils (list) : List of Coil or RegularizedCoil objects with symmetries applied. If regularizations are provided, then RegularizedCoil objects are returned.
    """

    assert len(curves) == len(currents)
    curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    currents = apply_symmetries_to_currents(currents, nfp, stellsym)
    if regularizations is None:
        coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
    else:
        regularizations = regularizations * (nfp * (1 + stellsym))
        coils = [RegularizedCoil(curv, curr, regularization) for (curv, curr, regularization) in zip(curves, currents, regularizations)]
    return coils


def load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None):
    """
    Load coils from a mgrid input file, returning a list of Coil objects.

    This function loads a file in MAKEGRID input format containing the Cartesian coordinates 
    and the currents for several coils and returns an array with the corresponding coils. 
    The format is described at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID

    Args:
        filename (str) : Path to the MAKEGRID input file.
        order (int) : Maximum mode number in the Fourier expansion.
        ppp (int, optional) : Points per period for quadrature (default: 20).
        group_names (list of str or str or None, optional) : If provided, only load coils in these groups.

    Returns:
        coils (list) : List of Coil objects loaded from the file.
    """

    if isinstance(group_names, str):
        # Handle case of a single string
        group_names = [group_names]

    with open(filename, 'r') as f:
        all_coils_values = f.read().splitlines()[3:]

    currents = []
    flag = True
    for j in range(len(all_coils_values)-1):
        vals = all_coils_values[j].split()
        if flag:
            curr = float(vals[3])
            flag = False
        if len(vals) > 4:
            flag = True
            if group_names is None:
                currents.append(curr)
            else:
                this_group_name = vals[5]
                if this_group_name in group_names:
                    currents.append(curr)

    curves = CurveXYZFourier.load_curves_from_makegrid_file(filename, order=order, ppp=ppp, group_names=group_names)
    coils = [Coil(curves[i], Current(currents[i])) for i in range(len(curves))]

    return coils


def coils_to_makegrid(filename, curves, currents, groups=None, nfp=1, stellsym=False):
    """
    Export a list of Curve objects and currents to a mgrid (MAKEGRID) input file.

    The output can be used by MAKEGRID and FOCUS. The format is described at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID

    Args:
        filename (str): Name of the file to write.
        curves (list) : list of Curve objects.
        currents (list) : list of Current objects.
        groups (list or None, optional): Coil current group. Coils in the same group are assembled together.
        nfp (int, optional): Number of field periods (default: 1).
        stellsym (bool, optional): Whether to apply stellarator symmetry (default: False).
    """

    assert len(curves) == len(currents)
    coils = coils_via_symmetries(curves, currents, nfp, stellsym)
    ncoils = len(coils)
    if groups is None:
        groups = np.arange(ncoils) + 1
    else:
        assert len(groups) == ncoils
        # should be careful. SIMSOPT flips the current, but actually should change coil order
    with open(filename, "w") as wfile:
        wfile.write("periods {:3d} \n".format(nfp))
        wfile.write("begin filament \n")
        wfile.write("mirror NIL \n")
        for icoil in range(ncoils):
            x = coils[icoil].curve.gamma()[:, 0]
            y = coils[icoil].curve.gamma()[:, 1]
            z = coils[icoil].curve.gamma()[:, 2]
            for iseg in range(len(x)):  # the last point matches the first one;
                wfile.write(
                    "{:23.15E} {:23.15E} {:23.15E} {:23.15E}\n".format(
                        x[iseg], y[iseg], z[iseg], coils[icoil].current.get_value()
                    )
                )
            wfile.write(
                "{:23.15E} {:23.15E} {:23.15E} {:23.15E} {:} {:10} \n".format(
                    x[0], y[0], z[0], 0.0, groups[icoil], coils[icoil].curve.name
                )
            )
        wfile.write("end \n")
    return


def coils_to_focus(filename, curves, currents, nfp=1, stellsym=False, Ifree=False, Lfree=False):
    """
    Export a list of CurveXYZFourier objects and currents to a FOCUS input file.

    The output can be used by FOCUS. The format is described at
    https://princetonuniversity.github.io/FOCUS/rdcoils.pdf

    Args:
        filename (str) : Name of the file to write.
        curves (list) : list of CurveXYZFourier objects.
        currents (list) : list of Current objects.
        nfp (int, optional) : Number of field periods (default: 1).
        stellsym (bool, optional) : Whether to apply stellarator symmetry (default: False).
        Ifree (bool, optional) : Whether the coil current is free (default: False).
        Lfree (bool, optional) : Whether the coil geometry is free (default: False).
    """
    from simsopt.geo import CurveLength

    assert len(curves) == len(currents)
    ncoils = len(curves)
    if stellsym:
        symm = 2  # both periodic and symmetric
    elif nfp > 1 and not stellsym:
        symm = 1  # only periodicity
    else:
        symm = 0  # no periodicity or symmetry
    if nfp > 1:
        print('Please note: FOCUS sets Nfp in the plasma file.')
    with open(filename, 'w') as f:
        f.write('# Total number of coils \n')
        f.write('  {:d} \n'.format(ncoils))
        for i in range(ncoils):
            assert isinstance(curves[i], CurveXYZFourier)
            nf = curves[i].order
            xyz = curves[i].full_x.reshape((3, -1))
            xc = xyz[0, ::2]
            xs = np.concatenate(([0.], xyz[0, 1::2]))
            yc = xyz[1, ::2]
            ys = np.concatenate(([0.], xyz[1, 1::2]))
            zc = xyz[2, ::2]
            zs = np.concatenate(([0.], xyz[2, 1::2]))
            length = CurveLength(curves[i]).J()
            nseg = len(curves[i].quadpoints)
            f.write('#------------{:d}----------- \n'.format(i+1))
            f.write('# coil_type  symm  coil_name \n')
            f.write('  {:d}   {:d}  {:} \n'.format(1, symm, curves[i].name))
            f.write('# Nseg current Ifree Length Lfree target_length \n')
            f.write('  {:d} {:23.15E} {:d} {:23.15E} {:d} {:23.15E} \n'.format(nseg, currents[i].get_value(), Ifree, length, Lfree, length))
            f.write('# NFcoil \n')
            f.write('  {:d} \n'.format(nf))
            f.write('# Fourier harmonics for coils ( xc; xs; yc; ys; zc; zs) \n')
            for r in [xc, xs, yc, ys, zc, zs]:  # 6 lines
                for k in range(nf+1):
                    f.write('{:23.15E} '.format(r[k]))
                f.write('\n')
        f.write('\n')
    return
