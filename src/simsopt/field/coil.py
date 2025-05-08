from math import pi
import numpy as np
from jax import vjp, jacrev

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import Derivative
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.jit import jit
import simsoptpp as sopp
from simsopt.field.force import regularization_circ

__all__ = ['Coil', 'JaxCurrent',
           'Current', 'coils_via_symmetries',
           'load_coils_from_makegrid_file',
           'apply_symmetries_to_currents', 'apply_symmetries_to_curves',
           'coils_to_makegrid', 'coils_to_focus', 'coils_to_vtk'
           ]


class Coil(sopp.Coil, Optimizable):
    """
    Represents a magnetic coil as a combination of a geometric curve and an electric current.

    This class combines a :class:`~simsopt.geo.curve.Curve` and a :class:`Current` object, and is used as input for
    :class:`~simsopt.field.biotsavart.BiotSavart` field calculations. 

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        The geometric curve describing the coil shape.
    current : Current
        The current object describing the electric current in the coil.
    regularization : Regularization
        The regularization object for the coil corresponding to the coil cross section. 
        Default is a circular cross section with radius 0.05.
    """

    def __init__(self, curve, current, regularization=regularization_circ(0.05)):
        self._curve = curve
        self._current = current
        self.regularization = regularization
        sopp.Coil.__init__(self, curve, current)
        Optimizable.__init__(self, depends_on=[curve, current])

    def vjp(self, v_gamma, v_gammadash, v_current):
        return self.curve.dgamma_by_dcoeff_vjp(v_gamma) \
            + self.curve.dgammadash_by_dcoeff_vjp(v_gammadash) \
            + self.current.vjp(v_current)

    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.curve.Curve.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.curve.Curve.plot()`
        """
        return self.curve.plot(**kwargs)


class CurrentBase(Optimizable):
    """
    Abstract base class for current objects that are optimizable.
    """

    def __init__(self, **kwargs):
        Optimizable.__init__(self, **kwargs)

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __rmul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, 1.0/other)

    def __neg__(self):
        return ScaledCurrent(self, -1.)

    def __add__(self, other):
        return CurrentSum(self, other)

    def __sub__(self, other):
        return CurrentSum(self, -other)

    # https://stackoverflow.com/questions/11624955/avoiding-python-sum-default-start-arg-behavior
    def __radd__(self, other):
        # This allows sum() to work (the default start value is zero)
        if other == 0:
            return self
        return self.__add__(other)


class Current(sopp.Current, CurrentBase):
    """
    An optimizable object that wraps around a single scalar degree of freedom representing 
    an electric current.

    This class is used for the current in a coil, or in a set of coils constrained 
    to use the same current.

    Parameters
    ----------
    current : float
        Initial value of the current.
    dofs : array-like or None, optional
        Degrees of freedom for optimization. If None, uses the current value.
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
        return Derivative({self: v_current})

    @property
    def current(self):
        return self.get_value()


class ScaledCurrent(sopp.CurrentBase, CurrentBase):
    """
    Represents a current that is a scaled version of another current object (Scales :mod:`Current` by a factor.)

    Used, for example, to flip currents for stellarator symmetric coils.

    Parameters
    ----------
    current_to_scale : CurrentBase
        The current object to scale.
    scale : float
        The scaling factor.
    """

    def __init__(self, current_to_scale, scale, **kwargs):
        self.current_to_scale = current_to_scale
        self.scale = scale
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_to_scale], **kwargs)

    def vjp(self, v_current):
        return self.scale * self.current_to_scale.vjp(v_current)

    def get_value(self):
        return self.scale * self.current_to_scale.get_value()

    def set_dofs(self, dofs):
        self.current_to_scale.set_dofs(dofs / self.scale)


def current_pure(dofs):
    return dofs


class JaxCurrent(sopp.Current, CurrentBase):
    """
    A current object supporting JAX-based automatic differentiation. Can be used for
    entirely jax-based optimizations with JaxCurves and JaxCurrents. Probably
    need a full Jax-based BiotSavart class to support this fully.

    Parameters
    ----------
    current : float
        Initial value of the current.
    dofs : array-like or None, optional
        Degrees of freedom for optimization. If None, uses the current value.
    """
    def __init__(self, current, dofs=None, **kwargs):
        sopp.Current.__init__(self, current)
        if dofs is None:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 x0=self.get_dofs(), **kwargs)
        else:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 dofs=dofs, **kwargs)

        self.current_jax = jit(lambda dofs: current_pure(dofs))
        self.dcurrent_by_dcurrent_jax = jit(jacrev(self.current_jax))
        self.dcurrent_by_dcurrent_vjp_jax = jit(lambda x, v: vjp(self.current_jax, x)[1](v)[0])

    def current_impl(self, dofs):
        return self.current_jax(dofs)

    def vjp(self, v):
        r"""
        """
        return Derivative({self: self.dcurrent_by_dcurrent_vjp_jax(self.get_dofs(), v)})

    def set_dofs(self, dofs):
        self.local_x = dofs
        sopp.Current.set_dofs(self, dofs)

    @property
    def current(self):
        return self.get_value()


class CurrentSum(sopp.CurrentBase, CurrentBase):
    """
    Represents the sum of two :mod:`Current` objects.

    Used to enforce current constraints or combine currents in optimization.

    Parameters
    ----------
    current_a : CurrentBase
        First current object.
    current_b : CurrentBase
        Second current object.
    """

    def __init__(self, current_a, current_b):
        self.current_a = current_a
        self.current_b = current_b
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_a, current_b])

    def vjp(self, v_current):
        return self.current_a.vjp(v_current) + self.current_b.vjp(v_current)

    def get_value(self):
        return self.current_a.get_value() + self.current_b.get_value()


def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    """
    Generate a list of curves by applying rotational and (optionally) stellarator symmetries.

    Take a list of ``n`` :mod:`simsopt.geo.curve.Curve`s and return ``n * nfp *
    (1+int(stellsym))`` :mod:`simsopt.geo.curve.Curve` objects obtained by
    applying rotations and flipping corresponding to ``nfp`` fold rotational
    symmetry and optionally stellarator symmetry.

    Parameters
    ----------
    base_curves : list of Curve
        List of base curves to replicate.
    nfp : int
        Number of field periods (rotational symmetry).
    stellsym : bool
        Whether to apply stellarator symmetry (flipping).

    Returns
    -------
    curves : list of Curve
        List of curves with symmetries applied.
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

    Parameters
    ----------
    base_currents : list of Current
        List of base current objects to replicate.
    nfp : int
        Number of field periods (rotational symmetry).
    stellsym : bool
        Whether to apply stellarator symmetry (sign flip).

    Returns
    -------
    currents : list of Current
        List of current objects with symmetries applied.
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
        coils: A python list of Coil objects.
        filename: Name of the file to write.
        close: Whether to draw the segment from the last quadrature point back to the first.
    """
    from simsopt.field.force import coil_net_force, coil_net_torque, coil_force, coil_torque
    from simsopt.geo.curve import curves_to_vtk
    curves = [coil.curve for coil in coils]
    currents = [coil.current.get_value() for coil in coils]
    if close:
        ppl = np.asarray([c.gamma().shape[0]+1 for c in curves])
    else:
        ppl = np.asarray([c.gamma().shape[0] for c in curves])
    contig = np.ascontiguousarray
    pointData = {}
    data = np.concatenate([i*np.ones((ppl[i], )) for i in range(len(curves))])
    coil_data = np.zeros(data.shape)
    for i in range(len(currents)):
        coil_data[i * ppl[i]: (i + 1) * ppl[i]] = currents[i]
    coil_data = np.ascontiguousarray(coil_data)
    pointData['I'] = coil_data
    pointData['I_mag'] = contig(np.abs(coil_data))

    NetForces = np.array([coil_net_force(c, coils) for c in coils])
    NetTorques = np.array([coil_net_torque(c, coils) for c in coils])
    coil_data = np.zeros((data.shape[0], 3))
    for i in range(len(coils)):
        coil_data[i * ppl[i]: (i + 1) * ppl[i], :] = NetForces[i, :]
    coil_data = np.ascontiguousarray(coil_data)
    pointData['NetForces'] = (contig(coil_data[:, 0]),
                                contig(coil_data[:, 1]),
                                contig(coil_data[:, 2]))
    coil_data = np.zeros((data.shape[0], 3))
    for i in range(len(coils)):
        coil_data[i * ppl[i]: (i + 1) * ppl[i], :] = NetTorques[i, :]
    coil_data = np.ascontiguousarray(coil_data)
    pointData['NetTorques'] = (contig(coil_data[:, 0]),
                                contig(coil_data[:, 1]),
                                contig(coil_data[:, 2]))
    
    ppl2 = np.asarray([c.gamma().shape[0] for c in curves])
    data2 = np.concatenate([i*np.ones((ppl2[i], )) for i in range(len(curves))])
    forces = np.zeros((data2.shape[0], 3))
    torques = np.zeros((data2.shape[0], 3))
    for i, c in enumerate(coils):
        forces[i * ppl2[i]: (i + 1) * ppl2[i], :] = coil_force(c, coils)
        torques[i * ppl2[i]: (i + 1) * ppl2[i], :] = coil_torque(c, coils)
        if close:
            forces[i + 1 * ppl2[i], :]  = forces[i * ppl2[i], :]
            torques[i + 1 * ppl2[i], :] = torques[i * ppl2[i], :]
    pointData["Pointwise_Forces"] = (contig(forces[:, 0]), contig(forces[:, 1]), contig(forces[:, 2]))
    pointData["Pointwise_Torques"] = (contig(torques[:, 0]), contig(torques[:, 1]), contig(torques[:, 2]))
    if extra_data is not None:
        pointData = {**pointData, **extra_data}
    curves_to_vtk(curves, filename, close=close, extra_data=pointData)

def coils_via_symmetries(curves, currents, nfp, stellsym):
    """
    Generate a list of Coil objects by applying rotational and (optionally) stellarator symmetries.

    Take a list of ``n`` curves and return ``n * nfp * (1+int(stellsym))``
    ``Coil`` objects obtained by applying rotations and flipping corresponding
    to ``nfp`` fold rotational symmetry and optionally stellarator symmetry.

    Parameters
    ----------
    curves : list of Curve
        List of base curves.
    currents : list of Current
        List of base current objects.
    nfp : int
        Number of field periods (rotational symmetry).
    stellsym : bool
        Whether to apply stellarator symmetry.

    Returns
    -------
    coils : list of Coil
        List of Coil objects with symmetries applied.
    """

    assert len(curves) == len(currents)
    curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    currents = apply_symmetries_to_currents(currents, nfp, stellsym)
    coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
    return coils


def load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None):
    """
    Load coils from a MAKEGRID input file, returning a list of Coil objects.

    This function loads a file in MAKEGRID input format containing the Cartesian coordinates 
    and the currents for several coils and returns an array with the corresponding coils. 
    The format is described at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID

    Parameters
    ----------
    filename : str
        Path to the MAKEGRID input file.
    order : int
        Maximum mode number in the Fourier expansion.
    ppp : int, optional
        Points per period for quadrature (default: 20).
    group_names : list of str or str or None, optional
        If provided, only load coils in these groups.

    Returns
    -------
    coils : list of Coil
        List of Coil objects loaded from the file.
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
    Export a list of Curve objects and currents to a MAKEGRID input file.

    The output can be used by MAKEGRID and FOCUS. The format is described at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID

    Parameters
    ----------
    filename : str
        Name of the file to write.
    curves : list of Curve objects.
        List of Curve objects.
    currents : list of Current objects.
        List of current objects.
    groups : list or None, optional
        Coil current group. Coils in the same group are assembled together.
    nfp : int, optional
        Number of field periods (default: 1).
    stellsym : bool, optional
        Whether to apply stellarator symmetry (default: False).
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

    Parameters
    ----------
    filename : str
        Name of the file to write.
    curves : list of CurveXYZFourier
        List of CurveXYZFourier objects.
    currents : list of Current
        List of current objects.
    nfp : int, optional
        Number of field periods (default: 1).
    stellsym : bool, optional
        Whether to apply stellarator symmetry (default: False).
    Ifree : bool, optional
        Whether the coil current is free (default: False).
    Lfree : bool, optional
        Whether the coil geometry is free (default: False).
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
