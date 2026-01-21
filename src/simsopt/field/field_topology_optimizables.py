# field_topology_optimizables.py
# simsopt-pyoculus interface to utilize pyoculus methods for locating fixed points
# and using their properties (Residue, location, etc) in optimizations.
# Also provides methods for calculating turnstile areas (measure for stochastic transport)
# Chris Smiet Feb 2025  christopher.smiet@epfl.ch
#

import numpy as np
from scipy.integrate import solve_ivp
try :
    from pyoculus.solvers import FixedPoint, Manifold
    from pyoculus.fields import SimsoptBfield
    from pyoculus.maps import CylindricalBfieldSection
    from pyoculus import __version__ as pyoc_version
except ImportError:
    pyoc_version = -1
    pyoculus = None
from .._core import Optimizable
from .._core.dev import SimsoptRequires
from .._core.util import ObjectiveFailure
from numpy.typing import NDArray
from typing import Union, Iterable, TYPE_CHECKING
import itertools

if TYPE_CHECKING:
    from ..field import MagneticField, Integrator

try:
    newpyoculus = int(str(pyoc_version).split('.')[0]) >= 1
except Exception:
    newpyoculus = False


__all__ = ['PyOculusFixedPoint', 'TwoFixedPointDifference', 'ClinicConnection', 'IntegratorFixedPoint', 'FDMapJac', 'FDEllipticity', 'FDResidue', 'FDTrace']

def _ellipticity(jacobian_matrix):
    """
    return the ellipticity-like quantity of the calculated jacobian, as described
    by Greene 1968: https://doi.org/10.1063/1.1664639
    The ellipticity is +1 for cirular elliptic fixed points, 0 when parabolic, and negative
    when hyperbolic. Ellipticity of -1 correspoinds to perpendicular intersection of stable and unstable manifolds.
    TODO: check if indices are correct (though anser is invariant to index permutations)
    """
    a = (jacobian_matrix[0, 0] + jacobian_matrix[1, 1])/2
    b = (jacobian_matrix[0, 0] - jacobian_matrix[1, 1])/2
    c = (jacobian_matrix[0, 1] + jacobian_matrix[1, 0])/2
    d = (jacobian_matrix[0, 1] - jacobian_matrix[1, 0])/2
    return (1-a**2)/(b**2 + c**2 + d**2)



class PyOculusFixedPoint(Optimizable):
    """
    A FixedPoint is a point in the Poincar\'e section of a magnetic field
    that maps to itself after n field periods.
    Pyoculus contains routines to efficiently locate fixed points, and calculate
    their properties (residue, location, etc). This class provides an interface 
    to those calculations. 
    """

    @SimsoptRequires(newpyoculus, "This PyOculusFixedPoint class requres the additions by L. Rais in version 1.0.0")
    def __init__(self,
                 pyocmap: "CylindricalBfieldSection",
                 start_guess: Iterable[np.float64],
                 fp_order: int = 1,
                 finder_args: dict = dict(),
                 parent_field: "MagneticField" = None,
                 with_axis: bool = False,
                 ):
        """
        Initialize a FixedPoint object
        Args:
            field: MagneticField object
            start_guess: initial guess for the fixed point location (R, Z).
            nfp: Number of field period
            phi0: Toroidal angle of the fixed point
            fp_order: Order of the fixed point
            axis: Location of the axis, len 2 array. The axis is also a fixed point, necessary if the poloidal mode number (winding with the axis) is required.
                   if False a nonsense axis will be given to pyoculus.
            integration_tol: Tolerance for integration
            integration_args: Additional arguments for constructing the Poincar\'e section.
                    See pyoculus.maps.CylindricalBfieldSection and pyoculus.maps.IntegratedMap for details.
            finder_args: Additional arguments for fixed point finder.
                    See pyoculus.solvers.FixedPoint for details.
        """
        self._pyocmap = pyocmap
        if parent_field is None: 
            parent_field = pyocmap._mf._mf
        self._parent_field = parent_field
        self._with_axis = with_axis
        self.start_guess = start_guess
        self.fp_order = fp_order
        self.finder_args = finder_args

        self._fixed_point = FixedPoint(self._pyocmap)  # create the fixed point
        self._fixed_point.find(self.fp_order, start_guess, **self.finder_args)  # find it
        if not self._fixed_point.successful:
            raise ValueError("Fixed point finder failed")
        self._current_location = np.copy(self._fixed_point.coords[0])
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[self._parent_field])
        self._refind_fp = False

    @classmethod
    def from_field(cls,
                   field: "MagneticField",
                   nfp: int,
                   start_guess: NDArray[np.float64],
                   finder_args: dict = dict(),
                   fp_order: int = 1, 
                   phi0: float = 0., 
                   axis: Union[bool, NDArray[np.float64]] = False, 
                   integration_tol: float = 1e-12, 
                   integration_args: dict = dict()
                   ):
        """
        Create the map first, using a simsopt MagneticField object, then instantiate the PyOculusFixedPoint object.
        """
        if 'rtol' not in integration_args:
            integration_args['rtol'] = integration_tol
        oculus_field_representation = SimsoptBfield(nfp, field)  # pyoculus' interfacing class
        if axis is True or isinstance(axis, Iterable):
            with_axis = True
        else:
           with_axis = False

        if with_axis:  # create the map with an axis so that rotational transform and such can be calculated.
            if not isinstance(axis, Iterable):  # if axis evaluates true, start with a random point (bound to fail)
                axis = np.array([1., 0.])
            pyocmap = CylindricalBfieldSection.without_axis(oculus_field_representation, phi0=phi0, guess=axis, **integration_args)
        else:
            pyocmap = CylindricalBfieldSection(oculus_field_representation, R0=0, Z0=0, phi0=phi0, **integration_args)  # create the map with 'axis' at the origin (methods calculating rotational transform will fail)
        return cls(pyocmap=pyocmap,
                   start_guess=start_guess,
                   fp_order=fp_order,
                   finder_args=finder_args,
                   parent_field=field,
                   with_axis=with_axis,
                   )
    
    @property
    def map(self):
        """
        The fixed point map object
        """
        return self._pyocmap

    @map.setter
    def map(self, value):
        raise ValueError("Cannot set map")
                         
    @property
    def fixed_point(self):
        """
        Return the fixed point object
        """
        return self._fixed_point

    @fixed_point.setter
    def fixed_point(self, value):
        raise ValueError("Cannot set fixed point")


    def recompute_bell(self, parent=None):
        self._pyocmap.clear_cache()  #remove cached maps as field changed
        self._refind_fp = True

    def refind(self):
        """
        triggers when targets are called and
        field has changed (recompute bell rung).
        Re-finds the fixed point (and axis if given).
        """
        try:
            self._fixed_point.find(self.fp_order, self._current_location, **self.finder_args)
            if self.fp_order > 1 and np.linalg.norm(self._fixed_point.coords[0] - self._fixed_point.coords[1]) < 1e-3:  # if accidentally jumped to the axis, raise hell
                raise ObjectiveFailure("Fixed point finder jumped to the axis")
            if not self._fixed_point.successful:
                raise ObjectiveFailure("Fixed point finder failed")
            closest_orbit = np.argmin(np.linalg.norm(self._fixed_point.coords[:-1] - self._current_location, axis=1))  # not the last!
            if closest_orbit !=0: 
                print(f'Fixed point had to jump back to other orbit, {closest_orbit} -> 0')
            self._current_location = np.copy(self._fixed_point.coords[closest_orbit])
        except Exception as e:
            raise ObjectiveFailure("Failed to find fixed point") from e
        if self._with_axis:
            try:
                self._pyocmap.find_axis()  # re-find the axis
            except Exception as e:
                raise ObjectiveFailure("Failed to find axis") from e
        self._refind_fp = False

    def loc_RZ(self):
        if self._refind_fp:
            self.refind()
        return self._fixed_point.coords[0]

    def loc_xyz(self):
        if self._refind_fp:
            self.refind()
        RZ = self.loc_RZ()
        return np.array(RZ[0]*np.cos(self.phi0), RZ[0]*np.sin(self.phi0), RZ[1])

    def R(self):
        return self.loc_RZ()[0]

    def Z(self):
        return self.loc_RZ()[1]

    def x_coord(self):
        return self.loc_xyz()[0]

    def y_coord(self):
        return self.loc_xyz()[1]

    def z_coord(self):
        return self.loc_xyz()[2]

    def residue(self):
        if self._refind_fp:
            self.refind()
        return self._fixed_point.GreenesResidue

    def jacobian(self):
        if self._refind_fp:
            self.refind()
        return self._fixed_point.Jacobian
    
    def trace(self):
        return np.trace(self.jacobian())
    
    def loc_at_other_angle(self, phi): 
        """
        return the location of the fixed point at a different toroidal angle.
        This is done by integrating the field line from the known location
        to the new angle.
        """
        if self._refind_fp:
            self.refind()
        fraction_of_nfp = phi * self.map._mf.Nfp / (2 * np.pi)  # convert to normalized toroidal angle
        RZ_end = self.map.f(fraction_of_nfp, self.loc_RZ())
        return RZ_end
    
    def ellipticity(self):
        """
        return the ellipticity-like quantity of the calculated jacobian, as described
        by Greene 1968: https://doi.org/10.1063/1.1664639
        The ellipticity is +1 for cirular elliptic fixed points, 0 when parabolic, and negative
        when hyperbolic. Ellipticity of -1 correspoinds to perpendicular intersection of stable and unstable manifolds.
        """
        return _ellipticity(self.jacobian())

    def difference_optimizable(self, position_RZ):
        """ 
        return an optimizable child object whose J function is the distance
        from this fixed point to the given position_RZ. 
        """
        _difference_optimizable = Optimizable(x0=np.asarray([]), depends_on=[self,])
        setattr(_difference_optimizable, 'J', lambda: np.linalg.norm(self.loc_RZ() - position_RZ))
        return _difference_optimizable
    
    def ellipticity_optimizable(self, target_ellipticity=0, penalize="difference"):
        """
        return an optimizable child object whose J function is the ellipticity"""
        _ellipticity_optimizable = Optimizable(x0=np.asarray([]), depends_on=[self,])
        if penalize == "above":
            setattr(_ellipticity_optimizable, 'J', lambda: np.maximum(0, self.ellipticity() - target_ellipticity))
        elif penalize == "below":
            setattr(_ellipticity_optimizable, 'J', lambda: np.minimum(0, self.ellipticity() - target_ellipticity))
        else:
            setattr(_ellipticity_optimizable, 'J', lambda: self.ellipticity() - target_ellipticity)
        return _ellipticity_optimizable
    
    
    def residue_optimizable(self):
        """
        return an optimizable child object whose J function is the Greenes residue
        """
        _residue_optimizable = Optimizable(x0=np.asarray([]), depends_on=[self,])
        setattr(_residue_optimizable, 'J', lambda: np.linalg.norm(self.residue()))
        return _residue_optimizable
    
    def trace_optimizable(self, target_trace=0, penalize="difference"):
        """
        return an optimizable child object whose J function is the trace.
        penalize can be "difference" (default), "above", or "below", 
        """
        _trace_optimizable = Optimizable(x0=np.asarray([]), depends_on=[self,])
        if penalize == "above":
            setattr(_trace_optimizable, 'J', lambda: np.maximum(0, self.trace() - target_trace))
        elif penalize == "below":
            setattr(_trace_optimizable, 'J', lambda: np.minimum(0, self.trace() - target_trace))
        elif penalize == "difference":
            setattr(_trace_optimizable, 'J', lambda: self.trace() - target_trace)
        else:
            raise ValueError("penalize must be 'difference', 'above', or 'below'")
        return _trace_optimizable

    def trace_in_range_optimizable(self, min_trace=-2, max_trace=2):
        """
        return an optimizable child object that is a QuadraticPenalty applied to the trace
        of the fixed point.
        """
        return self.trace_optimizable(target_trace=min_trace, penalize="below") + self.trace_optimizable(target_trace=max_trace, penalize="above")
    
    def other_angle_location_optimizable(self, phi, target_RZ):
        """
        return an optimizable child object whose J function is the distance
        from the location of this fixed point at toroidal angle phi to the given target_RZ. 
        """
        _other_angle_location_optimizable = Optimizable(x0=np.asarray([]), depends_on=[self,])
        setattr(_other_angle_location_optimizable, 'J', lambda: np.linalg.norm(self.loc_at_other_angle(phi) - target_RZ))
        return _other_angle_location_optimizable
    
class TwoFixedPointDifference(Optimizable):
    """
    The difference between two fixed points. Can also return Optimizables 
    that give the distance between the two trajectories at other angles.
    """
    def __init__(self, fp1: PyOculusFixedPoint, fp2: PyOculusFixedPoint):
        """
        Initialize a TwoFixedPointDifference
        Args:
            fp1: First fixed point
            fp2: Second fixed point
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp1, fp2])
        self.fp1 = fp1
        self.fp2 = fp2

    def J(self):
        return np.linalg.norm(self.fp1.loc_RZ() - self.fp2.loc_RZ())
    
    def other_angle_difference_optimizable(self, phi, desired_distance=0, penalize="above"):
        """
        Create an `Optimizable` object that evaluates the distance between the trajectories of
        the two fixed points at a different toroidal angle `phi`. 
        This is useful to constrain the two trajectories
        """
        _other_angle_location_difference = Optimizable(x0=np.asarray([]), depends_on=[self.fp1, self.fp2])
        if penalize == "above":
            setattr(_other_angle_location_difference, 'J', lambda: np.maximum(0, np.linalg.norm(self.fp1.loc_at_other_angle(phi) - self.fp2.loc_at_other_angle(phi)) - desired_distance))
        elif penalize == "below":
            setattr(_other_angle_location_difference, 'J', lambda: np.minimum(0, np.linalg.norm(self.fp1.loc_at_other_angle(phi) - self.fp2.loc_at_other_angle(phi)) - desired_distance))
        else:
            setattr(_other_angle_location_difference, 'J', lambda: np.linalg.norm(self.fp1.loc_at_other_angle(phi) - self.fp2.loc_at_other_angle(phi)) - desired_distance)
        return _other_angle_location_difference
    




@SimsoptRequires(newpyoculus, "This PyOculusFixedPoint class requres the additions by L. Rais in 1.0.0")
class ClinicConnection(Optimizable):
    """
    The ClinicConnection class allows calculation of the turnstile flux and
    other properties of homo- and heteroclinic connections using the PyOculus library.

    The fixed points must both be hyperbolic (topological index = -1)
    It is the Simsopt-binding to the pyoculus Manifold class.
    It allows calculation of the turnstile flux, which is a measure
    for stochastic transport.

    fixed points must be of the same map, etc!
    """
    def __init__(self,
                 fp1: PyOculusFixedPoint,
                 fp2: PyOculusFixedPoint,
                 dir1: NDArray = None,
                 dir2: NDArray = None,
                 first_stable=True,
                 ns: int = 5,
                 nu: int = 5,
                 order: int = 2,
                 stable_epsilons: Iterable = None,
                 unstable_epsilons: Iterable = None,
                 nretry_clinicfinding: int = 1,
                 nextratries_clinicfinding: int = 0,
                 clinicfinding_argument_dict = dict(),
                 ERR = 1e-3
                 ):
        """
        Initialize a ClinicConnection
        Args:
            fp1: PyOculusFixedPoint object
            fp2: PyOculusFixedPoint object
            dir1: Direction (approximate) that the manifold leaves the first fixed point
            dir2: Direction (approximate) that the manifold leaves the second fixed point
            first_stable: True if the first fixed point is stable
            ns: Number of times to map along the stable direction to find homo/heteroclinic trajectories
            nu: Number of times to map along the unstable direction to find homo/heteroclinic trajectories
            order: number of homo/heteroclinic trajectories to find
            stable_epsilons: floats, len=order: The distance along the stable manifold to start with the heteroclinic point finding
            unstable_epsilons: floats, len=order: The distance along the unstable manifold to start with the heteroclinic point finding
            nretry_clinicfinding: number of times to re-try the finding of each clinic if it field_topology_optimizables
            nextratries_clinicfinding: number of extra attempts to find more clinics
        """
        if fp1.map != fp2.map:
            raise ValueError("Fixed points must be of the same map. \n "
                             "Hint: use x1 = PyOculusFixedPoint.from_field() and give x2\n"
                             "the map using x2=PyOculusFixedPoint(x1.map, ...)")
        if fp1.fp_order != fp2.fp_order:
            raise ValueError("Fixed points must have the same order (for now)")  # could be allowed in future pyoculus
        if fp1._fixed_point.topological_index != -1 or fp2._fixed_point.topological_index != -1:
            raise ValueError("Fixed points must be hyperbolic (negative topological index)")
        self.fp1 = fp1
        self.fp2 = fp2
        self._dir1 = dir1
        self._dir2 = dir2
        self._first_stable = first_stable
        self._order = order
        self._ns = ns
        self._nu = nu
        self._stable_epsilons = stable_epsilons
        self._unstable_epsilons = unstable_epsilons
        self._nretry_clinicfinding = nretry_clinicfinding
        self._nextratries_clinicfinding = nextratries_clinicfinding
        self._clinicfinding_argument_dict = clinicfinding_argument_dict
        self._ERR = ERR
        self._manifold = Manifold(fp1._pyocmap, fp1._fixed_point, fp2._fixed_point, dir1, dir2, first_stable)
        self.set_true_directions()
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp1, fp2])
        self._need_to_reset = True

    def set_true_directions(self):
        """
        set the true directions for the manifold, such that in optimization the next
        manifold is found that is closest to the previous one.
        """
        if self._first_stable:
            self._truedir1 = self._manifold.vector_s
            self._truedir2 = self._manifold.vector_u
        else:
            self._truedir1 = self._manifold.vector_u
            self._truedir2 = self._manifold.vector_s

    def recompute_bell(self, parent=None):
        self._need_to_reset = True

    def reset(self):
        """
        if DoFs changed, trash all properties and refind fixed points.
        Do not perform expensive calculations, this is only
        when properties are requested.
        """
        self._manifold.clinics.reset()  # remove found clinics
        self._manifold._stable_trajectory = None  # remove other properties
        self._manifold._unstable_trajectory = None
        self._manifold._areas = None
        if self.fp1._refind_fp:  # re-calculate if fp bell was rung (should not trigger re-compute if one was re-found for other target)
            self.fp1.refind()
        if self.fp2._refind_fp:
            self.fp2.refind()
        self._manifold.choose(self._truedir1, self._truedir2, self._first_stable)
        self.set_true_directions()
        try:
            # find the first fundamental clinic:
            self._manifold.find_clinic_single(self._stable_epsilons[0], self._unstable_epsilons[0], n_s=self._ns, n_u=self._nu, nretry=self._nretry_clinicfinding, reset_clinics=True, root_args=self._clinicfinding_argument_dict, ERR=self._ERR)
        except Exception as e:
            raise ObjectiveFailure("Failed to find fundamental clinic") from e
        # find the others with one less nu:
        for eps_s_guess, eps_u_guess in zip(self._stable_epsilons[1:], self._unstable_epsilons[1:]):
            try:
                self._manifold.find_clinic_single(eps_s_guess, eps_u_guess, n_s=self._ns-1, n_u=self._nu, nretry=self._nretry_clinicfinding, ERR=self._ERR, root_args=self._clinicfinding_argument_dict)
            except Exception as e:
                print(f"one of the clinic finders failed due to {e}, will only fail if total found clinics becomes less than order")
        for shift in np.linspace(0, 1, self._nextratries_clinicfinding +1, endpoint=False)[1:]:
            try:
                self._manifold.find_other_clinic(shift_in_stable=shift, nretry=self._nretry_clinicfinding, root_args=self._clinicfinding_argument_dict, ERR=self._ERR)
            except Exception as e:
                print(f"extra finding attempt did not succeed because {e}, will only fail if total found clinics becomes less than order")
        if self._manifold.clinics.size < self._order:
            raise ObjectiveFailure("Failed to find all clinics")
        if np.abs(np.sum(self._manifold.turnstile_areas)) > np.max(np.abs(self._manifold.turnstile_areas))/1e3:
            raise ObjectiveFailure("Escaping Chickens! extra clinics required")
        self._stable_epsilons = self._manifold.clinics.stable_epsilons[:-1]
        self._unstable_epsilons = self._manifold.clinics.unstable_epsilons[:-1]
        self._need_to_reset = False

    def J(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.turnstile_areas).sum()/2
    
    def first_area(self):
        if self._need_to_reset:
            self.reset()
        if len(self._manifold.turnstile_areas) < 1:
            raise ObjectiveFailure("No turnstile areas found")
        return np.abs(self._manifold.turnstile_areas[0])
    
    def second_area(self):
        if self._need_to_reset:
            self.reset()
        if len(self._manifold.turnstile_areas) < 2:
            raise ObjectiveFailure("Less than two turnstile areas found")
        return np.abs(self._manifold.turnstile_areas[1])
    
    def third_area(self):
        if self._need_to_reset:
            self.reset()
        if len(self._manifold.turnstile_areas) < 3:
            raise ObjectiveFailure("Less than three turnstile areas found")
        return np.abs(self._manifold.turnstile_areas[2])
    
    def fourth_area(self):
        if self._need_to_reset:
            self.reset()
        if len(self._manifold.turnstile_areas) < 4:
            raise ObjectiveFailure("Less than four turnstile areas found")
        return np.abs(self._manifold.turnstile_areas[3])

    def fifth_area(self):
        if self._need_to_reset:
            self.reset()
        if len(self._manifold.turnstile_areas) < 5:
            raise ObjectiveFailure("Less than five turnstile areas found")
        return np.abs(self._manifold.turnstile_areas[4])
    
    def sixth_area(self):
        if self._need_to_reset:
            self.reset()
        if len(self._manifold.turnstile_areas) < 6:
            raise ObjectiveFailure("Less than six turnstile areas found")
        return np.abs(self._manifold.turnstile_areas[5])

class IntegratorFixedPoint(Optimizable):
    """
    The IntegratorFixedPoint class is a very robust and simple method of optimizing for a fixed point at a given location.
    The cost is simply the diffference between the (R, Z) coordinates of the starting point, and 
    the (R,Z) coordinates where the field line ends up after integration over `nfp` field periods.

    This makes the IntegratorFixedPoint very reliable, as long as the field line integration does not error, it will 
    return a quantity that can be optimized to zero. 
    In contrast, the PyoculusFixedPoint will raise an ObjectiveFailure if the fixed point finder fails. 
    This is useful for cold starts for example in a purely toroidal field or very chaotic field, 
    there is no prior knowledge of the fixed point location.

    The axis is a fixed point of the 1-field-period map (and every multiple map as well).
    In an N nfp stellarator, a ntor/mpol island chain is a fixed point of mpol field periods.
    For example to try to put an x- or o-point of the 5/5 island chain at (R, Z) in the phi=0 plane:
    ```
    [code to generate a BiotSavart object]
    fp_at_RZ = IntegratorFixedPoint(field, (R, Z), 5, 5, 0)
    prob= LeastSquaresProblem.from_tuples((fp_at_RZ.J(), 0, 1), ...[other objectives]))
    least_squares_serial_solve(prob)
    ```

    """
    def __init__(self, integrator: "Integrator", RZ: NDArray[np.float64], integration_nfp: int, phi0: float = 0.):
        """
        Initialize a SimpleFixedPoint
        Args:
            field: MagneticField object
            RZ: the R and Z-coordinate where a fixed point is required
            nfp: Number of field periods of the fixed point
            phi0: Toroidal angle of the fixed point
        """
        self.RZ = RZ
        self.field_nfp = integrator.nfp
        self.integration_nfp = integration_nfp
        self.phi0 = phi0
        self.integrator = integrator
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[integrator])

    def J(self):
        """
        Integrate the field line and return the distance form the starting point.
        """
        phi_end = self.phi0 + 2*np.pi*self.integration_nfp/self.field_nfp
        try:
            dist_mapped = np.linalg.norm(self.integrator.integrate_in_phi_cyl(self.RZ, self.phi0, phi_end)[::2] - self.RZ)
        except Exception as e:
            raise ObjectiveFailure("Failed to integrate field line") from e
        return dist_mapped

    # def dJ(self): can be made with integration of the derivatives

    def mapjac(self):
        """
        Calculate the jacobian of the field line map at the fixed point
        """
        phi_end = self.phi0 + 2*np.pi*self.integration_nfp/self.field_nfp
        try:
            jac = self.integrator.integrate_in_phi_tangentmap(self.RZ, self.phi0, phi_end)[::2, ::2]  # only R,Z
        except Exception as e:
            raise ObjectiveFailure("Failed to integrate field line") from e
        return jac



class FDMapJac(Optimizable):
    """
    The FDMapJac uses finite-differences to evaluate the jacobian of the field line map. 
    The J() returns the trace of the jacobian, for residue-like optimisations (though if the point is not a fixed point, you can get strange results)

    ```

    """
    def __init__(self, integrator: "Integrator", RZ: NDArray[np.float64], integration_nfp: int, FD_stepsize=1e-7, phi0: float = 0., tol=1e-9, max_step=1e5):
        """
        Initialize a SimpleFixedPoint
        Args:
            field: MagneticField object
            RZ: the R and Z-coordinate where the Jacobian of the field line map needs to be evaluated
            nfp: Number of field periods in the magnetic field
            integration_nfp: number of field periods to integrate. 
            phi0: Toroidal angle where integration is started. 
            FD_stepsize: size of the FD step to take. 
            tol: integartion Tolerance.
            max_step= maximum number of steps. 
        """
        self.RZ = RZ
        self.field_nfp = integrator.nfp
        self.integration_nfp = integration_nfp
        self.phi0 = phi0
        self.FD_stepsize = FD_stepsize
        self.integrator = integrator
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[integrator])
        self._mapjac = None

    @property
    def mapjac(self):
        if self._mapjac is None:
            self._mapjac = self.calculate_mapjac()
        return self._mapjac
    
    def recompute_bell(self, parent=None):
        self._mapjac = None

    def calculate_mapjac(self):
        """
        Calculate the distance from the target fixed point
        """
        phi_end = self.phi0 + 2*np.pi*self.integration_nfp/self.field_nfp
        startpoints_posdiff = self.RZ + np.eye(2) * self.FD_stepsize
        startpoints_negdiff = self.RZ - np.eye(2) * self.FD_stepsize
        try:
            centermap = self.integrator.integrate_in_phi_cyl(self.RZ, self.phi0, phi_end)[::2]  # only R,Z
            posdiffmaps = np.array([self.integrator.integrate_in_phi_cyl(diffstart, self.phi0, phi_end)[::2] - centermap for diffstart in startpoints_posdiff])
            negdiffmaps = np.array([self.integrator.integrate_in_phi_cyl(diffstart, self.phi0, phi_end)[::2] - centermap for diffstart in startpoints_negdiff])
            jac = ((posdiffmaps - negdiffmaps)/(2*self.FD_stepsize)).T
        except Exception as e:
            raise ObjectiveFailure("Failed to integrate field line") from e
        return jac

    def J(self):
        """
        return the trace of the calculated jacobian
        """
        return np.trace(self.mapjac)

    def residue(self):
        """
        return the residue-like quantity of the calculated jacobian
        """
        tr = self.J()
        return (2 - tr)/4
    
    def ellipticity(self):
        """
        return the ellipticity-like quantity of the calculated jacobian, as described
        by Greene 1968: https://doi.org/10.1063/1.1664639
        The ellipticity is +1 for cirular elliptic fixed points, 0 when parabolic, and negative
        when hyperbolic. Ellipticity of -1 correspoinds to perpendicular intersection of stable and unstable manifolds.
        TODO: check if indices are correct (though anser is invariant to index permutations)
        """
        return _ellipticity(self.mapjac)


class FDEllipticity(Optimizable):
    """
    Wrapper around SimpleMapJac to return the ellipticity of the map jacobian
    as described by Greene 1968: https://doi.org/10.1063/1.1664639
    
    The ellipticity is +1 for cirular elliptic fixed points, 0 when parabolic, and negative
    when hyperbolic. Ellipticity of -1 correspoinds to perpendicular intersection of stable and unstable manifolds.
    """
    def __init__(self, simplemapjac: FDMapJac):
        self.FDmapjac = simplemapjac
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[simplemapjac])
    
    def J(self):
        return self.FDmapjac.ellipticity()


class FDResidue(Optimizable):
    """
    Wrapper around SimpleMapJac to return the Greenes residue of the map jacobian
    as described by Greene 1968: https://doi.org/10.1063/1.1664639
    The residue is 0 for regular parabolic fixed points, and positive
    for regular hyperbolic fixed points. 
    Negative residue can be elliptic or alternating hyperbolic. 
    """
    def __init__(self, fdmapjac: FDMapJac):
        self.fdmapjac = fdmapjac
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fdmapjac])  

    def J(self):
        return self.fdmapjac.residue()
    
class FDTrace(Optimizable):
    """
    Wrapper around FDMapJac to return the trace of the map jacobian
    """
    def __init__(self, fdmapjac: FDMapJac):
        self.fdmapjac = fdmapjac
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fdmapjac])  

    def J(self):
        return self.fdmapjac.J()  