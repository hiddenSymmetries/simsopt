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
from typing import Union, Iterable
import itertools

try:
    newpyoculus = int(str(pyoc_version).split('.')[0]) >= 1
except Exception:
    newpyoculus = False


__all__ = ['PyOculusFixedPoint', 'TwoFixedPointDifference', 'ClinicConnection', 'SimpleFixedPoint_RZ', 'SimpleIntegrator', 'SimpleMapJac', 'SimpleEllipticity', 'SimpleResidue', 'SimpleTrace']

def _rphiz_to_xyz(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert an array of R, phi, Z to x, y, z
    """
    if array.ndim == 1:
        array2 = array[None, :]
    else:
        array2 = array[:, :]
    return np.array([array2[:, 0]*np.cos(array2[:, 1]), array2[:, 0]*np.sin(array[:, 1]), array[:, 2]]).T

def _xyz_to_rphiz(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert an array of x, y, z to R, phi, Z
    """
    if array.ndim == 1:
        array2 = array[None, :]
    else:
        array2 = array[:, :]
    return np.array([np.sqrt(array2[:, 0]**2 + array2[:, 1]**2), np.arctan2(array2[:, 1], array2[:, 0]), array2[:, 2]]).T


def _ellipticity(matrix):
    """
    return the ellipticity-like quantity of the calculated jacobian, as described
    by Greene 1968: https://doi.org/10.1063/1.1664639
    The ellipticity is +1 for cirular elliptic fixed points, 0 when parabolic, and negative
    when hyperbolic. Ellipticity of -1 correspoinds to perpendicular intersection of stable and unstable manifolds.
    TODO: check if indices are correct (though anser is invariant to index permutations)
    """
    a = (matrix[0, 0] + matrix[1, 1])/2
    b = (matrix[0, 0] - matrix[1, 1])/2
    c = (matrix[0, 1] + matrix[1, 0])/2
    d = (matrix[0, 1] - matrix[1, 0])/2
    return (1-a**2)/(b**2 + c**2 + d**2)

class IndexableGenerator(object):

    def __init__(self, it):
        self.it = iter(it)
        self.already_computed = []

    def __iter__(self):
        for elt in self.it:
            self.already_computed.append(elt)
            yield elt

    def __getitem__(self, index):
        try:
            max_idx = index.stop
        except AttributeError:
            max_idx = index
        n = max_idx - len(self.already_computed) + 1
        if n > 0:
            self.already_computed.extend(itertools.islice(self.it, n))
        return self.already_computed[index]


class PyOculusFixedPoint(Optimizable):
    """
    A FixedPoint is a point in the Poincar\'e section of a magnetic field
    that maps to itself after n field periods.
    Pyoculus contains routines to efficiently locate fixed points, and calculate
    their properties (residue, location, etc). This class provides an interface
    """

    @SimsoptRequires(newpyoculus, "This PyOculusFixedPoint class requres the additions by L. Rais in 1.0.0")
    def __init__(self,
                 pyocmap: "pyoculus.maps.CylindricalBfieldSection",
                 start_guess: Iterable[np.float64],
                 fp_order: int = 1,
                 finder_args: dict = dict(), 
                 parent_field: "simsopt.field.MagneticField" = None,
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
                   field: "simsopt.field.MagneticField",
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
    An optimizable that is the distance between two fixed points.
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
        return an optimizable child object whose J function is the distance
        between the location of the two fixed points at toroidal angle phi. 
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
    A ClinicConnection is a connection between two fixed points.
    It is the Simsopt-binding to the pyoculus Manifold class.
    It gives allows calculation of the turnstile flux, which is a measure
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
        when properties are changed.
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
        return np.abs(self._manifold.turnstile_areas).sum()
    
    def first_area(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.turnstile_areas[0])
    
    def second_area(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.turnstile_areas[1])
    
    def third_area(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.turnstile_areas[2])
    
    def fourth_area(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.turnstile_areas[3])

    def fifth_area(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.turnstile_areas[4])
    
    def sixth_area(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.turnstile_areas[5])

class SimpleIntegrator(Optimizable):
    """
    wrappers around a field to integrate trajectories with scipy's odeint.
    Can be used to integrate towards an endpoint, or get points equally spaced in phi
    along a field line.
    """

    def __init__(self, field, tol=1e-9):
        self.field = field
        self.TOL = tol
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def integration_fn(self, t, rz):
        """
        Integrand for simple field line following, ugly written to speed operations
        dR/dphi = R*B_R/B_phi,
        dZ/dphi = R*B_Z/B_phi
        """
        if np.any(np.isnan(rz)):
            return np.ones_like(rz)*np.nan
        R, Z = rz
        rphiz = np.array([R, t, Z])
        B = self.field.set_points_cyl(rphiz[None, :])
        B = self.field.B_cyl().flatten()
        B = B.flatten()
        return np.array([R*B[0]/B[1], R*B[2]/B[1]])

    @property
    def event_function(self):
        """
        event function that stops integration when the field is gets close to a coil.
        This is determined by the B_phi component getting smaller than 1e-3 the full field stength.
        Not setting points cause event is only called
        after function eval, so super fast.

        Abs and subtraction because zero crossing is not
        passed by integrator.
        """
        def _event(t, rz):
            return np.abs(self.field.B_cyl().dot(np.array([0,1.,0.])/np.linalg.norm(self.field.B()))) - 1e-3  # B_phi = 0
        _event.terminal = True
        return _event

    def integrate_in_phi_cyl(self, RZ_start, phi_start, phi_end):
        """
        Integrate the field line using scipy's odeint method
        """
        sol = solve_ivp(self.integration_fn, [phi_start, phi_end], RZ_start, events=self.event_function, method='RK45', rtol=self.TOL, atol=self.TOL)
        if not sol.success:
            return np.array(np.nan, np.nan)
        return sol.y[:, -1]

    def integrate_fieldlinepoints_RZ(self, RZ_start, phi_start, phi_end, n_points, endpoint=False, return_cartesian=False):
        """
        Integrate the field line using scipy's odeint method,
        to get a string of RZ points equally spaced in phi.

        Default returns cylindrical coordinates, can also
        return cartesian coordinates.
        """
        sol = solve_ivp(self.integration_fn, [phi_start, phi_end], RZ_start, t_eval=np.linspace(phi_start, phi_end, n_points, endpoint=endpoint), events=self.event_function, method='RK45', rtol=self.TOL, atol=self.TOL)
        if not sol.success:
            raise ObjectiveFailure("Failed to integrate field line")
        rphiz = np.array([sol.y[0, :], sol.t, sol.y[1, :]]).T
        if return_cartesian:
            return _rphiz_to_xyz(rphiz)
        else:
            return rphiz

    def integrate_fieldlinepoints_xyz(self, xyz_start, phi_total, n_points, endpoint=False, return_cartesian=True):
        """
        integrate a fieldline for a given toroidal distance, giving the start point in xyz. 
        """
        rphiz_start = _xyz_to_rphiz(xyz_start)
        RZ_start = rphiz_start[::2]
        phi_start = rphiz_start[1]
        phi_end = phi_start + phi_total
        points = self.integrate_fieldlinepoints_RZ(RZ_start, phi_start, phi_end, n_points, endpoint=endpoint, return_cartesian=return_cartesian)
        return points

    def integration_function3d(self, t, xyz):
        self.field.set_points(xyz[None,:])
        B = self.field.B().flatten()
        return B/np.linalg.norm(B)

    def integrate_3d_fieldlinepoints_xyz(self, xyz_start, l_total, n_points):
        """
        integrate a fieldline for a given toroidal distance, return the points in xyz
        """
        sol = solve_ivp(self.integration_function3d, [0, l_total], xyz_start, t_eval=np.linspace(0, l_total, n_points),  method='RK45', rtol=self.TOL, atol=self.TOL)
        return sol.y.T


    def _return_fieldline_iterator(self, RZ_start, phi_start, phi_distance):
        """
        Calculate the next point in a field line
        """
        def _fieldline_iterator():
            coords = RZ_start
            while True:
                coords = self.integrate_in_phi_cyl(coords, phi_start, phi_start + phi_distance)
                yield coords
        return IndexableGenerator(_fieldline_iterator())


    def simple_poincare(self, RZ_starts, phi_plane, phi_distance, n_points):
        """
        Calculate the points in a Poincar\'e section
        """
        poincare_points = []
        for RZ_start in RZ_starts:
            try:
                print(f"Integrating field line: {RZ_start}")
                poincare_points.append(np.array(self._return_fieldline_iterator(RZ_start, phi_plane, phi_distance)[:n_points]))
            except Exception as e:
                print(f"Failed to integrate field line: {RZ_start} because: {e}")
        return poincare_points







class SimpleFixedPoint_RZ(Optimizable):
    """
    The SimpleFixedPoint class is a very robust and simple method of optimizing for a fixed point at a given location.
    The cost is simply the distance that a field line is mapped away from its starting point after nfp field periods in an R,Z plane.

    Whereas PyoculusFixedPoint requires a fixed point to exist before it can be optimized, (This fixed point, change its location),
    this class can be used to generate a fixed point anywhere (This point in my section: make it a fixed point)
    useful for cold starts or when the fixed point is not known.

    The axis is a fixed point of the 1-field-period map (and every multiple map as well).
    In an N nfp stellarator, a ntor/mpol island chain is a fixed point of mpol field periods.
    For example to try to put an x- or o-point of the 5/5 island chain at (R, Z) in the phi=0 plane:
    ```
    [code to generate your field]
    fp_at_RZ = SimpleFixedPoint(field, (R, Z), 5, 5, 0)
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
            nfp: Number of field period
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
        Calculate the distance from the target fixed point
        """
        phi_end = self.phi0 + 2*np.pi*self.integration_nfp/self.field_nfp
        try:
            dist_mapped = np.linalg.norm(self.integrator.integrate_in_phi_cyl(self.RZ, self.phi0, phi_end)[::2] - self.RZ)
        except Exception as e:
            raise ObjectiveFailure("Failed to integrate field line") from e
        return dist_mapped

    # def dJ(self): can be made with integration of the derivatives



class SimpleMapJac(SimpleIntegrator, Optimizable):
    """
    The SimpleMapJac class is a simple method of calculating the jacobian of the field line map using finite differencing. 
    The J() returns the trace of the jacobian, for residue-like optimisations (though if the point is not a fixed point, you might get strange results)

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



class SimpleEllipticity(Optimizable):
    """
    Wrapper around SimpleMapJac to return the ellipticity of the map jacobian
    as described by Greene 1968: https://doi.org/10.1063/1.1664639
    
    The ellipticity is +1 for cirular elliptic fixed points, 0 when parabolic, and negative
    when hyperbolic. Ellipticity of -1 correspoinds to perpendicular intersection of stable and unstable manifolds.
    """
    def __init__(self, simplemapjac: SimpleMapJac):
        self.simplemapjac = simplemapjac
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[simplemapjac])
    
    def J(self):
        return self.simplemapjac.ellipticity()
    

class SimpleResidue(Optimizable):
    """
    Wrapper around SimpleMapJac to return the Greenes residue of the map jacobian
    as described by Greene 1968: https://doi.org/10.1063/1.1664639
    The residue is 0 for regular parabolic fixed points, and positive
    for regular hyperbolic fixed points. 
    Negative residue can be elliptic or alternating hyperbolic. 
    """
    def __init__(self, simplemapjac: SimpleMapJac):
        self.simplemapjac = simplemapjac
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[simplemapjac])  
    
    def J(self):
        return self.mapjac.residue()
    
class SimpleTrace(Optimizable):
    """
    Wrapper around SimpleMapJac to return the trace of the map jacobian
    """
    def __init__(self, simplemapjac: SimpleMapJac):
        self.simplemapjac = simplemapjac
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[simplemapjac])  

    def J(self):
        return self.simplemapjac.J()  