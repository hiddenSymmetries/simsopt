# field_topology_optimizables.py
# simsopt-pyoculus interface to utilize pyoculus methods for locating fixed points
# and using their properties (Residue, location, etc) in optimizations.
# Also provides methods for calculating turnstile areas (measure for stochastic transport)
# Chris Smiet Feb 2025  christopher.smiet@epfl.ch
#

import numpy as np
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
from .field_topology_optimizables_integrated import ellipticity
from typing import Union, Iterable, TYPE_CHECKING
import itertools

if TYPE_CHECKING:
    from ..field import MagneticField

try:
    newpyoculus = int(str(pyoc_version).split('.')[0]) >= 1
except Exception:
    newpyoculus = False


__all__ = ['PyOculusFixedPoint',
           'PyOculusFixedPointLocationTarget',
           'PyOculusEllipticityTarget', 'PyOculusResidueTarget',
           'PyOculusTraceTarget', 'PyOculusTraceInRangeTarget',
           'PyOculusLocationAtOtherAngleTarget',
           'PyOculusTwoFixedPointLocationDifference',
           'ClinicConnection',
          ]


class PyOculusFixedPoint(Optimizable):
    """
    A FixedPoint is a point in the Poincar\'e section of a magnetic field
    that maps to itself after n field periods.
    PyOculus contains routines to efficiently locate fixed points, and calculate
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
        # set good default finder args if not given:
        finder_args['method'] = finder_args.get('method', 'scipy.root')
        if finder_args['method'] == 'scipy.root':
            finder_args['options'] = finder_args.get('options', {'xtol':1e-10, 'factor':1e-2})
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
        return np.array([RZ[0]*np.cos(self._pyocmap.phi0), 
                         RZ[0]*np.sin(self._pyocmap.phi0), 
                         RZ[1]]
                         )

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
        return ellipticity(self.jacobian())

    
class PyOculusFixedPointLocationTarget(Optimizable):
    """
    Optimizable that evaluates the distance between a PyOculusFixedPoint and a target location.
    """
    def __init__(self, fp: PyOculusFixedPoint, target_RZ: NDArray[np.float64]):
        """
        Initialize a PyOculusFixedPointTargetLocation
        Args:
            fp: PyOculusFixedPoint object
            target_RZ: target location (R, Z)
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp])
        self.fp = fp
        self.target_RZ = target_RZ
    
    def J(self):
        return np.linalg.norm(self.fp.loc_RZ() - self.target_RZ)


class PyOculusEllipticityTarget(Optimizable):
    """
    Optimizable that evaluates the difference between the PyOculusFixedPoint's ellipticity and a target ellipticity. 
    The ellipticity of a fixed point is calculated from the map
    Jacobian, as described  by Greene 1968: 
    https://doi.org/10.1063/1.1664639
    The ellipticity is +1 for cirular elliptic fixed points, 0 when parabolic, and negative
    when hyperbolic. Ellipticity of -1 correspoinds to perpendicular intersection of stable and unstable manifolds.
    """
    def __init__(self, fp: PyOculusFixedPoint, target_ellipticity: float = 0.0):
        """
        Initialize a PyOculusFixedPointEllipticityTarget
        Args:
            fp: PyOculusFixedPoint object
            target_ellipticity: target ellipticity
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp])
        self.fp = fp
        self.target_ellipticity = target_ellipticity

    def J(self):
        return self.fp.ellipticity() - self.target_ellipticity
    
class PyOculusResidueTarget(Optimizable):
    """
    Optimizable that evaluates the Greenes residue of a PyOculusFixedPoint.
    """
    def __init__(self, fp: PyOculusFixedPoint, target_residue: float = 0):
        """
        Initialize a PyOculusResidueTarget
        Args:
            fp: PyOculusFixedPoint object
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp])
        self.fp = fp
        self.target_residue = target_residue

    def J(self):
        return np.linalg.norm(self.fp.residue() - self.target_residue)

class PyOculusTraceTarget(Optimizable):
    """
    Optimizable that evaluates the trace of the map Jacobian of a PyOculusFixedPoint.
    """
    def __init__(self, fp: PyOculusFixedPoint, target_trace: float = 0, penalize="difference"):
        """
        Initialize a PyOculusTraceOptimizable
        Args:
            fp: PyOculusFixedPoint object
            target_trace: target trace value
            penalize: penalization strategy, can be "difference" (default), "above", or "below"
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp])
        self.fp = fp
        self.target_trace = target_trace
        self.penalize = penalize

    def J(self):
        if self.penalize == "above":
            return np.maximum(0, self.fp.trace() - self.target_trace)
        elif self.penalize == "below":
            return np.minimum(0, self.fp.trace() - self.target_trace)
        else:
            return self.fp.trace() - self.target_trace

class PyOculusTraceInRangeTarget(Optimizable):
    """
    Optimizable that evaluates a penalty on the trace of the map Jacobian of a PyOculusFixedPoint.
    """
    def __init__(self, fp: PyOculusFixedPoint, min_trace: float = -2, max_trace: float = 2):
        """
        Initialize a PyOculusTraceInRangeTarget
        Args:
            fp: PyOculusFixedPoint object
            min_trace: minimum acceptable trace value
            max_trace: maximum acceptable trace value
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp])
        self.fp = fp
        self.min_trace = min_trace
        self.max_trace = max_trace

    def J(self):
        trace = self.fp.trace()
        return max(0, self.min_trace - trace) + max(0, trace - self.max_trace)
    
class PyOculusLocationAtOtherAngleTarget(Optimizable):
    """
    Optimizable that evaluates the distance between the location of a PyOculusFixedPoint
    at a different toroidal angle and a target location.
    """
    def __init__(self, fp: PyOculusFixedPoint, other_phi: float, target_RZ: NDArray[np.float64]):
        """
        Initialize a PyOculusLocationAtOtherAngleTarget
        Args:
            fp: PyOculusFixedPoint object
            other_phi: toroidal angle at which to evaluate the location
            target_RZ: target location (R, Z)
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp])
        self.fp = fp
        self.other_phi = other_phi
        self.target_RZ = target_RZ

    def J(self):
        return np.linalg.norm(self.fp.loc_at_other_angle(self.other_phi) - self.target_RZ)


class PyOculusTwoFixedPointLocationDifference(Optimizable):
    """
    Target the distance between two fixed points.
    """
    def __init__(self, fp1: PyOculusFixedPoint, fp2: PyOculusFixedPoint, target_distance: float, other_phi: float = 0, penalize="difference"):
        """
        Initialize a TwoFixedPointDifference
        Args:
            fp1: First fixed point
            fp2: Second fixed point
            target_distance: target distance between the two fixed points
            other_phi: toroidal angle at which to evaluate the distance (default 0)
            penalize: penalization strategy, can be "difference" (default), "above", or "below"
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp1, fp2])
        self.fp1 = fp1
        self.fp2 = fp2
        self.other_phi = other_phi
        self.target_distance = target_distance
        self.penalize = penalize

    def J(self):
        if self.other_phi == 0:
            loc1 = self.fp1.loc_RZ()
            loc2 = self.fp2.loc_RZ()
        else:
            loc1 = self.fp1.loc_at_other_angle(self.other_phi)
            loc2 = self.fp2.loc_at_other_angle(self.other_phi)
        if self.penalize == "difference":
            return np.linalg.norm(loc1 - loc2) - self.target_distance
        elif self.penalize == "above":
            return np.maximum(0, np.linalg.norm(loc1 - loc2) - self.target_distance)
        elif self.penalize == "below":
            return np.minimum(0, np.linalg.norm(loc1 - loc2) - self.target_distance)
    
    




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