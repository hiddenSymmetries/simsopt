# fixedpoint_turnstile.py
# simsopt-pyoculus interface to utilize pyoculus methods for locating fixed points
# and using their properties (Residue, location, etc) in optimizations. 
# Also provides methods for calculating turnstile areas (measure for stochastic transport)
# Chris Smiet Feb 2025  christopher.smiet@epfl.ch

import numpy as np
from scipy.integrate import solve_ivp
try : 
    from pyoculus.solvers import FixedPoint, Manifold
    from pyoculus.fields import SimsoptBfield
    from pyoculus.maps import CylindricalBfieldSection
except ImportError:
    newpyoculus = False
from ..field import MagneticField
from .._core import Optimizable
from .._core.dev import SimsoptRequires
from .._core.util import ObjectiveFailure

from numpy.typing import NDArray
from typing import Union
from pyoculus import __version__ as pyoc_version

if pyoc_version >= '1.0.0':
    newpyoculus = True
else:
    newpyoculus = False


__all__ = ['PyOculusFixedPoint', 'ClinicConnection', 'SimpleFixedPoint_RZ', 'SimpleIntegrator']


@SimsoptRequires(newpyoculus, "This PyOculusFixedPoint class requres the additions by L. Rais in 1.0.0")
class PyOculusFixedPoint(Optimizable):
    """
    A FixedPoint is a point in the Poincar\'e section of a magnetic field
    that maps to itself after n field periods. 
    Pyoculus contains routines to efficiently locate fixed points, and calculate
    their properties (residue, location, etc). This class provides an interface
    """

    def __init__(self, field: MagneticField, 
                 start_guess: NDArray[np.float64], 
                 nfp: int, 
                 phi_0: float = 0., 
                 fp_order: int = 1, 
                 axis: Union[bool, NDArray[np.float64]] = False,
                 integration_tol: float = 1e-9, 
                 finder_tol: float = 1e-9, 
                 integration_args: dict = None, 
                 finder_args: dict = None):
        """
        Initialize a FixedPoint object
        Args:
            field: MagneticField object
            start_guess: initial guess for the fixed point location (R, Z). 
            nfp: Number of field period
            phi_0: Toroidal angle of the fixed point
            fp_order: Order of the fixed point
            axis: Location of the axis, len 2 array. The axis is also a fixed point, necessary if the poloidal mode number (winding with the axis) is required. 
                   if False a nonsense axis will be given to pyoculus.
            integration_tol: Tolerance for integration
            finder_tol: Tolerance for fixed point finder
            integration_args: Additional arguments for constructing the Poincar\'e section. 
                    See pyoculus.maps.CylindricalBfieldSection and pyoculus.maps.IntegratedMap for details.
            finder_args: Additional arguments for fixed point finder.
                    See pyoculus.solvers.FixedPoint for details.
        """
        self.field = field
        self.start_guess = start_guess
        self.nfp = nfp
        self.phi_0 = phi_0
        self.fp_order = fp_order
        self.axis = axis
        self.integration_tol = integration_tol
        self.finder_tol = finder_tol
        self.integration_args = integration_args
        self.finder_args = finder_args

        self._oculus_field_representation = SimsoptBfield(nfp, field)  # pyoculus' interfacing class
        if axis:  # create the map with an axis so that rotational transform and such can be calculated. 
            if not isinstance(axis, np.ndarray):  # if axis evaluates true, start with a random point (bound to fail)
                axis = np.array([1., 0.])
            self._map = CylindricalBfieldSection.without_axis(self._oculus_field_representation, phi0=phi_0, guess=axis, **integration_args)
        else: 
            self._map = CylindricalBfieldSection(self._oculus_field_representation, R0=0, Z0=0, phi0=phi_0, **integration_args)  # create the map with 'axis' at the origin (methods calculating rotational transform will fail)
        self._fixed_point = FixedPoint(self._map)  # create the fixed point
        self._fixed_point.find(self.fp_order, start_guess, tol=finder_tol, **finder_args)  # find it
        self._current_location = np.copy(self._fixed_point.coords[0])
        self._refind_fp = False

        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def recompute_bell(self):
        self._fixed_point.successful = False
        self._refind_fp = True
    
    def refind(self): 
        try: 
            self._fixed_point.find(self.fp_order, self._current_location, tol=self.finder_tol, **self.finder_args)
        except Exception as e:
            raise ObjectiveFailure("Failed to find fixed point") from e
        if self.axis:
            try:  
                self._map.find_axis()  # re-find the axis
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
        return np.array(RZ[0]*np.cos(self.phi_0), RZ[0]*np.sin(self.phi_0), RZ[1])

    def R(self):
        return self.loc()[0]
    
    def Z(self):
        return self.loc()[1]
    
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
    
    def Trace(self):
        return np.trace(self.jacobian)
    

@SimsoptRequires(newpyoculus, "This PyOculusFixedPoint class requres the additions by L. Rais in 1.0.0")
class ClinicConnection(Optimizable): 
    """
    A ClinicConnection is a connection between two fixed points.
    It is the Simsopt-binding to the pyoculus Manifold class. 
    It gives allows calculation of the turnstile area, which is a measure
    for stochastic transport. 

    fixed points must be of the same map, etc!
    """
    def __init__(self, 
                 fp1: PyOculusFixedPoint, 
                 fp2: PyOculusFixedPoint,
                 dir1: NDArray = None,
                 dir2: NDArray = None,
                 first_stable=True,
                 ):
        """
        Initialize a ClinicConnection
        """
        if fp1._map != fp2._map:
            raise ValueError("Fixed points must be of the same map")
        if fp1.nfp != fp2.nfp:
            raise ValueError("Fixed points must have the same nfp")  # could be allowed in future pyoculus
        self.fp1 = fp1
        self.fp2 = fp2
        self._dir1 = dir1
        self._dir2 = dir2
        self._first_stable = first_stable
        self._manifold = Manifold(fp1._fixed_point, fp2._fixed_point, dir1, dir2, first_stable)
        self.set_true_directions()
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[fp1, fp2])
        self._invalidate = False

    def set_true_directions(self):
        """
        set the true directions for the manifold, such that in optimization the next
        manifold is found that is closest to the previous one.
        """
        if self._first_stable: 
            self._truedir1 = self._manifold.stable_direction
            self._truedir2 = self._manifold.unstable_direction
        else:
            self._truedir1 = self._manifold.unstable_direction
            self._truedir2 = self._manifold.stable_direction
    
    def recompute_bell(self):
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
    
    def J(self):
        if self._need_to_reset:
            self.reset()
        return np.abs(self._manifold.areas[0])

    
class SimpleIntegrator(Optimizable):
    """
    wrappers around a field to integrate trajectories with scipy's odeint. 
    Can be used to integrate towards an endpoint, or get points equally spaced in phi 
    along a field line. 
    """

    def __init__(self, field):
        self.field = field
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def integration_fn(self, t, rz):
        """
        Integrand for simple field line following, ugly written to speed operations
        dR/dphi = R*B_R/B_phi, 
        dZ/dphi = R*B_Z/B_phi
        """
        R, Z = rz
        rphiz = np.array([R, t, Z])
        B = self.field.set_points_cyl(rphiz[None, :])
        B = self.field.B_cyl().flatten()
        B = B.flatten()
        return np.array([R*B[0]/B[1], R*B[2]/B[1]])
        
    def integrate_in_phi(self, RZ_start, phi_start, phi_end):
        """
        Integrate the field line using scipy's odeint method
        """
        sol = solve_ivp(self.integration_fn, [phi_start, phi_end], RZ_start, method='LSODA', rtol=1e-9, atol=1e-9, max_step=1e5)
        return sol.y[:, -1] 
    
    def integrate_fieldlinepoints(self, RZ_start, phi_start, phi_end, n_points):
        """
        Integrate the field line using scipy's odeint method
        """
        sol = solve_ivp(self.integration_fn, [phi_start, phi_end], RZ_start, t_eval=np.linspace(phi_start, phi_end, n_points), method='LSODA', rtol=1e-9, atol=1e-9, max_step=1e5)
        return sol.y


class SimpleFixedPoint_RZ(SimpleIntegrator, Optimizable):
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
    def __init__(self, field: MagneticField, RZ: NDArray[np.float64], field_nfp: int, integration_nfp: int, phi_0: float = 0.):
        """
        Initialize a SimpleFixedPoint
        Args:
            field: MagneticField object
            RZ: the R and Z-coordinate where a fixed point is required
            nfp: Number of field period
            phi_0: Toroidal angle of the fixed point
        """
        self.field = field
        self.RZ = RZ
        self.field_nfp = field_nfp
        self.integration_nfp = integration_nfp
        self.phi_0 = phi_0
        self.integrator = SimpleIntegrator(field, RZ, field_nfp, integration_nfp, phi_0)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        """
        Calculate the distance from the target fixed point
        """
        phi_end = self.phi_0 + 2*np.pi*self.integration_nfp/self.field_nfp
        return np.linalg.norm(self.integrator.integrate(self.RZ, self.phi_0, phi_end) - self.RZ)
    
    # def dJ(self): can be made with integration of the derivatives
        
