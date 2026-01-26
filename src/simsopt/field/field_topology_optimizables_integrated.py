# field_topology_optimizables.py
# simsopt-pyoculus interface to utilize pyoculus methods for locating fixed points
# and using their properties (Residue, location, etc) in optimizations.
# Also provides methods for calculating turnstile areas (measure for stochastic transport)
# Chris Smiet Feb 2025  christopher.smiet@epfl.ch
#

import numpy as np
from .._core import Optimizable
from .._core.util import ObjectiveFailure
from numpy.typing import NDArray
from typing import Union, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..field import MagneticField, ScipyFieldlineIntegrator, Integrator

__all__ = ['IntegratorFixedPoint', 'FDMapJac', 'FDEllipticity', 'FDResidue', 'FDTrace', 'ellipticity']


def ellipticity(jacobian_matrix):
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
    def __init__(self, integrator: "ScipyFieldlineIntegrator", RZ: NDArray[np.float64], integration_nfp: int, phi0: float = 0.):
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

    # def dJ(self): can be made with adjoint methods

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