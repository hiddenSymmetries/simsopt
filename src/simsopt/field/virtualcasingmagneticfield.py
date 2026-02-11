from typing import TYPE_CHECKING
import warnings
import logging

from nptyping import NDArray
import numpy as np
from scipy.spatial import cKDTree

from .magneticfield import MagneticField
from .._core.json import GSONDecoder

from virtual_casing import VirtualCasing
from ..mhd.vmec import Vmec
from ..geo.surface import best_nphi_over_ntheta
from ..mhd.vmec_diagnostics import B_cartesian
from .._core.optimizable import Optimizable

if TYPE_CHECKING:
    from simsopt.geo.surface import Surface



logger = logging.getLogger(__name__)


__all__ = ['VirtualCasingField',]


class VirtualCasingField(MagneticField):
    """
    Calculates the magnetic field by currents inside or outside a toroidal surface using the virtual casing principle. 
    The calculation is impemented using an efficient high-order 
    singular quadrature scheme by Malhotra et al. (https://iopscience.iop.org/article/10.1088/1361-6587/ab57f4). 

    The calculation requres a surface on which the total field is 
    known, and can compute the field in a vacuum region on either side. 
    For example currents inside can be modeled by using ``simsopt.mhd.vmec_diagnostics.B_cartesian()`` to evaluate the total field on the surface of a VMEC equilibrium. 

    Alternatively, the currents in the coil can be projected onto the surface in a stage-two calculation. 

    
    Attributes:
        on_surface_tol (float): Distance threshold (in meters) for using precomputed
            on-surface values instead of off-surface computation.
        gamma (ndarray): Source grid positions, shape (n_trgt, 3).
        B_external_onsurf (ndarray): Precomputed B field at source grid, shape (n_trgt, 3).
    """

    def __init__(self, surf:'Surface', total_B_on_surf:'NDArray', digits:int = 7, src_nphi=None, src_ntheta=None, trgt_nphi=None, trgt_ntheta=None, max_upsampling = 10, currents_inside = True):
        """
        Initialize a plain VirtualCasingField

        Args: 
            surf: Surface on which the virtual casing integral is evaluated. 
            total_B_on_surf: Total magnetic field evaluated on the source surface, shape (src_nphi * src_ntheta, 3), for example evaluated by simsopt.mhd.vmec_diagnostics.B_cartesian() on the source grid.
            digits: Approximate number of digits of precision for the calculation.
            src_nphi: Number of grid points toroidally for the input of the calculation. If None, taken from surf.
            src_ntheta: Number of grid points poloidally for the input of the calculation. If None, taken from surf.
            trgt_nphi: Number of grid points toroidally for the output of the calculation. If None, taken from src_nphi.
            trgt_ntheta: Number of grid points poloidally for the output of the calculation. If None, taken from src_ntheta.
            max_upsampling: Maximum upsampling factor for the quadrature scheme. Higher values may be needed for higher accuracy, but will increase computation time.
            currents_inside: If True, compute the field from currents inside the surface. If False, compute the field from currents outside the surface. This determines which side of the surface is considered the "source" side for the virtual casing calculation.
        """
        # input juggling for versatile calling
        if src_nphi is None:
            src_nphi = surf.nphi
        if src_ntheta is None:
            src_ntheta = surf.ntheta
        if trgt_nphi is None:
            trgt_nphi = src_nphi
        if trgt_ntheta is None:
            trgt_ntheta = src_ntheta
        stellsym = surf.stellsym
        nfp = surf.nfp

        surf_range = 'half period' if stellsym else 'field period'
        self._surf = surf.copy(nphi=src_nphi, ntheta=src_ntheta, range=surf_range)  # secret, immutable copy

        # store inputs for serialization
        self._digits = digits
        self._src_nphi = src_nphi
        self._src_ntheta = src_ntheta
        self._trgt_nphi = trgt_nphi
        self._trgt_ntheta = trgt_ntheta
        self._max_upsampling = max_upsampling


        self._gamma1d = surf.gamma().reshape(-1, 3).flatten(order='C')  # Shape: (src_nphi * src_ntheta * 3,)
        self._B1d = total_B_on_surf.reshape(-1, 3).flatten(order='C')  # Shape: (src_nphi * src_ntheta * 3,)

        self.vcasing = VirtualCasing()
        self.vcasing.setup(
            digits, nfp, stellsym,
            src_nphi, src_ntheta, self._gamma1d,
            src_nphi, src_ntheta,
            trgt_nphi, trgt_ntheta)

        onsurf_b_from_internal_currents = self.vcasing.compute_internal_B(self._B1d) 

        if currents_inside is True:
            self.eval_field = self.vcasing.compute_internal_B_offsurf
            self.eval_grad_field = self.vcasing.compute_internal_B_offsurf_gradient
        else: 
            self.eval_field = self.vcasing.compute_external_B_offsurf
            self.eval_grad_field = self.vcasing.compute_external_B_offsurf_gradient

        super().__init__(self)
    
    def _B_impl(self, B):
        """
        Evaluate the magnetic field at the points that have previously been set using ``self.set_points()``. This method is called by the parent class's ``B()`` method, which handles caching and interpolation."""
        points = self.get_points_cart_ref()
        n_points = len(points)
        
        points1d = points.flatten(order='F')
        Bext1d = np.array(self.eval_field(self.B1d, points1d))
        Bext = Bext1d.reshape(n_points, 3, order='F')
        B[:] = Bext
        return
    
    def _dB_by_dX_impl(self, dB_by_dX):
        """
        Evaluate the derivatives of the magnetic field at the points that have previously been set using ``self.set_points()``. This method is called by the parent class's ``dB_by_dX()`` method, which handles caching and interpolation."""
        points = self.get_points_cart_ref()
        n_points = len(points)
        
        points1d = points.flatten(order='F')
        dB_by_dX_ext1d = np.array(self.eval_grad_field(self.B1d, points1d))
        dB_by_dX_ext = dB_by_dX_ext1d.reshape(n_points, 3, 3, order='F')
        dB_by_dX[:] = dB_by_dX_ext


class VmecVirtualCasingField(VirtualCasingField):
    """
    A VirtualCasingField coupled to a VMEC object, with logic to update when the degrees-of-freedom change. 
    """
    
    def __init__(self, vmec: Vmec);
        pass
        #calculate vmec properties needed by virtual_casing
        #self.get_vdata_from_vmec(vmec)
        #super.__init__(properties)
        #self.need_to_recompute_data = False
        # Optimizable.__init__(parent=...)

    def recompute_bell(self, parent=None):
        self.need_to_recompute_data = True
    
    def recompute_data(self):
    
    @staticmethod
    def get_vcdata_from_vmec(vmec, ntheta, nzeta):
        """
        """
        pass

