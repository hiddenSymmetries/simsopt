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


__all__ = ['VirtualCasingField', 'VmecVirtualCasingField']


class VirtualCasingField(MagneticField):
    """
    Calculates the magnetic field by currents inside or outside a toroidal surface using the virtual casing principle. 
    The calculation is impemented using an efficient high-order 
    singular quadrature scheme by Malhotra et al. (https://iopscience.iop.org/article/10.1088/1361-6587/ab57f4).

    There are two common use cases for this class: 
        1) To compute the field on the surface, in which case the method onsurf_field_due_currents should be used. This is often used in stage-two
        opimization where all the complexity of the coils can be projected to the surface.
        2) To compute the field in a vacuum region off the surface, where the evaluation points are in vacuum, and the field is due to currents on the opposite side 
         of the surface. This can be used to compute the total field in the vacuum region by summing with the Biot-Savart field from coils. 

    The calculation requres a surface on which the total field is 
    known, and can compute the field on that surface due to 
    currents external or internal to it, as well as
    the field in a in a vacuum region on either side. 

    Important Attributes: 
        onsurf_field_due_to_currents: The magnetic field on the surface due 
            to the currents on the "source" side of the surface. Cached
            after first calculation, updated when surface or total field changes.
        B: The magnetic field at the points that have previously been set
            using ``self.set_points()``. This method is called by the parent
            class's ``B()`` method, which handles caching and interpolation.
        
        surf: The surface on which the virtual casing integral is 
            evaluated. The class can be instantiated with any surface, but
            will be cast into SurfaceRZFourier with appropriate symmetry and quadrature points (from ntheta and nphi if provided)
        trgt_surf: the SurfaceRZFourier with quadrature points
            corresponding to where the on-surface virtual casing field is evaluated.

    Note: 
        The VirtualCasing principle can both be used to compute the field
        ON the surface, for which the method onsurf_field_due_to_currents
        needs to be used, as well as in a vacuum region off the surface, for which the method B must be used. 
        Evaluating B on the surface will give incorrect results, as kernel singularites are not handled.
        No checks are performed

    Note on symmetry and grid points from BIEST documentation:
     * If you do not exploit stellarator symmetry, then half_period
     * is set to false.  In this case the grids in the toroidal angles
     * begin at phi=0.  The grid spacing is 1 / (NFP * Nt), and there
     * is no point at the symmetry plane phi = 1 / NFP.
     *
     * If you do wish to exploit stellarator symmetry, set half_period
     * to true. In this case the toroidal grids are each shifted by
     * half a grid point, so there is no grid point at phi = 0. The
     * phi grid for the surface shape has points at 0.5 / (NFP * Nt),
     * 1.5 / (NFP * Nt), ..., (Nt - 0.5) / (NFP * Nt). The phi grid
     * for the output B_external has grid points at 0.5 / (NFP *
     * trg_Nt), 1.5 / (NFP * trg_Nt), ..., (trg_Nt - 0.5) / (NFP *
     * Nt), and similarly for the input B field with trg -> src.
    """

    def __init__(self, surf: 'Surface', total_B_in_surf: 'NDArray', currents_inside=True, src_nphi=None, src_ntheta=None, trgt_nphi=None, trgt_ntheta=None, digits: int = 7, max_upsampling=1000):
        """
        Initialize a plain VirtualCasingField

        Args: 
            surf: Surface on which the virtual casing integral is evaluated. 
            total_B_on_surf: Total magnetic field evaluated on the source surface, shape (src_nphi * src_ntheta, 3), for example evaluated by simsopt.mhd.vmec_diagnostics.B_cartesian() on the source grid. 
                Inherited classes are allowed to pass None provided they have methods that compute this when recompute_bell is rung.
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
            src_nphi = len(surf.quadpoints_phi)
        if src_ntheta is None:
            src_ntheta = len(surf.quadpoints_theta)
        if trgt_nphi is None:
            trgt_nphi = src_nphi
        if trgt_ntheta is None:
            trgt_ntheta = src_ntheta
        self.stellsym = surf.stellsym
        self.nfp = surf.nfp

        self.surf_range = 'half period' if self.stellsym else 'field period'

        if total_B_in_surf is None: 
            total_B_in_surf = np.zeros(src_nphi*src_ntheta*3) # will be set at first eval, but we need to pass something to the C++ side for now.

        # store inputs for serialization
        self._digits = digits
        self._total_B_in_surf = total_B_in_surf
        self._src_nphi = src_nphi
        self._src_ntheta = src_ntheta
        self._trgt_nphi = trgt_nphi
        self._trgt_ntheta = trgt_ntheta
        self._max_upsampling = max_upsampling
        self.surf = surf.copy(nphi=src_nphi, ntheta=src_ntheta, range=self.surf_range)  # uses setter, extracts gamma1d

        self.vcasing = VirtualCasing()
        self._onsurf_field = None

        if currents_inside is True:
            self.field_eval_fn = self.vcasing.compute_internal_B_offsurf
        else: 
            self.field_eval_fn = self.vcasing.compute_external_B_offsurf

        super().__init__()

    @property
    def surf(self):
        return self._surf
    
    @surf.setter
    def surf(self, value):
        self._surf = value.copy(nphi=self._src_nphi, ntheta=self._src_ntheta, range=self.surf_range)
        # new surface, new points to evaluate on.
        # surface returns array nxmx3, we need x11...xnm, y11...ynm z... so we just move the last axis to the front and then ravel
        self._gamma1d = np.moveaxis(self._surf.gamma(), -1, 0).ravel()
        self.need_to_recompute_data = True
    
    @property
    def trgt_surf(self):
        """
        Surface with quadrature points corresponding to where
        the onsurf evaluation is performed
        """
        trgt_surf = self._surf.copy(nphi=self._trgt_nphi, ntheta=self._trgt_ntheta, range=self.surf_range)
        return trgt_surf
    
    @trgt_surf.setter
    def trgt_surf(self, value):
        raise ValueError('trgt_surf is determined by the source surface and the number of grid points, cannot be set directly. Set the source surface or the number of grid points instead.')
    
    @property 
    def total_B_in_surf(self):
        return self._total_B_in_surf
    
    @total_B_in_surf.setter
    def total_B_in_surf(self, value):
        self._total_B_in_surf = value
        self.need_to_recompute_data = True

    def recompute_data(self):
        """
        Surface or total field has changed
        """
        self._onsurf_field_ext = None # reset cached field on surface
        self._onsurf_field_int = None # reset cached field on surface
        self.vcasing.setup(
            self._digits, self.nfp, self.stellsym,
            self._src_nphi, self._src_ntheta, self._gamma1d,
            self._src_nphi, self._src_ntheta,
            self._trgt_nphi, self._trgt_ntheta)
        self.need_to_recompute_data = False
    
    @property
    def onsurf_field_due_ext(self):
        """
        The cartesian magnetic field on the surface due imagining the currents on the outside of the surface. Cached after first calculation, updated when surface or total field changes.
        Will have shape (trgt_nphi, trgt_ntheta, 3).
        """
        if self.need_to_recompute_data:
            self.recompute_data()
        if self._onsurf_field_ext is None:
            onsurf_field_1d = np.array(self.vcasing.compute_external_B(self._total_B_in_surf))
            self._onsurf_field_ext = onsurf_field_1d.reshape(3, self._trgt_nphi, self._trgt_ntheta).transpose(1, 2, 0)
        return self._onsurf_field_ext

    @property
    def onsurf_field_due_int(self):
        """
        The cartesian magnetic field on the surface due imagining the currents on the inside of the surface. Cached after first calculation, updated when surface or total field changes.
        Will have shape (trgt_nphi, trgt_ntheta, 3).
        """
        if self.need_to_recompute_data:
            self.recompute_data()
        if self._onsurf_field_int is None:
            onsurf_field_1d = np.array(self.vcasing.compute_internal_B(self._total_B_in_surf))
            self._onsurf_field_int = onsurf_field_1d.reshape(3, self._trgt_nphi, self._trgt_ntheta).transpose(1, 2, 0)
        return self._onsurf_field_int
    
    @property
    def Bnormal_due_ext(self):
        """
        The normal component of the magnetic field on the surface due to the currents on the "source" side of the surface. Cached after first calculation, updated when surface or total field changes.
        Will have shape (trgt_nphi, trgt_ntheta).
        """
        return np.sum(self.onsurf_field_due_ext * self.trgt_surf.normal(), axis=-1)
    

    @property
    def Bnormal_due_int(self):
        """
        The normal component of the magnetic field on the surface due to the currents on the "source" side of the surface. Cached after first calculation, updated when surface or total field changes.
        Will have shape (trgt_nphi, trgt_ntheta).
        """
        return np.sum(self.onsurf_field_due_int * self.trgt_surf.normal(), axis=-1)

    
    def _B_impl(self, B):
        """
        Evaluate the magnetic field at the points that have previously been set using ``self.set_points()``. This method is called by the parent class's ``B()`` method, which handles caching and interpolation."""
        if self.need_to_recompute_data:
            self.recompute_data()
        points = self.get_points_cart()
        n_points = len(points)
        
        # virtual casing expects points in the form x11...xnm, y11...ynm z... so we just move the last axis to the front and then ravel
        points1d = np.moveaxis(points, -1, 0).ravel()
        calculation_result_1d = np.array(self.field_eval_fn(self.total_B_in_surf, points1d, max_Np=self._max_upsampling, max_Nt=self._max_upsampling))
        # Result is [Bx1...Bxn, By1...Byn, Bz1...Bzn], reshape to (n_points, 3)
        field = calculation_result_1d.reshape(3, n_points).T
        B[:] = field
        return
    


class VmecVirtualCasingField(VirtualCasingField):
    """
    A VirtualCasingField coupled to a VMEC object, with logic to update when the degrees-of-freedom change. 
    The field is is only valid outside of the VMEC LCFS, and can be 
    used to calculate the field in the vacuum region by summing with 
    the Biot-Savart field from the coils.

    The calculation is impemented using an efficient high-order 
    singular quadrature scheme by Malhotra et al. (https://iopscience.iop.org/article/10.1088/1361-6587/ab57f4). 
    """
    
    def __init__(self, vmec: Vmec, src_nphi=None, src_ntheta=None, trgt_nphi=None, trgt_ntheta=None, digits: int = 7, max_upsampling=10):
        """
         Initialize a VmecVirtualCasingField, which is a decorated VirtualCasingField that updates when the VMEC object (or its parents) change."""
        self._vmec = vmec
        tmp_surf = vmec.boundary
        this_range = 'half period' if tmp_surf.stellsym else 'field period'

        super().__init__(surf=tmp_surf,
                         total_B_in_surf= None,
                         src_nphi=src_nphi, 
                         src_ntheta=src_ntheta, 
                         trgt_nphi=trgt_nphi,
                         trgt_ntheta=trgt_ntheta,
                         max_upsampling=max_upsampling,
                         digits=digits,
                         currents_inside=True)
        Optimizable.__init__(self, x0=[], depends_on=[self._vmec,])
        self.need_to_recompute_data = True


    def recompute_bell(self, parent=None):
        self.need_to_recompute_data = True
    
    def recompute_data(self):
        """
        run if evaluation happens and parent (VMEC object) has changed
        """
        self._vmec.run()
        # the boundary might have changed, so we update
        self.surf = self._vmec.boundary  # uses parent setter, updates gamma1d
        self.total_B_in_surf = self.B1d_from_vmec(self.vmec, quadpoints_phi=self.surf.quadpoints_phi, quadpoints_theta=self.surf.quadpoints_theta)  # parent setter
        super().recompute_data() #passes to C++ side

    @property
    def vmec(self):
        return self._vmec
    
    @vmec.setter
    def vmec(self):
        raise ValueError('Cannot change the VMEC, create a new field instead')
    
    @staticmethod
    def B1d_from_vmec(vmec, quadpoints_theta=None, quadpoints_phi=None):
        """
        Generate the 1D representation of the magnetic field the way VirtualCasing expects. 
        """
        Bx, By, Bz = B_cartesian(vmec, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        B1d = np.concatenate([Bx.flatten(order='C'),
                              By.flatten(order='C'),
                              Bz.flatten(order='C')])
        return B1d

