import warnings
import logging

import numpy as np
from scipy.spatial import cKDTree

from .magneticfield import MagneticField
from .._core.json import GSONDecoder

import virtual_casing as vc_module
from ..mhd.vmec import Vmec
from ..geo.surface import best_nphi_over_ntheta
from ..mhd.vmec_diagnostics import B_cartesian
from .._core.optimizable import Optimizable



logger = logging.getLogger(__name__)


__all__ = ['VirtualCasingField',]


class VirtualCasingField(MagneticField):
    """
    Computes the magnetic field contribution from plasma currents using the 
    virtual casing principle. Can evaluate the field at arbitrary points in space.
    
    For points close to the plasma surface (within ``on_surface_tol``), precomputed
    on-surface values are used via nearest-neighbor lookup. For points far from the
    surface, ``compute_external_B_offsurf`` is used.
    
    NOTE: Evaluating exactly on the plasma surface using ``compute_external_B_offsurf``
    can be slow or numerically unstable due to singular integrals. The ``on_surface_tol``
    parameter controls when to use the faster precomputed on-surface values. This may need
    to be adjusted for different geometries or grid resolutions.
    
    Attributes:
        on_surface_tol (float): Distance threshold (in meters) for using precomputed
            on-surface values instead of off-surface computation.
        gamma (ndarray): Source grid positions, shape (n_trgt, 3).
        B_external_onsurf (ndarray): Precomputed B field at source grid, shape (n_trgt, 3).
    """

    def __init__(self, digits, nfp, stellsym, src_nphi, src_ntheta, gamma1d, src_nphi, src_ntheta, trgt_nphi, trgt_ntheta)
    
    @classmethod
    def from_vmec(cls, vmec, src_nphi=80, src_ntheta=None, trgt_nphi=32, trgt_ntheta=32, 
                  use_stellsym=True, digits=3, on_surface_tol=0.01, filename="auto"):
        """
        Given a :obj:`~simsopt.mhd.vmec.Vmec` object, define MagneticField 
        for the contribution to the total magnetic field due to currents 
        in the plasma.

        This function requires the python ``virtual_casing`` package to be
        installed. Currently only supports calculation of B, no dB_dX or d2B_dXdX,
        and no vector potential A.

        The argument ``src_nphi`` refers to the number of points around a half
        field period if stellarator symmetry is exploited, or a full field
        period if not.

        To set the grid resolutions ``src_nphi`` and ``src_ntheta``, it can be
        convenient to use the function
        :func:`simsopt.geo.surface.best_nphi_over_ntheta`. This is
        done automatically if you omit the ``src_ntheta`` argument.

        For now, this routine only works for stellarator symmetry.

        Args:
            vmec: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`, or the name of a
              Vmec ``input.*`` or ``wout*`` file.
            src_nphi (int, default=80): Number of grid points toroidally for the input of the calculation.
            src_ntheta (int, default=None): Number of grid points poloidally for the input of the calculation. If ``None``,
              the number of grid points will be calculated automatically using
              :func:`simsopt.geo.surface.best_nphi_over_ntheta()` to minimize
              the grid anisotropy, given the specified ``nphi``.
            trgt_nphi (int, default=32): Number of grid points toroidally for the output of the calculation.
              If unspecified, ``src_nphi`` will be used.
            trgt_ntheta (int, default=32): Number of grid points poloidally for the output of the calculation.
              If unspecified, ``src_ntheta`` will be used.
            use_stellsym (bool, default=True): whether to exploit stellarator symmetry in the calculation.
            digits (int, default=3): Approximate number of digits of precision for the calculation.
            on_surface_tol (float, default=0.01): Distance threshold (in meters) for using 
              precomputed on-surface values instead of off-surface computation.
            filename (str, default="auto"): If not ``None``, the results of the virtual casing calculation
              will be saved in this file. For the default value of ``"auto"``, the
              filename will automatically be set to ``"vcasing_<extension>.nc"``
              where ``<extension>`` is the string associated with Vmec input and output
              files, analogous to the Vmec output file ``"wout_<extension>.nc"``.
        """
        
        if not isinstance(vmec, Vmec):
            vmec = Vmec(vmec)

        vmec.run()
        nfp = vmec.wout.nfp
        stellsym = (not bool(vmec.wout.lasym)) and use_stellsym
        if vmec.wout.lasym:
            raise RuntimeError('virtual casing presently only works for stellarator symmetry')

        if src_ntheta is None:
            src_ntheta = int((1+int(stellsym)) * nfp * src_nphi / best_nphi_over_ntheta(vmec.boundary))
            logger.info(f'new src_ntheta: {src_ntheta}')

        # The requested nphi and ntheta may not match the quadrature
        # points in vmec.boundary, and the range may not be "full torus",
        # so generate a SurfaceRZFourier with the desired resolution:
        if stellsym:
            ran = "half period"
        else:
            ran = "field period"
        surf = vmec.boundary.copy(
            range=ran, 
            nphi=src_nphi, 
            ntheta=src_ntheta,  
            mpol=vmec.wout.mpol, 
            ntor=vmec.wout.ntor, 
            nfp=nfp,
            stellsym=stellsym
        )
        Bxyz = B_cartesian(vmec, nphi=src_nphi, ntheta=src_ntheta, range=ran)
        gamma = surf.gamma()

        if trgt_nphi is None:
            trgt_nphi = src_nphi
        if trgt_ntheta is None:
            trgt_ntheta = src_ntheta

        # virtual_casing wants all input arrays to be 1D. The order is
        # {x11, x12, ..., x1Np, x21, x22, ... , xNtNp, y11, ... , z11, ...}
        # where Nt is toroidal (not theta!) and Np is poloidal (not phi!)
        src_res = src_nphi * src_ntheta
        gamma1d = np.zeros(src_res * 3)
        B1d = np.zeros(src_res * 3)
        for jxyz in range(3):
            gamma1d[jxyz * src_res: (jxyz + 1) * src_res] = gamma[:, :, jxyz].flatten(order='C')
            B1d[jxyz * src_res: (jxyz + 1) * src_res] = Bxyz[jxyz].flatten(order='C')

        vcasing = vc_module.VirtualCasing()
        vcasing.setup(
            digits, nfp, stellsym,
            src_nphi, src_ntheta, gamma1d,
            src_nphi, src_ntheta,
            trgt_nphi, trgt_ntheta)
        
        # Create target surface for on-surface B field evaluation
        # This surface has the resolution where B_external_onsurf is computed
        trgt_surf = vmec.boundary.copy(
            range=ran,
            nphi=trgt_nphi,
            ntheta=trgt_ntheta,
            mpol=vmec.wout.mpol,
            ntor=vmec.wout.ntor,
            nfp=nfp,
            stellsym=stellsym
        )
        trgt_gamma = trgt_surf.gamma().reshape(-1, 3)  # Shape: (trgt_nphi * trgt_ntheta, 3)
        
        # Store source grid positions (for reference)
        gamma = gamma.reshape(-1, 3)  # Shape: (src_nphi * src_ntheta, 3)
        
        # Precompute on-surface B field values using compute_external_B
        Bexternal1d_onsurf = np.array(vcasing.compute_external_B(B1d))
        n_trgt = trgt_nphi * trgt_ntheta
        B_external_onsurf = np.zeros((n_trgt, 3))
        for jxyz in range(3):
            B_external_onsurf[:, jxyz] = Bexternal1d_onsurf[jxyz * n_trgt: (jxyz + 1) * n_trgt]
        
        # Initialize VirtualCasingField
        vc = cls()
        vc.vcasing = vcasing 
        vc.surf = surf  # Source surface
        vc.trgt_surf = trgt_surf  # Target surface (used for nearest-neighbor lookup)
        vc.B1d = B1d
        vc.gamma = gamma  # Source grid (for reference)
        vc.trgt_gamma = trgt_gamma  # Target grid (used for nearest-neighbor lookup)
        vc.trgt_nphi = trgt_nphi
        vc.trgt_ntheta = trgt_ntheta
        vc.B_external_onsurf = B_external_onsurf
        vc.on_surface_tol = on_surface_tol
        MagneticField.__init__(vc)
        return vc
    
    @staticmethod
    def _compute_max_grid_spacing(gamma):
        """
        Compute the maximum distance between adjacent grid points.
        
        This is used to set a safe minimum for on_surface_tol.
        
        Args:
            gamma: Surface grid positions, shape (src_nphi * src_ntheta, 3).
            
        Returns:
            float: Maximum distance between any point and its nearest neighbor.
        """
        tree = cKDTree(gamma)
        # Query for 2 nearest neighbors (first is self with distance 0)
        distances, _ = tree.query(gamma, k=2)
        # Return max of the nearest-neighbor distances (excluding self)
        return np.max(distances[:, 1])
    
    def get_close_mask(self, points=None):
        """
        Determine which evaluation points are close to the target grid points.
        
        Points within ``on_surface_tol`` of a target grid point are considered "close"
        and will use precomputed on-surface values via nearest-neighbor lookup.
        
        Args:
            points: Evaluation points, shape (n_points, 3). If None, uses current points.
                
        Returns:
            tuple: (close_mask, distances, nearest_idx) where:
                - close_mask: boolean array, True for points within on_surface_tol of grid
                - distances: Euclidean distance to nearest grid point for each point
                - nearest_idx: index of nearest grid point for each point
        """
        if points is None:
            points = self.get_points_cart_ref()
        
        # Compute distance to each target grid point
        distances_to_grid = np.linalg.norm(
            points[:, np.newaxis, :] - self.trgt_gamma[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(distances_to_grid, axis=1)
        distances = distances_to_grid[np.arange(len(points)), nearest_idx]
        
        # Use precomputed values if within on_surface_tol of a grid point
        close_mask = distances < self.on_surface_tol
        
        return close_mask, distances, nearest_idx
    
    def _eval_close(self, B, close_mask, nearest_idx):
        """
        Evaluate B field for points close to the surface using precomputed values.
        
        Currently uses nearest-neighbor lookup. This method is isolated to allow
        future implementation of interpolation schemes.
        
        Args:
            B: Output array to fill, shape (n_points, 3).
            close_mask: Boolean mask for close points.
            nearest_idx: Index of nearest grid point for each point.
        """
        # Nearest-neighbor lookup from precomputed on-surface values
        # Could be extended to use interpolation in the future
        B[close_mask] = self.B_external_onsurf[nearest_idx[close_mask]]

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        n_points = len(points)
        
        # Determine which points are close to the surface
        close_mask, distances, nearest_idx = self.get_close_mask(points)
        n_close = np.sum(close_mask)
        n_far = n_points - n_close
        
        # Evaluate close points using precomputed on-surface values
        if n_close > 0:
            self._eval_close(B, close_mask, nearest_idx)

            # Warn about using nearest-neighbor approximation
            max_close_dist = np.max(distances[close_mask])
            if max_close_dist > 1e-10:  # Not exactly on grid
                warnings.warn(
                    f"{n_close} evaluation point(s) are within {self.on_surface_tol:.4f}m "
                    f"of the target grid (max distance: {max_close_dist:.4f}m). "
                    f"Using nearest-neighbor lookup from precomputed on-surface values.",
                    stacklevel=3
                )
        
        # Evaluate far points using compute_external_B_offsurf
        if n_far > 0:
            far_mask = ~close_mask
            far_points = points[far_mask]
            points1d = far_points.flatten(order='F')
            Bext1d = np.array(self.vcasing.compute_external_B_offsurf(self.B1d, points1d))
            B[far_mask] = Bext1d.reshape((-1, 3), order='F')


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

