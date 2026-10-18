from typing import TYPE_CHECKING
import logging

import numpy as np

from .magneticfield import MagneticField

if TYPE_CHECKING:
    from ..geo.surfacerzfourier import SurfaceRZFourier
    from ..mhd.vmec import Vmec

logger = logging.getLogger(__name__)

__all__ = ['VirtualCasingField', 'VmecVirtualCasingField']


def _virtual_casing():
    """The ``virtual_casing`` module, which must provide off-surface evaluation."""
    try:
        import virtual_casing
    except ImportError as e:
        raise ImportError("VirtualCasingField requires the virtual_casing package: "
                          "pip install git+https://github.com/hiddenSymmetries/virtual-casing") from e
    if not hasattr(virtual_casing.VirtualCasing, "compute_internal_B_offsurf"):
        raise ImportError("This version of virtual_casing has no off-surface evaluation; install the current one: "
                          "pip install git+https://github.com/hiddenSymmetries/virtual-casing")
    return virtual_casing


class VirtualCasingField(MagneticField):
    r"""
    Magnetic field of the currents on one side of a closed toroidal surface, computed from the
    total field on the surface by the virtual casing principle, with the high-order quadrature of
    the ``virtual_casing`` package (Malhotra et al., Plasma Phys. Control. Fusion 62, 024004 (2020)).

    With ``currents_inside=True``, :meth:`B` is the field of the currents inside the surface (e.g.
    the plasma), valid outside it. With ``currents_inside=False`` it is the field of the currents
    outside the surface (e.g. the coils), valid inside it. Points on the wrong side give meaningless
    values. On the surface itself use :attr:`onsurf_field_due_int` and :attr:`onsurf_field_due_ext`.
    Only :meth:`B` is available.

    Grids are per half field period for a stellarator-symmetric surface and per field period
    otherwise. Within about a grid cell of the surface the accuracy is set by how well the source
    grid resolves :math:`\hat{\mathbf{n}}\times\mathbf{B}`, which needs a considerably finer grid
    than :math:`\mathbf{B}` itself.

    Args:
        surf: a :obj:`~simsopt.geo.SurfaceRZFourier`, copied onto the source grid.
        total_B_in_surf: total field at the source grid points (``surf.gamma()`` of the copy), of shape
          ``(src_nphi, src_ntheta, 3)``, or flattened with all x components first as returned by
          :meth:`VmecVirtualCasingField.B1d_from_vmec`.
        src_nphi: toroidal points of the source grid; default the quadrature points of ``surf``.
        src_ntheta: poloidal points of the source grid; default the quadrature points of ``surf``.
        trgt_nphi: toroidal points of the grid of the on-surface fields; default ``src_nphi``.
        trgt_ntheta: poloidal points of the grid of the on-surface fields; default ``src_ntheta``.
        digits: approximate number of accurate digits.
        currents_inside: whether the currents of interest are inside the surface.
    """

    def __init__(self, surf: 'SurfaceRZFourier', total_B_in_surf, src_nphi: int = None, src_ntheta: int = None,
                 trgt_nphi: int = None, trgt_ntheta: int = None, digits: int = 7, currents_inside: bool = True):
        self._vcasing = _virtual_casing().VirtualCasing()
        self.nfp = surf.nfp
        self.stellsym = surf.stellsym
        self.src_nphi = len(surf.quadpoints_phi) if src_nphi is None else int(src_nphi)
        self.src_ntheta = len(surf.quadpoints_theta) if src_ntheta is None else int(src_ntheta)
        self.trgt_nphi = self.src_nphi if trgt_nphi is None else int(trgt_nphi)
        self.trgt_ntheta = self.src_ntheta if trgt_ntheta is None else int(trgt_ntheta)
        self.digits = int(digits)
        self.currents_inside = bool(currents_inside)
        self._range = 'half period' if self.stellsym else 'field period'
        MagneticField.__init__(self)
        self.surf = surf
        self.total_B_in_surf = total_B_in_surf

    @property
    def surf(self):
        """The surface on the source grid."""
        return self._surf

    @surf.setter
    def surf(self, surf):
        self._surf = surf.copy(nphi=self.src_nphi, ntheta=self.src_ntheta, range=self._range)
        self._gamma1d = self._surf.gamma().transpose(2, 0, 1).ravel()
        self._need_setup = True
        self._onsurf = {}
        self.clear_cached_properties()

    @property
    def trgt_surf(self):
        """The surface on the grid of the on-surface fields."""
        return self._surf.copy(nphi=self.trgt_nphi, ntheta=self.trgt_ntheta, range=self._range)

    @property
    def total_B_in_surf(self):
        """Total field at the source grid points, shape ``(src_nphi, src_ntheta, 3)``."""
        return self._total_B

    @total_B_in_surf.setter
    def total_B_in_surf(self, B):
        if B is not None:
            B = np.asarray(B, dtype=float)
            if B.ndim == 1:
                B = B.reshape(3, self.src_nphi, self.src_ntheta).transpose(1, 2, 0)
            if B.shape != (self.src_nphi, self.src_ntheta, 3):
                raise ValueError(f"total_B_in_surf has shape {B.shape}, expected "
                                 f"({self.src_nphi}, {self.src_ntheta}, 3)")
            B = np.ascontiguousarray(B)
        self._total_B = B
        self._onsurf = {}
        self.clear_cached_properties()

    def _setup(self):
        if self._total_B is None:
            raise RuntimeError("total_B_in_surf has not been set")
        if self._need_setup:
            self._vcasing.setup(self.digits, self.nfp, self.stellsym, self.src_nphi, self.src_ntheta,
                                self._gamma1d, self.src_nphi, self.src_ntheta, self.trgt_nphi, self.trgt_ntheta)
            self._need_setup = False
        return self._total_B.transpose(2, 0, 1).ravel()

    def _onsurf_field(self, side):
        if side not in self._onsurf:
            B1d = self._setup()
            f = self._vcasing.compute_external_B if side == "ext" else self._vcasing.compute_internal_B
            self._onsurf[side] = np.asarray(f(B1d)).reshape(3, self.trgt_nphi, self.trgt_ntheta).transpose(1, 2, 0)
        return self._onsurf[side]

    @property
    def onsurf_field_due_ext(self):
        """Field of the currents outside the surface, on :attr:`trgt_surf`, shape ``(trgt_nphi, trgt_ntheta, 3)``."""
        return self._onsurf_field("ext")

    @property
    def onsurf_field_due_int(self):
        """Field of the currents inside the surface, on :attr:`trgt_surf`, shape ``(trgt_nphi, trgt_ntheta, 3)``."""
        return self._onsurf_field("int")

    @property
    def Bnormal_due_ext(self):
        """Normal component of :attr:`onsurf_field_due_ext`, shape ``(trgt_nphi, trgt_ntheta)``."""
        return np.sum(self.onsurf_field_due_ext * self.trgt_surf.unitnormal(), axis=-1)

    @property
    def Bnormal_due_int(self):
        """Normal component of :attr:`onsurf_field_due_int`, shape ``(trgt_nphi, trgt_ntheta)``."""
        return np.sum(self.onsurf_field_due_int * self.trgt_surf.unitnormal(), axis=-1)

    @staticmethod
    def _offsurf(f, B1d, points):
        return np.asarray(f(B1d, points.T.ravel())).reshape(3, -1).T

    def _B_impl(self, B):
        B1d = self._setup()
        f = (self._vcasing.compute_internal_B_offsurf if self.currents_inside
             else self._vcasing.compute_external_B_offsurf)
        B[:] = self._offsurf(f, B1d, self.get_points_cart_ref())


class VmecVirtualCasingField(VirtualCasingField):
    """
    :class:`VirtualCasingField` of the plasma currents of a VMEC equilibrium, valid outside the
    VMEC boundary. The boundary and its field are taken from the VMEC output, and are recomputed,
    rerunning VMEC, after the degrees of freedom of the :obj:`~simsopt.mhd.vmec.Vmec` object change.
    Only stellarator-symmetric equilibria are supported.

    Args:
        vmec: a :obj:`~simsopt.mhd.vmec.Vmec` object, or the name of a VMEC input or ``wout`` file.
        src_nphi: toroidal points of the source grid, per half field period; if ``None``, from
          :meth:`default_source_grid`.
        src_ntheta: poloidal points of the source grid; if ``None``, from :meth:`default_source_grid`.
        trgt_nphi: as for :class:`VirtualCasingField`.
        trgt_ntheta: as for :class:`VirtualCasingField`.
        digits: as for :class:`VirtualCasingField`.
    """

    def __init__(self, vmec: 'Vmec', src_nphi: int = None, src_ntheta: int = None, trgt_nphi: int = None,
                 trgt_ntheta: int = None, digits: int = 7):
        from ..mhd.vmec import Vmec

        if not isinstance(vmec, Vmec):
            vmec = Vmec(vmec)
        if not vmec.boundary.stellsym:
            raise ValueError("VmecVirtualCasingField supports only stellarator-symmetric equilibria")
        if src_nphi is None or src_ntheta is None:
            nphi, ntheta = self.default_source_grid(vmec, digits)
            src_nphi = nphi if src_nphi is None else src_nphi
            src_ntheta = ntheta if src_ntheta is None else src_ntheta
        self._vmec = vmec
        self._need_vmec = True
        super().__init__(vmec.boundary, None, src_nphi, src_ntheta, trgt_nphi, trgt_ntheta, digits,
                         currents_inside=True)
        self.append_parent(vmec)

    @property
    def vmec(self):
        """The :obj:`~simsopt.mhd.vmec.Vmec` object."""
        return self._vmec

    @staticmethod
    def default_source_grid(vmec, digits: int = 7):
        r"""
        Default source grid ``(src_nphi, src_ntheta)``: the smallest grid whose Nyquist modes of the virtual
        casing density :math:`\hat{\mathbf{n}}\times\mathbf{B}` on the VMEC boundary are below
        :math:`10^{-(\mathrm{digits}+2)}` of its largest Fourier coefficient, with at least 8 toroidal and 16
        poloidal points. The spectrum is that of the cylindrical components over one field period, sampled finely
        enough to be free of aliasing. VMEC is run if needed.

        Args:
            vmec: a :obj:`~simsopt.mhd.vmec.Vmec` object.
            digits: as for :class:`VirtualCasingField`.

        Returns:
            ``(src_nphi, src_ntheta)``, with ``src_nphi`` per half field period.
        """
        from ..geo.surfacerzfourier import SurfaceRZFourier
        from ..mhd.vmec_diagnostics import B_cartesian

        vmec.run()
        nfp = vmec.wout.nfp
        tol = 10.0 ** -(digits + 2)
        nphi, ntheta = 64, 128
        while True:
            qphi = np.arange(nphi) / (nphi * nfp)
            qtheta = np.arange(ntheta) / ntheta
            surf = SurfaceRZFourier.from_wout(vmec.output_file, quadpoints_phi=qphi, quadpoints_theta=qtheta)
            B = np.concatenate([np.stack(B_cartesian(vmec, quadpoints_phi=qphi[j:j + 8], quadpoints_theta=qtheta),
                                         axis=-1) for j in range(0, nphi, 8)])
            F = np.cross(surf.unitnormal(), B)
            c, s = np.cos(2 * np.pi * qphi)[:, None], np.sin(2 * np.pi * qphi)[:, None]
            F = np.stack([c * F[..., 0] + s * F[..., 1], c * F[..., 1] - s * F[..., 0], F[..., 2]], axis=-1)
            amp = np.abs(np.fft.fft2(F, axes=(0, 1))).max(axis=2)
            above = amp > tol * amp.max()
            m_max = np.minimum(np.arange(ntheta), ntheta - np.arange(ntheta))[above.any(axis=0)].max()
            n_max = np.minimum(np.arange(nphi), nphi - np.arange(nphi))[above.any(axis=1)].max()
            if 3 * m_max >= ntheta and ntheta < 2048:
                ntheta *= 2
            elif 3 * n_max >= nphi and nphi < 1024:
                nphi *= 2
            else:
                break
        if 3 * m_max >= ntheta or 3 * n_max >= nphi:
            logger.warning(f"default_source_grid: the spectrum of n x B is not resolved by {nphi} x {ntheta} samples")
        return max(int(n_max) + 1, 8), max(2 * (int(m_max) + 1), 16)

    def recompute_bell(self, parent=None):
        self._need_vmec = True
        self._onsurf = {}
        self.clear_cached_properties()

    def _setup(self):
        if self._need_vmec:
            from ..geo.surfacerzfourier import SurfaceRZFourier

            self._vmec.run()
            if self._vmec.wout.lasym:
                raise RuntimeError("VmecVirtualCasingField supports only stellarator-symmetric equilibria")
            self.surf = SurfaceRZFourier.from_wout(self._vmec.output_file, nphi=self.src_nphi,
                                                   ntheta=self.src_ntheta, range=self._range)
            self.total_B_in_surf = self.B1d_from_vmec(self._vmec, self.surf.quadpoints_theta,
                                                      self.surf.quadpoints_phi)
            self._need_vmec = False
        return super()._setup()

    @staticmethod
    def B1d_from_vmec(vmec, quadpoints_theta=None, quadpoints_phi=None):
        """
        Total field on the VMEC boundary at the given quadrature points, flattened as
        ``virtual_casing`` expects: all x components first, toroidal index outer.
        """
        from ..mhd.vmec_diagnostics import B_cartesian

        Bx, By, Bz = B_cartesian(vmec, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        return np.concatenate([Bx.ravel(), By.ravel(), Bz.ravel()])

