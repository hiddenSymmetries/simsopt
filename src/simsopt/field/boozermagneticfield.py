import simsoptpp as sopp
from simsopt.mhd import Boozer
from scipy.interpolate import interp1d
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BoozerMagneticField(sopp.BoozerMagneticField):
    """
    Generic class that represents a magnetic field in Boozer coordinates
    :math:`(s,\theta,\zeta)`. Here :math:`s = \psi/\psi_0` is the normalized
    toroidal flux. The magnetic field in the covariant form is,

    .. math::

        \textbf B(s,\theta,\zeta) = G(s) \nabla \zeta + I(s) \nabla \theta + K(s,\theta,\zeta) \nabla \psi,

    and the contravariant form is,
    .. math::

        textbf B(s,\theta,\zeta) = \frac{1}{\sqrt{g}} \left(\frac{\partial \mathbf r}{\partial \zeta} + \iota(s)\frac{\partial \mathbf r}{\partial \theta}\right),

    where,

    .. math::
        \sqrt{g}(s,\theta,\zeta) = \frac{G(s) + \iota(s)I(s)}{B^2}.

    Here :math:`\iota(s) = \psi_P'(\psi)` where :math:`2\pi\psi_P` is the
    poloidal flux and :math:`2\pi\psi` is the toroidal flux. Each subclass of
    BoozerMagneticField implements functions to compute
    :math:`B`, :math:`G`, :math:`I`, :math:`\iota`, :math:`\psi_P`, and their derivatives.
    The usage is similar to the MagneticField class.

    The usage of ``BoozerMagneticField`` is as follows:

    .. code-block::

        booz = BoozerAnalytic(etabar,B0,Bbar,N,G0,psi0,iota0) # An instance of BoozerMagneticField
        points = ... # points is a (n, 3) numpy array defining :math:`(s,\theta,\zeta)`
        booz.set_points(points)
        modB = bfield.modB() # returns the magnetic field strength at `points`

    ``BoozerMagneticField`` has a cache to avoid repeated calculations.
    To clear this cache manually, call the `clear_cached_properties()` function.
    The cache is automatically cleared when ``set_points`` is called or one of the dependencies
    changes.
    """

    def __init__(self, psi0):
        self.psi0 = psi0
        sopp.BoozerMagneticField.__init__(self, psi0)

    def clear_cached_properties(self):
        """Clear the cache."""
        sopp.BoozerMagneticField.invalidate_cache(self)

    def recompute_bell(self, parent=None):
        if np.any(self.dofs_free_status):
            self.clear_cached_properties()


class BoozerAnalytic(BoozerMagneticField):
    """
    Computes a BoozerMagneticField based on a first-order expansion in
    distance from the magnetic axis (Landreman & Sengupta, Journal of Plasma
    Physics 2018). Here the magnetic field strength is expressed as,

    .. math::
        B(s,\theta,\zeta) = B_0 \left(1 + \frac{\etabar \sqrt{2s\psi_0}}{\overline{B}}\cos(\theta - N \zeta)\right)

    and the covariant forms are,

    .. math::
        G(s) = G_0 + \frac{\sqrt{2s\psi_0}}{\overline{B}} G_1
        I(s) = I_0 + \frac{\sqrt{2s\psi_0}}{\overline{B}} I_1,

    and the rotational transform is,

    .. math::
        \iota(s) = \iota_0.

    While formally :math:`I_0 = I_1 = G_1 = 0`, these terms have been included
    in order to test the guiding center equations at finite beta.

    Args:
        etabar: magnitude of first order correction to magnetic field strength
        B0: magnetic field strength on the axis
        Bbar: normalizing magnetic field strength
        N: helicity of symmetry (integer)
        G0: lowest order toroidal covariant component
        psi0: (toroidal flux)/ (2*pi) on the boundary
        iota0: lowest order rotational transform
        I0: lowest order poloidal covariant component (defaults to 0)
        G1: first order correction to toroidal covariant component (defaults to 0)
        I1: first order correction to poloidal covariant component (defaults to 0)
    """

    def __init__(self, etabar, B0, Bbar, N, G0, psi0, iota0, I0=0., G1=0., I1=0.):
        self.etabar = etabar
        self.B0 = B0
        self.Bbar = Bbar
        self.N = N
        self.G0 = G0
        self.I0 = I0
        self.I1 = I1
        self.G1 = G1
        self.iota0 = iota0
        self.psi0 = psi0
        BoozerMagneticField.__init__(self, psi0)

    def set_etabar(self, etabar):
        self.invalidate_cache()
        self.etabar = etabar

    def set_B0(self, B0):
        self.invalidate_cache()
        self.B0 = B0

    def set_Bbar(self, Bbar):
        self.invalidate_cache()
        self.Bbar = Bbar

    def set_N(self, N):
        self.invalidate_cache()
        self.N = N

    def set_G0(self, G0):
        self.invalidate_cache()
        self.G0 = G0

    def set_I0(self, I0):
        self.invalidate_cache()
        self.I0 = I0

    def set_G1(self, G1):
        self.invalidate_cache()
        self.G1 = G1

    def set_I1(self, I1):
        self.invalidate_cache()
        self.I1 = I1

    def set_iota0(self, iota0):
        self.invalidate_cache()
        self.iota0 = iota0

    def set_psi0(self, psi0):
        self.invalidate_cache()
        self.psi0 = psi0

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        psip[:, 0] = self.psi0*s*self.iota0

    def _iota_impl(self, iota):
        iota[:, 0] = self.iota0

    def _diotads_impl(self, diotads):
        diotads[:, 0] = 0

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        G[:, 0] = self.G0 + s*self.G1

    def _dGds_impl(self, dGds):
        dGds[:, 0] = self.G1

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        I[:, 0] = self.I0 + s*self.I1

    def _dIds_impl(self, dIds):
        dIds[:, 0] = self.I1

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        modB[:, 0] = self.B0*(1 + self.etabar*r*np.cos(thetas-self.N*zetas))

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        drdpsi = 0.5*r/psi
        drds = drdpsi*self.psi0
        dmodBds[:, 0] = self.B0*self.etabar*drds*np.cos(thetas-self.N*zetas)

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dmodBdtheta[:, 0] = -self.B0*self.etabar*r*np.sin(thetas-self.N*zetas)

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dmodBdzeta[:, 0] = self.N*self.B0*self.etabar*r*np.sin(thetas-self.N*zetas)


class BoozerRadialInterpolant(BoozerMagneticField):
    """
    Given a Vmec instance, performs a Boozer coordinate transformation using
    BOOZXFORM. The magnetic field can be computed at any point in Boozer
    coordinates using radial interpolation and an inverse Fourier Transform
    in the two angles.

    Args:
        vmec: instance of Vmec
        order: order for radial interpolation
        N: helicity of symmetry. If specified, then the non-symmetric Fourier
            harmonics are filtered out. Otherwise, all harmonics are kept.
        enforce_vacuum: If True, a vacuum field is assumed, and ::math::`G` is
            set to its mean value and ::math::`I = 0`
    """

    def __init__(self, vmec, order, N=None, enforce_vacuum=False):
        self.vmec = vmec
        self.vmec.run()
        self.booz = Boozer(vmec)
        self.booz.register(self.vmec.s_half_grid)
        self.booz.run()
        self.order = order
        self.enforce_qs = False
        self.enforce_vacuum = enforce_vacuum
        if (N is not None):
            self.N = N
            self.enforce_qs = True
        BoozerMagneticField.__init__(self, vmec.wout.phi[-1]/(2*np.pi))
        self.init_splines()

    def init_splines(self):
        # Define quantities on extended half grid
        iota = np.zeros((self.vmec.wout.ns+1))
        G = np.zeros((self.vmec.wout.ns+1))
        I = np.zeros((self.vmec.wout.ns+1))
        bmnc = np.zeros((len(self.booz.bx.xm_b), self.vmec.wout.ns+1))
        dbmncds = np.zeros((len(self.booz.bx.xm_b), self.vmec.wout.ns))

        s_half_ext = np.zeros((self.vmec.wout.ns+1))

        psip = self.vmec.wout.chi/(2*np.pi)
        iota[1:-1] = self.vmec.wout.iotas[1::]
        G[1:-1] = self.vmec.wout.bvco[1::]
        I[1:-1] = self.vmec.wout.buco[1::]
        bmnc[:, 1:-1] = self.booz.bx.bmnc_b
        # Extrapolate to get points at s = 0 and s = 1
        iota[0] = 1.5*iota[1] - 0.5*iota[2]
        G[0] = 1.5*G[1] - 0.5*G[2]
        I[0] = 1.5*I[1] - 0.5*I[2]
        bmnc[:, 0] = 1.5*bmnc[:, 1] - 0.5*bmnc[:, 2]
        iota[-1] = 1.5*iota[-2] - 0.5*iota[-3]
        G[-1] = 1.5*G[-2] - 0.5*G[-3]
        I[-1] = 1.5*I[-2] - 0.5*I[-3]
        bmnc[:, -1] = 1.5*bmnc[:, -2] - 0.5*bmnc[:, -3]
        # Compute first derivatives - on full grid points in [1,ns-1]
        dGds = (G[2:-1] - G[1:-2])/self.vmec.ds
        dIds = (I[2:-1] - I[1:-2])/self.vmec.ds
        diotads = (iota[2:-1]-iota[1:-2])/self.vmec.ds
        dbmncds = (bmnc[:, 2:-1] - bmnc[:, 1:-2])/self.vmec.ds

        s_half_ext[1:-1] = self.vmec.s_half_grid
        s_half_ext[-1] = 1

        self.psip_spline = interp1d(self.vmec.s_full_grid, psip, kind=self.order)
        if not self.enforce_vacuum:
            self.G_spline = interp1d(s_half_ext, G, kind=self.order)
            self.I_spline = interp1d(s_half_ext, I, kind=self.order)
            self.dGds_spline = interp1d(self.vmec.s_full_grid[1:-1], dGds, kind=self.order, fill_value='extrapolate')
            self.dIds_spline = interp1d(self.vmec.s_full_grid[1:-1], dIds, kind=self.order, fill_value='extrapolate')
        else:
            self.G0 = np.mean(G)
        self.iota_spline = interp1d(s_half_ext, iota, kind=self.order)
        self.diotads_spline = interp1d(self.vmec.s_full_grid[1:-1], diotads, kind=self.order, fill_value='extrapolate')

        self.bmnc_splines = []
        self.dbmncds_splines = []
        for im in range(len(self.booz.bx.xm_b)):
            if (self.enforce_qs and (self.booz.bx.xn_b[im] != self.N * self.booz.bx.xm_b[im])):
                self.bmnc_splines.append(interp1d(s_half_ext, 0*bmnc[im, :], kind=self.order))
                self.dbmncds_splines.append(interp1d(self.vmec.s_full_grid[1:-1], 0*dbmncds[im, :], kind=self.order, fill_value='extrapolate'))
            else:
                self.bmnc_splines.append(interp1d(s_half_ext, bmnc[im, :], kind=self.order))
                self.dbmncds_splines.append(interp1d(self.vmec.s_full_grid[1:-1], dbmncds[im, :], kind=self.order, fill_value='extrapolate'))

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        psip[:] = self.psip_spline(s)[:, None]

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        if not self.enforce_vacuum:
            G[:] = self.G_spline(s)[:, None]
        else:
            G[:] = self.G0

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        if not self.enforce_vacuum:
            I[:] = self.I_spline(s)[:, None]
        else:
            I[:] = 0.

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        s = points[:, 0]
        iota[:] = self.iota_spline(s)[:, None]

    def _dGds_impl(self, dGds):
        points = self.get_points_ref()
        s = points[:, 0]
        if not self.enforce_vacuum:
            dGds[:] = self.dGds_spline(s)[:, None]
        else:
            dGds[:] = 0.

    def _dIds_impl(self, dIds):
        points = self.get_points_ref()
        s = points[:, 0]
        if not self.enforce_vacuum:
            dIds[:] = self.dIds_spline(s)[:, None]
        else:
            dIds[:] = 0.

    def _diotads_impl(self, diotads):
        points = self.get_points_ref()
        s = points[:, 0]
        diotads[:] = self.diotads_spline(s)[:, None]

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        modB[:, 0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            modB[:, 0] += bmnc*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        dmodBdtheta[:, 0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            dmodBdtheta[:, 0] += -self.booz.bx.xm_b[im]*bmnc*np.sin(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        dmodBdzeta[:, 0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            dmodBdzeta[:, 0] += self.booz.bx.xn_b[im]*bmnc*np.sin(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        dmodBds[:, 0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            dbmncds = self.dbmncds_splines[im](s)
            dmodBds[:, 0] += dbmncds*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)


class InterpolatedBoozerField(sopp.InterpolatedBoozerField, BoozerMagneticField):
    r"""
    This field takes an existing BoozerMagneticField and interpolates it on a
    regular grid in :math:`s,\theta,\zeta`. This resulting interpolant can then
    be evaluated very quickly. This is modeled after InterpolatedField.
    """

    def __init__(self, field, degree, srange, thetarange, zetarange, extrapolate=True, nfp=1, stellsym=True):
        r"""
        Args:
            field: the underlying :mod:`simsopt.field.boozermagneticfield.BoozerMagneticField` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            srange: a 3-tuple of the form ``(smin, smax, ns)``. This mean that
                the interval :math:`[smin, smax]` is split into ``ns`` many subintervals.
            thetarange: a 3-tuple of the form ``(thetamin, thetamax, ntheta)``.
            zetarange: a 3-tuple of the form ``(zetamin, zetamax, nzeta)``.
            extrapolate: whether to extrapolate the field when evaluate outside
                         the integration domain or to throw an error.
            nfp: Whether to exploit rotational symmetry. In this case any toroidal angle
                 is always mapped into the interval :math:`[0, 2\pi/\mathrm{nfp})`,
                 hence it makes sense to use ``zetamin=0`` and
                 ``zetamax=2*np.pi/nfp``.
            stellsym: Whether to exploit stellarator symmetry. In this case
                      ``theta`` is always mapped to be positive, hence it makes sense to use
                      ``thetamin=0``.
        """
        BoozerMagneticField.__init__(self, field.psi0)
        if stellsym and (np.any(np.asarray(thetarange) < 0) or np.any(np.asarray(thetarange) > np.pi)):
            logger.warning(fr"Sure about thetarange=[{thetarange[0]},{thetarange[1]}]? When exploiting stellarator symmetry, the interpolant is only evaluated for theta in [0,pi].")
        if nfp > 1 and (np.any(np.asarray(zetarange) < 0) or np.any(np.asarray(zetarange) > 2*np.pi/nfp)):
            logger.warning(fr"Sure about zetarange=[{zetarange[0]},{zetarange[1]}]? When exploiting rotational symmetry, the interpolant is only evaluated for zeta in [0,2\pi/nfp].")

        sopp.InterpolatedBoozerField.__init__(self, field, degree, srange, thetarange, zetarange, extrapolate, nfp, stellsym)
