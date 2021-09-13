import simsoptpp as sopp
from simsopt.mhd import Boozer
from scipy.interpolate import interp1d
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BoozerMagneticField(sopp.BoozerMagneticField):
    def __init__(self,psi0):
        self.psi0 = psi0
        sopp.BoozerMagneticField.__init__(self,psi0)

    def clear_cached_properties(self):
        """Clear the cache."""
        sopp.BoozerMagneticField.invalidate_cache(self)

    def recompute_bell(self, parent=None):
        if np.any(self.dofs_free_status):
            self.clear_cached_properties()

class BoozerAnalytic(BoozerMagneticField):
    """
    First order direct QS axis expansion
    """
    def __init__(self,etabar,B0,Bbar,N,G0,psi0,iota0):
        self.etabar = etabar
        self.B0 = B0
        self.Bbar = Bbar
        self.N = N
        self.G0 = G0
        self.iota0 = iota0
        self.psi0 = psi0
        BoozerMagneticField.__init__(self,psi0)

    def _iota_impl(self, iota):
        iota[:] = self.iota0

    def _G_impl(self, G):
        G[:] = self.G0

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(2*psi/self.Bbar)
        modB[:,0] = self.B0*(1 + self.etabar*r*np.cos(thetas-self.N*zetas))

    def _dmodBds_impl(self,dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(2*psi/self.Bbar)
        drdpsi = 0.5*r/psi
        drds = drdpsi*self.psi0
        dmodBds[:] = self.B0*self.etabar*drds*np.cos(thetas-self.N*zetas)

    def _dmodBdtheta_impl(self,dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(2*psi/self.Bbar)
        dmodBdtheta[:] = -self.B0*self.etabar*r*np.sin(thetas-self.N*zetas)

    def _dmodBdzeta_impl(self,dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(2*psi/self.Bbar)
        dmodBdzeta[:] = self.N*self.B0*self.etabar*r*np.sin(thetas-self.N*zetas)

class BoozerRadialInterpolant(BoozerMagneticField):
    "Perform 1d radial interpolation and IFT"

    def __init__(self, vmec, order, N=None, enforce_vacuum=True):
        self.vmec = vmec
        self.vmec.run()
        self.booz = Boozer(vmec)
        self.booz.register(self.vmec.s_half_grid)
        self.booz.run()
        self.order = order
        self.enforce_qs = False
        self.enforce_vacuum = True
        if (N is not None):
            self.N = N
            self.enforce_qs = True
        BoozerMagneticField.__init__(self,vmec.wout.phi[-1]/(2*np.pi))
        self.init_splines()

    def init_splines(self):
        # Define quantities on extended half grid
        iota = np.zeros((self.vmec.wout.ns+1))
        G = np.zeros((self.vmec.wout.ns+1))
        bmnc = np.zeros((len(self.booz.bx.xm_b),self.vmec.wout.ns+1))
        # diotads = np.zeros((self.vmec.wout.ns))
        # dGds = np.zeros((self.vmec.wout.ns))
        dbmncds = np.zeros((len(self.booz.bx.xm_b),self.vmec.wout.ns))
        # d2iotads2 = np.zeros((self.vmec.wout.ns+1))
        # d2Gds2 = np.zeros((self.vmec.wout.ns+1))
        # d2bmncds2 = np.zeros((len(self.booz.bx.xm_b),self.vmec.wout.ns+1))

        s_half_ext = np.zeros((self.vmec.wout.ns+1))

        psip = self.vmec.wout.chi/(2*np.pi)
        iota[1:-1] = self.vmec.wout.iotas[1::]
        G[1:-1] = self.vmec.wout.bvco[1::]
        bmnc[:,1:-1] = self.booz.bx.bmnc_b
        # Extrapolate to get points at s = 0 and s = 1
        iota[0] = 1.5*iota[1] - 0.5*iota[2]
        G[0] = 1.5*G[1] - 0.5*G[2]
        bmnc[:,0] = 1.5*bmnc[:,1] - 0.5*bmnc[:,2]
        iota[-1] = 1.5*iota[-2] - 0.5*iota[-3]
        G[-1] = 1.5*G[-2] - 0.5*G[-3]
        bmnc[:,-1] = 1.5*bmnc[:,-2] - 0.5*bmnc[:,-3]
        # Compute first derivatives - on full grid points in [1,ns-1]
        dGds = (G[2:-1] - G[1:-2])/self.vmec.ds
        diotads = (iota[2:-1]-iota[1:-2])/self.vmec.ds
        dbmncds = (bmnc[:,2:-1] - bmnc[:,1:-2])/self.vmec.ds
        # dGds[1:-1] = (G[2:-1] - G[1:-2])/self.vmec.ds
        # diotads[1:-1] = (iota[2:-1]-iota[1:-2])/self.vmec.ds
        # dbmncds[:,1:-1] = (bmnc[:,2:-1] - bmnc[:,1:-2])/self.vmec.ds
        # Compute second derivatives - on half grid points in [2,ns-2]
        d2Gds2  = (G[3:-1]      - 2*G[2:-2]     + G[1:-3])/self.vmec.ds**2
        d2iotads2 = (iota[3:-1] - 2*iota[2:-2]  + iota[1:-3])/self.vmec.ds**2
        d2bmncds2 = (bmnc[:,3:-1]-2*bmnc[:,2:-2]+bmnc[:,1:-3])/self.vmec.ds**2
        # d2Gds2[2:-2]  = (G[3:-1]      - 2*G[2:-2]     + G[1:-3])/self.vmec.ds**2
        # d2iotads2[2:-2] = (iota[3:-1] - 2*iota[2:-2]  + iota[1:-3])/self.vmec.ds**2
        # d2bmncds2[:,2:-2] = (bmnc[:,3:-1]-2*bmnc[:,2:-2]+bmnc[:,1:-3])/self.vmec.ds**2
        # # Compute first derivatives at s = 0 and s = 1 using 3 point stencil
        # dGds[0] = (-8*G[0] + 9*G[1] - G[2])/(3*self.vmec.ds)
        # diotads[0] = (-8*iota[0] + 9*iota[1] - iota[2])/(3*self.vmec.ds)
        # dbmncds[:,0] = (-8*bmnc[:,0] + 9*bmnc[:,1] - bmnc[:,2])/(3*self.vmec.ds)
        # dGds[-1] = (8*G[-1] - 9*G[-2] + G[-3])/(3*self.vmec.ds)
        # diotads[-1] = (8*iota[-1] - 9*iota[-2] + iota[-3])/(3*self.vmec.ds)
        # dbmncds[:,-1] = (8*bmnc[:,-1] - 9*bmnc[:,-2] + bmnc[:,-3])/(3*self.vmec.ds)
        # # Compute second derivative at first and last half grid points using 3 point stencil
        # d2Gds2[1] = (G[1] - 2*G[2] + G[3])/self.vmec.ds**2
        # d2iotads2[1] = (iota[1]- 2*iota[2] + iota[3])/self.vmec.ds**2
        # d2bmncds2[:,1] = (bmnc[:,1] - 2*bmnc[:,2] + bmnc[:,3])/self.vmec.ds**2
        # d2Gds2[-2] = (G[-4] - 2*G[-3] + G[-2])/self.vmec.ds**2
        # d2iotads2[-2] = (iota[-4] - 2*iota[-3] + iota[-2])/self.vmec.ds**2
        # d2bmncds2[:,-2] = (bmnc[:,-4] - 2*bmnc[:,-3] + bmnc[:,-2])/self.vmec.ds**2
        # # Compute second derivative at s=0 and s=1 using 3 point stencil
        # d2Gds2[0] = (2*G[0] - 3*G[1] + G[2])/(3*self.vmec.ds**2/4)
        # d2iotads2[0] = (2*iota[0] - 3*iota[1] + iota[2])/(3*self.vmec.ds**2/4)
        # d2bmncds2[:,0] = (2*bmnc[:,0] - 3*bmnc[:,1] + bmnc[:,2])/(3*self.vmec.ds**2/4)
        # d2Gds2[-1] = (2*G[0] - 3*G[1] + G[2])/(3*self.vmec.ds**2/4)
        # d2iotads2[-1] = (2*iota[0] - 3*iota[1] + iota[2])/(3*self.vmec.ds**2/4)
        # d2bmncds2[:,-1] = (2*bmnc[:,0] - 3*bmnc[:,-1] + bmnc[:,-2])/(3*self.vmec.ds**2/4)

        s_half_ext[1:-1] = self.vmec.s_half_grid
        s_half_ext[-1] = 1


        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(s_half_ext,G)
        #
        # plt.figure()
        # plt.plot(s_half_ext,iota)
        #
        # plt.figure()
        # plt.plot(self.vmec.s_full_grid,dGds)
        #
        # plt.figure()
        # plt.plot(self.vmec.s_full_grid,diotads)
        #
        # plt.figure()
        # plt.plot(s_half_ext,d2Gds2)
        #
        # plt.figure()
        # plt.plot(s_half_ext,d2iotads2)
        #
        # plt.show()

        self.psip_spline = interp1d(self.vmec.s_full_grid, psip, kind=self.order)
        if not self.enforce_vacuum:
            self.G_spline = interp1d(s_half_ext, G, kind=self.order)
        else:
            self.G0 = np.mean(G)
        self.iota_spline = interp1d(s_half_ext, iota, kind=self.order)
        self.dGds_spline = interp1d(self.vmec.s_full_grid[1:-1], dGds, kind=self.order, fill_value='extrapolate')
        self.diotads_spline = interp1d(self.vmec.s_full_grid[1:-1], diotads, kind=self.order, fill_value='extrapolate')
        self.d2Gds2_spline = interp1d(self.vmec.s_half_grid[1:-1], d2Gds2, kind=self.order, fill_value='extrapolate')
        self.d2iotads2_spline = interp1d(self.vmec.s_half_grid[1:-1], d2iotads2, kind=self.order, fill_value='extrapolate')

        self.bmnc_splines = []
        self.dbmncds_splines = []
        self.d2bmncds2_splines = []
        for im in range(len(self.booz.bx.xm_b)):
            if (self.enforce_qs and (self.booz.bx.xn_b[im] != self.N * self.booz.bx.xm_b[im])):
                self.bmnc_splines.append(interp1d(s_half_ext, 0*bmnc[im,:], kind=self.order))
                self.dbmncds_splines.append(interp1d(self.vmec.s_full_grid[1:-1], 0*dbmncds[im,:], kind=self.order, fill_value='extrapolate'))
                self.d2bmncds2_splines.append(interp1d(self.vmec.s_half_grid[1:-1], 0*d2bmncds2[im,:], kind=self.order, fill_value='extrapolate'))
            else:
                self.bmnc_splines.append(interp1d(s_half_ext, bmnc[im,:], kind=self.order))
                self.dbmncds_splines.append(interp1d(self.vmec.s_full_grid[1:-1], dbmncds[im,:], kind=self.order, fill_value='extrapolate'))
                self.d2bmncds2_splines.append(interp1d(self.vmec.s_half_grid[1:-1], d2bmncds2[im,:], kind=self.order, fill_value='extrapolate'))

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        psip[:] = self.psip_spline(s)[:,None]

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        if not self.enforce_vacuum:
            G[:] = self.G_spline(s)[:,None]
        else:
            G[:] = self.G0

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        s = points[:, 0]
        iota[:] = self.iota_spline(s)[:,None]

    def _dGds_impl(self, dGds):
        points = self.get_points_ref()
        s = points[:, 0]
        dGds[:] = self.dGds_spline(s)[:,None]

    def _diotads_impl(self, diotads):
        points = self.get_points_ref()
        s = points[:, 0]
        diotads[:] = self.diotads_spline(s)[:,None]

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        modB[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            modB[:,0] += bmnc*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        dmodBdtheta[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            dmodBdtheta[:,0] += -self.booz.bx.xm_b[im]*bmnc*np.sin(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        dmodBdzeta[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            dmodBdzeta[:,0] += self.booz.bx.xn_b[im]*bmnc*np.sin(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _d2modBdzeta2_impl(self, d2modBdzeta2):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        d2modBdzeta2[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            d2modBdzeta2[:,0] += -self.booz.bx.xn_b[im]*self.booz.bx.xn_b[im]*bmnc*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _d2modBdtheta2_impl(self, d2modBdtheta2):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        d2modBdtheta2[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            d2modBdtheta2[:,0] += -self.booz.bx.xm_b[im]*self.booz.bx.xm_b[im]*bmnc*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _d2modBdthetadzeta_impl(self, d2modBdthetadzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        d2modBdthetadzeta[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            bmnc = self.bmnc_splines[im](s)
            d2modBdthetadzeta[:,0] += self.booz.bx.xm_b[im]*self.booz.bx.xn_b[im]*bmnc*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        dmodBds[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            dbmncds = self.dbmncds_splines[im](s)
            dmodBds[:,0] += dbmncds*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _d2modBdsdtheta_impl(self, d2modBdsdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        d2modBdsdtheta[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            dbmncds = self.dbmncds_splines[im](s)
            d2modBdsdtheta[:,0] += -self.booz.bx.xm_b[im]*dbmncds*np.sin(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _d2modBdsdzeta_impl(self, dmodBdsdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        dmodBdsdzeta[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            dbmncds = self.dbmncds_splines[im](s)
            dmodBdsdzeta[:,0] += self.booz.bx.xn_b[im]*dbmncds*np.sin(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

    def _d2modBds2_impl(self, d2modBds2):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        d2modBds2[:,0] = 0.
        for im in range(len(self.booz.bx.xm_b)):
            d2bmncds2 = self.d2bmncds2_splines[im](s)
            d2modBds2[:,0] += d2bmncds2*np.cos(self.booz.bx.xm_b[im]*thetas - self.booz.bx.xn_b[im]*zetas)

class InterpolatedBoozerField(sopp.InterpolatedBoozerField, BoozerMagneticField):
    r"""
    This field takes an existing field and interpolates it on a regular grid in :math:`s,\theta,\zeta`.
    This resulting interpolant can then be evaluated very quickly.
    """

    def __init__(self, field, degree, srange, thetarange, zetarange, extrapolate=True, nfp=1, stellsym=True):
        r"""
        Args:
            field: the underlying :mod:`simsopt.field.magneticfield.MagneticField` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            rrange: a 3-tuple of the form ``(rmin, rmax, nr)``. This mean that the interval :math:`[rmin, rmax]` is
                    split into ``nr`` many subintervals.
            phirange: a 3-tuple of the form ``(phimin, phimax, nphi)``.
            zrange: a 3-tuple of the form ``(zmin, zmax, nz)``.
            extrapolate: whether to extrapolate the field when evaluate outside
                         the integration domain or to throw an error.
            nfp: Whether to exploit rotational symmetry. In this case any angle
                 is always mapped into the interval :math:`[0, 2\pi/\mathrm{nfp})`,
                 hence it makes sense to use ``phimin=0`` and
                 ``phimax=2*np.pi/nfp``.
            stellsym: Whether to exploit stellarator symmetry. In this case
                      ``zeta`` is always mapped to be positive, hence it makes sense to use
                      ``zmin=0``.
        """
        BoozerMagneticField.__init__(self, field.psi0)
        if stellsym and (np.any(np.asarray(thetarange) < 0) or np.any(np.asarray(thetarange) > np.pi)):
            logger.warning(fr"Sure about thetarange=[{thetarange[0]},{thetarange[1]}]? When exploiting stellarator symmetry, the interpolant is only evaluated for thetaeta in [0,pi].")
        if nfp > 1 and (np.any(np.asarray(zetarange) < 0) or np.any(np.asarray(zetarange) > 2*np.pi/nfp)):
            logger.warning(fr"Sure about zetarange=[{zetarange[0]},{zetarange[1]}]? When exploiting rotational symmetry, the interpolant is only evaluated for zeta in [0,2\pi/nfp].")

        sopp.InterpolatedBoozerField.__init__(self, field, degree, srange, thetarange, zetarange, extrapolate, nfp, stellsym)
