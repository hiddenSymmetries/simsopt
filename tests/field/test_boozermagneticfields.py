from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField, BoozerAnalytic
import numpy as np
import unittest
from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = str((TEST_DIR / 'input.LandremanPaul2021_QA_lowres').resolve())
filename_mhd = str((TEST_DIR / 'input.n3are_R7.75B5.7').resolve())
filename_mhd_lowres = str((TEST_DIR / 'input.n3are_R7.75B5.7_lowres').resolve())

try:
    import vmec
except ImportError as e:
    vmec = None

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None

if (MPI is not None) and (vmec is not None):
    from simsopt.mhd.vmec import Vmec


class TestingAnalytic(unittest.TestCase):
    def test_boozeranalytic(self):
        etabar = 1.1
        B0 = 1.0
        Bbar = 1.0
        N = 0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 0.4
        ba = BoozerAnalytic(etabar, B0, Bbar, N, G0, psi0, iota0)

        ntheta = 101
        nzeta = 100
        thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        zetas = np.linspace(0, 2*np.pi, nzeta, endpoint=False)
        [zetas, thetas] = np.meshgrid(zetas, thetas)
        points = np.zeros((len(thetas.flatten()), 3))
        points[:, 0] = 0.5*np.ones_like(thetas.flatten())
        points[:, 1] = thetas.flatten()
        points[:, 2] = zetas.flatten()
        ba.set_points(points)

        # Check that get_points returns correct points
        points_get = ba.get_points()
        thetas_get = points_get[:, 1]
        assert np.allclose(thetas_get, thetas.flatten())

        # Check that angular derivatives integrate to zero
        assert np.allclose(np.sum(ba.dmodBdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
        assert np.allclose(np.sum(ba.dmodBdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)

        # Check that zeta derivatives are small since we are QA
        assert np.allclose(ba.dmodBdzeta(), 0, atol=1e-12)

        # Check that (theta + zeta) derivatives are small since we are QH
        ba.set_N(1)
        assert np.allclose(ba.dmodBdtheta()+ba.dmodBdzeta(), 0, atol=1e-12)

        # Check that G = G0
        assert np.allclose(ba.G(), G0, atol=1e-12)

        # Check that dGds = 0 since G1 = 0
        assert np.allclose(ba.dGds(), 0, atol=1e-12)

        # Check that I = 0 and I'=0 since I0 = I1 = 0
        assert np.allclose(ba.I(), 0, atol=1e-12)
        assert np.allclose(ba.dIds(), 0, atol=1e-12)

        # Check that iota = iota0
        assert np.allclose(ba.iota(), iota0, atol=1e-12)

        # Check that diotads = 0
        assert np.allclose(ba.diotads(), 0, atol=1e-12)

        # Check that if etabar = 0, dBdtheta = dBdzeta = dBds = 0
        ba.set_etabar(0.)
        assert np.allclose(ba.dmodBdtheta(), 0, atol=1e-12)
        assert np.allclose(ba.dmodBdzeta(), 0, atol=1e-12)
        assert np.allclose(ba.dmodBds(), 0, atol=1e-12)

        # Check other set_ functions
        ba.set_B0(1.3)
        assert(ba.B0 == 1.3)
        ba.set_Bbar(3.)
        assert(ba.Bbar == 3.)
        ba.set_G0(3.1)
        assert(ba.G0 == 3.1)
        ba.set_I0(3.2)
        assert(ba.I0 == 3.2)
        ba.set_G1(3.3)
        assert(ba.G1 == 3.3)
        ba.set_I1(3.4)
        assert(ba.I1 == 3.4)
        ba.set_iota0(3.5)
        assert(ba.iota0 == 3.5)
        ba.set_psi0(3.6)
        assert(ba.psi0 == 3.6)


@unittest.skipIf(vmec is None, "vmec python package is not found")
class TestingVmec(unittest.TestCase):
    def test_boozerradialinterpolant(self):
        """
        This first loop tests a finite-beta equilibria
        """
        vmec = Vmec(filename_mhd)
        order = 3
        ns_delete = 1
        ntheta = 201
        nzeta = 50
        thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        zetas = np.linspace(0, 2*np.pi/vmec.indata.nfp, nzeta, endpoint=False)

        dtheta = thetas[1]-thetas[0]
        dzeta = zetas[1]-zetas[0]
        thetas, zetas = np.meshgrid(thetas, zetas)
        thetas_flat = thetas.flatten()
        zetas_flat = zetas.flatten()

        isurf = round(0.75*len(vmec.s_full_grid))

        for rescale in [True, False]:
            bri = BoozerRadialInterpolant(vmec, order, rescale=rescale, ns_delete=ns_delete)

            """
            These evaluation points test that the Jacobian sqrtg = (G + iota I)/B^2
            matches sqrt(det(g_ij)).
            """
            # Perform interpolation from full grid
            points = np.zeros((len(thetas_flat), 3))
            points[:, 0] = vmec.s_full_grid[isurf]
            points[:, 1] = thetas_flat
            points[:, 2] = zetas_flat
            bri.set_points(points)

            G = bri.G()[:, 0]
            I = bri.I()[:, 0]
            iota = bri.iota()[:, 0]
            B = bri.modB()[:, 0]
            sqrtg = (G + iota * I)/(B*B)

            R = bri.R()[:, 0]
            dRdtheta = bri.dRdtheta()[:, 0]
            dRdzeta = bri.dRdzeta()[:, 0]
            dRdpsi = bri.dRds()[:, 0]/bri.psi0
            dZdtheta = bri.dZdtheta()[:, 0]
            dZdzeta = bri.dZdzeta()[:, 0]
            dZdpsi = bri.dZds()[:, 0]/bri.psi0
            nu = bri.nu()[:, 0]
            dnudtheta = bri.dnudtheta()[:, 0]
            dnudzeta = bri.dnudzeta()[:, 0]
            dnudpsi = bri.dnuds()[:, 0]/bri.psi0

            phi = zetas_flat - nu
            dphidpsi = - dnudpsi
            dphidtheta = - dnudtheta
            dphidzeta = 1 - dnudzeta

            dXdtheta = dRdtheta * np.cos(phi) - R * np.sin(phi) * dphidtheta
            dYdtheta = dRdtheta * np.sin(phi) + R * np.cos(phi) * dphidtheta
            dXdpsi = dRdpsi * np.cos(phi) - R * np.sin(phi) * dphidpsi
            dYdpsi = dRdpsi * np.sin(phi) + R * np.cos(phi) * dphidpsi
            dXdzeta = dRdzeta * np.cos(phi) - R * np.sin(phi) * dphidzeta
            dYdzeta = dRdzeta * np.sin(phi) + R * np.cos(phi) * dphidzeta

            gpsipsi = dXdpsi**2 + dYdpsi**2 + dZdpsi**2
            gpsitheta = dXdpsi*dXdtheta + dYdpsi*dYdtheta + dZdpsi*dZdtheta
            gpsizeta = dXdpsi*dXdzeta + dYdpsi*dYdzeta + dZdpsi*dZdzeta
            gthetatheta = dXdtheta**2 + dYdtheta**2 + dZdtheta**2
            gthetazeta = dXdtheta*dXdzeta + dYdtheta*dYdzeta + dZdtheta*dZdzeta
            gzetazeta = dXdzeta**2 + dYdzeta**2 + dZdzeta**2

            detg = gpsipsi*(gthetatheta*gzetazeta - gthetazeta**2) \
                - gpsitheta*(gpsitheta*gzetazeta - gthetazeta*gpsizeta) \
                + gpsizeta*(gpsitheta*gthetazeta - gpsizeta*gthetatheta)

            np.allclose(np.sqrt(detg), sqrtg, atol=1e-2)

            """
            These evluation points test that K() satisfies the magnetic differential
            equation: iota dK/dtheta + dK/dzeta = sqrt(g) mu0*p'(psi) + G'(psi) + iota*I'(psi)
            """
            isurf = round(0.75*len(vmec.s_full_grid))
            points = np.zeros((len(thetas_flat), 3))
            points[:, 0] = vmec.s_full_grid[isurf]
            points[:, 1] = thetas_flat
            points[:, 2] = zetas_flat

            bri.set_points(points)

            K = bri.K()[:, 0]
            I = bri.I()[:, 0]
            G = bri.G()[:, 0]
            iota = bri.iota()[:, 0]
            modB = bri.modB()[:, 0]
            sqrtg = (G + iota*I)/(modB*modB)
            dGdpsi = bri.dGds()[:, 0]/bri.psi0
            dIdpsi = bri.dIds()[:, 0]/bri.psi0

            pres = vmec.wout.pres  # on half grid
            dpdpsi = (pres[isurf+1] - pres[isurf-1])/(2*vmec.ds*bri.psi0)
            mu0 = 4*np.pi*1e-7
            rhs = mu0*dpdpsi*sqrtg + dGdpsi + iota*dIdpsi

            K = K.reshape(np.shape(thetas))
            dKdtheta = (np.roll(K, -1, axis=1) - np.roll(K, +1, axis=1))/(2*dtheta)
            dKdzeta = (np.roll(K, -1, axis=0) - np.roll(K, +1, axis=0))/(2*dzeta)
            lhs = iota.reshape(np.shape(thetas))*dKdtheta + dKdzeta

            np.allclose(rhs, lhs.flatten(), atol=1e-2)

        """
        The next loop tests a vacuum equilibria
        """
        vmec = Vmec(filename)
        order = 3
        ns_delete = 1
        for rescale in [True, False]:
            bri = BoozerRadialInterpolant(vmec, order, rescale=rescale, ns_delete=ns_delete)

            """
            These evaluation points test G(), iota(), modB(), R(), and
            associated derivatives by comparing with linear interpolation onto
            vmec full grid.
            """
            # Perform interpolation from full grid
            points = np.zeros((len(vmec.s_half_grid)-1, 3))
            points[:, 0] = vmec.s_full_grid[1:-1]
            bri.set_points(points)
            # Check with linear interpolation from half grid
            G_full = (vmec.wout.bvco[1:-1]+vmec.wout.bvco[2::])/2.
            iota_full = (vmec.wout.iotas[1:-1]+vmec.wout.iotas[2::])/2.
            # magnitude of B at theta = 0, zeta = 0
            modB00 = np.sum(bri.booz.bx.bmnc_b, axis=0)
            modB_full = (modB00[0:-1]+modB00[1::])/2
            rmnc_full = vmec.wout.rmnc
            # major radius at theta = 0, zeta = 0
            R00 = np.sum(rmnc_full, axis=0)

            # Compare splines of derivatives with spline derivatives
            from scipy.interpolate import InterpolatedUnivariateSpline
            G_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bvco[1::])
            iota_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1::])
            modB00_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, modB00)
            R00_spline = InterpolatedUnivariateSpline(vmec.s_full_grid, R00)

            assert np.allclose(bri.G(), G_full, atol=1e-4)
            assert np.allclose(bri.iota(), iota_full, atol=1e-2)
            assert np.allclose(bri.modB()[:, 0], modB_full, atol=1e-2)
            assert np.allclose(bri.R()[:, 0], R00[1:-1], atol=1e-2)
            assert np.allclose(bri.dGds(), G_spline.derivative()(vmec.s_full_grid[1:-1]), atol=1e-3)
            assert np.allclose(bri.diotads(), iota_spline.derivative()(vmec.s_full_grid[1:-1]), atol=1e-2)
            assert np.allclose(bri.dmodBds()[ns_delete+1::, 0], modB00_spline.derivative()(vmec.s_full_grid[ns_delete+2:-1]), atol=1e-2)
            # Only compare away from axis since inacurracies are introduced through
            # spline due to r ~ sqrt(s) behavior
            assert np.allclose(bri.dRds()[5::].T, R00_spline.derivative()(vmec.s_full_grid[6:-1]), atol=1e-2)

            """
            The next evaluation points test Z() and nu()
            """
            points = np.zeros((len(vmec.s_half_grid)-1, 3))
            points[:, 0] = vmec.s_full_grid[1:-1]
            points[:, 1] = 0.
            points[:, 2] = np.pi/3
            bri.set_points(points)

            nu = bri.nu()
            iota = bri.iota()

            zmns_full = vmec.wout.zmns[:, 1:-1]
            lmns_full = vmec.wout.lmns[:, 1:-1]
            # To determine the vmec theta corresponding to theta_b = 0,
            # solve a minimization problem to achieve theta = theta_b - lambda - iota*nu

            def theta_diff(theta, isurf):
                lam = np.sum(lmns_full[:, isurf] * np.sin(vmec.wout.xm*theta-vmec.wout.xn*(np.pi/3-nu[isurf, 0])), axis=0)
                return ((theta + lam) + iota[isurf, 0]*nu[isurf, 0])**2

            from scipy.optimize import minimize
            thetas_vmec = np.zeros((len(lmns_full[0, :])))
            for isurf in range(len(lmns_full[0, :])):
                opt = minimize(theta_diff, 0, args=(isurf))
                thetas_vmec[isurf] = opt.x

            # Compute Z at theta_b = 0, zeta_b = pi/2  and compare with vmec result
            Z0pi = np.sum(zmns_full * np.sin(vmec.wout.xm[:, None]*thetas_vmec[None, :]-vmec.wout.xn[:, None]*(np.pi/3-nu.T)), axis=0)
            Z0pi_spline = InterpolatedUnivariateSpline(vmec.s_full_grid[1:-1], Z0pi)

            assert np.allclose(bri.Z()[:, 0], Z0pi, atol=1e-2)
            assert np.allclose(bri.dZds()[5::].T, Z0pi_spline.derivative()(vmec.s_full_grid[6:-1]), atol=1e-2)

            """
            The next evaluation points test the derivatives of modB, R, Z, and nu
            """

            points = np.zeros((len(vmec.s_half_grid), 3))
            points[:, 0] = vmec.s_half_grid
            bri.set_points(points)

            # Check that angular derivatives integrate to zero
            ntheta = 101
            nzeta = 100
            thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
            zetas = np.linspace(0, 2*np.pi, nzeta, endpoint=False)
            [zetas, thetas] = np.meshgrid(zetas, thetas)
            points = np.zeros((len(thetas.flatten()), 3))
            points[:, 0] = 0.5*np.ones_like(thetas.flatten())
            points[:, 1] = thetas.flatten()
            points[:, 2] = zetas.flatten()
            bri.set_points(points)
            # Check that get_points returns correct points
            points_get = bri.get_points()
            thetas_get = points_get[:, 1]
            assert np.allclose(thetas_get, thetas.flatten())
            assert np.allclose(np.sum(bri.dmodBdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dmodBdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dRdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dRdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dZdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dZdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dnudtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dnudzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)
            # Check that zeta derivatives are small since we are close to QA
            assert np.allclose(bri.dmodBdzeta(), 0, atol=1e-2)

    def test_interpolatedboozerfield_sym(self):
        """
        Here we perform 3D interpolation on a random set of points. Compare
        BoozerRadialInterpolant with InterpolatedBoozerField. We enforce
        nfp and stellarator symmetry in the 3D interpolant.
        """
        vmec = Vmec(filename)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order)

        points = np.zeros((len(vmec.s_half_grid)-1, 3))
        points[:, 0] = vmec.s_full_grid[1:-1]
        bri.set_points(points)

        nfp = vmec.indata.nfp
        n = 12
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = np.pi*(1/4)
        thetamax = np.pi*(3/4)
        thetasteps = n
        zetamin = -2*np.pi/(4*nfp)
        zetamax = 2*np.pi/(4*nfp)
        zetasteps = n*2
        bsh = InterpolatedBoozerField(
            bri, 4, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
            True, nfp=nfp, stellsym=True)
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        points[:, 0] = points[:, 0]*(smax-smin) + smin
        points[:, 1] = points[:, 1]*(thetamax-thetamin) + thetamin
        points[:, 2] = points[:, 2]*(zetamax-zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()

        bsh.set_points(points)
        modBh = bsh.modB()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose((dmodBds - dmodBdsh)/np.mean(np.abs(dmodBds)), 0, atol=1e-2)
        assert np.allclose((dmodBdtheta - dmodBdthetah)/np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2)
        assert np.allclose(dmodBdzeta, 0, atol=1e-3)
        assert np.allclose(dmodBdzetah, 0, atol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(I, 0, atol=1e-3)
        assert np.allclose(Ih, 0, atol=1e-3)
        assert np.allclose(dGds, 0, atol=1e-3)
        assert np.allclose(dGdsh, 0, atol=1e-3)
        assert np.allclose(dIds, 0, atol=1e-3)
        assert np.allclose(dIdsh, 0, atol=1e-3)

    def test_interpolatedboozerfield_no_sym(self):
        """
        Here we perform 3D interpolation on a random set of points. Compare
        BoozerRadialInterpolant with InterpolatedBoozerField. We don't enforce
        nfp and stellarator symmetry in the 3D interpolant.
        """
        vmec = Vmec(filename)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order)

        # Perform interpolation from full grid
        points = np.zeros((len(vmec.s_half_grid)-1, 3))
        points[:, 0] = vmec.s_full_grid[1:-1]
        bri.set_points(points)

        nfp = vmec.indata.nfp

        n = 12
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = np.pi*(1/4)
        thetamax = np.pi*(3/4)
        thetasteps = n
        zetamin = -2*np.pi/(4*nfp)
        zetamax = 2*np.pi/(4*nfp)
        zetasteps = n*2
        bsh = InterpolatedBoozerField(
            bri, 4, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
            True)
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        points[:, 0] = points[:, 0]*(smax-smin) + smin
        points[:, 1] = points[:, 1]*(thetamax-thetamin) + thetamin
        points[:, 2] = points[:, 2]*(zetamax-zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()

        bsh.set_points(points)
        modBh = bsh.modB()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose((dmodBds - dmodBdsh)/np.mean(np.abs(dmodBds)), 0, atol=1e-2)
        assert np.allclose((dmodBdtheta - dmodBdthetah)/np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2)
        assert np.allclose(dmodBdzeta, 0, atol=1e-3)
        assert np.allclose(dmodBdzetah, 0, atol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(I, 0, atol=1e-3)
        assert np.allclose(Ih, 0, atol=1e-3)
        assert np.allclose(dGds, 0, atol=1e-3)
        assert np.allclose(dGdsh, 0, atol=1e-3)
        assert np.allclose(dIds, 0, atol=1e-3)
        assert np.allclose(dIdsh, 0, atol=1e-3)

    def test_interpolated_field_convergence_rate(self):
        """
        Here we test the convergence rate of modB, R, Z, nu, G, I, and iota from
        InterpolatedBoozerField.
        """
        vmec = Vmec(filename_mhd_lowres)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order)

        # Perform interpolation from full grid
        points = np.zeros((len(vmec.s_half_grid)-1, 3))
        points[:, 0] = vmec.s_full_grid[1:-1]
        bri.set_points(points)

        nfp = vmec.indata.nfp
        smin = 0.1
        smax = 0.9
        thetamin = np.pi*(1/4)
        thetamax = np.pi*(3/4)
        zetamin = -2*np.pi/(4*nfp)
        zetamax = 2*np.pi/(4*nfp)
        old_err_1 = 1e6
        old_err_2 = 1e6
        old_err_3 = 1e6
        old_err_4 = 1e6
        old_err_5 = 1e6
        old_err_6 = 1e6
        old_err_7 = 1e6
        for n in [4, 8, 16]:
            ssteps = n
            thetasteps = n
            zetasteps = n
            bsh = InterpolatedBoozerField(
                bri, 1, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
                True, nfp=nfp, stellsym=True)
            err_1 = np.mean(bsh.estimate_error_modB(1000))
            err_2 = np.mean(bsh.estimate_error_I(1000))
            err_3 = np.mean(bsh.estimate_error_G(1000))
            err_4 = np.mean(bsh.estimate_error_iota(1000))
            err_5 = np.mean(bsh.estimate_error_R(1000))
            err_6 = np.mean(bsh.estimate_error_Z(1000))
            err_7 = np.mean(bsh.estimate_error_nu(1000))

            assert err_1 < 0.6**2 * old_err_1
            assert err_2 < 0.6**2 * old_err_2
            assert err_3 < 0.6**2 * old_err_3
            assert err_4 < 0.6**2 * old_err_4
            assert err_5 < 0.6**2 * old_err_5
            assert err_6 < 0.6**2 * old_err_6
            assert err_7 < 0.6**2 * old_err_7

            old_err_1 = err_1
            old_err_2 = err_2
            old_err_3 = err_3
            old_err_4 = err_4
            old_err_5 = err_5
            old_err_6 = err_6
            old_err_7 = err_7


if __name__ == "__main__":
    unittest.main()
