from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField, BoozerAnalytic
import numpy as np
import unittest
from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = str((TEST_DIR / 'wout_LandremanPaul2021_QA_lowres.nc').resolve())
filename_mhd = str((TEST_DIR / 'wout_n3are_R7.75B5.7.nc').resolve())
filename_mhd_lowres = str((TEST_DIR / 'wout_n3are_R7.75B5.7_lowres.nc').resolve())
filename_mhd_lasym = str((TEST_DIR / 'wout_10x10.nc').resolve())

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
    from simsopt.mhd.boozer import Boozer


class TestingAnalytic(unittest.TestCase):
    def test_boozeranalytic(self):
        etabar = 1.1
        B0 = 1.0
        Bbar = 1.0
        N = 0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 0.4
        K1 = 1.8
        ba = BoozerAnalytic(etabar, B0, N, G0, psi0, iota0, K1)

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

        # Check that K_derivs matches dKdtheta and dKdzeta
        K_derivs = ba.K_derivs()
        assert np.allclose(ba.dKdtheta(), K_derivs[:, 0])
        assert np.allclose(ba.dKdzeta(), K_derivs[:, 1])

        # Check that modB_derivs matches (dmodBds,dmodBdtheta,dmodBdzeta)
        modB_derivs = ba.modB_derivs()
        assert np.allclose(ba.dmodBds(), modB_derivs[:, 0])
        assert np.allclose(ba.dmodBdtheta(), modB_derivs[:, 1])
        assert np.allclose(ba.dmodBdzeta(), modB_derivs[:, 2])

        # Check other set_ functions
        ba.set_B0(1.3)
        assert (ba.B0 == 1.3)
        ba.set_Bbar(3.)
        assert (ba.Bbar == 3.)
        ba.set_G0(3.1)
        assert (ba.G0 == 3.1)
        ba.set_I0(3.2)
        assert (ba.I0 == 3.2)
        ba.set_G1(3.3)
        assert (ba.G1 == 3.3)
        ba.set_I1(3.4)
        assert (ba.I1 == 3.4)
        ba.set_iota0(3.5)
        assert (ba.iota0 == 3.5)
        ba.set_psi0(3.6)
        assert (ba.psi0 == 3.6)
        ba.set_K1(3.7)
        assert (ba.K1 == 3.7)


@unittest.skipIf(vmec is None, "vmec python package is not found")
class TestingVmec(unittest.TestCase):
    def test_boozerradialinterpolant_finite_beta(self):
        """
        This first loop tests a finite-beta equilibria
        """
        # This one is stellarator symmetric
        vmec_sym = Vmec(filename_mhd)
        vmec_asym = Vmec(filename_mhd_lasym)
        order = 3
        ns_delete = 1
        ntheta = 21
        nzeta = 20
        thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        zetas = np.linspace(0, 2*np.pi/vmec_sym.wout.nfp, nzeta, endpoint=False)

        dtheta = thetas[1]-thetas[0]
        dzeta = zetas[1]-zetas[0]
        thetas, zetas = np.meshgrid(thetas, zetas)
        thetas_flat = thetas.flatten()
        zetas_flat = zetas.flatten()

        # The following tests different initializations of BoozerRadialInterpolant
        for asym in [True, False]:
            if asym:
                vmec = vmec_asym
            else:
                vmec = vmec_sym
            for rescale in [True, False]:
                # First, do not initialize grid
                booz = Boozer(vmec, mpol=1, ntor=1)
                bri = BoozerRadialInterpolant(booz, order, rescale=rescale,
                                              ns_delete=ns_delete, mpol=1, ntor=1)

                s_0 = np.copy(bri.s_half_ext)
                G_0 = bri.G_spline(0.5)

                # Next, initialize wrong size of radial grid
                booz = Boozer(vmec, mpol=1, ntor=1)
                booz.register(vmec.s_half_grid[0:5])
                booz.run()
                bri = BoozerRadialInterpolant(booz, order, rescale=False,
                                              ns_delete=ns_delete, mpol=1, ntor=1)

                s_1 = np.copy(bri.s_half_ext)
                G_1 = bri.G_spline(0.5)

                # Next, intialize correct size, but wrong values
                booz = Boozer(vmec, mpol=1, ntor=1)
                s_grid = np.asarray(vmec.s_half_grid)
                s_grid[0] = s_grid[0]*0.5
                booz.register(s_grid)
                booz.run()
                bri = BoozerRadialInterpolant(booz, order, rescale=False,
                                              ns_delete=ns_delete, mpol=1, ntor=1)

                s_2 = np.copy(bri.s_half_ext)
                G_2 = bri.G_spline(0.5)

                assert np.allclose(s_0, s_1)
                assert np.allclose(s_0, s_2)
                assert G_0 == G_1
                assert G_0 == G_2

        for asym in [True, False]:
            if asym:
                vmec = vmec_asym
            else:
                vmec = vmec_sym
            # Compute full boozer transformation once
            booz = Boozer(vmec, mpol=20, ntor=18)
            booz.register(vmec.s_half_grid)
            booz.run()
            for rescale in [False, True]:
                bri = BoozerRadialInterpolant(booz, order, rescale=rescale,
                                              ns_delete=ns_delete, mpol=20,
                                              ntor=18)
                isurf = round(0.75*len(vmec.s_full_grid))

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

                assert np.allclose(np.sqrt(detg)/np.mean(np.abs(sqrtg)), np.abs(sqrtg)/np.mean(np.abs(sqrtg)), atol=1e-2)

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
                dKdtheta = bri.dKdtheta()[:, 0]
                dKdzeta = bri.dKdzeta()[:, 0]
                K_derivs = bri.K_derivs()

                assert np.allclose(K_derivs[:, 0], dKdtheta)
                assert np.allclose(K_derivs[:, 1], dKdzeta)

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
                dKdtheta = dKdtheta.reshape(np.shape(thetas))
                dKdzeta = dKdzeta.reshape(np.shape(zetas))
                lhs = iota.reshape(np.shape(thetas))*dKdtheta + dKdzeta

                assert np.allclose(rhs, lhs.flatten(), atol=1e-2)

    def test_boozerradialinterpolant_vacuum(self):
        vmec_asym = Vmec(filename_mhd_lasym)
        """
        The next loop tests a vacuum equilibria
        """
        order = 3
        ns_delete = 2
        for asym in [True, False]:
            if asym:
                vmec = vmec_asym
            else:
                vmec = Vmec(filename)
            for rescale in [True, False]:
                bri = BoozerRadialInterpolant(vmec, order, mpol=20, ntor=18,
                                              rescale=rescale, ns_delete=ns_delete,
                                              no_K=True)

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

                # Compare splines of derivatives with spline derivatives
                from scipy.interpolate import InterpolatedUnivariateSpline
                G_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bvco[1::])
                iota_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1::])
                modB00_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, modB00)

                rmnc_half = bri.booz.bx.rmnc_b
                rmnc_full = 0.5*(bri.booz.bx.rmnc_b[:, 0:-1] + bri.booz.bx.rmnc_b[:, 1::])
                # major radius at theta = 0, zeta = 0
                R00_half = np.sum(rmnc_half, axis=0)
                R00_full = np.sum(rmnc_full, axis=0)
                R00_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, R00_half)

                assert np.allclose(bri.G()[:, 0], G_full, rtol=1e-4)
                assert np.allclose(bri.iota()[:, 0], iota_full, rtol=1e-2)
                assert np.allclose(bri.modB()[:, 0], modB_full, rtol=1e-2)
                assert np.allclose(bri.R()[:, 0], R00_full, rtol=1e-2)

                # Only compare away from axis since inacurracies are introduced through
                # spline due to r ~ sqrt(s) behavior
                if not bri.stellsym:
                    mean_dGds = np.mean(np.abs(bri.dGds()[5::, 0]))
                else:
                    # This is a vacuum case, so dGds is close to zero
                    mean_dGds = 1

                assert np.allclose(bri.dGds()[5::, 0]/mean_dGds, G_spline.derivative()(vmec.s_full_grid[6:-1])/mean_dGds, atol=1e-2)
                mean_diotads = np.mean(np.abs(bri.diotads()[5::, 0]))
                assert np.allclose(bri.diotads()[5::, 0]/mean_diotads, iota_spline.derivative()(vmec.s_full_grid[6:-1])/mean_diotads, atol=1e-2)
                assert np.allclose(bri.dmodBds()[5::, 0], modB00_spline.derivative()(vmec.s_full_grid[6:-1]), rtol=1e-2)
                mean_dRds = np.mean(np.abs(bri.dRds()))
                assert np.allclose(bri.dRds()[5::, 0]/mean_dRds, R00_spline.derivative()(vmec.s_full_grid[6:-1])/mean_dRds, atol=1e-2)

                """
                The next evaluation points test Z() and nu()
                """
                points = np.zeros((len(vmec.s_half_grid), 3))
                points[:, 0] = vmec.s_half_grid
                points[:, 1] = 0.
                points[:, 2] = np.pi/3
                bri.set_points(points)

                nu = bri.nu()
                iota = bri.iota()

                # zmns/zmnc on full grid
                # lmnc/lmns on half grid
                zmns_full = vmec.wout.zmns[:, 1:]
                zmns_half = 0.5*(vmec.wout.zmns[:, 0:-1] + vmec.wout.zmns[:, 1::])
                lmns_full = 0.5*(vmec.wout.lmns[:, 1:-1] + vmec.wout.lmns[:, 2::])
                lmns_half = vmec.wout.lmns[:, 1::]
                if not bri.stellsym:
                    lmnc_full = 0.5*(vmec.wout.lmnc[:, 1:-1] + vmec.wout.lmnc[:, 2::])
                    lmnc_half = vmec.wout.lmnc[:, 1::]
                    zmnc_full = vmec.wout.zmnc[:, 1:-1]
                    zmnc_half = 0.5*(vmec.wout.zmnc[:, 0:-1] + vmec.wout.zmnc[:, 1::])
                else:
                    lmnc_full = np.zeros_like(lmns_full)
                    zmnc_full = np.zeros_like(zmns_full)
                    lmnc_half = np.zeros_like(lmns_half)
                    zmnc_half = np.zeros_like(zmns_half)

                # Determine the vmec theta/phi corresponding to theta_b = 0, zeta_b = pi/3
                # Here zeta_b = phi + nu
                # theta + lambda - iota * phi = theta_b - iota * zeta_b
                # theta + lambda - iota * (pi/3 - nu) = - iota * pi/3
                # theta + lambda + iota * nu = 0
                def theta_diff(theta, isurf):
                    lam = np.sum(lmns_half[:, isurf] * np.sin(vmec.wout.xm*theta-vmec.wout.xn*(np.pi/3-nu[isurf, 0]))
                                 + lmnc_half[:, isurf] * np.cos(vmec.wout.xm*theta-vmec.wout.xn*(np.pi/3-nu[isurf, 0])), axis=0)
                    return ((theta + lam) + iota[isurf, 0]*(nu[isurf, 0]))**2

                from scipy.optimize import minimize
                thetas_vmec = np.zeros((len(vmec.s_half_grid),))
                for isurf in range(len(vmec.s_half_grid)):
                    opt = minimize(theta_diff, 0, args=(isurf))
                    thetas_vmec[isurf] = opt.x

                # Compute Z at theta_b = 0, zeta_b = pi/2  and compare with vmec result
                Z0pi = np.sum(zmns_half * np.sin(vmec.wout.xm[:, None]*thetas_vmec[None, :]-vmec.wout.xn[:, None]*(np.pi/3-nu[None, :, 0]))
                              + zmnc_half * np.cos(vmec.wout.xm[:, None]*thetas_vmec[None, :]-vmec.wout.xn[:, None]*(np.pi/3-nu[None, :, 0])), axis=0)
                Z0pi_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, Z0pi)

                mean_dZds = np.mean(np.abs(bri.dZds()[5::, 0]))

                assert np.allclose(bri.Z()[:, 0], Z0pi, atol=1e-2)
                assert np.allclose(bri.dZds()[5::, 0]/mean_dZds, Z0pi_spline.derivative()(vmec.s_half_grid[5::])/mean_dZds, atol=5e-2)

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
                assert np.allclose(np.sum(bri.dKdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
                assert np.allclose(np.sum(bri.dKdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)
                # Check that zeta derivatives are small since we are close to QA
                if not asym:
                    assert np.allclose(bri.dmodBdzeta(), 0, atol=1e-2)

                assert np.allclose(bri.R_derivs()[:, 0], bri.dRds()[:, 0])
                assert np.allclose(bri.R_derivs()[:, 1], bri.dRdtheta()[:, 0])
                assert np.allclose(bri.R_derivs()[:, 2], bri.dRdzeta()[:, 0])
                assert np.allclose(bri.Z_derivs()[:, 0], bri.dZds()[:, 0])
                assert np.allclose(bri.Z_derivs()[:, 1], bri.dZdtheta()[:, 0])
                assert np.allclose(bri.Z_derivs()[:, 2], bri.dZdzeta()[:, 0])
                assert np.allclose(bri.nu_derivs()[:, 0], bri.dnuds()[:, 0])
                assert np.allclose(bri.nu_derivs()[:, 1], bri.dnudtheta()[:, 0])
                assert np.allclose(bri.nu_derivs()[:, 2], bri.dnudzeta()[:, 0])
                assert np.allclose(bri.modB_derivs()[:, 0], bri.dmodBds()[:, 0])
                assert np.allclose(bri.modB_derivs()[:, 1], bri.dmodBdtheta()[:, 0])
                assert np.allclose(bri.modB_derivs()[:, 2], bri.dmodBdzeta()[:, 0])

    def test_interpolatedboozerfield_sym(self):
        """
        Here we perform 3D interpolation on a random set of points. Compare
        BoozerRadialInterpolant with InterpolatedBoozerField. We enforce
        nfp and stellarator symmetry in the 3D interpolant.
        """
        vmec = Vmec(filename_mhd_lowres)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order, mpol=5, ntor=5, rescale=True)

        nfp = vmec.wout.nfp
        n = 12
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = 0
        thetamax = np.pi
        thetasteps = n
        zetamin = 0
        zetamax = 2*np.pi/(nfp)
        zetasteps = n*2
        bsh = InterpolatedBoozerField(
            bri, 4, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
            True, stellsym=True, nfp=nfp)

        # Compute points outside of interpolation range
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        thetamin = -np.pi
        thetamax = 2*np.pi
        zetamin = -2*np.pi/nfp
        zetamax = 4*np.pi/nfp
        points[:, 0] = points[:, 0]*(smax-smin) + smin
        points[:, 1] = points[:, 1]*(thetamax-thetamin) + thetamin
        points[:, 2] = points[:, 2]*(zetamax-zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        R = bri.R()
        dRdtheta = bri.dRdtheta()
        dRdzeta = bri.dRdzeta()
        dRds = bri.dRds()
        Z = bri.Z()
        dZdtheta = bri.dZdtheta()
        dZdzeta = bri.dZdzeta()
        dZds = bri.dZds()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        nu = bri.nu()
        dnudtheta = bri.dnudtheta()
        dnudzeta = bri.dnudzeta()
        dnuds = bri.dnuds()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()
        K = bri.K()
        dKdtheta = bri.dKdtheta()
        dKdzeta = bri.dKdzeta()

        bsh.set_points(points)
        modBh = bsh.modB()
        Rh = bsh.R()
        dRdthetah = bsh.dRdtheta()
        dRdzetah = bsh.dRdzeta()
        dRdsh = bsh.dRds()
        Zh = bsh.Z()
        dZdthetah = bsh.dZdtheta()
        dZdzetah = bsh.dZdzeta()
        dZdsh = bsh.dZds()
        nuh = bsh.nu()
        dnudthetah = bsh.dnudtheta()
        dnudzetah = bsh.dnudzeta()
        dnudsh = bsh.dnuds()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()
        Kh = bsh.K()
        dKdthetah = bsh.dKdtheta()
        dKdzetah = bsh.dKdzeta()

        assert np.allclose(K, Kh, rtol=1e-3)
        assert np.allclose(dKdtheta, dKdthetah, rtol=1e-3)
        assert np.allclose((dKdzeta - dKdzetah)/np.mean(np.abs(dKdzeta)), 0, atol=1e-3)

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose((dmodBds - dmodBdsh)/np.mean(np.abs(dmodBds)), 0, atol=1e-2)
        assert np.allclose((dmodBdtheta - dmodBdthetah)/np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2)
        assert np.allclose(dmodBdzeta - dmodBdzetah, 0, atol=1e-3)

        assert np.allclose(R, Rh, rtol=1e-3)
        assert np.allclose(dRds, dRdsh, rtol=1e-3)
        assert np.allclose(dRdtheta, dRdthetah, rtol=1e-3)
        assert np.allclose(dRdzeta, dRdzetah, rtol=1e-3)

        assert np.allclose(Z, Zh, rtol=1e-3)
        assert np.allclose(dZds, dZdsh, rtol=1e-3)
        assert np.allclose(dZdtheta, dZdthetah, rtol=1e-3)
        assert np.allclose(dZdzeta, dZdzetah, rtol=1e-3)

        assert np.allclose(nu, nuh, rtol=1e-3)
        assert np.allclose(dnuds, dnudsh, rtol=1e-3)
        assert np.allclose(dnudtheta, dnudthetah, rtol=1e-3)
        assert np.allclose(dnudzeta, dnudzetah, rtol=1e-3)

        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(I, Ih, rtol=1e-3)

        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(dGds, dGdsh, rtol=1e-3)
        assert np.allclose(dIds, dIdsh, rtol=1e-3)

        assert np.allclose(bsh.K_derivs()[:, 0], bsh.dKdtheta()[:, 0])
        assert np.allclose(bsh.K_derivs()[:, 1], bsh.dKdzeta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 0], bsh.dRds()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 1], bsh.dRdtheta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 2], bsh.dRdzeta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 0], bsh.dZds()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 1], bsh.dZdtheta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 2], bsh.dZdzeta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 0], bsh.dnuds()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 1], bsh.dnudtheta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 2], bsh.dnudzeta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 0], bsh.dmodBds()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 1], bsh.dmodBdtheta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 2], bsh.dmodBdzeta()[:, 0])

    def test_interpolatedboozerfield_no_sym(self):
        """
        Here we perform 3D interpolation on a random set of points. Compare
        BoozerRadialInterpolant with InterpolatedBoozerField. We don't enforce
        nfp and stellarator symmetry in the 3D interpolant.
        """
        vmec = Vmec(filename_mhd_lowres)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order, mpol=5, ntor=5, rescale=True)

        nfp = vmec.wout.nfp
        n = 12
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = 0
        thetamax = 2*np.pi
        thetasteps = n
        zetamin = 0
        zetamax = 2*np.pi
        zetasteps = n*2

        with self.assertRaises(ValueError):
            bsh = InterpolatedBoozerField(
                bri, 4, [smin, smax, ssteps], [-np.pi, 0, thetasteps], [zetamin, zetamax, zetasteps],
                True, stellsym=False)
        with self.assertRaises(ValueError):
            bsh = InterpolatedBoozerField(
                bri, 4, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [-np.pi, 0, zetasteps],
                True, stellsym=False)

        bsh = InterpolatedBoozerField(
            bri, 4, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
            True, stellsym=False)

        # Compute points outside of interpolation range
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        thetamin = -2*np.pi
        thetamax = 4*np.pi
        zetamin = -2*np.pi
        zetamax = 4*np.pi
        points[:, 0] = points[:, 0]*(smax-smin) + smin
        points[:, 1] = points[:, 1]*(thetamax-thetamin) + thetamin
        points[:, 2] = points[:, 2]*(zetamax-zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        R = bri.R()
        dRdtheta = bri.dRdtheta()
        dRdzeta = bri.dRdzeta()
        dRds = bri.dRds()
        Z = bri.Z()
        dZdtheta = bri.dZdtheta()
        dZdzeta = bri.dZdzeta()
        dZds = bri.dZds()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        nu = bri.nu()
        dnudtheta = bri.dnudtheta()
        dnudzeta = bri.dnudzeta()
        dnuds = bri.dnuds()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()
        K = bri.K()
        dKdtheta = bri.dKdtheta()
        dKdzeta = bri.dKdzeta()

        bsh.set_points(points)
        modBh = bsh.modB()
        Rh = bsh.R()
        dRdthetah = bsh.dRdtheta()
        dRdzetah = bsh.dRdzeta()
        dRdsh = bsh.dRds()
        Zh = bsh.Z()
        dZdthetah = bsh.dZdtheta()
        dZdzetah = bsh.dZdzeta()
        dZdsh = bsh.dZds()
        nuh = bsh.nu()
        dnudthetah = bsh.dnudtheta()
        dnudzetah = bsh.dnudzeta()
        dnudsh = bsh.dnuds()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()
        Kh = bsh.K()
        dKdthetah = bsh.dKdtheta()
        dKdzetah = bsh.dKdzeta()

        assert np.allclose(K, Kh, rtol=1e-3)
        assert np.allclose(dKdtheta, dKdthetah, rtol=1e-3)
        assert np.allclose((dKdzeta - dKdzetah)/np.mean(np.abs(dKdzeta)), 0, atol=1e-2)

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose((dmodBds - dmodBdsh)/np.mean(np.abs(dmodBds)), 0, atol=1e-2)
        assert np.allclose((dmodBdtheta - dmodBdthetah)/np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2)
        assert np.allclose(dmodBdzeta - dmodBdzetah, 0, atol=1e-3)

        assert np.allclose(R, Rh, rtol=1e-3)
        assert np.allclose((dRds - dRdsh)/np.mean(np.abs(dRds)), 0, atol=1e-3)
        assert np.allclose((dRdtheta - dRdthetah)/np.mean(np.abs(dRdtheta)), 0, atol=1e-3)
        assert np.allclose((dRdtheta - dRdthetah)/np.mean(np.abs(dRdtheta)), 0, atol=1e-3)

        assert np.allclose(Z, Zh, rtol=1e-3)
        assert np.allclose((dZds - dZdsh)/np.mean(np.abs(dZds)), 0, atol=1e-3)
        assert np.allclose((dZdtheta - dZdthetah)/np.mean(np.abs(dZdtheta)), 0, atol=1e-3)
        assert np.allclose((dZdtheta - dZdthetah)/np.mean(np.abs(dZdtheta)), 0, atol=1e-3)

        assert np.allclose(nu, nuh, rtol=1e-3)
        assert np.allclose(dnuds, dnudsh, rtol=1e-3)
        assert np.allclose(dnudtheta, dnudthetah, rtol=1e-3)
        assert np.allclose((dnudzeta - dnudzetah)/np.mean(np.abs(dnudzeta)), 0, atol=1e-3)

        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(I, Ih, rtol=1e-3)

        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(dGds, dGdsh, rtol=1e-3)
        assert np.allclose(dIds, dIdsh, rtol=1e-3)

        assert np.allclose(bsh.K_derivs()[:, 0], bsh.dKdtheta()[:, 0])
        assert np.allclose(bsh.K_derivs()[:, 1], bsh.dKdzeta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 0], bsh.dRds()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 1], bsh.dRdtheta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 2], bsh.dRdzeta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 0], bsh.dZds()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 1], bsh.dZdtheta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 2], bsh.dZdzeta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 0], bsh.dnuds()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 1], bsh.dnudtheta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 2], bsh.dnudzeta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 0], bsh.dmodBds()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 1], bsh.dmodBdtheta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 2], bsh.dmodBdzeta()[:, 0])

    def test_interpolatedboozerfield_convergence_rate(self):
        """
        Here we test the convergence rate of modB, R, Z, nu, K, G, I, and iota from
        InterpolatedBoozerField.
        """
        vmec = Vmec(filename_mhd_lowres)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order, mpol=10, ntor=10)

        # Perform interpolation from full grid
        points = np.zeros((len(vmec.s_half_grid)-1, 3))
        points[:, 0] = vmec.s_full_grid[1:-1]
        bri.set_points(points)

        nfp = vmec.wout.nfp
        smin = 0.1
        smax = 0.9
        thetamin = np.pi*(1/4)
        thetamax = np.pi*(3/4)
        zetamin = 2*np.pi/(4*nfp)
        zetamax = 2*np.pi*3/(4*nfp)
        old_err_modB = 1e6
        old_err_I = 1e6
        old_err_G = 1e6
        old_err_iota = 1e6
        old_err_R = 1e6
        old_err_Z = 1e6
        old_err_nu = 1e6
        old_err_K = 1e6
        for n in [4, 8, 16]:
            ssteps = n
            thetasteps = n
            zetasteps = n
            bsh = InterpolatedBoozerField(
                bri, 1, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
                True, nfp=nfp, stellsym=True)
            err_modB = np.mean(bsh.estimate_error_modB(1000))
            err_I = np.mean(bsh.estimate_error_I(1000))
            err_G = np.mean(bsh.estimate_error_G(1000))
            err_iota = np.mean(bsh.estimate_error_iota(1000))
            err_R = np.mean(bsh.estimate_error_R(1000))
            err_Z = np.mean(bsh.estimate_error_Z(1000))
            err_nu = np.mean(bsh.estimate_error_nu(1000))
            err_K = np.mean(bsh.estimate_error_K(1000))

            assert err_modB < 0.6**2 * old_err_modB
            assert err_I < 0.6**2 * old_err_I
            assert err_G < 0.6**2 * old_err_G
            assert err_iota < 0.6**2 * old_err_iota
            assert err_R < 0.6**2 * old_err_R
            assert err_Z < 0.6**2 * old_err_Z
            assert err_nu < 0.6**2 * old_err_nu
            assert err_K < 0.6**2 * old_err_K

            old_err_modB = err_modB
            old_err_I = err_I
            old_err_G = err_G
            old_err_iota = err_iota
            old_err_R = err_R
            old_err_Z = err_Z
            old_err_nu = err_nu
            old_err_K = err_K


if __name__ == "__main__":
    unittest.main()
