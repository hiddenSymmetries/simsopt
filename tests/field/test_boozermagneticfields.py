from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField, BoozerAnalytic
from simsoptpp import inverse_fourier_transform_odd, inverse_fourier_transform_even
import numpy as np
import unittest
from pathlib import Path
from scipy.io import netcdf_file
from simsopt._core.util import align_and_pad, allocate_aligned_and_padded_array
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename_vac = str((TEST_DIR / 'boozmn_LandremanPaul2021_QA_lowres.nc').resolve())
filename_vac_wout = str((TEST_DIR / 'wout_LandremanPaul2021_QA_lowres.nc').resolve())

filename_mhd = str((TEST_DIR / 'boozmn_n3are_R7.75B5.7.nc').resolve())
filename_mhd_wout = str((TEST_DIR / 'wout_n3are_R7.75B5.7.nc').resolve())
filename_mhd_reduced = str((TEST_DIR / 'boozmn_n3are_R7.75B5.7_reduced.nc').resolve())
filename_mhd_reordered = str((TEST_DIR / 'boozmn_n3are_R7.75B5.7_reordered.nc').resolve())
filename_mhd_lasym = str((TEST_DIR / 'boozmn_ITERModel_reference.nc').resolve())
filename_mhd_lasym_wout = str((TEST_DIR / 'wout_ITERModel_reference.nc').resolve())
filename_mhd_lasym_reduced = str((TEST_DIR / 'boozmn_ITERModel_reference_reduced.nc').resolve())
filename_mhd_lasym_reordered = str((TEST_DIR / 'boozmn_ITERModel_reference_reordered.nc').resolve())

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError as e:
    comm = None

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
        ba.set_K1(3.7)
        assert(ba.K1 == 3.7)


class TestingFiniteBeta(unittest.TestCase):
    def test_boozerradialinterpolant_finite_beta(self):
        """
        This first loop tests a finite-beta equilibria
        """
        # This one is stellarator symmetric
        filename_sym = filename_mhd
        filename_sym_wout = filename_mhd_wout
        filename_sym_reduced = filename_mhd_reduced
        filename_sym_reordered = filename_mhd_reordered
        filename_asym = filename_mhd_lasym
        filename_asym_wout = filename_mhd_lasym_wout
        filename_asym_reduced = filename_mhd_lasym_reduced
        filename_asym_reordered = filename_mhd_lasym_reordered
        order = 3
        ns_delete = 1
        ntheta = 21
        nzeta = 20

        # The following tests different initializations of BoozerRadialInterpolant
        for asym in [True,False]:
            if asym:
                filename = filename_asym
                filename_wout = filename_asym_wout
                filename_reduced = filename_asym_reduced
                filename_reordered = filename_asym_reordered
            else:
                filename = filename_sym
                filename_wout = filename_sym_wout
                filename_reduced = filename_sym_reduced
                filename_reordered = filename_sym_reordered
                
            for rescale in [True,False]:
                # First, initialize correctly-sized grid (booz_xform)
                bri = BoozerRadialInterpolant(filename, order, rescale=rescale,
                                              ns_delete=ns_delete, comm=comm)
                
                thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
                zetas = np.linspace(0, 2*np.pi/bri.nfp, nzeta, endpoint=False)

                dtheta = thetas[1]-thetas[0]
                dzeta = zetas[1]-zetas[0]
                thetas, zetas = np.meshgrid(thetas, zetas)
                thetas_flat = thetas.flatten()
                zetas_flat = zetas.flatten()

                s_0 = np.copy(bri.s_half_ext)
                G_0 = bri.G_spline(0.5)

                # Next, initialize with wout file and check for consistency
                bri = BoozerRadialInterpolant(filename_wout, order, rescale=rescale,ns_delete=ns_delete, comm=comm)

                s_1 = np.copy(bri.s_half_ext)
                G_1 = bri.G_spline(0.5)

                assert np.allclose(s_0, s_1)
                assert G_0 == G_1

                # Next, initialize wrong size of radial grid
                with self.assertRaises(ValueError):
                    bri = BoozerRadialInterpolant(filename_reduced, order, rescale=False,ns_delete=ns_delete, comm=comm)
                
                # Next, initialize grid with incorrect order or s points
                with self.assertRaises(ValueError):
                    bri = BoozerRadialInterpolant(filename_reordered, order,  rescale=False,ns_delete=ns_delete, comm=comm)


        # These tests require higher resolution equilibrium, since they check for consistency of the Jacobian and satisfying the magnetic differential equation
        for asym in [False,True]:
            if asym:
                filename = filename_mhd_lasym
                filename_wout = filename_mhd_lasym_wout
            else:
                filename = filename_mhd
                filename_wout = filename_mhd_wout
            
            for rescale in [False,True]:
                bri = BoozerRadialInterpolant(filename, order, rescale=rescale,
                                              ns_delete=ns_delete, comm=comm)
                isurf = round(0.75*len(bri.s_half_ext))

                """
                These evaluation points test that the Jacobian sqrtg = (G + iota I)/B^2
                matches sqrt(det(g_ij)).
                """
                # Perform interpolation from full grid
                points = np.zeros((len(thetas_flat), 3))
                points[:, 0] = bri.s_half_ext[isurf]
                points[:, 1] = thetas_flat
                points[:, 2] = zetas_flat
                bri.set_points(points)

                G = bri.G()[:, 0]
                I = bri.I()[:, 0]
                iota = bri.iota()[:, 0]
                B = bri.modB()[:, 0]
                sqrtg = (G + iota * I)/(B*B)

                detg = np.abs(np.sqrt(np.abs(bri.get_covariant_metric().det()))/bri.psi0)

                assert np.allclose(detg/np.mean(np.abs(sqrtg)), np.abs(sqrtg)/np.mean(np.abs(sqrtg)), atol=1e-2)

                """
                These evluation points test that K() satisfies the magnetic differential equation: iota dK/dtheta + dK/dzeta = sqrt(g) mu0*p'(psi) + G'(psi) + iota*I'(psi)
                """
                isurf = round(0.75*len(bri.s_half_ext))
                points = np.zeros((len(thetas_flat), 3))
                s_full = np.linspace(0, 1, bri.bx.ns_b+1)
                points[:, 0] = s_full[isurf]
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

                f = netcdf_file(filename_wout, mmap=False)
                pres = f.variables['pres'][()]
                ds = bri.s_half_ext[2] - bri.s_half_ext[1]
                dpdpsi = (pres[isurf+1] - pres[isurf-1])/(2*ds*bri.psi0)
                mu0 = 4*np.pi*1e-7
                rhs = mu0*dpdpsi*sqrtg + dGdpsi + iota*dIdpsi

                K = K.reshape(np.shape(thetas))
                dKdtheta = dKdtheta.reshape(np.shape(thetas))
                dKdzeta = dKdzeta.reshape(np.shape(zetas))
                lhs = iota.reshape(np.shape(thetas))*dKdtheta + dKdzeta

                assert np.allclose(rhs, lhs.flatten(), atol=1e-2)

    def test_boozerradialinterpolant_vacuum(self):
        """
        The next loop tests a vacuum equilibria
        """
        order = 3
        ns_delete = 2
        for asym in [True, False]:
            if asym:
                filename = filename_mhd_lasym
                filename_wout = filename_mhd_lasym_wout
            else:
                filename = filename_vac
                filename_wout = filename_vac_wout
            for rescale in [True, False]:
                bri = BoozerRadialInterpolant(filename, order, mpol=20, ntor=18,
                                              rescale=rescale, ns_delete=ns_delete,
                                              no_K=True, comm=comm)

                """
                These evaluation points test G(), iota(), modB(), R(), and
                associated derivatives by comparing with linear interpolation onto vmec full grid.
                """
                # Perform interpolation from full grid
                points = np.zeros((len(bri.s_half_ext)-3, 3))
                s_full = np.linspace(0, 1, bri.bx.ns_b+1)
                points[:, 0] = s_full[1:-1]
                bri.set_points(points)

                # Check with linear interpolation from half grid
                f = netcdf_file(filename_wout, mmap=False)
                bvco = f.variables['bvco'][()]
                iotas = f.variables['iotas'][()]
                G_full = (bvco[1:-1]+bvco[2::])/2.
                iota_full = (iotas[1:-1]+iotas[2::])/2.
                # magnitude of B at theta = 0, zeta = 0
                if comm is not None:
                    if comm.rank == 0:
                        modB00 = np.sum(bri.bx.bmnc_b, axis=0)
                    else:
                        modB00 = None
                    modB00 = comm.bcast(modB00, root=0)
                else:
                    modB00 = np.sum(bri.bx.bmnc_b, axis=0)
                modB_full = (modB00[0:-1]+modB00[1::])/2

                # Compare splines of derivatives with spline derivatives
                f = netcdf_file(filename_wout, mmap=False)

                bvco = f.variables['bvco'][()][1::]
                iotas = f.variables['iotas'][()][1::]
                G_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], bvco)
                iota_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], iotas)
                modB00_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], modB00)

                if comm is not None:
                    if comm.rank == 0:
                        rmnc_half = bri.bx.rmnc_b
                        rmnc_full = 0.5*(bri.bx.rmnc_b[:, 0:-1] + bri.bx.rmnc_b[:, 1::])
                    else:
                        rmnc_half = None
                        rmnc_full = None
                    rmnc_half = comm.bcast(rmnc_half, root=0)
                    rmnc_full = comm.bcast(rmnc_full, root=0)
                else:
                    rmnc_half = bri.bx.rmnc_b
                    rmnc_full = 0.5*(bri.bx.rmnc_b[:, 0:-1] + bri.bx.rmnc_b[:, 1::])
                # major radius at theta = 0, zeta = 0
                R00_half = np.sum(rmnc_half, axis=0)
                R00_full = np.sum(rmnc_full, axis=0)
                R00_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], R00_half)

                assert np.allclose(bri.G()[:, 0], G_full, rtol=1e-4)
                assert np.allclose(bri.iota()[:, 0], iota_full, rtol=1e-2)
                assert np.allclose(bri.modB()[:, 0], modB_full, rtol=1e-2)
                assert np.allclose(bri.R()[:, 0], R00_full, rtol=1e-2)

                # Only compare away from axis since inacurracies are introduced through
                # spline due to r ~ sqrt(s) behavior
                if bri.asym:
                    mean_dGds = np.mean(np.abs(bri.dGds()[5::, 0]))
                else:
                    # This is a vacuum case, so dGds is close to zero
                    mean_dGds = 1

                s_full = np.linspace(0, 1, bri.bx.ns_b+1)
                assert np.allclose(bri.dGds()[5::, 0]/mean_dGds, G_spline.derivative()(s_full[6:-1])/mean_dGds, atol=1e-2)
                mean_diotads = np.mean(np.abs(bri.diotads()[5::, 0]))
                assert np.allclose(bri.diotads()[5::, 0]/mean_diotads, iota_spline.derivative()(s_full[6:-1])/mean_diotads, atol=1e-2)
                assert np.allclose(bri.dmodBds()[5::, 0], modB00_spline.derivative()(s_full[6:-1]), rtol=1e-2)
                mean_dRds = np.mean(np.abs(bri.dRds()))
                assert np.allclose(bri.dRds()[5::, 0]/mean_dRds, R00_spline.derivative()(s_full[6:-1])/mean_dRds, atol=1e-2)

                """
                The next evaluation points test Z() and nu()
                """
                points = np.zeros((len(bri.s_half_ext[1:-1]), 3))
                points[:, 0] = bri.s_half_ext[1:-1]
                points[:, 1] = 0.
                points[:, 2] = np.pi/3
                bri.set_points(points)

                nu = bri.nu()
                iota = bri.iota()

                # zmns/zmnc on full grid
                # lmnc/lmns on half grid
                f = netcdf_file(filename_wout, mmap=False)

                zmns = f.variables['zmns'][()].T
                lmns = f.variables['lmns'][()].T
                zmns_full = (f.variables['zmns'][()].T)[:,1:]
                lmns_half = (f.variables['lmns'][()].T)[:,1::]
                zmns_half = 0.5*(zmns[:, 0:-1] + zmns[:, 1::])
                lmns_full = 0.5*(lmns[:, 1:-1] + lmns[:, 2::])
                if bri.asym:
                    lmnc = f.variables['lmnc'][()].T
                    lmnc_half = (f.variables['lmnc'][()].T)[:,1::]
                    lmnc_full = 0.5*(lmnc[:, 1:-1] + lmnc[:, 2::])
                    zmnc = f.variables['zmnc'][()].T
                    zmnc_full = (f.variables['zmnc'][()].T)[:,1:-1]
                    zmnc_half = 0.5*(zmnc[:, 0:-1] + zmnc[:, 1::])
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

                xm = f.variables['xm'][()]
                xn = f.variables['xn'][()]
                def theta_diff(theta, isurf):
                    lam = np.sum(lmns_half[:, isurf] * np.sin(xm*theta-xn*(np.pi/3-nu[isurf, 0])) + lmnc_half[:, isurf] * np.cos(xm*theta-xn*(np.pi/3-nu[isurf, 0])), axis=0)
                    return ((theta + lam) + iota[isurf, 0]*(nu[isurf, 0]))**2

                s_half_grid = bri.s_half_ext[1:-1]
                thetas_vmec = np.zeros((len(s_half_grid),))
                for isurf in range(len(s_half_grid)):
                    opt = minimize(theta_diff, 0, args=(isurf))
                    thetas_vmec[isurf] = opt.x

                # Compute Z at theta_b = 0, zeta_b = pi/2  and compare with vmec result
                Z0pi = np.sum(zmns_half * np.sin(xm[:, None]*thetas_vmec[None, :]-xn[:, None]*(np.pi/3-nu[None, :, 0]))
                              + zmnc_half * np.cos(xm[:, None]*thetas_vmec[None, :]-xn[:, None]*(np.pi/3-nu[None, :, 0])), axis=0)
                Z0pi_spline = InterpolatedUnivariateSpline(s_half_grid, Z0pi)

                mean_dZds = np.mean(np.abs(bri.dZds()[5::, 0]))

                assert np.allclose(bri.Z()[:, 0], Z0pi, atol=1e-2)
            
                if not asym:
                    assert np.allclose(bri.dZds()[5::, 0]/mean_dZds, Z0pi_spline.derivative()(s_half_grid[5::])/mean_dZds, atol=1e-2)
                # For asymmetric case, the dZds = 0
                else:
                    assert np.allclose(bri.dZds()[5::, 0], 0, atol=1e-2)
                    assert np.allclose(Z0pi_spline.derivative()(s_half_grid[5::]), 0, atol=1e-2)

                """
                The next evaluation points test the derivatives of modB, R, Z, and nu
                """
                points = np.zeros((len(s_half_grid), 3))
                points[:, 0] = s_half_grid
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

        order = 3
        bri = BoozerRadialInterpolant(filename_vac, order, rescale=True, comm=comm)

        nfp = bri.nfp
        n = 8
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

        assert np.allclose((K-Kh)/np.mean(np.abs(K)), 0, atol=1e-2)
        assert np.allclose((dKdtheta-dKdthetah)/np.mean(np.abs(dKdtheta)), 0, atol=1e-2)
        assert np.allclose((dKdzeta - dKdzetah)/np.mean(np.abs(dKdzeta)), 0, atol=5e-2)

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

        order = 3
        bri = BoozerRadialInterpolant(filename_vac, order, rescale=True, comm=comm)

        nfp = bri.nfp
        n = 16
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

        assert np.allclose((K-Kh)/np.mean(np.abs(K)), 0, atol=1e-2)
        assert np.allclose((dKdtheta-dKdthetah)/np.mean(np.abs(dKdtheta)), 0, atol=1e-2)
        assert np.allclose((dKdzeta - dKdzetah)/np.mean(np.abs(dKdzeta)), 0, atol=5e-2)

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

    def test_interpolatedboozerfield_convergence_rate(self):
        """
        Here we test the convergence rate of modB, R, Z, nu, K, G, I, and iota from
        InterpolatedBoozerField.
        """
        order = 3
        bri = BoozerRadialInterpolant(filename_mhd, order, mpol=10, ntor=10, comm=comm)

        # Perform interpolation from full grid
        points = np.zeros((bri.bx.ns_b-1, 3))
        s_full = np.linspace(0, 1, bri.bx.ns_b+1)
        points[:, 0] = s_full[1:-1]
        bri.set_points(points)

        nfp = bri.nfp
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


class TestingInverseFourier(unittest.TestCase):
    def test_inverse_fourier(self):
        thetas = np.linspace(0,2*np.pi, 131)
        zetas = np.linspace(0,2*np.pi, 131)
        [thetas, zetas] = np.meshgrid(thetas, zetas)
        thetas, zetas = thetas.flatten(), zetas.flatten()
        num_points = len(thetas)
        padded_thetas = align_and_pad(thetas)
        padded_zetas = align_and_pad(zetas)

        if (comm is not None):
            size = comm.size
            rank = comm.rank
        else:
            size = 1
            rank = 0
        
        num_modes = [1, 20, 48]
        for mpol in num_modes:
            for ntor in num_modes:
                nfp = np.random.randint(1, 8) if rank == 0 else None
                kmns = np.random.random(mpol * (ntor*2+1) - ntor)*2 - 1 if rank == 0 else None
                if comm is not None:
                    nfp = comm.bcast(nfp, root=0)
                    kmns = comm.bcast(kmns, root=0)
                                    
                xm = np.repeat(np.array(range(mpol)), ntor*2+1)[ntor:]
                xn = np.tile(np.array(range(-ntor*nfp, ntor*nfp+1, nfp)), mpol)[ntor:]
                assert(len(kmns)==len(xn))
                assert(len(kmns)==len(xm))

                even_K = sum(kmns[:, np.newaxis] * np.cos(np.outer(xm, thetas) - np.outer(xn, zetas)))
                odd_K = sum(kmns[:, np.newaxis] * np.sin(np.outer(xm, thetas) - np.outer(xn, zetas)))


                mn_idxs = np.array([i * len(xm) // size for i in range(size + 1)])
                first_mn, last_mn = mn_idxs[rank], mn_idxs[rank + 1]
                padded_kmns = align_and_pad(np.tile(kmns[first_mn:last_mn, np.newaxis], (1,num_points)))
                padded_even_output = allocate_aligned_and_padded_array(thetas.shape)
                padded_odd_output = allocate_aligned_and_padded_array(thetas.shape)

                inverse_fourier_transform_even(padded_even_output, padded_kmns, xm[first_mn:last_mn], xn[first_mn:last_mn], padded_thetas, padded_zetas, ntor, nfp)
                inverse_fourier_transform_odd(padded_odd_output, padded_kmns, xm[first_mn:last_mn], xn[first_mn:last_mn], padded_thetas, padded_zetas, ntor, nfp)

                if (comm is not None):
                    comm.Allreduce(MPI.IN_PLACE, [padded_even_output[:num_points], MPI.DOUBLE], op=MPI.SUM)
                    comm.Allreduce(MPI.IN_PLACE, [padded_odd_output[:num_points], MPI.DOUBLE], op=MPI.SUM)

                assert np.allclose(even_K, padded_even_output[:num_points], rtol=1e-12, atol=1e-11)
                assert np.allclose(odd_K, padded_odd_output[:num_points], rtol=1e-12, atol=1e-11)

                # a single point
                padded_kmns = align_and_pad(kmns[first_mn:last_mn])
                padded_xm = align_and_pad(xm[first_mn:last_mn])
                padded_xn = align_and_pad(xn[first_mn:last_mn])
                for i in range(num_points):
                    theta = np.array([thetas[i]])
                    zeta = np.array([zetas[i]])
                    even_output = np.zeros(1)
                    odd_output = np.zeros(1)

                    inverse_fourier_transform_even(even_output, padded_kmns, padded_xm, padded_xn, theta, zeta, ntor, nfp)
                    inverse_fourier_transform_odd(odd_output, padded_kmns, padded_xm, padded_xn, theta, zeta, ntor, nfp)

                    if (comm is not None):
                        comm.Allreduce(MPI.IN_PLACE, [even_output, MPI.DOUBLE], op=MPI.SUM)
                        comm.Allreduce(MPI.IN_PLACE, [odd_output, MPI.DOUBLE], op=MPI.SUM)

                    assert np.allclose(even_K[i], even_output[0], rtol=1e-12, atol=1e-11)
                    assert np.allclose(odd_K[i], odd_output[0], rtol=1e-12, atol=1e-11)

if __name__ == "__main__":
    unittest.main()
