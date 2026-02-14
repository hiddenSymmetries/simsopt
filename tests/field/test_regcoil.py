import json
import unittest
import warnings
from simsopt.geo import SurfaceRZFourier
from matplotlib import pyplot as plt
import numpy as np
from simsoptpp import WindingSurfaceBn_REGCOIL
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.objectives import SquaredFlux
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.util import in_github_actions
from simsopt._core.json import SIMSON, GSONEncoder, GSONDecoder
from scipy.special import ellipk, ellipe
from pathlib import Path
from scipy.io import netcdf_file
np.random.seed(100)

TEST_DIR = Path(__file__).parent / ".." / "test_files"


class Testing(unittest.TestCase):

    def test_windingsurface_exact(self):
        """
            Make an infinitesimally thin current loop in the Z = 0 plane
            Following approximate analytic solution in Jackson 5.37 for the
            vector potential A. From this, we can also check calculations for
            B, dA/dX and dB/dX using the WindingSurface class.
        """
        nphi = 128
        ntheta = 16

        # Make winding surface with major radius = 1, no minor radius
        winding_surface = SurfaceRZFourier()
        winding_surface = winding_surface.from_nphi_ntheta(nphi=nphi, ntheta=ntheta)
        for i in range(winding_surface.mpol + 1):
            for j in range(-winding_surface.ntor, winding_surface.ntor + 1):
                winding_surface.set_rc(i, j, 0.0)
                winding_surface.set_zs(i, j, 0.0)
        winding_surface.set_rc(0, 0, 1.0)
        eps = 1e-12
        winding_surface.set_rc(1, 0, eps)  # current loop must have finite width for simsopt
        winding_surface.set_zs(1, 0, eps)  # current loop must have finite width for simsopt

        # Make CurrentPotential class from this winding surface with 1 amp toroidal current
        current_potential = CurrentPotentialFourier(winding_surface, net_poloidal_current_amperes=0, net_toroidal_current_amperes=-1)

        # compute the Bfield from this current loop at some points
        Bfield = WindingSurfaceField(current_potential)
        N = 2000
        _phi = winding_surface.quadpoints_phi

        # Check that the full expression is correct
        points = (np.random.rand(N, 3) - 0.5) * 10
        Bfield.set_points(np.ascontiguousarray(points))
        B_predict = Bfield.B()
        dB_predict = Bfield.dB_by_dX()
        A_predict = Bfield.A()
        _dA_predict = Bfield.dA_by_dX()

        # calculate the Bfield analytically in spherical coordinates
        mu_fac = 1e-7

        # See Jackson 5.37 for the vector potential in terms of the elliptic integrals
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        k = np.sqrt(4 * r * np.sin(theta) / (1 + r ** 2 + 2 * r * np.sin(theta)))

        # Note scipy is very annoying... scipy function ellipk(k^2)
        # is equivalent to what Jackson calls ellipk(k) so call it with k^2
        Aphi = mu_fac * (4 / np.sqrt(1 + r ** 2 + 2 * r * np.sin(theta))) * ((2 - k ** 2) * ellipk(k ** 2) - 2 * ellipe(k ** 2)) / k ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        phi_points = np.arctan2(points[:, 1], points[:, 0])
        theta_points = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic_elliptic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        np.testing.assert_allclose(A_predict, A_analytic_elliptic, rtol=1e-2, atol=1e-12, err_msg="A_predict != A_analytic (near-field elliptic)")

        # now check the Bfield and shape derivatives using the analytic
        # expressions that can be derived by hand or found here
        # https://ntrs.nasa.gov/citations/20010038494
        C = 4 * mu_fac
        alpha2 = 1 + r ** 2 - 2 * r * np.sin(theta)
        beta2 = 1 + r ** 2 + 2 * r * np.sin(theta)
        k2 = 1 - alpha2 / beta2
        Br = C * np.cos(theta) * ellipe(k2) / (alpha2 * np.sqrt(beta2))
        Btheta = C * ((r ** 2 + np.cos(2 * theta)) * ellipe(k2) - alpha2 * ellipk(k2)) / (2 * alpha2 * np.sqrt(beta2) * np.sin(theta))

        # convert B_analytic to Cartesian
        Bx = np.zeros(len(Br))
        By = np.zeros(len(Br))
        Bz = np.zeros(len(Br))
        for i in range(N):
            Bx[i] = np.sin(theta_points[i]) * np.cos(phi_points[i]) * Br[i] + np.cos(theta_points[i]) * np.cos(phi_points[i]) * Btheta[i]
            By[i] = np.sin(theta_points[i]) * np.sin(phi_points[i]) * Br[i] + np.cos(theta_points[i]) * np.sin(phi_points[i]) * Btheta[i]
            Bz[i] = np.cos(theta_points[i]) * Br[i] - np.sin(theta_points[i]) * Btheta[i]
        B_analytic = np.array([Bx, By, Bz]).T

        np.testing.assert_allclose(B_predict, B_analytic, rtol=1e-2, atol=1e-12, err_msg="B_predict != B_analytic (near-field)")

        x = points[:, 0]
        y = points[:, 1]
        gamma = x ** 2 - y ** 2
        z = points[:, 2]
        rho = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        Bx_dx = C * z * (((- gamma * (3 * z ** 2 + 1) + rho ** 2 * (8 * x ** 2 - y ** 2)) - (rho ** 4 * (5 * x ** 2 + y ** 2) - 2 * rho ** 2 * z ** 2 * (2 * x ** 2 + y ** 2) + 3 * z ** 4 * gamma) - r ** 4 * (2 * x ** 4 + gamma * (y ** 2 + z ** 2))) * ellipe(k2) + (gamma * (1 + 2 * z ** 2) - rho ** 2 * (3 * x ** 2 - 2 * y ** 2) + r ** 2 * (2 * x ** 4 + gamma * (y ** 2 + z ** 2))) * alpha2 * ellipk(k2)) / (2 * alpha2 ** 2 * beta2 ** (3 / 2) * rho ** 4)

        Bx_dy = C * x * y * z * ((3 * (3 * rho ** 2 - 2 * z ** 2) - r ** 4 * (2 * r ** 2 + rho ** 2) - 2 - 2 * (2 * rho ** 4 - rho ** 2 * z ** 2 + 3 * z ** 4)) * ellipe(k2) + (r ** 2 * (2 * r ** 2 + rho ** 2) - (5 * rho ** 2 - 4 * z ** 2) + 2) * alpha2 * ellipk(k2)) / (2 * alpha2 ** 2 * beta2 ** (3 / 2) * rho ** 4)
        Bx_dz = C * x * (((rho ** 2 - 1) ** 2 * (rho ** 2 + 1) + 2 * z ** 2 * (1 - 6 * rho ** 2 + rho ** 4) + z ** 4 * (1 + rho ** 2)) * ellipe(k2) - ((rho ** 2 - 1) ** 2 + z ** 2 * (rho ** 2 + 1)) * alpha2 * ellipk(k2)) / (2 * alpha2 ** 2 * beta2 ** (3 / 2) * rho ** 2)
        By_dx = Bx_dy
        By_dy = C * z * (((gamma * (3 * z ** 2 + 1) + rho ** 2 * (8 * y ** 2 - x ** 2)) - (rho ** 4 * (5 * y ** 2 + x ** 2) - 2 * rho ** 2 * z ** 2 * (2 * y ** 2 + x ** 2) - 3 * z ** 4 * gamma) - r ** 4 * (2 * y ** 4 - gamma * (x ** 2 + z ** 2))) * ellipe(k2) + ((- gamma * (1 + 2 * z ** 2) - rho ** 2 * (3 * y ** 2 - 2 * x ** 2)) + r ** 2 * (2 * y ** 4 - gamma * (x ** 2 + z ** 2))) * alpha2 * ellipk(k2)) / (2 * alpha2 ** 2 * beta2 ** (3 / 2) * rho ** 4)
        By_dz = y / x * Bx_dz
        Bz_dx = Bx_dz
        Bz_dy = By_dz
        Bz_dz = C * z * ((6 * (rho ** 2 - z ** 2) - 7 + r ** 4) * ellipe(k2) + alpha2 * (1 - r ** 2) * ellipk(k2)) / (2 * alpha2 ** 2 * beta2 ** (3 / 2))
        dB_analytic = np.transpose(np.array([[Bx_dx, Bx_dy, Bx_dz],
                                             [By_dx, By_dy, By_dz],
                                             [Bz_dx, Bz_dy, Bz_dz]]), [2, 0, 1])

        np.testing.assert_allclose(dB_predict, dB_analytic, rtol=1e-2, atol=1e-12, err_msg="dB_predict != dB_analytic")

        # Now check that the far-field looks like a dipole
        points = (np.random.rand(N, 3) + 1) * 1000
        gamma = winding_surface.gamma().reshape((-1, 3))

        Bfield.set_points(np.ascontiguousarray(points))
        B_predict = Bfield.B()
        A_predict = Bfield.A()

        # calculate the Bfield analytically in spherical coordinates
        mu_fac = 1e-7

        # See Jackson 5.37 for the vector potential in terms of the elliptic integrals
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        k = np.sqrt(4 * r * np.sin(theta) / (1 + r ** 2 + 2 * r * np.sin(theta)))
        Aphi = mu_fac * (4 / np.sqrt(1 + r ** 2 + 2 * r * np.sin(theta))) * ((2 - k ** 2) * ellipk(k ** 2) - 2 * ellipe(k ** 2)) / k ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        phi_points = np.arctan2(points[:, 1], points[:, 0])
        theta_points = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic_elliptic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        np.testing.assert_allclose(A_predict, A_analytic_elliptic, rtol=1e-2, atol=1e-12, err_msg="A_predict != A_analytic (far-field elliptic, pass 1)")

        # Now check that the far-field looks like a dipole
        points = (np.random.rand(N, 3) + 1) * 1000
        gamma = winding_surface.gamma().reshape((-1, 3))

        Bfield.set_points(np.ascontiguousarray(points))
        B_predict = Bfield.B()
        A_predict = Bfield.A()

        # calculate the Bfield analytically in spherical coordinates
        mu_fac = 1e-7

        # See Jackson 5.37 for the vector potential in terms of the elliptic integrals
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        k = np.sqrt(4 * r * np.sin(theta) / (1 + r ** 2 + 2 * r * np.sin(theta)))

        # Note scipy is very annoying... scipy function ellipk(k^2)
        # is equivalent to what Jackson calls ellipk(k) so call it with k^2
        Aphi = mu_fac * (4 / np.sqrt(1 + r ** 2 + 2 * r * np.sin(theta))) * ((2 - k ** 2) * ellipk(k ** 2) - 2 * ellipe(k ** 2)) / k ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        phi_points = np.arctan2(points[:, 1], points[:, 0])
        theta_points = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic_elliptic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        np.testing.assert_allclose(A_predict, A_analytic_elliptic, rtol=1e-3, atol=1e-12, err_msg="A_predict != A_analytic (far-field elliptic, pass 2)")

        # double check with vector potential of a dipole
        Aphi = np.pi * mu_fac * np.sin(theta) / r ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        np.testing.assert_allclose(A_predict, A_analytic, rtol=1e-3, atol=1e-12, err_msg="A_predict != A_analytic (far-field dipole)")

    def test_regcoil_K_solve(self):
        """
            This function tests the Tikhonov solve with lambda -> infinity
            and extensively checks the SIMSOPT grid, currents, solution, etc.
            against the REGCOIL variables.
        """
        for filename in ['regcoil_out.w7x_infty.nc', 'regcoil_out.li383_infty.nc']:
            print(filename)
            filename = TEST_DIR / filename
            cpst = CurrentPotentialSolve.from_netcdf(filename)
            # initialize a solver object for the cp CurrentPotential
            s_plasma = cpst.plasma_surface
            s_coil = cpst.winding_surface
            # Check B and K RHS's -> these are independent of lambda
            b_rhs_simsopt, _ = cpst.B_matrix_and_rhs()
            k_rhs = cpst.K_rhs()

            for ilambda in range(1, 3):
                # Load in big list of variables from REGCOIL to check agree with SIMSOPT
                f = netcdf_file(filename, 'r', mmap=False)
                Bnormal_regcoil_total = f.variables['Bnormal_total'][()][ilambda, :, :]
                Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
                Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
                r_plasma = f.variables['r_plasma'][()]
                r_coil = f.variables['r_coil'][()]
                nzeta_plasma = f.variables['nzeta_plasma'][()]
                nzeta_coil = f.variables['nzeta_coil'][()]
                _ntheta_coil = f.variables['ntheta_coil'][()]
                _nfp = f.variables['nfp'][()]
                _ntheta_plasma = f.variables['ntheta_plasma'][()]
                K2_regcoil = f.variables['K2'][()][ilambda, :, :]
                lambda_regcoil = f.variables['lambda'][()][ilambda]
                b_rhs_regcoil = f.variables['RHS_B'][()]
                k_rhs_regcoil = f.variables['RHS_regularization'][()]
                single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][ilambda, :]
                _xm_potential = f.variables['xm_potential'][()]
                _xn_potential = f.variables['xn_potential'][()]
                _theta_coil = f.variables['theta_coil'][()]
                _zeta_coil = f.variables['zeta_coil'][()]
                f_B_regcoil = 0.5 * f.variables['chi2_B'][()][ilambda]
                f_K_regcoil = 0.5 * f.variables['chi2_K'][()][ilambda]
                norm_normal_plasma = f.variables['norm_normal_plasma'][()]
                current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()][ilambda, :, :]
                f.close()
                Bnormal_single_valued = Bnormal_regcoil_total - Bnormal_from_plasma_current - Bnormal_from_net_coil_currents
                print('ilambda index = ', ilambda, lambda_regcoil)

                np.testing.assert_allclose(b_rhs_regcoil, b_rhs_simsopt, rtol=1e-3, atol=1e-12, err_msg="b_rhs_regcoil != b_rhs_simsopt")
                np.testing.assert_allclose(k_rhs, k_rhs_regcoil, rtol=1e-3, atol=1e-12, err_msg="k_rhs != k_rhs_regcoil")

                # Compare Bnormal from plasma
                np.testing.assert_allclose(cpst.Bnormal_plasma, Bnormal_from_plasma_current.flatten(), rtol=1e-3, atol=1e-12, err_msg="Bnormal_plasma mismatch")

                # Compare optimized dofs
                cp = cpst.current_potential

                # when lambda -> infinity, the L1 and L2 regularized problems should agree
                optimized_phi_mn_lasso, f_B_lasso, f_K_lasso, _, _ = cpst.solve_lasso(lam=lambda_regcoil)
                optimized_phi_mn_lasso_ista, f_B_lasso_ista, f_K_lasso_ista, _, _ = cpst.solve_lasso(lam=lambda_regcoil, acceleration=False, max_iter=5000)
                optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=lambda_regcoil)
                np.testing.assert_allclose(single_valued_current_potential_mn, optimized_phi_mn, rtol=1e-3, atol=1e-12, err_msg="single_valued_current_potential_mn != optimized_phi_mn (Tikhonov)")
                print(optimized_phi_mn_lasso, optimized_phi_mn)
                print(f_B, f_B_lasso, f_B_regcoil)
                np.testing.assert_allclose(f_B, f_B_regcoil, rtol=1e-3, atol=1e-12, err_msg="f_B (Tikhonov) != f_B_regcoil")
                # assert np.isclose(f_K_lasso, f_K)
                np.testing.assert_allclose(optimized_phi_mn_lasso, optimized_phi_mn, rtol=1e-3, atol=1e-12, err_msg="optimized_phi_mn_lasso != optimized_phi_mn")
                np.testing.assert_allclose(optimized_phi_mn_lasso_ista, optimized_phi_mn, rtol=1e-3, atol=1e-12, err_msg="optimized_phi_mn_lasso (ISTA) != optimized_phi_mn")

                # Compare plasma surface position
                np.testing.assert_allclose(r_plasma[0:nzeta_plasma, :, :], s_plasma.gamma(), rtol=1e-3, atol=1e-12, err_msg="plasma surface position mismatch")

                # Compare plasma surface normal
                np.testing.assert_allclose(
                    norm_normal_plasma[0:nzeta_plasma, :],
                    np.linalg.norm(s_plasma.normal(), axis=2) / (2 * np.pi * 2 * np.pi),
                    rtol=1e-3, atol=1e-12,
                    err_msg="plasma surface normal mismatch"
                )

                # Compare winding surface position
                s_coil = cp.winding_surface
                np.testing.assert_allclose(r_coil, s_coil.gamma(), rtol=1e-3, atol=1e-12, err_msg="winding surface position mismatch")

                # Compare field from net coil currents
                cp_GI = CurrentPotentialFourier.from_netcdf(filename)
                Bfield = WindingSurfaceField(cp_GI)
                points = s_plasma.gamma().reshape(-1, 3)
                Bfield.set_points(points)
                B = Bfield.B()
                _norm_normal = np.linalg.norm(s_plasma.normal(), axis=2) / (2 * np.pi * 2 * np.pi)
                normal = s_plasma.unitnormal().reshape(-1, 3)
                B_GI_winding_surface = np.sum(B * normal, axis=1)
                np.testing.assert_allclose(B_GI_winding_surface, np.ravel(Bnormal_from_net_coil_currents), rtol=1e-3, atol=1e-12, err_msg="B_GI from WindingSurfaceField != Bnormal_from_net_coil_currents")
                np.testing.assert_allclose(cpst.B_GI, np.ravel(Bnormal_from_net_coil_currents), rtol=1e-3, atol=1e-12, err_msg="cpst.B_GI != Bnormal_from_net_coil_currents")

                # Compare single-valued current potential
                # Initialization not from netcdf
                cp_no_GI = CurrentPotentialFourier(
                    cp_GI.winding_surface,
                    net_poloidal_current_amperes=0.0,
                    net_toroidal_current_amperes=0.0,
                    mpol=cp_GI.mpol,  # critical line here
                    ntor=cp_GI.ntor,  # critical line here
                )
                cp_no_GI.set_dofs(optimized_phi_mn)
                np.testing.assert_allclose(cp_no_GI.Phi()[0:nzeta_coil, :], current_potential_thetazeta, rtol=1e-3, atol=1e-12, err_msg="single-valued current potential Phi mismatch")

                # Check that f_B from SquaredFlux and f_B from least-squares agree
                Bfield_opt = WindingSurfaceField(cp)
                Bfield_opt.set_points(s_plasma.gamma().reshape(-1, 3))
                B = Bfield_opt.B()
                normal = s_plasma.unitnormal().reshape(-1, 3)
                _Bn_opt = np.sum(B * normal, axis=1)
                _nfp = cpst.plasma_surface.nfp
                nphi = len(cpst.plasma_surface.quadpoints_phi)
                ntheta = len(cpst.plasma_surface.quadpoints_theta)
                f_B_sq = SquaredFlux(
                    s_plasma,
                    Bfield_opt,
                    -np.ascontiguousarray(cpst.Bnormal_plasma.reshape(nphi, ntheta))
                ).J()

                # These will not exactly agree
                # assert np.isclose(f_B, f_B_sq, rtol=1e-2)

                # These will not exactly agree because using different integral discretizations
                # assert np.isclose(f_B, f_B_regcoil, rtol=1e-2)

                # These should agree much better
                np.testing.assert_allclose(f_B_regcoil, f_B_sq, rtol=1e-3, atol=1e-12, err_msg="f_B_regcoil != f_B from SquaredFlux")

                # Compare current density
                cp.set_dofs(optimized_phi_mn)
                K = cp.K()
                K2 = np.sum(K ** 2, axis=2)
                K2_average = np.mean(K2, axis=(0, 1))
                np.testing.assert_allclose(K2[0:nzeta_coil, :] / K2_average, K2_regcoil / K2_average, rtol=1e-3, atol=1e-12, err_msg="K2 mismatch")

                # Compare values of f_K computed in three different ways
                normal = s_coil.normal().reshape(-1, 3)
                normN = np.linalg.norm(normal, axis=-1)
                f_K_direct = 0.5 * np.sum(np.ravel(K2) * normN) / (normal.shape[0])
                # print(f_K_regcoil, f_K_direct, f_K)
                np.testing.assert_allclose(f_K_regcoil, f_K_direct, rtol=1e-3, atol=1e-12, err_msg="f_K_regcoil != f_K_direct")
                np.testing.assert_allclose(f_K_regcoil, f_K, rtol=1e-3, atol=1e-12, err_msg="f_K_regcoil != f_K (from solve)")

                # Check normal field
                Bfield_opt = WindingSurfaceField(cp)
                Bfield_opt.set_points(s_plasma.gamma().reshape(-1, 3))
                B_opt = Bfield_opt.B()
                normal = s_plasma.unitnormal().reshape(-1, 3)
                Bnormal = np.sum(B_opt*normal, axis=1).reshape(np.shape(s_plasma.gamma()[:, :, 0]))
                Bnormal_regcoil = Bnormal_regcoil_total - Bnormal_from_plasma_current
                np.testing.assert_allclose(np.sum(Bnormal), 0, rtol=1e-3, atol=1e-12, err_msg="sum(Bnormal) != 0")
                np.testing.assert_allclose(np.sum(Bnormal_regcoil), 0, rtol=1e-3, atol=1e-12, err_msg="sum(Bnormal_regcoil) != 0")

                # B computed from inductance, i.e. equation A8 in REGCOIL paper
                normal_plasma = s_plasma.normal().reshape(-1, 3)
                r_plasma = s_plasma.gamma().reshape(-1, 3)
                normal_coil = s_coil.normal().reshape(-1, 3)
                r_coil = s_coil.gamma().reshape(-1, 3)
                rdiff = r_plasma[None, :, :] - r_coil[:, None, :]
                rdiff_norm = np.linalg.norm(rdiff, axis=2)
                n_dot_nprime = np.sum(normal_plasma[None, :, :] * normal_coil[:, None, :], axis=2)
                rdiff_dot_n = np.sum(rdiff * normal_plasma[None, :, :], axis=2)
                rdiff_dot_nprime = np.sum(rdiff * normal_coil[:, None, :], axis=2)
                inductance_simsopt = (n_dot_nprime / rdiff_norm ** 3 - 3 * rdiff_dot_n * rdiff_dot_nprime / rdiff_norm ** 5) * 1e-7
                dtheta_coil = s_coil.quadpoints_theta[1]
                dzeta_coil = s_coil.quadpoints_phi[1]
                Bnormal_g = (np.sum(inductance_simsopt * cp.Phi().reshape(-1)[:, None], axis=0) * dtheta_coil * dzeta_coil / np.linalg.norm(normal_plasma, axis=1)).reshape(np.shape(s_plasma.gamma()[:, :, 0]))

                # REGCOIL calculation in c++
                points = s_plasma.gamma().reshape(-1, 3)
                normal = s_plasma.normal().reshape(-1, 3)
                ws_points = s_coil.gamma().reshape(-1, 3)
                ws_normal = s_coil.normal().reshape(-1, 3)
                Bnormal_REGCOIL = WindingSurfaceBn_REGCOIL(points, ws_points, ws_normal, cp.Phi(), normal) * dtheta_coil * dzeta_coil
                np.testing.assert_allclose(Bnormal_REGCOIL, np.ravel(Bnormal_single_valued), rtol=1e-3, atol=1e-12, err_msg="Bnormal_REGCOIL (C++) != Bnormal_single_valued")
                normN = np.linalg.norm(normal, axis=-1)
                res = (np.ravel(Bnormal_regcoil_total) ** 2) @ normN
                f_B_manual = 0.5 * res / (nphi * ntheta)
                np.testing.assert_allclose(f_B_regcoil, f_B_manual, rtol=1e-3, atol=1e-12, err_msg="f_B_regcoil != f_B_manual")

                Bnormal_g += B_GI_winding_surface.reshape(np.shape(s_plasma.gamma()[:, :, 0]))
                Bnormal_REGCOIL += B_GI_winding_surface

                # Check that Bnormal calculations using the REGCOIL discretization all agree
                np.testing.assert_allclose(np.ravel(Bnormal_g), Bnormal_REGCOIL, rtol=1e-3, atol=1e-12, err_msg="Bnormal_g != Bnormal_REGCOIL")
                np.testing.assert_allclose(np.ravel(Bnormal_g), np.ravel(Bnormal_regcoil), rtol=1e-3, atol=1e-12, err_msg="Bnormal_g != Bnormal_regcoil")

                # will be some substantial disagreement here because of the different discretizations,
                # although it should improve with higher resolution
                # assert np.allclose(Bnormal / np.mean(np.abs(Bnormal_regcoil)), Bnormal_regcoil / np.mean(np.abs(Bnormal_regcoil)), atol=1e-2)

    def test_winding_surface_regcoil(self):
        """
            Extensive tests are done for multiple stellarators (stellarator symmetric and
            stellarator asymmetric) to verify that REGCOIL, as implemented in SIMSOPT,
            agrees with the REGCOIL test file solutions.
        """
        for filename in ['regcoil_out.near_axis_asym.nc', 'regcoil_out.near_axis.nc', 'regcoil_out.w7x.nc', 'regcoil_out.li383.nc']:
            print(filename)

            # Load big list of variables from REGCOIL to check against SIMSOPT implementation
            filename = TEST_DIR / filename
            f = netcdf_file(filename, 'r', mmap=False)
            Bnormal_regcoil_total = f.variables['Bnormal_total'][()]
            Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
            Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
            r_plasma = f.variables['r_plasma'][()]
            r_coil = f.variables['r_coil'][()]
            nzeta_plasma = f.variables['nzeta_plasma'][()]
            nzeta_coil = f.variables['nzeta_coil'][()]
            K2_regcoil = f.variables['K2'][()]
            lambda_regcoil = f.variables['lambda'][()]
            f_B_regcoil = 0.5 * f.variables['chi2_B'][()]
            f_K_regcoil = 0.5 * f.variables['chi2_K'][()]
            b_rhs_regcoil = f.variables['RHS_B'][()]
            k_rhs_regcoil = f.variables['RHS_regularization'][()]
            _xm_potential = f.variables['xm_potential'][()]
            _xn_potential = f.variables['xn_potential'][()]
            _nfp = f.variables['nfp'][()]
            single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()]
            norm_normal_plasma = f.variables['norm_normal_plasma'][()]
            current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()]
            norm_normal_coil = f.variables['norm_normal_coil'][()]
            f.close()

            # Compare K and B RHS's -> these are independent of lambda
            cpst = CurrentPotentialSolve.from_netcdf(filename)

            b_rhs_simsopt, _ = cpst.B_matrix_and_rhs()

            np.testing.assert_allclose(b_rhs_regcoil, b_rhs_simsopt, rtol=1e-3, atol=1e-12, err_msg=f"{filename}: b_rhs mismatch")

            k_rhs = cpst.K_rhs()
            np.testing.assert_allclose(k_rhs, k_rhs_regcoil, rtol=1e-3, atol=1e-12, err_msg=f"{filename}: k_rhs mismatch")

            # Compare plasma current
            np.testing.assert_allclose(cpst.Bnormal_plasma, Bnormal_from_plasma_current.flatten(), rtol=1e-3, atol=1e-12, err_msg=f"{filename}: Bnormal_plasma mismatch")

            # Compare Bnormal from net coil currents
            np.testing.assert_allclose(cpst.B_GI, np.ravel(Bnormal_from_net_coil_currents), rtol=1e-3, atol=1e-12, err_msg=f"{filename}: B_GI mismatch")

            cp = cpst.current_potential
            s_plasma = cpst.plasma_surface

            s_plasma_full = SurfaceRZFourier(
                nfp=s_plasma.nfp,
                mpol=s_plasma.mpol,
                ntor=s_plasma.ntor,
                stellsym=s_plasma.stellsym
            )
            s_plasma_full = s_plasma_full.from_nphi_ntheta(
                nfp=s_plasma.nfp, ntheta=len(s_plasma.quadpoints_theta),
                nphi=len(s_plasma.quadpoints_phi)*s_plasma.nfp,
                mpol=s_plasma.mpol, ntor=s_plasma.ntor,
                stellsym=s_plasma.stellsym, range="full torus"
            )
            s_plasma_full.set_dofs(s_plasma.get_dofs())
            # Compare plasma surface position
            np.testing.assert_allclose(r_plasma, s_plasma_full.gamma(), rtol=1e-3, atol=1e-12, err_msg=f"{filename}: plasma surface position mismatch")

            # Compare plasma surface normal
            norm_normal_plasma_simsopt = np.linalg.norm(s_plasma_full.normal(), axis=-1)
            np.testing.assert_allclose(norm_normal_plasma*2*np.pi*2*np.pi, norm_normal_plasma_simsopt[0:nzeta_plasma, :], rtol=1e-3, atol=1e-12, err_msg=f"{filename}: plasma surface normal mismatch")

            # Compare winding surface position
            s_coil = cp.winding_surface
            np.testing.assert_allclose(r_coil, s_coil.gamma(), rtol=1e-3, atol=1e-12, err_msg=f"{filename}: winding surface position mismatch")

            # Compare winding surface normal
            norm_normal_coil_simsopt = np.linalg.norm(s_coil.normal(), axis=-1)
            np.testing.assert_allclose(norm_normal_coil*2*np.pi*2*np.pi, norm_normal_coil_simsopt[0:nzeta_coil, :], rtol=1e-3, atol=1e-12, err_msg=f"{filename}: winding surface normal mismatch")

            # Compare two different ways of computing K()
            K = cp.K().reshape(-1, 3)
            winding_surface = cp.winding_surface
            normal_vec = winding_surface.normal().reshape(-1, 3)
            dzeta_coil = (winding_surface.quadpoints_phi[1] - winding_surface.quadpoints_phi[0])
            dtheta_coil = (winding_surface.quadpoints_theta[1] - winding_surface.quadpoints_theta[0])
            normn = np.sqrt(np.sum(normal_vec**2, axis=-1))  # |N|
            K_2 = -(cpst.fj @ cp.get_dofs() - cpst.d) / \
                (np.sqrt(dzeta_coil * dtheta_coil) * normn[:, None])
            np.testing.assert_allclose(K, K_2, rtol=1e-3, atol=1e-12, err_msg=f"{filename}: K from cp.K() != K from matrix computation")

            # Compare field from net coil currents
            cp_GI = CurrentPotentialFourier.from_netcdf(filename)
            Bfield = WindingSurfaceField(cp_GI)
            points = s_plasma.gamma().reshape(-1, 3)
            Bfield.set_points(points)
            B = Bfield.B()
            normal = s_plasma.unitnormal().reshape(-1, 3)
            B_GI_winding_surface = np.sum(B * normal, axis=1)
            np.testing.assert_allclose(B_GI_winding_surface, np.ravel(Bnormal_from_net_coil_currents), rtol=1e-3, atol=1e-12, err_msg=f"{filename}: B_GI from WindingSurfaceField mismatch")
            # Make sure single-valued part of current potential is working
            cp_no_GI = CurrentPotentialFourier.from_netcdf(filename)
            cp_no_GI.set_net_toroidal_current_amperes(0)
            cp_no_GI.set_net_poloidal_current_amperes(0)

            # Now loop over all the regularization values in the REGCOIL solution
            for i, lambda_reg in enumerate(lambda_regcoil):

                # Set current potential Fourier harmonics from regcoil file
                cp.set_current_potential_from_regcoil(filename, i)

                # Compare current potential Fourier harmonics
                np.testing.assert_allclose(cp.get_dofs(), single_valued_current_potential_mn[i, :], rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: current potential dofs mismatch")

                # Compare current density
                K = cp.K()
                K2 = np.sum(K ** 2, axis=2)
                K2_average = np.mean(K2, axis=(0, 1))
                np.testing.assert_allclose(K2[0:nzeta_plasma, :] / K2_average, K2_regcoil[i, :, :] / K2_average, rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: K2 mismatch")

                f_B_REGCOIL = f_B_regcoil[i]
                f_K_REGCOIL = f_K_regcoil[i]

                cp_no_GI.set_current_potential_from_regcoil(filename, i)

                # Compare single-valued current potential
                np.testing.assert_allclose(cp_no_GI.Phi()[0:nzeta_plasma, :], current_potential_thetazeta[i, :, :], rtol=1e-3, atol=1e-10, err_msg=f"{filename} lambda[{i}]: single-valued Phi mismatch")

                f_K_direct = 0.5 * np.sum(K2 * norm_normal_coil_simsopt) / (norm_normal_coil_simsopt.shape[0]*norm_normal_coil_simsopt.shape[1])
                np.testing.assert_allclose(f_K_direct/np.abs(f_K_REGCOIL), f_K_REGCOIL/np.abs(f_K_REGCOIL), rtol=1e-3, atol=1e-10, err_msg=f"{filename} lambda[{i}]: f_K_direct/|f_K_REGCOIL| != f_K_REGCOIL/|f_K_REGCOIL|, got {f_K_direct} vs {f_K_REGCOIL}")

                normal = s_plasma.unitnormal().reshape(-1, 3)
                norm_normal_plasma_simsopt = np.linalg.norm(s_plasma.normal(), axis=-1)
                Bnormal_regcoil = Bnormal_regcoil_total[i, :, :] - Bnormal_from_plasma_current

                # REGCOIL calculation in c++
                points = s_plasma.gamma().reshape(-1, 3)
                normal = s_plasma.normal().reshape(-1, 3)
                ws_points = s_coil.gamma().reshape(-1, 3)
                ws_normal = s_coil.normal().reshape(-1, 3)
                dtheta_coil = s_coil.quadpoints_theta[1]
                dzeta_coil = s_coil.quadpoints_phi[1]
                Bnormal = WindingSurfaceBn_REGCOIL(points, ws_points, ws_normal, cp.Phi(), normal) * dtheta_coil * dzeta_coil
                Bnormal += cpst.B_GI
                Bnormal = Bnormal.reshape(Bnormal_regcoil.shape)

                np.testing.assert_allclose(Bnormal, Bnormal_regcoil, rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: Bnormal (C++) mismatch")

                # check Bnormal and Bnormal_regcoil integrate over the surface to zero
                # This is only true in the stellarator symmetric case!
                if s_plasma.stellsym:
                    np.testing.assert_allclose(np.sum(Bnormal*norm_normal_plasma_simsopt), 0, rtol=1e-3, atol=1e-10, err_msg=f"{filename} lambda[{i}]: sum(Bnormal*norm) != 0")
                    np.testing.assert_allclose(np.sum(Bnormal_regcoil*norm_normal_plasma_simsopt[0:nzeta_plasma, :]), 0, rtol=1e-3, atol=1e-10, err_msg=f"{filename} lambda[{i}]: sum(Bnormal_regcoil*norm) != 0")

                # Check that L1 optimization agrees if lambda = 0
                # With lambda=0, FISTA converges slowly (ill-conditioned); need many iterations
                if lambda_reg == 0.0:
                    optimized_phi_mn_lasso, f_B_lasso, _, _, _ = cpst.solve_lasso(lam=lambda_reg, max_iter=10000, acceleration=True)
                    optimized_phi_mn_lasso_ista, f_B_lasso_ista, _, _, _ = cpst.solve_lasso(lam=lambda_reg, max_iter=10000, acceleration=False)

                # Check the optimization in SIMSOPT is working
                optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=lambda_reg)
                np.testing.assert_allclose(single_valued_current_potential_mn[i, :], optimized_phi_mn, rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: Tikhonov optimized_phi_mn mismatch")
                if lambda_reg == 0.0:
                    np.testing.assert_allclose(f_B, f_B_lasso, rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: f_B (Tikhonov) != f_B (Lasso) at lambda=0")
                    # np.testing.assert_allclose(f_B, f_B_lasso_ista, rtol=1e-1, atol=1e-12, err_msg=f"{filename} lambda[{i}]: f_B (Tikhonov) != f_B (Lasso ISTA) at lambda=0")

                # Check f_B from SquaredFlux and f_B from least-squares agree
                Bfield_opt = WindingSurfaceField(cpst.current_potential)
                Bfield_opt.set_points(s_plasma.gamma().reshape(-1, 3))
                _nfp = cpst.plasma_surface.nfp
                nphi = len(cpst.plasma_surface.quadpoints_phi)
                ntheta = len(cpst.plasma_surface.quadpoints_theta)
                _f_B_sq = SquaredFlux(
                    s_plasma,
                    Bfield_opt,
                    -np.ascontiguousarray(cpst.Bnormal_plasma.reshape(nphi, ntheta))
                ).J()

                # These do not agree well when lambda >> 1
                # or other situations where the exact plasma surface
                # locations are critical, so the REGCOIL Bnormal
                # calculation must be used
                #print(f_B, f_B_sq)
                #assert np.isclose(f_B, f_B_sq, rtol=1e-1)
                np.testing.assert_allclose(f_B, f_B_REGCOIL, rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: f_B != f_B_REGCOIL")
                np.testing.assert_allclose(f_K, f_K_REGCOIL, rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: f_K != f_K_REGCOIL")

                # check the REGCOIL Bnormal calculation in c++ """
                points = s_plasma.gamma().reshape(-1, 3)
                normal = s_plasma.normal().reshape(-1, 3)
                ws_points = s_coil.gamma().reshape(-1, 3)
                ws_normal = s_coil.normal().reshape(-1, 3)
                dtheta_coil = s_coil.quadpoints_theta[1]
                dzeta_coil = s_coil.quadpoints_phi[1]
                Bnormal_REGCOIL = WindingSurfaceBn_REGCOIL(points, ws_points, ws_normal, cp.Phi(), normal) * dtheta_coil * dzeta_coil

                normN = np.linalg.norm(normal, axis=-1)
                res = (np.ravel(Bnormal_regcoil_total[i, :, :]) ** 2) @ normN
                f_B_manual = 0.5 * res / (nphi * ntheta)
                np.testing.assert_allclose(f_B_REGCOIL, f_B_manual, rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: f_B_REGCOIL != f_B_manual")

                Bnormal_REGCOIL += B_GI_winding_surface
                np.testing.assert_allclose(Bnormal_REGCOIL, np.ravel(Bnormal_regcoil), rtol=1e-3, atol=1e-12, err_msg=f"{filename} lambda[{i}]: Bnormal_REGCOIL (C++) != Bnormal_regcoil")

    def test_Bnormal_interpolation_from_netcdf(self):
        """
        Test the branch that interpolates Bnormal_from_plasma when increasing
        plasma surface resolution via from_netcdf(plasma_ntheta_res > 1 or plasma_nzeta_res > 1).
        """
        filename = TEST_DIR / 'regcoil_out.w7x_infty.nc'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cpst = CurrentPotentialSolve.from_netcdf(
                filename, plasma_ntheta_res=2.0, plasma_nzeta_res=1.0
            )
        self.assertGreater(len(w), 0, msg="Expected interpolation accuracy warning")
        self.assertTrue(any("interpolated" in str(warning.message).lower() for warning in w))

        # Bnormal_plasma should have shape matching higher-resolution grid
        nzeta = cpst.nzeta_plasma
        ntheta = cpst.ntheta_plasma
        self.assertEqual(len(cpst.Bnormal_plasma), nzeta * ntheta,
                         msg="Bnormal_plasma length should match nzeta*ntheta")

        # solve_tikhonov and B_matrix_and_rhs should work
        b_rhs, B_matrix = cpst.B_matrix_and_rhs()
        self.assertEqual(len(b_rhs), cpst.ndofs)
        optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=1e-6)
        self.assertEqual(len(optimized_phi_mn), cpst.ndofs)

    def test_Bnormal_interpolation_plasma_nzeta_res(self):
        """Cover plasma_nzeta_res > 1 branch (plasma_ntheta_res=1, plasma_nzeta_res=2)."""
        filename = TEST_DIR / 'regcoil_out.w7x_infty.nc'
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cpst = CurrentPotentialSolve.from_netcdf(
                filename, plasma_ntheta_res=1.0, plasma_nzeta_res=2.0
            )
        self.assertEqual(len(cpst.Bnormal_plasma), cpst.nzeta_plasma * cpst.ntheta_plasma)
        optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=1e-6)
        self.assertEqual(len(optimized_phi_mn), cpst.ndofs)

    def test_CurrentPotentialSolve_Bnormal_plasma_shape_mismatch(self):
        """CurrentPotentialSolve raises ValueError when Bnormal_plasma shape mismatches."""
        cp = CurrentPotentialFourier.from_netcdf(TEST_DIR / 'regcoil_out.w7x_infty.nc')
        s_plasma = SurfaceRZFourier(
            nfp=cp.nfp, mpol=4, ntor=4, stellsym=True
        ).from_nphi_ntheta(nfp=cp.nfp, ntheta=32, nphi=32, mpol=4, ntor=4, stellsym=True, range="field period")
        # Bnormal with wrong size (e.g. 10 elements instead of 32*32)
        bad_Bnormal = np.ones(10)
        with self.assertRaises(ValueError) as cm:
            CurrentPotentialSolve(cp, s_plasma, bad_Bnormal)
        self.assertIn("shape", str(cm.exception).lower())

    def test_WindingSurfaceField_as_dict_from_dict(self):
        """Test WindingSurfaceField serialization via as_dict and from_dict.

        Uses surfaces from winding_surface_test.json (created via set_dofs so _dofs
        are in sync) since SurfaceRZFourier from from_netcdf uses set_rc/set_zs
        which does not sync _dofs for serialization.
        """
        from simsopt import load
        winding_surface, _ = load(TEST_DIR / 'winding_surface_test.json')
        cp = CurrentPotentialFourier(
            winding_surface, mpol=4, ntor=4,
            net_poloidal_current_amperes=11884578.094260072,
            net_toroidal_current_amperes=0,
            stellsym=True)
        cp.set_dofs(np.array([
            235217.63668779, -700001.94517193, 1967024.36417348,
            -1454861.01406576, -1021274.81793687, 1657892.17597651,
            -784146.17389912, 136356.84602536, -670034.60060171,
            194549.6432583, 1006169.72177152, -1677003.74430119,
            1750470.54137804, 471941.14387043, -1183493.44552104,
            1046707.62318593, -334620.59690486, 658491.14959397,
            -1169799.54944824, -724954.843765, 1143998.37816758,
            -2169655.54190455, -106677.43308896, 761983.72021537,
            -986348.57384563, 532788.64040937, -600463.7957275,
            1471477.22666607, 1009422.80860728, -2000273.40765417,
            2179458.3105468, -55263.14222144, -315581.96056445,
            587702.35409154, -637943.82177418, 609495.69135857,
            -1050960.33686344, -970819.1808181, 1467168.09965404,
            -198308.0580687
        ]))
        bfield = WindingSurfaceField(cp)
        points = np.ascontiguousarray(np.random.RandomState(42).rand(20, 3) * 2)
        bfield.set_points(points)
        B_orig = bfield.B()
        A_orig = bfield.A()

        field_json_str = json.dumps(SIMSON(bfield), cls=GSONEncoder)
        bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
        bfield_regen.set_points(points)
        np.testing.assert_allclose(bfield_regen.B(), B_orig, rtol=1e-3, atol=1e-12,
                                   err_msg="WindingSurfaceField B() mismatch after load")
        np.testing.assert_allclose(bfield_regen.A(), A_orig, rtol=1e-3, atol=1e-12,
                                   err_msg="WindingSurfaceField A() mismatch after load")

    def test_K_calculations(self):
        from simsopt import load
        winding_surface, plasma_surface = load(TEST_DIR / 'winding_surface_test.json')
        cp = CurrentPotentialFourier(
            winding_surface, mpol=4, ntor=4,
            net_poloidal_current_amperes=11884578.094260072,
            net_toroidal_current_amperes=0,
            stellsym=True)
        cp.set_dofs(np.array([
            235217.63668779, -700001.94517193, 1967024.36417348,
            -1454861.01406576, -1021274.81793687, 1657892.17597651,
            -784146.17389912, 136356.84602536, -670034.60060171,
            194549.6432583, 1006169.72177152, -1677003.74430119,
            1750470.54137804, 471941.14387043, -1183493.44552104,
            1046707.62318593, -334620.59690486, 658491.14959397,
            -1169799.54944824, -724954.843765, 1143998.37816758,
            -2169655.54190455, -106677.43308896, 761983.72021537,
            -986348.57384563, 532788.64040937, -600463.7957275,
            1471477.22666607, 1009422.80860728, -2000273.40765417,
            2179458.3105468, -55263.14222144, -315581.96056445,
            587702.35409154, -637943.82177418, 609495.69135857,
            -1050960.33686344, -970819.1808181, 1467168.09965404,
            -198308.0580687
        ]))
        cpst = CurrentPotentialSolve(cp, plasma_surface, np.zeros(1024))
        np.testing.assert_allclose(cpst.current_potential.get_dofs(), cp.get_dofs(), rtol=1e-3, atol=1e-12, err_msg="CurrentPotentialSolve dofs != original cp dofs")
        # Pre-compute some important matrices
        cpst.B_matrix_and_rhs()

        # Copied over from the packaged grid K operator.
        winding_surface = cp.winding_surface
        normal_vec = winding_surface.normal()
        normn = np.sqrt(np.sum(normal_vec**2, axis=-1))  # |N|

        test_K_1 = (
            cp.winding_surface.gammadash2()
            * (cp.Phidash1()+cp.net_poloidal_current_amperes)[:, :, None]
            - cp.winding_surface.gammadash1()
            * (cp.Phidash2()+cp.net_toroidal_current_amperes)[:, :, None])/normn[:, :, None]

        test_K_3 = cp.K()

        normn = normn.reshape(-1)
        dzeta_coil = (winding_surface.quadpoints_phi[1] - winding_surface.quadpoints_phi[0])
        dtheta_coil = (winding_surface.quadpoints_theta[1] - winding_surface.quadpoints_theta[0])

        # Notice Equation A.13 for the current in Matt L's regcoil paper has factor of 1/nnorm in it
        # But cpst.fj and cpst.d have factor of only 1/sqrt(normn)
        test_K_2 = -(cpst.fj @ cp.get_dofs() - cpst.d) / \
            (np.sqrt(dzeta_coil * dtheta_coil) * normn[:, None])
        nzeta_coil = cpst.nzeta_coil
        test_K_2 = test_K_2.reshape(nzeta_coil, nzeta_coil // cp.nfp, 3)

        # Figure 1: Surface current density K (A/m) from three equivalent formulations.
        # Row 1: K from analytic formula (Eq. A.13 REGCOIL).
        # Row 2: K from matrix computation (fj @ phi - d) / (sqrt(dzeta*dtheta) * |N|).
        # Row 3: K from cp.K() C++ implementation.
        # Columns: x, y, z components of K on (zeta, theta) winding surface grid.
        fig1, axes1 = plt.subplots(3, 3, figsize=(12, 10), squeeze=True)
        fig1.suptitle(r'Surface current density $\mathbf{K}$ (A/m): comparison of three formulations',
                      fontsize=12)
        for j in range(3):
            ax = axes1[0, j]
            im = ax.pcolor(test_K_1[:, :, j])
            fig1.colorbar(im, ax=ax)
            ax.set_title(r'Analytic: $K_{}$'.format('xyz'[j]))
            if j == 0:
                ax.set_ylabel(r'Analytic (Eq. A.13)')
        for j in range(3):
            ax = axes1[1, j]
            im = ax.pcolor(test_K_2[:, :, j])
            fig1.colorbar(im, ax=ax)
            ax.set_title(r'Matrix: $K_{}$'.format('xyz'[j]))
            if j == 0:
                ax.set_ylabel(r'Matrix (fj@phi-d)')
        for j in range(3):
            ax = axes1[2, j]
            im = ax.pcolor(test_K_3[:, :, j])
            fig1.colorbar(im, ax=ax)
            ax.set_title(r'cp.K(): $K_{}$'.format('xyz'[j]))
            ax.set_xlabel(r'$\theta$ (poloidal)')
            if j == 0:
                ax.set_ylabel(r'cp.K() (C++)')
        fig1.tight_layout()

        # Figure 2: Differences between formulations (should be ~0 if implementations agree).
        # Row 1: analytic - cp.K() for x, y, z.
        # Row 2: matrix - cp.K() for x, y, z.
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8), squeeze=True)
        fig2.suptitle(r'Difference in $\mathbf{K}$: analytic vs cp.K() and matrix vs cp.K() '
                      r'(should be ~0)', fontsize=12)
        # Row 0: analytic - cp.K()
        for j, comp in enumerate(['$K_x$', '$K_y$', '$K_z$']):
            ax = axes2[0, j]
            im = ax.pcolor(test_K_1[:, :, j] - test_K_3[:, :, j])
            fig2.colorbar(im, ax=ax)
            ax.set_title(r'Analytic $-$ cp.K(): ' + comp)
            if j == 0:
                ax.set_ylabel(r'Analytic $-$ cp.K()')
        # Row 1: matrix - cp.K()
        for j, comp in enumerate(['$K_x$', '$K_y$', '$K_z$']):
            ax = axes2[1, j]
            im = ax.pcolor(test_K_2[:, :, j] - test_K_3[:, :, j])
            fig2.colorbar(im, ax=ax)
            ax.set_title(r'Matrix $-$ cp.K(): ' + comp)
            ax.set_xlabel(r'$\theta$ (poloidal)')
            if j == 0:
                ax.set_ylabel(r'Matrix $-$ cp.K()')
        fig2.tight_layout()
        if not in_github_actions:
            plt.show()

        np.testing.assert_allclose(test_K_1, test_K_2, rtol=1e-3, atol=1e-12, err_msg="K from analytic formula != K from matrix computation")
        np.testing.assert_allclose(test_K_1, test_K_3, rtol=1e-3, atol=1e-12, err_msg="K from analytic formula != K from cp.K()")


if __name__ == "__main__":
    unittest.main()
