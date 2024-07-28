import unittest
from simsopt.geo import Surface, SurfaceRZFourier
from matplotlib import pyplot as plt
import numpy as np
from simsoptpp import WindingSurfaceBn_REGCOIL
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.objectives import SquaredFlux
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart, CurrentPotential, CurrentPotentialFourier, CurrentPotentialSolve
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
        ntheta = 8

        # Make winding surface with major radius = 1, no minor radius
        winding_surface = SurfaceRZFourier()
        winding_surface = winding_surface.from_nphi_ntheta(nphi=nphi, ntheta=ntheta)
        for i in range(winding_surface.mpol + 1):
            for j in range(-winding_surface.ntor, winding_surface.ntor + 1):
                winding_surface.set_rc(i, j, 0.0)
                winding_surface.set_zs(i, j, 0.0)
        winding_surface.set_rc(0, 0, 1.0)
        eps = 1e-8
        winding_surface.set_rc(1, 0, eps)  # current loop must have finite width for simsopt
        winding_surface.set_zs(1, 0, eps)  # current loop must have finite width for simsopt

        # Make CurrentPotential class from this winding surface with 1 amp toroidal current
        current_potential = CurrentPotentialFourier(winding_surface, net_poloidal_current_amperes=0, net_toroidal_current_amperes=-1)

        # compute the Bfield from this current loop at some points
        Bfield = WindingSurfaceField(current_potential)
        N = 1000
        phi = winding_surface.quadpoints_phi

        # Check that the full expression is correct
        points = (np.random.rand(N, 3) - 0.5) * 10
        Bfield.set_points(np.ascontiguousarray(points))
        B_predict = Bfield.B()
        dB_predict = Bfield.dB_by_dX()
        A_predict = Bfield.A()
        dA_predict = Bfield.dA_by_dX()

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

        assert np.allclose(A_predict, A_analytic_elliptic)

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

        assert np.allclose(B_predict, B_analytic)

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

        assert np.allclose(dB_predict, dB_analytic, rtol=1e-2)

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

        assert np.allclose(A_predict, A_analytic_elliptic)

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

        assert np.allclose(A_predict, A_analytic_elliptic)

        # double check with vector potential of a dipole
        Aphi = np.pi * mu_fac * np.sin(theta) / r ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        assert np.allclose(A_predict, A_analytic)

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
                f = netcdf_file(filename, 'r')
                Bnormal_regcoil_total = f.variables['Bnormal_total'][()][ilambda, :, :]
                Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
                Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
                r_plasma = f.variables['r_plasma'][()]
                r_coil = f.variables['r_coil'][()]
                nzeta_plasma = f.variables['nzeta_plasma'][()]
                nzeta_coil = f.variables['nzeta_coil'][()]
                ntheta_coil = f.variables['ntheta_coil'][()]
                nfp = f.variables['nfp'][()]
                ntheta_plasma = f.variables['ntheta_plasma'][()]
                K2_regcoil = f.variables['K2'][()][ilambda, :, :]
                lambda_regcoil = f.variables['lambda'][()][ilambda]
                b_rhs_regcoil = f.variables['RHS_B'][()]
                k_rhs_regcoil = f.variables['RHS_regularization'][()]
                single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][ilambda, :]
                xm_potential = f.variables['xm_potential'][()]
                xn_potential = f.variables['xn_potential'][()]
                theta_coil = f.variables['theta_coil'][()]
                zeta_coil = f.variables['zeta_coil'][()]
                f_B_regcoil = 0.5 * f.variables['chi2_B'][()][ilambda]
                f_K_regcoil = 0.5 * f.variables['chi2_K'][()][ilambda]
                norm_normal_plasma = f.variables['norm_normal_plasma'][()]
                current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()][ilambda, :, :]
                f.close()
                Bnormal_single_valued = Bnormal_regcoil_total - Bnormal_from_plasma_current - Bnormal_from_net_coil_currents
                print('ilambda index = ', ilambda, lambda_regcoil)

                assert np.allclose(b_rhs_regcoil, b_rhs_simsopt)
                assert np.allclose(k_rhs, k_rhs_regcoil)

                # Compare Bnormal from plasma
                assert np.allclose(cpst.Bnormal_plasma, Bnormal_from_plasma_current.flatten())

                # Compare optimized dofs
                cp = cpst.current_potential

                # when lambda -> infinity, the L1 and L2 regularized problems should agree
                optimized_phi_mn_lasso, f_B_lasso, f_K_lasso, _, _ = cpst.solve_lasso(lam=lambda_regcoil)
                optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=lambda_regcoil)
                assert np.allclose(single_valued_current_potential_mn, optimized_phi_mn)
                print(optimized_phi_mn_lasso, optimized_phi_mn)
                print(f_B, f_B_lasso, f_B_regcoil)
                assert np.isclose(f_B, f_B_regcoil)
                # assert np.isclose(f_K_lasso, f_K)
                assert np.allclose(optimized_phi_mn_lasso, optimized_phi_mn)

                # Compare plasma surface position
                assert np.allclose(r_plasma[0:nzeta_plasma, :, :], s_plasma.gamma())

                # Compare plasma surface normal
                assert np.allclose(
                    norm_normal_plasma[0:nzeta_plasma, :],
                    np.linalg.norm(s_plasma.normal(), axis=2) / (2 * np.pi * 2 * np.pi)
                )

                # Compare winding surface position
                s_coil = cp.winding_surface
                assert np.allclose(r_coil, s_coil.gamma())

                # Compare field from net coil currents
                cp_GI = CurrentPotentialFourier.from_netcdf(filename)
                Bfield = WindingSurfaceField(cp_GI)
                points = s_plasma.gamma().reshape(-1, 3)
                Bfield.set_points(points)
                B = Bfield.B()
                norm_normal = np.linalg.norm(s_plasma.normal(), axis=2) / (2 * np.pi * 2 * np.pi)
                normal = s_plasma.unitnormal().reshape(-1, 3)
                B_GI_winding_surface = np.sum(B * normal, axis=1)
                assert np.allclose(B_GI_winding_surface, np.ravel(Bnormal_from_net_coil_currents))
                assert np.allclose(cpst.B_GI, np.ravel(Bnormal_from_net_coil_currents))

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
                assert np.allclose(cp_no_GI.Phi()[0:nzeta_coil, :], current_potential_thetazeta)

                # Check that f_B from SquaredFlux and f_B from least-squares agree
                Bfield_opt = WindingSurfaceField(cp)
                Bfield_opt.set_points(s_plasma.gamma().reshape(-1, 3))
                B = Bfield_opt.B()
                normal = s_plasma.unitnormal().reshape(-1, 3)
                Bn_opt = np.sum(B * normal, axis=1)
                nfp = cpst.plasma_surface.nfp
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
                assert np.isclose(f_B_regcoil, f_B_sq, rtol=1e-4)

                # Compare current density
                cp.set_dofs(optimized_phi_mn)
                K = cp.K()
                K2 = np.sum(K ** 2, axis=2)
                K2_average = np.mean(K2, axis=(0, 1))
                assert np.allclose(K2[0:nzeta_coil, :] / K2_average, K2_regcoil / K2_average)

                # Compare values of f_K computed in three different ways
                normal = s_coil.normal().reshape(-1, 3)
                normN = np.linalg.norm(normal, axis=-1)
                f_K_direct = 0.5 * np.sum(np.ravel(K2) * normN) / (normal.shape[0])
                # print(f_K_regcoil, f_K_direct, f_K)
                assert np.isclose(f_K_regcoil, f_K_direct)
                assert np.isclose(f_K_regcoil, f_K)

                # Check normal field
                Bfield_opt = WindingSurfaceField(cp)
                Bfield_opt.set_points(s_plasma.gamma().reshape(-1, 3))
                B_opt = Bfield_opt.B()
                normal = s_plasma.unitnormal().reshape(-1, 3)
                Bnormal = np.sum(B_opt*normal, axis=1).reshape(np.shape(s_plasma.gamma()[:, :, 0]))
                Bnormal_regcoil = Bnormal_regcoil_total - Bnormal_from_plasma_current
                self.assertAlmostEqual(np.sum(Bnormal), 0)
                self.assertAlmostEqual(np.sum(Bnormal_regcoil), 0)

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
                assert np.allclose(Bnormal_REGCOIL, np.ravel(Bnormal_single_valued))
                normN = np.linalg.norm(normal, axis=-1)
                res = (np.ravel(Bnormal_regcoil_total) ** 2) @ normN
                f_B_manual = 0.5 * res / (nphi * ntheta)
                assert np.isclose(f_B_regcoil, f_B_manual, rtol=1e-4)

                Bnormal_g += B_GI_winding_surface.reshape(np.shape(s_plasma.gamma()[:, :, 0]))
                Bnormal_REGCOIL += B_GI_winding_surface

                # Check that Bnormal calculations using the REGCOIL discretization all agree
                assert np.allclose(np.ravel(Bnormal_g), Bnormal_REGCOIL)
                assert np.allclose(np.ravel(Bnormal_g), np.ravel(Bnormal_regcoil))

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
            f = netcdf_file(filename, 'r')
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
            xm_potential = f.variables['xm_potential'][()]
            xn_potential = f.variables['xn_potential'][()]
            nfp = f.variables['nfp'][()]
            single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()]
            norm_normal_plasma = f.variables['norm_normal_plasma'][()]
            current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()]
            norm_normal_coil = f.variables['norm_normal_coil'][()]
            f.close()

            # Compare K and B RHS's -> these are independent of lambda
            cpst = CurrentPotentialSolve.from_netcdf(filename)

            b_rhs_simsopt, _ = cpst.B_matrix_and_rhs()

            assert np.allclose(b_rhs_regcoil, b_rhs_simsopt)

            k_rhs = cpst.K_rhs()
            assert np.allclose(k_rhs, k_rhs_regcoil)

            # Compare plasma current
            assert np.allclose(cpst.Bnormal_plasma, Bnormal_from_plasma_current.flatten())

            # Compare Bnormal from net coil currents
            assert np.allclose(cpst.B_GI, np.ravel(Bnormal_from_net_coil_currents))

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
            assert np.allclose(r_plasma, s_plasma_full.gamma())

            # Compare plasma surface normal
            norm_normal_plasma_simsopt = np.linalg.norm(s_plasma_full.normal(), axis=-1)
            assert np.allclose(norm_normal_plasma*2*np.pi*2*np.pi, norm_normal_plasma_simsopt[0:nzeta_plasma, :])

            # Compare winding surface position
            s_coil = cp.winding_surface
            assert np.allclose(r_coil, s_coil.gamma())

            # Compare winding surface normal
            norm_normal_coil_simsopt = np.linalg.norm(s_coil.normal(), axis=-1)
            assert np.allclose(norm_normal_coil*2*np.pi*2*np.pi, norm_normal_coil_simsopt[0:nzeta_coil, :])

            # Compare two different ways of computing K()
            K = cp.K().reshape(-1, 3)
            winding_surface = cp.winding_surface
            normal_vec = winding_surface.normal().reshape(-1, 3)
            dzeta_coil = (winding_surface.quadpoints_phi[1] - winding_surface.quadpoints_phi[0])
            dtheta_coil = (winding_surface.quadpoints_theta[1] - winding_surface.quadpoints_theta[0])
            normn = np.sqrt(np.sum(normal_vec**2, axis=-1)) # |N|
            K_2 = -(cpst.fj @ cp.get_dofs() - cpst.d) / \
                    (np.sqrt(dzeta_coil * dtheta_coil) * normn[:, None]) 
            assert np.allclose(K, K_2)

            # Compare field from net coil currents
            cp_GI = CurrentPotentialFourier.from_netcdf(filename)
            Bfield = WindingSurfaceField(cp_GI)
            points = s_plasma.gamma().reshape(-1, 3)
            Bfield.set_points(points)
            B = Bfield.B()
            normal = s_plasma.unitnormal().reshape(-1, 3)
            B_GI_winding_surface = np.sum(B * normal, axis=1)
            assert np.allclose(B_GI_winding_surface, np.ravel(Bnormal_from_net_coil_currents))
            # Make sure single-valued part of current potential is working
            cp_no_GI = CurrentPotentialFourier.from_netcdf(filename)
            cp_no_GI.set_net_toroidal_current_amperes(0)
            cp_no_GI.set_net_poloidal_current_amperes(0)

            # Now loop over all the regularization values in the REGCOIL solution
            for i, lambda_reg in enumerate(lambda_regcoil):

                # Set current potential Fourier harmonics from regcoil file
                cp.set_current_potential_from_regcoil(filename, i)

                # Compare current potential Fourier harmonics
                assert np.allclose(cp.get_dofs(), single_valued_current_potential_mn[i, :])

                # Compare current density
                K = cp.K()
                K2 = np.sum(K ** 2, axis=2)
                K2_average = np.mean(K2, axis=(0, 1))
                assert np.allclose(K2[0:nzeta_plasma, :] / K2_average, K2_regcoil[i, :, :] / K2_average)

                f_B_REGCOIL = f_B_regcoil[i]
                f_K_REGCOIL = f_K_regcoil[i]

                cp_no_GI.set_current_potential_from_regcoil(filename, i)

                # Compare single-valued current potential
                assert np.allclose(cp_no_GI.Phi()[0:nzeta_plasma, :], current_potential_thetazeta[i, :, :])

                f_K_direct = 0.5 * np.sum(K2 * norm_normal_coil_simsopt) / (norm_normal_coil_simsopt.shape[0]*norm_normal_coil_simsopt.shape[1])
                self.assertAlmostEqual(f_K_direct/np.abs(f_K_REGCOIL), f_K_REGCOIL/np.abs(f_K_REGCOIL))

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

                assert np.allclose(Bnormal, Bnormal_regcoil)

                # check Bnormal and Bnormal_regcoil integrate over the surface to zero
                # This is only true in the stellarator symmetric case!
                if s_plasma.stellsym:
                    self.assertAlmostEqual(np.sum(Bnormal*norm_normal_plasma_simsopt), 0)
                    self.assertAlmostEqual(np.sum(Bnormal_regcoil*norm_normal_plasma_simsopt[0:nzeta_plasma, :]), 0)

                # Check that L1 optimization agrees if lambda = 0
                if lambda_reg == 0.0:
                    optimized_phi_mn_lasso, f_B_lasso, f_K_lasso, fB_history, _ = cpst.solve_lasso(lam=lambda_reg, max_iter=100, acceleration=True)

                # Check the optimization in SIMSOPT is working
                optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=lambda_reg)
                assert np.allclose(single_valued_current_potential_mn[i, :], optimized_phi_mn)
                if lambda_reg == 0.0:
                    assert np.isclose(f_B, f_B_lasso, rtol=1e-3)

                # Check f_B from SquaredFlux and f_B from least-squares agree
                Bfield_opt = WindingSurfaceField(cpst.current_potential)
                Bfield_opt.set_points(s_plasma.gamma().reshape(-1, 3))
                nfp = cpst.plasma_surface.nfp
                nphi = len(cpst.plasma_surface.quadpoints_phi)
                ntheta = len(cpst.plasma_surface.quadpoints_theta)
                f_B_sq = SquaredFlux(
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
                assert np.isclose(f_B, f_B_REGCOIL, rtol=1e-2)
                assert np.isclose(f_K, f_K_REGCOIL, rtol=1e-2)

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
                assert np.isclose(f_B_REGCOIL, f_B_manual, rtol=1e-4)

                Bnormal_REGCOIL += B_GI_winding_surface
                assert np.allclose(Bnormal_REGCOIL, np.ravel(Bnormal_regcoil))

    def test_K_calculations(self):
        from simsopt import load
        winding_surface, plasma_surface = load(TEST_DIR / 'winding_surface_test.json')
        cp = CurrentPotentialFourier(
            winding_surface, mpol=4, ntor=4,
            net_poloidal_current_amperes=11884578.094260072,
            net_toroidal_current_amperes=0,
            stellsym=True)
        cp.set_dofs(np.array([  
            235217.63668779,  -700001.94517193,  1967024.36417348,
            -1454861.01406576, -1021274.81793687,  1657892.17597651,
            -784146.17389912,   136356.84602536,  -670034.60060171,
            194549.6432583 ,  1006169.72177152, -1677003.74430119,
            1750470.54137804,   471941.14387043, -1183493.44552104,
            1046707.62318593,  -334620.59690486,   658491.14959397,
            -1169799.54944824,  -724954.843765  ,  1143998.37816758,
            -2169655.54190455,  -106677.43308896,   761983.72021537,
            -986348.57384563,   532788.64040937,  -600463.7957275 ,
            1471477.22666607,  1009422.80860728, -2000273.40765417,
            2179458.3105468 ,   -55263.14222144,  -315581.96056445,
            587702.35409154,  -637943.82177418,   609495.69135857,
            -1050960.33686344,  -970819.1808181 ,  1467168.09965404,
            -198308.0580687 
        ]))
        cpst = CurrentPotentialSolve(cp, plasma_surface, np.zeros(1024))
        assert np.allclose(cpst.current_potential.get_dofs(), cp.get_dofs())
        # Pre-compute some important matrices
        cpst.B_matrix_and_rhs()

        # Copied over from the packaged grid K operator.
        winding_surface = cp.winding_surface
        normal_vec = winding_surface.normal()
        normn = np.sqrt(np.sum(normal_vec**2, axis=-1)) # |N|

        test_K_1 = (
            cp.winding_surface.gammadash2()
            *(cp.Phidash1()+cp.net_poloidal_current_amperes)[:, :, None]
            - cp.winding_surface.gammadash1()
            *(cp.Phidash2()+cp.net_toroidal_current_amperes)[:, :, None])/normn[:, :, None]

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

        plt.figure(1)
        plt.subplot(3, 3, 1)
        plt.pcolor(test_K_1[:, :, 0])
        plt.colorbar()
        plt.subplot(3, 3, 2)
        plt.pcolor(test_K_1[:, :, 1])
        plt.colorbar()
        plt.subplot(3, 3, 3)
        plt.pcolor(test_K_1[:, :, 2])
        plt.colorbar()
        plt.subplot(3, 3, 4)
        plt.pcolor(test_K_2[:, :, 0])
        plt.colorbar()
        plt.subplot(3, 3, 5)
        plt.pcolor(test_K_2[:, :, 1])
        plt.colorbar()
        plt.subplot(3, 3, 6)
        plt.pcolor(test_K_2[:, :, 2])
        plt.colorbar()
        plt.subplot(3, 3, 7)
        plt.pcolor(test_K_3[:, :, 0])
        plt.colorbar()
        plt.subplot(3, 3, 8)
        plt.pcolor(test_K_3[:, :, 1])
        plt.colorbar()
        plt.subplot(3, 3, 9)
        plt.pcolor(test_K_3[:, :, 2])
        plt.colorbar()


        # In[21]:

        plt.figure(2)
        # Errors
        plt.subplot(3, 3, 1)
        plt.pcolor(test_K_1[:, :, 0] - test_K_3[:, :, 0])
        plt.colorbar()
        plt.subplot(3, 3, 2)
        plt.pcolor(test_K_1[:, :, 1] - test_K_3[:, :, 1])
        plt.colorbar()
        plt.subplot(3, 3, 3)
        plt.pcolor(test_K_1[:, :, 2] - test_K_3[:, :, 2])
        plt.colorbar()
        plt.subplot(3, 3, 4)
        plt.pcolor(test_K_2[:, :, 0] - test_K_3[:, :, 0])
        plt.colorbar()
        plt.subplot(3, 3, 5)
        plt.pcolor(test_K_2[:, :, 1] - test_K_3[:, :, 1])
        plt.colorbar()
        plt.subplot(3, 3, 6)
        plt.pcolor(test_K_2[:, :, 2] - test_K_3[:, :, 2])
        plt.colorbar()
        plt.show()

        assert np.allclose(test_K_1, test_K_2)
        assert np.allclose(test_K_1, test_K_3)

if __name__ == "__main__":
    unittest.main()
