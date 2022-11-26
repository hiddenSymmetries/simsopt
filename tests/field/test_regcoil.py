import unittest

import numpy as np
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart, CurrentPotential, CurrentPotentialFourier, CurrentPotentialSolveTikhonov
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

        # uniform grid with half-step shift
        # qphi = np.linspace(0, 1, nphi) + 1 / (2 * nphi)
        # qtheta = np.linspace(0, 1, ntheta) + 1 / (2 * ntheta)

        # Make winding surface with major radius = 1, no minor radius
        winding_surface = SurfaceRZFourier()
        winding_surface = winding_surface.from_nphi_ntheta(nphi=nphi, ntheta=ntheta)
        #winding_surface = SurfaceRZFourier(quadpoints_phi=qphi, quadpoints_theta=qtheta)
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

        assert np.allclose(dB_predict, dB_analytic)

        # Now check that the far-field looks like a dipole
        points = (np.random.rand(N, 3) + 1) * 1000
        gamma = winding_surface.gamma().reshape((-1, 3))
        #print('Inner boundary of the infinitesimally thin wire = ', np.min(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        #print('Outer boundary of the infinitesimally thin wire = ', np.max(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        #print('Area and volume of the wire = ', winding_surface.area(), winding_surface.volume())

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

        # Now check that the far-field looks like a dipole
        points = (np.random.rand(N, 3) + 1) * 1000
        gamma = winding_surface.gamma().reshape((-1, 3))
        #print('Inner boundary of the infinitesimally thin wire = ', np.min(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        #print('Outer boundary of the infinitesimally thin wire = ', np.max(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        #print('Area and volume of the wire = ', winding_surface.area(), winding_surface.volume())

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
        Here we check the solve with lambda -> infinity to test the K matrices and rhs
        """
        for filename in ['regcoil_out.li383_infty.nc','regcoil_out.w7x_infty.nc']:
            filename = TEST_DIR / filename
            f = netcdf_file(filename, 'r')
            ilambda = 1
            Bnormal_regcoil_total = f.variables['Bnormal_total'][()][ilambda,:,:]
            Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
            Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
            r_plasma = f.variables['r_plasma'][()]
            r_coil = f.variables['r_coil'][()]
            nzeta_plasma = f.variables['nzeta_plasma'][()]
            K2_regcoil = f.variables['K2'][()][ilambda,:,:]
            lambda_regcoil = f.variables['lambda'][()][ilambda]
            b_rhs_regcoil = f.variables['RHS_B'][()]
            k_rhs_regcoil = f.variables['RHS_regularization'][()]
            single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][ilambda,:]
            norm_normal_plasma = f.variables['norm_normal_plasma'][()]
            current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()][ilambda,:,:]
            f.close()

            # initialize a solver object for the cp CurrentPotential
            cpst = CurrentPotentialSolveTikhonov.from_netcdf(filename)

            # Check B and K RHS's -> these are independent of lambda
            if False:
                assert np.allclose(b_rhs_regcoil, b_rhs_simsopt)

            k_rhs = cpst.K_rhs()
            assert np.allclose(k_rhs, k_rhs_regcoil)

            # Compare Bnormal from plasma
            assert np.allclose(cpst.Bnormal_plasma, Bnormal_from_plasma_current.flatten())

            #  Compare optimized dofs
            optimized_phi_mn = cpst.solve(lam=lambda_regcoil)
            assert np.allclose(single_valued_current_potential_mn, optimized_phi_mn)

            cp = cpst.current_potential

            s_plasma = cpst.plasma_surface

            # Compare plasma surface position
            assert np.allclose(r_plasma[0:nzeta_plasma, :, :], s_plasma.gamma())

            # Compare plasma surface normal
            assert np.allclose(norm_normal_plasma[0:nzeta_plasma, :], np.linalg.norm(s_plasma.normal(),axis=2)/(2*np.pi*2*np.pi))

            # Compare winding surface position
            s_coil = cp.winding_surface
            assert np.allclose(r_coil, s_coil.gamma())

            # Compare field from net coil currents
            cp_GI = CurrentPotentialFourier.from_netcdf(filename)
            Bfield = WindingSurfaceField(cp_GI)
            points = s_plasma.gamma().reshape(-1, 3)
            Bfield.set_points(points)
            B = Bfield.B()
            normal = s_plasma.unitnormal().reshape(-1, 3)
            B_GI_winding_surface = np.sum(B*normal, axis=1)
            assert np.allclose(B_GI_winding_surface, np.ravel(Bnormal_from_net_coil_currents))
            assert np.allclose(cpst.B_GI, np.ravel(Bnormal_from_net_coil_currents))

            # Compare single-valued current potential
            cp_no_GI = CurrentPotentialFourier.from_netcdf(filename)
            cp_no_GI.net_toroidal_current_amperes = 0
            cp_no_GI.net_poloidal_current_amperes = 0
            cp_no_GI.set_dofs(optimized_phi_mn)
            assert np.allclose(cp_no_GI.Phi()[0:nzeta_plasma, :],current_potential_thetazeta)

            # Compare current density
            cp.set_dofs(optimized_phi_mn)
            K = cp.K()
            K2 = np.sum(K*K, axis=2)
            K2_average = np.mean(K2, axis=(0, 1))

            assert np.allclose(K2[0:nzeta_plasma, :]/K2_average, K2_regcoil/K2_average)

            # Check normal field
            Bfield_opt = WindingSurfaceField(cp)
            Bfield_opt.set_points(s_plasma.gamma().reshape(-1,3))
            B_opt = Bfield_opt.B()
            normal = s_plasma.unitnormal().reshape(-1, 3)
            Bnormal = np.sum(B_opt*normal, axis=1).reshape(np.shape(s_plasma.gamma()[:, :, 0]))
            Bnormal_regcoil = Bnormal_regcoil_total - Bnormal_from_plasma_current

            self.assertAlmostEqual(np.sum(Bnormal), 0)
            self.assertAlmostEqual(np.sum(Bnormal_regcoil), 0)

            if False:
                print(np.max(np.abs(Bnormal-Bnormal_regcoil)))
                print(np.mean(np.abs(Bnormal)))
                print(np.mean(np.abs(Bnormal_regcoil)))
                assert np.allclose(Bnormal,Bnormal_regcoil)

    def test_winding_surface_regcoil(self):
        # This compares the normal field from regcoil with that computed from
        # WindingSurface for W7-X and NCSX configuration
        for filename in ['regcoil_out.w7x.nc','regcoil_out.axisymmetry.nc','regcoil_out.li383.nc','regcoil_out.axisymmetry_asym.nc']:
        # for filename in [TEST_DIR / , TEST_DIR / 'regcoil_out.li383.nc', TEST_DIR / ]:
        # for filename in [TEST_DIR / 'regcoil_out.li383.nc', TEST_DIR / 'regcoil_out.w7x.nc']:
        # for filename in [TEST_DIR / 'regcoil_out.li383.nc']:
            print(filename)
            filename = TEST_DIR / filename
            f = netcdf_file(filename, 'r')
            Bnormal_regcoil_total = f.variables['Bnormal_total'][()]
            Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
            Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
            r_plasma = f.variables['r_plasma'][()]
            r_coil = f.variables['r_coil'][()]
            nzeta_plasma = f.variables['nzeta_plasma'][()]
            K2_regcoil = f.variables['K2'][()]
            lambda_regcoil = f.variables['lambda'][()]
            #lambda_regcoil = f.variables['lambda'][()][1]
            b_rhs_regcoil = f.variables['RHS_B'][()]
            k_rhs_regcoil = f.variables['RHS_regularization'][()]
            single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()]
            norm_normal_plasma = f.variables['norm_normal_plasma'][()]
            current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()]
            f.close()

            # Compare K and B RHS's -> these are independent of lambda
            cpst = CurrentPotentialSolveTikhonov.from_netcdf(filename)
            b_rhs_simsopt = cpst.B_rhs()
            if False:
                assert np.allclose(b_rhs_regcoil, b_rhs_simsopt)

            # this comparison doesn't work for stellarator asymmetric
            k_rhs = cpst.K_rhs()
            assert np.allclose(k_rhs, k_rhs_regcoil)

            # Compare Bnormal from net coil currents
            assert np.allclose(cpst.B_GI, np.ravel(Bnormal_from_net_coil_currents))

            # Compare plasma current
            assert np.allclose(cpst.Bnormal_plasma, Bnormal_from_plasma_current.flatten())

            cp = cpst.current_potential
            cp_no_GI = CurrentPotentialFourier.from_netcdf(filename)
            cp_no_GI.net_toroidal_current_amperes = 0
            cp_no_GI.net_poloidal_current_amperes = 0

            s_plasma = cpst.plasma_surface

            # Compare plasma surface position
            assert np.allclose(r_plasma[0:nzeta_plasma, :, :], s_plasma.gamma())

            # Compare plasma surface normal
            assert np.allclose(norm_normal_plasma[0:nzeta_plasma, :], np.linalg.norm(s_plasma.normal(),axis=2)/(2*np.pi*2*np.pi))

            # Compare winding surface position
            s_coil = cp.winding_surface
            assert np.allclose(r_coil, s_coil.gamma())

            # Compare field from net coil currents
            cp_GI = CurrentPotentialFourier.from_netcdf(filename)
            Bfield = WindingSurfaceField(cp_GI)
            points = s_plasma.gamma().reshape(-1, 3)
            Bfield.set_points(points)
            B = Bfield.B()
            normal = s_plasma.unitnormal().reshape(-1, 3)
            B_GI_winding_surface = np.sum(B*normal, axis=1)
            assert np.allclose(B_GI_winding_surface, np.ravel(Bnormal_from_net_coil_currents))

            # Solve the least-squares problem with the specified plasma
            # quadrature points, normal vector, and Bnormal at these quadrature points
            for i, lambda_reg in enumerate(lambda_regcoil):
                # Set current potential Fourier harmonis from regcoil file
                cp.set_current_potential_from_regcoil(filename,i)

                # Compare current potential Fourier harmonics
                assert np.allclose(cp.get_dofs(), single_valued_current_potential_mn[i,:])

                # I'm not sure why I need to make a new object here, but otherwise it doesn't work
                cp_no_GI = CurrentPotentialFourier.from_netcdf(filename)
                cp_no_GI.net_toroidal_current_amperes = 0
                cp_no_GI.net_poloidal_current_amperes = 0
                cp_no_GI.set_dofs(0*cp_no_GI.get_dofs())
                cp_no_GI.set_current_potential_from_regcoil(filename,i)

                # Compare single-valued current potential
                assert np.allclose(cp_no_GI.Phi()[0:nzeta_plasma, :],current_potential_thetazeta[i,:,:])

                # I'm not sure why I need to make a new object here, but otherwise it doesn't work
                cp = CurrentPotentialFourier.from_netcdf(filename)
                cp.set_current_potential_from_regcoil(filename,i)
                K = cp.K()
                K2 = np.sum(K*K, axis=2)
                K2_average = np.mean(K2, axis=(0, 1))

                assert np.allclose(K2[0:nzeta_plasma, :]/K2_average, K2_regcoil[i,:,:]/K2_average)

                # Initialize a solver object for the cp CurrentPotential
                cp = CurrentPotentialFourier.from_netcdf(filename)
                cp.set_current_potential_from_regcoil(filename,i)
                Bfield_opt = WindingSurfaceField(cp)
                Bfield_opt.set_points(s_plasma.gamma().reshape(-1,3))
                B_opt = Bfield_opt.B()
                normal = s_plasma.unitnormal().reshape(-1, 3)
                Bnormal = np.sum(B_opt*normal, axis=1).reshape(np.shape(s_plasma.gamma()[:, :, 0]))
                Bnormal_regcoil = Bnormal_regcoil_total[i,:,:] - Bnormal_from_plasma_current

                if False:
                    self.assertAlmostEqual(np.sum(Bnormal), 0)
                    self.assertAlmostEqual(np.sum(Bnormal_regcoil), 0)
                    print(np.max(np.abs(Bnormal-Bnormal_regcoil)))
                    print(np.mean(np.abs(Bnormal)))
                    print(np.mean(np.abs(Bnormal_regcoil)))
                    assert np.allclose(Bnormal,Bnormal_regcoil)

                    optimized_phi_mn = cpst.solve(lam=lambda_reg)
                    assert np.allclose(single_valued_current_potential_mn[i,:], optimized_phi_mn)

if __name__ == "__main__":
    unittest.main()
