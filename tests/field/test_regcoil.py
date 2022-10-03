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
        print('Inner boundary of the infinitesimally thin wire = ', np.min(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        print('Outer boundary of the infinitesimally thin wire = ', np.max(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        print('Area and volume of the wire = ', winding_surface.area(), winding_surface.volume())

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
        print('Inner boundary of the infinitesimally thin wire = ', np.min(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        print('Outer boundary of the infinitesimally thin wire = ', np.max(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        print('Area and volume of the wire = ', winding_surface.area(), winding_surface.volume())

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

    def test_winding_surface_regcoil(self):
        # This compares the normal field from regcoil with that computed from
        # WindingSurface for W7-X and NCSX configuration

        stellsym = True
        for filename in [TEST_DIR / 'regcoil_out.li383.nc', TEST_DIR / 'regcoil_out.w7x.nc']:
            f = netcdf_file(filename, 'r')
            Bnormal_regcoil = f.variables['Bnormal_total'][()][-1, :, :]
            Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
            Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
            Bnormal_regcoil = Bnormal_regcoil - Bnormal_from_plasma_current
            r_plasma = f.variables['r_plasma'][()]
            r_coil = f.variables['r_coil'][()]
            rmnc_plasma = f.variables['rmnc_plasma'][()]
            zmns_plasma = f.variables['zmns_plasma'][()]
            xm_plasma = f.variables['xm_plasma'][()]
            xn_plasma = f.variables['xn_plasma'][()]
            nfp = f.variables['nfp'][()]
            mpol_plasma = int(np.max(xm_plasma))
            ntor_plasma = int(np.max(xn_plasma)/nfp)
            ntheta_plasma = f.variables['ntheta_plasma'][()]
            nzeta_plasma = f.variables['nzeta_plasma'][()]
            mpol_potential = f.variables['mpol_potential'][()]
            ntor_potential = f.variables['ntor_potential'][()]
            net_poloidal_current_amperes = f.variables['net_poloidal_current_Amperes'][()]
            net_toroidal_current_amperes = f.variables['net_toroidal_current_Amperes'][()]
            xm_potential = f.variables['xm_potential'][()]
            xn_potential = f.variables['xn_potential'][()]
            K2_regcoil = f.variables['K2'][()][-1, :, :]
            lambda_regcoil = f.variables['lambda'][()]
            b_rhs_regcoil = f.variables['RHS_B'][()]
            #B_matrix_regcoil = f.variables['matrix_B'][()]
            #print('Bregcoil = ', B_matrix_regcoil)
            print('b_rhs = ', b_rhs_regcoil)
            print('B_GI_regcoil = ', Bnormal_from_net_coil_currents, Bnormal_from_net_coil_currents.shape)

            rmnc_coil = f.variables['rmnc_coil'][()]
            zmns_coil = f.variables['zmns_coil'][()]
            xm_coil = f.variables['xm_coil'][()]
            xn_coil = f.variables['xn_coil'][()]
            ntheta_coil = f.variables['ntheta_coil'][()]
            nzeta_coil = f.variables['nzeta_coil'][()]
            single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][-1, :]
            mpol_coil = int(np.max(xm_coil))
            ntor_coil = int(np.max(xn_coil)/nfp)

            s_plasma = SurfaceRZFourier(nfp=nfp, 
                                        mpol=mpol_plasma, ntor=ntor_plasma, stellsym=stellsym)
            s_plasma = s_plasma.from_nphi_ntheta(nfp=nfp, ntheta=ntheta_plasma, nphi=nzeta_plasma,
                                                 mpol=mpol_plasma, ntor=ntor_plasma, stellsym=stellsym, range="field period")
            s_plasma.set_dofs(0*s_plasma.get_dofs())
            for im in range(len(xm_plasma)):
                s_plasma.set_rc(xm_plasma[im], int(xn_plasma[im]/nfp), rmnc_plasma[im])
                s_plasma.set_zs(xm_plasma[im], int(xn_plasma[im]/nfp), zmns_plasma[im])

            assert np.allclose(r_plasma[0:nzeta_plasma, :, :], s_plasma.gamma())

            quadpoints_phi = np.linspace(0, nzeta_coil * nfp, 1) + 1 / (2 * nzeta_coil * nfp)
            quadpoints_theta = np.linspace(0, ntheta_coil, 1) + 1 / (2 * ntheta_coil)
            s_coil = SurfaceRZFourier(nfp=nfp, 
                                      mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym)
            s_coil = s_coil.from_nphi_ntheta(nfp=nfp, ntheta=ntheta_coil, nphi=nzeta_coil*nfp,
                                             mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym, range='full torus')
            s_coil.set_dofs(0*s_coil.get_dofs())
            for im in range(len(xm_coil)):
                s_coil.set_rc(xm_coil[im], int(xn_coil[im]/nfp), rmnc_coil[im])
                s_coil.set_zs(xm_coil[im], int(xn_coil[im]/nfp), zmns_coil[im])

            assert np.allclose(r_coil, s_coil.gamma())

            cp = CurrentPotentialFourier(s_coil, mpol=mpol_potential, ntor=ntor_potential,
                                         net_poloidal_current_amperes=net_poloidal_current_amperes,
                                         net_toroidal_current_amperes=net_toroidal_current_amperes)
            for im in range(len(xm_potential)):
                cp.set_phis(xm_potential[im], int(xn_potential[im]/nfp), single_valued_current_potential_mn[im])

            K = cp.K()
            K2 = np.sum(K*K, axis=2)
            K2_average = np.mean(K2, axis=(0, 1))

            assert np.allclose(K2[0:nzeta_plasma, :]/K2_average, K2_regcoil/K2_average)

            # initialize a solver object for the cp CurrentPotential
            cpst = CurrentPotentialSolveTikhonov(cp)

            # Solve the least-squares problem with the specified plasma 
            # quadrature points, normal vector, and Bnormal at these quadrature points
            optimized_dofs = cpst.solve(s_plasma, np.ravel(Bnormal_from_plasma_current), lam=lambda_regcoil)
            print('optimized dofs = ', optimized_dofs)
            cp.set_dofs(optimized_dofs)
            print('Current potential MN = ', single_valued_current_potential_mn)
            exit()

            # Initialize Bfield from optimized CurrentPotential
            Bfield = WindingSurfaceField(cp)
            points = s_plasma.gamma().reshape((int(len(s_plasma.gamma().flatten())/3), 3))
            Bfield.set_points(points)
            B = Bfield.B()

            normal = s_plasma.unitnormal().reshape((int(len(s_plasma.gamma().flatten())/3), 3))
            Bnormal = np.sum(B*normal, axis=1).reshape(np.shape(s_plasma.gamma()[:, :, 0]))

            self.assertAlmostEqual(np.sum(Bnormal), 0)
            self.assertAlmostEqual(np.sum(Bnormal_regcoil), 0)

            Bnormal_average = np.mean(np.abs(Bnormal))

            assert np.allclose(Bnormal.flatten()/Bnormal_average, Bnormal_regcoil.flatten()/Bnormal_average)


if __name__ == "__main__":
    unittest.main()
