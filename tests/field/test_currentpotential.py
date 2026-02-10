import unittest
from pathlib import Path

import numpy as np

from simsopt.field import CurrentPotentialFourier
from simsopt.field import CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier
from scipy.io import netcdf_file
from simsopt.util import in_github_actions

TEST_DIR = Path(__file__).parent / ".." / "test_files"

stellsym_list = [True, False]

try:
    import pyevtk  # noqa: F401
    pyevtk_found = True
except ImportError:
    pyevtk_found = False


class CurrentPotentialTests(unittest.TestCase):
    def test_compare_K_with_regcoil(self):
        # axisymmetric case with no Phimnc, non-axisymmetric, axisymmetric with Phimnc
        for filename in ['regcoil_out.w7x.nc', 'regcoil_out.near_axis_asym.nc']:
            filename = TEST_DIR / filename
            cp = CurrentPotentialFourier.from_netcdf(filename)
            cp.set_current_potential_from_regcoil(filename, -1)
            f = netcdf_file(filename, 'r', mmap=False)
            cp_regcoil = f.variables['single_valued_current_potential_mn'][()][-1, :]
            _xm_regcoil = f.variables['xm_potential'][()]
            stellsym = f.variables['symmetry_option'][()]
            if stellsym == 1:
                stellsym = True
            else:
                stellsym = False

            # need to add phic(0, 0) term to cp_regcoil to compare them
            assert np.allclose(cp.get_dofs(), cp_regcoil)
            K2_regcoil = f.variables['K2'][()][-1, :, :]
            K = cp.K()
            K2 = np.sum(K*K, axis=2)
            K2_average = np.mean(K2, axis=(0, 1))
            print(K2[0:int(len(cp.quadpoints_phi)/cp.nfp), :]/K2_average, K2.shape, K2_average)
            print(K2_regcoil[0:int(len(cp.quadpoints_phi)/cp.nfp), :]/K2_average, K2_regcoil.shape)
            assert np.allclose(K2[0:int(len(cp.quadpoints_phi)/cp.nfp), :]/K2_average, K2_regcoil/K2_average)


def get_currentpotential(cptype, stellsym, phis=None, thetas=None):
    np.random.seed(2)
    mpol = 4
    ntor = 3
    nfp = 2
    phis = phis if phis is not None else np.linspace(0, 1, 31, endpoint=False)
    thetas = thetas if thetas is not None else np.linspace(0, 1, 31, endpoint=False)

    if cptype == "CurrentPotentialFourier":
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor,
                                     quadpoints_phi=phis, quadpoints_theta=thetas)
        cp.x = cp.x * 0.
        cp.phis[1, ntor + 0] = 0.3
    else:
        assert False

    dofs = cp.get_dofs()
    np.random.seed(2)
    rand_scale = 0.01
    cp.x = dofs + rand_scale * np.random.rand(len(dofs))  # .reshape(dofs.shape)
    return cp


class CurrentPotentialFourierTests(unittest.TestCase):

    def test_init(self):
        """
        Modeled after test_init in test_surface_rzfourier
        """

        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=2, mpol=3, ntor=2)
        self.assertEqual(cp.phis.shape, (4, 5))
        self.assertEqual(cp.phic.shape, (4, 5))

        cp = CurrentPotentialFourier(s, nfp=10, mpol=1, ntor=3, stellsym=False)
        self.assertEqual(cp.phis.shape, (2, 7))
        self.assertEqual(cp.phic.shape, (2, 7))

    def test_get_dofs(self):
        """
        Modeled after test_get_dofs from test_surface_rzfourier
        Test that we can convert the degrees of freedom into a 1D vector
        """

        # axisymmetric current potential
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=3, ntor=0)
        cp.phis[1, 0] = 0.2
        cp.phis[2, 0] = 1.3
        cp.phis[3, 0] = 0.7
        dofs = cp.get_dofs()
        self.assertEqual(dofs.shape, (3,))
        self.assertAlmostEqual(dofs[0], 0.2)
        self.assertAlmostEqual(dofs[1], 1.3)
        self.assertAlmostEqual(dofs[2], 0.7)

        # non-axisymmetric current potential
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1)
        cp.phis[:, :] = [[100, 101, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        dofs = cp.get_dofs()
        self.assertEqual(dofs.shape, (3*(2*1 + 1) + 1,))
        for j in range(len(dofs)):
            self.assertAlmostEqual(dofs[j], j + 3)

    def test_set_dofs(self):
        """
        Modeled after test_get_dofs from test_surface_rzfourier
        Test that we can set the shape from a 1D vector
        """

        # First try an axisymmetric current potential for simplicity:
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=3, ntor=0)
        cp.set_dofs([-1.1, 0.7, 0.5])
        self.assertAlmostEqual(cp.phis[1, 0], -1.1)
        self.assertAlmostEqual(cp.phis[2, 0], 0.7)
        self.assertAlmostEqual(cp.phis[3, 0], 0.5)

        # Now try a nonaxisymmetric shape:
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1)
        ndofs = len(cp.get_dofs())
        cp.set_dofs(np.array(list(range(ndofs))) + 1)
        self.assertAlmostEqual(cp.phis[0, 0], 0)
        self.assertAlmostEqual(cp.phis[0, 1], 0)
        self.assertAlmostEqual(cp.phis[0, 2], 1)
        self.assertAlmostEqual(cp.phis[1, 0], 2)
        self.assertAlmostEqual(cp.phis[1, 1], 3)
        self.assertAlmostEqual(cp.phis[1, 2], 4)
        self.assertAlmostEqual(cp.phis[2, 0], 5)
        self.assertAlmostEqual(cp.phis[2, 1], 6)
        self.assertAlmostEqual(cp.phis[2, 2], 7)
        self.assertAlmostEqual(cp.phis[3, 0], 8)
        self.assertAlmostEqual(cp.phis[3, 1], 9)
        self.assertAlmostEqual(cp.phis[3, 2], 10)

        # Now try a nonaxisymmetric shape without stell sym
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1, stellsym=False)
        ndofs = len(cp.get_dofs())
        cp.set_dofs(np.array(list(range(ndofs))) + 1)
        self.assertAlmostEqual(cp.phis[0, 0], 0)
        self.assertAlmostEqual(cp.phis[0, 1], 0)
        self.assertAlmostEqual(cp.phis[0, 2], 1)
        self.assertAlmostEqual(cp.phis[1, 0], 2)
        self.assertAlmostEqual(cp.phis[1, 1], 3)
        self.assertAlmostEqual(cp.phis[1, 2], 4)
        self.assertAlmostEqual(cp.phis[2, 0], 5)
        self.assertAlmostEqual(cp.phis[2, 1], 6)
        self.assertAlmostEqual(cp.phis[2, 2], 7)
        self.assertAlmostEqual(cp.phis[3, 0], 8)
        self.assertAlmostEqual(cp.phis[3, 1], 9)
        self.assertAlmostEqual(cp.phis[3, 2], 10)
        self.assertAlmostEqual(cp.phic[0, 0], 0)
        self.assertAlmostEqual(cp.phic[0, 1], 0)
        self.assertAlmostEqual(cp.phic[0, 2], 11)
        self.assertAlmostEqual(cp.phic[1, 0], 12)
        self.assertAlmostEqual(cp.phic[1, 1], 13)
        self.assertAlmostEqual(cp.phic[1, 2], 14)
        self.assertAlmostEqual(cp.phic[2, 0], 15)
        self.assertAlmostEqual(cp.phic[2, 1], 16)
        self.assertAlmostEqual(cp.phic[2, 2], 17)
        self.assertAlmostEqual(cp.phic[3, 0], 18)
        self.assertAlmostEqual(cp.phic[3, 1], 19)
        self.assertAlmostEqual(cp.phic[3, 2], 20)

    def test_change_resolution(self):
        """
        Modeled after test_get_dofs from test_surface_rzfourier
        Check that we can change mpol and ntor.
        """
        for mpol in [1, 2]:
            for ntor in [0, 1]:
                for stellsym in [True, False]:
                    s = SurfaceRZFourier()
                    cp = CurrentPotentialFourier(s, mpol=mpol, ntor=ntor)
                    ndofs = len(cp.get_dofs())
                    cp.set_dofs((np.random.rand(ndofs) - 0.5) * 0.01)
                    cp.set_phis(1, 0, 0.13)
                    phi1 = cp.Phi()

                    cp.change_resolution(mpol+1, ntor)
                    s.recalculate = True
                    phi2 = cp.Phi()
                    self.assertTrue(np.allclose(phi1, phi2))

    def test_get_phic(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=1, ntor=0, stellsym=False)
        cp.x = [0.7, -1.1]
        self.assertAlmostEqual(cp.get_phis(1, 0), 0.7)
        self.assertAlmostEqual(cp.get_phic(1, 0), -1.1)
        self.assertAlmostEqual(cp.get_phis(0, 0), 0.0)
        self.assertAlmostEqual(cp.get_phic(0, 0), 0.0)

    def test_get_phis(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1)
        ndofs = len(cp.get_dofs())
        cp.x = np.array(list(range(ndofs))) + 1

        self.assertAlmostEqual(cp.get_phis(0, -1), 0)
        self.assertAlmostEqual(cp.get_phis(0, 0), 0)
        self.assertAlmostEqual(cp.get_phis(0, 1), 1)
        self.assertAlmostEqual(cp.get_phis(1, -1), 2)
        self.assertAlmostEqual(cp.get_phis(1, 0), 3)
        self.assertAlmostEqual(cp.get_phis(1, 1), 4)
        self.assertAlmostEqual(cp.get_phis(2, -1), 5)
        self.assertAlmostEqual(cp.get_phis(2, 0), 6)
        self.assertAlmostEqual(cp.get_phis(2, 1), 7)
        self.assertAlmostEqual(cp.get_phis(3, -1), 8)
        self.assertAlmostEqual(cp.get_phis(3, 0), 9)
        self.assertAlmostEqual(cp.get_phis(3, 1), 10)

    def test_set_phic(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=1, ntor=0, stellsym=False)
        cp.x = [2.9, -1.1]
        cp.set_phic(1, 0, 3.1)
        self.assertAlmostEqual(cp.x[1], 3.1)
        self.assertAlmostEqual(cp.get_phic(1, 0), 3.1)

    def test_set_phis(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=1, ntor=0, stellsym=True)
        s.x = [2.9, -1.1, 0.7]
        cp.set_phis(1, 0, 1.4)
        self.assertAlmostEqual(cp.x[0], 1.4)

    def test_names_order(self):
        """
        Verify that the order of phis and phic in the dof names is
        correct. This requires that the order of these four arrays in
        ``_make_names()`` matches the order in the C++ functions
        ``set_dofs_impl()`` and ``get_dofs()`` in
        ``src/simsoptpp/currentpotentialfourier.h``.
        """
        mpol = 1
        ntor = 1
        nfp = 4
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        cp.set_phic(0, 1, 100.0)
        cp.set_phis(0, 1, 200.0)
        self.assertAlmostEqual(cp.get('Phic(0,1)'), 100.0)
        self.assertAlmostEqual(cp.get('Phis(0,1)'), 200.0)

    def test_mn(self):
        """
        Test the arrays of mode numbers m and n.
        """
        mpol = 3
        ntor = 2
        nfp = 4
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor)
        m_correct = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        n_correct = [1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
        np.testing.assert_array_equal(cp.m, m_correct)
        np.testing.assert_array_equal(cp.n, n_correct)

        m_correct = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        n_correct = [1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]

        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        np.testing.assert_array_equal(cp.m, m_correct + m_correct)
        np.testing.assert_array_equal(cp.n, n_correct + n_correct)

    def test_mn_matches_names(self):
        """
        Verify that the m and n attributes match the dof names.
        """
        mpol = 2
        ntor = 3
        nfp = 5
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=True)
        names = [name[4:] for name in cp.local_dof_names]
        names2 = [f'({m},{n})' for m, n in zip(cp.m, cp.n)]
        self.assertEqual(names, names2)

        # Now try a non-stellarator-symmetric case:
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        assert 'Phic(0,0)' not in cp.local_dof_names
        names = [name[4:] for name in cp.local_dof_names]
        names2 = [f'({m},{n})' for m, n in zip(cp.m, cp.n)]
        self.assertEqual(names, names2)

    def test_target(self):
        """
        """
        from matplotlib import pyplot as plt
        def run_target_test(
            filename='regcoil_out.hsx.nc', 
            lambda_reg=0.0,
        ):
            """
            Run REGCOIL (L2 regularization) and Lasso (L1 regularization)
            starting from high regularization to low. When fB < fB_target
            is achieved, the algorithms quit. This allows one to compare
            L2 and L1 results at comparable levels of fB, which seems
            like the fairest way to compare them.

            Args:
                filename (str): Path to the REGCOIL netcdf file
                lambda_reg (float): Regularization parameter
            """
            _fB_target = 1e-2
            mpol = 4
            ntor = 4
            coil_ntheta_res = 1
            coil_nzeta_res = coil_ntheta_res
            plasma_ntheta_res = coil_ntheta_res
            plasma_nzeta_res = coil_ntheta_res

            # Load in low-resolution NCSX file from REGCOIL
            cpst = CurrentPotentialSolve.from_netcdf(
                TEST_DIR / filename, plasma_ntheta_res, plasma_nzeta_res, coil_ntheta_res, coil_nzeta_res
            )
            cp = CurrentPotentialFourier.from_netcdf(TEST_DIR / filename, coil_ntheta_res, coil_nzeta_res)

            # Overwrite low-resolution file with more mpol and ntor modes
            cp = CurrentPotentialFourier(
                cpst.winding_surface, mpol=mpol, ntor=ntor,
                net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
                net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
                stellsym=True)
            cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma)  #, cpst.B_GI)

            optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
            cp_opt = cpst.current_potential
            return(cp_opt, cp, cpst, optimized_phi_mn, f_B)


        cp_opt, cp, cpst, optimized_phi_mn, f_B = run_target_test(lambda_reg=0.1)

        theta_study1d, phi_study1d = cp_opt.quadpoints_theta, cp_opt.quadpoints_phi
        theta_study2d, phi_study2d = np.meshgrid(theta_study1d, phi_study1d)

        # Plotting K
        K_mag_study = np.sum(cp_opt.K()**2, axis=2)
        plt.figure()
        plt.pcolor(phi_study1d, theta_study1d, K_mag_study.T, shading='nearest')
        plt.colorbar()
        plt.title('|K|')
        plt.figure()
        plt.pcolor(phi_study2d.T, theta_study2d.T, K_mag_study.T, shading='nearest')
        plt.ylabel(r'$\theta$')
        plt.xlabel(r'$\phi$')
        plt.colorbar()
        plt.title('|K|')

        # Contours for Phi
        Phi_study = cp_opt.Phi()
        plt.figure()
        plt.pcolor(phi_study1d, theta_study1d, Phi_study.T, shading='nearest')
        plt.colorbar()
        plt.ylabel(r'$\theta$')
        plt.xlabel(r'$\phi$')
        plt.title(r'$\Phi$')

        # Contours for normal component
        plt.figure()
        plt.pcolor(cp_opt.winding_surface.normal()[:,:,0].T)
        plt.colorbar()
        plt.ylabel(r'$\theta$')
        plt.xlabel(r'$\phi$')
        plt.title(r'$\Phi_{normal}$')
        if not in_github_actions:
            plt.show()


if __name__ == "__main__":
    unittest.main()
