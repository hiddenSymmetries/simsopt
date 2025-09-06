from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

import simsoptpp as sopp
from simsopt.solve.permanent_magnet_optimization import prox_l0, prox_l1
from simsopt.solve.permanent_magnet_optimization import setup_initial_condition
from simsopt.solve import relax_and_split, GPMO
from simsopt.util import *
from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
from simsopt.field import BiotSavart


class Testing(unittest.TestCase):

    def test_prox(self):
        m = np.random.rand(3000)
        mmax = np.ones(1000)
        reg_l0 = 0.5
        nu = 0.5
        m_thresholded = prox_l0(m, mmax, reg_l0, nu)
        m_thresholded = m_thresholded[~np.isclose(m_thresholded, 0.0)]
        assert np.all(m_thresholded >= 0.5)
        nu = 1
        m_thresholded = prox_l1(m, mmax, reg_l0, nu)
        assert np.linalg.norm(m_thresholded) < np.linalg.norm(m)

    def test_MwPGP(self):
        """ 
            Test the MwPGP algorithm for solving the convex
            part of the permanent magnet problem. 
        """
        ndipoles = 100
        nquad = 512
        max_iter = 100
        m_maxima = np.random.rand(ndipoles) * 10
        m0 = np.zeros((ndipoles, 3))
        b = np.random.rand(nquad)
        A = np.random.rand(nquad, ndipoles, 3)
        ATA = np.tensordot(A, A, axes=([1, 1]))
        alpha = 2.0 / np.linalg.norm(ATA.reshape(nquad * 3, nquad * 3), ord=2)
        ATb = np.tensordot(A, b, axes=([0, 0]))
        with ScratchDir("."):
            MwPGP_hist, RS_hist, m_hist, dipoles = sopp.MwPGP_algorithm(
                A_obj=A, b_obj=b, ATb=ATb, m_proxy=m0, m0=m0, m_maxima=m_maxima,
                alpha=alpha, nu=1e100, epsilon=1e-4, max_iter=max_iter,  # verbose=True,
                reg_l0=0.0, reg_l1=0.0, reg_l2=0.0)
            m_hist = np.array(m_hist)
            assert dipoles.shape == (ndipoles, 3)
            assert m_hist.shape == (ndipoles, 3, 21)

    def test_algorithms(self):
        """ 
            Test the relax and split algorithm for solving
            the permanent magnet problem. Test the GPMO
            algorithm variants in the limit that they should
            all produce the same solution.
        """
        nphi = 8  # nphi = ntheta >= 64 needed for accurate full-resolution runs
        ntheta = 8
        dr = 0.04  # cylindrical bricks with radial extent 4 cm
        coff = 0.1  # PM grid starts offset ~ 10 cm from the plasma surface
        poff = 0.05  # PM grid end offset ~ 15 cm from the plasma surface
        input_name = 'input.LandremanPaul2021_QA_lowres'

        # Read in the plasma equilibrium file
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

        # Make the inner and outer surfaces by extending the plasma surface
        s_inner.extend_via_projected_normal(poff)
        s_outer.extend_via_projected_normal(poff + coff)

        # optimize the currents in the TF coils
        with ScratchDir("."):
            base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s)
            bs = BiotSavart(coils)
            bs = coil_optimization(s, bs, base_curves, curves)
            bs.set_points(s.gamma().reshape((-1, 3)))
            Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

            kwargs_geo = {"dr": dr}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, Bnormal, s_inner, s_outer, **kwargs_geo
            )
            setup_initial_condition(pm_opt, np.zeros(pm_opt.ndipoles * 3))

            reg_l0 = 0.05  # Threshold off magnets with 5% or less strength
            nu = 1e10  # how strongly to make proxy variable w close to values in m

            # Rescale the hyperparameters and then add contributions to ATA and ATb
            reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu)

            # Set some hyperparameters for the optimization
            kwargs = initialize_default_kwargs()
            kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
            kwargs['max_iter'] = 40  # Number of iterations to take in a convex step
            kwargs['max_iter_RS'] = 20  # Number of total iterations of the relax-and-split algorithm
            kwargs['reg_l0'] = reg_l0
            relax_and_split(pm_opt, **kwargs)
            w = pm_opt.m_proxy[~np.isclose(pm_opt.m_proxy, 0.0)]
            assert np.all(np.abs(w) >= reg_l0 * pm_opt.m_maxima[0])

            # Try again with more aggressive thresholding
            reg_l0 = 0.5  # Threshold off magnets with 50% or less strength
            nu = 1e10  # how strongly to make proxy variable w close to values in m

            # Rescale the hyperparameters and then add contributions to ATA and ATb
            reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu)

            # Set some hyperparameters for the optimization
            kwargs = initialize_default_kwargs()
            kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
            kwargs['reg_l0'] = reg_l0
            relax_and_split(pm_opt, **kwargs)
            w = pm_opt.m_proxy[~np.isclose(pm_opt.m_proxy, 0.0)]
            assert np.all(np.abs(w) >= reg_l0 * pm_opt.m_maxima[0])
            kwargs['reg_l1'] = reg_l0
            with self.assertRaises(ValueError):
                relax_and_split(pm_opt, **kwargs)
            kwargs['reg_l0'] = 0.0
            kwargs['epsilon_RS'] = 1e5
            relax_and_split(pm_opt, **kwargs)

            # Test that all the GPMO variants return the same solutions
            # in various limits.
            kwargs = initialize_default_kwargs('GPMO')
            with self.assertRaises(ValueError):
                GPMO(pm_opt, algorithm='baseline', **kwargs)
            kwargs['nhistory'] = 10
            kwargs['K'] = 10
            errors1, Bn_errors1, m_history1 = GPMO(pm_opt, algorithm='baseline', **kwargs)
            m1 = pm_opt.m
            ndipoles = pm_opt.ndipoles
            pol_vector_x = np.zeros((ndipoles, 3))
            pol_vector_x[:, 0] = 1.0
            pol_vector_y = np.zeros((ndipoles, 3))
            pol_vector_y[:, 1] = 1.0
            pol_vector_z = np.zeros((ndipoles, 3))
            pol_vector_z[:, 2] = 1.0
            pol_vectors = np.transpose(np.array([pol_vector_x, pol_vector_y, pol_vector_z]), [1, 0, 2])
            pm_opt.pol_vectors = pol_vectors
            errors2, Bn_errors2, m_history2 = GPMO(pm_opt, algorithm='ArbVec', **kwargs)
            m2 = pm_opt.m
            assert np.allclose(m1, m2)
            assert np.allclose(errors1, errors2)
            assert np.allclose(Bn_errors1, Bn_errors2)
            assert np.allclose(m_history1, m_history2)
            kwargs['Nadjacent'] = 1
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            errors3, Bn_errors3, m_history3 = GPMO(pm_opt, algorithm='multi', **kwargs)
            m3 = pm_opt.m
            assert np.allclose(m1, m3)
            assert np.allclose(errors1, errors3)
            assert np.allclose(Bn_errors1, Bn_errors3)
            assert np.allclose(m_history1, m_history3)
            kwargs['backtracking'] = 500
            kwargs['max_nMagnets'] = 1000
            errors4, Bn_errors4, m_history4 = GPMO(pm_opt, algorithm='backtracking', **kwargs)
            m4 = pm_opt.m
            assert np.allclose(m1, m4)
            assert np.allclose(errors1, errors4)
            assert np.allclose(Bn_errors1, Bn_errors4)
            assert np.allclose(m_history1, m_history4)

            # Note: ArbVec_backtracking history arrays contain one additional
            # entry at the beginning for the initialized solution

            errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            m5 = pm_opt.m
            assert np.allclose(m1, m5)
            assert np.allclose(errors1, errors5[1:])
            assert np.allclose(Bn_errors1, Bn_errors5[1:])
            assert np.allclose(m_history1, m_history5[:, :, 1:])
            with self.assertRaises(ValueError):
                pm_opt.coordinate_flag = 'cylindrical'
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            with self.assertRaises(NotImplementedError):
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='random_name', **kwargs)

            kwargs['m_init'] = pm_opt.m.reshape([-1, 3])
            pm_opt.coordinate_flag = 'cartesian'
            errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            # Note: when K = n_history, m_history[:,:,-1] will be zeros
            assert np.allclose(m_history5[:, :, -2], m_history6[:, :, 0])
            with self.assertRaises(ValueError):
                kwargs['m_init'] = m_history6[:-1, :, -1]
                errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)

    def _make_pm_opt_tiny(self, TEST_DIR, nphi=2, ntheta=2, dr=0.05, coff=0.05, poff=0.03, vmec_name='input.LandremanPaul2021_QA_lowres'):
        """
        Tiny helper to build a very small problem instance quickly.
        """
        surface_filename = TEST_DIR / vmec_name
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner.extend_via_projected_normal(poff)
        s_outer.extend_via_projected_normal(poff + coff)

        base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s)
        bs = BiotSavart(coils)
        bs = coil_optimization(s, bs, base_curves, curves)
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

        pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
            s, Bnormal, s_inner, s_outer, **{"dr": dr}
        )
        setup_initial_condition(pm_opt, np.zeros(pm_opt.ndipoles * 3))
        # Use simple Cartesian {x,y,z} polarization set so the greedy path is deterministic:
        ndip = pm_opt.ndipoles
        pvx = np.zeros((ndip, 3)); pvx[:, 0] = 1.0
        pvy = np.zeros((ndip, 3)); pvy[:, 1] = 1.0
        pvz = np.zeros((ndip, 3)); pvz[:, 2] = 1.0
        pm_opt.pol_vectors = np.transpose(np.array([pvx, pvy, pvz]), (1, 0, 2))
        return s, pm_opt

    def test_arbvec_backtracking_py_matches_cpp(self):
        """
        Verify the pure-Python ArbVec_backtracking implementation matches the C++/sopp backend.
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(TEST_DIR, nphi=4, ntheta=4, dr=0.05, coff=0.06, poff=0.03)

            # Shared kwargs (small problem so it runs fast)
            kwargs = initialize_default_kwargs('GPMO')
            kwargs['nhistory'] = 5
            kwargs['K'] = 20
            kwargs['Nadjacent'] = 1
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            kwargs['backtracking'] = 10
            kwargs['max_nMagnets'] = 30
            kwargs['thresh_angle'] = np.pi

            # C++ backend
            errors_cpp, Bn_cpp, m_hist_cpp = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            m_cpp = pm_opt.m.copy()

            # Python backend, same init
            pm_opt.m = np.zeros_like(pm_opt.m)
            errors_py, Bn_py, m_hist_py = GPMO(pm_opt, algorithm='ArbVec_backtracking_py', **kwargs)
            m_py = pm_opt.m.copy()

            # Exact match (history for ArbVec_backtracking variants includes initial snapshot)
            np.testing.assert_allclose(errors_py, errors_cpp, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(Bn_py, Bn_cpp, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_hist_py, m_hist_cpp, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_py, m_cpp, rtol=1e-12, atol=1e-12)


    def test_arbvec_backtracking_macromag_smoke(self):
        """
        MacroMag-driven ArbVec_backtracking: massively downsampled test.

        Verifies:
        - The run completes on a tiny instance.
        - The magnetization history tensor has shape (ndipoles, 3, >= nhistory+1)
        - With m_init = 0, the initial objective equals 0.5‖b‖^2 (b = pm_opt.b_obj)
        - If the initial residual is non-tiny (> 1e-12), the objective strictly decreases
        - If the initial residual is tiny (<= 1e-12), the final objective remains tiny (<= 1e-6)..
        - When improvement is possible, at least one magnet is placed; in all cases
        the number of placed magnets is capped by max_nMagnets (4 here)
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            # Tiny instance to keep demag tensor assembly cheap.... 
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=2, ntheta=2, dr=0.08, coff=0.08, poff=0.04
            )
            kwargs = initialize_default_kwargs('GPMO')
            kwargs['nhistory'] = 3
            kwargs['K'] = 8
            kwargs['Nadjacent'] = 1
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            kwargs['backtracking'] = 4
            kwargs['max_nMagnets'] = 4
            kwargs['thresh_angle'] = np.pi
            kwargs['m_init'] = np.zeros((pm_opt.ndipoles, 3))
            kwargs['verbose'] = False               
            kwargs['mm_refine_every'] = 1 
            kwargs['use_coils'] = False
            errors_mm, Bn_mm, m_hist_mm = GPMO(pm_opt, algorithm='ArbVec_backtracking_macromag_py', **kwargs)
            m_mm = pm_opt.m.copy()

            # Basic shape checks
            self.assertEqual(m_hist_mm.shape[0], pm_opt.ndipoles)
            self.assertEqual(m_hist_mm.shape[1], 3)
            self.assertGreaterEqual(m_hist_mm.shape[2], kwargs['nhistory'] + 1)  # includes initial snapshot

            # Initial objective equals 0.5||b||^2 (since m=0)
            b = pm_opt.b_obj
            init_R2 = 0.5 * float(b @ b)
            self.assertAlmostEqual(
                errors_mm[0],
                init_R2,
                delta=max(1e-12, 1e-6 * max(1.0, init_R2))
            )

            # If the initial residual is not tiny, require improvement; otherwise just require it stays tiny
            tiny_tol = 1e-12
            if init_R2 > tiny_tol:
                self.assertLess(errors_mm[-1], errors_mm[0])
            else:
                self.assertLessEqual(errors_mm[-1], 1e-6)

            # Magnet count: require at least 1 only when improvement is possible; always enforce the cap
            nonzero = np.count_nonzero(np.linalg.norm(m_mm.reshape(-1, 3), axis=1))
            if init_R2 > tiny_tol:
                self.assertGreaterEqual(nonzero, 1)
            self.assertLessEqual(nonzero, kwargs['max_nMagnets'])
            
    def test_macromag_finite_mu_consistency(self):
        """
        Consistency check:
        With cube side = 4 mm and mu_ea=mu_oa=1 (i.e. a isotropic case), MacroMag driven
        ArbVec+backtracking must match the non-Macromag enhanced ArbVec+backtracking result,
        provided we set the GPMO 'm_maxima' to the physical full-strength
        moment m_max = M_rem * V (with V = cube_dim^3).
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=4, ntheta=4, dr=0.05, coff=0.06, poff=0.03
            )

            cube_dim = 0.004  # 4 mm
            V = cube_dim ** 3
            mu0 = 4.0 * np.pi * 1e-7
            B_max = 1.465
            M_rem = B_max / mu0  # A/m
            m_max = M_rem * V    # A·m^2 full-strength dipole for each site

            # Force GPMO to use this exact full-strength magnitude at every site
            pm_opt.m_maxima[:] = m_max

            base_kwargs = initialize_default_kwargs('GPMO')
            base_kwargs.update({
                'nhistory': 6,
                'K': 24,
                'Nadjacent': 1,
                'dipole_grid_xyz': pm_opt.dipole_grid_xyz,
                'backtracking': 6,
                'max_nMagnets': 30,
                'thresh_angle': np.pi,
                'm_init': np.zeros((pm_opt.ndipoles, 3)),
                'verbose': False,
            })

            errors_ref, Bn_ref, m_hist_ref = GPMO(pm_opt, algorithm='ArbVec_backtracking', **base_kwargs)
            m_ref = pm_opt.m.copy()

            pm_opt.m = np.zeros_like(pm_opt.m)
            pm_opt.m_proxy = np.zeros_like(pm_opt.m)

            mm_kwargs = dict(base_kwargs)
            mm_kwargs.update({
                'cube_dim': cube_dim,
                'mu_ea': 1.0,
                'mu_oa': 1.0,
                'mm_refine_every': 1,   
                'use_coils': False, 
            })
            errors_mm, Bn_mm, m_hist_mm = GPMO(pm_opt, algorithm='ArbVec_backtracking_macromag_py', **mm_kwargs)
            m_mm = pm_opt.m.copy()

            np.testing.assert_allclose(errors_mm,errors_ref,rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(Bn_mm,Bn_ref,rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_hist_mm,m_hist_ref,rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_mm,m_ref,rtol=1e-12, atol=1e-12)

            # just a quick check to veify if at least one magnet is placed (unless the residual was already aprox. 0)
            if errors_ref[0] > 1e-12:
                self.assertGreater(np.count_nonzero(np.linalg.norm(m_ref.reshape(-1, 3), axis=1)), 0)



if __name__ == "__main__":
    unittest.main()
