from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

import simsoptpp as sopp
from simsopt.solve.permanent_magnet_optimization import (
    _connectivity_matrix_py,
    _initialize_GPMO_ArbVec_py,
    prox_l0,
    prox_l1,
    setup_initial_condition,
)
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
            base_curves, curves, coils = initialize_coils_for_pm_optimization('qa', TEST_DIR, s)
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
            errors2, Bn_errors2, m_history2 = GPMO(pm_opt, algorithm='GPMO_ArbVec', **kwargs)
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
            errors4, Bn_errors4, m_history4 = GPMO(pm_opt, algorithm='GPMO_Backtracking', **kwargs)
            m4 = pm_opt.m
            assert np.allclose(m1, m4)
            assert np.allclose(errors1, errors4)
            assert np.allclose(Bn_errors1, Bn_errors4)
            assert np.allclose(m_history1, m_history4)

            # Note: GPMO history arrays contain one additional entry at the beginning
            # for the initialized solution.

            errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='GPMO', **kwargs)
            m5 = pm_opt.m
            assert np.allclose(m1, m5)
            assert np.allclose(errors1, errors5[1:])
            assert np.allclose(Bn_errors1, Bn_errors5[1:])
            assert np.allclose(m_history1, m_history5[:, :, 1:])
            with self.assertRaises(ValueError):
                pm_opt.coordinate_flag = 'cylindrical'
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='GPMO', **kwargs)
            with self.assertRaises(NotImplementedError):
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='random_name', **kwargs)

            kwargs['m_init'] = pm_opt.m.reshape([-1, 3])
            pm_opt.coordinate_flag = 'cartesian'
            errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='GPMO', **kwargs)
            # Note: when K = n_history, m_history[:,:,-1] will be zeros
            assert np.allclose(m_history5[:, :, -2], m_history6[:, :, 0])
            with self.assertRaises(ValueError):
                kwargs['m_init'] = m_history6[:-1, :, -1]
                errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='GPMO', **kwargs)

    def _make_pm_opt_tiny(self, TEST_DIR, nphi=2, ntheta=2, dr=0.05, coff=0.05, poff=0.03, vmec_name='input.LandremanPaul2021_QA_lowres'):
        """
        Tiny helper to build a very small problem instance quickly.

        Args:
            TEST_DIR: Path to the test directory.
            nphi: Number of phi points.
            ntheta: Number of theta points.
            dr: Radial extent of the PM grid.
            coff: Offset from the plasma surface for the inner surface.
            poff: Offset from the plasma surface for the outer surface.
            vmec_name: Name of the VMEC input file.

        Returns:
            s: SurfaceRZFourier object.
            pm_opt: PermanentMagnetGrid object.
        """
        surface_filename = TEST_DIR / vmec_name
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner.extend_via_projected_normal(poff)
        s_outer.extend_via_projected_normal(poff + coff)

        base_curves, curves, coils = initialize_coils_for_pm_optimization('qa', TEST_DIR, s)
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

    def test_gpmo_py_matches_cpp(self):
        """
        Verify the pure-Python GPMO_py implementation matches the C++/sopp backend.
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
            errors_cpp, Bn_cpp, m_hist_cpp = GPMO(pm_opt, algorithm='GPMO', **kwargs)
            m_cpp = pm_opt.m.copy()

            # Python backend, same init
            pm_opt.m = np.zeros_like(pm_opt.m)
            errors_py, Bn_py, m_hist_py = GPMO(pm_opt, algorithm='GPMO_py', **kwargs)
            m_py = pm_opt.m.copy()

            # Exact match (history for these variants includes initial snapshot)
            np.testing.assert_allclose(errors_py, errors_cpp, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(Bn_py, Bn_cpp, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_hist_py, m_hist_cpp, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_py, m_cpp, rtol=1e-12, atol=1e-12)

    def test_initialize_GPMO_ArbVec_py(self):
        """Unit test for _initialize_GPMO_ArbVec_py: maps x_init to nearest pol vectors."""
        N = 4
        nPolVecs = 3
        ngrid = 8
        pol_vectors = np.zeros((N, nPolVecs, 3))
        pol_vectors[:, 0, :] = [1, 0, 0]
        pol_vectors[:, 1, :] = [0, 1, 0]
        pol_vectors[:, 2, :] = [0, 0, 1]

        x_init = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [-0.1, -0.9, 0.0],
        ], dtype=np.float64)

        A_obj = np.random.randn(N, 3, ngrid).astype(np.float64) * 0.01
        b_obj = np.random.randn(ngrid).astype(np.float64) * 0.1

        x = np.zeros((N, 3), dtype=np.float64)
        x_vec = np.zeros(N, dtype=np.int32)
        x_sign = np.zeros(N, dtype=np.int8)
        Aij_mj_sum = -b_obj.copy()
        R2s = np.full(2 * N * nPolVecs, 1e50, dtype=np.float64)
        Gamma_complement = np.ones(N, dtype=bool)
        num_nonzero_ref = [0]

        _initialize_GPMO_ArbVec_py(
            x_init, pol_vectors, x, x_vec, x_sign,
            A_obj, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref,
        )

        self.assertEqual(num_nonzero_ref[0], 3)
        np.testing.assert_allclose(x[0], [1, 0, 0])
        np.testing.assert_allclose(x[1], [1, 0, 0])
        np.testing.assert_allclose(x[2], [0, 0, 0])
        np.testing.assert_allclose(x[3], [0, -1, 0])
        self.assertTrue(Gamma_complement[2])
        self.assertFalse(Gamma_complement[0])
        self.assertFalse(Gamma_complement[1])
        self.assertFalse(Gamma_complement[3])

    def test_gpmo_py_backtracking_removes_pairs(self):
        """
        With small thresh_angle, backtracking removes adjacent pairs.
        Compare runs: small thresh_angle should yield fewer magnets than pi.
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=4, ntheta=4, dr=0.05, coff=0.06, poff=0.03
            )
            ndip = pm_opt.ndipoles

            kwargs_base = {
                "nhistory": 10,
                "K": 30,
                "Nadjacent": 4,
                "dipole_grid_xyz": pm_opt.dipole_grid_xyz,
                "backtracking": 2,
                "max_nMagnets": ndip,
                "verbose": False,
            }
            kwargs_base.update(initialize_default_kwargs('GPMO'))

            kwargs_lo = dict(kwargs_base, thresh_angle=0.4)
            errors_lo, _, _ = GPMO(pm_opt, algorithm='GPMO_py', **kwargs_lo)
            nonzero_lo = np.count_nonzero(np.linalg.norm(pm_opt.m.reshape(-1, 3), axis=1))

            pm_opt.m = np.zeros_like(pm_opt.m)
            kwargs_hi = dict(kwargs_base, thresh_angle=np.pi)
            errors_hi, _, _ = GPMO(pm_opt, algorithm='GPMO_py', **kwargs_hi)
            nonzero_hi = np.count_nonzero(np.linalg.norm(pm_opt.m.reshape(-1, 3), axis=1))

            self.assertLess(nonzero_lo, nonzero_hi, msg="Small thresh_angle should remove more magnets")
            self.assertLess(len(errors_lo), len(errors_hi) + 2)

    def test_gpmo_vs_gpmo_py_nontrivial(self):
        """GPMO and GPMO_py produce the same result on a nontrivial instance."""
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=6, ntheta=6, dr=0.04, coff=0.07, poff=0.04
            )

            kwargs = initialize_default_kwargs('GPMO')
            kwargs['nhistory'] = 8
            kwargs['K'] = 25
            kwargs['Nadjacent'] = 3
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            kwargs['backtracking'] = 5
            kwargs['max_nMagnets'] = 20
            kwargs['thresh_angle'] = np.pi
            kwargs['verbose'] = False

            errors_cpp, _, _ = GPMO(pm_opt, algorithm='GPMO', **kwargs)
            m_cpp = pm_opt.m.copy()

            pm_opt.m = np.zeros_like(pm_opt.m)
            errors_py, _, _ = GPMO(pm_opt, algorithm='GPMO_py', **kwargs)
            m_py = pm_opt.m.copy()

            np.testing.assert_allclose(m_py, m_cpp, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(errors_py, errors_cpp, rtol=1e-11, atol=1e-11)

    def test_connectivity_matrix_py(self):
        """_connectivity_matrix_py returns correct shape and self-inclusion."""
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]], dtype=np.float64)
        conn = _connectivity_matrix_py(xyz, Nadjacent=3)
        self.assertEqual(conn.shape, (4, 3))
        for i in range(4):
            self.assertEqual(conn[i, 0], i)

    def test_gpmomr_smoke(self):
        """
        MacroMag-driven GPMOmr: massively downsampled test.

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
            errors_mm, Bn_mm, m_hist_mm = GPMO(pm_opt, algorithm='GPMOmr', **kwargs)
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

    def test_gpmomr_final_fb_matches_recompute(self):
        """
        Regression test for GPMOmr logging/return consistency:

        - Run a very small GPMOmr instance that stops early via max_nMagnets.
        - Verify the logged final f_B equals recomputing f_B from the returned pm_opt.m
          on the same objective grid (pm_opt.A_obj / pm_opt.b_obj).

        This guards against mismatches between the last recorded objective and the
        final magnetization state returned by the solver.
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            nphi = 8
            ntheta = 8
            dr = 0.05
            coff = 0.06
            poff = 0.03

            surface_filename = TEST_DIR / "input.LandremanPaul2021_QA_lowres"
            s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_inner.extend_via_projected_normal(poff)
            s_outer.extend_via_projected_normal(poff + coff)

            # Avoid expensive coil setup; any deterministic Bnormal is sufficient for this consistency check.
            rng = np.random.default_rng(1)
            Bnormal = rng.standard_normal((nphi, ntheta))

            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, Bnormal, s_inner, s_outer, **{"dr": dr}
            )
            setup_initial_condition(pm_opt, np.zeros(pm_opt.ndipoles * 3))

            # Deterministic Cartesian {x,y,z} polarization set so the greedy path is repeatable.
            ndip = pm_opt.ndipoles
            pvx = np.zeros((ndip, 3)); pvx[:, 0] = 1.0
            pvy = np.zeros((ndip, 3)); pvy[:, 1] = 1.0
            pvz = np.zeros((ndip, 3)); pvz[:, 2] = 1.0
            pm_opt.pol_vectors = np.transpose(np.array([pvx, pvy, pvz]), (1, 0, 2))

            max_nMagnets = 20
            kwargs = initialize_default_kwargs("GPMO")
            kwargs.update({
                "K": 30,  # > max_nMagnets so we trigger the early-stop final snapshot
                "nhistory": 5,
                "Nadjacent": 1,
                "dipole_grid_xyz": np.ascontiguousarray(pm_opt.dipole_grid_xyz),
                "backtracking": 0,
                "max_nMagnets": max_nMagnets,
                "thresh_angle": np.pi,
                "verbose": False,  # rely on the forced early-stop snapshot for logging
                "mm_refine_every": 5,  # keep it cheap, but ensures caches exist
                "use_coils": False,
            })

            errors, _, _ = GPMO(pm_opt, algorithm="GPMOmr", **kwargs)
            self.assertGreater(len(errors), 0)

            # k_history should reflect the actual early-stop iteration, not a nominal schedule.
            self.assertTrue(hasattr(pm_opt, "k_history"))
            self.assertEqual(len(pm_opt.k_history), len(errors))
            self.assertEqual(int(pm_opt.k_history[-1]), max_nMagnets + 1)

            # num_nonzeros should report the active count at the final snapshot.
            self.assertTrue(hasattr(pm_opt, "num_nonzeros"))
            self.assertEqual(int(pm_opt.num_nonzeros[-1]), max_nMagnets)

            # Recompute f_B from the returned dipole moments on the same grid:
            # f_B = 0.5 * ||A m - b||^2 (matches the objective used in the greedy routines).
            res = pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj
            fb_recomputed = 0.5 * float(res @ res)
            fb_logged = float(errors[-1])

            np.testing.assert_allclose(
                fb_logged,
                fb_recomputed,
                rtol=1e-12,
                atol=max(1e-12, 1e-12 * max(1.0, abs(fb_recomputed))),
            )
            
    def test_macromag_finite_mu_consistency(self):
        """
        Consistency check:
        With cube side = 4 mm and mu_ea=mu_oa=1 (i.e. a isotropic case), MacroMag driven
        GPMOmr must match the non-MacroMag GPMO result,
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

            errors_ref, Bn_ref, m_hist_ref = GPMO(pm_opt, algorithm='GPMO', **base_kwargs)
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
            errors_mm, Bn_mm, m_hist_mm = GPMO(pm_opt, algorithm='GPMOmr', **mm_kwargs)
            m_mm = pm_opt.m.copy()

            np.testing.assert_allclose(errors_mm,errors_ref,rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(Bn_mm,Bn_ref,rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_hist_mm,m_hist_ref,rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m_mm,m_ref,rtol=1e-12, atol=1e-12)

            # just a quick check to veify if at least one magnet is placed (unless the residual was already aprox. 0)
            if errors_ref[0] > 1e-12:
                self.assertGreater(np.count_nonzero(np.linalg.norm(m_ref.reshape(-1, 3), axis=1)), 0)

    # ------------------------------------------------------------------
    # Enhanced _initialize_GPMO_ArbVec_py tests
    # ------------------------------------------------------------------

    def test_initialize_GPMO_ArbVec_py_all_zero(self):
        """All-zero x_init should place zero magnets and leave buffers untouched."""
        N, nPolVecs, ngrid = 5, 3, 6
        pol_vectors = np.zeros((N, nPolVecs, 3))
        pol_vectors[:, 0, :] = [1, 0, 0]
        pol_vectors[:, 1, :] = [0, 1, 0]
        pol_vectors[:, 2, :] = [0, 0, 1]

        x_init = np.zeros((N, 3), dtype=np.float64)
        A_obj = np.random.default_rng(0).standard_normal((N, 3, ngrid)).astype(np.float64)
        b_obj = np.random.default_rng(1).standard_normal(ngrid).astype(np.float64)

        x = np.zeros((N, 3), dtype=np.float64)
        x_vec = np.zeros(N, dtype=np.int32)
        x_sign = np.zeros(N, dtype=np.int8)
        Aij_mj_sum_orig = -b_obj.copy()
        Aij_mj_sum = Aij_mj_sum_orig.copy()
        R2s = np.full(2 * N * nPolVecs, 1e50, dtype=np.float64)
        R2s_orig = R2s.copy()
        Gamma_complement = np.ones(N, dtype=bool)
        num_nonzero_ref = [0]

        _initialize_GPMO_ArbVec_py(
            x_init, pol_vectors, x, x_vec, x_sign,
            A_obj, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref,
        )

        self.assertEqual(num_nonzero_ref[0], 0)
        np.testing.assert_array_equal(x, 0.0)
        np.testing.assert_array_equal(Gamma_complement, True)
        np.testing.assert_allclose(Aij_mj_sum, Aij_mj_sum_orig)
        np.testing.assert_allclose(R2s, R2s_orig)

    def test_initialize_GPMO_ArbVec_py_negative_pol(self):
        """x_init matching a negative pol vector should assign sign=-1."""
        N, nPolVecs, ngrid = 2, 2, 4
        pol_vectors = np.zeros((N, nPolVecs, 3))
        pol_vectors[:, 0, :] = [1, 0, 0]
        pol_vectors[:, 1, :] = [0, 1, 0]

        x_init = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)

        A_obj = np.random.default_rng(7).standard_normal((N, 3, ngrid)).astype(np.float64)
        b_obj = np.random.default_rng(8).standard_normal(ngrid).astype(np.float64)

        x = np.zeros((N, 3), dtype=np.float64)
        x_vec = np.zeros(N, dtype=np.int32)
        x_sign = np.zeros(N, dtype=np.int8)
        Aij_mj_sum = -b_obj.copy()
        R2s = np.full(2 * N * nPolVecs, 1e50, dtype=np.float64)
        Gamma_complement = np.ones(N, dtype=bool)
        num_nonzero_ref = [0]

        _initialize_GPMO_ArbVec_py(
            x_init, pol_vectors, x, x_vec, x_sign,
            A_obj, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref,
        )

        self.assertEqual(num_nonzero_ref[0], 2)
        np.testing.assert_allclose(x[0], [-1, 0, 0])
        np.testing.assert_allclose(x[1], [0, -1, 0])
        self.assertEqual(x_sign[0], -1)
        self.assertEqual(x_sign[1], -1)
        self.assertEqual(x_vec[0], 0)
        self.assertEqual(x_vec[1], 1)

    def test_initialize_GPMO_ArbVec_py_residual_and_R2s(self):
        """Verify Aij_mj_sum and R2s are correctly updated after initialization."""
        N, nPolVecs, ngrid = 3, 2, 5
        rng = np.random.default_rng(42)
        pol_vectors = np.zeros((N, nPolVecs, 3))
        pol_vectors[:, 0, :] = [1, 0, 0]
        pol_vectors[:, 1, :] = [0, 1, 0]

        x_init = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)

        A_obj = rng.standard_normal((N, 3, ngrid)).astype(np.float64)
        b_obj = rng.standard_normal(ngrid).astype(np.float64)

        x = np.zeros((N, 3), dtype=np.float64)
        x_vec = np.zeros(N, dtype=np.int32)
        x_sign = np.zeros(N, dtype=np.int8)
        Aij_mj_sum = -b_obj.copy()
        R2s = np.full(2 * N * nPolVecs, 1e50, dtype=np.float64)
        Gamma_complement = np.ones(N, dtype=bool)
        num_nonzero_ref = [0]

        _initialize_GPMO_ArbVec_py(
            x_init, pol_vectors, x, x_vec, x_sign,
            A_obj, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref,
        )

        # Manually recompute expected residual
        expected_resid = -b_obj.copy()
        for j in range(N):
            if x_sign[j] != 0:
                pv = pol_vectors[j, x_vec[j], :]
                expected_resid += x_sign[j] * (pv[:, None] * A_obj[j]).sum(axis=0)
        np.testing.assert_allclose(Aij_mj_sum, expected_resid, atol=1e-14)

        # R2s should be 1e50 for placed sites, unchanged for available sites
        NNp = N * nPolVecs
        for j in range(N):
            base = j * nPolVecs
            if not Gamma_complement[j]:
                np.testing.assert_array_equal(R2s[base:base + nPolVecs], 1e50)
                np.testing.assert_array_equal(R2s[NNp + base:NNp + base + nPolVecs], 1e50)

    # ------------------------------------------------------------------
    # Backtracking with large thresh_angle
    # ------------------------------------------------------------------

    def test_gpmo_py_backtracking_large_thresh_angle(self):
        """
        With an intermediate thresh_angle (~2.0 radians), backtracking should
        remove some but not all pairs, yielding a magnet count between the
        small-threshold and no-removal (pi) cases.
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=4, ntheta=4, dr=0.05, coff=0.06, poff=0.03
            )
            ndip = pm_opt.ndipoles

            kwargs_base = {
                "nhistory": 10,
                "K": 30,
                "Nadjacent": 4,
                "dipole_grid_xyz": pm_opt.dipole_grid_xyz,
                "backtracking": 2,
                "max_nMagnets": ndip,
                "verbose": False,
            }
            kwargs_base.update(initialize_default_kwargs('GPMO'))

            # No removal: thresh_angle = pi
            kwargs_pi = dict(kwargs_base, thresh_angle=np.pi)
            GPMO(pm_opt, algorithm='GPMO_py', **kwargs_pi)
            nonzero_pi = np.count_nonzero(
                np.linalg.norm(pm_opt.m.reshape(-1, 3), axis=1)
            )

            # Aggressive removal: thresh_angle = 0.4
            pm_opt.m = np.zeros_like(pm_opt.m)
            kwargs_lo = dict(kwargs_base, thresh_angle=0.4)
            GPMO(pm_opt, algorithm='GPMO_py', **kwargs_lo)
            nonzero_lo = np.count_nonzero(
                np.linalg.norm(pm_opt.m.reshape(-1, 3), axis=1)
            )

            # Intermediate: thresh_angle = 2.0 (large enough to cause some removal)
            pm_opt.m = np.zeros_like(pm_opt.m)
            kwargs_mid = dict(kwargs_base, thresh_angle=2.0)
            GPMO(pm_opt, algorithm='GPMO_py', **kwargs_mid)
            nonzero_mid = np.count_nonzero(
                np.linalg.norm(pm_opt.m.reshape(-1, 3), axis=1)
            )

            # Monotonicity: more aggressive threshold -> fewer magnets
            self.assertLessEqual(nonzero_lo, nonzero_mid)
            self.assertLessEqual(nonzero_mid, nonzero_pi)
            # At least one magnet placed in the intermediate case
            self.assertGreater(nonzero_mid, 0)

    # ------------------------------------------------------------------
    # GPMO vs GPMO_py with active backtracking
    # ------------------------------------------------------------------

    def test_gpmo_vs_gpmo_py_with_backtracking(self):
        """
        GPMO (C++) and GPMO_py must agree even when backtracking is active
        with a non-trivial thresh_angle.
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=4, ntheta=4, dr=0.05, coff=0.06, poff=0.03
            )

            kwargs = initialize_default_kwargs('GPMO')
            kwargs['nhistory'] = 6
            kwargs['K'] = 20
            kwargs['Nadjacent'] = 2
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            kwargs['backtracking'] = 5
            kwargs['max_nMagnets'] = 20
            kwargs['thresh_angle'] = np.pi
            kwargs['verbose'] = False

            errors_cpp, Bn_cpp, m_hist_cpp = GPMO(pm_opt, algorithm='GPMO', **kwargs)
            m_cpp = pm_opt.m.copy()

            pm_opt.m = np.zeros_like(pm_opt.m)
            errors_py, Bn_py, m_hist_py = GPMO(pm_opt, algorithm='GPMO_py', **kwargs)
            m_py = pm_opt.m.copy()

            np.testing.assert_allclose(errors_py, errors_cpp, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(Bn_py, Bn_cpp, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(m_py, m_cpp, rtol=1e-11, atol=1e-11)

    def test_gpmo_vs_gpmo_py_different_nadjacent(self):
        """
        GPMO (C++) and GPMO_py must agree with Nadjacent=4 (larger neighbourhood).
        """
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=4, ntheta=4, dr=0.05, coff=0.06, poff=0.03
            )

            kwargs = initialize_default_kwargs('GPMO')
            kwargs['nhistory'] = 5
            kwargs['K'] = 18
            kwargs['Nadjacent'] = 4
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            kwargs['backtracking'] = 6
            kwargs['max_nMagnets'] = 25
            kwargs['thresh_angle'] = np.pi
            kwargs['verbose'] = False

            errors_cpp, _, _ = GPMO(pm_opt, algorithm='GPMO', **kwargs)
            m_cpp = pm_opt.m.copy()

            pm_opt.m = np.zeros_like(pm_opt.m)
            errors_py, _, _ = GPMO(pm_opt, algorithm='GPMO_py', **kwargs)
            m_py = pm_opt.m.copy()

            np.testing.assert_allclose(errors_py, errors_cpp, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(m_py, m_cpp, rtol=1e-11, atol=1e-11)

    # ------------------------------------------------------------------
    # Enhanced connectivity matrix tests
    # ------------------------------------------------------------------

    def test_connectivity_matrix_py_neighbors(self):
        """Verify nearest neighbors on a known grid."""
        xyz = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [10, 10, 10],
        ], dtype=np.float64)

        conn = _connectivity_matrix_py(xyz, Nadjacent=2)
        self.assertEqual(conn.shape, (4, 2))
        # Self is always first
        for i in range(4):
            self.assertEqual(conn[i, 0], i)
        # For point 0, nearest non-self neighbor is 1 or 2 (both distance 1)
        self.assertIn(conn[0, 1], [1, 2])
        # For the isolated point 3, nearest neighbor is one of 0,1,2
        self.assertIn(conn[3, 1], [0, 1, 2])

    def test_connectivity_matrix_py_nadjacent_1(self):
        """Nadjacent=1 should return only self-connections."""
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        conn = _connectivity_matrix_py(xyz, Nadjacent=1)
        self.assertEqual(conn.shape, (3, 1))
        for i in range(3):
            self.assertEqual(conn[i, 0], i)

    # ------------------------------------------------------------------
    # Timing: GPMO (C++) should be faster than GPMO_py (Python)
    # ------------------------------------------------------------------

    def test_gpmo_cpp_faster_than_py(self):
        """
        Verify that the C++ GPMO backend is faster than the pure-Python
        GPMO_py reference implementation on the same problem instance.
        """
        import time

        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        with ScratchDir("."):
            _, pm_opt = self._make_pm_opt_tiny(
                TEST_DIR, nphi=4, ntheta=4, dr=0.05, coff=0.06, poff=0.03
            )

            kwargs = initialize_default_kwargs('GPMO')
            kwargs['nhistory'] = 5
            kwargs['K'] = 20
            kwargs['Nadjacent'] = 2
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            kwargs['backtracking'] = 5
            kwargs['max_nMagnets'] = 20
            kwargs['thresh_angle'] = np.pi
            kwargs['verbose'] = False

            t0 = time.perf_counter()
            GPMO(pm_opt, algorithm='GPMO', **kwargs)
            t_cpp = time.perf_counter() - t0

            pm_opt.m = np.zeros_like(pm_opt.m)

            t0 = time.perf_counter()
            GPMO(pm_opt, algorithm='GPMO_py', **kwargs)
            t_py = time.perf_counter() - t0

            self.assertLess(
                t_cpp, t_py,
                msg=f"C++ GPMO ({t_cpp:.4f}s) should be faster than "
                    f"Python GPMO_py ({t_py:.4f}s)"
            )


if __name__ == "__main__":
    unittest.main()
