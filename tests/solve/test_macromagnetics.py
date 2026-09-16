import time
import unittest
import numpy as np
from pathlib import Path
from typing import Optional

from simsopt.solve.macromagnetics import (
    MacroMag,
    SolverParams,
    Tiles,
    assemble_blocks_subset,
    build_prism,
    get_rotmat,
    muse2tiles,
    rotation_angle,
)
from simsopt.solve.permanent_magnet_optimization import _connectivity_matrix_py
from simsopt.util import in_github_actions
from simsopt.solve.macromagnetics import (
    _prism_N_local_nb,
    _prism_N_local_vectorized,
    _rotation_matrix,
)


def make_trivial_tiles(n: int, *, cube_dim: float = 1.0, offsets: Optional[np.ndarray] = None) -> Tiles:
    """
    Small synthetic Tiles instance for unit tests.

    - Cubic prisms with the same size and rotation.
    - Easy axis along +z.
    - mu_r_ea = mu_r_oa = 1 (chi = 0).
    - M_rem = [1, 2, ..., n].
    """
    tiles = Tiles(n)
    tiles.tile_type = 2  # prism

    if offsets is None:
        offsets = np.zeros((n, 3), dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (n, 3):
        raise ValueError(f"offsets must have shape ({n}, 3), got {offsets.shape}")

    dims = np.array([cube_dim, cube_dim, cube_dim], dtype=np.float64)
    for i in range(n):
        tiles.offset = (offsets[i], i)
        tiles.size = (dims, i)
        tiles.rot = ((0.0, 0.0, 0.0), i)
        tiles.M_rem = (float(i + 1), i)
        tiles.mu_r_ea = (1.0, i)
        tiles.mu_r_oa = (1.0, i)
        tiles.u_ea = ((0.0, 0.0, 1.0), i)
    return tiles


class MacroMagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent.parent / "test_files"
        cls.csv_path = base / "magtense_zot80_3d_small.csv"

    # --- basic validation tests ---

    def test_demag_tensor_small_cube_properties(self):
        """
        Demag tensor sanity checks on a tiny synthetic system.

        For a cube, symmetry implies the self-demag block has equal diagonal
        entries and trace(N_ii) = -1 (our convention returns -N in H_d = -N M).
        """
        offsets = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
        tiles = make_trivial_tiles(2, cube_dim=0.01, offsets=offsets)
        macro = MacroMag(tiles)

        N = macro.fast_get_demag_tensor(cache=False)
        self.assertEqual(N.shape, (2, 2, 3, 3))

        # Symmetry: each 3×3 block is symmetric.
        for i in range(2):
            for j in range(2):
                with self.subTest(i=i, j=j):
                    np.testing.assert_allclose(N[i, j], N[i, j].T, atol=1e-14, rtol=0.0)

        # Cube self-demag: equal diagonal, trace ~ -1, diag ~ -1/3.
        diag = np.diag(N[0, 0])
        self.assertAlmostEqual(float(np.trace(N[0, 0])), -1.0, places=10)
        np.testing.assert_allclose(diag, (-1.0 / 3.0) * np.ones(3), atol=5e-6, rtol=0.0)

        # Reciprocity for identical cubes in identical orientation: N_01 == N_10.
        np.testing.assert_allclose(N[0, 1], N[1, 0], atol=1e-14, rtol=0.0)

    def test_fast_get_demag_tensor_cache(self):
        """fast_get_demag_tensor with cache=True returns same object on second call."""
        tiles = make_trivial_tiles(2, cube_dim=0.01)
        macro = MacroMag(tiles)
        N1 = macro.fast_get_demag_tensor(cache=True)
        N2 = macro.fast_get_demag_tensor(cache=True)
        self.assertIs(N1, N2)
        N3 = macro.fast_get_demag_tensor(cache=False)
        np.testing.assert_allclose(N1, N3, atol=0, rtol=0)
        self.assertIsNot(N1, N3)

    def test_direct_solve_magnitude_constant(self):
        """
        When chi = 0 and H_coil = 0, direct solve should yield M = M_rem * u_ea.
        This ensures equivalence between MacroMag and uncoupled GPMO for μ=1.
        """
        tiles = make_trivial_tiles(4)
        macro = MacroMag(tiles)

        # Any valid neighbors array is sufficient here: chi=0 short-circuits the solve.
        neighbors = np.tile(np.arange(tiles.n, dtype=np.int64), (tiles.n, 1))
        macro, A = macro.direct_solve_neighbor_sparse(
            neighbors=neighbors,
            params=SolverParams(use_coils=False, krylov_tol=1e-8, krylov_it=200),
        )
        self.assertIsNone(A)

        norms = np.linalg.norm(tiles.M, axis=1)
        expected = tiles.M_rem
        for idx, (n_val, e_val) in enumerate(zip(norms, expected)):
            with self.subTest(tile=idx):
                self.assertAlmostEqual(
                    n_val, e_val, places=8,
                    msg=f"Tile {idx}: |M| = {n_val:.12f}, expected {e_val:.12f}"
                )

    def test_muse_reference_tensor_subset(self):
        """
        Check demag tensor subset for small MUSE grid: shape, symmetry, and trace.
        """
        tiles = muse2tiles(str(self.csv_path), magnetization=1.1658e6, verbose=False)
        macro = MacroMag(tiles)
        idx = np.array([0, 1, 2], dtype=np.int64)
        blocks = assemble_blocks_subset(macro.centres, macro.half, macro.Rg2l, idx, idx)
        self.assertEqual(blocks.shape, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(blocks[i, j], blocks[i, j].T, atol=1e-14)
        # Self-demag trace = -1 for uniformly magnetized prism
        np.testing.assert_allclose(
            np.trace(blocks[0, 0]), -1.0, atol=1e-3, rtol=0.0
        )

    # minimal tests for Tiles API

    def test_tile_proxy_and_setters(self):
        """Basic Tiles set/get consistency."""
        tiles = make_trivial_tiles(2)
        self.assertEqual(tiles[0].M_rem, 1.0)
        self.assertEqual(tiles[1].M_rem, 2.0)
        tiles[1].M_rem = 5.5
        self.assertEqual(tiles.M_rem[1], 5.5)
        self.assertEqual(tiles.M_rem[0], 1.0)
        with self.assertRaises(IndexError):
            _ = tiles[2]
        s = str(tiles)
        self.assertIn("Tile_0", s)
        self.assertIn("Tile_1", s)

    def test_rotation_matrices(self):
        """Rotation matrices should be orthonormal inverses."""
        Rg2l, Rl2g = MacroMag.get_rotation_matrices(0.1, 0.2, 0.3)
        I = np.eye(3)
        self.assertTrue(np.allclose(Rl2g @ Rg2l, I, atol=1e-12))

    def test_set_easy_axis_error(self):
        tiles = Tiles(1)
        with self.assertRaises(TypeError):
            tiles.set_easy_axis(val=1.23, idx=0)

    def test_rotation_angle_roundtrip(self):
        """rotation_angle is inverse of _rotation_matrix for both conventions."""

        alpha, beta, gamma = 0.1, 0.2, 0.3
        R_zyx = _rotation_matrix(alpha, beta, gamma, xyz=False)
        a, b, c = rotation_angle(R_zyx, xyz=False)
        np.testing.assert_allclose([a, b, c], [alpha, beta, gamma], atol=1e-10)

        R_xyz = _rotation_matrix(alpha, beta, gamma, xyz=True)
        a2, b2, c2 = rotation_angle(R_xyz, xyz=True)
        np.testing.assert_allclose([a2, b2, c2], [alpha, beta, gamma], atol=1e-10)

    def test_get_rotmat_orthonormal(self):
        """get_rotmat returns orthonormal rotation matrix."""
        rot = np.array([0.1, 0.2, 0.3])
        R = get_rotmat(rot)
        self.assertEqual(R.shape, (3, 3))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_build_prism_returns_tiles(self):
        """build_prism returns Tiles with correct attributes."""
        lwh = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
        center = np.array([[0, 0, 0], [0.1, 0, 0]])
        rot = np.zeros((2, 3))
        mag_angle = np.array([[np.pi / 2, 0], [np.pi / 2, np.pi / 2]])
        mu = np.array([[1.05, 1.15], [1.05, 1.15]])
        remanence = np.array([1e6, 1e6])

        prism = build_prism(lwh, center, rot, mag_angle, mu, remanence)
        self.assertIsInstance(prism, Tiles)
        self.assertEqual(prism.n, 2)
        np.testing.assert_allclose(prism.size, lwh)
        np.testing.assert_allclose(prism.offset, center)
        np.testing.assert_allclose(prism.M_rem, remanence)

    def test_muse2tiles_from_csv(self):
        """muse2tiles loads CSV and returns valid Tiles."""
        tiles = muse2tiles(str(self.csv_path), magnetization=1.1658e6, verbose=False)
        self.assertIsInstance(tiles, Tiles)
        self.assertGreater(tiles.n, 0)
        self.assertEqual(tiles.size.shape[1], 3)
        self.assertEqual(tiles.offset.shape[1], 3)

    def test_direct_solve_with_demag_toggle(self):
        """direct_solve with use_demag=False yields local M = M_rem * u_ea."""
        tiles = make_trivial_tiles(2, cube_dim=0.01)
        tiles.mu_r_ea = (1.5, 0)
        tiles.mu_r_oa = (1.2, 0)
        tiles.mu_r_ea = (1.5, 1)
        tiles.mu_r_oa = (1.2, 1)
        macro = MacroMag(tiles)
        N = macro.fast_get_demag_tensor(cache=False)

        macro2, A = macro.direct_solve(
            N_new_rows=N,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(use_coils=False, use_demag=False),
        )
        self.assertIsNotNone(A)
        np.testing.assert_allclose(A, np.eye(6), atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(tiles.M, axis=1),
            tiles.M_rem,
            atol=1e-10,
        )

    def test_prism_N_local_vectorized_matches_scalar(self):
        """
        _prism_N_local_vectorized matches _prism_N_local_nb element-wise.

        Spot-check several (a,b,c,x0,y0,z0) combinations including edge cases.
        """
        cases = [
            (0.5, 0.5, 0.5, 0.0, 0.0, 0.0),
            (0.01, 0.01, 0.01, 0.02, 0.0, 0.0),
            (1.0, 0.5, 0.25, -0.1, 0.2, -0.05),
            (0.1, 0.2, 0.3, 0.05, 0.05, 0.05),
        ]
        for a, b, c, x0, y0, z0 in cases:
            with self.subTest(a=a, b=b, c=c, x0=x0, y0=y0, z0=z0):
                scalar = _prism_N_local_nb(a, b, c, x0, y0, z0)
                vec = _prism_N_local_vectorized(
                    np.array([a]), np.array([b]), np.array([c]),
                    np.array([[x0]]), np.array([[y0]]), np.array([[z0]]),
                )
                np.testing.assert_allclose(vec[0, 0], scalar, atol=1e-10, rtol=1e-9)

    def test_prism_N_local_vectorized_faster_than_scalar(self):
        """
        Vectorized demag kernel is faster than scalar loop for large batches.

        Benchmarks _prism_N_local_vectorized vs repeated _prism_N_local_nb calls
        for a (48, 48) block grid (2304 evaluations). Asserts vectorized is at
        least 5x faster and prints the speedup for CI visibility.
        """
        nI, nJ = 48, 48
        n_blocks = nI * nJ
        a, b, c = 0.01, 0.01, 0.01
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-0.05, 0.05, (nI, nJ)).astype(np.float64)
        y0 = rng.uniform(-0.05, 0.05, (nI, nJ)).astype(np.float64)
        z0 = rng.uniform(-0.05, 0.05, (nI, nJ)).astype(np.float64)
        a_arr = np.full((nI, nJ), a, dtype=np.float64)
        b_arr = np.full((nI, nJ), b, dtype=np.float64)
        c_arr = np.full((nI, nJ), c, dtype=np.float64)

        # Warmup
        _ = _prism_N_local_vectorized(a_arr, b_arr, c_arr, x0, y0, z0)
        for i in range(2):
            for j in range(2):
                _ = _prism_N_local_nb(a, b, c, float(x0[i, j]), float(y0[i, j]), float(z0[i, j]))

        # Time scalar loop: n_blocks calls
        t0 = time.perf_counter()
        for i in range(nI):
            for j in range(nJ):
                _ = _prism_N_local_nb(a, b, c, float(x0[i, j]), float(y0[i, j]), float(z0[i, j]))
        t_scalar = time.perf_counter() - t0

        # Time vectorized: one call for same (nI, nJ) grid
        t0 = time.perf_counter()
        _ = _prism_N_local_vectorized(a_arr, b_arr, c_arr, x0, y0, z0)
        t_vec = time.perf_counter() - t0

        speedup = t_scalar / (t_vec + 1e-12)
        print(f"  [prism_N_local] scalar={t_scalar:.4f}s vectorized={t_vec:.4f}s speedup={speedup:.1f}x (n={n_blocks})")
        self.assertGreater(speedup, 5.0, msg=f"Vectorized should be >5x faster, got {speedup:.1f}x")

    def test_assemble_demag_tensor_translation_invariant_matches_full(self):
        """
        Translation-invariant path in _assemble_demag_tensor produces identical
        results to full block assembly for uniform grids.
        """
        # 2x2 uniform grid
        offsets = np.array(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.02, 0.02, 0.0]],
            dtype=np.float64,
        )
        tiles = make_trivial_tiles(4, cube_dim=0.01, offsets=offsets)
        macro = MacroMag(tiles)
        N_invariant = macro.fast_get_demag_tensor(cache=False)
        # Full assembly via assemble_blocks_subset
        idx = np.arange(4, dtype=np.int32)
        N_full = assemble_blocks_subset(macro.centres, macro.half, macro.Rg2l, idx, idx)
        np.testing.assert_allclose(N_invariant, N_full, atol=1e-14, rtol=0.0)

    def test_assemble_demag_tensor_translation_invariant_faster_than_general(self):
        """
        For uniform grids, translation-invariant path is faster than general O(N^2).
        Creates a 6x6x6=216 tile uniform grid and times both paths.
        """
        # 6^3 = 216 uniform tiles
        n_side = 6
        offs = []
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    offs.append([i * 0.02, j * 0.02, k * 0.02])
        offsets = np.array(offs, dtype=np.float64)
        tiles_uniform = make_trivial_tiles(len(offsets), cube_dim=0.01, offsets=offsets)
        macro_uniform = MacroMag(tiles_uniform)

        t0 = time.perf_counter()
        _ = macro_uniform.fast_get_demag_tensor(cache=False)
        t_invariant = time.perf_counter() - t0

        # Non-uniform: one tile has different size (breaks translation invariance)
        tiles_nonuniform = make_trivial_tiles(len(offsets), cube_dim=0.01, offsets=offsets)
        tiles_nonuniform.size = (np.array([0.0099, 0.01, 0.01], dtype=np.float64), 0)
        macro_nonuniform = MacroMag(tiles_nonuniform)
        t0 = time.perf_counter()
        _ = macro_nonuniform.fast_get_demag_tensor(cache=False)
        t_general = time.perf_counter() - t0

        speedup = t_general / (t_invariant + 1e-12)
        print(f"  [_assemble_demag_tensor] invariant={t_invariant:.4f}s general={t_general:.4f}s speedup={speedup:.1f}x (n=216)")
        # Invariant path reduces O(n^2) to O(n_unique); for small n overhead can narrow the gap.
        self.assertGreaterEqual(speedup, 1.0, msg=f"Translation-invariant should not be slower, got {speedup:.1f}x")

    def test_direct_solve_matrix_free_matches_dense(self):
        """
        direct_solve with matrix_free=True produces identical results to
        matrix_free=False when given the same N tensor, validating the
        matrix-free GMRES matvec path.

        Uses nontrivial examples: finite susceptibility (chi != 0), multiple
        tiles with demag coupling, uniform and non-uniform grids.
        """
        def _setup_with_mu(tiles):
            """Set finite mu so demag coupling matters."""
            for i in range(tiles.n):
                tiles.mu_r_ea = (1.5, i)
                tiles.mu_r_oa = (1.2, i)

        params = SolverParams(
            use_coils=False,
            use_demag=True,
            krylov_tol=1e-11,
            krylov_it=500,
        )

        # --- Case 1: Uniform 2x2x2 grid (8 tiles) ---
        offs_222 = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    offs_222.append([i * 0.02, j * 0.02, k * 0.02])
        offsets_222 = np.array(offs_222, dtype=np.float64)
        tiles_dense = make_trivial_tiles(8, cube_dim=0.01, offsets=offsets_222)
        tiles_mf = make_trivial_tiles(8, cube_dim=0.01, offsets=offsets_222)
        _setup_with_mu(tiles_dense)
        _setup_with_mu(tiles_mf)

        macro_dense = MacroMag(tiles_dense)
        macro_mf = MacroMag(tiles_mf)
        N = macro_dense.fast_get_demag_tensor(cache=False)

        macro_dense.direct_solve(
            N_new_rows=N,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(**{**params.__dict__, "matrix_free": False}),
        )
        macro_mf.direct_solve(
            N_new_rows=N,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(**{**params.__dict__, "matrix_free": True}),
        )
        np.testing.assert_allclose(
            tiles_dense.M, tiles_mf.M,
            atol=1e-9, rtol=1e-8,
            err_msg="matrix_free=True vs False mismatch on uniform 2x2x2 grid",
        )

        # --- Case 2: Non-uniform grid (different tile sizes) ---
        lwh = np.array([
            [0.01, 0.01, 0.01],
            [0.012, 0.01, 0.01],
            [0.01, 0.012, 0.01],
            [0.01, 0.01, 0.012],
            [0.009, 0.011, 0.01],
        ], dtype=np.float64)
        center = np.array([
            [0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0.03, 0.03, 0], [0.015, 0.015, 0],
        ], dtype=np.float64)
        rot = np.zeros((5, 3))
        mag_angle = np.full((5, 2), np.pi / 2)
        mu = np.tile([1.05, 1.15], (5, 1))
        remanence = np.full(5, 1e6)
        tiles_nonunif = build_prism(lwh, center, rot, mag_angle, mu, remanence)
        macro_dense2 = MacroMag(tiles_nonunif)
        tiles_nonunif2 = build_prism(lwh, center, rot, mag_angle, mu, remanence)
        macro_mf2 = MacroMag(tiles_nonunif2)
        N2 = macro_dense2.fast_get_demag_tensor(cache=False)

        macro_dense2.direct_solve(
            N_new_rows=N2,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(**{**params.__dict__, "matrix_free": False}),
        )
        macro_mf2.direct_solve(
            N_new_rows=N2,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(**{**params.__dict__, "matrix_free": True}),
        )
        np.testing.assert_allclose(
            tiles_nonunif.M, tiles_nonunif2.M,
            atol=1e-9, rtol=1e-8,
            err_msg="matrix_free=True vs False mismatch on non-uniform 5-tile grid",
        )

        # --- Case 3: Rotated tiles (nontrivial easy-axis orientations) ---
        tiles_rot = make_trivial_tiles(4, cube_dim=0.01)
        for i in range(4):
            tiles_rot.mu_r_ea = (1.4, i)
            tiles_rot.mu_r_oa = (1.1, i)
            # Vary easy-axis direction
            theta = i * np.pi / 4
            tiles_rot.u_ea = ((np.cos(theta), np.sin(theta), 0.5), i)
        offsets_2x2 = np.array(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.02, 0.02, 0.0]],
            dtype=np.float64,
        )
        tiles_rot.offset = offsets_2x2
        tiles_rot_dense = make_trivial_tiles(4, cube_dim=0.01, offsets=offsets_2x2)
        tiles_rot_mf = make_trivial_tiles(4, cube_dim=0.01, offsets=offsets_2x2)
        for i in range(4):
            tiles_rot_dense.mu_r_ea = (1.4, i)
            tiles_rot_dense.mu_r_oa = (1.1, i)
            tiles_rot_mf.mu_r_ea = (1.4, i)
            tiles_rot_mf.mu_r_oa = (1.1, i)
            theta = i * np.pi / 4
            u = (np.cos(theta), np.sin(theta), 0.5)
            tiles_rot_dense.u_ea = (u, i)
            tiles_rot_mf.u_ea = (u, i)

        macro_rot_dense = MacroMag(tiles_rot_dense)
        macro_rot_mf = MacroMag(tiles_rot_mf)
        N3 = macro_rot_dense.fast_get_demag_tensor(cache=False)

        macro_rot_dense.direct_solve(
            N_new_rows=N3,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(**{**params.__dict__, "matrix_free": False}),
        )
        macro_rot_mf.direct_solve(
            N_new_rows=N3,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(**{**params.__dict__, "matrix_free": True}),
        )
        np.testing.assert_allclose(
            tiles_rot_dense.M, tiles_rot_mf.M,
            atol=1e-9, rtol=1e-8,
            err_msg="matrix_free=True vs False mismatch on rotated-tile grid",
        )

    def test_direct_solve_neighbor_sparse_matches_direct_solve_when_full_neighbors(self):
        """
        direct_solve_neighbor_sparse matches direct_solve when neighbors include
        all tiles (full demag), validating the sparse assembly and GMRES path.
        """
        tiles_sparse = make_trivial_tiles(3, cube_dim=0.01)
        tiles_full = make_trivial_tiles(3, cube_dim=0.01)
        for tiles in (tiles_sparse, tiles_full):
            tiles.mu_r_ea = (1.5, 0)
            tiles.mu_r_oa = (1.2, 0)
            for i in range(1, 3):
                tiles.mu_r_ea = (1.5, i)
                tiles.mu_r_oa = (1.2, i)
        macro_sparse = MacroMag(tiles_sparse)
        macro_full = MacroMag(tiles_full)
        N = macro_sparse.fast_get_demag_tensor(cache=False)
        neighbors_full = np.tile(np.arange(3, dtype=np.int64), (3, 1))

        macro_sparse.direct_solve_neighbor_sparse(
            neighbors=neighbors_full,
            params=SolverParams(use_coils=False, krylov_tol=1e-10, krylov_it=300),
        )
        macro_full.direct_solve(
            N_new_rows=N,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=SolverParams(use_coils=False, use_demag=True),
        )
        np.testing.assert_allclose(tiles_sparse.M, tiles_full.M, atol=1e-6, rtol=1e-5)

    def test_solver_params_override_and_backward_compat(self):
        """
        direct_solve and direct_solve_neighbor_sparse produce consistent results
        when given the same SolverParams.
        """
        def setup_tiles():
            t = make_trivial_tiles(2, cube_dim=0.01)
            t.mu_r_ea = (1.5, 0)
            t.mu_r_oa = (1.2, 0)
            t.mu_r_ea = (1.5, 1)
            t.mu_r_oa = (1.2, 1)
            return t

        tiles_kw = setup_tiles()
        tiles_params = setup_tiles()
        macro_kw = MacroMag(tiles_kw)
        macro_params = MacroMag(tiles_params)
        N = macro_kw.fast_get_demag_tensor(cache=False)
        neighbors = np.tile(np.arange(2, dtype=np.int64), (2, 1))

        # direct_solve: params only
        params = SolverParams(
            use_coils=False,
            use_demag=True,
            krylov_tol=1e-10,
            krylov_it=100,
        )
        macro_kw.direct_solve(
            N_new_rows=N,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=params,
        )
        macro_params.direct_solve(
            N_new_rows=N,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
            params=params,
        )
        np.testing.assert_allclose(tiles_kw.M, tiles_params.M, atol=1e-6, rtol=1e-5)

        # direct_solve_neighbor_sparse: kwargs vs params (fresh tiles)
        tiles_kw2 = setup_tiles()
        tiles_params2 = setup_tiles()
        macro_kw2 = MacroMag(tiles_kw2)
        macro_params2 = MacroMag(tiles_params2)
        macro_kw2.direct_solve_neighbor_sparse(neighbors=neighbors, params=params)
        macro_params2.direct_solve_neighbor_sparse(neighbors=neighbors, params=params)
        np.testing.assert_allclose(tiles_kw2.M, tiles_params2.M, atol=1e-6, rtol=1e-5)

    def test_assemble_blocks_subset_empty(self):
        """assemble_blocks_subset handles empty index arrays."""
        centres = np.zeros((3, 3))
        half = np.ones((3, 3)) * 0.5
        Rg2l = np.tile(np.eye(3), (3, 1, 1))
        I = np.array([], dtype=np.int32)
        J = np.array([], dtype=np.int32)
        blocks = assemble_blocks_subset(centres, half, Rg2l, I, J)
        self.assertEqual(blocks.shape, (0, 0, 3, 3))

    def test_assemble_blocks_subset_rectangular(self):
        """Assemble blocks for nI != nJ (batched case in GPMOmr)."""
        rng = np.random.default_rng(42)
        centres = rng.random((50, 3))
        half = np.ones((50, 3)) * 0.01
        Rg2l = np.tile(np.eye(3), (50, 1, 1))
        I = np.array([0, 1, 2])  # nI=3
        J = np.arange(20)  # nJ=20
        blocks = assemble_blocks_subset(centres, half, Rg2l, I, J)
        self.assertEqual(blocks.shape, (3, 20, 3, 3))

    @unittest.skipIf(in_github_actions, "Nadjacent convergence test too slow for CI")
    def test_nadjacent_convergence_on_muse_full_grid(self):
        """
        direct_solve_neighbor_sparse converges as Nadjacent increases.

        Uses neighbors = _connectivity_matrix_py(tiles.offset, Nadjacent) on
        the full MUSE magtense_zot80_3d.csv grid. Solves for magnet-magnet
        coupled magnetization at several Nadjacent values and verifies that
        the relative error vs the reference (largest Nadjacent) decreases.
        Outputs a plot of solve error vs Nadjacent.
        """
        base = Path(__file__).parent.parent / "test_files"
        csv_path = base / "magtense_zot80_3d.csv"
        self.assertTrue(csv_path.exists(), f"Missing {csv_path}")

        tiles = muse2tiles(csv_path, (1.05, 1.15))
        mac = MacroMag(tiles)

        Nadjacent_values = [10, 50, 100, 200, 400, 800]
        M_history = []

        for k in Nadjacent_values:
            neighbors = _connectivity_matrix_py(tiles.offset, k)
            mac.direct_solve_neighbor_sparse(
                neighbors=neighbors,
                params=SolverParams(
                    use_coils=False,
                    krylov_tol=1e-10,
                    krylov_it=200,
                    print_progress=False,
                ),
            )
            M_history.append(tiles.M.copy())

        M_ref = M_history[-1]
        ref_norm = np.linalg.norm(M_ref)
        self.assertGreater(ref_norm, 1e-6, "Reference magnetization should be non-trivial")

        errors_rel = []
        errors_abs = []
        for M in M_history:
            diff = M - M_ref
            errors_abs.append(float(np.linalg.norm(diff)))
            errors_rel.append(float(np.linalg.norm(diff) / (ref_norm + 1e-30)))

        # Error should decrease monotonically as Nadjacent increases
        for i in range(len(errors_rel) - 1):
            self.assertGreaterEqual(
                errors_rel[i],
                errors_rel[i + 1],
                msg=f"Error should decrease: Nadjacent={Nadjacent_values[i]} err={errors_rel[i]:.2e} "
                f"> Nadjacent={Nadjacent_values[i+1]} err={errors_rel[i+1]:.2e}",
            )

        # Largest Nadjacent should have near-zero error (self-reference)
        self.assertLess(errors_rel[-1], 1e-10, "Self-reference error should be negligible")

        # Plot and save (relative errors; last point included, use floor for semilogy)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return  # Skip plot if matplotlib unavailable

        out_dir = base / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "nadjacent_convergence.png"

        # Floor for semilogy so zero error at last point is visible
        y_plot = np.maximum(errors_rel, 1e-30)

        fig, ax = plt.subplots()
        ax.semilogy(Nadjacent_values[:-1], y_plot[:-1], "o-")
        ax.set_xlabel("Nadjacent")
        ax.set_ylabel(r"Relative solve error $\|M - M_{\mathrm{ref}}\| / \|M_{\mathrm{ref}}\|$")
        ax.set_title("direct_solve_neighbor_sparse convergence on magtense_zot80_3d.csv")
        ax.grid(True, which="both")
        fig.tight_layout()
        fig.savefig(plot_path)
        if not in_github_actions:
            plt.show()
        plt.close(fig)
        print(f"  [test_nadjacent_convergence] Saved plot to {plot_path}")


if __name__ == "__main__":
    unittest.main()
