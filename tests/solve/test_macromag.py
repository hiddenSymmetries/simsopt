import unittest
import json
import numpy as np
from pathlib import Path
from typing import Optional

from simsopt.solve.macromag import (
    MacroMag,
    Tiles,
    assemble_blocks_subset,
    build_prism,
    get_rotmat,
    muse2tiles,
    rotation_angle,
)
from simsopt.solve.macromag import _rotation_matrix


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
        cls.csv_path = base / "magtense_zot80_3d.csv"
        cls.ref_subset_path = base / "muse_tensor_subset.json"

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
            use_coils=False,
            krylov_tol=1e-8,
            krylov_it=200,
            print_progress=False,
            x0=None,
            H_a_override=None,
            drop_tol=0.0,
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
        Regression check against a fixed reference subset of the MUSE demag tensor computed via MAGTENSE.

        This avoids storing the full 9736×9736 tensor in the repository while still
        pinning a handful of representative demag blocks to known values.
        """
        with open(self.ref_subset_path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        idx = np.asarray(ref["indices"], dtype=np.int64)
        ref_blocks = np.asarray(ref["blocks"], dtype=np.float64)

        tiles = muse2tiles(str(self.csv_path), magnetization=1.1658e6)
        macro = MacroMag(tiles)

        blocks = assemble_blocks_subset(macro.centres, macro.half, macro.Rg2l, idx, idx)
        np.testing.assert_allclose(blocks, ref_blocks, atol=1e-12, rtol=0.0)

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
        tiles = muse2tiles(str(self.csv_path), magnetization=1.1658e6)
        self.assertIsInstance(tiles, Tiles)
        self.assertGreater(tiles.n, 0)
        self.assertEqual(tiles.size.shape[1], 3)
        self.assertEqual(tiles.offset.shape[1], 3)

    def test_direct_solve_with_demag_toggle(self):
        """direct_solve_toggle with use_demag=False yields local M = M_rem * u_ea."""
        tiles = make_trivial_tiles(2, cube_dim=0.01)
        tiles.mu_r_ea = (1.5, 0)
        tiles.mu_r_oa = (1.2, 0)
        tiles.mu_r_ea = (1.5, 1)
        tiles.mu_r_oa = (1.2, 1)
        macro = MacroMag(tiles)
        N = macro.fast_get_demag_tensor(cache=False)

        macro2, A = macro.direct_solve_toggle(
            use_coils=False,
            use_demag=False,
            N_new_rows=N,
            A_prev=np.zeros((0, 0)),
            prev_n=0,
        )
        self.assertIsNotNone(A)
        np.testing.assert_allclose(A, np.eye(6), atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(tiles.M, axis=1),
            tiles.M_rem,
            atol=1e-10,
        )

    def test_assemble_blocks_subset_empty(self):
        """assemble_blocks_subset handles empty index arrays."""
        centres = np.zeros((3, 3))
        half = np.ones((3, 3)) * 0.5
        Rg2l = np.tile(np.eye(3), (3, 1, 1))
        I = np.array([], dtype=np.int32)
        J = np.array([], dtype=np.int32)
        blocks = assemble_blocks_subset(centres, half, Rg2l, I, J)
        self.assertEqual(blocks.shape, (0, 0, 3, 3))


if __name__ == "__main__":
    unittest.main()
