import unittest
import numpy as np
from pathlib import Path
import logging

from simsopt.solve.macromag import MacroMag, muse2tiles, Tiles
from simsopt.solve.permanent_magnet_optimization import _neighbors_from_radius

logger = logging.getLogger(__name__)


def make_trivial_tiles(n: int):
    """
    Create a trivial Tiles object with:
      - all centers at the origin
      - unit cube size
      - easy axis along +z
      - mu_r_ea = mu_r_oa = 1 (chi = 0)
      - M_rem = [1, 2, ..., n]
    """
    tiles = Tiles(n)
    tiles.center_pos = (0.0, 0)
    tiles.dev_center = (0.0, 0)
    tiles.size = (1.0, 0)
    tiles.rot = (np.array([0.0, 0.0, 0.0]), 0)

    for i in range(n):
        tiles.M_rem = (float(i + 1), i)
        tiles.mu_r_ea = (1.0, i)
        tiles.mu_r_oa = (1.0, i)
        tiles.u_ea = (np.array([0.0, 0.0, 1.0]), i)
    return tiles


class MacroMagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent.parent / "test_files"
        cls.csv_path = base / "magtense_zot80_3d.csv"
        cls.npz_path = base / "muse_tensor.npy"
        cls.hfield_path = base / "muse_demag_field.npy"

        # load reference demag tensor
        arr = np.load(cls.npz_path, allow_pickle=False)
        cls.demag_ref = (
            arr["demag"].astype(np.float64)
            if isinstance(arr, np.lib.npyio.NpzFile)
            else arr.astype(np.float64)
        )

        # build tiles + MacroMag
        tiles = muse2tiles(str(cls.csv_path), magnetization=1.1658e6)
        cls.macro = MacroMag(tiles)

        # also precompute our tensor once
        cls.demag_test = cls.macro.fast_get_demag_tensor().astype(np.float64)
        cls.demag_test[np.abs(cls.demag_test) < 1e-12] = 0.0
        cls.pts = cls.macro.centres

    # --- basic validation tests ---

    def test_shape(self):
        """Demag tensor shape should match reference."""
        self.assertEqual(
            self.demag_test.shape,
            self.demag_ref.shape,
            f"expected shape {self.demag_ref.shape}, got {self.demag_test.shape}",
        )

    def test_symmetry(self):
        """N[i,j] == N[i,j].T for representative entries."""
        for i, j in [(0, 0), (1, 1), (0, 1), (2, 3)]:
            with self.subTest(i=i, j=j):
                M = self.demag_test[i, j]
                self.assertTrue(np.allclose(M, M.T, atol=0),
                                f"N[{i},{j}] not symmetric")

    def test_trace_invariant(self):
        """Trace(N_ii) ≈ 1 for self-demag blocks."""
        for i in range(min(5, self.demag_test.shape[0])):
            tr = np.trace(self.demag_test[i, i])
            self.assertAlmostEqual(abs(tr), 1.0, places=12,
                                   msg=f"trace N[{i},{i}] = {tr}")

    def test_against_stored(self):
        """Compare computed tensor against stored reference."""
        diff = np.abs(self.demag_test - self.demag_ref)
        max_err = diff.max()
        self.assertLessEqual(max_err, 1e-9,
                             f"max abs error {max_err:.2e} exceeds tol 1e-9")

    def test_direct_solve_magnitude_constant(self):
        """
        When chi = 0 and H_coil = 0, direct solve should yield M = M_rem * u_ea.
        This ensures equivalence between MacroMag and uncoupled GPMO for μ=1.
        """
        tiles = make_trivial_tiles(4)
        macro = MacroMag(tiles)

        pts = np.zeros((tiles.n, 3))
        neighbors = _neighbors_from_radius(pts, radius=1.0, max_neighbors=None)

        macro, _ = macro.solve(
            neighbors=neighbors,
            use_coils=False,
            krylov_tol=1e-8,
            krylov_it=200,
            print_progress=False,
            x0=None,
            H_a_override=None,
            drop_tol=0.0,
        )

        norms = np.linalg.norm(tiles.M, axis=1)
        expected = tiles.M_rem
        for idx, (n_val, e_val) in enumerate(zip(norms, expected)):
            with self.subTest(tile=idx):
                self.assertAlmostEqual(
                    n_val, e_val, places=8,
                    msg=f"Tile {idx}: |M| = {n_val:.12f}, expected {e_val:.12f}"
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


if __name__ == "__main__":
    unittest.main()
