import unittest
from pathlib import Path
import numpy as np
import logging
from simsopt.solve.macromag import MacroMag, muse2tiles, Tiles
from coilpy.magtense_interface import muse2magntense
from scipy.constants import mu_0

logger = logging.getLogger(__name__)

def make_trivial_tiles(n: int):
    """
    Create a trivial Tiles with:
      - all centers at 0
      - unit prisms
      - easy axis = z
      - mu_r_ea = mu_r_oa = 1 (chi=0)
      - M_rem = [1,2,...,n]
    """
    tiles = Tiles(n)
    # broadcast setters
    tiles.center_pos = (0.0, 0)
    tiles.dev_center = (0.0, 0)
    tiles.size       = (1.0, 0)
    tiles.rot        = (np.array([0.0,0.0,0.0]), 0)
    # set remanent magnetization per tile
    for i in range(n):
        tiles.M_rem  = (float(i+1), i)
        # unit permeability
        tiles.mu_r_ea = (1.0, i)
        tiles.mu_r_oa = (1.0, i)
        # easy axis along +z
        tiles.u_ea    = (np.array([0.0, 0.0, 1.0]), i)
    return tiles

class MacroMagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent.parent / "test_files"
        cls.csv_path     = base / "magtense_zot80_3d.csv"
        cls.npz_path     = base / "muse_tensor.npy"
        cls.hfield_path  = base / "muse_demag_field.npy"

        # load reference demag tensor
        arr = np.load(cls.npz_path, allow_pickle=False)
        cls.demag_ref = arr["demag"].astype(np.float64) if isinstance(arr, np.lib.npyio.NpzFile) else arr.astype(np.float64)

        # load reference H_field
        cls.hfield_ref = np.load(cls.hfield_path, allow_pickle=False).astype(np.float64)

        # build tiles + MacroMag
        tiles = muse2tiles(str(cls.csv_path),magnetization=1.1658e6)
        cls.macro = MacroMag(tiles)
        

        # also precompute our tensor once
        cls.demag_test = cls.macro.fast_get_demag_tensor().astype(np.float64)
        cls.demag_test[np.abs(cls.demag_test) < 1e-12] = 0.0

        cls.pts = cls.macro.centres


    def test_shape(self):
        """Test that the computed tensor has the same shape as the stored one."""
        self.assertEqual(self.demag_test.shape, self.demag_ref.shape,
                         f"expected shape {self.demag_ref.shape}, got {self.demag_test.shape}")

    def test_symmetry(self):
        """Test N[i,j] == N[i,j].T for a few representative entries."""
        # pick a small sample so this doesn't blow up memory/time
        for i, j in [(0,0), (1,1), (0,1), (2,3)]:
            with self.subTest(i=i, j=j):
                M = self.demag_test[i, j]
                self.assertTrue(np.allclose(M, M.T, atol=0),
                                f"N[{i},{j}] not symmetric")

    def test_trace_invariant(self):
        """For a rectangular prism, the self‐demag trace should be ≈1."""
        # only check diagonal entries (i==j)
        n = self.demag_test.shape[0]
        # sample first 5 to avoid huge loops
        for i in range(min(5, n)):
            tr = np.trace(self.demag_test[i, i])
            self.assertAlmostEqual(abs(tr), 1.0, places=12,
                msg=f"trace N[{i},{i}] = {tr}, |trace| expected 1.0")

    def test_against_stored(self):
        """Compare absolute errors with the reference."""
        diff    = np.abs(self.demag_test - self.demag_ref)
        max_err = diff.max()

        # strict absolute‐error tolerance
        ABS_TOL = 1e-9
        self.assertLessEqual(
            max_err, ABS_TOL,
            f"max absolute error {max_err:.2e} exceeds tol={ABS_TOL:.2e}"
        )
        
    def test_mismatch_report(self):
        """If there is a big mismatch, print its location (for debugging)."""
        diff = np.abs(self.demag_test - self.demag_ref)
        max_err = diff.max()
        
        logger.info("Max |ΔN|:", max_err, "(rms", np.sqrt(np.mean(diff**2)), ")")

        tol = 1e-9
        mask = diff > tol
        if mask.any():
            idx = np.unravel_index(np.argmax(diff * mask), diff.shape)
            msg = (
                f"largest absolute mismatch at {idx}: "
                f"test={self.demag_test[idx]:.6e} "
                f"ref={self.demag_ref[idx]:.6e} "
                f"abs delta={diff[idx]:.2e} (tol={tol})"
            )
            self.fail(msg)
            
            
    def test_demag_field(self):
        """Compare our _demag_field_at to saved MagTense H_field."""
        H_test = self.macro._demag_field_at(self.pts, self.demag_ref)
        diff   = np.linalg.norm(H_test - self.hfield_ref, axis=1)
        max_err = diff.max()
        rms     = np.sqrt(np.mean(diff**2))

        # absolute‐error tolerance ~1e-9 A/m
        self.assertLessEqual(
            max_err, 1e-9,
            f"Max |ΔH| = {max_err:.3e}, RMS = {rms:.3e}"
        )
        
    def test_sor_magnitude_constant(self):
        """When chi = 0 and H_coil = 0, SOR should leave ‖M‖ = M_rem."""
        tiles = make_trivial_tiles(5)
        macro = MacroMag(tiles)
        # zero‐field, demag‐only mode
        macro.sor_solve(
            Ms=1.0, K=1.0,
            omega=1.0, max_it=20, tol=1e-8,
            const_H=(0.0,0.0,0.0),
            use_coils=False,
            demag_only=True,
            log_every=100
        )
        norms   = np.linalg.norm(tiles.M, axis=1)
        expected = tiles.M_rem
        self.assertTrue(
            np.allclose(norms, expected, atol=1e-8),
            f"Direct solve changed magnitudes beyond tolerance: got {norms}, expected {expected}"
        )

    def test_direct_solve_magnitude_constant(self):
        """When chi = 0 and H_coil = 0, direct solve should yield M = M_rem u."""
        tiles = make_trivial_tiles(4)
        macro = MacroMag(tiles)
        macro.direct_solve(
            const_H=(0.0,0.0,0.0),
            use_coils=False,
            demag_only=True,
            krylov_tol=1e-8
        )
        norms   = np.linalg.norm(tiles.M, axis=1)
        expected = tiles.M_rem
        for idx, (n_val, e_val) in enumerate(zip(norms, expected)):
            with self.subTest(tile=idx):
                self.assertAlmostEqual(n_val, e_val, places=8,msg=f"Tile {idx}: |M| = {n_val:.12f}, expected {e_val:.12f}")

    def test_tile_proxy(self):
        """Tile proxy should get/set individual elements and enforce bounds."""
        tiles = make_trivial_tiles(2)
        # both start at 1,2
        self.assertEqual(tiles[0].M_rem, 1.0)
        self.assertEqual(tiles[1].M_rem, 2.0)
        # setting via proxy updates parent
        tiles[1].M_rem = 5.5
        self.assertEqual(tiles.M_rem[1], 5.5)
        # other unchanged
        self.assertEqual(tiles.M_rem[0], 1.0)
        # out‐of‐bounds __getitem__
        with self.assertRaises(IndexError):
            _ = tiles[2]
        # __str__ should mention both tiles
        s = str(tiles)
        self.assertIn("Tile_0", s)
        self.assertIn("Tile_1", s)
        
        def test_rotation_and_ffgg_hh(self):
            # get_rotation_matrices ↔ orthonormal inverses:
            Rg2l, Rl2g = MacroMag.get_rotation_matrices(0.1, 0.2, 0.3)
            I = np.eye(3)
            self.assertTrue(np.allclose(Rl2g @ Rg2l, I, atol=1e-12))

            # simple calls to the f/g/h/FF/GG/HH wrappers (via MacroMag static methods)
            for func in ("_f_3D","_g_3D","_h_3D","_FF_3D","_GG_3D","_HH_3D"):
                # pick a non‐degenerate point
                val = getattr(MacroMag, func)(1.0, 2.0, 3.0, 0.1, -0.2, 0.5)
                self.assertIsInstance(val, float)

    def test_anisotropy_and_coil_field(self):
        # tiny 2‐tile system
        tiles = make_trivial_tiles(2)
        macro = MacroMag(tiles)

        # anisotropy_field: if M parallel to u_ea, should be (2K/(μ0Ms²))*|M|
        M = np.array([[0,0,1.0],[0,0,2.0]])
        K, Ms = 4.0, 2.0
        Ha = macro._anisotropy_field(M, K, Ms)
        expected = (2*K/(mu_0*Ms*Ms)) * np.array([1.0,2.0])[:,None] * tiles.u_ea
        self.assertTrue(np.allclose(Ha, expected))

        # coil_field_at: uniform, no _bs_coil
        pts = np.zeros((3,3))
        Hc = macro._coil_field_at(pts, (1.0,2.0,3.0), use_coils=False)
        self.assertEqual(Hc.shape, (3,3))
        self.assertTrue(np.allclose(Hc, [[1,2,3]]*3))

    def test_set_easy_axis_error(self):
        tiles = Tiles(1)
        # passing a single float should TypeError
        with self.assertRaises(TypeError):
            tiles.set_easy_axis(val=1.23, idx=0)

    def test_vertices_setter_and_errors(self):
        tiles = Tiles(2)
        # valid 3×4 shape broadcast
        mat34 = np.ones((3,4))
        tiles.vertices = mat34
        self.assertTrue((tiles.vertices == mat34).all())

        # valid tuple‐setter with 4×3
        mat43 = np.arange(12).reshape(4,3)
        tiles.vertices = (mat43, 1)
        self.assertTrue((tiles.vertices[1] == mat43.T).all())

        # invalid shape
        with self.assertRaises(AssertionError):
            tiles.vertices = np.zeros((5,5))
            
    def test_other_tiles_setters(self):
        n = 3
        tiles = Tiles(n)

        # tile_type: tuple, scalar, list
        tiles.tile_type = (7,1)
        self.assertEqual(tiles.tile_type[1], 7)
        tiles.tile_type = 9
        self.assertTrue((tiles.tile_type == 9).all())
        tiles.tile_type = [1,2,3]
        self.assertTrue((tiles.tile_type == np.array([1,2,3])).all())

        # offset
        tiles.offset = (np.array([1,1,1]), 2)
        self.assertTrue((tiles.offset[2] == [1,1,1]).all())
        tiles.offset = [0.5,0.5,0.5]
        self.assertTrue((tiles.offset == [0.5,0.5,0.5]).all())

        # M setter: tuple
        tiles.M_rem = [1,2,3]
        vec = np.array([1.0, 2.0, 3.0])
        tiles.u_ea = [ [0,0,1], [0,1,0], [1,0,0] ]
        tiles.M = (vec, 1)
        self.assertTrue((tiles.M[1] == vec).all())

        # u_oa1 / u_oa2 tuple and broadcast
        oa = np.array([1,0,0])
        tiles.u_oa1 = (oa, 0)
        self.assertTrue((tiles.u_oa1[0] == oa).all())
        tiles.u_oa2 = [ [0,1,0] ]*n
        self.assertTrue((tiles.u_oa2 == [[0,1,0]]*n).all())

        # mu_r sets
        tiles.mu_r_ea = (4.2, 2)
        self.assertEqual(tiles.mu_r_ea[2], 4.2)
        tiles.mu_r_oa = 5.5
        self.assertTrue((tiles.mu_r_oa == 5.5).all())

        # M_rel, stfcn_index, incl_it
        for prop in ("M_rel","stfcn_index","incl_it"):
            arr = getattr(tiles, prop)
            setattr(tiles, prop, (7,1))
            self.assertEqual(getattr(tiles, prop)[1], 7)

    def test_add_tiles_and_refine_prism(self):
        # start from a single-tile trivial Tiles
        tiles = make_trivial_tiles(1)
        # mark it as a prism
        tiles.tile_type = (2, 0)
        self.assertEqual(tiles.n, 1)

        # refine it into 2×2×1 = 4 prisms
        tiles.refine_prism(0, [2,2,1])
        self.assertEqual(tiles.n, 4)
        self.assertAlmostEqual(tiles.M_rem[0], 1.0)

        # newly added tiles default to zero
        np.testing.assert_array_equal(tiles.M_rem[1:], np.zeros(3))

        # but u_ea _is_ inherited across all four
        u0 = tiles.u_ea[0]
        for idx in range(1, tiles.n):
            with self.subTest(tile=idx):
                np.testing.assert_allclose(tiles.u_ea[idx], u0)









if __name__ == "__main__":
    unittest.main()
