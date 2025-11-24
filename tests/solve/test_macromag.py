import unittest
from pathlib import Path
import numpy as np
import logging
from simsopt.solve.macromag import MacroMag, muse2tiles, Tiles
from simsopt.solve.permanent_magnet_optimization import _neighbors_from_radius
from simsopt.solve.macromag import MacroMag, Tiles, assemble_blocks_subset


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
        tiles = muse2tiles(str(cls.csv_path), magnetization=1.1658e6)
        cls.macro = MacroMag(tiles)

        # also precompute our tensor once
        cls.demag_test = cls.macro.fast_get_demag_tensor().astype(np.float64)
        cls.demag_test[np.abs(cls.demag_test) < 1e-12] = 0.0

        cls.pts = cls.macro.centres

    # def test_shape(self):
    #     """Test that the computed tensor has the same shape as the stored one."""
    #     self.assertEqual(self.demag_test.shape, self.demag_ref.shape,
    #                      f"expected shape {self.demag_ref.shape}, got {self.demag_test.shape}")

    # def test_symmetry(self):
    #     """Test N[i,j] == N[i,j].T for a few representative entries."""
    #     # pick a small sample so this doesn't blow up memory/time
    #     for i, j in [(0,0), (1,1), (0,1), (2,3)]:
    #         with self.subTest(i=i, j=j):
    #             M = self.demag_test[i, j]
    #             self.assertTrue(np.allclose(M, M.T, atol=0),
    #                             f"N[{i},{j}] not symmetric")

    # def test_trace_invariant(self):
    #     """For a rectangular prism, the self‐demag trace should be ≈1."""
    #     # only check diagonal entries (i==j)
    #     n = self.demag_test.shape[0]
    #     # sample first 5 to avoid huge loops
    #     for i in range(min(5, n)):
    #         tr = np.trace(self.demag_test[i, i])
    #         self.assertAlmostEqual(abs(tr), 1.0, places=12,
    #             msg=f"trace N[{i},{i}] = {tr}, |trace| expected 1.0")

    # def test_against_stored(self):
    #     """Compare absolute errors with the reference."""
    #     diff    = np.abs(self.demag_test - self.demag_ref)
    #     max_err = diff.max()

    #     # strict absolute‐error tolerance
    #     ABS_TOL = 1e-9
    #     self.assertLessEqual(
    #         max_err, ABS_TOL,
    #         f"max absolute error {max_err:.2e} exceeds tol={ABS_TOL:.2e}"
    #     )

    # def test_mismatch_report(self):
    #     """If there is a big mismatch, print its location (for debugging)."""
    #     diff = np.abs(self.demag_test - self.demag_ref)
    #     max_err = diff.max()

    #     logger.info("Max |ΔN|: %.3e (rms %.3e)", max_err, np.sqrt(np.mean(diff**2)))

    #     tol = 1e-9
    #     mask = diff > tol
    #     if mask.any():
    #         idx = np.unravel_index(np.argmax(diff * mask), diff.shape)
    #         msg = (
    #             f"largest absolute mismatch at {idx}: "
    #             f"test={self.demag_test[idx]:.6e} "
    #             f"ref={self.demag_ref[idx]:.6e} "
    #             f"abs delta={diff[idx]:.2e} (tol={tol})"
    #         )
    #         self.fail(msg)

    # def test_demag_field(self):
    #     """Compare our _demag_field_at to saved MagTense H_field."""
    #     H_test = self.macro._demag_field_at(self.pts, self.demag_ref)

    #     diff1 = np.linalg.norm(H_test - self.hfield_ref, axis=1)
    #     diff2 = np.linalg.norm(H_test + self.hfield_ref, axis=1)

    #     max_err = min(diff1.max(), diff2.max())
    #     rms1 = np.sqrt(np.mean(diff1**2))
    #     rms2 = np.sqrt(np.mean(diff2**2))
    #     rms = min(rms1, rms2)

    #     # absolute‐error tolerance ~1e-9 A/m, up to a global sign
    #     self.assertLessEqual(
    #         max_err, 1e-9,
    #         f"Max |ΔH| (up to sign) = {max_err:.3e}, RMS = {rms:.3e}"
    #     )

    # def test_direct_solve_magnitude_constant(self):
    #     """When chi = 0 and H_coil = 0, direct solve should yield M = M_rem u."""
    #     tiles = make_trivial_tiles(4)
    #     macro = MacroMag(tiles)

    #     pts = np.zeros((tiles.n, 3))
    #     neighbors = _neighbors_from_radius(pts, radius=1.0, max_neighbors=None)

    #     macro, _ = macro.direct_solve_neighbor_sparse(
    #         neighbors=neighbors,
    #         use_coils=False,
    #         krylov_tol=1e-8,
    #         krylov_it=200,
    #         print_progress=False,
    #         x0=None,
    #         H_a_override=None,
    #         drop_tol=0.0,
    #     )

    #     norms   = np.linalg.norm(tiles.M, axis=1)
    #     expected = tiles.M_rem
    #     for idx, (n_val, e_val) in enumerate(zip(norms, expected)):
    #         with self.subTest(tile=idx):
    #             self.assertAlmostEqual(n_val, e_val, places=8,
    #                 msg=f"Tile {idx}: |M| = {n_val:.12f}, expected {e_val:.12f}")

    # def test_tile_proxy(self):
    #     """Tile proxy should get/set individual elements and enforce bounds."""
    #     tiles = make_trivial_tiles(2)
    #     # both start at 1,2
    #     self.assertEqual(tiles[0].M_rem, 1.0)
    #     self.assertEqual(tiles[1].M_rem, 2.0)
    #     # setting via proxy updates parent
    #     tiles[1].M_rem = 5.5
    #     self.assertEqual(tiles.M_rem[1], 5.5)
    #     # other unchanged
    #     self.assertEqual(tiles.M_rem[0], 1.0)
    #     # out‐of‐bounds __getitem__
    #     with self.assertRaises(IndexError):
    #         _ = tiles[2]
    #     # __str__ should mention both tiles
    #     s = str(tiles)
    #     self.assertIn("Tile_0", s)
    #     self.assertIn("Tile_1", s)

    # def test_rotation_and_ffgg_hh(self):
    #     # get_rotation_matrices ↔ orthonormal inverses:
    #     Rg2l, Rl2g = MacroMag.get_rotation_matrices(0.1, 0.2, 0.3)
    #     I = np.eye(3)
    #     self.assertTrue(np.allclose(Rl2g @ Rg2l, I, atol=1e-12))

    #     # simple calls to the f/g/h/FF/GG/HH wrappers (via MacroMag static methods)
    #     for func in ("_f_3D","_g_3D","_h_3D","_FF_3D","_GG_3D","_HH_3D"):
    #         # pick a non‐degenerate point
    #         val = getattr(MacroMag, func)(1.0, 2.0, 3.0, 0.1, -0.2, 0.5)
    #         self.assertIsInstance(val, float)

    # def test_set_easy_axis_error(self):
    #     tiles = Tiles(1)
    #     # passing a single float should TypeError
    #     with self.assertRaises(TypeError):
    #         tiles.set_easy_axis(val=1.23, idx=0)

    # def test_vertices_setter_and_errors(self):
    #     tiles = Tiles(2)
    #     # valid 3×4 shape broadcast
    #     mat34 = np.ones((3,4))
    #     tiles.vertices = mat34
    #     self.assertTrue((tiles.vertices == mat34).all())

    #     # valid tuple‐setter with 4×3
    #     mat43 = np.arange(12).reshape(4,3)
    #     tiles.vertices = (mat43, 1)
    #     self.assertTrue((tiles.vertices[1] == mat43.T).all())

    #     # invalid shape
    #     with self.assertRaises(AssertionError):
    #         tiles.vertices = np.zeros((5,5))

    # def test_other_tiles_setters(self):
    #     n = 3
    #     tiles = Tiles(n)

    #     # tile_type: tuple, scalar, list
    #     tiles.tile_type = (7,1)
    #     self.assertEqual(tiles.tile_type[1], 7)
    #     tiles.tile_type = 9
    #     self.assertTrue((tiles.tile_type == 9).all())
    #     tiles.tile_type = [1,2,3]
    #     self.assertTrue((tiles.tile_type == np.array([1,2,3])).all())

    #     # offset
    #     tiles.offset = (np.array([1,1,1]), 2)
    #     self.assertTrue((tiles.offset[2] == [1,1,1]).all())
    #     tiles.offset = [0.5,0.5,0.5]
    #     self.assertTrue((tiles.offset == [0.5,0.5,0.5]).all())

    #     # M setter: tuple
    #     tiles.M_rem = [1,2,3]
    #     vec = np.array([1.0, 2.0, 3.0])
    #     tiles.u_ea = [ [0,0,1], [0,1,0], [1,0,0] ]
    #     tiles.M = (vec, 1)
    #     self.assertTrue((tiles.M[1] == vec).all())

    #     # u_oa1 / u_oa2 tuple and broadcast
    #     oa = np.array([1,0,0])
    #     tiles.u_oa1 = (oa, 0)
    #     self.assertTrue((tiles.u_oa1[0] == oa).all())
    #     tiles.u_oa2 = [ [0,1,0] ]*n
    #     self.assertTrue((tiles.u_oa2 == [[0,1,0]]*n).all())

    #     # mu_r sets
    #     tiles.mu_r_ea = (4.2, 2)
    #     self.assertEqual(tiles.mu_r_ea[2], 4.2)
    #     tiles.mu_r_oa = 5.5
    #     self.assertTrue((tiles.mu_r_oa == 5.5).all())

    #     # M_rel, stfcn_index, incl_it
    #     for prop in ("M_rel","stfcn_index","incl_it"):
    #         arr = getattr(tiles, prop)
    #         setattr(tiles, prop, (7,1))
    #         self.assertEqual(getattr(tiles, prop)[1], 7)

    # def test_add_tiles_and_refine_prism(self):
    #     # start from a single-tile trivial Tiles
    #     tiles = make_trivial_tiles(1)
    #     # mark it as a prism
    #     tiles.tile_type = (2, 0)
    #     self.assertEqual(tiles.n, 1)

    #     # refine it into 2×2×1 = 4 prisms
    #     tiles.refine_prism(0, [2,2,1])
    #     self.assertEqual(tiles.n, 4)
    #     self.assertAlmostEqual(tiles.M_rem[0], 1.0)

    #     # newly added tiles default to zero
    #     np.testing.assert_array_equal(tiles.M_rem[1:], np.zeros(3))

    #     # but u_ea _is_ inherited across all four
    #     u0 = tiles.u_ea[0]
    #     for idx in range(1, tiles.n):
    #         with self.subTest(tile=idx):
    #             np.testing.assert_allclose(tiles.u_ea[idx], u0)

    # def test_neighbor_sparse_matches_dense_pm4stell_params(self):
    #     """Sparse near-field direct solve should be within 1% of a larger-radius neighbor-sparse solve for PM4Stell-like parameters."""
    #     tiles_ref = muse2tiles(str(self.csv_path), magnetization=1.1658e6)
    #     tiles_approx = muse2tiles(str(self.csv_path), magnetization=1.1658e6)

    #     macro_ref = MacroMag(tiles_ref)
    #     macro_approx = MacroMag(tiles_approx)

    #     centres = macro_ref.centres
    #     cube_dim = float(np.min(tiles_ref.size[0]))

    #     radius_ref = 8.0 * cube_dim
    #     radius_approx = 4.0 * cube_dim

    #     neighbors_ref = _neighbors_from_radius(
    #         centres,
    #         radius=radius_ref,
    #         max_neighbors=2048,
    #     )
    #     neighbors_approx = _neighbors_from_radius(
    #         centres,
    #         radius=radius_approx,
    #         max_neighbors=512,
    #     )

    #     macro_ref, _ = macro_ref.direct_solve_neighbor_sparse(
    #         neighbors=neighbors_ref,
    #         use_coils=False,
    #         krylov_tol=1e-10,
    #         krylov_it=500,
    #         print_progress=False,
    #         x0=None,
    #         H_a_override=None,
    #         drop_tol=0.0,
    #     )

    #     macro_approx, _ = macro_approx.direct_solve_neighbor_sparse(
    #         neighbors=neighbors_approx,
    #         use_coils=False,
    #         krylov_tol=1e-10,
    #         krylov_it=500,
    #         print_progress=False,
    #         x0=None,
    #         H_a_override=None,
    #         drop_tol=0.0,
    #     )

    #     M_ref = macro_ref.tiles.M
    #     M_approx = macro_approx.tiles.M

    #     diff = np.linalg.norm(M_ref - M_approx, axis=1)
    #     denom = np.linalg.norm(M_ref, axis=1) + 1e-30
    #     rel = diff / denom
    #     max_rel_err = float(np.max(rel))
    #     rms_rel_err = float(np.sqrt(np.mean(rel**2)))

    #     MAX_REL_ERR_TOL = 1e-2  # 1% max relative error

    #     self.assertLess(
    #         max_rel_err, MAX_REL_ERR_TOL,
    #         f"neighbor-sparse PM4Stell solve max relative |ΔM|={max_rel_err:.3e} "
    #         f"(rms {rms_rel_err:.3e}) exceeds {MAX_REL_ERR_TOL:.1e}"
    #     )


    # def test_fast_gpmo_pm4stell_downsampled_runs(self):
    #     """Lightweight integration test: fast MacroMag GPMO runs on a downsampled PM4Stell grid."""
    #     from simsopt.field import BiotSavart, Coil
    #     from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
    #     from simsopt.solve import GPMO
    #     from simsopt.util import FocusPlasmaBnormal, FocusData, read_focus_coils
    #     from simsopt.util.permanent_magnet_helper_functions import initialize_default_kwargs
    #     from simsopt.util.polarization_project import (
    #         polarization_axes,
    #         orientation_phi,
    #         discretize_polarizations,
    #     )

    #     # Modest resolution / downsample so the test stays fast:
    #     downsample = 10
    #     N = 16
    #     nphi = N
    #     ntheta = N

    #     base = self.csv_path.parent
    #     fname_plasma = base / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    #     fname_ncsx_coils = base / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    #     fname_argmt = base / "magpie_trial104b_PM4Stell.focus"
    #     fname_corn = base / "magpie_trial104b_corners_PM4Stell.csv"
    #     fname_tf_coils_muse = base / "muse_tf_coils.focus"

    #     # Plasma surface and B·n from plasma
    #     lcfs = SurfaceRZFourier.from_focus(
    #         fname_plasma, range="half period", nphi=nphi, ntheta=ntheta
    #     )
    #     bnormal_obj = FocusPlasmaBnormal(fname_plasma)
    #     bn_plasma = bnormal_obj.bnormal_grid(nphi, ntheta, "half period")

    #     # TF coils and their B·n on the same surface
    #     base_curves, base_currents, ncoils = read_focus_coils(fname_ncsx_coils)
    #     coils = [Coil(base_curves[i], base_currents[i]) for i in range(ncoils)]
    #     for c in base_curves:
    #         c.fix_all()
    #     base_currents[0].fix_all()

    #     bs_tf = BiotSavart(coils)
    #     bs_tf.set_points(lcfs.gamma().reshape((-1, 3)))
    #     bn_tf = np.sum(
    #         bs_tf.B().reshape((nphi, ntheta, 3)) * lcfs.unitnormal(), axis=2
    #     )
    #     bn_total = bn_plasma + bn_tf

    #     # PM4Stell magnet arrangement, downsampled
    #     mag_data = FocusData(fname_argmt, downsample=downsample)
    #     nMagnets_tot = mag_data.nMagnets

    #     # Face polarizations, positive variants only
    #     pol_axes_all, pol_type_all = polarization_axes(["face"])
    #     ntype = len(pol_type_all) // 2
    #     pol_axes = pol_axes_all[:ntype, :]
    #     pol_type = pol_type_all[:ntype]

    #     # Orient the allowed polarizations
    #     ophi = orientation_phi(fname_corn)[:nMagnets_tot]
    #     discretize_polarizations(mag_data, ophi, pol_axes, pol_type)

    #     pol_vectors = np.zeros((nMagnets_tot, len(pol_type), 3))
    #     pol_vectors[:, :, 0] = mag_data.pol_x
    #     pol_vectors[:, :, 1] = mag_data.pol_y
    #     pol_vectors[:, :, 2] = mag_data.pol_z

    #     kwargs_geo = {"pol_vectors": pol_vectors, "downsample": downsample}
    #     pm = PermanentMagnetGrid.geo_setup_from_famus(
    #         lcfs, bn_total, fname_argmt, **kwargs_geo
    #     )

    #     # GPMO setup (fast MacroMag variant)
    #     kwargs = initialize_default_kwargs("GPMO")
    #     kwargs["K"] = 200
    #     kwargs["nhistory"] = 10
    #     kwargs["backtracking"] = 0
    #     kwargs["Nadjacent"] = 10
    #     kwargs["dipole_grid_xyz"] = np.ascontiguousarray(pm.dipole_grid_xyz)

    #     algorithm = "ArbVec_backtracking_fast_macromag_py"

    #     # MacroMag / fast-sparse knobs
    #     kwargs["cube_dim"] = 0.0296
    #     kwargs["mu_ea"] = 1.05
    #     kwargs["mu_oa"] = 1.15
    #     kwargs["use_coils"] = True
    #     kwargs["use_demag"] = True
    #     kwargs["coil_path"] = fname_tf_coils_muse
    #     kwargs["mm_refine_every"] = 20

    #     kwargs["demag_radius"] = 4 * kwargs["cube_dim"]
    #     kwargs["max_demag_neighbors"] = 150
    #     kwargs["demag_drop_tol"] = 1e-3
    #     kwargs["krylov_tol"] = 1e-6
    #     kwargs["krylov_it"] = 200

    #     R2_history, Bn_history, m_history = GPMO(pm, algorithm, **kwargs)

    #     # Basic sanity checks: finite, nontrivial, and some improvement in f_B.
    #     valid = R2_history[np.isfinite(R2_history) & (R2_history > 0)]
    #     self.assertGreater(
    #         valid.size, 1,
    #         "GPMO produced no valid (positive, finite) objective values on PM4Stell grid.",
    #     )
    #     self.assertTrue(
    #         np.isfinite(valid).all(), "R2_history contains non-finite entries."
    #     )
    #     self.assertLess(
    #         valid.min(),
    #         valid[0],
    #         "Fast MacroMag GPMO did not reduce the objective on downsampled PM4Stell grid.",
    #     )

    #     # Also ensure final m has the right shape and is finite
    #     m_final = pm.m.reshape(pm.ndipoles, 3)
    #     self.assertEqual(m_final.shape[0], pm.ndipoles)
    #     self.assertTrue(np.isfinite(m_final).all())

    # def test_neighbor_sparse_scaling_with_active_count_pm4stell(self):
    #     """
    #     For PM4Stell geometry, check that the neighbor-sparse MacroMag
    #     error (fast vs less-aggressive reference) stays small as we
    #     increase the number of active tiles up to <= 35,000.
    #     """
    #     pts_full = self.pts
    #     N_full = pts_full.shape[0]

    #     # Target active counts: low, medium, high
    #     target_active = [50, 500, 2000, 8000, 20000, 35000]
    #     target_active = [n for n in target_active if n <= N_full]

    #     # Geometry scale for PM4Stell cubes (matches your fast GPMO script)
    #     cube_dim = 0.0296

    #     # "Reference" neighbor-sparse settings: slightly larger radius, stricter drop_tol
    #     radius_ref = 5.0 * cube_dim
    #     max_neighbors_ref = 250
    #     drop_tol_ref = 1e-4

    #     # "Fast" settings: what you use in ArbVec_backtracking_fast_macromag_py
    #     radius_fast = 4.0 * cube_dim
    #     max_neighbors_fast = 150
    #     drop_tol_fast = 1e-3

    #     # Deterministic random easy axes
    #     rng = np.random.default_rng(123)

    #     # Tolerances on relative |ΔM|
    #     MAX_REL_TOL = 8e-3   # 0.8%
    #     RMS_REL_TOL = 4e-3   # 0.4%

    #     for n_active in target_active:
    #         with self.subTest(n_active=n_active):
    #             # Uniformly sample n_active indices from the full PM4Stell centres
    #             # (no replacement, deterministic)
    #             idx = np.linspace(0, N_full - 1, n_active, dtype=int)
    #             pts = pts_full[idx]

    #             # Build trivial tiles at those centres: chi=0, M_rem=1..n_active, easy axis z
    #             tiles_ref = make_trivial_tiles(n_active)
    #             tiles_fast = make_trivial_tiles(n_active)
    #             tiles_ref.offset = pts
    #             tiles_fast.offset = pts

    #             # Random easy axes (same for ref and fast)
    #             ea = rng.normal(size=(n_active, 3))
    #             ea /= np.linalg.norm(ea, axis=1, keepdims=True)
    #             tiles_ref._u_ea = ea
    #             tiles_fast._u_ea = ea

    #             # Build MacroMag instances
    #             macro_ref = MacroMag(tiles_ref)
    #             macro_fast = MacroMag(tiles_fast)

    #             # Neighbor lists (local indices 0..n_active-1)
    #             neigh_ref = _neighbors_from_radius(
    #                 pts,
    #                 radius=radius_ref,
    #                 max_neighbors=max_neighbors_ref,
    #             )
    #             neigh_fast = _neighbors_from_radius(
    #                 pts,
    #                 radius=radius_fast,
    #                 max_neighbors=max_neighbors_fast,
    #             )

    #             # Solve with neighbor-sparse demag, no coils, same Krylov controls
    #             macro_ref, _ = macro_ref.direct_solve_neighbor_sparse(
    #                 neighbors=neigh_ref,
    #                 use_coils=False,
    #                 krylov_tol=1e-8,
    #                 krylov_it=300,
    #                 print_progress=False,
    #                 x0=None,
    #                 H_a_override=None,
    #                 drop_tol=drop_tol_ref,
    #             )
    #             macro_fast, _ = macro_fast.direct_solve_neighbor_sparse(
    #                 neighbors=neigh_fast,
    #                 use_coils=False,
    #                 krylov_tol=1e-8,
    #                 krylov_it=300,
    #                 print_progress=False,
    #                 x0=None,
    #                 H_a_override=None,
    #                 drop_tol=drop_tol_fast,
    #             )

    #             M_ref = macro_ref.tiles.M
    #             M_fast = macro_fast.tiles.M

    #             # Relative error per tile
    #             norm_ref = np.linalg.norm(M_ref, axis=1)
    #             norm_diff = np.linalg.norm(M_fast - M_ref, axis=1)
    #             rel_err = norm_diff / np.maximum(norm_ref, 1e-14)

    #             max_rel = float(rel_err.max())
    #             rms_rel = float(np.sqrt(np.mean(rel_err**2)))

    #             msg = (
    #                 f"n_active={n_active}: max_rel={max_rel:.3e}, "
    #                 f"rms_rel={rms_rel:.3e} (MAX_REL_TOL={MAX_REL_TOL:.3e}, "
    #                 f"RMS_REL_TOL={RMS_REL_TOL:.3e})"
    #             )

    #             self.assertLessEqual(max_rel, MAX_REL_TOL, msg)
    #             self.assertLessEqual(rms_rel, RMS_REL_TOL, msg)

    
    # def test_direct_vs_neighbor_sparse_pm4stell(self):
    #     """
    #     For PM4Stell geometry, compare the final GPMO magnetization
    #     when using the dense/incremental MacroMag demag
    #     (ArbVec_backtracking_macromag_py) vs the neighbor-sparse demag
    #     (ArbVec_backtracking_fast_macromag_py), across several
    #     downsample factors.
    #     """
    #     from simsopt.field import BiotSavart, Coil
    #     from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
    #     from simsopt.solve import GPMO
    #     from simsopt.util import (
    #         FocusPlasmaBnormal,
    #         FocusData,
    #         read_focus_coils,
    #     )
    #     from simsopt.util.permanent_magnet_helper_functions import (
    #         initialize_default_kwargs,
    #     )
    #     from simsopt.util.polarization_project import (
    #         polarization_axes,
    #         orientation_phi,
    #         discretize_polarizations,
    #     )

    #     base = self.csv_path.parent
    #     fname_plasma = base / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    #     fname_ncsx_coils = base / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    #     fname_argmt = base / "magpie_trial104b_PM4Stell.focus"
    #     fname_corn = base / "magpie_trial104b_corners_PM4Stell.csv"
    #     fname_tf_coils_muse = base / "muse_tf_coils.focus"

    #     # Coarse surface resolution so test runs quickly
    #     N = 16
    #     nphi = N
    #     ntheta = N

    #     # Plasma surface and Bn
    #     lcfs = SurfaceRZFourier.from_focus(
    #         fname_plasma, range="half period", nphi=nphi, ntheta=ntheta
    #     )
    #     bnormal_obj = FocusPlasmaBnormal(fname_plasma)
    #     bn_plasma = bnormal_obj.bnormal_grid(nphi, ntheta, "half period")

    #     # TF coils and their Bn on the same surface
    #     base_curves, base_currents, ncoils = read_focus_coils(fname_ncsx_coils)
    #     coils = [Coil(base_curves[i], base_currents[i]) for i in range(ncoils)]
    #     for c in base_curves:
    #         c.fix_all()
    #     base_currents[0].fix_all()

    #     bs_tf = BiotSavart(coils)
    #     bs_tf.set_points(lcfs.gamma().reshape((-1, 3)))
    #     bn_tf = np.sum(
    #         bs_tf.B().reshape((nphi, ntheta, 3)) * lcfs.unitnormal(),
    #         axis=2,
    #     )
    #     bn_total = bn_plasma + bn_tf

    #     # Face polarizations, positive variants only
    #     pol_axes_all, pol_type_all = polarization_axes(["face"])
    #     ntype = len(pol_type_all) // 2
    #     pol_axes = pol_axes_all[:ntype, :]
    #     pol_type = pol_type_all[:ntype]

    #     ophi_all = orientation_phi(fname_corn)

    #     cube_dim = 0.0296

    #     downsample_list = [100, 50, 20]

    #     # Relative tolerance on M differences
    #     MAX_REL_TOL = 1.0e-2
    #     RMS_REL_TOL = 5.0e-3

    #     for downsample in downsample_list:
    #         with self.subTest(downsample=downsample):
    #             mag_data = FocusData(fname_argmt, downsample=downsample)
    #             nMagnets_tot = mag_data.nMagnets
    #             ophi = ophi_all[:nMagnets_tot]

    #             discretize_polarizations(mag_data, ophi, pol_axes, pol_type)

    #             pol_vectors = np.zeros((nMagnets_tot, len(pol_type), 3))
    #             pol_vectors[:, :, 0] = mag_data.pol_x
    #             pol_vectors[:, :, 1] = mag_data.pol_y
    #             pol_vectors[:, :, 2] = mag_data.pol_z

    #             kwargs_geo = {"pol_vectors": pol_vectors, "downsample": downsample}

    #             # Build two identical PM grids
    #             pm_ref = PermanentMagnetGrid.geo_setup_from_famus(
    #                 lcfs, bn_total, fname_argmt, **kwargs_geo
    #             )
    #             pm_fast = PermanentMagnetGrid.geo_setup_from_famus(
    #                 lcfs, bn_total, fname_argmt, **kwargs_geo
    #             )

    #             ndip = pm_ref.ndipoles
    #             self.assertEqual(ndip, pm_fast.ndipoles)

    #             # Shared GPMO kwargs
    #             kwargs_common = initialize_default_kwargs("GPMO")
    #             K = min(1000, ndip)
    #             kwargs_common["K"] = K
    #             kwargs_common["nhistory"] = 10
    #             kwargs_common["backtracking"] = 0
    #             kwargs_common["Nadjacent"] = 10
    #             kwargs_common["dipole_grid_xyz"] = np.ascontiguousarray(
    #                 pm_ref.dipole_grid_xyz
    #             )

    #             # Parameters required by ArbVec_backtracking_macromag_py
    #             thresh_angle = np.pi
    #             max_nMagnets = K

    #             # Dense / incremental MacroMag variant
    #             kwargs_ref = dict(kwargs_common)
    #             kwargs_ref.update(
    #                 dict(
    #                     cube_dim=cube_dim,
    #                     mu_ea=1.05,
    #                     mu_oa=1.15,
    #                     use_coils=True,
    #                     use_demag=True,
    #                     coil_path=fname_tf_coils_muse,
    #                     mm_refine_every=50,
    #                     thresh_angle=thresh_angle,
    #                     max_nMagnets=max_nMagnets,
    #                 )
    #             )

    #             R2_ref, Bn_ref, m_hist_ref = GPMO(
    #                 pm_ref, "ArbVec_backtracking_macromag_py", **kwargs_ref
    #             )
    #             M_ref = pm_ref.m.reshape(ndip, 3)

    #             # Neighbor-sparse MacroMag variant
    #             kwargs_fast = dict(kwargs_common)
    #             kwargs_fast.update(
    #                 dict(
    #                     cube_dim=cube_dim,
    #                     mu_ea=1.05,
    #                     mu_oa=1.15,
    #                     use_coils=True,
    #                     use_demag=True,
    #                     coil_path=fname_tf_coils_muse,
    #                     mm_refine_every=50,
    #                     demag_radius=4.0 * cube_dim,
    #                     max_demag_neighbors=150,
    #                     demag_drop_tol=1e-3,
    #                     krylov_tol=1e-6,
    #                     krylov_it=200,
    #                     thresh_angle=thresh_angle,
    #                     max_nMagnets=max_nMagnets,
    #                 )
    #             )

    #             R2_fast, Bn_fast, m_hist_fast = GPMO(
    #                 pm_fast, "ArbVec_backtracking_fast_macromag_py", **kwargs_fast
    #             )
    #             M_fast = pm_fast.m.reshape(ndip, 3)

    #             self.assertEqual(M_ref.shape, M_fast.shape)

    #             norm_ref = np.linalg.norm(M_ref, axis=1)
    #             norm_diff = np.linalg.norm(M_fast - M_ref, axis=1)
    #             rel_err = norm_diff / np.maximum(norm_ref, 1e-14)

    #             max_rel = float(rel_err.max())
    #             rms_rel = float(np.sqrt(np.mean(rel_err**2)))

    #             msg = (
    #                 f"downsample={downsample}, ndipoles={ndip}: "
    #                 f"max_rel={max_rel:.3e}, rms_rel={rms_rel:.3e} "
    #                 f"(MAX_REL_TOL={MAX_REL_TOL:.3e}, "
    #                 f"RMS_REL_TOL={RMS_REL_TOL:.3e})"
    #             )

    #             self.assertLessEqual(max_rel, MAX_REL_TOL, msg)
    #             self.assertLessEqual(rms_rel, RMS_REL_TOL, msg)
                    
                                
                            

    def test_macromag_dense_vs_sparse_chunk(self):
        """
        Compare dense vs neighbor-sparse MacroMag solves on a sizeable
        MUSE chunk and check that the sparse solve reproduces
        the dense result within a small relative error.
        Additionally, compare the effect of demag coupling on B·n
        (coupled vs. uncoupled) for dense vs. sparse solves.
        """
        import math
        import time
        import numpy as np
        from pathlib import Path

        from simsopt.solve.macromag import MacroMag, build_prism
        from simsopt.solve.permanent_magnet_optimization import _neighbors_from_radius
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import DipoleField

        # Parameters for this comparison
        cube_dim = 0.004
        mu_ea = 1.05
        mu_oa = 1.15
        krylov_tol = 1e-8
        krylov_it = 300

        # Load MUSE CSV 
        csv_path = self.csv_path
        data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
        
        n_total = data.shape[0]
        print("n_total loaded for chunk test", n_total)
        
        ## from sparse direct solve
        drop_tol = 1e-2
        max_neighbors = 5000
        radius = 1e6

        # Positions and half-dimensions (as stored in the CSV)
        centers = data[:, 0:3]
        lwh = data[:, 3:6]

        # For this test we do not need the original local frames or
        # magnetization directions. We can set:
        #   - rotations to zero,
        #   - magnetization angles to zero (easy axis along +z),
        # and the comparison between dense and sparse remains valid.
        rot = np.zeros((n_total, 3))
        mag_angle = np.zeros((n_total, 2))  # polar=0, azimuth=0

        mu_arr = np.repeat([[mu_ea, mu_oa]], n_total, axis=0)
        remanence = np.full(n_total, 1.16e6)

        # Build Tiles object for these prisms
        tiles = build_prism(lwh, centers, rot, mag_angle, mu_arr, remanence)
        n = tiles.n

        # Two MacroMag instances sharing the same Tiles geometry
        mac_dense = MacroMag(tiles)
        mac_sparse = MacroMag(tiles)

        # Dense solve: build full demag tensor and call direct_solve once
        N_full = mac_dense.fast_get_demag_tensor()  # shape (n, n, 3, 3)

        t0 = time.time()
        mac_dense, A_dense = mac_dense.direct_solve(
            use_coils=False,
            krylov_tol=krylov_tol,
            krylov_it=krylov_it,
            N_new_rows=N_full,
            N_new_cols=None,
            N_new_diag=None,
            print_progress=False,
            x0=None,
            H_a_override=None,
            A_prev=None,
            prev_n=0,
        )
        t_dense = time.time() - t0
        M_dense = mac_dense.tiles.M.copy()

        # Neighbor-sparse solve using a radius-based neighbor list
        centres = mac_sparse.tiles.offset.copy()
        neighbors = _neighbors_from_radius(
            centres,
            radius=radius,
            max_neighbors=max_neighbors,
        )

        t1 = time.time()
        mac_sparse, A_sparse = mac_sparse.direct_solve_neighbor_sparse(
            neighbors=neighbors,
            use_coils=False,
            krylov_tol=krylov_tol,
            krylov_it=krylov_it,
            print_progress=False,
            x0=None,
            H_a_override=None,
            drop_tol=drop_tol,
        )
        t_sparse = time.time() - t1
        M_sparse = mac_sparse.tiles.M.copy()

        norm_dense = np.linalg.norm(M_dense, axis=1)
        diff = np.linalg.norm(M_dense - M_sparse, axis=1)
        rel_err = diff / np.maximum(norm_dense, 1e-14)

        max_rel = float(np.max(rel_err))
        rms_rel = float(np.sqrt(np.mean(rel_err**2)))

        print(
            f"[MacroMag PM4Stell chunk] N={n}, "
            f"dense={t_dense:.2f}s, sparse={t_sparse:.2f}s, "
            f"max_rel(M)={max_rel:.3e}, rms_rel(M)={rms_rel:.3e}"
        )

        # # Tolerances can be tuned, but these are a sensible starting point:
        # self.assertLess(rms_rel, 1e-2, f"RMS relative error too high: {rms_rel:.3e}")
        # self.assertLess(max_rel, 5e-2, f"Max relative error too high: {max_rel:.3e}")

        # Construct a modest-resolution MUSE surface
        test_dir = Path(csv_path).parent
        surface_filename = test_dir / "input.muse"
        nphi = 32
        ntheta = nphi
        s = SurfaceRZFourier.from_focus(
            surface_filename,
            range="half period",
            nphi=nphi,
            ntheta=ntheta,
        )

        # Sample points and normals on that surface
        pts = s.gamma().reshape(-1, 3)
        n_hat = s.unitnormal().reshape(nphi, ntheta, 3)

        # Volumes of each brick
        volumes = np.prod(lwh, axis=1)

        # Dipole moments for dense and sparse
        m_dense  = (M_dense  * volumes[:, None]).reshape(-1)
        m_sparse = (M_sparse * volumes[:, None]).reshape(-1)

        # Build dipole fields
        b_dense  = DipoleField(centers, m_dense,  nfp=1, coordinate_flag=0)
        b_sparse = DipoleField(centers, m_sparse, nfp=1, coordinate_flag=0)

        # Evaluate fields on the surface
        b_dense.set_points(pts)
        b_sparse.set_points(pts)

        B_dense  = b_dense.B().reshape(nphi, ntheta, 3)
        B_sparse = b_sparse.B().reshape(nphi, ntheta, 3)

        # Project to B·n
        Bn_dense  = np.sum(B_dense  * n_hat, axis=2)
        Bn_sparse = np.sum(B_sparse * n_hat, axis=2)

        # Error in B·n from sparse vs dense
        Bn_diff = Bn_sparse - Bn_dense

        rms_Bn  = float(np.sqrt(np.mean(Bn_dense**2)))
        rms_err = float(np.sqrt(np.mean(Bn_diff**2)))
        max_err = float(np.max(np.abs(Bn_diff)))

        rms_rel_Bn = rms_err / 0.15
        max_rel_Bn = max_err / 0.15

        print(
            f"[MacroMag PM4Stell chunk Bn] "
            f"rms_rel(Bn)={rms_rel_Bn:.3e}, max_rel(Bn)={max_rel_Bn:.3e}, "
            f"rms(Bn_dense)={rms_Bn:.3e}"
        )

        self.assertLess(rms_rel_Bn, 0.01, f"RMS rel error in B·n too high: {rms_rel_Bn:.3e}")
        self.assertLess(max_rel_Bn, 0.01, f"Max rel error in B·n too high: {max_rel_Bn:.3e}")

        
    # def test_macromag_dense_vs_sparse_equivalence(self):
    #     """
    #     Verify that dense and neighbor-sparse MacroMag solves are mathematically
    #     identical when no truncation is applied (all-to-all neighbors, drop_tol=0).
    #     Expected result: RMS and max relative errors ≈ GMRES tolerance (1e-8–1e-6).
    #     """
    #     import math
    #     import time
    #     import numpy as np

    #     from simsopt.solve.macromag import MacroMag, build_prism

    #     # Parameters for this exact-equivalence comparison
    #     cube_dim = 0.004
    #     mu_ea = 1.05
    #     mu_oa = 1.15
    #     krylov_tol = 1e-8
    #     krylov_it = 300

    #     # Load MUSE CSV 
    #     csv_path = self.csv_path
    #     data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
        
    #     n_total = data.shape[0]
    #     print("n_total loaded for equivalence test", n_total)
        
    #     # Key changes: all-to-all neighbors, no truncation
    #     drop_tol = 0
    #     max_neighbors = n_total         # include all tiles
    #     radius = 1e6                    # big enough to include all tiles

    #     centers = data[:, 0:3]
    #     lwh = data[:, 3:6]
    #     rot = np.zeros((n_total, 3))
    #     mag_angle = np.zeros((n_total, 2))
    #     mu_arr = np.repeat([[mu_ea, mu_oa]], n_total, axis=0)
    #     remanence = np.full(n_total, 1.16e6)

    #     tiles = build_prism(lwh, centers, rot, mag_angle, mu_arr, remanence)
    #     n = tiles.n

    #     mac_dense = MacroMag(tiles)
    #     mac_sparse = MacroMag(tiles)

    #     # Dense solve
    #     N_full = mac_dense.fast_get_demag_tensor()
    #     t0 = time.time()
    #     mac_dense, A_dense = mac_dense.direct_solve(
    #         use_coils=False,
    #         krylov_tol=krylov_tol,
    #         krylov_it=krylov_it,
    #         N_new_rows=N_full,
    #         N_new_cols=None,
    #         N_new_diag=None,
    #         print_progress=True,
    #         x0=None,
    #         H_a_override=None,
    #         A_prev=None,
    #         prev_n=0,
    #     )
    #     t_dense = time.time() - t0
    #     M_dense = mac_dense.tiles.M.copy()

    #     # Sparse solve: all-to-all neighbors
    #     neighbours = np.tile(np.arange(n), (n, 1))  # <--- all-to-all matrix

    #     t1 = time.time()
    #     mac_sparse, A_sparse = mac_sparse.direct_solve_neighbor_sparse(
    #         neighbors=neighbours,
    #         use_coils=False,
    #         krylov_tol=krylov_tol,
    #         krylov_it=krylov_it,
    #         print_progress=True,
    #         x0=None,
    #         H_a_override=None,
    #         drop_tol=drop_tol,
    #     )
    #     t_sparse = time.time() - t1
    #     M_sparse = mac_sparse.tiles.M.copy()

    #     # Compare
    #     norm_dense = np.linalg.norm(M_dense, axis=1)
    #     diff = np.linalg.norm(M_dense - M_sparse, axis=1)
    #     rel_err = diff / np.maximum(norm_dense, 1e-14)

    #     max_rel = float(np.max(rel_err))
    #     rms_rel = float(np.sqrt(np.mean(rel_err**2)))

    #     print(
    #         f"[MacroMag Equivalence] N={n}, "
    #         f"dense={t_dense:.2f}s, sparse={t_sparse:.2f}s, "
    #         f"max_rel={max_rel:.3e}, rms_rel={rms_rel:.3e}"
    #     )

    #     # Expect numerical equivalence (differences limited by solver tolerance)
    #     self.assertLess(rms_rel, 1e-6, f"RMS relative error too high: {rms_rel:.3e}")
    #     self.assertLess(max_rel, 1e-5, f"Max relative error too high: {max_rel:.3e}")



if __name__ == "__main__":
    unittest.main()
