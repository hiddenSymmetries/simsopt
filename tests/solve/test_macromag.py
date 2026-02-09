import unittest
import json
import time
import os
import tempfile
import numpy as np
from pathlib import Path
import logging
from typing import Optional

# Matplotlib is imported at module scope by parts of simsopt, but some CI / sandboxed
# environments lack writable default cache directories. Pre-set a writable config
# dir to avoid noisy warnings during import.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-simsopt-tests-"))

from simsopt.solve.macromag import (
    MacroMag,
    Tiles,
    assemble_blocks_subset,
    assemble_blocks_subset_dispatch,
    muse2tiles,
)

logger = logging.getLogger(__name__)


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

    def test_muse_reference_tensor_subset_jax(self):
        """
        Validate the experimental JAX demag-block assembly against the same MAGTENSE reference subset.

        This is intentionally the same numerical target as
        :meth:`test_muse_reference_tensor_subset`, but routed through the JAX
        backend to ensure the two implementations remain consistent.
        """
        with open(self.ref_subset_path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        idx = np.asarray(ref["indices"], dtype=np.int64)
        ref_blocks = np.asarray(ref["blocks"], dtype=np.float64)

        tiles = muse2tiles(str(self.csv_path), magnetization=1.1658e6)
        macro = MacroMag(tiles)

        blocks = assemble_blocks_subset_dispatch(
            macro.centres, macro.half, macro.Rg2l, idx, idx, backend="jax"
        )
        np.testing.assert_allclose(blocks, ref_blocks, atol=1e-12, rtol=0.0)

    def test_assemble_blocks_subset_dispatch_benchmark(self):
        """
        Compare Numba vs JAX demag-block assembly for correctness and basic timing.

        Notes
        -----
        Performance depends heavily on the machine and configuration (OpenMP/BLAS,
        Numba thread settings, and JAX/XLA versions). This test prints a small
        timing summary and the maximum absolute deviation between backends. For
        more representative timings, run locally with a larger ``n``.

        This test also writes a simple scaling plot of assembly time vs ``n`` to
        help evaluate the Numba vs JAX tradeoff for larger optimization runs.
        """

        import simsopt.solve.macromag as macromag

        rng = np.random.default_rng(0)
        nJ = 128
        n_values = [1000, 5000, 10000, 20000]

        def _time_backend(n_tiles: int, *, backend: str) -> tuple[float, float]:
            offsets = rng.normal(scale=0.05, size=(n_tiles, 3))
            tiles = make_trivial_tiles(n_tiles, cube_dim=0.01, offsets=offsets)
            macro = MacroMag(tiles)
            I = np.arange(n_tiles, dtype=np.int64)
            J = np.arange(min(n_tiles, nJ), dtype=np.int64)

            t0 = time.perf_counter()
            _ = assemble_blocks_subset_dispatch(macro.centres, macro.half, macro.Rg2l, I, J, backend=backend)
            t1 = time.perf_counter()
            _ = assemble_blocks_subset_dispatch(macro.centres, macro.half, macro.Rg2l, I, J, backend=backend)
            t2 = time.perf_counter()
            return (t1 - t0), (t2 - t1)

        # Scaling timings: cold includes compilation for a new shape (JAX) or first-ever call (Numba).
        numba_cold: list[float] = []
        numba_warm: list[float] = []
        jax_cold: list[float] = []
        jax_warm: list[float] = []

        print(f"[bench] numba installed: {macromag._HAS_NUMBA}")
        for n_tiles in n_values:
            if macromag._HAS_NUMBA:
                t_without_precomp, t_with_precomp = _time_backend(n_tiles, backend="numba")
                numba_cold.append(t_without_precomp)
                numba_warm.append(t_with_precomp)
                print(
                    f"[bench] (n={n_tiles}, nJ={nJ}) "
                    f"numba without_precomp={t_without_precomp:.3f}s "
                    f"with_precomp={t_with_precomp:.3f}s"
                )
            else:
                numba_cold.append(float("nan"))
                numba_warm.append(float("nan"))

            t_without_precomp, t_with_precomp = _time_backend(n_tiles, backend="jax")
            jax_cold.append(t_without_precomp)
            jax_warm.append(t_with_precomp)
            print(
                f"[bench] (n={n_tiles}, nJ={nJ}) "
                f"jax without_precomp={t_without_precomp:.3f}s "
                f"with_precomp={t_with_precomp:.3f}s"
            )

        # Small correctness + cold/warm sanity print (first call includes compilation).
        n = 16
        offsets = rng.normal(scale=0.05, size=(n, 3))
        tiles = make_trivial_tiles(n, cube_dim=0.01, offsets=offsets)
        macro = MacroMag(tiles)
        I = np.arange(n, dtype=np.int64)
        J = np.arange(n, dtype=np.int64)

        t0 = time.perf_counter()
        blocks_numba = assemble_blocks_subset_dispatch(macro.centres, macro.half, macro.Rg2l, I, J, backend="numba")
        t1 = time.perf_counter()
        _ = assemble_blocks_subset_dispatch(macro.centres, macro.half, macro.Rg2l, I, J, backend="numba")
        t2 = time.perf_counter()

        t3 = time.perf_counter()
        blocks_jax = assemble_blocks_subset_dispatch(macro.centres, macro.half, macro.Rg2l, I, J, backend="jax")
        t4 = time.perf_counter()
        _ = assemble_blocks_subset_dispatch(macro.centres, macro.half, macro.Rg2l, I, J, backend="jax")
        t5 = time.perf_counter()

        max_abs = float(np.max(np.abs(blocks_numba - blocks_jax)))
        denom = float(np.max(np.abs(blocks_numba)))
        max_rel = max_abs / denom if denom > 0 else float("nan")

        print(f"[bench] max abs(N_numba - N_jax): {max_abs:.3e}")
        print(f"[bench] max rel(N_numba - N_jax): {max_rel:.3e}")
        print(f"[bench] assemble_blocks_subset (numba) first call:  {t1 - t0:.3f} s")
        print(f"[bench] assemble_blocks_subset (numba) second call: {t2 - t1:.3f} s")
        print(f"[bench] assemble_blocks_subset (jax) first call:    {t4 - t3:.3f} s")
        print(f"[bench] assemble_blocks_subset (jax) second call:   {t5 - t4:.3f} s")

        np.testing.assert_allclose(blocks_jax, blocks_numba, atol=1e-12, rtol=0.0)

        # Plot scaling results. This is a diagnostic artifact only; do not assert on timings.
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        x = np.asarray(n_values, dtype=np.int64)
        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

        ax.plot(x, jax_cold, marker="o", linestyle="--", label="JAX without precompilation")
        ax.plot(x, jax_warm, marker="o", linestyle="-", label="JAX with precompilation")

        if macromag._HAS_NUMBA:
            ax.plot(x, numba_cold, marker="s", linestyle="--", label="Numba without precompilation")
            ax.plot(x, numba_warm, marker="s", linestyle="-", label="Numba with precompilation")

        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        ax.set_yscale("log")
        ax.set_xlabel(r"$n = |I|$ (tiles)")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(rf"Subset assembly: $|I|=n$, $|J|={nJ}$ (total blocks $=|I||J|$)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(loc="best")

        out_png = Path(tempfile.mkdtemp(prefix="macromag-bench-")) / "assemble_blocks_subset_scaling.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[bench] wrote scaling plot: {out_png}")

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
