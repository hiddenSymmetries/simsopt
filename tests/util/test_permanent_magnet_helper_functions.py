"""
Unit tests for permanent_magnet_helper_functions module.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.field import BiotSavart
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.solve import GPMO
from simsopt.util import (
    FocusData,
    build_polarization_vectors_from_focus_data,
    compute_final_squared_flux,
    compute_m_maxima_and_cube_dim_from_focus_file,
    get_utc_timestamp_iso,
    initialize_coils_for_pm_optimization,
    initialize_default_kwargs,
    run_gpmo_optimization_on_muse_grid,
    write_gpmo_run_history_csv,
    write_gpmo_run_metadata_yaml,
)

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


class TestGetUtcTimestampIso(unittest.TestCase):
    """Tests for get_utc_timestamp_iso."""

    def test_returns_iso_format_string(self):
        ts = get_utc_timestamp_iso()
        self.assertIsInstance(ts, str)
        # ISO 8601 format: YYYY-MM-DDTHH:MM:SS+00:00
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}$"
        self.assertRegex(ts, pattern)


class TestWriteGpmoRunHistoryCsv(unittest.TestCase):
    """Tests for write_gpmo_run_history_csv."""

    def test_writes_csv_with_defaults(self):
        with ScratchDir("."):
            out_dir = Path("test_output")
            run_id = "test_run_001"
            fB_hist = np.array([1e-3, 5e-4, 2e-4])
            absBn_hist = np.array([0.1, 0.08, 0.06])

            class MockPmOpt:
                pass

            pm_opt = MockPmOpt()
            path = write_gpmo_run_history_csv(out_dir, run_id, pm_opt, fB_hist, absBn_hist)

            self.assertTrue(path.exists())
            self.assertEqual(path.name, "runhistory_test_run_001.csv")
            content = path.read_text()
            self.assertIn("k,fB,absBn,n_active", content)
            self.assertIn("0,0.001,0.1,0", content)

    def test_writes_csv_with_k_history_and_num_nonzeros(self):
        with ScratchDir("."):
            out_dir = Path("test_output")
            run_id = "test_run_002"
            fB_hist = np.array([1e-3, 5e-4])
            absBn_hist = np.array([0.1, 0.08])

            class MockPmOpt:
                k_history = np.array([10, 20])
                num_nonzeros = np.array([5, 10])

            pm_opt = MockPmOpt()
            path = write_gpmo_run_history_csv(out_dir, run_id, pm_opt, fB_hist, absBn_hist)

            content = path.read_text()
            self.assertIn("10,0.001,0.1,5", content)
            self.assertIn("20,0.0005,0.08,10", content)


class TestWriteGpmoRunMetadataYaml(unittest.TestCase):
    """Tests for write_gpmo_run_metadata_yaml."""

    def test_writes_yaml_file(self):
        if importlib.util.find_spec("ruamel.yaml") is None:
            self.skipTest("ruamel.yaml not installed")

        with ScratchDir("."):
            out_dir = Path("test_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            run_id = "test_run_yaml"
            run_params = {"nphi": 8, "ntheta": 8, "downsample": 6, "dr": 0.01}
            paths = {
                "test_dir": str(TEST_DIR),
                "famus_filename": "zot80.focus",
                "surface_filename": "input.muse",
                "coil_path": "muse_tf_coils.focus",
            }
            material = {"B_max": 1.465, "mu_ea": 1.05, "mu_oa": 1.15}
            metrics = {"fB_last_history": 1e-4, "fB_recomputed": 1e-4}

            path = write_gpmo_run_metadata_yaml(
                out_dir,
                run_id,
                algorithm="GPMO",
                current_scale=1.0,
                metrics=metrics,
                material_name="N52",
                material=material,
                run_params=run_params,
                paths=paths,
            )

            self.assertTrue(path.exists())
            self.assertEqual(path.name, "run_test_run_yaml.yaml")
            content = path.read_text()
            self.assertIn("run_id: test_run_yaml", content)
            self.assertIn("algorithm: GPMO", content)
            self.assertIn("N52", content)


class TestBuildPolarizationVectorsFromFocusData(unittest.TestCase):
    """Tests for build_polarization_vectors_from_focus_data."""

    def test_returns_correct_shape(self):
        mag_data = FocusData(TEST_DIR / "zot80.focus", downsample=100)
        pol_vectors = build_polarization_vectors_from_focus_data(mag_data)
        self.assertEqual(pol_vectors.ndim, 3)
        self.assertEqual(pol_vectors.shape[0], mag_data.nMagnets)
        self.assertEqual(pol_vectors.shape[2], 3)

    def test_polarization_vectors_are_unit_like(self):
        mag_data = FocusData(TEST_DIR / "zot80.focus", downsample=100)
        pol_vectors = build_polarization_vectors_from_focus_data(mag_data)
        norms = np.linalg.norm(pol_vectors, axis=2)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10)


class TestComputeMMaximaAndCubeDimFromFocusFile(unittest.TestCase):
    """Tests for compute_m_maxima_and_cube_dim_from_focus_file."""

    def test_returns_tuple_of_correct_types(self):
        m_maxima, cube_dim = compute_m_maxima_and_cube_dim_from_focus_file(
            TEST_DIR / "zot80.focus", downsample=100, B_max=1.465
        )
        self.assertIsInstance(m_maxima, np.ndarray)
        self.assertIsInstance(cube_dim, float)
        self.assertEqual(m_maxima.ndim, 1)
        self.assertGreater(cube_dim, 0)
        self.assertGreater(len(m_maxima), 0)

    def test_m_maxima_scales_with_B_max(self):
        m1, _ = compute_m_maxima_and_cube_dim_from_focus_file(
            TEST_DIR / "zot80.focus", downsample=100, B_max=1.465
        )
        m2, _ = compute_m_maxima_and_cube_dim_from_focus_file(
            TEST_DIR / "zot80.focus", downsample=100, B_max=0.7325
        )
        np.testing.assert_allclose(m2, m1 * 0.5, rtol=1e-10)


class TestComputeFinalSquaredFlux(unittest.TestCase):
    """Tests for compute_final_squared_flux."""

    def test_returns_nonnegative_float(self):
        nphi = 4
        ntheta = 4
        s = SurfaceRZFourier.from_focus(
            TEST_DIR / "input.muse", range="half period", nphi=nphi, ntheta=ntheta
        )
        base_curves, curves, coils = initialize_coils_for_pm_optimization(
            "muse_famus", TEST_DIR, s
        )
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(
            bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2
        )
        pm_opt = PermanentMagnetGrid.geo_setup_from_famus(
            s, Bnormal, TEST_DIR / "zot80.focus", downsample=100
        )
        kwargs = initialize_default_kwargs("GPMO")
        kwargs["K"] = 5
        kwargs["nhistory"] = 2
        kwargs["verbose"] = False
        GPMO(pm_opt, "baseline", **kwargs)

        fB = compute_final_squared_flux(
            s_plot=s, bs=bs, pm_opt=pm_opt, nphi=nphi, ntheta=ntheta
        )
        self.assertIsInstance(fB, float)
        self.assertGreaterEqual(fB, 0)


class TestRunGpmoOptimizationOnMuseGrid(unittest.TestCase):
    """Integration test for run_gpmo_optimization_on_muse_grid."""

    def test_runs_gpmo_and_writes_outputs(self):
        nphi = 2
        ntheta = 2
        s = SurfaceRZFourier.from_focus(
            TEST_DIR / "input.muse", range="half period", nphi=nphi, ntheta=ntheta
        )
        base_curves, curves, coils = initialize_coils_for_pm_optimization(
            "muse_famus", TEST_DIR, s
        )
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(
            bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2
        )
        mag_data = FocusData(TEST_DIR / "zot80.focus", downsample=100)
        pol_vectors = build_polarization_vectors_from_focus_data(mag_data)
        m_maxima, cube_dim = compute_m_maxima_and_cube_dim_from_focus_file(
            TEST_DIR / "zot80.focus", downsample=100, B_max=1.465
        )
        material = {"B_max": 1.465, "mu_ea": 1.05, "mu_oa": 1.15}
        gpmo_run_params = {
            "downsample": 100,
            "dr": 0.01,
            "nIter_max": 5,
            "nHistory": 5,
            "history_every": 1,
            "nBacktracking": 0,
            "Nadjacent": 12,
            "max_nMagnets": 5,
            "thresh_angle": 3.0,
            "mm_refine_every": 0,
            "nphi": nphi,
            "ntheta": ntheta,
            "coil_path": TEST_DIR / "muse_tf_coils.focus",
            "test_dir": TEST_DIR,
            "surface_filename": TEST_DIR / "input.muse",
            "dx": 0.01,
            "dy": 0.01,
            "dz": 0.01,
        }

        with ScratchDir("."):
            out_dir = Path("gpmo_test_output")
            run_gpmo_optimization_on_muse_grid(
                "GPMO",
                subdir="GPMO",
                s=s,
                bs=bs,
                Bnormal=Bnormal,
                pol_vectors=pol_vectors,
                m_maxima=m_maxima,
                cube_dim=cube_dim,
                current_scale=1.0,
                out_dir=out_dir,
                famus_filename=TEST_DIR / "zot80.focus",
                material_name="N52",
                material=material,
                **gpmo_run_params,
            )
            run_subdir = out_dir / "GPMO"
            self.assertTrue(run_subdir.exists())
            csv_files = list(run_subdir.glob("runhistory_*.csv"))
            self.assertGreater(len(csv_files), 0)
            npz_files = list(run_subdir.glob("dipoles_final_*.npz"))
            self.assertGreater(len(npz_files), 0)
