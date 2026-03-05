"""
Unit tests for permanent_magnet_helper_functions module.
"""
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.field import BiotSavart
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.solve import GPMO
from simsopt.util import (
    FocusData,
    GPMORunConfig,
    build_polarization_vectors_from_focus_data,
    check_magnet_volume_stellarator_symmetry,
    compute_angle_between_vectors_degrees,
    compute_final_squared_flux,
    compute_m_maxima_and_cube_dim_from_focus_file,
    compute_normal_field_component_from_dipoles,
    get_utc_timestamp_iso,
    initialize_coils_for_pm_optimization,
    initialize_default_kwargs,
    normalize_vectors_to_unit_length,
    print_bnormal_error_summary_statistics,
    reshape_to_vtk_field_format,
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

    def test_raises_if_yaml_dependency_missing(self):
        import simsopt.util.permanent_magnet_helper_functions as pmhf

        with ScratchDir("."):
            out_dir = Path("test_output")
            out_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(pmhf, "YAML", None):
                with self.assertRaises(ImportError):
                    pmhf.write_gpmo_run_metadata_yaml(
                        out_dir,
                        "test_run_yaml_missing_dep",
                        algorithm="GPMO",
                        current_scale=1.0,
                        metrics={},
                        material_name="N52",
                        material={},
                        run_params={},
                        paths={},
                    )


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

    def _make_test_fixtures(self, nphi=2, ntheta=2, downsample=100):
        """Build surface, coils, Bnormal, pol_vectors, m_maxima, cube_dim."""
        s = SurfaceRZFourier.from_focus(
            TEST_DIR / "input.muse", range="half period", nphi=nphi, ntheta=ntheta
        )
        _, _, coils = initialize_coils_for_pm_optimization("muse_famus", TEST_DIR, s)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(
            bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2
        )
        mag_data = FocusData(TEST_DIR / "zot80.focus", downsample=downsample)
        pol_vectors = build_polarization_vectors_from_focus_data(mag_data)
        m_maxima, cube_dim = compute_m_maxima_and_cube_dim_from_focus_file(
            TEST_DIR / "zot80.focus", downsample=downsample, B_max=1.465
        )
        return s, bs, Bnormal, pol_vectors, m_maxima, cube_dim

    def test_runs_gpmo_and_writes_outputs(self):
        nphi = ntheta = 2
        s, bs, Bnormal, pol_vectors, m_maxima, cube_dim = self._make_test_fixtures(nphi, ntheta)
        config = GPMORunConfig(
            downsample=100, dr=0.01, dx=0.01, dy=0.01, dz=0.01,
            nphi=nphi, ntheta=ntheta,
            nIter_max=5, nHistory=5, history_every=1,
            nBacktracking=0, Nadjacent=12, max_nMagnets=5,
            thresh_angle=3.0, mm_refine_every=0,
            coil_path=TEST_DIR / "muse_tf_coils.focus",
            test_dir=TEST_DIR, surface_filename=TEST_DIR / "input.muse",
            material_name="N52",
            material={"B_max": 1.465, "mu_ea": 1.05, "mu_oa": 1.15},
        )
        with ScratchDir("."):
            out_dir = Path("gpmo_test_output")
            run_gpmo_optimization_on_muse_grid(
                "GPMO", subdir="GPMO", s=s, bs=bs, Bnormal=Bnormal,
                pol_vectors=pol_vectors, m_maxima=m_maxima, cube_dim=cube_dim,
                current_scale=1.0, out_dir=out_dir,
                famus_filename=TEST_DIR / "zot80.focus", config=config,
            )
            run_subdir = out_dir / "GPMO"
            self.assertTrue(run_subdir.exists())
            self.assertGreater(len(list(run_subdir.glob("runhistory_*.csv"))), 0)
            self.assertGreater(len(list(run_subdir.glob("dipoles_final_*.npz"))), 0)

    def test_runs_gpmomr_and_writes_outputs(self):
        """Integration test for run_gpmo_optimization_on_muse_grid with GPMOmr algorithm."""
        nphi = ntheta = 2
        s, bs, Bnormal, pol_vectors, m_maxima, cube_dim = self._make_test_fixtures(nphi, ntheta)
        config = GPMORunConfig(
            downsample=100, dr=0.01, dx=0.01, dy=0.01, dz=0.01,
            nphi=nphi, ntheta=ntheta,
            nIter_max=5, nHistory=5, history_every=1,
            nBacktracking=0, Nadjacent=12, max_nMagnets=5,
            thresh_angle=3.0, mm_refine_every=2,
            coil_path=TEST_DIR / "muse_tf_coils.focus",
            test_dir=TEST_DIR, surface_filename=TEST_DIR / "input.muse",
            material_name="N52",
            material={"B_max": 1.465, "mu_ea": 1.05, "mu_oa": 1.15},
        )
        with ScratchDir("."):
            out_dir = Path("gpmo_test_output")
            run_gpmo_optimization_on_muse_grid(
                "GPMOmr", subdir="GPMOmr", s=s, bs=bs, Bnormal=Bnormal,
                pol_vectors=pol_vectors, m_maxima=m_maxima, cube_dim=cube_dim,
                current_scale=1.0, out_dir=out_dir,
                famus_filename=TEST_DIR / "zot80.focus", config=config,
            )
            run_subdir = out_dir / "GPMOmr"
            self.assertTrue(run_subdir.exists())
            self.assertGreater(len(list(run_subdir.glob("runhistory_*.csv"))), 0)
            self.assertGreater(len(list(run_subdir.glob("dipoles_final_*.npz"))), 0)
            yaml_files = list(run_subdir.glob("run_*.yaml"))
            self.assertGreater(len(yaml_files), 0)
            for yf in yaml_files:
                self.assertIn("GPMOmr", yf.read_text())


class TestComputeNormalFieldComponentFromDipoles(unittest.TestCase):
    """Tests for compute_normal_field_component_from_dipoles."""

    def test_returns_correct_shape(self):
        """Output has shape (M,) matching number of evaluation points."""
        dipole_centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        dipole_moments = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        eval_pts = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Bn = compute_normal_field_component_from_dipoles(
            dipole_centers, dipole_moments, eval_pts, normals
        )
        self.assertEqual(Bn.shape, (3,))
        self.assertIsInstance(Bn, np.ndarray)

    def test_zero_dipoles_gives_zero_field(self):
        """With zero dipole moments, B·n should be zero."""
        dipole_centers = np.array([[0.0, 0.0, 0.0]])
        dipole_moments = np.array([[0.0, 0.0, 0.0]])
        eval_pts = np.array([[1.0, 0.0, 0.0]])
        normals = np.array([[1.0, 0.0, 0.0]])
        Bn = compute_normal_field_component_from_dipoles(
            dipole_centers, dipole_moments, eval_pts, normals
        )
        np.testing.assert_allclose(Bn, 0.0, atol=1e-14)

    def test_matches_dipole_field_B_dot_n(self):
        """B·n matches manual DipoleField.B() dotted with normals."""
        from simsopt.field import DipoleField

        dipole_centers = np.array([[0.0, 0.0, 0.0]])
        dipole_moments = np.array([[1e-3, 0.0, 0.0]])  # Small moment
        eval_pts = np.array([[0.5, 0.0, 0.0]])
        normals = np.array([[1.0, 0.0, 0.0]])
        Bn = compute_normal_field_component_from_dipoles(
            dipole_centers, dipole_moments, eval_pts, normals
        )
        dip = DipoleField(
            dipole_centers, dipole_moments,
            nfp=1, stellsym=False, coordinate_flag="cartesian"
        )
        dip.set_points(eval_pts)
        B = dip.B()
        Bn_manual = np.sum(B * normals, axis=1)
        np.testing.assert_allclose(Bn, Bn_manual, rtol=1e-10)


class TestCheckMagnetVolumeStellaratorSymmetry(unittest.TestCase):
    """Tests for check_magnet_volume_stellarator_symmetry."""

    def test_runs_without_error_on_symmetric_grid(self):
        """Should run without raising on a symmetric magnet grid."""
        nfp = 2
        n_per_half = 10
        phi = np.linspace(0, np.pi / nfp, n_per_half, endpoint=False)
        r = 1.0 + 0.1 * np.cos(2 * phi)
        z = 0.1 * np.sin(2 * phi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        centers = np.column_stack([x, y, z])
        volumes = np.ones(n_per_half) * 1e-6
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            check_magnet_volume_stellarator_symmetry(
                centers, volumes, nfp=nfp
            )
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        self.assertIn("nfp=", output)
        self.assertIn("toroidal symmetry", output)
        self.assertIn("midplane", output)
        self.assertIn("inversion", output)

    def test_accepts_z_tolerance_and_relative_tolerance(self):
        """Should accept optional z_tolerance and relative_tolerance args."""
        centers = np.array([[1.0, 0.0, 0.1], [1.0, 0.0, -0.1]])
        volumes = np.array([1e-6, 1e-6])
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            check_magnet_volume_stellarator_symmetry(
                centers, volumes, nfp=2,
                z_tolerance=1e-4, relative_tolerance=1e-2
            )
        finally:
            sys.stdout = old_stdout


class TestReshapeToVtkFieldFormat(unittest.TestCase):
    """Tests for reshape_to_vtk_field_format."""

    def test_returns_3d_with_last_dim_one(self):
        """Output has shape (nphi, ntheta, 1)."""
        field_2d = np.random.rand(8, 16)
        out = reshape_to_vtk_field_format(field_2d)
        self.assertEqual(out.shape, (8, 16, 1))
        np.testing.assert_allclose(out[:, :, 0], field_2d)

    def test_is_contiguous_and_float64(self):
        """Output is contiguous and float64."""
        field_2d = np.random.rand(4, 4).astype(np.float32)
        out = reshape_to_vtk_field_format(field_2d)
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        self.assertEqual(out.dtype, np.float64)


class TestPrintBnormalErrorSummaryStatistics(unittest.TestCase):
    """Tests for print_bnormal_error_summary_statistics."""

    def test_runs_without_error(self):
        """Should print without raising."""
        bnormal_error = np.array([1e-5, -2e-5, 3e-5])
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_bnormal_error_summary_statistics("Test case", bnormal_error)
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        self.assertIn("Test case", output)
        self.assertIn("mean", output)
        self.assertIn("max", output)
        self.assertIn("rms", output)


class TestNormalizeVectorsToUnitLength(unittest.TestCase):
    """Tests for normalize_vectors_to_unit_length."""

    def test_returns_unit_vectors(self):
        """Output vectors have unit norm."""
        v = np.array([[3.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 3.0]])
        out = normalize_vectors_to_unit_length(v)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-10)

    def test_preserves_direction(self):
        """Output direction matches input (for nonzero vectors)."""
        v = np.array([[1.0, 2.0, 3.0]])
        out = normalize_vectors_to_unit_length(v)
        expected = v / np.linalg.norm(v)
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_handles_zero_vector_with_eps(self):
        """Zero vector does not cause division by zero."""
        v = np.array([[0.0, 0.0, 0.0]])
        out = normalize_vectors_to_unit_length(v, eps=1e-30)
        self.assertTrue(np.all(np.isfinite(out)))


class TestComputeAngleBetweenVectorsDegrees(unittest.TestCase):
    """Tests for compute_angle_between_vectors_degrees."""

    def test_parallel_vectors_give_zero_degrees(self):
        """Parallel vectors give 0°."""
        a = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        angles = compute_angle_between_vectors_degrees(a, b)
        np.testing.assert_allclose(angles, [0.0, 0.0], atol=1e-10)

    def test_antiparallel_vectors_give_180_degrees(self):
        """Antiparallel vectors give 180°."""
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[-1.0, 0.0, 0.0]])
        angles = compute_angle_between_vectors_degrees(a, b)
        np.testing.assert_allclose(angles, [180.0], atol=1e-10)

    def test_perpendicular_vectors_give_90_degrees(self):
        """Perpendicular vectors give 90°."""
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        angles = compute_angle_between_vectors_degrees(a, b)
        np.testing.assert_allclose(angles, [90.0], atol=1e-10)

    def test_returns_correct_shape(self):
        """Output has shape (N,) for N vector pairs."""
        a = np.random.rand(5, 3)
        b = np.random.rand(5, 3)
        angles = compute_angle_between_vectors_degrees(a, b)
        self.assertEqual(angles.shape, (5,))


class TestInitializeDefaultKwargs(unittest.TestCase):
    """Tests for initialize_default_kwargs."""

    def test_rs_branch_returns_rsparams_dict(self):
        with self.assertWarns(DeprecationWarning):
            kwargs = initialize_default_kwargs("RS")
        self.assertIsInstance(kwargs, dict)
        self.assertIn("nu", kwargs)
        self.assertIn("max_iter", kwargs)
        self.assertIn("epsilon_RS", kwargs)

    def test_unknown_algorithm_falls_back(self):
        with self.assertWarns(DeprecationWarning):
            kwargs = initialize_default_kwargs("totally_unknown_algorithm")
        self.assertEqual(kwargs, {"verbose": True})
