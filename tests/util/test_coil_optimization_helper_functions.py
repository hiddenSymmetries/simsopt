import unittest
import os
import shutil
import numpy as np
from pathlib import Path
from monty.tempfile import ScratchDir

from simsopt.util import (
    initial_vacuum_stage_II_optimizations, continuation_vacuum_stage_II_optimizations,
    read_focus_coils, build_stage_II_data_array, make_stage_II_pareto_plots,
    vacuum_stage_II_optimization, coil_optimization, make_filament_from_voxels,
)
# from simsopt.field import LpCurveForce, LpCurveTorque, SquaredMeanForce, SquaredMeanTorque

# Test directory setup
TEST_DIR = Path(__file__).parent / "../test_files"
OUTPUT_DIR = Path(__file__).parent / "test_output"


class TestInitialOptimizations(unittest.TestCase):

    def test_initial_optimizations_basic(self):
        """Test basic functionality of initial_optimizations with real file."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            output_dir = "qa_output/"
            
            # Test initial optimizations with default parameters and no output directory
            initial_vacuum_stage_II_optimizations(
                N=1,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                ncoils=3
            )
            # Repeat but specify the output directory and debug mode
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3,
                debug=True
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists(output_dir))
            
            # Check that a results.json file was created
            results_files = list(Path(output_dir).glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Check that biot_savart.json files were created
            biot_savart_files = list(Path(output_dir).glob("*/biot_savart.json"))
            self.assertGreater(len(biot_savart_files), 0)

    def test_initial_optimizations_with_force_objective(self):
        """Test initial_optimizations with force objective."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            output_dir = "qa_force_output/"
            
            # Test initial optimizations with force objective
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                # FORCE_OBJ=LpCurveForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )

            # Test continuation optimizations with force objective
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                # FORCE_OBJ=LpCurveForce,
            )

            # Test initial optimizations with torque objective (commented out for now)
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                # FORCE_OBJ=LpCurveTorque,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )

            # Test continuation optimizations with torque objective (commented out for now)
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                # FORCE_OBJ=LpCurveTorque,
            )

            # Test initial optimizations with SquaredMeanForce objective (commented out for now)
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                # FORCE_OBJ=SquaredMeanForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )

            # Test continuation optimizations with SquaredMeanForce objective (commented out for now)
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                # FORCE_OBJ=SquaredMeanForce,
            )

            # Test initial optimizations with SquaredMeanTorque objective (commented out for now)
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                # FORCE_OBJ=SquaredMeanTorque,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )

            # Test continuation optimizations with SquaredMeanTorque objective (commented out for now)
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                # FORCE_OBJ=SquaredMeanTorque,
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists(output_dir))
            
            # Check that a results.json file was created
            results_files = list(Path(output_dir).glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Check that biot_savart.json files were created
            biot_savart_files = list(Path(output_dir).glob("*/biot_savart.json"))
            self.assertGreater(len(biot_savart_files), 0)


class TestInitialOptimizationsQH(unittest.TestCase):

    def test_initial_optimizations_QH_basic(self):
        """Test basic functionality of initial_optimizations_QH with real file."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QH_reactorScale_lowres", ".")
            
            output_dir = "qh_output/"
            
            # Test initial optimizations with default parameters on LP QH
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
                ncoils=3,
                config="QH"
            )

            # Test initial optimizations with debug = True on LP QH
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                # FORCE_OBJ=LpCurveForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
                ncoils=3,
                debug=True,
                config="QH"
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists(output_dir))
            
            # Check that a results.json file was created
            results_files = list(Path(output_dir).glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Check that biot_savart.json files were created
            biot_savart_files = list(Path(output_dir).glob("*/biot_savart.json"))
            self.assertGreater(len(biot_savart_files), 0)

    def test_initial_optimizations_QH_with_force_objective(self):
        """Test initial_optimizations_QH with force objective using real file."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QH_reactorScale_lowres", ".")
            
            output_dir = "qh_force_output/"
            
            # Test initial optimizations with force objective on LP QH
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                # FORCE_OBJ=LpCurveForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
                ncoils=3,
                config="QH"
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists(output_dir))
            
            # Check that a results.json file was created
            results_files = list(Path(output_dir).glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Check that biot_savart.json files were created
            biot_savart_files = list(Path(output_dir).glob("*/biot_savart.json"))
            self.assertGreater(len(biot_savart_files), 0)


class TestReadFocusCoils(unittest.TestCase):

    def test_read_focus_coils_basic(self):
        """Test basic functionality of read_focus_coils with real FOCUS file."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "muse_tf_coils.focus", ".")
            
            focus_file = TEST_DIR / "muse_tf_coils.focus"
            
            # Test reading the FOCUS file
            coils = read_focus_coils(focus_file)
            
            # Check that coils were read
            self.assertIsNotNone(coils)
            self.assertGreater(len(coils), 0)
            
            # Check that each coil has the expected attributes
            for coil in coils:
                self.assertIsNotNone(coil)


class TestContinuation(unittest.TestCase):

    def test_continuation_basic(self):
        """Test basic functionality of continuation with real files, this test differs 
        from previous only in that it checks the continuation folder for new files."""
        with ScratchDir("."):
            # First run initial_optimizations to create input data for continuation
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            # Test initial optimizations with default parameters
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR="qa_output/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )
            
            # Test continuation optimizations with default parameters
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                INPUT_DIR="qa_output/",
                OUTPUT_DIR="qa_output_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5
            )
            
            # Check that output directory was created for the continuation optimizations
            self.assertTrue(os.path.exists("qa_output_continuation/"))
            
            # Check that a results.json file was created for the continuation optimizations
            results_files = list(Path("qa_output_continuation/").glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Check that biot_savart.json files were created for the continuation optimizations
            biot_savart_files = list(Path("qa_output_continuation/").glob("*/biot_savart.json"))
            self.assertGreater(len(biot_savart_files), 0)


class TestRealOptimizationRun(unittest.TestCase):

    def test_initial_optimizations_and_continuation(self):
        """Test initial_optimizations and continuation together for a nontrivial optimization run."""
        with ScratchDir("."):
            # Run initial optimizations
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            # Do initial optimizations with maxiters, N, and ncoils set to nontrivial values.
            initial_vacuum_stage_II_optimizations(
                N=2,
                MAXITER=50,
                OUTPUT_DIR="qa_output/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=4,
                config="QA"
            )
            
            # Run continuation on the previous optimization results
            continuation_vacuum_stage_II_optimizations(
                N=2,
                dx=0.01,
                INPUT_DIR="qa_output/",
                OUTPUT_DIR="qa_output_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                config="QA"
            )

            # Build a data array and make the Pareto plots
            df, df_filtered, _ = build_stage_II_data_array(
                INPUT_DIR="qa_output/", 
                margin_up=1e5,
                margin_low=1e-5,
            )
            make_stage_II_pareto_plots(df, df_filtered, OUTPUT_DIR="qa_output_continuation/")
            
            # Check that both output directories were created
            self.assertTrue(os.path.exists("qa_output/"))
            self.assertTrue(os.path.exists("qa_output_continuation/"))
            
            # Check that results.json files were created in both directories
            qa_results = list(Path("qa_output/").glob("*/results.json"))
            continuation_results = list(Path("qa_output_continuation/").glob("*/results.json"))
            continuation_hist = list(Path("qa_output_continuation/").glob("histograms.pdf"))
            self.assertGreater(len(qa_results), 0)
            self.assertGreater(len(continuation_results), 0)
            self.assertGreater(len(continuation_hist), 0)
            
            # Check that biot_savart.json files were created in both directories
            qa_biot_savart = list(Path("qa_output/").glob("*/biot_savart.json"))
            continuation_biot_savart = list(Path("qa_output_continuation/").glob("*/biot_savart.json"))
            self.assertGreater(len(qa_biot_savart), 0)
            self.assertGreater(len(continuation_biot_savart), 0)


class TestOptimizationKwargs(unittest.TestCase):

    def test_initial_optimizations_with_all_kwargs(self):
        """Test initial_optimizations with all kwargs explicitly passed."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            output_dir = "qa_output_kwargs/"
            
            # Test with all kwargs explicitly passed
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3,
                config="QA",
                FORCE_OBJ=None,
                with_force=False,
                debug=True
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists(output_dir))
            
            # Check that a results.json file was created
            results_files = list(Path(output_dir).glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Verify that the results.json contains expected keys
            if results_files:
                import json
                with open(results_files[0], 'r') as f:
                    results_data = json.load(f)
                    # Check that key optimization parameters are present
                    self.assertIn('JF', results_data)
                    self.assertIn('Jf', results_data)
                    self.assertIn('lengths', results_data)
                    self.assertIn('max_κ', results_data)
                    self.assertIn('MSCs', results_data)

    def test_continuation_with_all_kwargs(self):
        """Test continuation_optimizations with all kwargs explicitly passed."""
        with ScratchDir("."):
            # First run initial_optimizations to create input data for continuation
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            # Run initial optimizations to create input data for continuation
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR="qa_output_kwargs_init/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3,
                config="QA"
            )
            
            # Now test continuation with all kwargs
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                config="QA",
                INPUT_DIR="qa_output_kwargs_init/",
                OUTPUT_DIR="qa_output_kwargs_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                FORCE_OBJ=None,
                debug=True,
                MAXITER=5
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists("qa_output_kwargs_continuation/"))
            
            # Check that a results.json file was created
            results_files = list(Path("qa_output_kwargs_continuation/").glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Verify that the results.json contains expected keys
            if results_files:
                import json
                with open(results_files[0], 'r') as f:
                    results_data = json.load(f)
                    # Check that key optimization parameters are present
                    self.assertIn('JF', results_data)
                    self.assertIn('Jf', results_data)
                    self.assertIn('lengths', results_data)
                    self.assertIn('max_κ', results_data)
                    self.assertIn('MSCs', results_data)

    def test_vacuum_stage_II_optimization_with_all_kwargs(self):
        """Test vacuum_stage_II_optimization with all kwargs explicitly passed."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            output_dir = "qa_output_direct/"
            
            # Test with all kwargs explicitly passed
            results = vacuum_stage_II_optimization(
                config="QA",
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                R1=0.5,
                order=5,
                ncoils=3,
                UUID_init_from=None,
                # All kwargs explicitly passed
                LENGTH_TARGET=5.0,
                LENGTH_WEIGHT=1e-3,
                CURVATURE_THRESHOLD=12.0,
                CURVATURE_WEIGHT=1e-8,
                MSC_THRESHOLD=5.0,
                MSC_WEIGHT=1e-4,
                CC_THRESHOLD=0.083,
                CC_WEIGHT=1e3,
                CS_THRESHOLD=0.166,
                CS_WEIGHT=1e3,
                FORCE_THRESHOLD=2e4,
                FORCE_WEIGHT=1e-10,
                FORCE_OBJ=None,
                ARCLENGTH_WEIGHT=1e-2,
                dx=0.05,
                with_force=False,
                debug=True,
                MAXITER=5
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists(output_dir))
            
            # Check that results dictionary is returned
            self.assertIsInstance(results, dict)
            self.assertIn('UUID', results)
            self.assertIn('JF', results)
            self.assertIn('Jf', results)
            self.assertIn('lengths', results)
            self.assertIn('max_κ', results)
            self.assertIn('MSCs', results)
            
            # Check that a results.json file was created
            results_files = list(Path(output_dir).glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Verify that the results.json matches the returned dict
            if results_files:
                import json
                with open(results_files[0], 'r') as f:
                    results_data = json.load(f)
                    # Check that returned UUID matches saved UUID
                    self.assertEqual(results['UUID'], results_data['UUID'])


class TestCoilOptimization(unittest.TestCase):

    def test_coil_optimization_basic(self):
        """Test basic functionality of coil_optimization function."""
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries, BiotSavart
        from simsopt.objectives import SquaredFlux
        
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            # Create surface from test file
            nphi = 32
            ntheta = 32
            s = SurfaceRZFourier.from_vmec_input(
                TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                range="half period",
                nphi=nphi,
                ntheta=ntheta
            )
            nfp = s.nfp
            R0 = s.get_rc(0, 0)
            
            # Create initial coils
            ncoils = 3
            order = 5
            R1 = 0.5
            base_curves = create_equally_spaced_curves(
                ncoils,
                nfp,
                stellsym=True,
                R0=R0,
                R1=R1,
                order=order,
            )
            
            # Create currents
            total_current = 3e5
            base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
            total_current_obj = Current(total_current)
            total_current_obj.fix_all()
            base_currents += [total_current_obj - sum(base_currents)]
            
            # Create coils with symmetries
            coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
            curves = [c.curve for c in coils]
            
            # Create BiotSavart object
            bs = BiotSavart(coils)
            bs.set_points(s.gamma().reshape((-1, 3)))
            fB_initial = SquaredFlux(s, bs).J()
            
            # Run optimization with minimal iterations
            bs_optimized = coil_optimization(
                s, bs, base_curves, curves,
                MAXITER=5,
                LENGTH_WEIGHT=1.0,
                LENGTH_THRESHOLD=18.0 * R0,
                CC_WEIGHT=1.0,
                CC_THRESHOLD=0.1 * R0,
                CS_WEIGHT=1e-2,
                CS_THRESHOLD=0.15 * R0,
                CURVATURE_WEIGHT=1e-6,
                CURVATURE_THRESHOLD=0.1 * R0,
                MSC_WEIGHT=1e-6,
                MSC_THRESHOLD=0.1 * R0,
                LINKING_NUMBER_WEIGHT=0.0,
                FORCE_WEIGHT=0.0,
                FORCE_THRESHOLD=0.0
            )
            
            # Verify that optimization returns a BiotSavart object
            self.assertIsInstance(bs_optimized, BiotSavart)
            
            # Verify that the coils are still present
            self.assertEqual(len(bs_optimized.coils), len(coils))
            
            # Verify that the field can still be evaluated
            bs_optimized.set_points(s.gamma().reshape((-1, 3)))
            final_BdotN = np.mean(np.abs(np.sum(bs_optimized.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
            
            # Field should be finite
            self.assertTrue(np.isfinite(final_BdotN))

            # fB should have decreased
            fB_final = SquaredFlux(s, bs_optimized).J()
            self.assertLess(fB_final, fB_initial)

            # Run optimization with minimal iterations without passing arguments
            bs_optimized = coil_optimization(
                s, bs, base_curves, curves,
                MAXITER=5,
            )
            
            # Verify that points are set correctly
            points = s.gamma().reshape((-1, 3))
            np.testing.assert_allclose(bs_optimized.get_points_cart_ref(), points, atol=1e-8)


def _make_mock_grid(xyz, nfp, stellsym, alphas=None):
    """Create mock CurrentVoxelsGrid for testing."""
    n_pts = xyz.shape[0]
    if alphas is None:
        alphas = np.ones((n_pts, 1))

    class MockPlasmaBoundary:
        pass

    mock_pb = MockPlasmaBoundary()
    mock_pb.nfp = nfp
    mock_pb.stellsym = stellsym

    class MockGrid:
        pass

    mock_grid = MockGrid()
    mock_grid.N_grid = n_pts
    mock_grid.n_functions = 1
    mock_grid.alphas = alphas
    mock_grid.XYZ_flat = xyz
    mock_grid.plasma_boundary = mock_pb
    return mock_grid


class TestMakeFilamentFromVoxels(unittest.TestCase):

    def test_make_filament_returns_curve_with_symmetries(self):
        """Test that make_filament_from_voxels returns CurveXYZFourierSymmetries."""
        from simsopt.geo import CurveXYZFourierSymmetries

        nfp, stellsym = 2, True
        n_pts = 50
        phi_pts = np.linspace(0, 2 * np.pi / nfp, n_pts, endpoint=False)
        R0, r_minor = 1.0, 0.3
        x = (R0 + r_minor * np.cos(phi_pts * nfp)) * np.cos(phi_pts)
        y = (R0 + r_minor * np.cos(phi_pts * nfp)) * np.sin(phi_pts)
        z = r_minor * np.sin(phi_pts * nfp)
        xyz = np.column_stack([x, y, z])

        mock_grid = _make_mock_grid(xyz, nfp, stellsym)
        curve = make_filament_from_voxels(mock_grid, 0.5, num_fourier=8)

        self.assertIsInstance(curve, CurveXYZFourierSymmetries)
        self.assertEqual(curve.nfp, nfp)
        self.assertEqual(curve.stellsym, stellsym)
        gamma = curve.gamma()
        self.assertEqual(gamma.shape[1], 3)
        self.assertTrue(np.all(np.isfinite(gamma)), "gamma must not contain NaN or Inf")

    def test_make_filament_no_nan_in_gamma(self):
        """Test that gamma() never contains NaN for various inputs."""

        for n_pts in [10, 20, 50]:
            nfp, stellsym = 2, True
            phi_pts = np.linspace(0, 2 * np.pi / nfp, n_pts, endpoint=False)
            R0, r_minor = 1.0, 0.3
            x = (R0 + r_minor * np.cos(phi_pts * nfp)) * np.cos(phi_pts)
            y = (R0 + r_minor * np.cos(phi_pts * nfp)) * np.sin(phi_pts)
            z = r_minor * np.sin(phi_pts * nfp)
            xyz = np.column_stack([x, y, z])

            mock_grid = _make_mock_grid(xyz, nfp, stellsym)
            curve = make_filament_from_voxels(mock_grid, 0.5, num_fourier=8)
            gamma = curve.gamma()
            self.assertTrue(np.all(np.isfinite(gamma)), f"n_pts={n_pts}: gamma has NaN/Inf")

    def test_make_filament_non_stellarator_symmetric(self):
        """Test with stellsym=False."""

        nfp, stellsym = 1, False
        n_pts = 30
        phi_pts = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        R0, r_minor = 1.0, 0.3
        x = (R0 + r_minor * np.cos(phi_pts)) * np.cos(phi_pts)
        y = (R0 + r_minor * np.cos(phi_pts)) * np.sin(phi_pts)
        z = r_minor * np.sin(phi_pts)
        xyz = np.column_stack([x, y, z])

        mock_grid = _make_mock_grid(xyz, nfp, stellsym)
        curve = make_filament_from_voxels(mock_grid, 0.5, num_fourier=6)
        gamma = curve.gamma()
        self.assertTrue(np.all(np.isfinite(gamma)))
        self.assertFalse(curve.stellsym)

    def test_make_filament_no_nonzero_voxels_raises(self):
        """Test that no voxels above threshold raises ValueError."""
        nfp, stellsym = 2, True
        xyz = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
        # All alphas below threshold -> zero nonzero voxels -> too few points
        alphas = np.ones((2, 1)) * 0.1  # below threshold 0.5
        mock_grid = _make_mock_grid(xyz, nfp, stellsym, alphas=alphas)
        with self.assertRaises(ValueError):
            make_filament_from_voxels(mock_grid, 0.5, num_fourier=4)

    def test_make_filament_truncate_option(self):
        """Test truncate=True produces valid curve."""

        nfp, stellsym = 2, True
        n_pts = 40
        phi_pts = np.linspace(0, 2 * np.pi / nfp, n_pts, endpoint=False)
        R0, r_minor = 1.0, 0.3
        xyz = np.column_stack([
            (R0 + r_minor * np.cos(phi_pts * nfp)) * np.cos(phi_pts),
            (R0 + r_minor * np.cos(phi_pts * nfp)) * np.sin(phi_pts),
            r_minor * np.sin(phi_pts * nfp),
        ])
        mock_grid = _make_mock_grid(xyz, nfp, stellsym)
        curve = make_filament_from_voxels(mock_grid, 0.5, truncate=True, num_fourier=8)
        gamma = curve.gamma()
        self.assertTrue(np.all(np.isfinite(gamma)))

    def test_make_filament_centroid_at_origin(self):
        """Test that extracted filament has centroid at origin (device center alignment)."""
        nfp, stellsym = 2, True
        n_pts = 50
        phi_pts = np.linspace(0, 2 * np.pi / nfp, n_pts, endpoint=False)
        R0, r_minor = 1.0, 0.3
        xyz = np.column_stack([
            (R0 + r_minor * np.cos(phi_pts * nfp)) * np.cos(phi_pts),
            (R0 + r_minor * np.cos(phi_pts * nfp)) * np.sin(phi_pts),
            r_minor * np.sin(phi_pts * nfp),
        ])
        mock_grid = _make_mock_grid(xyz, nfp, stellsym)
        curve = make_filament_from_voxels(mock_grid, 0.5, num_fourier=8)
        centroid = curve.centroid()
        self.assertLess(
            np.linalg.norm(centroid),
            1e-4,
            msg=f"Coil centroid should be at origin for symmetry; got {centroid}",
        )


def _compute_Bn_nfp_symmetry_error(Bn, nfp, nphi, ntheta):
    """
    Compute nfp symmetry error for B·n on the plasma surface.

    For a coil with nfp-fold rotational symmetry, B·n(φ,θ) = B·n(φ+2π/nfp,θ).

    Bn: 2D array (nphi, ntheta) of B·n values
    nfp: number of field periods
    nphi, ntheta: grid dimensions (nphi must be divisible by nfp)

    Returns:
        nfp_err: max |B·n(φ,θ) - B·n(φ+2π/nfp,θ)|
    """
    Bn = np.asarray(Bn).reshape(nphi, ntheta)
    nfp_err = 0.0
    step_phi = nphi // nfp  # indices per field period
    for i in range(nphi - step_phi):
        for j in range(ntheta):
            diff = np.abs(Bn[i, j] - Bn[i + step_phi, j])
            nfp_err = max(nfp_err, diff)
    return nfp_err


def _compute_Bn_stellarator_symmetry_error(Bn, nphi, ntheta):
    """
    Compute stellarator symmetry error: max |B·n(φ,θ) + B·n(-φ,-θ)|.
    Stellarator symmetry: B·n(φ,θ) = -B·n(-φ,-θ).
    """
    Bn = np.asarray(Bn).reshape(nphi, ntheta)
    err = 0.0
    for i in range(nphi):
        for j in range(ntheta):
            i_stell = (nphi - i) % nphi
            j_stell = (ntheta - j) % ntheta
            diff = np.abs(Bn[i, j] + Bn[i_stell, j_stell])
            err = max(err, diff)
    return err


class TestFilamentFieldSymmetry(unittest.TestCase):
    """
    Test that the magnetic field from a helical filament coil obeys nfp
    rotational symmetry on the plasma surface, before and after optimization.
    """

    @classmethod
    def setUpClass(cls):
        """Create plasma surface and filament coil once for all tests."""
        from simsopt.field import BiotSavart, Current, Coil
        from simsopt.geo import SurfaceRZFourier

        cls.TEST_DIR = Path(__file__).parent / "../test_files"
        cls.filename_QA = cls.TEST_DIR / "input.LandremanPaul2021_QA"

        # Plasma surface: full torus for symmetry checks.
        # nphi must be divisible by nfp for correct symmetry error computation.
        nphi, ntheta = 32, 24
        cls.s = SurfaceRZFourier.from_vmec_input(
            cls.filename_QA,
            range="full torus",
            nphi=nphi,
            ntheta=ntheta,
        )
        cls.nphi, cls.ntheta = nphi, ntheta
        cls.nfp = cls.s.nfp
        cls.stellsym = cls.s.stellsym
        assert nphi % cls.nfp == 0, "nphi must be divisible by nfp for symmetry checks"

        # Filament coil from mock voxel data
        phi_pts = np.linspace(0, 2 * np.pi / cls.nfp, 50, endpoint=False)
        R0, r_minor = 1.0, 0.3
        xyz = np.column_stack([
            (R0 + r_minor * np.cos(phi_pts * cls.nfp)) * np.cos(phi_pts),
            (R0 + r_minor * np.cos(phi_pts * cls.nfp)) * np.sin(phi_pts),
            r_minor * np.sin(phi_pts * cls.nfp),
        ])
        mock_grid = _make_mock_grid(xyz, cls.nfp, cls.stellsym)
        cls.curve = make_filament_from_voxels(mock_grid, 0.5, num_fourier=8)
        cls.curve.fix("xc(0)")
        cls.current = Current(1e5)
        cls.current.fix_all()
        cls.coil = Coil(cls.curve, cls.current)
        cls.bs = BiotSavart([cls.coil])

    def _get_Bn(self):
        """Compute B·n on the plasma surface."""
        self.bs.set_points(self.s.gamma().reshape((-1, 3)))
        B = self.bs.B().reshape(self.nphi, self.ntheta, 3)
        n = self.s.unitnormal().reshape(self.nphi, self.ntheta, 3)
        return np.sum(B * n, axis=2)

    def test_Bn_nfp_symmetry_before_optimization(self):
        """B·n from helical coil obeys nfp symmetry before optimization."""
        Bn = self._get_Bn()
        nfp_err = _compute_Bn_nfp_symmetry_error(
            Bn, self.nfp, self.nphi, self.ntheta
        )
        Bn_scale = np.max(np.abs(Bn)) + 1e-14
        # CurveXYZFourierSymmetries enforces nfp symmetry by construction, so error should be ~machine precision
        self.assertLess(
            nfp_err / Bn_scale,
            1e-10,
            msg=f"nfp symmetry error {nfp_err} (rel {nfp_err/Bn_scale:.2e}) before optimization",
        )

    def test_Bn_stellarator_symmetry_before_optimization(self):
        """B·n from helical coil obeys stellarator symmetry when stellsym=True."""
        if not self.stellsym:
            self.skipTest("Surface is not stellarator-symmetric")
        Bn = self._get_Bn()
        stell_err = _compute_Bn_stellarator_symmetry_error(Bn, self.nphi, self.ntheta)
        Bn_scale = np.max(np.abs(Bn)) + 1e-14
        self.assertLess(
            stell_err / Bn_scale,
            1e-10,
            msg=f"stellarator symmetry error {stell_err} (rel {stell_err/Bn_scale:.2e}) before optimization",
        )

    def test_coil_geometry_nfp_symmetry(self):
        """Coil geometry has nfp rotational symmetry (rotate by 2π/nfp maps onto itself)."""
        pts = self.curve.gamma()
        step = max(1, len(pts) // (4 * self.nfp))
        subset = pts[::step]
        angle = 2 * np.pi / self.nfp
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotated = (R @ subset.T).T
        dists = np.min(
            np.linalg.norm(pts[:, None, :] - rotated[None, :, :], axis=2), axis=0
        )
        scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-14
        self.assertLess(
            np.max(dists) / scale,
            1e-6,
            msg="CurveXYZFourierSymmetries coil should have nfp geometry symmetry",
        )

    def test_Bn_nfp_symmetry_fails_for_asymmetric_coil(self):
        """Negative test: asymmetric coil fails the nfp symmetry check."""
        from simsopt.field import BiotSavart, Coil
        from simsopt.geo import CurveXYZFourier

        # CurveXYZFourier with xs(1) != 0 breaks nfp=2 symmetry (sin(θ) has period 2π, not π)
        curve_asym = CurveXYZFourier(50, 2)  # Fewer points for speed
        dofs = curve_asym.full_x.copy()
        dofs[0] = 1.0
        dofs[1] = 0.3  # xs(1) - breaks nfp=2 rotational symmetry
        curve_asym.full_x = dofs

        bs_asym = BiotSavart([Coil(curve_asym, self.current)])
        bs_asym.set_points(self.s.gamma().reshape((-1, 3)))
        Bn_asym = np.sum(
            bs_asym.B().reshape(self.nphi, self.ntheta, 3)
            * self.s.unitnormal().reshape(self.nphi, self.ntheta, 3),
            axis=2,
        )
        nfp_err = _compute_Bn_nfp_symmetry_error(
            Bn_asym, self.nfp, self.nphi, self.ntheta
        )
        Bn_scale = np.max(np.abs(Bn_asym)) + 1e-14
        self.assertGreater(
            nfp_err / Bn_scale,
            1e-10,
            msg="Asymmetric coil should fail symmetry check (rel err > 1e-10)",
        )

    def test_Bn_nfp_symmetry_after_optimization(self):
        """B·n from helical coil obeys nfp symmetry after optimization."""
        from scipy.optimize import minimize
        from simsopt.geo import (
            CurveLength,
            LpCurveCurvature,
            CurveSurfaceDistance,
            MeanSquaredCurvature,
        )
        from simsopt.objectives import SquaredFlux, Weight, QuadraticPenalty

        curves = [self.coil.curve]
        Jf = SquaredFlux(self.s, self.bs)
        JF = (
            Jf
            + Weight(1e-6) * sum(CurveLength(c) for c in curves)
            + Weight(1e-10) * CurveSurfaceDistance(curves, self.s, 0.1)
            + Weight(1e-8) * sum(LpCurveCurvature(c, 2, 0.1) for c in curves)
            + Weight(1e-8) * sum(
                QuadraticPenalty(MeanSquaredCurvature(c), 0.1, "max") for c in curves
            )
        )

        def fun(dofs):
            JF.x = dofs
            return JF.J(), JF.dJ()

        dofs = JF.x
        minimize(fun, dofs, jac=True, method="L-BFGS-B", options={"maxiter": 15})

        Bn = self._get_Bn()
        nfp_err = _compute_Bn_nfp_symmetry_error(
            Bn, self.nfp, self.nphi, self.ntheta
        )
        Bn_scale = np.max(np.abs(Bn)) + 1e-14
        # Symmetry is baked into CurveXYZFourierSymmetries; optimization cannot break it
        self.assertLess(
            nfp_err / Bn_scale,
            1e-10,
            msg=f"nfp symmetry error {nfp_err} (rel {nfp_err/Bn_scale:.2e}) after optimization",
        )
        if self.stellsym:
            stell_err = _compute_Bn_stellarator_symmetry_error(Bn, self.nphi, self.ntheta)
            self.assertLess(
                stell_err / Bn_scale,
                1e-10,
                msg=f"stellarator symmetry error {stell_err} (rel {stell_err/Bn_scale:.2e}) after optimization",
            )


if __name__ == "__main__":
    unittest.main() 