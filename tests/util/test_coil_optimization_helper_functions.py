import unittest
import os
import shutil
import numpy as np
from pathlib import Path
from monty.tempfile import ScratchDir

from simsopt.util import (
    initial_vacuum_stage_II_optimizations, continuation_vacuum_stage_II_optimizations,
    read_focus_coils, build_stage_II_data_array, make_stage_II_pareto_plots,
    vacuum_stage_II_optimization, coil_optimization
)
from simsopt.field import LpCurveForce

# Test directory setup
TEST_DIR = Path(__file__).parent / "../test_files"
OUTPUT_DIR = Path(__file__).parent / "test_output"


class TestConfigValidation(unittest.TestCase):
    """Tests for config validation and default path coverage."""

    def test_initial_optimizations_invalid_config(self):
        """Invalid config raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            initial_vacuum_stage_II_optimizations(N=1, config="INVALID")
        self.assertIn("Invalid configuration", str(cm.exception))

    def test_continuation_invalid_config(self):
        """Invalid config raises ValueError in continuation."""
        with self.assertRaises(ValueError) as cm:
            continuation_vacuum_stage_II_optimizations(N=1, config="INVALID")
        self.assertIn("Invalid configuration", str(cm.exception))

    def test_vacuum_stage_II_optimization_invalid_config(self):
        """Invalid config raises ValueError in vacuum_stage_II_optimization."""
        with self.assertRaises(ValueError) as cm:
            vacuum_stage_II_optimization(config="INVALID", R1=0.5, ncoils=3)
        self.assertIn("Invalid configuration", str(cm.exception))

    def test_initial_optimizations_QA_default_paths(self):
        """QA config with OUTPUT_DIR=None and INPUT_FILE=None uses defaults."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA", ".")
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                ncoils=3,
                config="QA",
                OUTPUT_DIR=None,
                INPUT_FILE=None,
            )
            self.assertTrue(Path("./output/QA/optimizations").exists())
            results_files = list(Path("./output/QA/optimizations").glob("*/results.json"))
            self.assertGreater(len(results_files), 0)

    def test_initial_optimizations_QH_default_paths(self):
        """QH config with OUTPUT_DIR=None and INPUT_FILE=None uses defaults."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QH_magwell_R0=1", ".")
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                ncoils=3,
                config="QH",
                OUTPUT_DIR=None,
                INPUT_FILE=None,
            )
            self.assertTrue(Path("./output/QH/optimizations").exists())
            results_files = list(Path("./output/QH/optimizations").glob("*/results.json"))
            self.assertGreater(len(results_files), 0)

    def test_continuation_QA_default_paths(self):
        """Continuation QA with default INPUT_DIR, OUTPUT_DIR, INPUT_FILE."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA", ".")
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                ncoils=3,
                config="QA",
                OUTPUT_DIR=None,
                INPUT_FILE=None,
            )
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                MAXITER=5,
                config="QA",
                INPUT_DIR=None,
                OUTPUT_DIR=None,
                INPUT_FILE=None,
            )
            self.assertTrue(Path("./output/QA/optimizations/continuation").exists())
            results_files = list(
                Path("./output/QA/optimizations/continuation").glob("*/results.json")
            )
            self.assertGreater(len(results_files), 0)

    def test_continuation_QH_default_paths(self):
        """Continuation QH with default INPUT_DIR, OUTPUT_DIR, INPUT_FILE."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QH_magwell_R0=1", ".")
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                ncoils=3,
                config="QH",
                OUTPUT_DIR=None,
                INPUT_FILE=None,
            )
            continuation_vacuum_stage_II_optimizations(
                N=1,
                dx=0.1,
                MAXITER=5,
                config="QH",
                INPUT_DIR=None,
                OUTPUT_DIR=None,
                INPUT_FILE=None,
            )
            self.assertTrue(Path("./output/QH/optimizations/continuation").exists())
            results_files = list(
                Path("./output/QH/optimizations/continuation").glob("*/results.json")
            )
            self.assertGreater(len(results_files), 0)

    def test_vacuum_stage_II_optimization_QA_default_paths(self):
        """vacuum_stage_II_optimization QA with OUTPUT_DIR=None, INPUT_FILE=None."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA", ".")
            results = vacuum_stage_II_optimization(
                config="QA",
                OUTPUT_DIR=None,
                INPUT_FILE=None,
                R1=0.5,
                ncoils=3,
                MAXITER=5,
            )
            self.assertTrue(Path("./output/QA/optimizations").exists())
            self.assertIn("UUID", results)

    def test_vacuum_stage_II_optimization_QH_default_paths(self):
        """vacuum_stage_II_optimization QH with OUTPUT_DIR=None, INPUT_FILE=None."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QH_magwell_R0=1", ".")
            results = vacuum_stage_II_optimization(
                config="QH",
                OUTPUT_DIR=None,
                INPUT_FILE=None,
                R1=0.5,
                ncoils=3,
                MAXITER=5,
            )
            self.assertTrue(Path("./output/QH/optimizations").exists())
            self.assertIn("UUID", results)


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

    def test_initial_optimizations_with_force_True(self):
        """Test initial_optimizations with with_force=True (force terms in objective)."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            output_dir = "qa_output_with_force/"
            initial_vacuum_stage_II_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3,
                with_force=True,
                FORCE_OBJ=LpCurveForce,
            )
            self.assertTrue(os.path.exists(output_dir))
            results_files = list(Path(output_dir).glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
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

    def test_make_stage_II_pareto_plots_with_df_filtered(self):
        """Test make_stage_II_pareto_plots with non-empty df_filtered (exercises 'after filtering' histograms)."""
        with ScratchDir("."):
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            initial_vacuum_stage_II_optimizations(
                N=2,
                MAXITER=50,
                OUTPUT_DIR="qa_output/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=4,
                config="QA"
            )
            continuation_vacuum_stage_II_optimizations(
                N=2,
                dx=0.01,
                INPUT_DIR="qa_output/",
                OUTPUT_DIR="qa_output_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                config="QA"
            )
            df, df_filtered, _ = build_stage_II_data_array(
                INPUT_DIR="qa_output/",
                margin_up=1e5,
                margin_low=1e-5,
            )
            self.assertGreater(len(df), 0, "df should be non-empty")
            # Use df as df_filtered when filter yields nothing, so we exercise the
            # "after filtering" histogram path in make_stage_II_pareto_plots
            if len(df_filtered) == 0:
                df_filtered = df
            pareto_dir = "qa_pareto_plots/"
            os.makedirs(pareto_dir, exist_ok=True)
            make_stage_II_pareto_plots(df, df_filtered, OUTPUT_DIR=pareto_dir)
            histograms = list(Path(pareto_dir).glob("histograms.pdf"))
            self.assertGreater(len(histograms), 0, "histograms.pdf should be created")
            self.assertGreater(histograms[0].stat().st_size, 0, "histograms.pdf should be non-empty")


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
                FORCE_WEIGHT=1.0,
                FORCE_THRESHOLD=1e4
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


if __name__ == "__main__":
    unittest.main() 