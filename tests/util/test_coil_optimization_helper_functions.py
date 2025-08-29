import unittest
import os
import shutil
from pathlib import Path
from monty.tempfile import ScratchDir

from simsopt.util.coil_optimization_helper_functions import (
    initial_optimizations, initial_optimizations_QH, continuation,
    read_focus_coils, get_dfs, success_plt
)
from simsopt.field import LpCurveForce, LpCurveTorque, SquaredMeanForce, SquaredMeanTorque

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
            
            initial_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )
            initial_optimizations(
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
            
            initial_optimizations(
                N=1,
                MAXITER=5,
                FORCE_OBJ=LpCurveForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )
            continuation(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                FORCE_OBJ=LpCurveForce,
            )

            initial_optimizations(
                N=1,
                MAXITER=5,
                FORCE_OBJ=LpCurveTorque,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )
            continuation(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                FORCE_OBJ=LpCurveTorque,
            )

            initial_optimizations(
                N=1,
                MAXITER=5,
                FORCE_OBJ=SquaredMeanForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )
            continuation(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                FORCE_OBJ=SquaredMeanForce,
            )

            initial_optimizations(
                N=1,
                MAXITER=5,
                FORCE_OBJ=SquaredMeanTorque,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )
            continuation(
                N=1,
                dx=0.1,
                INPUT_DIR=output_dir,
                OUTPUT_DIR=output_dir + "_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5,
                FORCE_OBJ=SquaredMeanTorque,
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
            
            initial_optimizations_QH(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
                ncoils=3
            )
            initial_optimizations_QH(
                N=1,
                MAXITER=5,
                FORCE_OBJ=LpCurveForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
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

    def test_initial_optimizations_QH_with_force_objective(self):
        """Test initial_optimizations_QH with force objective using real file."""
        with ScratchDir("."):
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QH_reactorScale_lowres", ".")
            
            output_dir = "qh_force_output/"
            
            initial_optimizations_QH(
                N=1,
                MAXITER=5,
                FORCE_OBJ=LpCurveForce,
                OUTPUT_DIR=output_dir,
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
                ncoils=3
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
        """Test basic functionality of continuation with real files."""
        with ScratchDir("."):
            # First run initial_optimizations to create input data for continuation
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            initial_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR="qa_output/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=3
            )
            
            continuation(
                N=1,
                dx=0.1,
                INPUT_DIR="qa_output/",
                OUTPUT_DIR="qa_output_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists("qa_output_continuation/"))
            
            # Check that a results.json file was created
            results_files = list(Path("qa_output_continuation/").glob("*/results.json"))
            self.assertGreater(len(results_files), 0)
            
            # Check that biot_savart.json files were created
            biot_savart_files = list(Path("qa_output_continuation/").glob("*/biot_savart.json"))
            self.assertGreater(len(biot_savart_files), 0)


class TestRealOptimizationRun(unittest.TestCase):

    def test_initial_optimizations_and_continuation(self):
        """Test initial_optimizations and continuation together."""
        with ScratchDir("."):
            # Run initial optimizations
            # Copy required files into the temp dir
            shutil.copy(TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres", ".")
            
            initial_optimizations(
                N=2,
                MAXITER=50,
                OUTPUT_DIR="qa_output/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                ncoils=4
            )
            
            # Run continuation
            continuation(
                N=2,
                dx=0.01,
                INPUT_DIR="qa_output/",
                OUTPUT_DIR="qa_output_continuation/",
                INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
                MAXITER=5
            )

            df, df_filtered, _ = get_dfs(INPUT_DIR="qa_output/", 
                                         margin_up=1e5,
                                         margin_low=1e-5)
            print('Done filtering')
            success_plt(df, df_filtered, OUTPUT_DIR="qa_output_continuation/")
            
            # Check that both output directories were created
            self.assertTrue(os.path.exists("qa_output/"))
            self.assertTrue(os.path.exists("qa_output_continuation/"))
            
            # Check that results.json files were created in both directories
            qa_results = list(Path("qa_output/").glob("*/results.json"))
            continuation_results = list(Path("qa_output_continuation/").glob("*/results.json"))
            continuation_hist = list(Path("qa_output_continuation/").glob("hist.pdf"))
            self.assertGreater(len(qa_results), 0)
            self.assertGreater(len(continuation_results), 0)
            self.assertGreater(len(continuation_hist), 0)
            
            # Check that biot_savart.json files were created in both directories
            qa_biot_savart = list(Path("qa_output/").glob("*/biot_savart.json"))
            continuation_biot_savart = list(Path("qa_output_continuation/").glob("*/biot_savart.json"))
            self.assertGreater(len(qa_biot_savart), 0)
            self.assertGreater(len(continuation_biot_savart), 0)


if __name__ == "__main__":
    unittest.main() 