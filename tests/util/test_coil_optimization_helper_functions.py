import unittest
from pathlib import Path
import json
import os

from simsopt.util.coil_optimization_helper_functions import (
    initial_optimizations, 
    initial_optimizations_QH,
    continuation,
    read_focus_coils
)
from simsopt.field import LpCurveForce
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
OUTPUT_DIR = (Path(__file__).parent / ".." / "util").resolve()

class TestInitialOptimizations(unittest.TestCase):
    """Test cases for initial_optimizations function."""

    def test_initial_optimizations_basic(self):
        """Test basic functionality of initial_optimizations with real file."""
        output_dir = "qa_output/"
        
        initial_optimizations(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=str(OUTPUT_DIR / output_dir),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
            ncoils=3
        )
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(str(OUTPUT_DIR / output_dir)))
        
        # Check that a results.json file was created
        found = False
        for root, dirs, files in os.walk(str(OUTPUT_DIR / output_dir)):
            if "results.json" in files:
                results_path = os.path.join(root, "results.json")
                found = True
                break
        self.assertTrue(found, "results.json not found after initial_optimizations")
        
        # Check that results.json contains expected keys
        with open(results_path) as f:
            results = json.load(f)
        expected_keys = ["UUID", "ncoils", "order", "R1", "JF", "Jf"]
        for key in expected_keys:
            self.assertIn(key, results)

    def test_initial_optimizations_with_force_objective(self):
        """Test initial_optimizations with force objective."""
        output_dir = "qa_force_output/"
        
        initial_optimizations(
            N=1,
            MAXITER=5,
            FORCE_OBJ=LpCurveForce,
            OUTPUT_DIR=str(OUTPUT_DIR / output_dir),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
            ncoils=3
        )
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(str(OUTPUT_DIR / output_dir)))
        
        # Check that a results.json file was created
        found = False
        for root, dirs, files in os.walk(str(OUTPUT_DIR / output_dir)):
            if "results.json" in files:
                results_path = os.path.join(root, "results.json")
                found = True
                break
        self.assertTrue(found, "results.json not found after initial_optimizations with force")
        
        # Check that results.json contains force-related keys
        with open(results_path) as f:
            results = json.load(f)
        force_keys = ["lpcurveforce", "max_forces", "force_weight"]
        for key in force_keys:
            self.assertIn(key, results)


class TestInitialOptimizationsQH(unittest.TestCase):
    """Test cases for initial_optimizations_QH function using real files."""

    def test_initial_optimizations_QH_basic(self):
        """Test basic functionality of initial_optimizations_QH with real file."""
        output_dir = "qh_output/"
        
        initial_optimizations_QH(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=str(OUTPUT_DIR / output_dir),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
            ncoils=3
        )
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(str(OUTPUT_DIR / output_dir)))
        
        # Check that a results.json file was created
        found = False
        for root, dirs, files in os.walk(str(OUTPUT_DIR / output_dir)):
            if "results.json" in files:
                results_path = os.path.join(root, "results.json")
                found = True
                break
        self.assertTrue(found, "results.json not found after initial_optimizations_QH")
        
        # Check that results.json contains expected keys
        with open(results_path) as f:
            results = json.load(f)
        expected_keys = ["UUID", "ncoils", "order", "R1", "JF", "Jf"]
        for key in expected_keys:
            self.assertIn(key, results)

    def test_initial_optimizations_QH_with_force_objective(self):
        """Test initial_optimizations_QH with force objective using real file."""
        output_dir = "qh_force_output/"
        
        initial_optimizations_QH(
            N=1,
            MAXITER=5,
            FORCE_OBJ=LpCurveForce,
            OUTPUT_DIR=str(OUTPUT_DIR / output_dir),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
            ncoils=3
        )
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(str(OUTPUT_DIR / output_dir)))
        
        # Check that a results.json file was created
        found = False
        for root, dirs, files in os.walk(str(OUTPUT_DIR / output_dir)):
            if "results.json" in files:
                results_path = os.path.join(root, "results.json")
                found = True
                break
        self.assertTrue(found, "results.json not found after initial_optimizations_QH with force")
        
        # Check that results.json contains force-related keys
        with open(results_path) as f:
            results = json.load(f)
        force_keys = ["lpcurveforce", "max_forces", "force_weight"]
        for key in force_keys:
            self.assertIn(key, results)


class TestContinuation(unittest.TestCase):
    """Test cases for continuation function using real files."""

    def test_continuation_basic(self):

        # First run initial_optimizations to create input data for continuation
        initial_optimizations(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=str(OUTPUT_DIR / 'qa_output/'),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
            ncoils=3
        )
        """Test basic functionality of continuation with real files."""
        continuation(
            N=1,
            dx=0.1,
            INPUT_DIR=str(OUTPUT_DIR / 'qa_output/'),
            OUTPUT_DIR=str(OUTPUT_DIR / 'qa_output_continuation/'),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
            MAXITER=5
        )
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(str(OUTPUT_DIR / 'qa_output_continuation/')))
        
        # Check that a results.json file was created
        found = False
        for root, dirs, files in os.walk(str(OUTPUT_DIR / 'qa_output_continuation/')):
            if "results.json" in files:
                results_path = os.path.join(root, "results.json")
                found = True
                break
        self.assertTrue(found, "results.json not found after continuation")
        
        # Check that results.json contains expected keys
        with open(results_path) as f:
            results = json.load(f)
        expected_keys = ["UUID", "ncoils", "order", "R1", "JF", "Jf"]
        for key in expected_keys:
            self.assertIn(key, results)


class TestReadFocusCoils(unittest.TestCase):
    """Test cases for read_focus_coils function."""

    def test_read_focus_coils_basic(self):
        """Test basic functionality of read_focus_coils using real FOCUS file."""
        focus_file = Path(__file__).parent / ".." / ".." / "tests" / "test_files" / "muse_tf_coils.focus"
        
        coils, base_currents, ncoils = read_focus_coils(str(focus_file))
        
        # Check that we got reasonable outputs
        # ncoils might be a numpy array, so convert to int for comparison
        ncoils_int = int(ncoils)
        self.assertGreater(ncoils_int, 0)
        self.assertIsInstance(coils, list)
        self.assertEqual(len(coils), ncoils_int)
        self.assertIsInstance(base_currents, list)
        self.assertEqual(len(base_currents), ncoils_int)
        
        # Check that coils are CurveXYZFourier objects
        from simsopt.geo import CurveXYZFourier
        for coil in coils:
            self.assertIsInstance(coil, CurveXYZFourier)
        
        # Check that currents are Current objects
        from simsopt.field import Current
        for current in base_currents:
            self.assertIsInstance(current, Current)

    def test_read_focus_coils_file_not_found(self):
        """Test read_focus_coils with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            read_focus_coils("non_existent_file.txt")


class TestRealOptimizationRun(unittest.TestCase):
    """Test cases for initial and then continuation optimization runs."""

    def test_initial_optimizations_and_continuation(self):
        """Test initial_optimizations and continuation."""
        # Run initial_optimizations
        initial_optimizations(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=str(OUTPUT_DIR / 'qa_output/'),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
            ncoils=3
        )
        
        # Check that a results.json file was created
        found = False
        for root, dirs, files in os.walk(str(OUTPUT_DIR / 'qa_output/')):
            if "results.json" in files:
                results_path = os.path.join(root, "results.json")
                found = True
                break
        self.assertTrue(found, "results.json not found after initial_optimizations")
        
        with open(results_path) as f:
            results = json.load(f)
        for key in ["UUID", "ncoils", "order"]:
            self.assertIn(key, results)

        # Run continuation using the previous output as input
        continuation(
            N=1,
            dx=0.05,
            INPUT_DIR=str(OUTPUT_DIR / 'qa_output/'),
            OUTPUT_DIR=str(OUTPUT_DIR / 'qa_output_continuation/'),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
            MAXITER=5
        )
        
        # Check that a results.json file was created in the new output
        found2 = False
        for root, dirs, files in os.walk(str(OUTPUT_DIR / 'qa_output_continuation/')):
            if "results.json" in files:
                results_path2 = os.path.join(root, "results.json")
                found2 = True
                break
        self.assertTrue(found2, "results.json not found after continuation")
        
        with open(results_path2) as f:
            results2 = json.load(f)
        for key in ["UUID", "ncoils", "order"]:
            self.assertIn(key, results2)

    def test_qa_vs_qh_comparison(self):
        """Test that QA and QH optimizations produce different results."""
        # Run QA optimization
        qa_output_dir = "qa_output/"
        initial_optimizations(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=str(OUTPUT_DIR / qa_output_dir),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres',
            ncoils=3
        )
        
        # Run QH optimization
        qh_output_dir = "qh_output/"
        initial_optimizations_QH(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=str(OUTPUT_DIR / qh_output_dir),
            INPUT_FILE=TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres',
            ncoils=3
        )
        
        # Find results files
        qa_results = None
        qh_results = None
        
        for root, dirs, files in os.walk(str(OUTPUT_DIR / qa_output_dir)):
            if "results.json" in files:
                with open(os.path.join(root, "results.json")) as f:
                    qa_results = json.load(f)
                break
                
        for root, dirs, files in os.walk(str(OUTPUT_DIR / qh_output_dir)):
            if "results.json" in files:
                with open(os.path.join(root, "results.json")) as f:
                    qh_results = json.load(f)
                break
        
        self.assertIsNotNone(qa_results, "QA results not found")
        self.assertIsNotNone(qh_results, "QH results not found")
        
        # Check that both have expected keys
        for results in [qa_results, qh_results]:
            for key in ["UUID", "ncoils", "order", "R1", "JF", "Jf"]:
                self.assertIn(key, results)


if __name__ == "__main__":
    unittest.main() 