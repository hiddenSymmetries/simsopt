import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from simsopt.util.coil_optimization_helper_functions import (
    initial_optimizations, 
    initial_optimizations_QH
)
from simsopt.field import LpCurveForce, SquaredMeanForce, LpCurveTorque, SquaredMeanTorque, B2Energy


class TestInitialOptimizations(unittest.TestCase):
    """Test cases for initial_optimizations function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_file = Path(__file__).parent / ".." / ".." / "tests" / "test_files" / "input.LandremanPaul2021_QA"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_basic(self, mock_optimization):
        """Test basic functionality of initial_optimizations."""
        # Mock the optimization function to avoid actual optimization
        mock_optimization.return_value = None
        
        # Test with minimal parameters
        initial_optimizations(
            N=2,  # Small number for testing
            MAXITER=10,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            debug=False,
            ncoils=3
        )
        
        # Check that optimization was called twice (N=2)
        self.assertEqual(mock_optimization.call_count, 2)
        
        # Check that the calls were made with expected parameters
        calls = mock_optimization.call_args_list
        for call in calls:
            args, kwargs = call
            # Check that required parameters are present in positional args
            self.assertEqual(len(args), 20)  # 20 positional arguments
            self.assertIn('with_force', kwargs)
            self.assertIn('debug', kwargs)
            self.assertIn('MAXITER', kwargs)
            
            # Check parameter ranges (positional arguments)
            self.assertTrue(0.35 <= args[2] <= 0.75)  # R1
            self.assertTrue(5 <= args[8] <= 12)  # CURVATURE_THRESHOLD
            self.assertTrue(4 <= args[10] <= 6)  # MSC_THRESHOLD
            self.assertTrue(0.166 <= args[14] <= 0.300)  # CS_THRESHOLD
            self.assertTrue(0.083 <= args[12] <= 0.120)  # CC_THRESHOLD
            self.assertTrue(0 <= args[16] <= 5e+04)  # FORCE_THRESHOLD
            self.assertTrue(4.9 <= args[6] <= 5.0)  # LENGTH_TARGET

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_with_force_objective(self, mock_optimization):
        """Test initial_optimizations with force objective."""
        mock_optimization.return_value = None
        
        # Test with force objective
        initial_optimizations(
            N=1,
            MAXITER=10,
            FORCE_OBJ=LpCurveForce,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            debug=True,
            ncoils=4
        )
        
        # Check that optimization was called with force parameters
        call_args = mock_optimization.call_args
        args, kwargs = call_args
        
        self.assertTrue(kwargs['with_force'])
        self.assertEqual(args[18], LpCurveForce)  # FORCE_OBJ is positional arg
        self.assertTrue(1e-13 <= args[17] <= 1e-8)  # FORCE_WEIGHT is positional arg

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_without_force_objective(self, mock_optimization):
        """Test initial_optimizations without force objective."""
        mock_optimization.return_value = None
        
        # Test without force objective
        initial_optimizations(
            N=1,
            MAXITER=10,
            FORCE_OBJ=None,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            debug=False,
            ncoils=5
        )
        
        # Check that optimization was called without force parameters
        call_args = mock_optimization.call_args
        args, kwargs = call_args
        
        self.assertFalse(kwargs['with_force'])
        self.assertEqual(args[17], 0)  # FORCE_WEIGHT is positional arg

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_parameter_ranges(self, mock_optimization):
        """Test that parameter ranges are within expected bounds."""
        mock_optimization.return_value = None
        
        # Run multiple iterations to test parameter ranges
        initial_optimizations(
            N=10,
            MAXITER=5,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            debug=False,
            ncoils=3
        )
        
        calls = mock_optimization.call_args_list
        for call in calls:
            args, kwargs = call
            # Test weight parameter ranges (positional arguments)
            self.assertTrue(1e-4 <= args[7] <= 1e-2)  # LENGTH_WEIGHT
            self.assertTrue(1e-9 <= args[9] <= 1e-5)  # CURVATURE_WEIGHT
            self.assertTrue(1e-7 <= args[11] <= 1e-3)  # MSC_WEIGHT
            self.assertTrue(1e-1 <= args[15] <= 1e+4)  # CS_WEIGHT
            self.assertTrue(1e+2 <= args[13] <= 1e+5)  # CC_WEIGHT

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_different_force_objectives(self, mock_optimization):
        """Test initial_optimizations with different force objectives."""
        mock_optimization.return_value = None
        
        force_objectives = [LpCurveForce, SquaredMeanForce, B2Energy]
        
        for force_obj in force_objectives:
            with self.subTest(force_obj=force_obj.__name__):
                initial_optimizations(
                    N=1,
                    MAXITER=5,
                    FORCE_OBJ=force_obj,
                    OUTPUT_DIR=self.temp_dir + "/test_output/",
                    INPUT_FILE=str(self.test_input_file),
                    debug=False,
                    ncoils=3
                )
                
                call_args = mock_optimization.call_args
                args, kwargs = call_args
                
                self.assertTrue(kwargs['with_force'])
                self.assertEqual(args[18], force_obj)  # FORCE_OBJ is positional arg

    def test_initial_optimizations_invalid_input_file(self):
        """Test that initial_optimizations handles invalid input file gracefully."""
        with self.assertRaises(Exception):
            initial_optimizations(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=self.temp_dir + "/test_output/",
                INPUT_FILE="nonexistent_file.txt",
                debug=False,
                ncoils=3
            )


class TestInitialOptimizationsQH(unittest.TestCase):
    """Test cases for initial_optimizations_QH function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_file = Path(__file__).parent / ".." / ".." / "tests" / "test_files" / "input.LandremanPaul2021_QH_magwell_R0=1"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_QH_basic(self, mock_optimization):
        """Test basic functionality of initial_optimizations_QH."""
        # Mock the optimization function to return expected values
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-123'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        # Test with minimal parameters
        initial_optimizations_QH(
            N=2,  # Small number for testing
            MAXITER=10,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            ncoils=3
        )
        
        # Check that optimization was called twice (N=2)
        self.assertEqual(mock_optimization.call_count, 2)
        
        # Check that the calls were made with expected parameters
        calls = mock_optimization.call_args_list
        for call in calls:
            args, kwargs = call
            print(args)
            print(kwargs)
            # Check that required parameters are present in positional args
            self.assertEqual(len(args), 20, msg="There should be 20 positional arguments")
            self.assertIn('MAXITER', kwargs, msg="MAXITER should be present in kwargs")
            self.assertIn('with_force', kwargs, msg="with_force should be present in kwargs")
            self.assertFalse(kwargs['with_force'], msg="with_force should be False")
            # Check parameter ranges (different from QA)
            self.assertTrue(0.35 <= args[2] <= 0.75, msg="R1 should be between 0.35 and 0.75")  # R1
            self.assertTrue(5 <= args[8] <= 12, msg="CURVATURE_THRESHOLD should be between 5 and 12")  # CURVATURE_THRESHOLD
            self.assertTrue(4 <= args[10] <= 6, msg="MSC_THRESHOLD should be between 4 and 6")  # MSC_THRESHOLD
            self.assertTrue(0.166 <= args[14] <= 0.300, msg="CS_THRESHOLD should be between 0.166 and 0.300")  # CS_THRESHOLD
            self.assertTrue(0.083 <= args[12] <= 0.120, msg="CC_THRESHOLD should be between 0.083 and 0.120")  # CC_THRESHOLD
            self.assertTrue(0 <= args[16] <= 5e+04, msg="FORCE_THRESHOLD should be between 0 and 5e+04")  # FORCE_THRESHOLD
            self.assertTrue(4.9 <= args[6] <= 5.0, msg="LENGTH_TARGET should be between 4.9 and 5.0")  # LENGTH_TARGET

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_QH_with_force_objective(self, mock_optimization):
        """Test initial_optimizations_QH with force objective."""
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-456'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        # Test with force objective
        initial_optimizations_QH(
            N=1,
            MAXITER=10,
            FORCE_OBJ=LpCurveForce,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            ncoils=4
        )
        
        # Check that optimization was called with force parameters
        call_args = mock_optimization.call_args
        args, kwargs = call_args
        
        self.assertTrue(kwargs['with_force'])
        self.assertTrue(1e-14 <= args[17] <= 1e-9)  # FORCE_WEIGHT is positional arg

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_QH_without_force_objective(self, mock_optimization):
        """Test initial_optimizations_QH without force objective."""
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-789'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        # Test without force objective
        initial_optimizations_QH(
            N=1,
            MAXITER=10,
            FORCE_OBJ=None,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            ncoils=5
        )
        
        # Check that optimization was called without force parameters
        call_args = mock_optimization.call_args
        args, kwargs = call_args
        
        self.assertFalse(kwargs['with_force'])
        self.assertEqual(args[17], 0)  # FORCE_WEIGHT is positional arg

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_QH_parameter_ranges(self, mock_optimization):
        """Test that QH parameter ranges are within expected bounds."""
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-param'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        # Run multiple iterations to test parameter ranges
        initial_optimizations_QH(
            N=10,
            MAXITER=5,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            ncoils=3
        )
        
        calls = mock_optimization.call_args_list
        for call in calls:
            args, kwargs = call
            # Test weight parameter ranges (positional arguments)
            self.assertTrue(1e-3 <= args[7] <= 1e-1)  # LENGTH_WEIGHT
            self.assertTrue(1e-9 <= args[9] <= 1e-5)  # CURVATURE_WEIGHT
            self.assertTrue(1e-5 <= args[11] <= 1e-1)  # MSC_WEIGHT
            self.assertTrue(1e-1 <= args[15] <= 1e+4)  # CS_WEIGHT
            self.assertTrue(1e+2 <= args[13] <= 1e+5)  # CC_WEIGHT

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_QH_different_force_objectives(self, mock_optimization):
        """Test initial_optimizations_QH with different force objectives."""
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-force'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        force_objectives = [LpCurveForce, SquaredMeanForce, LpCurveTorque, SquaredMeanTorque, B2Energy]
        
        for force_obj in force_objectives:
            with self.subTest(force_obj=force_obj.__name__):
                initial_optimizations_QH(
                    N=1,
                    MAXITER=5,
                    FORCE_OBJ=force_obj,
                    OUTPUT_DIR=self.temp_dir + "/test_output/",
                    INPUT_FILE=str(self.test_input_file),
                    ncoils=3
                )
                
                call_args = mock_optimization.call_args
                args, kwargs = call_args
                
                self.assertTrue(kwargs['with_force'])

    def test_initial_optimizations_QH_invalid_input_file(self):
        """Test that initial_optimizations_QH handles invalid input file gracefully."""
        with self.assertRaises(Exception):
            initial_optimizations_QH(
                N=1,
                MAXITER=5,
                OUTPUT_DIR=self.temp_dir + "/test_output/",
                INPUT_FILE="nonexistent_file.txt",
                ncoils=3
            )

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_initial_optimizations_QH_return_values(self, mock_optimization):
        """Test that initial_optimizations_QH properly handles return values from optimization."""
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-return'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        # Test that the function completes without error
        initial_optimizations_QH(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=self.temp_dir + "/test_output/",
            INPUT_FILE=str(self.test_input_file),
            ncoils=3
        )
        
        # Verify that optimization was called
        mock_optimization.assert_called_once()


class TestInitialOptimizationsComparison(unittest.TestCase):
    """Test cases comparing QA and QH optimization functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.qa_input_file = Path(__file__).parent / ".." / ".." / "tests" / "test_files" / "input.LandremanPaul2021_QA"
        self.qh_input_file = Path(__file__).parent / ".." / ".." / "tests" / "test_files" / "input.LandremanPaul2021_QH_magwell_R0=1"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_qa_vs_qh_parameter_differences(self, mock_optimization):
        """Test that QA and QH functions use different parameter ranges."""
        mock_optimization.return_value = None
        
        # Run QA optimization
        initial_optimizations(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=self.temp_dir + "/qa_output/",
            INPUT_FILE=str(self.qa_input_file),
            ncoils=5
        )
        
        qa_call = mock_optimization.call_args
        qa_args, qa_kwargs = qa_call
        
        # Reset mock
        mock_optimization.reset_mock()
        
        # Run QH optimization
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-compare'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        initial_optimizations_QH(
            N=1,
            MAXITER=5,
            OUTPUT_DIR=self.temp_dir + "/qh_output/",
            INPUT_FILE=str(self.qh_input_file),
            ncoils=3
        )
        
        qh_call = mock_optimization.call_args
        qh_args, qh_kwargs = qh_call
        
        # Compare parameter ranges
        # QH should have different weight ranges than QA
        self.assertNotEqual(qa_args[7], qh_args[7])  # LENGTH_WEIGHT
        self.assertNotEqual(qa_args[11], qh_args[11])  # MSC_WEIGHT
        
        # Check that QH uses different ncoils default
        self.assertEqual(qa_args[4], 5)
        self.assertEqual(qh_args[4], 3)

    @patch('simsopt.util.coil_optimization_helper_functions.optimization')
    def test_qa_vs_qh_force_weight_ranges(self, mock_optimization):
        """Test that QA and QH use different force weight ranges."""
        mock_optimization.return_value = None
        
        # Test QA force weights
        initial_optimizations(
            N=1,
            MAXITER=5,
            FORCE_OBJ=LpCurveForce,
            OUTPUT_DIR=self.temp_dir + "/qa_output/",
            INPUT_FILE=str(self.qa_input_file),
            ncoils=3
        )
        
        qa_call = mock_optimization.call_args
        qa_args, qa_kwargs = qa_call
        qa_force_weight = qa_args[17]
        
        # Reset mock
        mock_optimization.reset_mock()
        
        # Test QH force weights
        mock_result = MagicMock()
        mock_results = {'UUID': 'test-uuid-force-compare'}
        mock_coils = [MagicMock()]
        mock_optimization.return_value = (mock_result, mock_results, mock_coils)
        
        initial_optimizations_QH(
            N=1,
            MAXITER=5,
            FORCE_OBJ=LpCurveForce,
            OUTPUT_DIR=self.temp_dir + "/qh_output/",
            INPUT_FILE=str(self.qh_input_file),
            ncoils=3
        )
        
        qh_call = mock_optimization.call_args
        qh_args, qh_kwargs = qh_call
        qh_force_weight = qh_args[17]
        
        # Both should be within their respective ranges
        self.assertTrue(1e-13 <= qa_force_weight <= 1e-8)
        self.assertTrue(1e-14 <= qh_force_weight <= 1e-9)


if __name__ == '__main__':
    unittest.main() 