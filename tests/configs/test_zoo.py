import unittest

from simsopt.configs.zoo import get_data, configurations
from simsopt.geo import Curve
from simsopt.field.coil import Current
from simsopt.field.biotsavart import BiotSavart


class TestGetData(unittest.TestCase):
    """Tests for the get_data function across all configurations except QUASR."""

    def test_all_configurations(self):
        """Test that all configurations (except QUASR) return correct types."""
        for config_name in configurations:
            if config_name.upper() == "QUASR":
                continue  # Skip QUASR for now

            with self.subTest(config=config_name):
                result = get_data(config_name)

                # Check that result is a 5-tuple
                self.assertEqual(
                    len(result), 5,
                    f"Expected 5 elements for {config_name}, got {len(result)}")

                base_curves, base_currents, ma, nfp, bs = result

                # Check base_curves is a list of Curve objects
                self.assertIsInstance(
                    base_curves, list,
                    f"base_curves should be a list for {config_name}")
                self.assertGreater(
                    len(base_curves), 0,
                    f"base_curves should not be empty for {config_name}")
                for i, curve in enumerate(base_curves):
                    self.assertIsInstance(
                        curve, Curve,
                        f"base_curves[{i}] should be a Curve for {config_name}, got {type(curve)}")

                # Check base_currents is a list of Current objects
                self.assertIsInstance(
                    base_currents, list,
                    f"base_currents should be a list for {config_name}")
                self.assertEqual(
                    len(base_currents), len(base_curves),
                    f"base_currents length should match base_curves for {config_name}")
                for i, current in enumerate(base_currents):
                    self.assertIsInstance(
                        current, Current,
                        f"base_currents[{i}] should be a Current for {config_name}, got {type(current)}")

                # Check ma is a CurveRZFourier
                self.assertIsInstance(
                    ma, Curve,
                    f"ma should be a Curve for {config_name}, got {type(ma)}")

                # Check nfp is an int
                self.assertIsInstance(
                    nfp, int,
                    f"nfp should be an int for {config_name}, got {type(nfp)}")
                self.assertGreater(
                    nfp, 0,
                    f"nfp should be positive for {config_name}")

                # Check bs is a BiotSavart object
                self.assertIsInstance(
                    bs, BiotSavart,
                    f"bs should be a BiotSavart for {config_name}, got {type(bs)}")

    def test_invalid_configuration_raises_error(self):
        """Test that an invalid configuration name raises ValueError."""
        with self.assertRaises(ValueError):
            get_data("invalid_config")

    def test_case_insensitivity(self):
        """Test that configuration names are case-insensitive."""
        # Test with ncsx in different cases
        result_lower = get_data("ncsx")
        result_upper = get_data("NCSX")
        result_mixed = get_data("NcSx")

        # All should return the same structure
        self.assertEqual(len(result_lower), 5)
        self.assertEqual(len(result_upper), 5)
        self.assertEqual(len(result_mixed), 5)


if __name__ == "__main__":
    unittest.main()
