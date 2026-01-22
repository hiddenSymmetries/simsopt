"""
Integration tests for QUASR database connectivity.

These tests make actual network calls to the QUASR database and are skipped
by default. To run these tests, set the environment variable:

    SIMSOPT_INTEGRATION_TESTS=1 python -m unittest tests.configs.test_quasr_integration

Or run from the tests directory:

    SIMSOPT_INTEGRATION_TESTS=1 python -m unittest configs.test_quasr_integration
"""

import os
import unittest

from simsopt.configs import get_data, download_ID_from_QUASR_database
from simsopt.geo import CurveXYZFourier, SurfaceXYZTensorFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.field.coil import ScaledCurrent, Coil
from simsopt.field.biotsavart import BiotSavart


# Skip integration tests unless explicitly enabled
SKIP_INTEGRATION = os.environ.get("SIMSOPT_INTEGRATION_TESTS", "0") != "1"
SKIP_REASON = (
    "Integration tests are disabled by default. "
    "Set SIMSOPT_INTEGRATION_TESTS=1 to enable."
)


@unittest.skipIf(SKIP_INTEGRATION, SKIP_REASON)
class QuasrIntegrationTests(unittest.TestCase):
    """
    Integration tests that make actual calls to the QUASR database.

    These tests verify that the QUASR database is accessible and returns
    data in the expected format.
    """

    def test_download_quasr_simsopt_style(self):
        """
        Test downloading a configuration from QUASR in simsopt-style format.
        Uses ID=952 as a known good configuration.
        """
        base_curves, base_currents, ma, nfp, bs = get_data("quasr", QUASR_ID=952)

        # Verify return types
        self.assertIsInstance(base_curves, list)
        self.assertGreater(len(base_curves), 0)
        for curve in base_curves:
            self.assertIsInstance(curve, CurveXYZFourier)

        self.assertIsInstance(base_currents, list)
        self.assertEqual(len(base_currents), len(base_curves))
        for current in base_currents:
            self.assertIsInstance(current, ScaledCurrent)

        # Note: ma is None for QUASR configurations (axis finding not implemented)
        self.assertIsNone(ma)
        self.assertIsInstance(nfp, int)
        self.assertGreater(nfp, 0)
        self.assertIsInstance(bs, BiotSavart)

    def test_download_quasr_quasr_style(self):
        """
        Test downloading a configuration from QUASR in quasr-style format.
        Uses ID=952 as a known good configuration.
        """
        surfaces, coils = download_ID_from_QUASR_database(
            952, return_style='quasr-style'
        )

        # Verify return types
        self.assertIsInstance(surfaces, list)
        self.assertGreater(len(surfaces), 0)
        for surface in surfaces:
            self.assertIsInstance(surface, SurfaceXYZTensorFourier)

        self.assertIsInstance(coils, list)
        self.assertGreater(len(coils), 0)
        for coil in coils:
            self.assertIsInstance(coil, Coil)

    def test_download_quasr_with_caching(self):
        """
        Test that caching works correctly by downloading the same ID twice.
        The second download should use the cached version.
        """
        # First download (may or may not be cached)
        result1 = get_data("quasr", QUASR_ID=952)

        # Second download (should use cache)
        result2 = get_data("quasr", QUASR_ID=952)

        # Both should return the same structure
        base_curves1, base_currents1, ma1, nfp1, bs1 = result1
        base_curves2, base_currents2, ma2, nfp2, bs2 = result2

        self.assertEqual(len(base_curves1), len(base_curves2))
        self.assertEqual(len(base_currents1), len(base_currents2))
        self.assertEqual(nfp1, nfp2)

    def test_download_quasr_without_caching(self):
        """
        Test downloading with caching disabled.
        """
        base_curves, base_currents, ma, nfp, bs = get_data(
            "quasr", QUASR_ID=952, use_cache=False
        )

        # Verify return types
        self.assertIsInstance(base_curves, list)
        self.assertGreater(len(base_curves), 0)
        # Note: ma is None for QUASR configurations (axis finding not implemented)
        self.assertIsNone(ma)
        self.assertIsInstance(nfp, int)
        self.assertIsInstance(bs, BiotSavart)

    def test_invalid_quasr_id_raises_error(self):
        """
        Test that requesting a non-existent QUASR ID raises an error.
        Uses ID=0 which should not exist.
        """
        with self.assertRaises(Exception):
            get_data("quasr", QUASR_ID=0)


if __name__ == "__main__":
    unittest.main()
