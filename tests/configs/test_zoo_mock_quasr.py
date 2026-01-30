import unittest
from simsopt.configs import get_data, download_ID_from_QUASR_database
from simsopt.configs.zoo import _prune_cache
from simsopt.geo import CurveXYZFourier, SurfaceXYZTensorFourier
from simsopt.field.coil import ScaledCurrent, Coil
from simsopt._core import load
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import time


from pathlib import Path
THIS_DIR = (Path(__file__).parent).resolve()


class QuasrTests(unittest.TestCase):
    @patch("simsopt.configs.zoo.requests")
    def test_QUASR_downloader(self, mock_requests):
        """
        This unit test checks that the get_QUASR_data functionality works as expected.
        We download the device with ID=0000952 is downloaded correctly.  We also check that
        exceptions are raised if an ID is requested, but the associated device
        does not exist, or if the improper return style is passed.

        The data is downloaded in get_QUASR_data using requests.get, which we mock in this
        unit test.
        """

        THIS_DIR = (Path(__file__).parent).resolve()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with open(THIS_DIR / '../test_files/serial0000952.json', "rb") as f:
            raw_bytes = f.read()
        mock_response.content = raw_bytes
        mock_requests.get.return_value = mock_response

        true_surfaces, true_coils = load(THIS_DIR / '../test_files/serial0000952.json')

        base_curves, base_currents, ma, nfp, bs = get_data("quasr", QUASR_ID=952)
        assert isinstance(base_curves[0], CurveXYZFourier)
        assert isinstance(base_currents[0], ScaledCurrent)
        np.testing.assert_allclose(base_curves[0].x, true_coils[0].curve.x)
        np.testing.assert_allclose(base_currents[0].get_value(), true_coils[0].current.get_value())

        surfaces, coils = download_ID_from_QUASR_database(952, return_style='quasr-style', verbose=True)
        assert isinstance(coils[0], Coil)
        assert isinstance(surfaces[0], SurfaceXYZTensorFourier)
        np.testing.assert_allclose(surfaces[0].x, true_surfaces[0].x)
        np.testing.assert_allclose(coils[0].x, true_coils[0].x)

        # invalid return style
        with self.assertRaises(Exception):
            surfaces, coils = download_ID_from_QUASR_database(952, return_style='invalid-style')

        # 404 means the file is not found
        mock_response.status_code = 404
        with self.assertRaises(Exception):
            base_curves, base_currents, ma, nfp, bs = get_data("quasr", 0)

        # no id provided
        with self.assertRaises(Exception):
            base_curves, base_currents, ma, nfp, bs = get_data("quasr")

        
        # requests.get raises an exception
        mock_requests.get.side_effect = Exception("something went wrong")
        with self.assertRaises(Exception):
            base_curves, base_currents, ma, nfp, bs = get_data("quasr", 952, return_style='')
        
        with self.assertRaises(Exception):
            # mock os.makedir only here so it throws an error
            with patch("simsopt.configs.zoo.os.makedirs") as mock_makedirs:
                mock_makedirs.side_effect = Exception("Failed to create directory")
                base_curves, base_currents, ma, nfp, bs = get_data(953, return_style='quasr-style')

        # reset mock for permission tests
        mock_requests.get.side_effect = None
        mock_requests.get.return_value = mock_response
        mock_response.status_code = 200

        # test fallback to cwd when THIS_DIR is not writable (line 596)
        # Also verify that cache directory is created and file is written
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("simsopt.configs.zoo.os.access") as mock_access, \
                 patch("simsopt.configs.zoo.os.getcwd", return_value=tmpdir):
                # First call (THIS_DIR) returns False, second call (cwd) returns True
                mock_access.side_effect = [False, True]
                download_ID_from_QUASR_database(952, use_cache=True, verbose=False)
                # Verify os.access was called twice
                self.assertEqual(mock_access.call_count, 2)
                # Verify cache directory was created
                cache_dir = Path(tmpdir) / 'QUASR_cache'
                self.assertTrue(cache_dir.exists())
                self.assertTrue(cache_dir.is_dir())
                # Verify JSON file was written
                cache_files = list(cache_dir.glob("*.json"))
                self.assertEqual(len(cache_files), 1)
                self.assertEqual(cache_files[0].name, "serial0000952.json")

        # test PermissionError when no writable location (lines 597-598)
        with patch("simsopt.configs.zoo.os.access", return_value=False):
            with self.assertRaises(PermissionError):
                download_ID_from_QUASR_database(952, use_cache=True)

    def test_prune_cache(self):
        """
        Test that _prune_cache removes oldest files when cache exceeds limit.
        Creates 101 empty .json files and verifies that the oldest file is removed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create 101 empty .json files with staggered modification times
            for i in range(101):
                filepath = cache_dir / f"test_file_{i:03d}.json"
                filepath.touch()
                # Ensure distinct modification times by setting mtime explicitly
                # Oldest file has the smallest mtime
                mtime = time.time() - (101 - i)
                filepath.touch()
                import os
                os.utime(filepath, (mtime, mtime))
            
            # Verify we have 101 files
            files_before = list(cache_dir.glob("*.json"))
            self.assertEqual(len(files_before), 101)
            
            # Run prune with default limit of 100
            _prune_cache(cache_dir, limit=100, verbose=True) # hit the print line to satisfy coverage
            
            # Verify we now have 100 files
            files_after = list(cache_dir.glob("*.json"))
            self.assertEqual(len(files_after), 100)
            
            # Verify the oldest file (test_file_000.json) was removed
            remaining_names = {f.name for f in files_after}
            self.assertNotIn("test_file_000.json", remaining_names)
            self.assertIn("test_file_001.json", remaining_names)
            self.assertIn("test_file_100.json", remaining_names)


if __name__ == "__main__":
    unittest.main()
