import unittest
from simsopt.configs import get_QUASR_data
from simsopt.geo import CurveXYZFourier, SurfaceXYZTensorFourier
from simsopt.field.coil import ScaledCurrent, Coil
from simsopt._core import load
from unittest.mock import patch, MagicMock
import numpy as np

from pathlib import Path
THIS_DIR = (Path(__file__).parent).resolve()

class ZooTests(unittest.TestCase):
    @patch("simsopt.configs.zoo.requests.get")
    def test_QUASR_downloader(self, mock_get):
        """
        This unit test checks that the get_QUASR_data functionality works as expected.
        We download the device with ID=2680021 is downloaded correctly.  We also check that
        exceptions are raised if an ID is requested, but the associated device
        does not exist, or if the improper return style is passed.

        The data is downloaded in get_QUASR_data using requests.get, which we mock in this
        unit test.
        """

        THIS_DIR = (Path(__file__).parent).resolve()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with open(THIS_DIR / '../test_files/serial2680021.json', "rb") as f:
            raw_bytes = f.read()
        mock_response.content = raw_bytes
        mock_get.return_value = mock_response

        true_surfaces, true_coils = load(THIS_DIR / '../test_files/serial2680021.json')

        curves, currents = get_QUASR_data(2680021, return_style='simsopt-style')
        assert isinstance(curves[0], CurveXYZFourier)
        assert isinstance(currents[0], ScaledCurrent)
        np.testing.assert_allclose(curves[0].x, true_coils[0].curve.x)
        np.testing.assert_allclose(currents[0].get_value(), true_coils[0].current.get_value())

        surfaces, coils = get_QUASR_data(2680021, return_style='quasr-style')
        assert isinstance(coils[0], Coil)
        assert isinstance(surfaces[0], SurfaceXYZTensorFourier)
        np.testing.assert_allclose(surfaces[0].x, true_surfaces[0].x)
        np.testing.assert_allclose(coils[0].x, true_coils[0].x)
        
        # 404 means the file is not found
        mock_response.status_code = 404
        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(0)
       
        # wrong return style
        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(2680021, return_style='')
        
        # requests.get raises an exception
        mock_response.side_effect = Exception("something went wrong")
        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(2680021, return_style='')

if __name__ == "__main__":
    unittest.main()
