import unittest
from simsopt.configs import get_QUASR_data
from simsopt.geo import CurveXYZFourier, SurfaceXYZTensorFourier
from simsopt.field.coil import ScaledCurrent, Coil

class ZooTests(unittest.TestCase):
    def test_QUASR_downloader(self):
        """
        This unit test checks that the get_QUASR_data functionality works as expected.
        We download the device with ID=0000952 is downloaded correctly.  We also check that
        exceptions are raised if an ID is requested, but the associated device
        does not exist, or if the improper return style is passed.
        """

        curves, currents = get_QUASR_data(952, return_style='simsopt-style')
        assert isinstance(curves[0], CurveXYZFourier)
        assert isinstance(currents[0], ScaledCurrent)

        surfaces, coils = get_QUASR_data(952, return_style='quasr-style')
        assert isinstance(coils[0], Coil)
        assert isinstance(surfaces[0], SurfaceXYZTensorFourier)
        
        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(0)

        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(952, return_style='')

if __name__ == "__main__":
    unittest.main()
