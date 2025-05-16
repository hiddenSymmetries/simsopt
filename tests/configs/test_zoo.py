import unittest
from simsopt.configs import get_QUASR_data

class ZooTests(unittest.TestCase):
    def test_QUASR_downloader(self):
        curves, currents = get_QUASR_data(952, return_style='simsopt-style')
        coils, surfaces = get_QUASR_data(952, return_style='quasr-style')
        
        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(0)

        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(952, return_style='')

if __name__ == "__main__":
    unittest.main()
