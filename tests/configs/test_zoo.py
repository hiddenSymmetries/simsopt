import unittest
from simsopt.configs import get_QUASR_data

class ZooTests(unittest.TestCase):
    def test_QUASR_downloader(self):
        curves, currents, ma = get_QUASR_data(952)
        coils, ma, surfaces = get_QUASR_data(952, return_style='json')
        
        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(0)

        with self.assertRaises(Exception):
            curves, currents, ma = get_QUASR_data(952, return_style='')

if __name__ == "__main__":
    unittest.main()
