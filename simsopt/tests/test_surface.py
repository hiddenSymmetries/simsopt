import unittest
import os
from simsopt.surface import *

class SurfaceTests(unittest.TestCase):
    def test_init(self):
        """
        This test case checks the most common use cases.
        """
        # Try initializing a Surface without or with the optional
        # arguments:
        s = Surface()
        self.assertEqual(s.nfp, 1)
        self.assertTrue(s.stelsym)

        s = Surface(nfp=3)
        self.assertEqual(s.nfp, 3)
        self.assertTrue(s.stelsym, True)

        s = Surface(stelsym=False)
        self.assertEqual(s.nfp, 1)
        self.assertFalse(s.stelsym)

        # Now let's check that we can change nfp and stelsym.
        s.nfp = 5
        self.assertEqual(s.nfp, 5)
        self.assertFalse(s.stelsym)

        s.stelsym = True
        self.assertEqual(s.nfp, 5)
        self.assertTrue(s.stelsym)

class SurfaceRZFourierTests(unittest.TestCase):
    def test_init(self):
        s = SurfaceRZFourier(nfp=2, mpol=3, ntor=2)
        self.assertEqual(s.rc.shape, (4, 5))
        self.assertEqual(s.zs.shape, (4, 5))

        s = SurfaceRZFourier(nfp=10, mpol=1, ntor=3, stelsym=False)
        self.assertEqual(s.rc.shape, (2, 7))
        self.assertEqual(s.zs.shape, (2, 7))
        self.assertEqual(s.rs.shape, (2, 7))
        self.assertEqual(s.zc.shape, (2, 7))

    def test_area_volume(self):
        """
        Test the calculation of area and volume for an axisymmetric surface
        """
        s = SurfaceRZFourier()
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[1, 0] = 0.2

        area, volume = s.area_volume()
        true_area = 15.827322032265993
        true_volume = 2.0528777154265874
        self.assertAlmostEqual(area, true_area, places=4)
        self.assertAlmostEqual(volume, true_volume, places=3)
        self.assertAlmostEqual(s.compute_area(), true_area, places=4)
        self.assertAlmostEqual(s.compute_volume(), true_volume, places=3)

    def test_from_focus(self):
        """
        Try reading in a focus-format file.
        """
        base_filename = 'tf_only_half_tesla.plasma'
        filename2 = os.path.join("simsopt", "tests", base_filename)
        if os.path.isfile(base_filename):
            filename = base_filename
        elif os.path.isfile(filename2):
            filename = filename2
        else:
            raise RuntimeError("Unable to find test file " + base_filename)
        s = SurfaceRZFourier.from_focus(filename)

        self.assertEqual(s.nfp, 3)
        self.assertTrue(s.stelsym)
        self.assertEqual(s.rc.shape, (11, 13))
        self.assertEqual(s.zs.shape, (11, 13))
        self.assertAlmostEqual(s.rc[0, 6], 1.408922E+00)
        self.assertAlmostEqual(s.rc[0, 7], 2.794370E-02)
        self.assertAlmostEqual(s.zs[0, 7], -1.909220E-02)
        self.assertAlmostEqual(s.rc[10, 12], -6.047097E-05)
        self.assertAlmostEqual(s.zs[10, 12], 3.663233E-05)

        self.assertAlmostEqual(s.get_rc(0,0), 1.408922E+00)
        self.assertAlmostEqual(s.get_rc(0,1), 2.794370E-02)
        self.assertAlmostEqual(s.get_zs(0,1), -1.909220E-02)
        self.assertAlmostEqual(s.get_rc(10,6), -6.047097E-05)
        self.assertAlmostEqual(s.get_zs(10,6), 3.663233E-05)

        area, volume = s.area_volume()
        true_area = 24.5871075268402
        true_volume = 2.96201898538042
        #print("computed area: ", area, ", correct value: ", true_area, \
        #    " , difference: ", area - true_area)
        #print("computed volume: ", volume, ", correct value: ", \
        #    true_volume, ", difference:", volume - true_volume)
        self.assertAlmostEqual(area, true_area, places=4)
        self.assertAlmostEqual(volume, true_volume, places=3)
        self.assertAlmostEqual(s.compute_area(), true_area, places=4)
        self.assertAlmostEqual(s.compute_volume(), true_volume, places=3)

if __name__ == "__main__":
    unittest.main()
