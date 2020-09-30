import unittest
import numpy as np
import os
from simsopt.vmec import *

class VmecTests(unittest.TestCase):
    def test_init_defaults(self):
        """
        Just create a Vmec instance using the standard constructor,
        and make sure we can read some of the attributes.
        """
        v = Vmec()
        self.assertEqual(v.nfp, 5)
        self.assertTrue(v.stelsym)
        self.assertEqual(v.mpol, 5)
        self.assertEqual(v.ntor, 4)
        self.assertEqual(v.delt, 0.5)
        self.assertEqual(v.tcon0, 2.0)
        self.assertEqual(v.phiedge, 1.0)
        self.assertEqual(v.curtor, 0.0)
        self.assertEqual(v.gamma, 0.0)
        self.assertEqual(v.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)
        v.finalize()

    def test_init_from_file(self):
        """
        Try creating a Vmec instance from a specified input file.
        """

        filename = os.path.join(os.path.dirname(__file__), \
                                    'input.li383_low_res')

        v = Vmec(filename)
        self.assertEqual(v.nfp, 3)
        self.assertEqual(v.mpol, 4)
        self.assertEqual(v.ntor, 3)
        self.assertEqual(v.boundary.mpol, 4)
        self.assertEqual(v.boundary.ntor, 3)

        # n = 0, m = 0:
        self.assertAlmostEqual(v.boundary.get_rc(0, 0), 1.3782)

        # n = 0, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 0), 4.6465E-01)

        # n = 1, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 1), 1.6516E-01)

        self.assertEqual(v.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)

        v.finalize()

if __name__ == "__main__":
    unittest.main()
