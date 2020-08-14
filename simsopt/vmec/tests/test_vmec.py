import unittest
import numpy as np
import os
from simsopt.vmec import *

class VmecTests(unittest.TestCase):
    def test_init(self):
        """
        Just create a Vmec instance using the standard constructor,
        and make sure we can read all the attributes.
        """
        v = Vmec()
        self.assertEqual(v.nfp.val, 1)
        self.assertTrue(v.stelsym.val)
        self.assertEqual(v.mpol.val, 1)
        self.assertEqual(v.ntor.val, 0)
        self.assertEqual(v.delt.val, 0.7)
        self.assertEqual(v.tcon0.val, 2.0)
        self.assertEqual(v.phiedge.val, 1.0)
        self.assertEqual(v.curtor.val, 0.0)
        self.assertEqual(v.gamma.val, 0.0)
        self.assertEqual(v.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)

    def test_parse_namelist_var(self):
        """
        Try adding a variable from an input namelist to a Vmec instance.
        """
        v = Vmec()
        myvars = {"foo":7, "bar":8, "oof":False}
        # Try a variable that IS in the namelist:
        v._parse_namelist_var(myvars, "foo", 12)
        self.assertEqual(v.foo.val, 7)
        # Try a variable that is NOT in the namelist:
        v._parse_namelist_var(myvars, "zzz", 13)
        self.assertEqual(v.zzz.val, 13)
        # Try renaming a variable:
        v._parse_namelist_var(myvars, "bar", -7, new_name="blorp")
        self.assertEqual(v.blorp.val, 8)
        # Try a variable that is not a parameter:
        v._parse_namelist_var(myvars, "nerp", -5, parameter=False)
        self.assertEqual(v.nerp, -5)

#    def test_from_input_file(self):
#        """
#        Try reading in a VMEC input namelist.
#        """
#        # We might run this script from this directory or from the
#        # project root directory. Handle both cases.
#        base_filename = "input.li383_1.4m"
#        filename2 = os.path.join("simsopt", "tests", base_filename)
#        if os.path.isfile(base_filename):
#            filename = base_filename
#        elif os.path.isfile(filename2):
#            filename = filename2
#        else:
#            raise RuntimeError("Unable to find test file " + base_filename)
#        v = Vmec.from_input_file(filename)
#
#        self.assertEqual(v.nfp.val, 3)
#        self.assertTrue(v.stelsym.val)
#        self.assertEqual(v.mpol.val, 9)
#        self.assertEqual(v.ntor.val, 5)
#        self.assertEqual(v.delt.val, 0.9)
#        self.assertEqual(v.tcon0.val, 2.0)
#        self.assertAlmostEqual(v.phiedge.val, 0.514386, places=13)
#        self.assertAlmostEqual(v.curtor.val, -1.7425E+05, places=13)
#        self.assertEqual(v.gamma.val, 0.0)
#        self.assertEqual(v.ncurr, 1)
#        self.assertFalse(v.free_boundary)
#        self.assertTrue(v.need_to_run_code)

if __name__ == "__main__":
    unittest.main()
