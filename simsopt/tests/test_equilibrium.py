import unittest
import numpy as np
from simsopt.equilibrium import *

class EquilibriumTests(unittest.TestCase):
    def test_init(self):
        """
        Just create an Equilibrium instance using the standard constructor,
        and make sure we can read all the attributes.
        """
        e = Equilibrium()
        self.assertEqual(e.nfp.val, 1)
        self.assertTrue(e.stelsym.val)
        self.assertEqual(e.nfp.val, e.boundary.nfp.val)
        self.assertEqual(e.stelsym.val, e.boundary.stelsym.val)

    def test_repr(self):
        """
        Test that objects are printed in the expected way.
        """
        e = Equilibrium()
        self.assertEqual(e.__repr__(), \
                             "Equilibrium instance (nfp=1 stelsym=True)")


if __name__ == "__main__":
    unittest.main()
