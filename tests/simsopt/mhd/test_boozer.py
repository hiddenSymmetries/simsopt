import unittest
import numpy as np
import os
import logging
from simsopt.mhd.boozer import Quasisymmetry, booz_xform_found
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

class MockBoozXform():
    """
    This class exists only for testing the Quasisymmetry class.  It
    returns similar data to the real Booz_xform class, but without doing a
    real calculation.
    """
    def __init__(self, mpol, ntor, nfp):
        mnmax = (ntor * 2 + 1) * mpol + ntor + 1
        xm = np.zeros(mnmax)
        xn = np.zeros(mnmax)
        xn[:ntor + 1] = np.arange(ntor + 1)
        for m in range(1, mpol + 1):
            index = ntor + 1 + (ntor * 2 + 1) * (m - 1)
            xm[index : index + (ntor * 2 + 1)] = m
            xn[index : index + (ntor * 2 + 1)] = np.arange(-ntor, ntor + 1)
        self.xm_b = xm
        self.xn_b = xn * nfp
        self.mnmax_b = mnmax
        self.nfp = nfp
        arr1 = np.arange(1.0, mnmax + 1) * 10
        arr2 = arr1 + 1
        arr2[0] = 100
        self.bmnc_b = np.stack((arr1, arr2)).transpose()
        print('bmnc_b:')
        print(self.bmnc_b)
        print('booz_xform_found:', booz_xform_found)

class MockBoozer():
    """
    This class exists only for testing the Quasisymmetry class.  It
    returns similar data to the real Boozer class, but without doing a
    real calculation.
    """
    def __init__(self, mpol, ntor, nfp):
        self.bx = MockBoozXform(mpol, ntor, nfp)
        self.s_to_index = {0: 0, 1: 1}
        
    def register(self, s):
        pass

    def run(self):
        pass

class QuasisymmetryTests(unittest.TestCase):
    def test_quasisymmetry_residuals(self):
        """
        Verify that the correct residual vectors are returned for QA, QP, and QH
        """
        b = MockBoozer(3, 2, 4)
        # xm:    [  0  0  0  1  1  1  1  1  2   2   2   2   2   3   3   3   3   3]
        # xn:    [  0  4  8 -8 -4  0  4  8 -8  -4   0   4   8  -8  -4   0   4   8]
        # bmnc: [[ 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
        # bmnc:  [100 21 31 41 51 61 71 81 91 101 111 121 131 141 151 161 171 181]]
        
        # QA
        s = 0; q = Quasisymmetry(b, s, 1, 0, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18])
        s = 1; q = Quasisymmetry(b, s, 1, 0, "B00", "even")
        np.testing.assert_allclose(q.J(), [.21, .31, .41, .51, .71, .81, .91, 1.01, 1.21, 1.31, 1.41, 1.51, 1.71, 1.81])
        s = (0, 1); q = Quasisymmetry(b, s, 1, 0, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18,
                                           .21, .31, .41, .51, .71, .81, .91, 1.01, 1.21, 1.31, 1.41, 1.51, 1.71, 1.81])

        # QP
        s = 0
        q = Quasisymmetry(b, s, 0, 1, "B00", "even")
        np.testing.assert_allclose(q.J(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # QH
        q = Quasisymmetry(b, s, 1, 1, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # QH, opposite "chirality"
        q = Quasisymmetry(b, s, 1, -1, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18])

    #@unittest.skipIf(not booz_xform_found, "booz_xform python package not found")

if __name__ == "__main__":
    unittest.main()
