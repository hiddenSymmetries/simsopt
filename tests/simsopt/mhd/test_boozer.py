import unittest
import numpy as np
import os
import logging
from simsopt.mhd.boozer import Quasisymmetry
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

class MockBoozer():
    """
    This class exists only for testing the Quasisymmetry class.  It
    returns similar data to the real Boozer class, but without doing a
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
        self.xm = xm
        self.xn = xn * nfp
        self.mnmax = mnmax
        self.nfp = nfp

    def bmnc(self, s):
        return np.arange(1.0, self.mnmax + 1)

class QuasisymmetryTests(unittest.TestCase):
    def test_quasisymmetry_residuals(self):
        """
        Verify that the correct residual vectors are returned for QA, QP, and QH
        """
        b = MockBoozer(3, 2, 4)
        # xm:   [  0  0  0  1  1  1  1  1  2   2   2   2   2   3   3   3   3   3]
        # xn:   [  0  4  8 -8 -4  0  4  8 -8  -4   0   4   8  -8  -4   0   4   8]
        # bmnc: [ 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
        s = 1
        
        # QA
        q = Quasisymmetry(b, s, 1, 0, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18])

        # QP
        q = Quasisymmetry(b, s, 0, 1, "B00", "even")
        np.testing.assert_allclose(q.J(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # QH
        q = Quasisymmetry(b, s, 1, 1, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # QH, opposite "chirality"
        q = Quasisymmetry(b, s, 1, -1, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18])

if __name__ == "__main__":
    unittest.main()
