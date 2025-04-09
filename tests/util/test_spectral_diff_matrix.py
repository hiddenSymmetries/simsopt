#!/usr/bin/env python3

import unittest
import numpy as np
from simsopt.util.spectral_diff_matrix import spectral_diff_matrix

class Tests(unittest.TestCase):

    def test_Fourier(self):
        """
        Confirm that the derivative of a sin/cos is computed exactly.
        """
        xmin = -3.4
        xmax = -0.7
        L = xmax - xmin
        for nphi in range(11, 21, 2):
            D = spectral_diff_matrix(nphi, xmin=xmin, xmax=xmax)
            for n in range(0, int(np.floor(nphi/2)) - 1):
                for phase in [0, 0.3]:
                    phi = np.linspace(xmin, xmax, nphi, endpoint=False)
                    x = np.sin(n * phi * 2 * np.pi / L + phase)
                    dx = (n * 2 * np.pi / L) * np.cos(n * phi * 2 * np.pi / L + phase)
                    np.testing.assert_allclose(np.matmul(D, x), dx,
                                               rtol = 1e-12, atol = 1e-12)

    def test_random(self):
        """
        Do a bunch of tests for random xmin and xmax.
        """
        for n in range(1, 10):
            for whichrand in range(10):
                xmin = np.random.rand() * 20 - 10
                xmax = xmin + np.random.rand() * 20
                D = spectral_diff_matrix(n, xmin=xmin, xmax=xmax)
                # Check that the diagonal entries are all 0:
                for j in range(n):
                    self.assertLess(np.abs(D[j, j]), 1e-14)
                # Check that the row and column sums are all 0
                self.assertLess(np.max(np.abs(np.sum(D, axis=0))), 1e-11)
                self.assertLess(np.max(np.abs(np.sum(D, axis=1))), 1e-11)

    def test_n2(self):
        """
        Test matrix for n = 2
        """
        D1 = np.array([[0.0, 0.0], \
                       [0.0, 0.0]])
        D2 = spectral_diff_matrix(2)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)
        
    def test_n3(self):
        """
        Test matrix for n = 3
        """
        x = 0.577350269189626
        D1 = np.array([[0, x, -x], \
                      [-x, 0, x], \
                      [x, -x, 0]])
        D2 = spectral_diff_matrix(3)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)
        
    def test_n4(self):
        """
        Test matrix for n = 4
        """
        D1 = np.array([[0, 0.5, 0, -0.5], \
                       [-0.5, 0, 0.5, 0], \
                       [0, -0.5, 0, 0.5], \
                       [0.5, 0,-0.5, 0]])
        D2 = spectral_diff_matrix(4)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)
        
    def test_n5(self):
        """
        Test matrix for n = 5
        """
        e = 0.85065080835204
        f = 0.525731112119134
        D1 = np.array([[0, e, -f, f, -e], \
                       [-e, 0, e, -f, f], \
                       [f, -e, 0, e, -f], \
                       [-f, f, -e, 0, e], \
                       [e, -f, f, -e, 0]])
        D2 = spectral_diff_matrix(5)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)

    def test_n2_shifted(self):
        """
        Test matrix for n = 2, with xmin and xmax set.
        """
        D1 = np.array([[0.0, 0.0], \
                       [0.0, 0.0]])
        D2 = spectral_diff_matrix(2, xmin=-2.1, xmax=3.7)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)
        
    def test_n3_shifted(self):
        """
        Test matrix for n = 3, with xmin and xmax set.
        """
        x = 0.625448056632489
        D1 = np.array([[0, x, -x], \
                      [-x, 0, x], \
                      [x, -x, 0]])
        D2 = spectral_diff_matrix(3, xmin=-2.1, xmax=3.7)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)
        
    def test_n4_shifted(self):
        """
        Test matrix for n = 4, with xmin and xmax set.
        """
        x = 0.541653905791344
        D1 = np.array([[0, x, 0, -x], \
                       [-x, 0, x, 0], \
                       [0, -x, 0, x], \
                       [x, 0, -x, 0]])
        D2 = spectral_diff_matrix(4, xmin=-2.1, xmax=3.7)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)
        
    def test_n5_shifted(self):
        """
        Test matrix for n = 5, with xmin and xmax set.
        """
        e = 0.921516665616892
        f = 0.569528620550711
        D1 = np.array([[0, e, -f, f, -e], \
                       [-e, 0, e, -f, f], \
                       [f, -e, 0, e, -f], \
                       [-f, f, -e, 0, e], \
                       [e, -f, f, -e, 0]])
        D2 = spectral_diff_matrix(5, xmin=-2.1, xmax=3.7)
        np.testing.assert_allclose(D1, D2, rtol=1e-13, atol=1e-13)
                
if __name__ == "__main__":
    unittest.main()
