from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo import PSCgrid

class Testing(unittest.TestCase):

    def test_L(self):
        """
        Tests the inductance calculation for some limiting cases:
            1. Identical, coaxial coils 
            (solution in Jacksons textbook, problem 5.28)
            2. Identical coils separated in the xy-plane by distance R0
            (solution in Jacksons textbook, problem 5.34a)
        """
        
        from scipy.special import ellipk, ellipe, jv
        # initialize coaxial coils
        R0 = 1
        R = 1 
        a = 0.01
        mu0 = 4 * np.pi * 1e-7
        points = np.array([[0, 0, 0], [0, 0, R0]])
        alphas = np.zeros(2)
        deltas = np.zeros(2)
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        L = psc_array.L
        L_self_analytic = mu0 * R * (np.log(8.0 * R / a) - 2.0)
        
        # Jackson problem 5.28
        k = np.sqrt(4.0 * R ** 2 / (4.0 * R ** 2 + R0 ** 2))
        # Jackson uses K(k) and E(k) but this corresponds to
        # K(k^2) and E(k^2) in scipy library
        L_mutual_analytic = mu0 * R * (
            (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        )
        assert(np.allclose(L_self_analytic, np.diag(L)))
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10))
        
        # Jackson problem 5.34a
        points = np.array([[0, 0, 0], [0, R0, 0]])
        # r_coord = np.linspace(0, 100, 100)
        # k = np.sqrt(4 * R * r_coord / (R ** 2 + r_coord ** 2 + 2 * R * r_coord))
        k = np.linspace(0, 100, 1000)
        dk = k[1] - k[0]
        # Jackson uses K(k) and E(k) but this corresponds to
        # K(k^2) and E(k^2) in scipy library
        L_mutual_analytic = mu0 * np.pi * R ** 2 * np.sum(np.exp(-1.0 * k * R0) * jv(1, k * R) ** 2) * dk
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10))

if __name__ == "__main__":
    unittest.main()
