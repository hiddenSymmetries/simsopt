from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo import PSCgrid
from simsopt.field import BiotSavart, coils_via_symmetries, Current

class Testing(unittest.TestCase):

    def test_L(self):
        """
        Tests the inductance calculation for some limiting cases:
            1. Identical, coaxial coils 
            (solution in Jacksons textbook, problem 5.28)
        and tests that the inductance and flux calculations agree,
        when we use the "TF field" as one of the coil B fields
        and compute the flux through the other coil, for coils with random 
        separations, random orientations, etc.
        """
        
        from scipy.special import ellipk, ellipe, jv
        from scipy.integrate import quad
        
        # initialize coaxial coils
        R0 = 4
        R = 1 
        a = 1e-5
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
        print(L_mutual_analytic, L[0, 1], L[1, 0])
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10))
        
        # Another simple test of the analytic formula   
        # Formulas only valid for R/a >> 1 so otherwise it will fail
        R = 1
        a = 1e-6
        R0 = 1
        mu0 = 4 * np.pi * 1e-7
        points = np.array([[0, 0, 0], [0, 0, R0]])
        alphas = np.zeros(2)
        deltas = np.zeros(2)
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas,
        )
        L = psc_array.L
        k = np.sqrt(4.0 * R ** 2 / (4.0 * R ** 2 + R0 ** 2))
        L_mutual_analytic = mu0 * R * (
            (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        )
        L_self_analytic = mu0 * R * (np.log(8.0 * R / a) - 2.0)
        assert(np.allclose(L_self_analytic, np.diag(L)))
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10))
        I = 1e10
        coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
        bs = BiotSavart(coils)
        center = np.array([0, 0, 0]).reshape(1, 3)
        bs.set_points(center)
        B_center = bs.B()
        Bz_center = B_center[:, 2]
        kwargs = {"B_TF": bs}
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
        )
        # print((psc_array.psi[0] / I)/ L[1, 0], psc_array.psi[0] / I, L[1, 0])
        assert(np.isclose(psc_array.psi[0] / I, L[1, 0]))
        # Only true because R << 1
        # assert(np.isclose(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center))
        
        # Test that inductance and flux calculations for wide set of
        # scenarios
        a = 1e-4
        R0 = 10
        for R in [0.1, 1]:
            for points in [np.array([[0, 0, 0], [0, 0, R0]]), 
                           np.array([[0, 0, 0], [0, R0, 0]]), 
                           np.array([[0, 0, 0], [R0, 0, 0]]),
                           np.array([[R0, 0, 0], [0, 0, 0]]), 
                           np.array([[0, R0, 0], [0, 0, 0]]), 
                           np.array([[0, 0, R0], [0, 0, 0]]),
                           np.array([[0, 0, 0], (np.random.rand(3) - 0.5) * 40]),
                           np.array([(np.random.rand(3) - 0.5) * 40, [0, 0, 0]]),
                           np.array([(np.random.rand(3) - 0.5) * 40, 
                                     (np.random.rand(3) - 0.5) * 40])]:
                for alphas in [np.zeros(2), np.random.rand(2) * 2 * np.pi]:
                    for deltas in [np.zeros(2), np.random.rand(2) * 2 * np.pi]:
                        psc_array = PSCgrid.geo_setup_manual(
                            points, R=R, a=a, alphas=alphas, deltas=deltas,
                        )
                        coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
                        bs = BiotSavart(coils)
                        kwargs = {"B_TF": bs}
                        psc_array = PSCgrid.geo_setup_manual(
                            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                        )
                        L = psc_array.L
                        print(psc_array.psi[0] / I * 1e10, L[1, 0] * 1e10)
                        assert(np.isclose(psc_array.psi[0] / I * 1e10, L[1, 0] * 1e10))

if __name__ == "__main__":
    unittest.main()
