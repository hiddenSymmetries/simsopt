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
            2. Identical coils separated in the xy-plane by distance R0
            (solution in Jacksons textbook, problem 5.34a)
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
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10))
        
        # Jackson problem 5.34a
        points = np.array([[0, 0, 0], [0, R0, 0]])
        # r_coord = np.linspace(0, 100, 100)
        # k = np.sqrt(4 * R * r_coord / (R ** 2 + r_coord ** 2 + 2 * R * r_coord))
        # k = np.linspace(0, 1000, 1000)
        # dk = k[1] - k[0]
        # Jackson uses K(k) and E(k) but this corresponds to
        # K(k^2) and E(k^2) in scipy library
        def analytic_soln(kk):
            return np.exp(-1.0 * kk * R0) * jv(1, kk * R) ** 2
            
        L_mutual_analytic, err = quad(analytic_soln, 0, 1e2)
        L_mutual_analytic = L_mutual_analytic * mu0 * np.pi * R ** 2
        print(L[0, 1])
        # print(L_mutual_analytic * 1e10, L[0, 1] * 1e10)
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10))
        
        # Test inductances and fluxes in simple cases like far apart coaxial coils        
        # Formulas only valid for R/a >> 1 so otherwise it will fail
        R = 0.001
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
        # L corrections
        # dwM_dx2 = mu0 * np.pi * k ** 3 / R * (ellipk(k ** 2) - (1 - 2 * k ** 2) / (1 - k ** 2) * ellipe(k ** 2))
        # dwM_dR2 = mu0 * np.pi * k / R * ((2 - k ** 2) * ellipk(k ** 2) - (2 - k ** 2 * (1 - 2 * k ** 2) / (1 - k ** 2) * ellipe(k ** 2)))
        # M_correction = psc_array.a ** 2 / 12.0 * (dwM_dx2 + dwM_dR2)
        # print(M_correction, L[1, 0] + M_correction)
        # print(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center)
        # print(psc_array.psi[0] / I, L[1, 0])
        assert(np.isclose(psc_array.psi[0] / I, L[1, 0]))
        assert(np.isclose(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center))
        
        # Test inductances and fluxes in simple cases like far apart coaxial coils        
        # Formulas only valid for R/a >> 1 so otherwise it will fail
        R = 1
        a = 1e-4
        R0 = 10
        points = np.array([[0, 0, 0], [0, 0, R0]])
        alphas = np.zeros(2)
        deltas = np.zeros(2)
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas,
        )
        L = psc_array.L
        L_self_analytic = mu0 * R * (np.log(8.0 * R / a) - 2.0)
        k = np.sqrt(4.0 * R ** 2 / (4.0 * R ** 2 + R0 ** 2))
        L_mutual_analytic = mu0 * R * (
            (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        )
        assert(np.allclose(L_self_analytic, np.diag(L)))
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10))
        coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
        bs = BiotSavart(coils)
        bs.set_points(center)
        B_center = bs.B()
        Bz_center = B_center[:, 2]
        kwargs = {"B_TF": bs}
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
        )
        print(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center)
        print(psc_array.psi[0] / I, L[1, 0])
        assert(np.isclose(psc_array.psi[0] / I, L[1, 0]))
        
        # Doesn't work unless coil major radius is small (so that we
        # can approximate the Bfield through the coil as just the Bfield
        # at the center)
        # assert(np.isclose(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center))
        
        # Test inductances and fluxes calculations give same results for very
        # general configurations
        R0 = 4
        R = 1
        a = 1e-5
        points = np.array([[0, 0, 0], [0, 0, R0]])
        # points = (np.random.rand(2, 3) - 0.5) * 40
        # print(points)
        alphas = [np.pi / 2.0, np.pi / 2.0]
        deltas = np.zeros(2)  # np.random.rand(2)
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas,
        )
        L = psc_array.L
        coils = coils_via_symmetries([psc_array.curves[1]], [Current(-I)], nfp=1, stellsym=False)
        bs = BiotSavart(coils)
        kwargs = {"B_TF": bs}
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
        )
        L = psc_array.L
        print(psc_array.psi[0] / I, L[1, 0])
        assert(np.isclose(psc_array.psi[0] / I * 1e10, L[1, 0] * 1e10))

if __name__ == "__main__":
    unittest.main()
