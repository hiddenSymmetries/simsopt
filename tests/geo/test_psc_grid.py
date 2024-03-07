from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo import PSCgrid
from simsopt.field import BiotSavart, coils_via_symmetries, Current, CircularCoil
import simsoptpp as sopp

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
        
        np.random.seed(1)
        
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
        I = 1e3
        coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
        bs = BiotSavart(coils)
        center = np.array([0, 0, 0]).reshape(1, 3)
        bs.set_points(center)
        B_center = bs.B()
        Bz_center = B_center[:, 2]
        kwargs = {"B_TF": bs, "ppp": 2000}
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
                        kwargs = {"B_TF": bs, "ppp": 2000}
                        psc_array = PSCgrid.geo_setup_manual(
                            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                        )
                        L = psc_array.L
                        
                        # This is not a robust check but it only converges when N >> 1
                        # points are used to do the integrations. Can easily check that 
                        # can increase rtol as you increase N
                        # print(psc_array.psi[0] / I * 1e10, L[1, 0] * 1e10)
                        assert(np.isclose(psc_array.psi[0] / I * 1e10, L[1, 0] * 1e10, rtol=1e-1))
                        
                        psc_array.setup_psc_biotsavart()
                        contig = np.ascontiguousarray
                        B_PSC = sopp.B_PSC(
                            contig(psc_array.grid_xyz),
                            contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                            contig(psc_array.alphas),
                            contig(psc_array.deltas),
                            contig(psc_array.I),
                            psc_array.R
                        )
                        Bn_PSC = sopp.Bn_PSC(
                            contig(psc_array.grid_xyz),
                            contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                            contig(psc_array.alphas),
                            contig(psc_array.deltas),
                            contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                            psc_array.R
                        ) @ psc_array.I
                        B_circular_coils = np.zeros(B_PSC.shape)
                        Bn_circular_coils = np.zeros(psc_array.Bn_PSC.shape)
                        for i in range(len(psc_array.alphas)):
                            PSC = CircularCoil(
                                psc_array.R, 
                                psc_array.grid_xyz[i, :], 
                                psc_array.I[i], 
                                psc_array.coil_normals[i, :]
                            )
                            PSC.set_points(psc_array.plasma_boundary.gamma().reshape(-1, 3))
                            B_circular_coils += PSC.B().reshape(-1, 3)
                            Bn_circular_coils += np.sum(PSC.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                        
                        # Robust test of all the B and Bn calculations from circular coils
                        assert(np.allclose(psc_array.B_PSC.B(), B_PSC))
                        assert(np.allclose(psc_array.B_PSC.B(), B_circular_coils))
                        assert(np.allclose(psc_array.Bn_PSC, Bn_PSC))
                        assert(np.allclose(psc_array.Bn_PSC, Bn_circular_coils))

if __name__ == "__main__":
    unittest.main()
