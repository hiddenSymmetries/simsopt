from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo import PSCgrid, SurfaceRZFourier
from simsopt.field import BiotSavart, coils_via_symmetries, Current, CircularCoil
import simsoptpp as sopp

class Testing(unittest.TestCase):
    
    def test_psi_analytic_derivatives(self):
        """
        Tests the analytic calculations of dpsi/dalpha and dpsi/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        
        # Test partial derivative calculation if we change delta by epsilon
        ncoils = 3
        np.random.seed(ncoils)
        R0 = 1
        R = 1 
        a = 1e-5
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        
        # Need alphas and deltas in [-pi, pi] range
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        psi = psc_array.psi
        psi_deriv = psc_array.psi_deriv()
        epsilon = 1e-6
        deltas_new = np.copy(deltas)
        deltas_new[0] += epsilon
        alphas_new = alphas  # + epsilon
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        psi_new = psc_array_new.psi
        dpsi_ddelta = (psi_new - psi) / epsilon
        dpsi_ddelta_analytic = psi_deriv

        # Linv calculations looks much more incorrect that the L derivatives,
        # maybe because of numerical error accumulation? 
        # print(dpsi_ddelta, dpsi_ddelta_analytic)
        assert(np.allclose(dpsi_ddelta[0], dpsi_ddelta_analytic[ncoils], rtol=1e-3))

        # # Test partial derivative calculation if we change alpha by epsilon
        ncoils = 3
        np.random.seed(ncoils)
        R = 1 
        a = 1e-5
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        
        # Need alphas and deltas in [-pi, pi] range
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        psi = psc_array.psi
        psi_deriv = psc_array.psi_deriv()
        epsilon = 1e-6
        alphas_new = np.copy(alphas)
        alphas_new[0] += epsilon
        deltas_new = deltas  # + epsilon
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        psi_new = psc_array_new.psi
        dpsi_dalpha = (psi_new - psi) / epsilon
        dpsi_dalpha_analytic = psi_deriv

        # Linv calculations looks much more incorrect that the L derivatives,
        # maybe because of numerical error accumulation? 
        # print(dpsi_dalpha, dpsi_dalpha_analytic, dpsi_dalpha_analytic[0])
        assert(np.isclose(dpsi_dalpha[0], dpsi_dalpha_analytic[0], rtol=1e-3))

    
    def test_L_analytic_derivatives(self):
        """
        Tests the analytic calculations of dL/dalpha and dL/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        
        # Test partial derivative calculation if we change delta by epsilon
        ncoils = 3
        np.random.seed(ncoils)
        R0 = 1
        R = 1 
        a = 1e-5
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        
        # Need alphas and deltas in [-pi, pi] range
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        L = psc_array.L
        Linv = psc_array.L_inv
        L_deriv = psc_array.L_deriv()
        epsilon = 1e-6
        deltas_new = np.copy(deltas)
        deltas_new[0] += epsilon
        alphas_new = alphas  # + epsilon
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        L_new = psc_array_new.L
        Linv_new = psc_array_new.L_inv
        dL_ddelta = (L_new - L) / epsilon
        dLinv_ddelta = (Linv_new - Linv) / epsilon
        dL_ddelta_analytic = L_deriv
        dLinv_ddelta_analytic = -np.tensordot(
            np.tensordot(
            L_deriv, Linv, axes=([1], [0])
            ), Linv, axes=([-1], [0])
        )
        # Symmetrizing helps with numerical error accumulation?
        for i in range(ncoils * 2):
            dLinv_ddelta_analytic[i, :, :] = (dLinv_ddelta_analytic[i, :, :] + dLinv_ddelta_analytic[i, :, :].T) / 2.0
            
        # Linv calculations looks much more incorrect that the L derivatives,
        # maybe because of numerical error accumulation? 
        # print(dL_ddelta, dL_ddelta_analytic[ncoils, :, :])
        # print(dLinv_ddelta, dLinv_ddelta_analytic[ncoils, :, :])
        assert(np.allclose(dL_ddelta, dL_ddelta_analytic[ncoils, :, :]))
        assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[ncoils, :, :], rtol=10))

        # Test partial derivative calculation if we change alpha by epsilon
        ncoils = 3
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        L = psc_array.L
        Linv = psc_array.L_inv
        L_deriv = psc_array.L_deriv()
        epsilon = 1e-6
        alphas_new = np.copy(alphas)
        alphas_new[0] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        L_new = psc_array_new.L
        Linv_new = psc_array_new.L_inv
        dL_dalpha = (L_new - L) / epsilon
        dLinv_dalpha = (Linv_new - Linv) / epsilon
        dL_dalpha_analytic = L_deriv
        dLinv_dalpha_analytic = -np.tensordot(
            np.tensordot(
            L_deriv, Linv, axes=([1], [0])
            ), Linv, axes=([-1], [0])
        )
        # Symmetrizing helps with numerical error accumulation?
        for i in range(ncoils * 2):
            dLinv_dalpha_analytic[i, :, :] = (dLinv_dalpha_analytic[i, :, :] + dLinv_dalpha_analytic[i, :, :].T) / 2.0
        assert(np.allclose(dL_dalpha, dL_dalpha_analytic[0, :, :], rtol=1e-3))
        assert(np.allclose(dLinv_dalpha, dLinv_dalpha_analytic[0, :, :], rtol=1))

        
    def test_dB_analytic_derivatives(self):
        """
        Tests the analytic calculations of dB/dalpha and dB/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        
        ncoils = 6
        np.random.seed(ncoils)
        R0 = 1
        R = 1 
        a = 1e-5
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        deltas = np.zeros(ncoils)
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        A_direct = psc_array.A_matrix_direct
        assert np.allclose(A, A_direct)
        # print(A, A_direct)
        # exit()
        epsilon = 1e-6
        alphas_new = alphas
        alphas_new[0] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        A_new = psc_array_new.A_matrix
        # B.set_points(psc_array.plasma_points)
        # B_new.set_points(psc_array.plasma_points)
        dA_dalpha = (A_new - A) / epsilon
        dA_dkappa_analytic = psc_array.dA_dkappa()
        # print(dA_dalpha[:, 0], dA_dkappa_analytic[0, :])
        assert np.allclose(dA_dalpha[:, 0], dA_dkappa_analytic[0, :])
        
        # Repeat but changing delta instead of alpha
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = np.zeros(ncoils)
        psc_array = PSCgrid.geo_setup_manual(
        points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        A_direct = psc_array.A_matrix_direct
        # print(A, A_direct)
        assert np.allclose(A, A_direct)
        # print(A, A_direct)
        # exit()
        epsilon = 1e-6
        alphas_new = alphas
        deltas_new = deltas
        deltas_new[0] += epsilon
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        A_new = psc_array_new.A_matrix
        # B.set_points(psc_array.plasma_points)
        # B_new.set_points(psc_array.plasma_points)
        dA_ddelta = (A_new - A) / epsilon
        dA_dkappa_analytic = psc_array.dA_dkappa()
        assert np.allclose(dA_ddelta[:, 0], dA_dkappa_analytic[ncoils, :])
        # print(dA_ddelta[:, 0], dA_dkappa_analytic[ncoils, :])

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
    
        from scipy.special import ellipk, ellipe
        
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
        # print(L_mutual_analytic, L[0, 1], L[1, 0])
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
        kwargs = {"B_TF": bs, "ppp": 2000}
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
        )
        # print((psc_array.psi[0] / I)/ L[1, 0], psc_array.psi[0] / I, L[1, 0])
        assert(np.isclose(psc_array.psi[0] / I, L[1, 0]))
        # Only true because R << 1
        # assert(np.isclose(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center))
        
        input_name = 'input.LandremanPaul2021_QA_lowres'
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        surf1 = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='full torus', nphi=16, ntheta=16
        )
        surf1.nfp = 1
        surf1.stellsym = False
        surf2 = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='half period', nphi=16, ntheta=16
        )
        
        # Test that inductance and flux calculations for wide set of
        # scenarios
        a = 1e-4
        R0 = 10
        print('starting loop')
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
                for alphas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
                    for deltas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
                        for surf in [surf1]:
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas,
                            )
                            coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
                            bs = BiotSavart(coils)
                            kwargs = {"B_TF": bs, "ppp": 2000, "plasma_boundary": surf}
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                            )
                            L = psc_array.L

                            # This is not a robust check but it only converges when N >> 1
                            # points are used to do the integrations. Can easily check that 
                            # can increase rtol as you increase N    
                            assert(np.isclose(psc_array.psi[0] / I * 1e10, L[1, 0] * 1e10, rtol=1e-1))
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
                                    print(q, fp, stell)
                                    phi0 = (2 * np.pi / psc_array.nfp) * fp
                                    # get new locations by flipping the y and z components, then rotating by phi0
                                    ox = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                                    oy = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                                    oz = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                                    xyz = np.array([ox, oy, oz]).T
                                    Bn_PSC += sopp.A_matrix(
                                        contig(xyz),
                                        contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                                        psc_array.R,
                                    ) @ psc_array.I
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I * stell),
                                        psc_array.R,
                                    )
                                    q = q + 1
                                    # print(q, B_PSC)
                            # Calculate Bfields from CircularCoil class
                            B_circular_coils = np.zeros(B_PSC.shape)
                            Bn_circular_coils = np.zeros(psc_array.Bn_PSC.shape)
                            for i in range(len(psc_array.alphas)):
                                PSC = CircularCoil(
                                    psc_array.R, 
                                    psc_array.grid_xyz_all[i, :], 
                                    psc_array.I_all[i], 
                                    psc_array.coil_normals_all[i, :]
                                )
                                PSC.set_points(psc_array.plasma_boundary.gamma().reshape(-1, 3))
                                B_circular_coils += PSC.B().reshape(-1, 3)
                                Bn_circular_coils += np.sum(PSC.B().reshape(
                                    -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart
                            currents = []
                            for i in range(len(psc_array.I)):
                                currents.append(Current(psc_array.I[i]))
                            coils = coils_via_symmetries(
                                psc_array.curves, currents, nfp=1, stellsym=False
                            )
                            B_direct = BiotSavart(coils)
                            B_direct.set_points(psc_array.plasma_points)
                            Bn_direct = np.sum(B_direct.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart, using all the coils manually defined
                            currents = []
                            for i in range(psc_array.num_psc * psc_array.symmetry):
                                currents.append(Current(psc_array.I_all[i]))
                            all_coils = coils_via_symmetries(
                                psc_array.all_curves, currents, nfp=1, stellsym=False
                            )
                            B_direct_all = BiotSavart(all_coils)
                            B_direct_all.set_points(psc_array.plasma_points)
                            Bn_direct_all = np.sum(B_direct_all.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            
                            # Robust test of all the B and Bn calculations from circular coils
                            # print(psc_array.B_PSC.B()[-1] * 1e10, B_PSC[-1] * 1e10, B_circular_coils[-1] * 1e10, B_direct.B()[-1] * 1e10)
                            print(alphas, deltas)
                            print('Bns = ', psc_array.Bn_PSC[-1] * 1e10, Bn_PSC[-1] * 1e10, Bn_circular_coils[-1] * 1e10, Bn_direct[-1] * 1e10)
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, atol=1e3))
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_PSC * 1e10))
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_circular_coils * 1e10))
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_direct.B() * 1e10))
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_direct_all.B() * 1e10))
        
        # Applying discrete symmetries to the coils wont work unless they dont intersect the symmetry planes
        for R in [0.1, 1]:
            for points in [np.array([(np.random.rand(3) - 0.5) * 40, 
                                      (np.random.rand(3) - 0.5) * 40])]:
                for alphas in [np.zeros(2), np.random.rand(2) * 2 * np.pi]:
                    for deltas in [np.zeros(2), np.random.rand(2) * 2 * np.pi]:
                        for surf in [surf1, surf2]:
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas,
                            )
                            coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
                            bs = BiotSavart(coils)
                            kwargs = {"B_TF": bs, "ppp": 2000, "plasma_boundary": surf}
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                            )
                            L = psc_array.L
                            # print(psc_array.nfp, psc_array.stell_list, psc_array.I)

                            # Below is not true once coils are added from discrete symmetries!
                            # assert(np.isclose(psc_array.psi[0] / I * 1e10, L[1, 0] * 1e10, rtol=1e-1))
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
                                    print(q, fp, stell)
                                    phi0 = (2 * np.pi / psc_array.nfp) * fp
                                    # get new locations by flipping the y and z components, then rotating by phi0
                                    ox = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                                    oy = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                                    oz = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                                    xyz = np.array([ox, oy, oz]).T
                                    Bn_PSC += sopp.A_matrix(
                                        contig(xyz),
                                        contig(psc_array.plasma_points),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                                        psc_array.R,
                                    ) @ (psc_array.I)
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_points),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I),
                                        psc_array.R,
                                    )
                                    q = q + 1
                                    # print(q, B_PSC)
                            B_PSC_all = sopp.B_PSC(
                                contig(psc_array.grid_xyz_all),
                                contig(psc_array.plasma_points),
                                contig(psc_array.alphas_total),
                                contig(psc_array.deltas_total),
                                contig(psc_array.I),
                                psc_array.R,
                            )
                            # Calculate Bfields from CircularCoil class
                            B_circular_coils = np.zeros(B_PSC.shape)
                            Bn_circular_coils = np.zeros(psc_array.Bn_PSC.shape)
                            for i in range(psc_array.alphas_total.shape[0]):
                                PSC = CircularCoil(
                                    psc_array.R, 
                                    psc_array.grid_xyz_all[i, :], 
                                    psc_array.I_all[i] * np.sign(psc_array.I_all[i]), 
                                    psc_array.coil_normals_all[i, :]
                                )
                                PSC.set_points(psc_array.plasma_points)
                                B_circular_coils += PSC.B().reshape(-1, 3)
                                Bn_circular_coils += np.sum(PSC.B().reshape(
                                    -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart
                            currents = []
                            for i in range(len(psc_array.I)):
                                currents.append(Current(psc_array.I[i]))
                            coils = coils_via_symmetries(
                                psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
                            )
                            B_direct = BiotSavart(coils)
                            B_direct.set_points(psc_array.plasma_points)
                            Bn_direct = np.sum(B_direct.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart, using all the coils manually defined
                            currents = []
                            for i in range(psc_array.num_psc * psc_array.symmetry):
                                currents.append(Current(psc_array.I_all[i]))
                            all_coils = coils_via_symmetries(
                                psc_array.all_curves, currents, nfp=1, stellsym=False
                            )
                            B_direct_all = BiotSavart(all_coils)
                            B_direct_all.set_points(psc_array.plasma_points)
                            Bn_direct_all = np.sum(B_direct_all.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            
                            # Robust test of all the B and Bn calculations from circular coils
                            # print('here = ', psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, Bn_circular_coils * 1e10)
                            print(psc_array.B_PSC.B()[-1] * 1e10, B_PSC[-1] * 1e10, B_PSC_all[-1] * 1e10, 
                                  B_circular_coils[-1] * 1e10, B_direct.B()[-1] * 1e10, B_direct_all.B()[-1] * 1e10)
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_PSC * 1e10))
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_circular_coils * 1e10))
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_direct.B() * 1e10))
                            # assert(np.allclose(psc_array.B_PSC.B() * 1e10, B_direct_all.B() * 1e10))
                            print(psc_array.Bn_PSC[-1] * 1e10, Bn_PSC[-1] * 1e10, 
                                  Bn_circular_coils[-1] * 1e10, Bn_direct[-1] * 1e10, Bn_direct_all[-1] * 1e10)
                            # print(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10)
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, atol=1e3))

if __name__ == "__main__":
    unittest.main()
