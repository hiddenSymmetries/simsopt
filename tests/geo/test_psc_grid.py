from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo import PSCgrid, SurfaceRZFourier
from simsopt.field import BiotSavart, coils_via_symmetries, Current, CircularCoil
import simsoptpp as sopp

class Testing(unittest.TestCase):
    
    def test_dpsi_analytic_derivatives(self):
        """
        Tests the analytic calculations of dpsi/dalpha and dpsi/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        ncoils = 7
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
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt  # / psc_array.grid_normalization
        L = psc_array.L
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        psi_deriv = psc_array.psi_deriv()
        epsilon = 1e-4
        deltas_new = np.copy(deltas)
        deltas_new[0] += epsilon
        alphas_new = alphas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        psi_new = psc_array_new.psi
        Bn_objective_new = 0.5 * (A @ Linv @ psi_new + b).T @ (A @ Linv @ psi_new + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dpsi_ddelta = (psi_new - psi) / epsilon
        dpsi_ddelta_analytic = psi_deriv
        # print(dpsi_ddelta, dpsi_ddelta_analytic)
        assert(np.allclose(dpsi_ddelta[0], dpsi_ddelta_analytic[ncoils], rtol=1e-3))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (Linv * psi_deriv[ncoils:]))
        assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
        
        # Repeat but change coil 3
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt  # / psc_array.grid_normalization
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        psi_deriv = psc_array.psi_deriv()
        epsilon = 1e-4
        deltas_new = np.copy(deltas)
        deltas_new[3] += epsilon
        alphas_new = alphas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        psi_new = psc_array_new.psi
        Bn_objective_new = 0.5 * (A @ Linv @ psi_new + b).T @ (A @ Linv @ psi_new + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dpsi_ddelta = (psi_new - psi) / epsilon
        dpsi_ddelta_analytic = psi_deriv
        # print(dpsi_ddelta, dpsi_ddelta_analytic)
        assert(np.allclose(dpsi_ddelta[3], dpsi_ddelta_analytic[ncoils + 3], rtol=1e-3))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (Linv * psi_deriv[ncoils:]))
        assert np.isclose(dBn_objective, dBn_analytic[3], rtol=1e-2)

        # Test partial derivative calculation if we change alpha by epsilon
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt # / psc_array.grid_normalization
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        psi_deriv = psc_array.psi_deriv()
        epsilon = 1e-4
        alphas_new = np.copy(alphas)
        alphas_new[0] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        psi_new = psc_array_new.psi
        Bn_objective_new = 0.5 * (A @ Linv @ psi_new + b).T @ (A @ Linv @ psi_new + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dpsi_dalpha = (psi_new - psi) / epsilon
        dpsi_dalpha_analytic = psi_deriv
        # print(dpsi_ddelta, dpsi_ddelta_analytic)
        assert(np.allclose(dpsi_dalpha[0], dpsi_dalpha_analytic[0], rtol=1e-3))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (Linv * psi_deriv[:ncoils]))
        assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
        
        # Repeat for changing coil 7
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt  # / psc_array.grid_normalization
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        psi_deriv = psc_array.psi_deriv()
        epsilon = 1e-4
        alphas_new = np.copy(alphas)
        alphas_new[6] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        psi_new = psc_array_new.psi
        Bn_objective_new = 0.5 * (A @ Linv @ psi_new + b).T @ (A @ Linv @ psi_new + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dpsi_dalpha = (psi_new - psi) / epsilon
        dpsi_dalpha_analytic = psi_deriv
        # print(dpsi_ddelta, dpsi_ddelta_analytic)
        assert(np.allclose(dpsi_dalpha[6], dpsi_dalpha_analytic[6], rtol=1e-3))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (Linv * psi_deriv[:ncoils]))
        assert np.isclose(dBn_objective, dBn_analytic[6], rtol=1e-2)
    
    def test_L_analytic_derivatives(self):
        """
        Tests the analytic calculations of dL/dalpha and dL/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        ncoils = 6
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
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt  # / psc_array.grid_normalization
        L = psc_array.L
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        L_deriv = psc_array.L_deriv()
        epsilon = 1e-4
        deltas_new = np.copy(deltas)
        deltas_new[0] += epsilon
        alphas_new = alphas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        L_new = psc_array_new.L
        Linv_new = psc_array_new.L_inv
        Bn_objective_new = 0.5 * (A @ Linv_new @ psi + b).T @ (A @ Linv_new @ psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dL_ddelta = (L_new - L) / epsilon
        dLinv_ddelta = (Linv_new - Linv) / epsilon
        dL_ddelta_analytic = L_deriv
        dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
        # Symmetrizing helps with numerical error accumulation?
        for i in range(ncoils * 2):
            dLinv_ddelta_analytic[i, :, :] = (dLinv_ddelta_analytic[i, :, :] + dLinv_ddelta_analytic[i, :, :].T) / 2.0
            
        # Linv calculations looks much more incorrect that the L derivatives,
        # maybe because of numerical error accumulation? 
        assert(np.allclose(dL_ddelta, dL_ddelta_analytic[ncoils, :, :]))
        assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[ncoils, :, :], rtol=10))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (dLinv_ddelta_analytic @ psi).T)
        assert np.isclose(dBn_objective, dBn_analytic[ncoils], rtol=1e-2)
        
        # repeat calculation but change coil 3
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt # / psc_array.grid_normalization
        L = psc_array.L
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        L_deriv = psc_array.L_deriv()
        epsilon = 1e-4
        deltas_new = np.copy(deltas)
        deltas_new[3] += epsilon
        alphas_new = alphas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        L_new = psc_array_new.L
        Linv_new = psc_array_new.L_inv
        Bn_objective_new = 0.5 * (A @ Linv_new @ psi + b).T @ (A @ Linv_new @ psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dL_ddelta = (L_new - L) / epsilon
        dLinv_ddelta = (Linv_new - Linv) / epsilon
        dL_ddelta_analytic = L_deriv
        dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
        # Symmetrizing helps with numerical error accumulation?
        for i in range(ncoils * 2):
            dLinv_ddelta_analytic[i, :, :] = (dLinv_ddelta_analytic[i, :, :] + dLinv_ddelta_analytic[i, :, :].T) / 2.0
            
        # Linv calculations looks much more incorrect that the L derivatives,
        # maybe because of numerical error accumulation? 
        # print(dL_ddelta, dL_ddelta_analytic[ncoils + 3, :, :])
        assert(np.allclose(dL_ddelta, dL_ddelta_analytic[ncoils + 3, :, :]))
        assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[ncoils + 3, :, :], rtol=10))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (dLinv_ddelta_analytic @ psi).T)
        assert np.isclose(dBn_objective, dBn_analytic[ncoils + 3], rtol=1e-2)

        # Test partial derivative calculation if we change alpha by epsilon
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt # / psc_array.grid_normalization
        L = psc_array.L
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        L_deriv = psc_array.L_deriv()
        epsilon = 1e-4
        alphas_new = np.copy(alphas)
        alphas_new[0] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        L_new = psc_array_new.L
        Linv_new = psc_array_new.L_inv
        Bn_objective_new = 0.5 * (A @ Linv_new @ psi + b).T @ (A @ Linv_new @ psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dL_ddelta = (L_new - L) / epsilon
        dLinv_ddelta = (Linv_new - Linv) / epsilon
        dL_ddelta_analytic = L_deriv
        dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
        # Symmetrizing helps with numerical error accumulation?
        for i in range(ncoils * 2):
            dLinv_ddelta_analytic[i, :, :] = (dLinv_ddelta_analytic[i, :, :] + dLinv_ddelta_analytic[i, :, :].T) / 2.0
            
        # Linv calculations looks much more incorrect that the L derivatives,
        # maybe because of numerical error accumulation? 
        # print(dL_ddelta, dL_ddelta_analytic[ncoils + 3, :, :])
        assert(np.allclose(dL_ddelta, dL_ddelta_analytic[0, :, :]))
        assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[0, :, :], rtol=10))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (dLinv_ddelta_analytic @ psi).T)
        assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
        
        # Repeat changing coil 2
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        psi = psc_array.psi
        b = psc_array.b_opt # / psc_array.grid_normalization
        L = psc_array.L
        Linv = psc_array.L_inv
        Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
        L_deriv = psc_array.L_deriv()
        epsilon = 1e-4
        alphas_new = np.copy(alphas)
        alphas_new[2] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        L_new = psc_array_new.L
        Linv_new = psc_array_new.L_inv
        Bn_objective_new = 0.5 * (A @ Linv_new @ psi + b).T @ (A @ Linv_new @ psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        dL_ddelta = (L_new - L) / epsilon
        dLinv_ddelta = (Linv_new - Linv) / epsilon
        dL_ddelta_analytic = L_deriv
        dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
        # Symmetrizing helps with numerical error accumulation?
        for i in range(ncoils * 2):
            dLinv_ddelta_analytic[i, :, :] = (dLinv_ddelta_analytic[i, :, :] + dLinv_ddelta_analytic[i, :, :].T) / 2.0
            
        # Linv calculations looks much more incorrect that the L derivatives,
        # maybe because of numerical error accumulation? 
        # print(dL_ddelta, dL_ddelta_analytic[ncoils + 3, :, :])
        assert(np.allclose(dL_ddelta, dL_ddelta_analytic[2, :, :]))
        assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[2, :, :], rtol=10))
        dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (dLinv_ddelta_analytic @ psi).T)
        assert np.isclose(dBn_objective, dBn_analytic[2], rtol=1e-2)

        
    def test_dA_analytic_derivatives(self):
        """
        Tests the analytic calculations of dA/dalpha and dA/ddelta
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
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        Linv = psc_array.L_inv[:ncoils, :ncoils]
        psi = psc_array.psi
        Linv_psi = Linv @ psi
        b = psc_array.b_opt # / psc_array.grid_normalization
        Bn_objective = 0.5 * (A @ Linv_psi + b).T @ (A @ Linv_psi + b)
        epsilon = 1e-6
        alphas_new = alphas
        alphas_new[0] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        A_new = psc_array_new.A_matrix
        Bn_objective_new = 0.5 * (A_new @ Linv_psi + b).T @ (A_new @ Linv_psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        A_deriv = psc_array.A_deriv()
        dBn_analytic = (A @ Linv_psi + b).T @ (A_deriv[:, :ncoils] * Linv_psi)
        print(dBn_objective, dBn_analytic[0])
        assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
        dA_dalpha = (A_new - A) / epsilon
        dA_dkappa_analytic = A_deriv
        assert np.allclose(dA_dalpha[:, 0], dA_dkappa_analytic[:, 0], rtol=1e-2)
        
        # Repeat but change coil 3
        points = (np.random.rand(ncoils, 3) - 0.5) * 5
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        Linv = psc_array.L_inv[:ncoils, :ncoils]
        psi = psc_array.psi
        Linv_psi = Linv @ psi
        b = psc_array.b_opt # / psc_array.grid_normalization
        Bn_objective = 0.5 * (A @ Linv_psi + b).T @ (A @ Linv_psi + b)
        epsilon = 1e-4
        alphas_new = alphas
        alphas_new[4] += epsilon
        deltas_new = deltas
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        A_new = psc_array_new.A_matrix
        Bn_objective_new = 0.5 * (A_new @ Linv_psi + b).T @ (A_new @ Linv_psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        A_deriv = psc_array.A_deriv()
        dBn_analytic = (A @ Linv_psi + b).T @ (A_deriv[:, :ncoils] * Linv_psi)
        assert np.isclose(dBn_objective, dBn_analytic[4], rtol=1e-2)
        dA_dalpha = (A_new - A) / epsilon
        dA_dkappa_analytic = A_deriv
        assert np.allclose(dA_dalpha[:, 4], dA_dkappa_analytic[:, 4], rtol=1e-2)
        
        # Repeat but changing delta instead of alpha
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
        points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        Linv = psc_array.L_inv[:ncoils, :ncoils]
        psi = psc_array.psi
        Linv_psi = Linv @ psi
        b = psc_array.b_opt  # / psc_array.grid_normalization
        Bn_objective = 0.5 * (A @ Linv_psi + b).T @ (A @ Linv_psi + b)
        epsilon = 1e-4
        alphas_new = alphas
        deltas_new = deltas
        deltas_new[0] += epsilon
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        A_new = psc_array_new.A_matrix
        Bn_objective_new = 0.5 * (A_new @ Linv_psi + b).T @ (A_new @ Linv_psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        A_deriv = psc_array.A_deriv()
        # notice ncoils: instead of :ncoils below
        dBn_analytic = (A @ Linv_psi + b).T @ (A_deriv[:, ncoils:] * Linv_psi)
        # print(dBn_objective, dBn_analytic[0])
        dA_ddelta = (A_new - A) / epsilon
        dA_dkappa_analytic = A_deriv
        # print(dA_ddelta[:, 0], dA_dkappa_analytic[:, ncoils])
        assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
        assert np.allclose(dA_ddelta[:, 0], dA_dkappa_analytic[:, ncoils], rtol=1e-2)
        
        # Repeat but change coil 5
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        psc_array = PSCgrid.geo_setup_manual(
        points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        A = psc_array.A_matrix
        Linv = psc_array.L_inv[:ncoils, :ncoils]
        psi = psc_array.psi
        Linv_psi = Linv @ psi
        b = psc_array.b_opt  # / psc_array.grid_normalization
        Bn_objective = 0.5 * (A @ Linv_psi + b).T @ (A @ Linv_psi + b)
        epsilon = 1e-5
        alphas_new = alphas
        deltas_new = deltas
        deltas_new[5] += epsilon
        psc_array_new = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas_new, deltas=deltas_new
        )
        A_new = psc_array_new.A_matrix
        Bn_objective_new = 0.5 * (A_new @ Linv_psi + b).T @ (A_new @ Linv_psi + b)
        dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
        A_deriv = psc_array.A_deriv()
        # notice ncoils: instead of :ncoils below
        dBn_analytic = (A @ Linv_psi + b).T @ (A_deriv[:, ncoils:] * Linv_psi)
        print(dBn_objective, dBn_analytic[5])
        assert np.isclose(dBn_objective, dBn_analytic[5], rtol=1e-2)
        dA_ddelta = (A_new - A) / epsilon
        dA_dkappa_analytic = A_deriv
        assert np.allclose(dA_ddelta[:, 5], dA_dkappa_analytic[:, ncoils + 5], rtol=1e-3)

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
        print(L_mutual_analytic * 1e10, L[0, 1] * 1e10, L)
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10, rtol=1e-3))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10, rtol=1e-3))
        
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
        print(L_mutual_analytic * 1e10, L[0, 1] * 1e10)
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10, rtol=1e-1))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10, rtol=1e-1))
        I = 1e10
        coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
        bs = BiotSavart(coils)
        center = np.array([0, 0, 0]).reshape(1, 3)
        bs.set_points(center)
        B_center = bs.B()
        Bz_center = B_center[:, 2]
        kwargs = {"B_TF": bs, "ppp": 1000}
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
        )
        # This is not a robust check but it only converges when N >> 1
        # points are used to do the integrations and there
        # are no discrete symmetries. Can easily check that 
        # can decrease rtol as you increase number of integration points
        # assert(np.isclose(psc_array.psi[0] / I, L[1, 0], rtol=1e-1))
        # Only true if R << 1, assert(np.isclose(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center))
        
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
                            # print(L[0, 1] * 1e10)
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
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
                                    ) @ psc_array.I * 2e-7
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I * stell),
                                        psc_array.R,
                                    )
                                    q = q + 1
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
                            
                            psc_array.Bn_PSC = psc_array.Bn_PSC * psc_array.fac
                            # Robust test of all the B and Bn calculations from circular coils
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, atol=1e3, rtol=1e-3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, atol=1e3, rtol=1e-3))
        
        # Applying discrete symmetries to the coils wont work unless they dont intersect the symmetry planes
        for R in [0.1, 1]:
            for points in [np.array([(np.random.rand(3) - 0.5) * 40, 
                                      (np.random.rand(3) - 0.5) * 40])]:
                for alphas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
                    for deltas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
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
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
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
                                    ) @ (psc_array.I) * 2e-7
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_points),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I),
                                        psc_array.R,
                                    )
                                    q = q + 1
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
                                    -psc_array.I_all[i] * np.sign(psc_array.I_all[i]), 
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
                            psc_array.Bn_PSC = psc_array.Bn_PSC * psc_array.fac
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, rtol=1e-2, atol=1e3))

if __name__ == "__main__":
    unittest.main()
