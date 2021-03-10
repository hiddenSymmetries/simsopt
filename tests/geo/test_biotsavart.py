import unittest

import numpy as np

from simsopt.geo.fouriercurve import FourierCurve
from simsopt.geo.magneticaxis import StellaratorSymmetricCylindricalFourierCurve
from simsopt.geo.biotsavart import BiotSavart


def get_coil(num_quadrature_points=200):
    coil = FourierCurve(num_quadrature_points, 3)
    coeffs = coil.dofs
    coeffs[1][0] = 1.
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    coil.set_dofs(np.concatenate(coeffs))
    return coil


class Testing(unittest.TestCase):

    def test_biotsavart_exponential_convergence(self):
        coil = get_coil()
        from time import time
        # points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        points = np.asarray(10 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        tic = time()
        btrue   = BiotSavart([get_coil(1000)], [1e4]).set_points(points).B(compute_derivatives=0)
        # print(btrue)
        bcoarse = BiotSavart([get_coil(10)]  , [1e4]).set_points(points).B(compute_derivatives=0)
        bfine   = BiotSavart([get_coil(20)]  , [1e4]).set_points(points).B(compute_derivatives=0)
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
        # print(time()-tic)

        tic = time()
        dbtrue   = BiotSavart([get_coil(1000)], [1e4]).set_points(points).dB_by_dX(compute_derivatives=1)
        # print(dbtrue)
        dbcoarse = BiotSavart([get_coil(10)]  , [1e4]).set_points(points).dB_by_dX(compute_derivatives=1)
        dbfine   = BiotSavart([get_coil(20)]  , [1e4]).set_points(points).dB_by_dX(compute_derivatives=1)
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
        # print(time()-tic)

        tic = time()
        dbtrue   = BiotSavart([get_coil(1000)], [1e4]).set_points(points).d2B_by_dXdX(compute_derivatives=2)
        # print("dbtrue", dbtrue)
        dbcoarse = BiotSavart([get_coil(10)]  , [1e4]).set_points(points).d2B_by_dXdX(compute_derivatives=2)
        dbfine   = BiotSavart([get_coil(20)]  , [1e4]).set_points(points).d2B_by_dXdX(compute_derivatives=2)
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
        # print(time()-tic)

    def test_dB_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        coil = get_coil()
        bs = BiotSavart([coil], [1e4])
        points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = np.asarray(coil.get_dofs())
        B = bs.B()
        dBdX = bs.dB_by_dX()
        J0 = np.sum(B**2)
        dJ = bs.B_and_dB_vjp(B, dBdX)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ[0][0] * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.set_dofs(coil_dofs + eps * h)
            bs.clear_cached_properties()
            Bh = bs.B()
            Jh = np.sum(Bh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dBdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        coil = get_coil()
        bs = BiotSavart([coil], [1e4])
        points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = np.asarray(coil.get_dofs())
        B = bs.B()
        dBdX = bs.dB_by_dX()
        J0 = np.sum(dBdX**2)
        dJ = bs.B_and_dB_vjp(B, dBdX)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ[1][0] * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.set_dofs(coil_dofs + eps * h)
            bs.clear_cached_properties()
            dBdXh = bs.dB_by_dX()
            Jh = np.sum(dBdXh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def subtest_biotsavart_dBdX_taylortest(self, idx):
        coil = get_coil()
        bs = BiotSavart([coil], [1e4])
        points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        B0 = bs.B()[idx]
        dB = bs.dB_by_dX()[idx]
        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            deriv = dB.T.dot(direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Beps = bs.B()[idx]
                deriv_est = (Beps-B0)/(eps)
                new_err = np.linalg.norm(deriv-deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_biotsavart_dBdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_dBdX_taylortest(idx)

    def subtest_biotsavart_gradient_symmetric_and_divergence_free(self, idx):
        coil = get_coil()
        bs = BiotSavart([coil], [1e4])
        points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        dB = bs.dB_by_dX()
        assert abs(dB[idx][0, 0] + dB[idx][1, 1] + dB[idx][2, 2]) < 1e-14
        assert np.allclose(dB[idx], dB[idx].T)

    def test_biotsavart_gradient_symmetric_and_divergence_free(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_gradient_symmetric_and_divergence_free(idx)

    def subtest_d2B_by_dXdX_is_symmetric(self, idx):
        coil = get_coil()
        bs = BiotSavart([coil], [1e4])
        points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        d2B_by_dXdX = bs.d2B_by_dXdX()
        for i in range(3):
            assert np.allclose(d2B_by_dXdX[idx, :, :, i], d2B_by_dXdX[idx, :, :, i].T)

    def test_d2B_by_dXdX_is_symmetric(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_d2B_by_dXdX_is_symmetric(idx)

    def subtest_biotsavart_d2B_by_dXdX_taylortest(self, idx):
        coil = get_coil()
        bs = BiotSavart([coil], [1e4])
        points = np.asarray(17 *[[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        bs.set_points(points)
        B0, dB_by_dX, d2B_by_dXdX = bs.B(), bs.dB_by_dX(), bs.d2B_by_dXdX()
        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            first_deriv = dB_by_dX[idx].T.dot(direction)
            second_deriv = np.einsum('ijk,i,j->k', d2B_by_dXdX[idx], direction, direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Beps = bs.B()[idx]
                deriv_est = (Beps-B0)/(eps)
                second_deriv_est = 2*(deriv_est - first_deriv)/eps
                new_err = np.linalg.norm(second_deriv-second_deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_biotsavart_d2B_by_dXdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_d2B_by_dXdX_taylortest(idx)

if __name__ == "__main__":
    unittest.main()
