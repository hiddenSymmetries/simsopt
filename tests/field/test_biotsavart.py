import unittest

import numpy as np

from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent


def get_curve(num_quadrature_points=200, perturb=False):
    coil = CurveXYZFourier(num_quadrature_points, 3)
    coeffs = coil.dofs
    coeffs[1][0] = 1.
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    coil.set_dofs(np.concatenate(coeffs))
    if perturb:
        d = coil.get_dofs()
        coil.set_dofs(d + np.random.uniform(size=d.shape))
    return coil


class Testing(unittest.TestCase):

    def test_biotsavart_both_interfaces_give_same_result(self):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        B1 = BiotSavart([coil]).set_points(points).B()
        from simsoptpp import biot_savart_B
        B2 = biot_savart_B(points, [curve.gamma()], [curve.gammadash()], [1e4])
        assert np.linalg.norm(B1) > 1e-5
        assert np.allclose(B1, B2)

    def test_biotsavart_exponential_convergence(self):
        coil = BiotSavart([Coil(get_curve(), Current(1e4))])
        from time import time
        # points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        tic = time()
        btrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).B()
        # print(btrue)
        bcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).B()
        bfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).B()
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
        # print(time()-tic)

        tic = time()
        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).dB_by_dX()
        # print(dbtrue)
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).dB_by_dX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).dB_by_dX()
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
        # print(time()-tic)

        tic = time()
        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).d2B_by_dXdX()
        # print("dbtrue", dbtrue)
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).d2B_by_dXdX()
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
        # print(time()-tic)

    def test_dB_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        curve_dofs = curve.x
        B = bs.B()
        J0 = np.sum(B**2)
        dJ = bs.B_vjp(B)(curve)

        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Bh = bs.B()
            Jh = np.sum(Bh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dBdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        curve_dofs = curve.x
        B = bs.B()
        dBdX = bs.dB_by_dX()
        J0 = np.sum(dBdX**2)
        dJ = bs.B_and_dB_vjp(B, dBdX)[1](curve)

        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            dBdXh = bs.dB_by_dX()
            Jh = np.sum(dBdXh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def subtest_biotsavart_dBdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
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
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
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
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
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
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        dB_by_dX, d2B_by_dXdX = bs.dB_by_dX(), bs.d2B_by_dXdX()
        for d1 in range(3):
            for d2 in range(3):
                second_deriv = d2B_by_dXdX[idx, d1, d2]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i

                    ed2 = np.zeros((1, 3))
                    ed2[0, d2] = 1.

                    bs.set_points(points + eps * ed2)
                    dB_dXp = bs.dB_by_dX()[idx, d1]

                    bs.set_points(points - eps * ed2)
                    dB_dXm = bs.dB_by_dX()[idx, d1]

                    second_deriv_est = (dB_dXp - dB_dXm)/(2. * eps)

                    new_err = np.linalg.norm(second_deriv-second_deriv_est)
                    assert new_err < 0.30 * err
                    err = new_err

    def test_biotsavart_d2B_by_dXdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_d2B_by_dXdX_taylortest(idx)

    def test_biotsavart_B_is_curlA(self):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        B, dA_by_dX = bs.B(), bs.dA_by_dX() 
        curlA1 = dA_by_dX[:, 1, 2] - dA_by_dX[:, 2, 1]
        curlA2 = dA_by_dX[:, 2, 0] - dA_by_dX[:, 0, 2]
        curlA3 = dA_by_dX[:, 0, 1] - dA_by_dX[:, 1, 0]
        curlA = np.concatenate((curlA1[:, None], curlA2[:, None], curlA3[:, None]), axis=1)
        err = np.max(np.abs(curlA - B))
        assert err < 1e-14

    def subtest_biotsavart_dAdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        A0 = bs.A()[idx]
        dA = bs.dA_by_dX()[idx]

        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            deriv = dA.T.dot(direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Aeps = bs.A()[idx]
                deriv_est = (Aeps-A0)/(eps)
                new_err = np.linalg.norm(deriv-deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_biotsavart_dAdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_dAdX_taylortest(idx)

    def subtest_biotsavart_d2A_by_dXdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        dA_by_dX, d2A_by_dXdX = bs.dA_by_dX(), bs.d2A_by_dXdX()
        for d1 in range(3):
            for d2 in range(3):
                second_deriv = d2A_by_dXdX[idx, d1, d2]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i

                    ed2 = np.zeros((1, 3))
                    ed2[0, d2] = 1.

                    bs.set_points(points + eps * ed2)
                    dA_dXp = bs.dA_by_dX()[idx, d1]

                    bs.set_points(points - eps * ed2)
                    dA_dXm = bs.dA_by_dX()[idx, d1]

                    second_deriv_est = (dA_dXp - dA_dXm)/(2. * eps)

                    new_err = np.linalg.norm(second_deriv-second_deriv_est)
                    #print("new_err", new_err)
                    assert new_err < 0.30 * err
                    err = new_err

    def test_biotsavart_d2A_by_dXdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_d2A_by_dXdX_taylortest(idx)

    def test_biotsavart_coil_current_taylortest(self):
        curve0 = get_curve()
        c0 = 1e4
        current0 = Current(c0)
        curve1 = get_curve(perturb=True)
        current1 = Current(1e3)
        bs = BiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        B = bs.B()
        J = bs.dB_by_dX()
        H = bs.d2B_by_dXdX()
        dB = bs.dB_by_dcoilcurrents()
        dJ = bs.d2B_by_dXdcoilcurrents()
        dH = bs.d3B_by_dXdXdcoilcurrents()

        # the B field is linear in the current, so a small stepsize is not necessary
        current0.x = [0]
        B0 = bs.B()
        J0 = bs.dB_by_dX()
        H0 = bs.d2B_by_dXdX()
        dB_approx = (B-B0)/(c0)
        dJ_approx = (J-J0)/(c0)
        dH_approx = (H-H0)/(c0)
        assert np.linalg.norm(dB[0]-dB_approx) < 1e-15
        assert np.linalg.norm(dJ[0]-dJ_approx) < 1e-15
        print(f"H norm is {np.linalg.norm(dH[0]-dH_approx)}")
        assert np.linalg.norm(dH[0]-dH_approx) < 1e-15

    def test_dA_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, ScaledCurrent(Current(1), 1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = coil.x
        A = bs.A()
        J0 = np.sum(A**2)
        dJ = bs.A_vjp(A)(coil)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.x = coil_dofs + eps * h
            Ah = bs.A()
            Jh = np.sum(Ah**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dAdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, ScaledCurrent(Current(1), 1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = coil.x
        A = bs.A()
        dAdX = bs.dA_by_dX()
        J0 = np.sum(dAdX**2)
        dJ = bs.A_and_dA_vjp(A, dAdX)[1](coil)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.x = coil_dofs + eps * h
            dAdXh = bs.dA_by_dX()
            Jh = np.sum(dAdXh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_flux_through_disk(self):
        # this test makes sure that the toroidal flux through a disk (D)
        # given by \int_D B \cdot n dB = \int_{\partial D} A \cdot dl

        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('zyx', [21.234, 8.431, -4.86392], degrees=True).as_matrix()
        new_n = rot @ np.array([0, 0, 1])

        curve = get_curve(perturb=True)
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])

        # define the disk
        def f(t, r):
            x = r * np.cos(t).reshape((-1, 1))
            y = r * np.sin(t).reshape((-1, 1))
            pts = np.concatenate((x, y, np.zeros((x.shape[1], 1))), axis=1) @ rot.T
            bs.set_points(pts)
            B = bs.B()
            return np.sum(B*new_n[None, :], axis=1)*r

        # int_r int_theta B int r dr dtheta
        from scipy import integrate
        r = 0.15
        fluxB = integrate.dblquad(f, 0, r, 0, 2*np.pi, epsabs=1e-15, epsrel=1e-15) 

        for num in range(20, 60):
            npoints = num
            angles = np.linspace(0, 2*np.pi, npoints, endpoint=False).reshape((-1, 1))
            t = np.concatenate((-np.sin(angles), np.cos(angles), np.zeros((angles.size, 1))), axis=1) @ rot.T
            pts = r*np.concatenate((np.cos(angles), np.sin(angles), np.zeros((angles.size, 1))), axis=1) @ rot.T
            bs.set_points(pts)
            A = bs.A()
            fluxA = r*np.sum(A*t) * 2 * np.pi/npoints

            assert np.abs(fluxB[0]-fluxA)/fluxB[0] < 1e-14

    def test_biotsavart_vector_potential_coil_current_taylortest(self):
        curve0 = get_curve()
        c0 = 1e4
        current0 = Current(c0)
        curve1 = get_curve(perturb=True)
        current1 = Current(1e3)
        bs = BiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        A = bs.A()
        J = bs.dA_by_dX()
        H = bs.d2A_by_dXdX()

        #trigger recompute bell for code coverage of field cache
        bs.recompute_bell()
        dA = bs.dA_by_dcoilcurrents()
        bs.recompute_bell()
        dJ = bs.d2A_by_dXdcoilcurrents()
        bs.recompute_bell()
        dH = bs.d3A_by_dXdXdcoilcurrents()

        # the A field is linear in the current, so a small stepsize is not necessary
        current0.x = [0]
        A0 = bs.A()
        J0 = bs.dA_by_dX()
        H0 = bs.d2A_by_dXdX()
        dA_approx = (A-A0)/(c0)
        dJ_approx = (J-J0)/(c0)
        dH_approx = (H-H0)/(c0)
        assert np.linalg.norm(dA[0]-dA_approx) < 1e-15
        assert np.linalg.norm(dJ[0]-dJ_approx) < 1e-15
        assert np.linalg.norm(dH[0]-dH_approx) < 1e-15


if __name__ == "__main__":
    unittest.main()
