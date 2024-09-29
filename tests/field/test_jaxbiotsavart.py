import unittest

import numpy as np
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier as CurveXYZFourier
from simsopt.field.biotsavart import JaxBiotSavart, BiotSavart, TotalVacuumEnergy, \
    CoilSelfNetForces, CoilCoilNetForces12, CoilCoilNetForces, CoilCoilNetTorques12
from simsopt.field.coil import Coil, ScaledCurrent
from simsopt.field.coil import JaxCurrent as Current

def get_curve(num_quadrature_points=200, perturb=False):
    coil = CurveXYZFourier(num_quadrature_points, 3)
    coeffs = np.zeros((3, len(coil.get_dofs()) // 3))
    coeffs[1][0] = 1.
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    coil.set_dofs(np.concatenate(coeffs))
    if perturb:
        d = coil.get_dofs()
        coil.set_dofs(d + np.random.uniform(size=d.shape))
    return coil

class Testing(unittest.TestCase):

    def test_jaxbiotsavart_both_interfaces_give_same_result(self):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        B1 = JaxBiotSavart([coil]).set_points(points).B()
        from simsoptpp import biot_savart_B
        B2 = biot_savart_B(points, [curve.gamma()], [curve.gammadash()], [1e4])
        assert np.linalg.norm(B1) > 1e-5
        assert np.allclose(B1, B2)

    def test_biotsavart_exponential_convergence(self):
        JaxBiotSavart([Coil(get_curve(), Current(1e4))])
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        btrue = JaxBiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).B()
        bcoarse = JaxBiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).B()
        bfine = JaxBiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).B()
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

        dbtrue = JaxBiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).dB_by_dX()
        dbcoarse = JaxBiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).dB_by_dX()
        dbfine = JaxBiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).dB_by_dX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

        dbtrue = JaxBiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbcoarse = JaxBiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbfine = JaxBiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).d2B_by_dXdX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

    def test_dB_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        bs2 = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        bs2.set_points(points)
        curve_dofs = curve.x 
        B = bs.B()
        B2 = bs2.B()
        J0 = np.sum(B**2)
        dJ = bs.B_vjp(B)(curve)
        dJ2 = bs2.B_vjp(B2)(curve)
        assert np.allclose(B, B2)
        assert np.allclose(dJ, dJ2)

        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Bh = bs.B()
            Bh2 = bs2.B()
            Jh = np.sum(Bh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dBdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
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
        bs = JaxBiotSavart([coil])
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
        bs = JaxBiotSavart([coil])
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
        bs = JaxBiotSavart([coil])
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
        bs = JaxBiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        d2B_by_dXdX = bs.d2B_by_dXdX()
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
        bs = JaxBiotSavart([coil])
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
        bs = JaxBiotSavart([coil])
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
        bs = JaxBiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        d2A_by_dXdX = bs.d2A_by_dXdX()
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
        bs = JaxBiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        gamma1 = bs._coils[0].curve.gamma()
        gammadash1 = bs._coils[0].curve.gammadash()
        B1 = bs.B()
        A1 = bs.A()
        dA1 = bs.dA_by_dcoilcurrents()
        dAJ1 = bs.dA_by_dX()
        dAH1 = bs.d2A_by_dXdX()
        dAdJ1 = bs.d2A_by_dXdcoilcurrents()
        dAdH1 = bs.d3A_by_dXdXdcoilcurrents()
        J1 = bs.dB_by_dX()
        H1 = bs.d2B_by_dXdX()
        dB1 = bs.dB_by_dcoilcurrents()
        dJ1 = bs.d2B_by_dXdcoilcurrents()
        dH1 = bs.d3B_by_dXdXdcoilcurrents()
        currents1 = [c.current.get_value() for c in bs._coils], 
        gammas1 = [c.curve.gamma() for c in bs._coils], 
        gammadashs1 = [c.curve.gammadash() for c in bs._coils]

        # Check agreement with analytic form
        bs = BiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        bs.set_points(points)
        currents2 = [c.current.get_value() for c in bs._coils], 
        gammas2 = [c.curve.gamma() for c in bs._coils], 
        gammadashs2 = [c.curve.gammadash() for c in bs._coils]
        assert np.allclose(currents1, currents2)
        assert np.allclose(gammas1, gammas2)
        assert np.allclose(gammadashs1, gammadashs2)
        B = bs.B()
        A = bs.A()
        dA = bs.dA_by_dcoilcurrents()
        dAJ = bs.dA_by_dX()
        dAH = bs.d2A_by_dXdX()
        dAdJ = bs.d2A_by_dXdcoilcurrents()
        dAdH = bs.d3A_by_dXdXdcoilcurrents()
        J = bs.dB_by_dX()
        H = bs.d2B_by_dXdX()
        dB = bs.dB_by_dcoilcurrents()
        dJ = bs.d2B_by_dXdcoilcurrents()
        dH = bs.d3B_by_dXdXdcoilcurrents()
        assert np.allclose(gamma1, bs._coils[0].curve.gamma())
        assert np.allclose(gammadash1, bs._coils[0].curve.gammadash())
        assert np.allclose(B1, B)
        assert np.allclose(J1, J)
        assert np.allclose(H1, H)
        assert np.allclose(dB1, dB)
        assert np.allclose(dJ1, dJ)
        assert np.allclose(dH1, dH)  # Not quite in agreement?
        assert np.allclose(A1, A)
        assert np.allclose(dAJ1, dAJ)
        assert np.allclose(dAH1, dAH)
        assert np.allclose(dA1, dA)
        assert np.allclose(dAdJ1, dAdJ)
        assert np.allclose(dAdH1, dAdH)

        bs = JaxBiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
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
        bs = JaxBiotSavart([coil])
        bs2 = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        bs2.set_points(points)
        coil_dofs = coil.x
        A = bs.A()
        A2 = bs2.A()
        J0 = np.sum(A**2)
        dJ = bs.A_vjp(A)(coil)
        dJ2 = bs2.A_vjp(A2)(coil)
        assert np.allclose(A, A2)
        assert np.allclose(dJ, dJ2)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ2 * h)
        err = 1e6
        for i in range(2, 10):
            eps = 0.5**i
            coil.x = coil_dofs + eps * h
            Ah = bs2.A()
            Ah2 = bs2.A()
            Jh = np.sum(Ah**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dAdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, ScaledCurrent(Current(1), 1e4))
        bs = JaxBiotSavart([coil])
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
        bs = JaxBiotSavart([coil])

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
        bs = JaxBiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
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

    def test_net_force_calculation(self):
        from scipy.special import ellipk, ellipe
        
        ncoils = 2
        I1 = 5.0e4
        I2 = -3.0e4
        Z1 = 1.0
        Z2 = 4.0
        R1 = 0.5
        R2 = R1
        # a = 1e-5
        # points = np.array([[0.0, 0.0, Z1], [0.0, 0.0, Z2]])
        # curve0 = get_curve()
        curve0 = CurveXYZFourier(200, 3)
        coeffs = np.zeros((3, len(curve0.get_dofs()) // 3))
        coeffs[0][2] = R1
        coeffs[1][1] = R1
        coeffs[2][0] = Z1
        curve0.set_dofs(np.concatenate(coeffs))
        current0 = Current(I1)
        curve1 = CurveXYZFourier(200, 3)
        coeffs = np.zeros((3, len(curve1.get_dofs()) // 3))
        coeffs[0][2] = R2
        coeffs[1][1] = R2
        coeffs[2][0] = Z2
        curve1.set_dofs(np.concatenate(coeffs))
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.plot(curve1.gamma()[:, 0], curve1.gamma()[:, 1])
        # plt.show()
        current1 = Current(I2)
        bs = JaxBiotSavart([Coil(curve0, current0), Coil(curve1, current1)])

        # alphas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * np.pi
        # deltas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        k = np.sqrt(4.0 * R1 * R2 / ((R1 + R2) ** 2 + (Z2 - Z1) ** 2))
        mu0 = 4 * np.pi * 1e-7
        # Jackson uses K(k) and E(k) but this corresponds to
        # K(k^2) and E(k^2) in scipy library
        F_analytic = mu0 * I1 * I2 * k * (Z2 - Z1) * (
            (2  - k ** 2) * ellipe(k ** 2) / (1 - k ** 2) - 2.0 * ellipk(k ** 2)
        ) / (4.0 * np.sqrt(R1 * R2))
        
        F_bs = bs.coil_coil_forces()
        print(F_bs, F_analytic)
        assert np.allclose(F_bs[:, 0], 0.0)
        assert np.allclose(F_bs[:, 1], 0.0)
        assert np.allclose(np.abs(F_bs[:, 2]), abs(F_analytic))

        T_bs = bs.coil_coil_torques()
        print(T_bs)
        assert np.allclose(T_bs, 0.0)


    def test_net_force_and_torque_calculation_far_field(self):       
        from simsopt.geo import JaxCurvePlanarFourier 
        ncoils = 2
        I1 = 5.0e4
        I2 = -3.0e4
        Z1 = 0.0
        Z2 = 100.0
        R1 = 0.5
        R2 = R1
        # a = 1e-5
        points = np.array([[0.0, 0.0, Z1], [0.0, 0.0, Z2]])
        # curve0 = get_curve()
        curve0 = JaxCurvePlanarFourier(200, 0)
        dofs = np.zeros(8)
        dofs[0] = R1
        dofs[1:5] = (np.random.rand(4) - 0.5)
        dofs[5:9] = points[0, :]
        curve0.set_dofs(dofs)
        q_norm = dofs[1:5] / np.linalg.norm(dofs[1:5])
        normal_orig = np.array([0, 0, 1]).T
        rot_mat = np.array(
            [[(1.0 - 2 * (q_norm[2] ** 2 + q_norm[3] ** 2)), 
            2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]), 
            2 * (q_norm[1] * q_norm[3] + q_norm[0] * q_norm[2])], 
            [2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]), 
            (1.0 - 2 * (q_norm[1] ** 2 + q_norm[3] ** 2)),
            2 * (q_norm[2] * q_norm[3] - q_norm[0] * q_norm[1])], 
            [2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]), 
            2 * (q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]),
            (1.0 - 2 * (q_norm[1] ** 2 + q_norm[2] ** 2))]])
        normal0 = rot_mat @ normal_orig
        current0 = Current(I1)
        curve1 = JaxCurvePlanarFourier(200, 0)
        dofs = np.zeros(8)
        dofs[0] = R2
        dofs[1:5] = (np.random.rand(4) - 0.5) * 2 * np.pi
        dofs[5:9] = points[1, :]
        curve1.set_dofs(dofs)
        q_norm = dofs[1:5] / np.linalg.norm(dofs[1:5])
        normal_orig = np.array([0, 0, 1]).T
        rot_mat = np.array(
            [[(1.0 - 2 * (q_norm[2] ** 2 + q_norm[3] ** 2)), 
            2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]), 
            2 * (q_norm[1] * q_norm[3] + q_norm[0] * q_norm[2])], 
            [2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]), 
            (1.0 - 2 * (q_norm[1] ** 2 + q_norm[3] ** 2)),
            2 * (q_norm[2] * q_norm[3] - q_norm[0] * q_norm[1])], 
            [2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]), 
            2 * (q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]),
            (1.0 - 2 * (q_norm[1] ** 2 + q_norm[2] ** 2))]])
        normal1 = rot_mat @ normal_orig
        current1 = Current(I2)
        bs = JaxBiotSavart([Coil(curve0, current0), Coil(curve1, current1)])

        # alphas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * np.pi
        # deltas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        r12 = points[1, :] - points[0, :]
        r12_mag = np.linalg.norm(r12)
        m1 = np.pi * R1 ** 2 * I1 * normal0
        m2 = np.pi * R2 ** 2 * I2 * normal1
        B1 = 1e-7 * (3 * r12 * np.dot(r12, m1) / r12_mag ** 5 - m1 / r12_mag ** 3)
        F_analytic = 3 * 1e-7 / r12_mag ** 5 * (np.dot(m1, r12) * m2 + \
            np.dot(m2, r12) * m1 + np.dot(m1, m2) * r12 - 5 * np.dot(m1, r12
            ) * np.dot(m2, r12) * r12 / r12_mag ** 2)
        T_analytic = np.cross(m2, B1)
        
        F_bs = bs.coil_coil_forces()
        F = CoilCoilNetForces(bs).J()
        F_12 = CoilCoilNetForces12(bs, bs).J() / 2.0
        T_12 = CoilCoilNetTorques12(bs, bs).J() / 2.0
        print(F_bs, F_analytic, F_12, F, np.sum(F_bs ** 2))
        print(T_12)

        # dJ should not match anyways 
        print('dJ = ', CoilCoilNetForces(bs).dJ(), CoilCoilNetForces12(bs, bs).dJ())
        assert np.allclose(F_bs[0, :], F_analytic)
        assert np.allclose(np.sum(F_bs ** 2), F)
        assert np.allclose(np.sum(F_bs ** 2), F_12)
        # assert np.allclose(CoilCoilNetForces(bs).dJ(), CoilCoilNetForces12(bs, bs).dJ())

        T_bs = bs.coil_coil_torques()
        print(T_bs, T_analytic)
        assert np.allclose(T_bs[0, :], T_analytic)

    def total_energy_test(self):       
        from simsopt.geo import JaxCurvePlanarFourier 
        ncoils = 1
        I1 = 5.0e6
        R1 = 1
        # a = 1e-5
        curve0 = JaxCurvePlanarFourier(200, 0)
        dofs = np.zeros(8)
        dofs[0] = R1
        dofs[1:5] = np.ones(1)
        dofs[5:9] = np.array([0.0, 0.0, 0.0])
        curve0.set_dofs(dofs)
        current0 = Current(I1)
        bs = JaxBiotSavart([Coil(curve0, current0)])
        E_dipoles = bs.total_vacuum_energy_pure(bs.get_curve_dofs(), bs.get_currents(), a=0.01, b=0.01)
        print('U = ', E_dipoles)
        dE_dipoles = bs.dtve_by_dX(bs.get_curve_dofs(), bs.get_currents(), a=0.01, b=0.01)
        print('dU = ', dE_dipoles)
        tve = TotalVacuumEnergy(bs)
        print('J = ', tve.J())
        print('dJ = ', tve.dJ())
        sf = CoilSelfNetForces(bs)
        print(sf.J(), sf.dJ())

if __name__ == "__main__":
    unittest.main()