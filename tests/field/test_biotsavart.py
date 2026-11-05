import unittest

import numpy as np

from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent

# Number of field evaluation points. The value is arbitrary, but should be at
# least 2 and differ from the other array dimensions (3 spatial components,
# 200 curve quadrature points) so that broadcasting or axis mix-ups fail.
NPOINTS = 17

# Coil current for the tests of derivatives with respect to the coil currents,
# which do not depend on it. Zero rather than 1, because a current of 1 would
# hide an accessor that returned the current-scaled total instead of the
# unit-current derivative.
NEUTRAL_CURRENT = 0.0

# Relative tolerance between two evaluations of the same Biot-Savart kernel at
# the same points, which agree to rounding.
SAME_KERNEL_RTOL = 1e-13


def get_curve(num_quadrature_points=200, perturb=False):
    coil = CurveXYZFourier(num_quadrature_points, 3)
    coeffs = coil.dofs_matrix
    coeffs[1][0] = 1.
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    coil.set_dofs(np.concatenate(coeffs))
    if perturb:
        d = coil.get_dofs()
        coil.set_dofs(d + np.random.uniform(size=d.shape))
    return coil


def random_points_away_from_coils(rng, curves):
    """
    Returns ``NPOINTS`` random field evaluation points in the cube
    ``[-0.3, 0.3]^3``, away from the coils.

    The ``get_curve()`` coil is a circle of radius 0.5 centred at ``(0, 1, 0)``
    in the ``x = 0`` plane. It spans y from 0.5 to 1.5, so the cube lies
    outside it, below it in y, and every point is at least 0.2 from the wire.
    Perturbed coils can come closer, so the distance from the points to the
    quadrature points of each curve in ``curves`` is checked against that 0.2
    clearance, which keeps the points away from the near-singular field close
    to a wire.
    """
    points = rng.uniform(-0.3, 0.3, size=(NPOINTS, 3))
    for curve in curves:
        distance = np.linalg.norm(points[:, None, :] - curve.gamma()[None, :, :], axis=-1).min()
        assert distance >= 0.2, f"an evaluation point is {distance:.3g} from a coil"
    return points


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
        BiotSavart([Coil(get_curve(), Current(1e4))])
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        btrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).B()
        bcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).B()
        bfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).B()
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).dB_by_dX()
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).dB_by_dX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).dB_by_dX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).d2B_by_dXdX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

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

    def test_coil_current_derivatives_are_not_updated_by_later_evaluations(self):
        """
        Arrays returned by the public coil-current derivative methods keep
        their values after the field is evaluated at new points.
        """
        curve = get_curve()
        bs = BiotSavart([Coil(curve, Current(NEUTRAL_CURRENT))])
        rng = np.random.default_rng(0)
        points0 = random_points_away_from_coils(rng, [curve])
        points1 = random_points_away_from_coils(rng, [curve])
        methods = [bs.dB_by_dcoilcurrents, bs.d2B_by_dXdcoilcurrents, bs.d3B_by_dXdXdcoilcurrents,
                   bs.dA_by_dcoilcurrents, bs.d2A_by_dXdcoilcurrents, bs.d3A_by_dXdXdcoilcurrents]

        bs.set_points(points0)
        held = [method()[0] for method in methods]
        expected = [h.copy() for h in held]
        bs.set_points(points1)
        new = [method()[0] for method in methods]

        for method, h, e, n in zip(methods, held, expected, new):
            assert not np.allclose(n, e), (
                f"{method.__name__}: the second point set must change the values, "
                "otherwise a live view would go undetected")
            np.testing.assert_array_equal(h, e, err_msg=f"{method.__name__} result changed after set_points")

    def subtest_vector_potential_coil_current_derivatives(self, evaluate_A_elsewhere_first):
        """
        Compares the vector potential coil-current derivatives of a two-coil
        field with the vector potential of each coil carrying unit current.
        If ``evaluate_A_elsewhere_first``, the field first evaluates ``A``
        and its derivatives at other points, so the per-coil caches hold
        values for those points when the evaluation points change.
        """
        rng = np.random.default_rng(1)
        curves = [get_curve(), get_curve()]
        # The perturbation only makes the two coils distinct, so that a mix-up
        # between their derivatives fails. It can also bring the perturbed coil
        # within 0.2 of a sampled point, which random_points_away_from_coils
        # rejects, so the seed must be one for which the points stay clear.
        # Seed 1 does; many others fail that assertion.
        curves[1].x = curves[1].x + 0.1 * rng.standard_normal(curves[1].x.shape)
        bs = BiotSavart([Coil(curves[0], Current(NEUTRAL_CURRENT)), Coil(curves[1], Current(NEUTRAL_CURRENT))])
        points_other = random_points_away_from_coils(rng, curves)
        points = random_points_away_from_coils(rng, curves)
        if evaluate_A_elsewhere_first:
            bs.set_points(points_other)
            bs.A()
            bs.dA_by_dX()
            bs.d2A_by_dXdX()
        bs.set_points(points)

        derivatives = [bs.dA_by_dcoilcurrents(), bs.d2A_by_dXdcoilcurrents(), bs.d3A_by_dXdXdcoilcurrents()]
        names = ["dA_by_dcoilcurrents", "d2A_by_dXdcoilcurrents", "d3A_by_dXdXdcoilcurrents"]

        for i, curve in enumerate(curves):
            unit = BiotSavart([Coil(curve, Current(1.0))])
            unit.set_points(points)
            expected = [unit.A(), unit.dA_by_dX(), unit.d2A_by_dXdX()]
            if evaluate_A_elsewhere_first:
                unit_other = BiotSavart([Coil(curve, Current(1.0))]).set_points(points_other)
                stale = [unit_other.A(), unit_other.dA_by_dX(), unit_other.d2A_by_dXdX()]
                for name, s, e in zip(names, stale, expected):
                    assert not np.allclose(s, e), (
                        f"{name}[{i}]: the other point set must change the values, "
                        "otherwise a stale cache would go undetected")
            for name, d, e in zip(names, derivatives, expected):
                np.testing.assert_allclose(d[i], e, rtol=SAME_KERNEL_RTOL, atol=0,
                                           err_msg=f"{name}[{i}] differs from coil {i} at unit current")

    def test_vector_potential_coil_current_derivatives_on_fresh_field(self):
        self.subtest_vector_potential_coil_current_derivatives(evaluate_A_elsewhere_first=False)

    def test_vector_potential_coil_current_derivatives_after_set_points(self):
        self.subtest_vector_potential_coil_current_derivatives(evaluate_A_elsewhere_first=True)

    def test_vector_potential_vjp_wrt_coil_currents_after_set_points(self):
        """
        The coil-current part of `A_vjp` and `A_and_dA_vjp` uses the vector
        potential at the current evaluation points, even if `A` was last
        evaluated at other points.
        """
        rng = np.random.default_rng(2)
        curve = get_curve()
        current = Current(NEUTRAL_CURRENT)
        bs = BiotSavart([Coil(curve, current)])
        points_other = random_points_away_from_coils(rng, [curve])
        points = random_points_away_from_coils(rng, [curve])
        v = rng.standard_normal((NPOINTS, 3))
        vgrad = rng.standard_normal((NPOINTS, 3, 3))
        unit = BiotSavart([Coil(curve, Current(1.0))])
        unit.set_points(points)
        expected = np.sum(v * unit.A())
        expected_grad = np.sum(vgrad * unit.dA_by_dX())
        unit_other = BiotSavart([Coil(curve, Current(1.0))]).set_points(points_other)
        assert not np.allclose(np.sum(v * unit_other.A()), expected), (
            "the other point set must change the A term, otherwise a stale cache would go undetected")
        assert not np.allclose(np.sum(vgrad * unit_other.dA_by_dX()), expected_grad), (
            "the other point set must change the grad A term, otherwise a stale cache would go undetected")

        bs.set_points(points_other)
        bs.A()
        bs.dA_by_dX()
        bs.set_points(points)
        dJ = bs.A_vjp(v)(current)

        bs.set_points(points_other)
        bs.A()
        bs.dA_by_dX()
        bs.set_points(points)
        dJ_pair = bs.A_and_dA_vjp(v, vgrad)

        np.testing.assert_allclose(dJ, [expected], rtol=SAME_KERNEL_RTOL, err_msg="A_vjp current derivative")
        np.testing.assert_allclose(dJ_pair[0](current), [expected], rtol=SAME_KERNEL_RTOL,
                                   err_msg="A_and_dA_vjp current derivative of A")
        np.testing.assert_allclose(dJ_pair[1](current), [expected_grad], rtol=SAME_KERNEL_RTOL,
                                   err_msg="A_and_dA_vjp current derivative of grad A")

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
        np.random.seed(1)

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
            temp = np.sum(B*new_n[None, :], axis=1)*r
            return temp[0]

        # int_r int_theta B int r dr dtheta
        from scipy import integrate
        r = 0.15
        fluxB = integrate.dblquad(f, 0, r, 0, 2*np.pi, epsabs=1e-15, epsrel=1e-15)

        # num range used to be (20, 60) but this fails for num <= 20-30 for certain
        # random coil initializations since don't have enough quadrature points
        # to integrate to numerical precision.
        for num in range(40, 100):
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
