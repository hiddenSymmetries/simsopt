import numpy as np
import unittest

from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo import parameters

parameters['jit'] = False


def taylor_test(f, df, x, epsilons=None, direction=None):
    np.random.seed(1)
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(7, 20)))
    # print("################################################################################")
    err_old = 1e9
    counter = 0
    for eps in epsilons:
        if counter > 8:
            break
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps-fminuseps)/(2*eps)
        err = np.linalg.norm(dfest - dfx)
        # print(err)
        assert err < 1e-9 or err < 0.3 * err_old
        if err < 1e-9:
            break
        err_old = err
        counter += 1
    if err > 1e-10:
        assert counter > 3
    # print("################################################################################")

def get_curve(curvetype, rotated, x=np.asarray([0.5])):
    np.random.seed(2)
    rand_scale=0.01
    order = 4

    if curvetype == "CurveXYZFourier":
        curve = CurveXYZFourier(x, order)
    elif curvetype == "JaxCurveXYZFourier":
        curve = JaxCurveXYZFourier(x, order)
    elif curvetype == "CurveRZFourier":
        curve = CurveRZFourier(x, order, 2, True)
    else:
        assert False
    dofs = np.zeros((curve.num_dofs(), ))
    if curvetype in ["CurveXYZFourier", "JaxCurveXYZFourier"]:
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
    elif curvetype in ["CurveRZFourier"]:
        dofs[0] = 1.
        dofs[1] = 0.1
        dofs[order+1] = 0.1
    else:
        assert False
    curve.set_dofs(dofs)

    dofs = np.asarray(curve.get_dofs())
    curve.set_dofs(dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape))
    if rotated:
        curve = RotatedCurve(curve, 0.5, flip=False)
    return curve

class Testing(unittest.TestCase):

    curvetypes = ["CurveXYZFourier", "JaxCurveXYZFourier", "CurveRZFourier"]

    def subtest_curve_first_derivative(self, curvetype, rotated):
        h = 0.1
        epss = [0.5**i for i in range(10, 15)] 
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)
        f0 = curve.gamma()[0]
        deriv = curve.gammadash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = curve.gamma()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_curve_first_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_first_derivative(curvetype, rotated)

    def subtest_curve_second_derivative(self, curvetype, rotated):
        h = 0.1
        epss = [0.5**i for i in range(10, 15)] 
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)
        f0 = curve.gammadash()[0]
        deriv = curve.gammadashdash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = curve.gammadash()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_curve_second_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_second_derivative(curvetype, rotated)

    def subtest_curve_third_derivative(self, curvetype, rotated):
        h = 0.1
        epss = [0.5**i for i in range(10, 15)] 
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)
        f0 = curve.gammadashdash()[0]
        deriv = curve.gammadashdashdash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = curve.gammadashdash()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def subtest_coil_dof_numbering(self, curvetype, rotated):
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.get_dofs()
        cfc.set_dofs(coeffs)
        assert(np.allclose(coeffs, cfc.get_dofs()))

    def test_coil_dof_numbering(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_coil_dof_numbering(curvetype, rotated)

    def subtest_coil_coefficient_derivative(self, curvetype, rotated):
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.get_dofs()
        cfc.invalidate_cache()
        def f(dofs):
            cfc.set_dofs(dofs)
            return cfc.gamma().copy()
        def df(dofs):
            cfc.set_dofs(dofs)
            return cfc.dgamma_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            cfc.set_dofs(dofs)
            return cfc.gammadash().copy()
        def df(dofs):
            cfc.set_dofs(dofs)
            return cfc.dgammadash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            cfc.set_dofs(dofs)
            return cfc.gammadashdash().copy()
        def df(dofs):
            cfc.set_dofs(dofs)
            return cfc.dgammadashdash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            cfc.set_dofs(dofs)
            return cfc.gammadashdashdash().copy()
        def df(dofs):
            cfc.set_dofs(dofs)
            return cfc.dgammadashdashdash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_coil_coefficient_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_coil_coefficient_derivative(curvetype, rotated)

    def subtest_coil_kappa_derivative(self, curvetype, rotated):
        # This implicitly also tests the higher order derivatives of gamma as these
        # are needed to compute the derivative of the curvature.
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.get_dofs()
        def f(dofs):
            cfc.set_dofs(dofs)
            return cfc.kappa().copy()
        def df(dofs):
            cfc.set_dofs(dofs)
            return cfc.dkappa_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_coil_kappa_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_coil_kappa_derivative(curvetype, rotated)

    def subtest_curve_kappa_first_derivative(self, curvetype, rotated):
        h = 0.1
        epss = [0.5**i for i in range(12, 17)] 
        x = np.asarray([0.1234] + [0.1234 + eps for eps in epss])
        ma = get_curve(curvetype, rotated, x)
        f0 = ma.kappa()[0]
        deriv = ma.kappadash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = ma.kappa()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            # print("err", err)
            assert err < 0.55 * err_old
            err_old = err

    def test_curve_kappa_first_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_kappa_first_derivative(curvetype, rotated)

    def subtest_curve_incremental_arclength_derivative(self, curvetype, rotated):
        # This implicitly also tests the higher order derivatives of gamma as these
        # are needed to compute the derivative of the curvature.
        ma = get_curve(curvetype, rotated)
        coeffs = ma.get_dofs()
        def f(dofs):
            ma.set_dofs(dofs)
            return ma.incremental_arclength().copy()
        def df(dofs):
            ma.set_dofs(dofs)
            return ma.dincremental_arclength_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_incremental_arclength_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_incremental_arclength_derivative(curvetype, rotated)

    def subtest_curve_kappa_derivative(self, curvetype, rotated):
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.get_dofs()
        def f(dofs):
            cfc.set_dofs(dofs)
            return cfc.kappa().copy()
        def df(dofs):
            cfc.set_dofs(dofs)
            return cfc.dkappa_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_kappa_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_kappa_derivative(curvetype, rotated)

    def subtest_curve_torsion_derivative(self, curvetype, rotated):
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.get_dofs()
        def f(dofs):
            cfc.set_dofs(dofs)
            return cfc.torsion().copy()
        def df(dofs):
            cfc.set_dofs(dofs)
            return cfc.dtorsion_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_torsion_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_torsion_derivative(curvetype, rotated)

    def subtest_curve_frenet_frame(self, curvetype, rotated):
        ma = get_curve(curvetype, rotated)
        (t, n, b) = ma.frenet_frame()
        assert np.allclose(np.sum(n*t, axis=1), 0)
        assert np.allclose(np.sum(n*b, axis=1), 0)
        assert np.allclose(np.sum(t*b, axis=1), 0)
        assert np.allclose(np.sum(t*t, axis=1), 1)
        assert np.allclose(np.sum(n*n, axis=1), 1)
        assert np.allclose(np.sum(b*b, axis=1), 1)

    def test_curve_frenet_frame(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_frenet_frame(curvetype, rotated)

    def subtest_curve_frenet_frame_derivative(self, curvetype, rotated):
        ma = get_curve(curvetype, rotated)
        coeffs = ma.get_dofs()
        def f(dofs):
            ma.set_dofs(dofs)
            return ma.frenet_frame()[0].copy()
        def df(dofs):
            ma.set_dofs(dofs)
            return ma.dfrenet_frame_by_dcoeff()[0].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            ma.set_dofs(dofs)
            return ma.frenet_frame()[1].copy()
        def df(dofs):
            ma.set_dofs(dofs)
            return ma.dfrenet_frame_by_dcoeff()[1].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            ma.set_dofs(dofs)
            return ma.frenet_frame()[2].copy()
        def df(dofs):
            ma.set_dofs(dofs)
            return ma.dfrenet_frame_by_dcoeff()[2].copy()
        taylor_test(f, df, coeffs)

    def test_curve_frenet_frame_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_frenet_frame_derivative(curvetype, rotated)

    def subtest_curve_dkappa_by_dphi_derivative(self, curvetype, rotated):
        ma = get_curve(curvetype, rotated)
        coeffs = ma.get_dofs()
        def f(dofs):
            ma.set_dofs(dofs)
            return ma.kappadash().copy()
        def df(dofs):
            ma.set_dofs(dofs)
            return ma.dkappadash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_dkappa_by_dphi_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_dkappa_by_dphi_derivative(curvetype, rotated)

if __name__ == "__main__":
    unittest.main()
