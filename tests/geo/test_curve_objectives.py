import unittest

import numpy as np

from simsopt.geo import parameters
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curveobjectives import CurveLength, LpCurveCurvature, \
    LpCurveTorsion, MinimumDistance

parameters['jit'] = False


class Testing(unittest.TestCase):

    curvetypes = ["CurveXYZFourier", "JaxCurveXYZFourier", "CurveRZFourier"]

    def create_curve(self, curvetype, rotated):
        np.random.seed(1)
        rand_scale = 0.01
        order = 4
        nquadpoints = 200

        if curvetype == "CurveXYZFourier":
            coil = CurveXYZFourier(nquadpoints, order)
        elif curvetype == "JaxCurveXYZFourier":
            coil = JaxCurveXYZFourier(nquadpoints, order)
        elif curvetype == "CurveRZFourier":
            coil = CurveRZFourier(nquadpoints, order, 2, False)
        else:
            # print('Could not find' + curvetype)
            assert False
        dofs = np.zeros((coil.dof_size, ))
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

        coil.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
        if rotated:
            coil = RotatedCurve(coil, 0.5, flip=False)
        return coil

    def subtest_curve_length_taylor_test(self, curve):
        J = CurveLength(curve)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

    def test_curve_length_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_length_taylor_test(curve)

    def subtest_curve_curvature_taylor_test(self, curve):
        J = LpCurveCurvature(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

    def test_curve_curvature_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_curvature_taylor_test(curve)

    def subtest_curve_torsion_taylor_test(self, curve):
        J = LpCurveTorsion(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(10, 20):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

    def test_curve_torsion_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_torsion_taylor_test(curve)

    def subtest_curve_minimum_distance_taylor_test(self, curve):
        ncurves = 3
        curve_t = curve.curve.__class__.__name__ if isinstance(curve, RotatedCurve) else curve.__class__.__name__
        curves = [curve] + [RotatedCurve(self.create_curve(curve_t, False), 0.1*i, True) for i in range(1, ncurves)]
        J = MinimumDistance(curves, 0.2)
        for k in range(ncurves):
            curve_dofs = curves[k].x
            h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
            J0 = J.J()
            dJ = J.dJ(partials=True)(curves[k].curve if isinstance(curves[k], RotatedCurve) else curves[k])
            deriv = np.sum(dJ * h)
            assert np.abs(deriv) > 1e-10
            err = 1e6
            for i in range(5, 15):
                eps = 0.5**i
                curves[k].x = curve_dofs + eps * h
                Jh = J.J()
                deriv_est = (Jh-J0)/eps
                err_new = np.linalg.norm(deriv_est-deriv)
                # print("err_new %s" % (err_new))
                assert err_new < 0.55 * err
                err = err_new

    def test_curve_minimum_distance_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_minimum_distance_taylor_test(curve)


if __name__ == "__main__":
    unittest.main()
