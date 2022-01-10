import unittest

import numpy as np

from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveobjectives import CurveLength
from simsopt.objectives.utilities import MPIObjective, QuadraticPenalty
from simsopt.geo import parameters
parameters['jit'] = False


class UtilityObjectiveTesting(unittest.TestCase):

    def create_curve(self):
        np.random.seed(1)
        rand_scale = 0.01
        order = 4
        nquadpoints = 200
        curve = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve.dof_size, ))
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
        curve.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
        return curve

    def subtest_quadratic_penalty(self, curve, threshold):
        J = QuadraticPenalty(CurveLength(curve), threshold)
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
            print("err_new %s" % (err_new))
            # assert err_new < 0.55 * err or err_new < 1e-13
            err = err_new

    def test_quadratic_penalty(self):
        curve = self.create_curve()
        J = CurveLength(curve)
        self.subtest_quadratic_penalty(curve, J.J()+0.1)
        self.subtest_quadratic_penalty(curve, J.J()-0.1)
