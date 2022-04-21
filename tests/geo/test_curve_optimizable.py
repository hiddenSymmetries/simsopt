import numpy as np
import unittest

from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo import parameters
from simsopt.geo.curveobjectives import CurveLength
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.serial import least_squares_serial_solve

parameters['jit'] = False


class Testing(unittest.TestCase):

    def subtest_curve_length_optimisation(self, rotated):
        nquadrature = 100
        nfourier = 4
        nfp = 5
        curve = CurveRZFourier(nquadrature, nfourier, nfp, True)

        # Initialize the Fourier amplitudes to some random values
        x0 = np.random.rand(curve.dof_size) - 0.5
        x0[0] = 3.0
        curve.x = x0
        print('Initial curve dofs: ', curve.x)

        # Tell the curve object that the first Fourier mode is fixed, whereas
        # all the other dofs are not.
        curve.fix(0)

        if rotated:
            curve = RotatedCurve(curve, 0.5, flip=False)

        # Presently in simsgeo, the length objective is a separate object
        # rather than a function of Curve itself.
        obj = CurveLength(curve)

        # For now, we need to add this attribute to CurveLength. Eventually
        # this would hopefully be done in simsgeo, but for now I'll put it here.

        print('Initial curve length: ', obj.J())

        # Each target function is then equipped with a shift and weight, to
        # become a term in a least-squares objective function.
        # A list of terms are combined to form a nonlinear-least-squares
        # problem.
        prob = LeastSquaresProblem.from_tuples([(obj.J, 0.0, 1.0)])

        # At the initial condition, get the Jacobian two ways: analytic
        # derivatives and finite differencing. The difference should be small.
        # Ignoring Jacs for the time being
        # fd_jac = prob.dofs.fd_jac()
        # jac = prob.dofs.jac()
        # print('finite difference Jacobian:')
        # print(fd_jac)
        # print('Analytic Jacobian:')
        # print(jac)
        # print('Difference:')
        # print(fd_jac - jac)
        # assert np.allclose(fd_jac, jac, rtol=1e-4, atol=1e-4)

        # Solve the minimization problem:
        least_squares_serial_solve(prob, ftol=1e-10, xtol=1e-10, gtol=1e-10)

        print('At the optimum, x: ', prob.x)
        print(' Final curve dofs: ', curve.local_full_x)
        print(' Final curve length:    ', obj.J())
        print(' Expected final length: ', 2 * np.pi * x0[0])
        print(' objective function: ', prob.objective())
        assert abs(obj.J() - 2 * np.pi * x0[0]) < 1e-8

    def test_curve_first_derivative(self):
        for rotated in [True, False]:
            with self.subTest(rotated=rotated):
                self.subtest_curve_length_optimisation(rotated=rotated)


if __name__ == "__main__":
    unittest.main()
