import numpy as np
import unittest

from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo import parameters
from simsopt.geo.curveobjectives import CurveLength
from simsopt.core.optimizable import optimizable
from simsopt.core.least_squares_problem import LeastSquaresProblem
from simsopt.solve.serial_solve import least_squares_serial_solve

parameters['jit'] = False


class Testing(unittest.TestCase):

    def subtest_curve_length_optimisation(self, rotated):
        nquadrature = 100
        nfourier = 4
        nfp = 5
        curve = CurveRZFourier(nquadrature, nfourier, nfp, True)
        if rotated:
            curve = RotatedCurve(curve, 0.5, flip=False)

        # Initialize the Fourier amplitudes to some random values
        x0 = np.random.rand(curve.num_dofs()) - 0.5
        x0[0] = 3.0
        curve.set_dofs(x0)
        print('Initial curve dofs: ', curve.get_dofs())

        # Tell the curve object that the first Fourier mode is fixed, whereas
        # all the other dofs are not.
        curve.all_fixed(False)
        curve.fixed[0] = True

        # Presently in simsgeo, the length objective is a separate object
        # rather than a function of Curve itself.
        obj = optimizable(CurveLength(curve))

        # For now, we need to add this attribute to CurveLength. Eventually
        # this would hopefully be done in simsgeo, but for now I'll put it here.
        obj.depends_on = ['curve']

        print('Initial curve length: ', obj.J())

        # Each target function is then equipped with a shift and weight, to
        # become a term in a least-squares objective function.
        # A list of terms are combined to form a nonlinear-least-squares
        # problem.
        prob = LeastSquaresProblem([(obj, 0.0, 1.0)])

        # At the initial condition, get the Jacobian two ways: analytic
        # derivatives and finite differencing. The difference should be small.
        fd_jac = prob.dofs.fd_jac()
        jac = prob.dofs.jac()
        print('finite difference Jacobian:')
        print(fd_jac)
        print('Analytic Jacobian:')
        print(jac)
        print('Difference:')
        print(fd_jac - jac)
        assert np.allclose(fd_jac, jac, rtol=1e-4, atol=1e-4)

        # Solve the minimization problem:
        least_squares_serial_solve(prob, ftol=1e-10, xtol=1e-10, gtol=1e-10)

        print('At the optimum, x: ', prob.x)
        print(' Final curve dofs: ', curve.get_dofs())
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
