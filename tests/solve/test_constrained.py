import unittest

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from simsopt._core.optimizable import Optimizable
from simsopt.objectives.functions import Identity, Rosenbrock, Affine
from simsopt.objectives.constrained import ConstrainedProblem
from simsopt.solve.serial import constrained_serial_solve, serial_solve
if MPI is not None:
    from simsopt.util.mpi import MpiPartition
    from simsopt.solve.mpi import constrained_mpi_solve


def mpi_solve_1group(prob, **kwargs):
    constrained_mpi_solve(prob, MpiPartition(ngroups=1), **kwargs)


solvers = [constrained_serial_solve]
if MPI is not None:
    solvers.append(mpi_solve_1group)


class TestFunc1(Optimizable):
    """
    Args:
        nparams: number of independent variables.
        nvals: number of dependent variables.
    """

    def __init__(self, nparams, nvals):
        self.nparams = nparams
        self.nvals = nvals
        self.A = (np.random.rand(nvals, nparams) - 0.5) * 4
        self.B = (np.random.rand(nvals) - 0.5) * 4
        super().__init__(np.zeros(nparams))

    def f(self):
        return np.sum(np.matmul(self.A, self.full_x)**2)

    return_fn_map = {'f': f}

    def f2(self):
        return np.sum(self.full_x**2)

    def c1(self):
        return np.matmul(self.A, self.full_x) + self.B

    def c2(self):
        return np.sum(self.full_x)


# "mpi" is included in the class name so the tests are discovered by
# run_tests_mpi.
class ConstrainedSolveTests_mpi(unittest.TestCase):

    def test_bound_constrained(self):
        grads = [True, False]

        # bound constraints with scalar bounds
        rosen = Rosenbrock()
        rosen.lower_bounds = np.zeros(len(rosen.x))
        rosen.upper_bounds = 5*np.ones(len(rosen.x))
        prob = ConstrainedProblem(rosen.f)
        options = {'ftol': 1e-9, 'maxiter': 2000}
        for solver in solvers:
            for grad in grads:
                prob.x = np.zeros(2)
                solver(prob, grad=grad, options=options)
                np.testing.assert_allclose(prob.x, np.ones(2), atol=1e-3)
                np.testing.assert_allclose(prob.objective(), 0.0, atol=1e-3)

        # bound constraints with vector bounds
        rosen = Rosenbrock()
        lb = np.array([-np.inf, 0.0])
        ub = [1.5, np.inf]
        rosen.lower_bounds = lb
        rosen.upper_bounds = ub
        prob = ConstrainedProblem(rosen.f)
        options = {'ftol': 1e-9, 'maxiter': 2000}
        for solver in solvers:
            for grad in grads:
                prob.x = np.zeros(2)
                solver(prob, grad=grad, options=options)
                np.testing.assert_allclose(prob.x, np.ones(2), atol=1e-3)
                np.testing.assert_allclose(prob.objective(), 0.0, atol=1e-3)

    def test_linear(self):
        grads = [True, False]

        rosen = Rosenbrock()
        A = np.atleast_2d(np.ones(2))
        b = np.array([10.0])
        lb = np.array([-np.inf, 0.0])
        ub = [1.5, np.inf]
        rosen.lower_bounds = lb
        rosen.upper_bounds = ub
        prob = ConstrainedProblem(rosen.f, tuple_lc=(A, -np.inf, b))
        options = {'ftol': 1e-9, 'maxiter': 2000}
        for solver in solvers:
            for grad in grads:
                prob.x = np.zeros(2)
                solver(prob, grad=grad, options=options)
                np.testing.assert_allclose(prob.x, np.ones(2), atol=1e-3)
                np.testing.assert_allclose(prob.objective(), 0.0, atol=1e-3)

        # QP program
        tester = TestFunc1(2, 2)
        A = np.atleast_2d(-np.ones(2))
        b = np.array([-.5])
        tester.lower_bounds = np.zeros(len(tester.x))
        prob = ConstrainedProblem(tester.f2, tuple_lc=(A, -np.inf, b))
        prob.x = np.ones(2)
        options = {'ftol': 1e-9, 'maxiter': 2000}
        for solver in solvers:
            for grad in grads:
                prob.x = np.zeros(2)
                solver(prob, grad=grad, options=options)
                np.testing.assert_allclose(prob.x, 0.25*np.ones(2), atol=1e-3)
                np.testing.assert_allclose(prob.objective(), 0.125, atol=1e-3)

    def test_nlc(self):
        grads = [True, False]

        # QP
        tester = TestFunc1(2, 2)
        prob = ConstrainedProblem(tester.f, tuples_nlc=[(tester.c2, 0.0, np.inf)])
        options = {'ftol': 1e-9, 'maxiter': 2000}
        for solver in solvers:
            for grad in grads:
                prob.x = np.ones(2)
                solver(prob, grad=grad, options=options)
                np.testing.assert_allclose(prob.x, np.zeros(2), atol=1e-3)
                np.testing.assert_allclose(prob.objective(), 0.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
