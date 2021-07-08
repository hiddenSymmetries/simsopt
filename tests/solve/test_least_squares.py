import unittest
import logging

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from simsopt.objectives.functions import Identity, Rosenbrock, Beale
from simsopt._core.optimizable import Target
from simsopt.objectives.least_squares import LeastSquaresProblem, \
    LeastSquaresTerm
from simsopt.solve.serial import least_squares_serial_solve, serial_solve
if MPI is not None:
    from simsopt.util.mpi import MpiPartition
    from simsopt.solve.mpi import least_squares_mpi_solve


def mpi_solve_1group(prob, **kwargs):
    least_squares_mpi_solve(prob, MpiPartition(ngroups=1), **kwargs)


solvers = [serial_solve, least_squares_serial_solve]
if MPI is not None:
    solvers.append(mpi_solve_1group)

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LeastSquaresProblemTests(unittest.TestCase):

    def test_solve_quadratic(self):
        """
        Minimize f(x,y,z) = 1 * (x - 1) ^ 2 + 2 * (y - 2) ^ 2 + 3 * (z - 3) ^ 2.
        The optimum is at (x,y,z)=(1,2,3), and f=0 at this point.
        """
        for solver in solvers:
            iden1 = Identity()
            iden2 = Identity()
            iden3 = Identity()
            term1 = (iden1.J, 1, 1)
            term2 = (iden2.J, 2, 2)
            term3 = (iden3.J, 3, 3)
            prob = LeastSquaresProblem([term1, term2, term3])
            if solver == serial_solve:
                solver(prob, tol=1e-12)
            else:
                solver(prob)
            self.assertAlmostEqual(prob.objective(), 0)
            self.assertAlmostEqual(iden1.x, 1)
            self.assertAlmostEqual(iden2.x, 2)
            self.assertAlmostEqual(iden3.x, 3)

    def test_solve_quadratic_fixed(self):
        """
        Same as test_solve_quadratic, except with different weights and x
        and z are fixed, so only y is optimized.
        """
        for solver in solvers:
            iden1 = Identity()
            iden2 = Identity()
            iden3 = Identity()
            iden1.x = 4
            iden2.x = 5
            iden3.x = 6
            iden1.names = ['x1']
            iden2.names = ['x2']
            iden3.names = ['x3']
            iden1.fixed = [True]
            iden3.fixed = [True]
            term1 = (iden1.J, 1, 1)
            term2 = (iden2.J, 2, 1 / 4.)
            term3 = (iden3.J, 3, 1 / 9.)
            prob = LeastSquaresProblem([term1, term2, term3])
            solver(prob)
            self.assertAlmostEqual(prob.objective(), 10)
            self.assertAlmostEqual(iden1.x, 4)
            self.assertAlmostEqual(iden2.x, 2)
            self.assertAlmostEqual(iden3.x, 6)

    def test_solve_quadratic_fixed_supplying_objects(self):
        """
        Same as test_solve_quadratic_fixed, except supplying objects
        rather than functions as targets.
        """
        for solver in solvers:
            iden1 = Identity()
            iden2 = Identity()
            iden3 = Identity()
            iden1.x = 4
            iden2.x = 5
            iden3.x = 6
            iden1.names = ['x1']
            iden2.names = ['x2']
            iden3.names = ['x3']
            iden1.fixed = [True]
            iden3.fixed = [True]
            term1 = [iden1, 1, 1]
            term2 = [iden2, 2, 1 / 4.]
            term3 = [iden3, 3, 1 / 9.]
            prob = LeastSquaresProblem([term1, term2, term3])
            solver(prob)
            self.assertAlmostEqual(prob.objective(), 10)
            self.assertAlmostEqual(iden1.x, 4)
            self.assertAlmostEqual(iden2.x, 2)
            self.assertAlmostEqual(iden3.x, 6)

    def test_solve_quadratic_fixed_supplying_attributes(self):
        """
        Same as test_solve_quadratic_fixed, except supplying attributes
        rather than functions as targets.
        """
        for solver in solvers:
            iden1 = Identity()
            iden2 = Identity()
            iden3 = Identity()
            iden1.x = 4
            iden2.x = 5
            iden3.x = 6
            iden1.names = ['x1']
            iden2.names = ['x2']
            iden3.names = ['x3']
            iden1.fixed = [True]
            iden3.fixed = [True]
            # Try a mix of explicit LeastSquaresTerms and tuples
            term1 = LeastSquaresTerm(Target(iden1, 'x'), 1, 1)
            term2 = (iden2, 'x', 2, 1 / 4.)
            term3 = (iden3, 'x', 3, 1 / 9.)
            prob = LeastSquaresProblem([term1, term2, term3])
            solver(prob)
            self.assertAlmostEqual(prob.objective(), 10)
            self.assertAlmostEqual(iden1.x, 4)
            self.assertAlmostEqual(iden2.x, 2)
            self.assertAlmostEqual(iden3.x, 6)

    def test_solve_quadratic_fixed_supplying_properties(self):
        """
        Same as test_solve_quadratic_fixed, except supplying @properties
        rather than functions as targets.
        """
        for solver in solvers:
            iden1 = Identity()
            iden2 = Identity()
            iden3 = Identity()
            iden1.x = 4
            iden2.x = 5
            iden3.x = 6
            iden1.names = ['x1']
            iden2.names = ['x2']
            iden3.names = ['x3']
            iden1.fixed = [True]
            iden3.fixed = [True]
            # Try a mix of explicit LeastSquaresTerms and lists
            term1 = [iden1, 'f', 1, 1]
            term2 = [iden2, 'f', 2, 1 / 4.]
            term3 = LeastSquaresTerm.from_sigma(Target(iden3, 'f'), 3, sigma=3)
            prob = LeastSquaresProblem([term1, term2, term3])
            solver(prob)
            self.assertAlmostEqual(prob.objective(), 10)
            self.assertAlmostEqual(iden1.x, 4)
            self.assertAlmostEqual(iden2.x, 2)
            self.assertAlmostEqual(iden3.x, 6)

    def test_solve_rosenbrock_using_scalars(self):
        """
        Minimize the Rosenbrock function using two separate least-squares
        terms.
        """
        for solver in solvers:
            for grad in [True, False]:
                r = Rosenbrock()
                term1 = (r.term1, 0, 1)
                term2 = (r.term2, 0, 1)
                prob = LeastSquaresProblem((term1, term2))
                if solver == serial_solve:
                    if grad == True:
                        continue
                    else:
                        solver(prob, tol=1e-12)
                else:
                    solver(prob, grad=grad)
                self.assertAlmostEqual(prob.objective(), 0)
                v = r.get_dofs()
                self.assertAlmostEqual(v[0], 1)
                self.assertAlmostEqual(v[1], 1)

    def test_solve_rosenbrock_using_vector(self):
        """
        Minimize the Rosenbrock function using a single vector-valued
        least-squares term.
        """
        for solver in solvers:
            for grad in [True, False]:
                r = Rosenbrock()
                prob = LeastSquaresProblem([(r.terms, 0, 1)])
                if solver == serial_solve:
                    if grad == True:
                        continue
                    else:
                        solver(prob, tol=1e-12)
                else:
                    solver(prob, grad=grad)
                self.assertAlmostEqual(prob.objective(), 0)
                v = r.get_dofs()
                self.assertAlmostEqual(v[0], 1)
                self.assertAlmostEqual(v[1], 1)

    def test_solve_with_finite_differences(self):
        """
        Minimize a function for which analytic derivatives are not
        provided. Provides test coverage for the finite-differencing
        options.
        """
        #for solver in [least_squares_serial_solve]:
        for solver in solvers:
            if solver == serial_solve:
                continue
            for abs_step in [0, 1.0e-7]:
                rel_steps = [0, 1.0e-7]
                if abs_step == 0:
                    rel_steps = [1.0e-7]
                for rel_step in rel_steps:
                    for diff_method in ["forward", "centered"]:
                        logger.debug(f'solver={solver} diff_method={diff_method} ' \
                                     f'abs_step={abs_step} rel_step={rel_step}')
                        b = Beale()
                        b.set_dofs([0.1, -0.2])
                        prob = LeastSquaresProblem([(b, 0, 1)], diff_method=diff_method,
                                                   abs_step=abs_step, rel_step=rel_step)
                        #least_squares_serial_solve(prob, grad=True)
                        solver(prob, grad=True)
                        np.testing.assert_allclose(prob.x, [3, 0.5])
                        np.testing.assert_allclose(prob.f(), [0, 0, 0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
