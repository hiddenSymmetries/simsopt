import unittest
import logging
from simsopt.core.functions import Identity, Rosenbrock
from simsopt.core.optimizable import Target
from simsopt.core.least_squares_problem import LeastSquaresProblem, LeastSquaresTerm
from simsopt.core.serial_solve import least_squares_serial_solve
from simsopt.core.mpi import MpiPartition
from simsopt.core.mpi_solve import least_squares_mpi_solve

def mpi_solve_1group(prob, **kwargs):
    least_squares_mpi_solve(prob, MpiPartition(ngroups=1), **kwargs)
    
solvers = [least_squares_serial_solve, mpi_solve_1group]

#logging.basicConfig(level=logging.DEBUG)

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
                solver(prob, grad=grad)
                self.assertAlmostEqual(prob.objective(), 0)
                v = r.get_dofs()
                self.assertAlmostEqual(v[0], 1)
                self.assertAlmostEqual(v[1], 1)

if __name__ == "__main__":
    unittest.main()
