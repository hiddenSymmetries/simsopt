import unittest

import numpy as np

from simsopt._core.optimizable import Target
from simsopt.objectives.new_functions import Identity, Rosenbrock
from simsopt.objectives.new_least_squares import LeastSquaresProblem
from simsopt.solve.new_serial import least_squares_serial_solve
from simsopt.util.mpi import MpiPartition
from simsopt.solve.new_mpi import least_squares_mpi_solve


def mpi_solve_1group(prob, **kwargs):
    least_squares_mpi_solve(prob, MpiPartition(ngroups=1), **kwargs)
    

solvers = [least_squares_serial_solve, mpi_solve_1group]


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
            term1 = (iden1.f, 1, 1)
            term2 = (iden2.f, 2, 2)
            term3 = (iden3.f, 3, 3)
            prob = LeastSquaresProblem.from_tuples([term1, term2, term3])
            solver(prob)
            self.assertAlmostEqual(prob.objective(), 0)
            self.assertTrue(np.allclose(iden1.x, [1]))
            self.assertTrue(np.allclose(iden2.x, [2]))
            self.assertTrue(np.allclose(iden3.x, [3]))


    def test_solve_quadratic_fixed(self):
        """
        Same as test_solve_quadratic, except with different weights and x
        and z are fixed, so only y is optimized.
        """
        for solver in solvers:
            iden1 = Identity(4, dof_name='x1', dof_fixed=True)
            iden2 = Identity(5, dof_name='x2')
            iden3 = Identity(6, dof_name='x3', dof_fixed=True)
            term1 = (iden1.f, 1, 1)
            term2 = (iden2.f, 2, 1 / 4.)
            term3 = (iden3.f, 3, 1 / 9.)
            prob = LeastSquaresProblem.from_tuples([term1, term2, term3])
            solver(prob)
            self.assertAlmostEqual(prob.objective(), 10)
            self.assertTrue(np.allclose(iden1.x, [4]))
            self.assertTrue(np.allclose(iden2.x, [2]))
            self.assertTrue(np.allclose(iden3.x, [6]))

    def test_solve_rosenbrock_using_scalars(self):
        """
        Minimize the Rosenbrock function using two separate least-squares
        terms.
        """
        for solver in solvers:
            #for grad in [True, False]:
                r = Rosenbrock()
                term1 = (r.term1, 0, 1)
                term2 = (r.term2, 0, 1)
                prob = LeastSquaresProblem.from_tuples((term1, term2))
                solver(prob) #, grad=grad)
                self.assertAlmostEqual(prob.objective(), 0)
                v = r.x
                self.assertTrue(np.allclose(v, [1, 1]))
                #self.assertAlmostEqual(v[0], 1)
                #self.assertAlmostEqual(v[1], 1)

    def test_solve_rosenbrock_using_vector(self):
        """
        Minimize the Rosenbrock function using a single vector-valued
        least-squares term.
        """
        for solver in solvers:
            #for grad in [True, False]:
                r = Rosenbrock()
                prob = LeastSquaresProblem.from_tuples([(r.terms, 0, 1)])
                solver(prob) #, grad=grad)
                self.assertAlmostEqual(prob.objective(), 0)
                v = r.x
                self.assertTrue(np.allclose(v, [1, 1]))
                # self.assertAlmostEqual(v[0], 1)
                # self.assertAlmostEqual(v[1], 1)

if __name__ == "__main__":
    unittest.main()
