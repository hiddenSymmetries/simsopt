import unittest
from simsopt.target import Identity
from simsopt.least_squares_term import LeastSquaresTerm
from simsopt.least_squares_problem import LeastSquaresProblem
from simsopt.rosenbrock import Rosenbrock

class LeastSquaresProblemTests(unittest.TestCase):

    def test_basic(self):
        """
        Test basic usage
        """
        # Objective function f(x) = ((x - 3) / 2) ** 2
        iden1 = Identity()
        term1 = LeastSquaresTerm(iden1.target, 3, 2)
        prob = LeastSquaresProblem([term1])
        self.assertAlmostEqual(prob.objective, 2.25)
        iden1.x.val = 10
        self.assertAlmostEqual(prob.objective, 12.25)
        self.assertEqual(prob.parameters, [iden1.x])

        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        iden2 = Identity()
        term2 = LeastSquaresTerm(iden2.target, -4, 5)
        prob = LeastSquaresProblem([term1, term2])
        self.assertAlmostEqual(prob.objective, 12.89)
        iden1.x.val = 5
        iden2.x.val = -7
        self.assertAlmostEqual(prob.objective, 1.36)
        #print("prob.parameters:",prob.parameters)
        #print("[iden1.x, iden2.x]:",[iden1.x, iden2.x])
        self.assertEqual(set(prob.parameters), {iden1.x, iden2.x})

    def test_exceptions(self):
        """
        Verify that exceptions are raised when invalid inputs are
        provided.
        """
        # Argument must be a list of LeastSquaresTerms
        with self.assertRaises(ValueError):
            prob = LeastSquaresProblem(7)
        with self.assertRaises(ValueError):
            prob = LeastSquaresProblem([])
        with self.assertRaises(ValueError):
            prob = LeastSquaresProblem([7, 1])

    def test_solve_quadratic(self):
        """
        Minimize f(x,y,z) = ((x-1)/1)^2 + ((y-2)/2)^2 + ((z-3)/3)^2.
        The optimum is at (x,y,z)=(1,2,3), and f=0 at this point.
        """
        iden1 = Identity()
        iden2 = Identity()
        iden3 = Identity()
        iden1.x.fixed = False
        iden2.x.fixed = False
        iden3.x.fixed = False
        term1 = LeastSquaresTerm(iden1.target, 1, 1)
        term2 = LeastSquaresTerm(iden2.target, 2, 2)
        term3 = LeastSquaresTerm(iden3.target, 3, 3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 0)
        self.assertAlmostEqual(iden1.x.val, 1)
        self.assertAlmostEqual(iden2.x.val, 2)
        self.assertAlmostEqual(iden3.x.val, 3)

    def test_solve_quadratic_fixed(self):
        """
        Same as test_solve_quadratic, except x and z are fixed, so
        only y is optimized.
        """
        iden1 = Identity()
        iden2 = Identity()
        iden3 = Identity()
        iden1.x.val = 4
        iden2.x.val = 5
        iden3.x.val = 6
        iden1.x.name = 'x1'
        iden2.x.name = 'x2'
        iden3.x.name = 'x3'
        iden2.x.fixed = False
        term1 = LeastSquaresTerm(iden1.target, 1, 1)
        term2 = LeastSquaresTerm(iden2.target, 2, 2)
        term3 = LeastSquaresTerm(iden3.target, 3, 3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 10)
        self.assertAlmostEqual(iden1.x.val, 4)
        self.assertAlmostEqual(iden2.x.val, 2)
        self.assertAlmostEqual(iden3.x.val, 6)

    def test_solve_rosenbrock(self):
        """
        Minimize the Rosenbrock function.
        """
        r = Rosenbrock()
        r.x1.fixed = False
        r.x2.fixed = False
        term1 = LeastSquaresTerm(r.target1, 0, 1)
        term2 = LeastSquaresTerm(r.target2, 0, 1)
        prob = LeastSquaresProblem([term1, term2])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 0)
        self.assertAlmostEqual(r.x1.val, 1)
        self.assertAlmostEqual(r.x2.val, 1)

if __name__ == "__main__":
    unittest.main()
