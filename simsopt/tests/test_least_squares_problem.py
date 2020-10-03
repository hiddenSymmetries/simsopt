import unittest
import logging
from simsopt.functions import Identity, Rosenbrock
from simsopt.optimizable import Target
from simsopt.least_squares_problem import LeastSquaresProblem, LeastSquaresTerm

#logging.basicConfig(level=logging.DEBUG)

class LeastSquaresTermTests(unittest.TestCase):

    def test_basic(self):
        """
        Test basic usage
        """
        iden = Identity()
        lst = LeastSquaresTerm(iden.J, 3, 0.1)
        self.assertEqual(lst.f_in, iden.J)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.sigma, 0.1, places=13)

        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.f_out(), correct_value, places=13)

    def test_supply_object(self):
        """
        Test that we can supply an object with a J function instead of a function.
        """
        iden = Identity()
        lst = LeastSquaresTerm(iden, 3, 0.1) # Note here we supply iden instead of iden.J
        self.assertEqual(lst.f_in, iden.J)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.sigma, 0.1, places=13)

        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.f_out(), correct_value, places=13)

    def test_supply_property(self):
        """
        Test that we can supply a property instead of a function.
        """
        iden = Identity()
        lst = LeastSquaresTerm(Target(iden, 'f'), 3, 0.1)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.sigma, 0.1, places=13)

        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.f_out(), correct_value, places=13)

    def test_supply_attribute(self):
        """
        Test that we can supply an attribute instead of a function.
        """
        iden = Identity()
        lst = LeastSquaresTerm(Target(iden, 'x'), 3, 0.1)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.sigma, 0.1, places=13)

        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.f_out(), correct_value, places=13)

    def test_exceptions(self):
        """
        Test that exceptions are thrown when invalid inputs are
        provided.
        """
        # First argument must be callable
        with self.assertRaises(TypeError):
            lst = LeastSquaresTerm(2, 3, 0.1)

        # Second and third arguments must be real numbers
        iden = Identity()
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, "hello", 0.1)
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, iden)

        # sigma cannot be zero
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, 0)
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, 0.0)

            
class LeastSquaresProblemTests(unittest.TestCase):

    def test_basic(self):
        """
        Test basic usage
        """
        # Objective function f(x) = ((x - 3) / 2) ** 2
        iden1 = Identity()
        term1 = LeastSquaresTerm(iden1.J, 3, 2)
        prob = LeastSquaresProblem([term1])
        self.assertAlmostEqual(prob.objective, 2.25)
        iden1.set_dofs([10])
        self.assertAlmostEqual(prob.objective, 12.25)
        self.assertEqual(prob.dofs.all_owners, [iden1])
        self.assertEqual(prob.dofs.dof_owners, [iden1])

        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        iden2 = Identity()
        term2 = LeastSquaresTerm(iden2.J, -4, 5)
        prob = LeastSquaresProblem([term1, term2])
        self.assertAlmostEqual(prob.objective, 12.89)
        iden1.set_dofs([5])
        iden2.set_dofs([-7])
        self.assertAlmostEqual(prob.objective, 1.36)
        self.assertEqual(prob.dofs.dof_owners, [iden1, iden2])
        self.assertEqual(prob.dofs.all_owners, [iden1, iden2])

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
        term1 = LeastSquaresTerm(iden1.J, 1, 1)
        term2 = LeastSquaresTerm(iden2.J, 2, 2)
        term3 = LeastSquaresTerm(iden3.J, 3, 3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 0)
        self.assertAlmostEqual(iden1.x, 1)
        self.assertAlmostEqual(iden2.x, 2)
        self.assertAlmostEqual(iden3.x, 3)

    def test_solve_quadratic_fixed(self):
        """
        Same as test_solve_quadratic, except x and z are fixed, so
        only y is optimized.
        """
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
        term1 = LeastSquaresTerm(iden1.J, 1, 1)
        term2 = LeastSquaresTerm(iden2.J, 2, 2)
        term3 = LeastSquaresTerm(iden3.J, 3, 3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 10)
        self.assertAlmostEqual(iden1.x, 4)
        self.assertAlmostEqual(iden2.x, 2)
        self.assertAlmostEqual(iden3.x, 6)

    def test_solve_quadratic_fixed_supplying_objects(self):
        """
        Same as test_solve_quadratic_fixed, except supplying objects
        rather than functions as targets.
        """
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
        term1 = LeastSquaresTerm(iden1, 1, 1)
        term2 = LeastSquaresTerm(iden2, 2, 2)
        term3 = LeastSquaresTerm(iden3, 3, 3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 10)
        self.assertAlmostEqual(iden1.x, 4)
        self.assertAlmostEqual(iden2.x, 2)
        self.assertAlmostEqual(iden3.x, 6)

    def test_solve_quadratic_fixed_supplying_attributes(self):
        """
        Same as test_solve_quadratic_fixed, except supplying attributes
        rather than functions as targets.
        """
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
        term1 = LeastSquaresTerm(Target(iden1, 'x'), 1, 1)
        term2 = LeastSquaresTerm(Target(iden2, 'x'), 2, 2)
        term3 = LeastSquaresTerm(Target(iden3, 'x'), 3, 3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 10)
        self.assertAlmostEqual(iden1.x, 4)
        self.assertAlmostEqual(iden2.x, 2)
        self.assertAlmostEqual(iden3.x, 6)

    def test_solve_quadratic_fixed_supplying_properties(self):
        """
        Same as test_solve_quadratic_fixed, except supplying @properties
        rather than functions as targets.
        """
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
        term1 = LeastSquaresTerm(Target(iden1, 'f'), 1, 1)
        term2 = LeastSquaresTerm(Target(iden2, 'f'), 2, 2)
        term3 = LeastSquaresTerm(Target(iden3, 'f'), 3, 3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 10)
        self.assertAlmostEqual(iden1.x, 4)
        self.assertAlmostEqual(iden2.x, 2)
        self.assertAlmostEqual(iden3.x, 6)

    def test_solve_rosenbrock(self):
        """
        Minimize the Rosenbrock function.
        """
        r = Rosenbrock()
        term1 = LeastSquaresTerm(r.term1, 0, 1)
        term2 = LeastSquaresTerm(r.term2, 0, 1)
        prob = LeastSquaresProblem([term1, term2])
        prob.solve()
        self.assertAlmostEqual(prob.objective, 0)
        v = r.get_dofs()
        self.assertAlmostEqual(v[0], 1)
        self.assertAlmostEqual(v[1], 1)

if __name__ == "__main__":
    unittest.main()
