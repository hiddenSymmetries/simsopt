import unittest
import logging
from simsopt.core.functions import Identity, Rosenbrock
from simsopt.core.optimizable import Target
from simsopt.core.least_squares_problem import LeastSquaresProblem, LeastSquaresTerm

#logging.basicConfig(level=logging.DEBUG)

class LeastSquaresTermTests(unittest.TestCase):

    def test_basic(self):
        """
        Test basic usage
        """
        iden = Identity()
        lst = LeastSquaresTerm(iden.J, 3, weight=0.1)
        self.assertEqual(lst.f_in, iden.J)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.weight, 0.1, places=13)

        lst = LeastSquaresTerm(iden.J, 3, sigma=0.1)
        self.assertEqual(lst.f_in, iden.J)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.weight, 100.0, places=13)
        
        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.f_out(), correct_value, places=11)

    def test_supply_object(self):
        """
        Test that we can supply an object with a J function instead of a function.
        """
        iden = Identity()
        lst = LeastSquaresTerm(iden, 3, sigma=0.1) # Note here we supply iden instead of iden.J
        self.assertEqual(lst.f_in, iden.J)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.weight, 100, places=13)

        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.f_out(), correct_value, places=11)

    def test_supply_property(self):
        """
        Test that we can supply a property instead of a function.
        """
        iden = Identity()
        lst = LeastSquaresTerm(Target(iden, 'f'), 3, sigma=0.1)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.weight, 100, places=13)

        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.f_out(), correct_value, places=11)

    def test_supply_attribute(self):
        """
        Test that we can supply an attribute instead of a function.
        """
        iden = Identity()
        lst = LeastSquaresTerm(Target(iden, 'x'), 3, weight=0.1)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.weight, 0.1, places=13)

        iden.set_dofs([17])
        self.assertEqual(lst.f_in(), 17)
        correct_value = 0.1 * ((17 - 3) ** 2)
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
        #with self.assertRaises(TypeError):
        #    lst = LeastSquaresTerm(iden.J, "hello", sigma=0.1)
        with self.assertRaises(TypeError):
            lst = LeastSquaresTerm(iden.J, 3, sigma=iden)
        #with self.assertRaises(TypeError):
        #    lst = LeastSquaresTerm(iden.J, "hello", weight=0.1)
        with self.assertRaises(TypeError):
            lst = LeastSquaresTerm(iden.J, 3, weight=iden)

        # sigma cannot be zero
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, sigma=0)
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, sigma=0.0)

        # Cannot specify both weight and sigma
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, sigma=1.2, weight=3.4)
        # Must specify either weight or sigma
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3)

        # Weight cannot be negative
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, weight=-1.0)
        
class LeastSquaresProblemTests(unittest.TestCase):

    def test_basic(self):
        """
        Test basic usage
        """
        # Objective function f(x) = ((x - 3) / 2) ** 2
        iden1 = Identity()
        term1 = LeastSquaresTerm(iden1.J, 3, sigma=2)
        prob = LeastSquaresProblem([term1])
        self.assertAlmostEqual(prob.objective(), 2.25)
        iden1.set_dofs([10])
        self.assertAlmostEqual(prob.objective(), 12.25)
        self.assertAlmostEqual(prob.objective([0]), 2.25)
        self.assertAlmostEqual(prob.objective([10]), 12.25)
        self.assertEqual(prob.dofs.all_owners, [iden1])
        self.assertEqual(prob.dofs.dof_owners, [iden1])

        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        iden2 = Identity()
        term2 = LeastSquaresTerm(iden2.J, -4, sigma=5)
        prob = LeastSquaresProblem([term1, term2])
        self.assertAlmostEqual(prob.objective(), 12.89)
        iden1.set_dofs([5])
        iden2.set_dofs([-7])
        self.assertAlmostEqual(prob.objective(), 1.36)
        self.assertAlmostEqual(prob.objective([10, 0]), 12.89)
        self.assertAlmostEqual(prob.objective([5, -7]), 1.36)
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
        term1 = LeastSquaresTerm(iden1.J, 1, sigma=1)
        term2 = LeastSquaresTerm(iden2.J, 2, sigma=2)
        term3 = LeastSquaresTerm(iden3.J, 3, sigma=3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective(), 0)
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
        term1 = LeastSquaresTerm(iden1.J, 1, sigma=1)
        term2 = LeastSquaresTerm(iden2.J, 2, sigma=2)
        term3 = LeastSquaresTerm(iden3.J, 3, sigma=3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective(), 10)
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
        term1 = LeastSquaresTerm(iden1, 1, sigma=1)
        term2 = LeastSquaresTerm(iden2, 2, sigma=2)
        term3 = LeastSquaresTerm(iden3, 3, sigma=3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective(), 10)
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
        term1 = LeastSquaresTerm(Target(iden1, 'x'), 1, sigma=1)
        term2 = LeastSquaresTerm(Target(iden2, 'x'), 2, sigma=2)
        term3 = LeastSquaresTerm(Target(iden3, 'x'), 3, sigma=3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective(), 10)
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
        term1 = LeastSquaresTerm(Target(iden1, 'f'), 1, sigma=1)
        term2 = LeastSquaresTerm(Target(iden2, 'f'), 2, sigma=2)
        term3 = LeastSquaresTerm(Target(iden3, 'f'), 3, sigma=3)
        prob = LeastSquaresProblem([term1, term2, term3])
        prob.solve()
        self.assertAlmostEqual(prob.objective(), 10)
        self.assertAlmostEqual(iden1.x, 4)
        self.assertAlmostEqual(iden2.x, 2)
        self.assertAlmostEqual(iden3.x, 6)

    def test_solve_rosenbrock_using_scalars(self):
        """
        Minimize the Rosenbrock function using two separate least-squares
        terms.
        """
        for grad in [True, False]:
            r = Rosenbrock()
            term1 = LeastSquaresTerm(r.term1, 0, sigma=1)
            term2 = LeastSquaresTerm(r.term2, 0, sigma=1)
            prob = LeastSquaresProblem([term1, term2])
            prob.solve(grad=grad)
            self.assertAlmostEqual(prob.objective(), 0)
            v = r.get_dofs()
            self.assertAlmostEqual(v[0], 1)
            self.assertAlmostEqual(v[1], 1)

    def test_solve_rosenbrock_using_vector(self):
        """
        Minimize the Rosenbrock function using a single vector-valued
        least-squares term.
        """
        for grad in [True, False]:
            r = Rosenbrock()
            term1 = LeastSquaresTerm(r.terms, 0, weight=1)
            prob = LeastSquaresProblem([term1])
            prob.solve(grad=grad)
            self.assertAlmostEqual(prob.objective(), 0)
            v = r.get_dofs()
            self.assertAlmostEqual(v[0], 1)
            self.assertAlmostEqual(v[1], 1)

if __name__ == "__main__":
    unittest.main()
