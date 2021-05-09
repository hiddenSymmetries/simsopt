import unittest
import numpy as np
from scipy.optimize import minimize

from simsopt.objectives.functions import Identity, Rosenbrock, \
    Failer, RosenbrockWithFailures
from simsopt.objectives.least_squares import LeastSquaresProblem, \
    LeastSquaresTerm
from simsopt._core.optimizable import Target


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

        lst = LeastSquaresTerm.from_sigma(iden.J, 3, sigma=0.1)
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
        # Note here we supply iden instead of iden.J
        lst = LeastSquaresTerm.from_sigma(iden, 3, sigma=0.1)
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
        lst = LeastSquaresTerm.from_sigma(Target(iden, 'f'), 3, sigma=0.1)
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
            lst = LeastSquaresTerm.from_sigma(iden.J, 3, sigma=iden)
        #with self.assertRaises(TypeError):
        #    lst = LeastSquaresTerm(iden.J, "hello", weight=0.1)
        with self.assertRaises(TypeError):
            lst = LeastSquaresTerm(iden.J, 3, weight=iden)

        # sigma cannot be zero
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm.from_sigma(iden.J, 3, sigma=0)
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm.from_sigma(iden.J, 3, sigma=0.0)

        # Cannot specify both weight and sigma
        with self.assertRaises(TypeError):
            lst = LeastSquaresTerm(iden.J, 3, sigma=1.2, weight=3.4)
        # Must specify either weight or sigma
        with self.assertRaises(TypeError):
            lst = LeastSquaresTerm(iden.J, 3)

        # Weight cannot be negative
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.J, 3, weight=-1.0)


class LeastSquaresProblemTests(unittest.TestCase):

    def test_supply_LeastSquaresTerm(self):
        """
        Test basic usage
        """
        # Objective function f(x) = ((x - 3) / 2) ** 2
        iden1 = Identity()
        term1 = LeastSquaresTerm.from_sigma(iden1.J, 3, sigma=2)
        prob = LeastSquaresProblem([term1])
        self.assertAlmostEqual(prob.objective(), 2.25)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertEqual(len(prob.dofs.f()), 1)
        self.assertAlmostEqual(prob.dofs.f()[0], 0)
        self.assertEqual(len(prob.f()), 1)
        self.assertAlmostEqual(prob.f()[0], -1.5)
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 2.25)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 2.25)
        iden1.set_dofs([10])
        self.assertAlmostEqual(prob.objective(), 12.25)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 12.25)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 12.25)
        self.assertAlmostEqual(prob.objective([0]), 2.25)
        self.assertAlmostEqual(prob.objective([10]), 12.25)
        self.assertEqual(prob.dofs.all_owners, [iden1])
        self.assertEqual(prob.dofs.dof_owners, [iden1])

        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        iden2 = Identity()
        term2 = LeastSquaresTerm.from_sigma(iden2.J, -4, sigma=5)
        prob = LeastSquaresProblem([term1, term2])
        self.assertAlmostEqual(prob.objective(), 12.89)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertEqual(len(prob.f()), 2)
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 12.89)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 12.89)
        iden1.set_dofs([5])
        iden2.set_dofs([-7])
        self.assertAlmostEqual(prob.objective(), 1.36)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertEqual(len(prob.f()), 2)
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 1.36)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 1.36)
        self.assertAlmostEqual(prob.objective([10, 0]), 12.89)
        self.assertAlmostEqual(prob.objective([5, -7]), 1.36)
        self.assertEqual(prob.dofs.dof_owners, [iden1, iden2])
        self.assertEqual(prob.dofs.all_owners, [iden1, iden2])

    def test_supply_tuples(self):
        """
        Test basic usage
        """
        # Objective function f(x) = ((x - 3) / 2) ** 2
        iden1 = Identity()
        term1 = (iden1.J, 3, 0.25)
        prob = LeastSquaresProblem([term1])
        self.assertAlmostEqual(prob.objective(), 2.25)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertEqual(len(prob.dofs.f()), 1)
        self.assertAlmostEqual(prob.dofs.f()[0], 0)
        self.assertEqual(len(prob.f()), 1)
        self.assertAlmostEqual(prob.f()[0], -1.5)
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 2.25)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 2.25)
        iden1.set_dofs([10])
        self.assertAlmostEqual(prob.objective(), 12.25)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 12.25)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 12.25)
        self.assertAlmostEqual(prob.objective([0]), 2.25)
        self.assertAlmostEqual(prob.objective([10]), 12.25)
        self.assertEqual(prob.dofs.all_owners, [iden1])
        self.assertEqual(prob.dofs.dof_owners, [iden1])

        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        iden2 = Identity()
        term2 = (iden2.J, -4, 0.04)
        prob = LeastSquaresProblem([term1, term2])
        self.assertAlmostEqual(prob.objective(), 12.89)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertEqual(len(prob.f()), 2)
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 12.89)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 12.89)
        iden1.set_dofs([5])
        iden2.set_dofs([-7])
        self.assertAlmostEqual(prob.objective(), 1.36)
        self.assertAlmostEqual(prob.objective(), sum(t.f_out() for t in prob.terms))
        self.assertEqual(len(prob.f()), 2)
        self.assertAlmostEqual(prob.objective_from_shifted_f(prob.f()), 1.36)
        self.assertAlmostEqual(prob.objective_from_unshifted_f(prob.dofs.f()), 1.36)
        self.assertAlmostEqual(prob.objective([10, 0]), 12.89)
        self.assertAlmostEqual(prob.objective([5, -7]), 1.36)
        self.assertEqual(prob.dofs.dof_owners, [iden1, iden2])
        self.assertEqual(prob.dofs.all_owners, [iden1, iden2])

    def test_exceptions(self):
        """
        Verify that exceptions are raised when invalid inputs are
        provided.
        """
        # Argument must be a list in which each element is a
        # LeastSquaresTerm or a 3- or 4-element tuple/list.
        with self.assertRaises(TypeError):
            prob = LeastSquaresProblem(7)
        with self.assertRaises(ValueError):
            prob = LeastSquaresProblem([])
        with self.assertRaises(TypeError):
            prob = LeastSquaresProblem([7, 1])

    def test_failures(self):
        """
        Verify that the expected residuals are returned in cases where the
        objective function evaluations fail.
        """
        o1 = Failer()
        r1 = Rosenbrock()
        fail_val = 1.0e6
        prob1 = LeastSquaresProblem([(r1.terms, 0, 1), (o1, 0, 1)],
                                    fail=fail_val)
        # First evaluation should not fail.
        f = prob1.f()
        print(f)
        np.testing.assert_allclose(f, [-1, 0, 1, 1, 1])
        # Second evaluation should fail.
        f = prob1.f()
        print(f)
        np.testing.assert_allclose(f, np.full(5, fail_val))
        # Third evaluation should not fail.
        f = prob1.f()
        print(f)
        np.testing.assert_allclose(f, [-1, 0, 1, 1, 1])

    def test_outside_optimizer(self):
        """
        Verify that a least-squares problem can be passed to an outside
        optimization package to be solved, even when failures occur
        during evaluations.
        """
        ros = RosenbrockWithFailures(fail_interval=20)
        fail_val = 1.0e2
        prob = LeastSquaresProblem([(ros.terms, 0, 1)], fail=fail_val)
        # Just call the bare scipy.optimize.minimize function. This is
        # not really an "outside" package, but it is similar to how we
        # might want to call outside optimization libraries that are
        # completely separate from simsopt.
        result = minimize(prob.objective, prob.x)
        # Need a large tolerance, since the failures confuse the
        # optimizer
        np.testing.assert_allclose(result.x, [1, 1], atol=1e-2)


if __name__ == "__main__":
    unittest.main()
