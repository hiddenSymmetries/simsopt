import unittest
import logging
from simsopt.core.new_functions import Identity, Rosenbrock
#from simsopt.core.optimizable import Target
from simsopt.core.new_least_squares import LeastSquaresProblem

#logging.basicConfig(level=logging.DEBUG)

class LeastSquaresProblemTests(unittest.TestCase):
    def test_single_value_func_in(self):
        iden = Identity()
        lst = LeastSquaresProblem.from_sigma(iden, 3, 0.1)

        iden.x = [17]
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst(), correct_value, places=11)

        iden.x = [0]
        term1 = LeastSquaresProblem.from_sigma(iden, 3, 2)
        self.assertAlmostEqual(term1(), 2.25)

        term1.x = [10]
        self.assertAlmostEqual(term1(), 12.25)
        self.assertAlmostEqual(term1(x=[0]), 2.25)
        self.assertAlmostEqual(term1(x=[10]), 12.25)

    def test_exceptions(self):
        """
        Test that exceptions are thrown when invalid inputs are
        provided.
        """
        iden = Identity()

        # sigma cannot be zero
        with self.assertRaises(ValueError):
            lst = LeastSquaresProblem.from_sigma(iden, 3, 0)

        # Weight cannot be negative
        with self.assertRaises(ValueError):
            lst = LeastSquaresProblem(iden, 3, -1.0)

    def test_multiple_funcs_single_input(self):
        iden1 = Identity(x=10)
        iden2 = Identity()
        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        term = LeastSquaresProblem.from_sigma([iden1, iden2], [3, -4], [2, 5])
        self.assertAlmostEqual(term(), 12.89)
        term.x = [5, -7]
        self.assertAlmostEqual(term(), 1.36)
        self.assertAlmostEqual(term([10, 0]), 12.89)
        self.assertAlmostEqual(term([5, -7]), 1.36)

    def test_parent_dof_transitive_behavior(self):
        iden1 = Identity()
        iden2 = Identity()
        term = LeastSquaresProblem.from_sigma([iden1, iden2], [3, -4], [2, 5])
        iden1.x = [10]
        self.assertAlmostEqual(term(), 12.89)

    def test_least_squares_combination(self):
        iden1 = Identity()
        iden2 = Identity()
        term1 = LeastSquaresProblem.from_sigma(iden1, 3, 2)
        term2 = LeastSquaresProblem.from_sigma(iden2, -4, 5)
        term = term1 + term2
        iden1.x = [10]
        self.assertAlmostEqual(term(), 12.89)


if __name__ == "__main__":
    unittest.main()
