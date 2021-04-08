import unittest
import logging
import numpy as np
from simsopt.core.new_functions import Identity, Rosenbrock
#from simsopt.core.optimizable import Target
from simsopt.core.new_least_squares import LeastSquaresProblem

#logging.basicConfig(level=logging.DEBUG)

class LeastSquaresProblemTests(unittest.TestCase):
    def test_single_value_opt_in(self):
        iden = Identity()
        lst = LeastSquaresProblem.from_sigma(iden, 3, 0.1)

        iden.x = [17]
        correct_value = ((17 - 3) / 0.1) #** 2
        self.assertAlmostEqual(np.abs(lst()[0]), correct_value, places=11)

        iden.x = [0]
        term1 = LeastSquaresProblem.from_sigma(iden, 3, 2)
        self.assertAlmostEqual(np.abs(term1()[0]), 1.5)

        term1.x = [10]
        self.assertAlmostEqual(np.abs(term1()[0]), 3.5)
        self.assertAlmostEqual(np.abs(term1(x=[0])), 1.5)
        self.assertAlmostEqual(np.abs(term1(x=[5])), 1)

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
        lsp = LeastSquaresProblem.from_sigma([iden1, iden2], [3, -4], [2, 5])
        self.assertAlmostEqual(np.abs(lsp()[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp()[1]), 0.8)
        lsp.x = [5, -7]
        self.assertAlmostEqual(np.abs(lsp()[0]), 1.0)
        self.assertAlmostEqual(np.abs(lsp()[1]), 0.6)
        self.assertAlmostEqual(np.abs(lsp([10, 0])[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp([10, 0])[1]), 0.8)
        self.assertAlmostEqual(np.abs(lsp([5, -7])[0]), 1.0)
        self.assertAlmostEqual(np.abs(lsp([5, -7])[1]), 0.6)

    def test_parent_dof_transitive_behavior(self):
        iden1 = Identity()
        iden2 = Identity()
        lsp = LeastSquaresProblem.from_sigma([iden1, iden2], [3, -4], [2, 5])
        iden1.x = [10]
        self.assertAlmostEqual(np.abs(lsp()[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp()[1]), 0.8)

    def test_least_squares_combination(self):
        iden1 = Identity()
        iden2 = Identity()
        term1 = LeastSquaresProblem.from_sigma(iden1, 3, 2)
        term2 = LeastSquaresProblem.from_sigma(iden2, -4, 5)
        lsp = term1 + term2
        iden1.x = [10]
        self.assertAlmostEqual(np.abs(lsp()[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp()[1]), 0.8)


if __name__ == "__main__":
    unittest.main()
