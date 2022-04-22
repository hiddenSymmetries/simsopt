import unittest
import logging
import numpy as np
from simsopt.objectives.functions import Identity, Rosenbrock
#from simsopt.core.optimizable import Target
from simsopt.objectives.least_squares import LeastSquaresProblem

#logging.basicConfig(level=logging.DEBUG)


class LeastSquaresProblemTests(unittest.TestCase):
    def test_single_value_opt_in(self):
        iden = Identity()
        lst = LeastSquaresProblem.from_sigma(3, 0.1, depends_on=iden)

        iden.x = [17]
        correct_value = ((17 - 3) / 0.1)  # ** 2
        self.assertAlmostEqual(np.abs(lst.residuals()[0]),
                               correct_value,
                               places=11)

        iden.x = [0]
        term1 = LeastSquaresProblem.from_sigma(3, 2, depends_on=iden)
        self.assertAlmostEqual(np.abs(term1.residuals()[0]), 1.5)

        term1.x = [10]
        self.assertAlmostEqual(np.abs(term1.residuals()[0]), 3.5)
        self.assertAlmostEqual(np.abs(term1.residuals(x=[0])), 1.5)
        self.assertAlmostEqual(np.abs(term1.residuals(x=[5])), 1)

    def test_exceptions(self):
        """
        Test that exceptions are thrown when invalid inputs are
        provided.
        """
        iden = Identity()

        # sigma cannot be zero
        with self.assertRaises(ValueError):
            lst = LeastSquaresProblem.from_sigma(3, 0, depends_on=iden)

        # Weight cannot be negative
        with self.assertRaises(ValueError):
            lst = LeastSquaresProblem(3, -1.0, depends_on=iden)

    def test_multiple_funcs_single_input(self):
        iden1 = Identity(x=10)
        iden2 = Identity()
        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        lsp = LeastSquaresProblem.from_sigma([3, -4], [2, 5], depends_on=[iden1, iden2])
        self.assertAlmostEqual(np.abs(lsp.residuals()[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp.residuals()[1]), 0.8)
        lsp.x = [5, -7]
        self.assertAlmostEqual(np.abs(lsp.residuals()[0]), 1.0)
        self.assertAlmostEqual(np.abs(lsp.residuals()[1]), 0.6)
        self.assertAlmostEqual(np.abs(lsp.residuals([10, 0])[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp.residuals([10, 0])[1]), 0.8)
        self.assertAlmostEqual(np.abs(lsp.residuals([5, -7])[0]), 1.0)
        self.assertAlmostEqual(np.abs(lsp.residuals([5, -7])[1]), 0.6)

    def test_parent_dof_transitive_behavior(self):
        iden1 = Identity()
        iden2 = Identity()
        lsp = LeastSquaresProblem.from_sigma([3, -4], [2, 5], depends_on=[iden1, iden2])
        iden1.x = [10]
        self.assertAlmostEqual(np.abs(lsp.residuals()[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp.residuals()[1]), 0.8)

    def test_least_squares_combination(self):
        iden1 = Identity()
        iden2 = Identity()
        term1 = LeastSquaresProblem.from_sigma(3, 2, depends_on=[iden1])
        term2 = LeastSquaresProblem.from_sigma(-4, 5, depends_on=[iden2])
        lsp = term1 + term2
        iden1.x = [10]
        self.assertAlmostEqual(np.abs(lsp.residuals()[0]), 3.5)
        self.assertAlmostEqual(np.abs(lsp.residuals()[1]), 0.8)


if __name__ == "__main__":
    unittest.main()
