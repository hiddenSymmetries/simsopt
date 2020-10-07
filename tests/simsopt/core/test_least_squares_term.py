import unittest
from simsopt.target import Identity
from simsopt.least_squares_term import LeastSquaresTerm

class LeastSquaresTermTests(unittest.TestCase):

    def test_basic(self):
        """
        Test basic usage
        """
        iden = Identity()
        lst = LeastSquaresTerm(iden.target, 3, 0.1)
        self.assertIs(lst.in_target, iden.target)
        self.assertEqual(lst.goal, 3)
        self.assertAlmostEqual(lst.sigma, 0.1, places=13)

        iden.x.val = 17
        self.assertEqual(lst.in_val, 17)
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst.out_val, correct_value, places=13)
        # Check that out_target gives the right value:
        self.assertAlmostEqual(lst.out_target.evaluate(), correct_value,
                               places=13)
        # Check that out_target correctly has iden.x as its parameter:
        self.assertEqual(lst.out_target.parameters, {iden.x})

    def test_exceptions(self):
        """
        Test that exceptions are thrown when invalid inputs are
        provided.
        """
        # First argument must have type Target
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(2, 3, 0.1)

        # Second and third arguments must be real numbers
        iden = Identity()
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.target, "hello", 0.1)
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.target, 3, iden)

        # sigma cannot be zero
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.target, 3, 0)
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden.target, 3, 0.0)

if __name__ == "__main__":
    unittest.main()
