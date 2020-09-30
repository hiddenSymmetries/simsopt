import unittest
from simsopt.util import Identity, Target
from simsopt.least_squares_term import LeastSquaresTerm

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

if __name__ == "__main__":
    unittest.main()
