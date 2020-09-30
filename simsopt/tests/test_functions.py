import unittest
import numpy as np
from simsopt.functions import Identity, Rosenbrock

class IdentityTests(unittest.TestCase):
    def test_basic(self):
        iden = Identity()
        self.assertAlmostEqual(iden.J(), 0, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([0.0]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

        x = 3.5
        iden = Identity(x)
        self.assertAlmostEqual(iden.J(), x, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([x]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

        y = -2
        iden.set_dofs([y])
        self.assertAlmostEqual(iden.J(), y, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([y]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

class RosenbrockTests(unittest.TestCase):
    def test_1(self):
        """
        This is the most common use case.
        """
        r = Rosenbrock()
        self.assertAlmostEqual(r.term1(), -1.0, places=13)
        self.assertAlmostEqual(r.term2(),  0.0, places=13)

        # Change the parameters:
        x_new = [3, 2]
        r.set_dofs(x_new)
        np.testing.assert_allclose(x_new, r.get_dofs(), rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(r.term1(), 2.0, places=13)
        self.assertAlmostEqual(r.term2(), 0.7, places=13)

if __name__ == "__main__":
    unittest.main()
