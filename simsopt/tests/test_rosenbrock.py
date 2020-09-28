import unittest
import numpy as np
from simsopt.rosenbrock import Rosenbrock

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
