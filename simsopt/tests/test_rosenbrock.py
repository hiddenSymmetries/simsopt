import unittest
from simsopt.rosenbrock import Rosenbrock

class RosenbrockTests(unittest.TestCase):
    def test_1(self):
        """
        This is the most common use case.
        """
        r = Rosenbrock()
        self.assertAlmostEqual(r.target1.evaluate(), 1.0, places=13)
        self.assertAlmostEqual(r.target2.evaluate(), 0.0, places=13)

        # Change one of the parameters:
        r.x1.val = -3.0
        self.assertAlmostEqual(r.target1.evaluate(), 4.0, places=13)
        self.assertAlmostEqual(r.target2.evaluate(), -90.0, places=13)

        # Change the second parameter:
        r.x2.val = 2.0
        self.assertAlmostEqual(r.target1.evaluate(), 4.0, places=13)
        self.assertAlmostEqual(r.target2.evaluate(), -70.0, places=13)

if __name__ == "__main__":
    unittest.main()
