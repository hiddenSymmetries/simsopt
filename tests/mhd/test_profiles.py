import unittest
import logging
import numpy as np
from simsopt.mhd.profiles import ProfilePolynomial

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

class ProfilesTests(unittest.TestCase):
    def test_polynomial(self):
        """
        Test the value and derivative of a polynomial profile
        """
        # Function: f(s) = 3 * (1 - s^3)
        prof = ProfilePolynomial([3, 0, 0, -3])
        s = np.linspace(0, 1, 10)
        np.testing.assert_allclose(prof.f(s), 3 * (1 - s ** 3))
        np.testing.assert_allclose(prof.dfds(s), 3 * (- 3 * s ** 2))
        
    def test_plot(self):
        """
        Test the plotting function.
        """
        # Function: f(s) = 3 * (1 - s^3)
        prof = ProfilePolynomial([3, 0, 0, -3])
        prof.plot(show=False)
