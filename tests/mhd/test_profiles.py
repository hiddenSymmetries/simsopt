import unittest
import logging
import numpy as np
from simsopt.mhd.profiles import ProfilePolynomial, ProfileScaled

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
        np.testing.assert_allclose(prof(s), 3 * (1 - s ** 3))
        np.testing.assert_allclose(prof.dfds(s), 3 * (- 3 * s ** 2))

    def test_plot(self):
        """
        Test the plotting function.
        """
        # Function: f(s) = 3 * (1 - s^3)
        prof = ProfilePolynomial([3, 0, 0, -3])
        prof.plot(show=False)

    def test_scaled(self):
        """
        Test ProfileScaled
        """
        prof1 = ProfilePolynomial([3, 0, 0, -3])
        scalefac = 0.6
        prof2 = ProfileScaled(prof1, scalefac)
        prof2.unfix_all()
        s = np.linspace(0, 1, 10)
        np.testing.assert_allclose(prof2(s), scalefac * prof1(s))
        np.testing.assert_allclose(prof2.dfds(s), scalefac * prof1.dfds(s))
        newscalefac = 1.3
        prof2.x = [newscalefac]
        np.testing.assert_allclose(prof2(s), newscalefac * prof1(s))
        np.testing.assert_allclose(prof2.dfds(s), newscalefac * prof1.dfds(s))
