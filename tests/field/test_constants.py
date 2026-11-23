import unittest

import numpy as np
from scipy import constants

import simsoptpp as sopp
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier


class TestMu0Consistency(unittest.TestCase):
    """The C++ core must use the same mu_0 as scipy.constants."""

    def test_cpp_mu0_matches_scipy(self):
        self.assertEqual(sopp.MU0, constants.mu_0)

    def test_biot_savart_circular_coil_center(self):
        # B = mu_0 I / (2 R) at the center of a circular loop
        R, I = 1.3, 2.7e5
        curve = CurveXYZFourier(200, 1)
        curve.set_dofs([0, 0, R, 0, R, 0, 0, 0, 0])  # x = R cos, y = R sin
        field = BiotSavart([Coil(curve, Current(I))])
        field.set_points(np.zeros((1, 3)))
        np.testing.assert_allclose(
            field.B()[0, 2], constants.mu_0 * I / (2 * R), rtol=1e-12)

    def test_dipole_field_on_axis(self):
        # B = mu_0 m / (2 pi r^3) on the axis of a dipole
        r, m = 0.7, 3.1
        pts = np.array([[0.0, 0.0, r]])
        dip_pos = np.zeros((1, 3))
        dip_m = np.array([[0.0, 0.0, m]])
        B = sopp.dipole_field_B(pts, dip_pos, dip_m)
        np.testing.assert_allclose(
            B[0, 2], constants.mu_0 * m / (2 * np.pi * r**3), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
