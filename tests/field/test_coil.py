import unittest
import numpy as np
from simsopt.field.coil import Current, ScaledCurrent


class CoilTesting(unittest.TestCase):

    def test_scaled_current(self):
        one = np.asarray([1.])
        c0 = Current(5.)
        fak = 3.
        c1 = fak * c0
        assert abs(c1.get_value()-fak * c0.get_value()) < 1e-15
        assert np.linalg.norm((c1.vjp(one)-fak * c0.vjp(one))(c0)) < 1e-15

        c2 = c0 * fak
        assert abs(c2.get_value()-fak * c0.get_value()) < 1e-15
        assert np.linalg.norm((c2.vjp(one)-fak * c0.vjp(one))(c0)) < 1e-15

        c3 = ScaledCurrent(c0, fak)
        assert abs(c3.get_value()-fak * c0.get_value()) < 1e-15
        assert np.linalg.norm((c3.vjp(one)-fak * c0.vjp(one))(c0)) < 1e-15

        c4 = -c0
        assert abs(c4.get_value() - (-1.) * c0.get_value()) < 1e-15
        assert np.linalg.norm((c4.vjp(one) - (-1.) * c0.vjp(one))(c0)) < 1e-15

        c00 = Current(6.)

        c5 = c0 + c00
        assert abs(c5.get_value() - (5. + 6.)) < 1e-15
        assert np.linalg.norm((c5.vjp(one)-c0.vjp(one))(c0)) < 1e-15
        assert np.linalg.norm((c5.vjp(one)-c00.vjp(one))(c00)) < 1e-15
        c6 = sum([c0, c00])
        assert abs(c6.get_value() - (5. + 6.)) < 1e-15
        assert np.linalg.norm((c6.vjp(one)-c0.vjp(one))(c0)) < 1e-15
        assert np.linalg.norm((c6.vjp(one)-c00.vjp(one))(c00)) < 1e-15
        c7 = c0 - c00
        assert abs(c7.get_value() - (5. - 6.)) < 1e-15
        assert np.linalg.norm((c7.vjp(one)-c0.vjp(one))(c0)) < 1e-15
        assert np.linalg.norm((c7.vjp(one)+c00.vjp(one))(c00)) < 1e-15
