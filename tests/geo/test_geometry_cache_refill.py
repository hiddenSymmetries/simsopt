"""Arrays returned by the C++ geometry caches are never rewritten.

The caches hand Python the cached array itself, so a dof change must
recompute into a new array: the caller may still hold the old one.
"""

import unittest

import numpy as np

from simsopt.geo import CurveXYZFourier, SurfaceRZFourier

CURVE_QUANTITIES = ("gamma", "gammadash", "gammadashdash", "kappa", "torsion")
SURFACE_QUANTITIES = ("gamma", "gammadash1", "gammadash2", "normal", "unitnormal")


def _curve():
    curve = CurveXYZFourier(32, 3)
    curve.set("xc(0)", 1.0)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("zs(2)", 0.1)
    return curve


def _surface():
    surface = SurfaceRZFourier(nfp=2, mpol=2, ntor=2)
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    surface.set_rc(1, 1, 0.02)
    return surface


class GeometryCacheRefillTests(unittest.TestCase):

    def _check(self, make, quantities):
        rng = np.random.default_rng(0)
        for name in quantities:
            with self.subTest(quantity=name):
                obj = make()
                held = getattr(obj, name)()
                held_values = np.array(held, copy=True)

                new_x = obj.x + 1e-2 * rng.standard_normal(obj.x.size)
                obj.x = new_x
                refreshed = getattr(obj, name)()

                np.testing.assert_array_equal(
                    held, held_values,
                    err_msg=f"{name}() rewrote the array it returned before the dofs changed")

                reference = make()
                reference.x = new_x
                np.testing.assert_array_equal(
                    refreshed, getattr(reference, name)(),
                    err_msg=f"{name}() after a dof change differs from a fresh object at those dofs")
                self.assertFalse(
                    np.array_equal(refreshed, held_values),
                    f"{name}() did not change with the dofs; the test is vacuous")

    def test_curve_arrays_survive_a_dof_change(self):
        self._check(_curve, CURVE_QUANTITIES)

    def test_surface_arrays_survive_a_dof_change(self):
        self._check(_surface, SURFACE_QUANTITIES)


if __name__ == "__main__":
    unittest.main()
