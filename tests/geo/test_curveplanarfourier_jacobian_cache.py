"""CurvePlanarFourier Jacobians must follow the dofs.

The curve's normalized quaternion rotation makes all four ``d*_by_dcoeff``
blocks depend on the dofs, so they must be recomputed when the dofs change.
"""

import unittest

import numpy as np

from simsopt.geo.curveplanarfourier import CurvePlanarFourier

FIRST_DOFS = np.array(
    [1.0, 0.1, -0.05, 0.02, 0.03, -0.01, 0.004, 0.9, 0.2, -0.3, 0.1, 0.5, -0.2, 0.3]
)
SECOND_DOFS = FIRST_DOFS + 0.3 * np.random.default_rng(0).standard_normal(FIRST_DOFS.size)

JACOBIANS = ("dgamma_by_dcoeff", "dgammadash_by_dcoeff",
             "dgammadashdash_by_dcoeff", "dgammadashdashdash_by_dcoeff")


def _curve_at(dofs):
    curve = CurvePlanarFourier(40, 3)
    curve.x = dofs
    return curve


class CurvePlanarFourierJacobianCacheTests(unittest.TestCase):

    def test_jacobian_does_not_depend_on_the_dof_history(self):
        for name in JACOBIANS:
            with self.subTest(jacobian=name):
                moved = _curve_at(FIRST_DOFS)
                getattr(moved, name)()  # fills the cache at FIRST_DOFS
                moved.x = SECOND_DOFS
                np.testing.assert_array_equal(
                    getattr(moved, name)(), getattr(_curve_at(SECOND_DOFS), name)(),
                    err_msg=f"{name} of a curve moved to the dofs differs from one built at them")


if __name__ == "__main__":
    unittest.main()
