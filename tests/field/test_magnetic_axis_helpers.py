#!/usr/bin/env python3
import unittest
import warnings
import numpy as np
from simsopt.field.magnetic_axis_helpers import compute_on_axis_iota, compute_on_axis_tangent_map
from simsopt.field import ScalarPotentialRZMagneticField
from simsopt.geo import CurveRZFourier
from simsopt.configs.zoo import get_data


def rotating_hyperbolic_field(nfp, lam, k):
    """
    A field with the circle R = 1, Z = 0 as a closed field line, whose linearization about the circle is a hyperbolic
    stretch of strength lam (per radian of toroidal angle) rotating by k half turns per field period. Its tangent map
    over one period is R(k pi) exp(K T), T = 2 pi / nfp, K = [[lam, w], [-w, -lam]], w = k nfp / 2, so its trace is
    (-1)^k 2 cosh(T sqrt(lam^2 - w^2)) for lam > w and (-1)^k 2 cos(T sqrt(w^2 - lam^2)) for lam < w.
    """
    field = ScalarPotentialRZMagneticField(
        f"phi + ({lam}/2)*(((R-1)**2 - Z**2)*cos({k * nfp}*phi) + 2*(R-1)*Z*sin({k * nfp}*phi))")
    axis = CurveRZFourier(quadpoints=np.linspace(0, 1 / nfp, 64, endpoint=False), order=1, nfp=nfp, stellsym=True)
    axis.rc[0] = 1.0
    axis.x = axis.get_dofs()
    T, w = 2 * np.pi / nfp, k * nfp / 2
    mu2 = lam**2 - w**2
    trace = (-1)**k * 2 * (np.cosh(T * np.sqrt(mu2)) if mu2 > 0 else np.cos(T * np.sqrt(-mu2)))
    return axis, field, trace

class MagneticAxisHelpers(unittest.TestCase):

    def test_magnetic_axis_iota(self):
        """
        Verify that the rotational transform can be computed on axis
        """
        for (config, target_iota) in zip(["hsx", "ncsx", "giuliani"], [1.0418687161633922, 0.39549339846119463, 0.42297724084249616]):
            self.subtest_magnetic_axis_iota(config, target_iota)

    def subtest_magnetic_axis_iota(self, config, target_iota):
        base_curves, base_currents, ma, nfp, bs = get_data(config)
        iota = compute_on_axis_iota(ma, bs)
        np.testing.assert_allclose(iota, target_iota, rtol=1e-10, atol=1e-10)

    def test_tangent_map_trace(self):
        """
        The trace of the on-axis tangent map matches its analytic value for an elliptic, a reflection-hyperbolic and
        a hyperbolic axis
        """
        for nfp, lam, k in [(2, 0.5, 1), (2, 1.05, 1), (3, 3.2, 2)]:
            axis, field, trace = rotating_hyperbolic_field(nfp, lam, k)
            M = compute_on_axis_tangent_map(axis, field)
            np.testing.assert_allclose(np.trace(M), trace, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(np.linalg.det(M), 1.0, rtol=1e-8, atol=1e-8)

    def test_iota_warns_when_the_axis_is_not_elliptic(self):
        """
        On a reflection-hyperbolic axis compute_on_axis_iota returns nfp/2 and on a hyperbolic axis 0, with a warning;
        on an elliptic axis it does not warn
        """
        for nfp, lam, k, kind, value in [(2, 1.05, 1, 'reflection-hyperbolic', 1.0), (3, 3.2, 2, 'hyperbolic', 0.0)]:
            axis, field, trace = rotating_hyperbolic_field(nfp, lam, k)
            with self.assertWarns(RuntimeWarning) as w:
                iota = compute_on_axis_iota(axis, field)
            self.assertIn(kind, str(w.warning))
            np.testing.assert_allclose(abs(iota), value, atol=1e-12)
        axis, field, trace = rotating_hyperbolic_field(2, 0.5, 1)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            iota = compute_on_axis_iota(axis, field)
        np.testing.assert_allclose(iota, 2 * np.arccos(trace / 2) / (2 * np.pi), rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
