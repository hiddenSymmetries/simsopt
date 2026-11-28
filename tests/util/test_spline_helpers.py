import io
import unittest
from contextlib import redirect_stdout

import numpy as np
from scipy.interpolate import BSpline

from simsopt.util.spline_helpers import (
    b_p,
    b_p_deriv,
    b_p_deriv2,
    double_reflection_rmf,
    print_dofs_nicely,
    uniform_knots,
)


def scipy_basis(t, p, x, nu=0):
    """Every basis function (or its nu-th derivative) from scipy, shape (len(x), len(t) - p - 1)."""
    nb = len(t) - p - 1
    return np.stack([BSpline(t, np.eye(nb)[i], p).derivative(nu)(x) if nu else BSpline(t, np.eye(nb)[i], p)(x)
                     for i in range(nb)], axis=1)


def periodic_control_points(P, p):
    """Pad control points by p on each side, as SurfaceBSpline does for its periodic directions."""
    return np.concatenate([P[-p:], P, P[:p]])


class UniformKnotsTests(unittest.TestCase):

    def test_length_spacing_and_domain(self):
        for p in (1, 2, 3, 4):
            for n, domain in ((4, 2 * np.pi), (9, 1.0)):
                t = uniform_knots(n, p, domain=domain)
                self.assertEqual(len(t), n + 3 * p + 2)
                np.testing.assert_allclose(np.diff(t), domain / (n + 1))
                # 0 and the period are both knots
                self.assertTrue(np.any(np.isclose(t, 0.0, atol=1e-14)))
                self.assertTrue(np.any(np.isclose(t, domain)))
                # one basis function per padded control point
                self.assertEqual(len(t) - p - 1, n + 1 + 2 * p)

    def test_periodic_curve_closes_smoothly(self):
        """With padded control points the curve and its first p-1 derivatives match at 0 and the period."""
        rng = np.random.default_rng(0)
        for p in (2, 3, 4):
            n = 7
            t = uniform_knots(n, p)
            P = periodic_control_points(rng.normal(size=(n + 1, 3)), p)
            ends = np.array([0.0, 2 * np.pi])
            B, D1, D2 = b_p_deriv2(t, p, ends)
            for M in (B, D1) + ((D2,) if p >= 3 else ()):
                c = M @ P
                np.testing.assert_allclose(c[0], c[1], atol=1e-12)


class BasisFunctionTests(unittest.TestCase):

    x = np.linspace(0, 2 * np.pi, 997, endpoint=False)

    def test_basis_matches_scipy(self):
        for p in (1, 2, 3, 4):
            t = uniform_knots(6, p)
            np.testing.assert_allclose(b_p(t, p, self.x), scipy_basis(t, p, self.x), atol=1e-14)

    def test_derivatives_match_scipy(self):
        for p in (1, 2, 3, 4):
            t = uniform_knots(6, p)
            B1, D1 = b_p_deriv(t, p, self.x)
            B2, D1b, D2 = b_p_deriv2(t, p, self.x)
            np.testing.assert_allclose(B1, scipy_basis(t, p, self.x), atol=1e-14)
            np.testing.assert_allclose(B2, B1, atol=1e-15)
            np.testing.assert_allclose(D1, scipy_basis(t, p, self.x, nu=1), atol=1e-12)
            np.testing.assert_allclose(D1b, D1, atol=1e-15)
            if p >= 2:
                np.testing.assert_allclose(D2, scipy_basis(t, p, self.x, nu=2), atol=1e-11)
            else:
                np.testing.assert_array_equal(D2, 0)

    def test_nonuniform_knots_match_scipy(self):
        """Unevenly spaced knots."""
        t = np.array([-0.9, -0.5, -0.2, 0.0, 0.3, 1.1, 1.4, 2.0, 2.5, 3.0, 3.4, 3.5, 4.1])
        p = 2
        x = np.linspace(0, 3.0, 301, endpoint=False)
        B, D1, D2 = b_p_deriv2(t, p, x)
        np.testing.assert_allclose(B, scipy_basis(t, p, x), atol=1e-14)
        np.testing.assert_allclose(D1, scipy_basis(t, p, x, nu=1), atol=1e-12)
        np.testing.assert_allclose(D2, scipy_basis(t, p, x, nu=2), atol=1e-11)

    def test_clamped_knots_match_scipy_including_last_knot(self):
        """Repeated end knots, evaluated with zero, one and two points at the last knot."""
        for p in (1, 2, 3):
            t = np.concatenate([np.zeros(p), [0.0, 0.4, 1.1, 2.0, 3.0], np.full(p, 3.0)])
            for x in (np.linspace(0, 3.0, 50, endpoint=False),
                      np.linspace(0, 3.0, 51),
                      np.append(np.linspace(0, 3.0, 51), 3.0)):
                B, D1, D2 = b_p_deriv2(t, p, x)
                np.testing.assert_allclose(B, scipy_basis(t, p, x), atol=1e-14)
                np.testing.assert_allclose(B.sum(axis=1), 1, atol=1e-14)
                np.testing.assert_allclose(b_p(t, p, x), B, atol=1e-15)
                np.testing.assert_allclose(b_p_deriv(t, p, x)[1], D1, atol=1e-15)
                np.testing.assert_allclose(D1, scipy_basis(t, p, x, nu=1), atol=1e-12)
                if p >= 2:
                    np.testing.assert_allclose(D2, scipy_basis(t, p, x, nu=2), atol=1e-11)

    def test_partition_of_unity(self):
        for p in (1, 2, 3, 4):
            t = uniform_knots(5, p)
            B, D1, D2 = b_p_deriv2(t, p, self.x)
            self.assertTrue(np.all(B >= 0))
            np.testing.assert_allclose(B.sum(axis=1), 1, atol=1e-14)
            np.testing.assert_allclose(D1.sum(axis=1), 0, atol=1e-12)
            np.testing.assert_allclose(D2.sum(axis=1), 0, atol=1e-11)

    def test_single_index(self):
        t = uniform_knots(5, 3)
        np.testing.assert_array_equal(b_p(t, 3, self.x, i=4), b_p(t, 3, self.x)[:, 4])


class DoubleReflectionRMFTests(unittest.TestCase):

    def test_frame_is_right_handed_orthonormal(self):
        s = np.linspace(0, 2 * np.pi, 400)
        gamma = np.stack([np.cos(s) * (1 + 0.2 * np.cos(3 * s)), np.sin(s), 0.3 * np.sin(2 * s)], axis=1)
        gammadash = np.gradient(gamma, s, axis=0)
        n0 = np.cross(gammadash[0], [0.0, 0.0, 1.0])
        T, N, B = double_reflection_rmf(gamma, gammadash, n0)
        np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(np.einsum('ij,ij->i', T, N), 0, atol=1e-12)
        np.testing.assert_allclose(B, np.cross(T, N), atol=1e-15)

    def test_planar_curve_has_no_twist(self):
        """For a planar curve, a normal starting perpendicular to the plane stays perpendicular to it."""
        s = np.linspace(0, 2 * np.pi, 200)
        gamma = np.stack([2 * np.cos(s), np.sin(s), np.zeros_like(s)], axis=1)
        gammadash = np.stack([-2 * np.sin(s), np.cos(s), np.zeros_like(s)], axis=1)
        _, N, _ = double_reflection_rmf(gamma, gammadash, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(N, np.tile([0.0, 0.0, 1.0], (len(s), 1)), atol=1e-13)

    def test_helix_rotates_against_frenet_frame_at_torsion_rate(self):
        """A rotation-minimizing normal turns relative to the Frenet frame at rate -torsion."""
        a, b = 1.0, 0.4
        t = np.linspace(0, 4 * np.pi, 4000)
        gamma = np.stack([a * np.cos(t), a * np.sin(t), b * t], axis=1)
        gammadash = np.stack([-a * np.sin(t), a * np.cos(t), b * np.ones_like(t)], axis=1)
        frenet_N = np.stack([-np.cos(t), -np.sin(t), np.zeros_like(t)], axis=1)
        T = gammadash / np.linalg.norm(gammadash, axis=1, keepdims=True)
        frenet_B = np.cross(T, frenet_N)

        _, N, _ = double_reflection_rmf(gamma, gammadash, frenet_N[0])
        angle = np.unwrap(np.arctan2(np.einsum('ij,ij->i', N, frenet_B), np.einsum('ij,ij->i', N, frenet_N)))
        arclength = np.sqrt(a**2 + b**2) * t
        torsion = b / (a**2 + b**2)
        np.testing.assert_allclose(angle, -torsion * arclength, atol=1e-5)


class PrintDofsNicelyTests(unittest.TestCase):

    def test_prints_header_and_one_row_per_dof(self):
        class FakeOpt:
            dof_names = ["a:r_0", "a:theta_1"]
            x = np.array([1.5, -0.25])
            lower_bounds = np.array([0.0, -1.0])
            upper_bounds = np.array([2.0, 1.0])

        for kwargs, bounds in (({}, ("0.0", "2.0")), ({"lb": [-9, -9], "ub": [9, 9]}, ("-9", "9"))):
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_dofs_nicely(FakeOpt(), **kwargs)
            lines = buf.getvalue().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0].split(), ["dof", "value", "lower", "bound", "upper", "bound"])
            self.assertEqual(lines[1].split(), ["a:r_0", "1.5", *bounds])


if __name__ == "__main__":
    unittest.main()
