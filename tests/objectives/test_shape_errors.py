import unittest

import numpy as np

from simsopt.geo import SurfaceBSpline, SurfaceRZFourier
from simsopt.objectives.shape_errors import (
    _closest_theta_newton,
    _eval_rz_fourier,
    _fourier_eval_with_theta_derivs,
    angle_matched_shape_error,
    build_angle_matched_reference,
    build_exact_shape_reference,
    exact_shape_error,
)


def circular_torus(R0=1.0, a=0.3, nfp=1):
    """Torus with circular cross sections of minor radius a centered at (R0, 0)."""
    s = SurfaceRZFourier(nfp=nfp, mpol=1, ntor=0)
    s.set_rc(0, 0, R0)
    s.set_rc(1, 0, a)
    s.set_zs(1, 0, a)
    return s


def rotating_ellipse(nfp=3, sign=1):
    """Shaped, phi-dependent stellarator-symmetric surface; sign flips the n != 0 modes."""
    s = SurfaceRZFourier(nfp=nfp, mpol=2, ntor=1)
    s.set_rc(0, 0, 1.0)
    s.set_rc(0, 1, sign * 0.05)
    s.set_zs(0, 1, sign * 0.05)
    s.set_rc(1, 0, 0.2)
    s.set_zs(1, 0, 0.2)
    s.set_rc(1, 1, sign * 0.06)
    s.set_zs(1, 1, -sign * 0.06)
    s.set_rc(2, 0, 0.02)
    s.set_zs(2, 0, 0.01)
    return s


def reversed_theta(surf):
    """Same shape as surf with the poloidal angle running the other way (theta -> -theta)."""
    s = SurfaceRZFourier(nfp=surf.nfp, mpol=surf.mpol, ntor=surf.ntor)
    for n in range(0, surf.ntor + 1):
        s.set_rc(0, n, surf.get_rc(0, n))
        if n > 0:
            s.set_zs(0, n, surf.get_zs(0, n))
    # for m > 0, cos(m*theta - n*nfp*phi) -> cos(m*theta + n*nfp*phi): (m, n) -> (m, -n), and Z flips sign
    for m in range(1, surf.mpol + 1):
        for n in range(-surf.ntor, surf.ntor + 1):
            s.set_rc(m, -n, surf.get_rc(m, n))
            s.set_zs(m, -n, -surf.get_zs(m, n))
    return s


class FourierEvalTests(unittest.TestCase):

    def test_eval_rz_fourier_matches_gamma(self):
        """_eval_rz_fourier uses the same angle convention as SurfaceRZFourier.gamma()."""
        surf = rotating_ellipse()
        nu, v_1d = 12, np.linspace(0, 2 * np.pi, 7, endpoint=False)
        R, Z = _eval_rz_fourier(surf, nu, v_1d)
        s = SurfaceRZFourier(
            nfp=surf.nfp, mpol=surf.mpol, ntor=surf.ntor,
            quadpoints_phi=v_1d / (2 * np.pi),
            quadpoints_theta=np.linspace(0, 1, nu, endpoint=False),
        )
        s.x = surf.x
        g = s.gamma()
        np.testing.assert_allclose(R, np.hypot(g[..., 0], g[..., 1]).T, atol=1e-13)
        np.testing.assert_allclose(Z, g[..., 2].T, atol=1e-13)

    def test_theta_derivatives_match_finite_differences(self):
        surf = rotating_ellipse()
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, 2 * np.pi, 20)
        phi = rng.uniform(0, 2 * np.pi, 20)
        h = 1e-5
        R, Z, dR, dZ, d2R, d2Z = _fourier_eval_with_theta_derivs(surf, theta, phi)
        Rp, Zp, dRp, dZp, *_ = _fourier_eval_with_theta_derivs(surf, theta + h, phi)
        Rm, Zm, dRm, dZm, *_ = _fourier_eval_with_theta_derivs(surf, theta - h, phi)
        np.testing.assert_allclose(dR, (Rp - Rm) / (2 * h), atol=1e-9)
        np.testing.assert_allclose(dZ, (Zp - Zm) / (2 * h), atol=1e-9)
        np.testing.assert_allclose(d2R, (dRp - dRm) / (2 * h), atol=1e-9)
        np.testing.assert_allclose(d2Z, (dZp - dZm) / (2 * h), atol=1e-9)

    def test_eval_with_derivs_matches_grid_eval(self):
        surf = rotating_ellipse()
        nu, v_1d = 10, np.array([0.0, 0.4, 1.3])
        R, Z = _eval_rz_fourier(surf, nu, v_1d)
        theta = np.linspace(0, 2 * np.pi, nu, endpoint=False)
        for j, v in enumerate(v_1d):
            R2, Z2, *_ = _fourier_eval_with_theta_derivs(surf, theta, v)
            np.testing.assert_allclose(R2, R[:, j], atol=1e-14)
            np.testing.assert_allclose(Z2, Z[:, j], atol=1e-14)


class ClosestThetaNewtonTests(unittest.TestCase):

    def test_circle_matches_analytic_distance_and_angle(self):
        """Distance from (R, Z) to a circle is | |p - c| - a |, at angle atan2(Z, R - R0)."""
        R0, a = 1.0, 0.3
        surf = circular_torus(R0, a)
        rng = np.random.default_rng(1)
        rho = rng.uniform(0.05, 0.6, 50)  # points both inside and outside the circle
        ang = rng.uniform(0, 2 * np.pi, 50)
        R, Z = R0 + rho * np.cos(ang), rho * np.sin(ang)
        d, theta = _closest_theta_newton(surf, 0.7, R, Z)
        np.testing.assert_allclose(d, np.abs(rho - a), atol=1e-12)
        dtheta = np.angle(np.exp(1j * (theta - ang)))
        np.testing.assert_allclose(dtheta, 0, atol=1e-10)

    def test_points_on_curve_have_zero_distance(self):
        surf = rotating_ellipse()
        phi = 0.37
        theta0 = np.linspace(0, 2 * np.pi, 31, endpoint=False) + 0.01
        R, Z, *_ = _fourier_eval_with_theta_derivs(surf, theta0, phi)
        d, theta = _closest_theta_newton(surf, phi, R, Z)
        np.testing.assert_allclose(d, 0, atol=1e-12)
        np.testing.assert_allclose(np.angle(np.exp(1j * (theta - theta0))), 0, atol=1e-10)

    def test_shaped_curve_matches_brute_force_and_is_stationary(self):
        surf = rotating_ellipse()
        phi = 1.1
        rng = np.random.default_rng(2)
        t = rng.uniform(0, 2 * np.pi, 40)
        R, Z, dR, dZ, *_ = _fourier_eval_with_theta_derivs(surf, t, phi)
        # offset along the normal, well inside the local radius of curvature
        n = np.stack([dZ, -dR]) / np.hypot(dR, dZ)
        off = rng.uniform(-0.03, 0.03, 40)
        R0, Z0 = R + off * n[0], Z + off * n[1]

        d, theta = _closest_theta_newton(surf, phi, R0, Z0)

        dense = np.linspace(0, 2 * np.pi, 400_000, endpoint=False)
        Rd, Zd, *_ = _fourier_eval_with_theta_derivs(surf, dense, phi)
        brute = np.min(np.hypot(Rd[None, :] - R0[:, None], Zd[None, :] - Z0[:, None]), axis=1)
        # never worse than any densely sampled point, and equal to the normal offset
        self.assertTrue(np.all(d <= brute + 1e-12))
        np.testing.assert_allclose(d, brute, atol=1e-7)
        np.testing.assert_allclose(d, np.abs(off), atol=1e-12)

        Rs, Zs, dRs, dZs, *_ = _fourier_eval_with_theta_derivs(surf, theta, phi)
        np.testing.assert_allclose((Rs - R0) * dRs + (Zs - Z0) * dZs, 0, atol=1e-12)


class ExactShapeErrorTests(unittest.TestCase):

    phi_1d = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.05

    def test_identical_surface_has_zero_error(self):
        surf = rotating_ellipse()
        ref = build_exact_shape_reference(surf, self.phi_1d, ntheta=24)
        err = exact_shape_error(surf, ref)
        self.assertEqual(err.shape, (len(self.phi_1d), 24))
        np.testing.assert_allclose(err, 0, atol=1e-10)

    def test_concentric_circles_give_radius_difference(self):
        ref = build_exact_shape_reference(circular_torus(a=0.3), self.phi_1d, ntheta=16)
        err = exact_shape_error(circular_torus(a=0.33), ref)
        np.testing.assert_allclose(err, 0.03, atol=1e-10)

    def test_shifted_circle_matches_analytic_distance(self):
        ref = build_exact_shape_reference(circular_torus(R0=1.0), self.phi_1d, ntheta=16)
        err = exact_shape_error(circular_torus(R0=1.04), ref)
        expected = np.abs(np.hypot(ref.R_ref - 1.04, ref.Z_ref) - 0.3)
        np.testing.assert_allclose(err, expected, atol=1e-10)

    def test_invariant_to_poloidal_parametrization(self):
        """Reversing theta changes the parametrization but not the shape."""
        surf = rotating_ellipse()
        ref = build_exact_shape_reference(surf, self.phi_1d, ntheta=24)
        np.testing.assert_allclose(exact_shape_error(reversed_theta(surf), ref), 0, atol=1e-10)

    def test_toroidal_angle_convention(self):
        """phi-dependent shape: zero error only if reference and surf agree on phi."""
        ref = build_exact_shape_reference(rotating_ellipse(sign=1), self.phi_1d, ntheta=24)
        np.testing.assert_allclose(exact_shape_error(rotating_ellipse(sign=1), ref), 0, atol=1e-10)
        # flipping the n != 0 modes mirrors the shape in phi, so the error must be clearly nonzero
        self.assertGreater(np.max(exact_shape_error(rotating_ellipse(sign=-1), ref)), 1e-2)

    def test_spline_fourier_error_decreases_with_resolution(self):
        """to_RZFourier of a SurfaceBSpline converges to the spline as M = N grows."""
        spline = SurfaceBSpline(p_u=3, p_v=3)
        phi_1d = np.linspace(0, 2 * np.pi / spline.nfp, 5, endpoint=False)
        ref = build_exact_shape_reference(spline, phi_1d, ntheta=16)
        errs = []
        for MN in (2, 4, 8):
            s = SurfaceBSpline(p_u=3, p_v=3, M=MN, N=MN)
            s.x = spline.x
            errs.append(np.max(exact_shape_error(s.to_RZFourier(spec_cond=None), ref)))
        self.assertTrue(errs[0] > errs[1] > errs[2], errs)
        self.assertLess(errs[2], 1e-3)


class AngleMatchedShapeErrorTests(unittest.TestCase):

    def test_identical_surface_has_zero_error(self):
        surf = rotating_ellipse()
        ref = build_angle_matched_reference(surf, nu=32, nv=8)
        err = angle_matched_shape_error(surf, ref)
        self.assertEqual(err.shape, (32, 8))
        np.testing.assert_allclose(err, 0, atol=1e-13)

    def test_concentric_circles_give_radius_difference(self):
        """Up to the chord error of linear interpolation between reference points."""
        nu, a, delta = 32, 0.3, 0.03
        ref = build_angle_matched_reference(circular_torus(a=a), nu=nu, nv=4)
        err = angle_matched_shape_error(circular_torus(a=a + delta), ref, nu=48)
        self.assertEqual(err.shape, (48, 4))
        chord = a * (1 - np.cos(np.pi / nu))
        self.assertTrue(np.all(err >= delta - 1e-13))
        self.assertTrue(np.all(err <= delta + chord + 1e-13))

    def test_reparametrization_error_shrinks_with_resolution(self):
        """Reversing theta: only interpolation error remains, and it falls off as nu grows."""
        surf = rotating_ellipse()
        rev = reversed_theta(surf)
        errs = []
        for nu in (16, 32, 64):
            ref = build_angle_matched_reference(surf, nu=nu, nv=6)
            # offset the evaluation grid so it never coincides with reference points
            errs.append(np.max(angle_matched_shape_error(rev, ref, nu=nu + 1)))
        self.assertTrue(errs[0] > errs[1] > errs[2], errs)
        self.assertLess(errs[2], 1e-3)


if __name__ == "__main__":
    unittest.main()
