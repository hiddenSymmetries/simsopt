import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from simsopt import load, save
from simsopt._core.json import GSONDecoder
from simsopt.geo import InteriorAxis, SurfaceBSpline, SurfaceRZFourier
from simsopt.geo.surfacespline import CrossSectionFixedZeta

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    matplotlib = None

try:
    import plotly
except ImportError:
    plotly = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import vmec
except ImportError:
    vmec = None


def perturbed_surface(seed=0, scale=0.02, **kwargs):
    """A SurfaceBSpline with every free dof randomly perturbed, so no shape is accidentally symmetric."""
    s = SurfaceBSpline(**kwargs)
    s.x = s.x + scale * np.random.default_rng(seed).normal(size=len(s.x))
    return s


def gamma_at(surf, phi, theta):
    """Surface points at paired (phi, theta) fractions."""
    phi, theta = np.broadcast_arrays(np.asarray(phi, float), np.asarray(theta, float))
    data = np.zeros((phi.size, 3))
    surf.gamma_lin(data, phi.ravel(), theta.ravel())
    return data


def rotate_z(xyz, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * xyz[:, 0] - s * xyz[:, 1], s * xyz[:, 0] + c * xyz[:, 1], xyz[:, 2]], axis=1)


VARIANTS = [
    dict(p_u=2, p_v=2),
    dict(p_u=3, p_v=3),
    dict(p_u=3, p_v=2, use_bishop_frame=True),
    dict(p_u=3, p_v=3, n_cs=3, toroidal_equispaced=False, cs_global_angle_free=True),
]


class SymmetryTests(unittest.TestCase):

    def test_stellarator_symmetry(self):
        """gamma(-phi, -theta) is gamma(phi, theta) reflected through the x axis."""
        rng = np.random.default_rng(1)
        phi, theta = rng.uniform(0, 1, 30), rng.uniform(0, 1, 30)
        for kw in VARIANTS:
            with self.subTest(**kw):
                s = perturbed_surface(**kw)
                np.testing.assert_allclose(
                    gamma_at(s, -phi % 1, -theta % 1), gamma_at(s, phi, theta) * [1, -1, -1], atol=1e-12
                )

    def test_field_period_symmetry(self):
        rng = np.random.default_rng(2)
        phi, theta = rng.uniform(0, 1, 30), rng.uniform(0, 1, 30)
        for kw in VARIANTS:
            with self.subTest(**kw):
                s = perturbed_surface(nfp=3, **kw)
                for k in (1, 2):
                    np.testing.assert_allclose(
                        gamma_at(s, (phi + k / 3) % 1, theta),
                        rotate_z(gamma_at(s, phi, theta), 2 * np.pi * k / 3),
                        atol=1e-12,
                    )

    def test_phi_is_the_cylindrical_angle(self):
        rng = np.random.default_rng(3)
        phi, theta = rng.uniform(0, 1, 50), rng.uniform(0, 1, 50)
        for kw in VARIANTS:
            with self.subTest(**kw):
                g = gamma_at(perturbed_surface(**kw), phi, theta)
                np.testing.assert_allclose(np.arctan2(g[:, 1], g[:, 0]) % (2 * np.pi), 2 * np.pi * phi, atol=1e-10)

    def test_gamma_grid_matches_gamma_lin(self):
        s = perturbed_surface(quadpoints_phi=[0.03, 0.2, 0.41], quadpoints_theta=[0.0, 0.35, 0.8])
        P, T = np.meshgrid(s.quadpoints_phi, s.quadpoints_theta, indexing="ij")
        np.testing.assert_allclose(s.gamma(), gamma_at(s, P, T).reshape(3, 3, 3), atol=1e-14)

    def test_theta_convention_matches_surface_rz_fourier(self):
        """theta = 0 is the outboard midplane and theta increases upward, so rc(1,0), zs(1,0) > 0."""
        for kw in VARIANTS:
            for ppcs in (6, 7):
                with self.subTest(points_per_cs=ppcs, **kw):
                    s = perturbed_surface(points_per_cs=ppcs, M=6, N=6, **kw)
                    g = gamma_at(s, [0.0, 0.0, 0.0], [0.0, 0.5, 0.05])
                    axis = np.zeros((1, 3))
                    s.axis.gamma_impl(axis, np.array([0.0]))
                    self.assertGreater(np.hypot(g[0, 0], g[0, 1]), axis[0, 0])  # outboard
                    self.assertLess(np.hypot(g[1, 0], g[1, 1]), axis[0, 0])  # theta = 1/2 inboard
                    np.testing.assert_allclose(g[:2, 2], 0, atol=1e-14)  # both on the midplane
                    self.assertGreater(g[2, 2], 0)  # increasing theta goes up
                    rz = s.to_RZFourier(spec_cond=None)
                    self.assertGreater(rz.get_rc(1, 0), 0)
                    self.assertGreater(rz.get_zs(1, 0), 0)

    def test_outward_normal(self):
        """Positive volume and negative minimum principal curvature, as for SurfaceRZFourier."""
        for kw in VARIANTS:
            with self.subTest(**kw):
                s = perturbed_surface(**kw)
                self.assertGreater(s.volume(), 0)
                self.assertTrue(np.all(s.surface_curvatures()[..., 3] < 0))


class UnsupportedOptionTests(unittest.TestCase):

    def test_non_stellarator_symmetric_raises(self):
        with self.assertRaises(NotImplementedError):
            SurfaceBSpline(stellsym=False)
        with self.assertRaises(NotImplementedError):
            InteriorAxis(stellsym=False)


class ResolutionLimitTests(unittest.TestCase):

    def test_default_surface_tends_to_axisymmetric_circular_torus(self):
        """
        Default dofs put every control point on a circular torus (R0 = 1, a = default_r).

        As n_cs, points_per_cs and axis_points grow together the surface becomes axisymmetric
        (fourth order) and approaches that torus (second order, from B-spline corner cutting).
        axis_points must grow too: a B-spline through axis points on a circle is not a circle.
        """
        a = 0.3
        q_phi = np.linspace(0, 0.5, 24, endpoint=False)
        q_theta = np.linspace(0, 1, 24, endpoint=False)
        axisym, torus = [], []
        for k in (4, 8, 16):
            s = SurfaceBSpline(p_u=3, p_v=3, n_cs=k, points_per_cs=2 * k + 3, axis_points=k + 2,
                               default_r=a, quadpoints_phi=q_phi, quadpoints_theta=q_theta)
            g = s.gamma()
            R, Z = np.hypot(g[..., 0], g[..., 1]), g[..., 2]
            axisym.append(max(np.ptp(R, axis=0).max(), np.ptp(Z, axis=0).max()))
            torus.append(np.abs(np.hypot(R - 1, Z) - a).max())
        for i in (1, 2):
            self.assertLess(axisym[i], axisym[i - 1] / 8)
            self.assertLess(torus[i], torus[i - 1] / 3)
        self.assertLess(axisym[-1], 1e-6)
        self.assertLess(torus[-1], 1e-2)


class DerivativeTests(unittest.TestCase):
    """Analytic phi/theta derivatives against central finite differences of gamma."""

    h = 1e-5

    def surface_and_points(self, kw, seed):
        rng = np.random.default_rng(seed)
        phi, theta = rng.uniform(0, 1, 6), rng.uniform(0, 1, 6)
        s = perturbed_surface(quadpoints_phi=phi, quadpoints_theta=theta, **kw)
        # derivatives are evaluated on the phi x theta grid; FD at the same grid points
        P, T = np.meshgrid(phi, theta, indexing="ij")
        return s, P.ravel(), T.ravel()

    def fd(self, f, P, T, direction):
        dP, dT = (self.h, 0) if direction == "phi" else (0, self.h)
        return (f(P + dP, T + dT) - f(P - dP, T - dT)) / (2 * self.h)

    def test_first_derivatives(self):
        for i, kw in enumerate(VARIANTS):
            with self.subTest(**kw):
                s, P, T = self.surface_and_points(kw, i)
                g = lambda p, t: gamma_at(s, p % 1, t % 1)
                np.testing.assert_allclose(s.gammadash1().reshape(-1, 3), self.fd(g, P, T, "phi"), atol=1e-8)
                np.testing.assert_allclose(s.gammadash2().reshape(-1, 3), self.fd(g, P, T, "theta"), atol=1e-8)

    def test_second_derivatives(self):
        for i, kw in enumerate(VARIANTS):
            with self.subTest(**kw):
                s, P, T = self.surface_and_points(kw, 10 + i)

                def dash(p, t, which):
                    out = np.zeros((p.size, 3))
                    for k, (pk, tk) in enumerate(zip(p % 1, t % 1)):
                        sk = SurfaceBSpline(quadpoints_phi=[pk], quadpoints_theta=[tk], **kw)
                        sk.x = s.x
                        out[k] = getattr(sk, which)()[0, 0]
                    return out

                g1 = lambda p, t: dash(p, t, "gammadash1")
                g2 = lambda p, t: dash(p, t, "gammadash2")
                np.testing.assert_allclose(s.gammadash1dash1().reshape(-1, 3), self.fd(g1, P, T, "phi"), atol=1e-6)
                np.testing.assert_allclose(s.gammadash1dash2().reshape(-1, 3), self.fd(g1, P, T, "theta"), atol=1e-6)
                np.testing.assert_allclose(s.gammadash2dash2().reshape(-1, 3), self.fd(g2, P, T, "theta"), atol=1e-6)

    def test_dof_derivatives_not_implemented(self):
        with self.assertRaises(RuntimeError):
            SurfaceBSpline().dgamma_by_dcoeff()


class CurvatureTests(unittest.TestCase):

    def test_curvature_identities(self):
        s = perturbed_surface(p_u=3, p_v=3)
        H, K, k1, k2 = np.moveaxis(s.surface_curvatures(), -1, 0)
        np.testing.assert_allclose(k1 + k2, 2 * H, atol=1e-10)
        np.testing.assert_allclose(k1 * k2, K, atol=1e-10)
        self.assertTrue(np.all(k1 >= k2))

    def test_gauss_bonnet(self):
        """The total Gaussian curvature of a torus is zero (per field period, by symmetry)."""
        rel = []
        for n in (128, 256):
            q = np.linspace(0, 1, n, endpoint=False)
            s = perturbed_surface(p_u=3, p_v=3, nfp=2, quadpoints_phi=q / 2, quadpoints_theta=q)
            K = s.surface_curvatures()[..., 1]
            dA = np.linalg.norm(s.normal(), axis=-1)
            rel.append(abs(np.mean(K * dA)) / np.mean(np.abs(K) * dA))
        # second-order quadrature convergence (K is only C0 for p = 3)
        self.assertLess(rel[1], 2e-5)
        self.assertLess(rel[1], rel[0] / 2.5)


class DofTests(unittest.TestCase):

    @staticmethod
    def local_names(s):
        return {n.split(":")[1] for n in s.dof_names}

    def test_default_dofs(self):
        s = SurfaceBSpline(axis_points=4, n_cs=2, points_per_cs=7)
        # z_sym cross section: 4 radii, 3 free angles; other: 7 radii, 6 free angles;
        # axis: 4 radii, 2 interior z, 2 interior zeta
        self.assertEqual(len(s.x), (4 + 3) + (7 + 6) + (4 + 2 + 2))

    def test_flags_change_the_expected_dofs(self):
        base = self.local_names(SurfaceBSpline(n_cs=3))
        cases = [
            (dict(toroidal_equispaced=False), {"cs_zeta1", "cs_zeta2"}, set()),
            (dict(cs_global_angle_free=True), {"cs_angle1", "cs_angle2"}, set()),
            (dict(axis_angles_fixed=True), set(), {"zeta_axis_1", "zeta_axis_2"}),
            (dict(poloidal_equispaced=True), set(), {f"theta_{i}" for i in range(1, 7)}),
            (dict(nurbs=True), {f"w_{i}" for i in range(6)}, set()),
        ]
        for kw, added, removed in cases:
            with self.subTest(**kw):
                names = self.local_names(SurfaceBSpline(n_cs=3, **kw))
                self.assertEqual(names - base, added)
                self.assertEqual(base - names, removed)

    def test_dof_order_is_cross_sections_then_axis_then_surface(self):
        """Natural sort keeps this order past 9 cross sections; logged dof vectors rely on it."""
        s = perturbed_surface(n_cs=12, toroidal_equispaced=False)
        expected = np.concatenate([cs.x for cs in s.cs_list] + [s.axis.x, s.local_x])
        np.testing.assert_array_equal(s.x, expected)

    def test_setting_x_reproduces_geometry(self):
        """Setting x (or full_x) on a fresh surface gives identical geometry."""
        kw = dict(n_cs=3, toroidal_equispaced=False, cs_global_angle_free=True)
        s = perturbed_surface(**kw)
        for attr in ("x", "full_x"):
            with self.subTest(attr=attr):
                t = SurfaceBSpline(**kw)
                setattr(t, attr, getattr(s, attr))
                np.testing.assert_array_equal(t.gamma(), s.gamma())

    def test_child_dof_changes_invalidate_cache(self):
        s = perturbed_surface()
        for child, name in ((s.axis, "r_axis_1"), (s.cs_list[1], "r_2"), (s, None)):
            g0 = s.gamma().copy()
            if name is None:
                s.x = s.x + 0.01
            else:
                child.set(name, child.get(name) + 0.01)
            self.assertFalse(np.allclose(s.gamma(), g0))

    def test_nurbs_fixes_one_weight_per_cross_section(self):
        """With or without fixed polar angles, exactly one weight per cross section stays fixed."""
        for equispaced in (False, True):
            with self.subTest(poloidal_equispaced=equispaced):
                s = SurfaceBSpline(nurbs=True, poloidal_equispaced=equispaced)
                for cs in s.cs_list:
                    w_fixed = [cs.is_fixed(f"w_{i}") for i in range(cs.n_pts)]
                    self.assertEqual(sum(w_fixed), 1)

    def test_nurbs_weight_scale_does_not_change_shape(self):
        """Why one weight per cross section is fixed: scaling all of them leaves the surface unchanged."""
        s = perturbed_surface(nurbs=True)
        g0 = s.gamma().copy()
        for cs in s.cs_list:
            cs.unfix_all()
            cs.full_x = np.concatenate([cs.r_ctrl, cs.theta_ctrl, 1.7 * cs.w_ctrl])
        np.testing.assert_allclose(s.gamma(), g0, atol=1e-12)


class CrossSectionTests(unittest.TestCase):

    def test_z_sym_full_arrays_are_mirrored(self):
        for n in (6, 7):
            with self.subTest(n_ctrl_pts=n):
                cs = CrossSectionFixedZeta(0, n_ctrl_pts=n, z_sym=True)
                cs.x = cs.x + 0.05 * np.random.default_rng(n).normal(size=len(cs.x))
                r, th = cs.get_r_ctrl_full(), cs.get_theta_ctrl_full()
                self.assertEqual(len(r), n)
                np.testing.assert_allclose(r[1:], r[1:][::-1])
                np.testing.assert_allclose(th[1:] + th[1:][::-1], 2 * np.pi)
                self.assertEqual(th[0], 0)

    def test_flipped_is_up_down_mirror(self):
        cs = CrossSectionFixedZeta(1, n_ctrl_pts=7)
        cs.x = cs.x + 0.05 * np.random.default_rng(0).normal(size=len(cs.x))
        fixed = [cs.is_fixed(n) for n in cs.local_full_dof_names]
        f = cs.flipped()

        def points(c):
            r, th = c.get_r_ctrl_full(), c.get_theta_ctrl_full()
            return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)

        p, q = points(cs), points(f)
        # same points mirrored in z, with the angle ordering reversed after the pinned first point
        np.testing.assert_allclose(q[0], p[0] * [1, -1], atol=1e-14)
        np.testing.assert_allclose(q[1:], (p[1:] * [1, -1])[::-1], atol=1e-14)
        self.assertEqual([cs.is_fixed(n) for n in cs.local_full_dof_names], fixed)
        self.assertEqual([f.is_fixed(n) for n in f.local_full_dof_names], fixed)
        z = CrossSectionFixedZeta(0, n_ctrl_pts=7, z_sym=True)
        self.assertIs(z.flipped(), z)


class InteriorAxisTests(unittest.TestCase):

    def perturbed_axis(self, seed=0, **kw):
        a = InteriorAxis(n_ctrl_pts=5, nfp=3, **kw)
        a.x = a.x + 0.03 * np.random.default_rng(seed).normal(size=len(a.x))
        return a

    def test_symmetries(self):
        q = np.random.default_rng(4).uniform(0, 1, 40)
        a = self.perturbed_axis()
        g = lambda t: (lambda d: (a.gamma_impl(d, np.asarray(t) % 1), d)[1])(np.zeros((len(t), 3)))
        np.testing.assert_allclose(g(-q), g(q) * [1, -1, -1], atol=1e-12)
        np.testing.assert_allclose(g(q + 1 / 3), rotate_z(g(q), 2 * np.pi / 3), atol=1e-12)
        np.testing.assert_allclose(np.arctan2(g(q)[:, 1], g(q)[:, 0]) % (2 * np.pi), 2 * np.pi * q, atol=1e-10)

    def test_derivatives(self):
        h = 1e-5
        q = np.random.default_rng(5).uniform(0, 1, 10)
        a = self.perturbed_axis()
        def shifted(dq):
            b = InteriorAxis(n_ctrl_pts=5, nfp=3, quadpoints=(q + dq) % 1)
            b.x = a.x
            return b

        A0, Ap, Am = shifted(0), shifted(h), shifted(-h)
        np.testing.assert_allclose(A0.gammadash(), (Ap.gamma() - Am.gamma()) / (2 * h), atol=1e-8)
        np.testing.assert_allclose(A0.gammadashdash(), (Ap.gammadash() - Am.gammadash()) / (2 * h), atol=1e-6)

    def test_toroidal_monotonicity_check(self):
        self.assertTrue(InteriorAxis(n_ctrl_pts=4, nfp=2).is_toroidally_monotonic())
        a = InteriorAxis(n_ctrl_pts=4, nfp=2)
        for name, val in (("r_axis_1", 0.3), ("r_axis_2", 2.5), ("zeta_axis_1", 0.9), ("zeta_axis_2", 0.3)):
            a.set(name, val)
        self.assertFalse(a.is_toroidally_monotonic())

    def test_bishop_frame(self):
        a = self.perturbed_axis()
        q = np.linspace(0, 1, 90, endpoint=False)
        T, N, B = a.bishop_frame(q)
        for v in (T, N, B):
            np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(np.einsum("ij,ij->i", T, N), 0, atol=1e-12)
        np.testing.assert_allclose(B, np.cross(T, N), atol=1e-12)
        A = InteriorAxis(n_ctrl_pts=5, nfp=3, quadpoints=q)
        A.x = a.x
        np.testing.assert_allclose(T, A.gammadash() / np.linalg.norm(A.gammadash(), axis=1, keepdims=True), atol=1e-10)
        # nfp-periodic: the frame one field period later is the same frame rotated by 2*pi/nfp
        k = len(q) // 3
        for v in (T, N, B):
            np.testing.assert_allclose(np.roll(v, -k, axis=0), rotate_z(v, 2 * np.pi / 3), atol=1e-10)

    def test_bishop_frame_of_planar_axis_has_no_twist(self):
        _, N, B = InteriorAxis(n_ctrl_pts=4, nfp=2).bishop_frame()
        np.testing.assert_allclose(np.abs(B[:, 2]), 1, atol=1e-12)
        np.testing.assert_allclose(N[:, 2], 0, atol=1e-12)


def violated(s, A, lb, ub, titles):
    """Titles of the constraint rows violated at the surface's current dofs."""
    v = A @ s.x
    return {t for t, vi, lo, hi in zip(titles, v, lb, ub) if vi < lo - 1e-12 or vi > hi + 1e-12}


class ConstraintTests(unittest.TestCase):

    def test_shapes_and_defaults_feasible(self):
        s = SurfaceBSpline(n_cs=3, toroidal_equispaced=False, cs_global_angle_free=True)
        A, lb, ub, titles = s.write_inequality_constraints(cs_angle_step_max=0.5)
        self.assertEqual(A.shape[1], len(s.x))
        self.assertEqual(len(lb), A.shape[0])
        self.assertEqual(len(ub), A.shape[0])
        self.assertEqual(len(titles), A.shape[0])
        v = A @ s.x
        self.assertTrue(np.all(v >= lb) and np.all(v <= ub))

    def test_cross_section_larger_than_axis_is_infeasible(self):
        s = SurfaceBSpline()
        A, lb, ub, _ = s.write_inequality_constraints()
        s.cs_list[1].set("r_3", 1.5)  # larger than every axis radius
        v = A @ s.x
        self.assertFalse(np.all(v >= lb) and np.all(v <= ub))

    def constraints(self, s, **kw):
        """Constraint rows for s, checked to be feasible at its current dofs."""
        A, lb, ub, titles = s.write_inequality_constraints(**kw)
        self.assertEqual(violated(s, A, lb, ub, titles), set())
        return A, lb, ub, titles

    def test_axis_height(self):
        s = SurfaceBSpline(axis_points=4)
        A, lb, ub, titles = self.constraints(s, z_axis_max=0.3)
        s.axis.set("z_axis_1", 0.29)
        self.assertEqual(violated(s, A, lb, ub, titles), set())
        s.axis.set("z_axis_1", -0.31)
        self.assertEqual(violated(s, A, lb, ub, titles), {f"-0.3 < {s.axis.name}:z_axis_1 < 0.3"})

    def test_axis_radii_with_free_r_axis_0(self):
        s = SurfaceBSpline(axis_points=4)
        ax = s.axis.name
        A, lb, ub, titles = self.constraints(s, axis_r_max=1.2)
        s.axis.set("r_axis_2", 1.05)  # above r_axis_0 = 1
        self.assertEqual(violated(s, A, lb, ub, titles), {f"{ax}:r_axis_2 <= {ax}:r_axis_0"})
        s.axis.set("r_axis_2", 1.0)
        s.axis.set("r_axis_0", 1.25)
        self.assertEqual(violated(s, A, lb, ub, titles), {f"0 < {ax}:r_axis_0 < 1.2"})

    def test_axis_radii_with_fixed_r_axis_0(self):
        s = SurfaceBSpline(axis_points=4)
        s.axis.fix("r_axis_0")
        ax = s.axis.name
        A, lb, ub, titles = self.constraints(s, axis_r_fixed_max=1.1)
        self.assertFalse(any(f"{ax}:r_axis_0" in t for t in titles))
        s.axis.set("r_axis_1", 1.15)
        self.assertEqual(violated(s, A, lb, ub, titles), {f"0 < {ax}:r_axis_1 < 1.1"})

    def test_cross_sections_inside_every_axis_radius(self):
        s = SurfaceBSpline(axis_points=4, default_r=0.5)
        ax = s.axis.name
        A, lb, ub, titles = self.constraints(s)
        s.axis.set("r_axis_3", 0.45)  # below every cross-section radius
        bad = violated(s, A, lb, ub, titles)
        expected = {f"{cs.name}:r_{j} <= {ax}:r_axis_3" for cs in s.cs_list for j in range(cs.n_pts)}
        self.assertEqual(bad, expected)
        s.axis.set("r_axis_3", 1.0)
        s.cs_list[1].set("r_4", -0.01)
        self.assertEqual(violated(s, A, lb, ub, titles), {f"0 <= {s.cs_list[1].name}:r_4"})

    def test_cs_axis_ratio(self):
        s = SurfaceBSpline(axis_points=4, default_r=0.5)
        A, lb, ub, titles = self.constraints(s, cs_axis_ratio=2.0)  # 2 * 0.5 <= mean(1, 1, 1, 1)
        # the ratio replaces the per-axis-point rows r_j <= r_axis_m
        self.assertFalse(any(t.startswith("CrossSection") and ":r_axis_" in t for t in titles))
        name = f"{s.cs_list[1].name}:r_2"
        s.cs_list[1].set("r_2", 0.51)
        self.assertEqual(violated(s, A, lb, ub, titles), {f"2.0*{name} <= mean(r_axis)"})
        s.cs_list[1].set("r_2", 0.5)
        s.axis.set("r_axis_1", 0.9)  # lowers the mean below 2 * 0.5 for every cross-section radius
        self.assertEqual(len(violated(s, A, lb, ub, titles)), sum(cs.n_pts for cs in s.cs_list))

    def test_fixed_dofs_are_constants(self):
        s = SurfaceBSpline(axis_points=4, default_r=0.5)
        ax = s.axis.name
        s.axis.set("r_axis_1", 0.8)
        s.axis.fix("r_axis_1")
        s.axis.fix("z_axis_1")  # fixed axis heights get no row (and must not crash)
        A, lb, ub, titles = self.constraints(s)
        self.assertFalse(any("z_axis_1" in t for t in titles))
        cs = s.cs_list[1]
        cs.set("r_3", 0.81)  # above the fixed r_axis_1 = 0.8
        self.assertEqual(violated(s, A, lb, ub, titles), {f"{cs.name}:r_3 <= {ax}:r_axis_1"})

        s = SurfaceBSpline(axis_points=4)
        ax = s.axis.name
        s.axis.set("r_axis_2", 1.05)  # fixed above the free r_axis_0 = 1
        s.axis.fix("r_axis_2")
        A, lb, ub, titles = s.write_inequality_constraints()
        self.assertEqual(violated(s, A, lb, ub, titles), {f"{ax}:r_axis_2 <= {ax}:r_axis_0"})

    def test_cross_section_angle_steps(self):
        s = SurfaceBSpline(n_cs=3, cs_global_angle_free=True)
        A, lb, ub, titles = self.constraints(s, cs_angle_step_max=0.1)
        s.set("cs_angle1", 0.04)
        s.set("cs_angle2", -0.04)  # steps 0.04 and 0.08, and 0.08 across the mirror plane
        self.assertEqual(violated(s, A, lb, ub, titles), set())
        s.set("cs_angle2", 0.16)  # step 0.12 from cs_angle1, and 0.32 across the mirror plane
        self.assertEqual(violated(s, A, lb, ub, titles), {
            "|cs_angle2 - cs_angle1| <= 0.1",
            "|2*cs_angle2| <= 0.1 (step across the zeta = pi/nfp mirror plane)",
        })
        s.set("cs_angle2", 0.0)
        s.set("cs_angle1", 0.11)  # step from the fixed cs_angle0 = 0
        self.assertIn("|cs_angle1 - cs_angle0| <= 0.1", violated(s, A, lb, ub, titles))

    def test_fix_axis_angles(self):
        s = SurfaceBSpline()
        n = len(s.x)
        A, *_ = s.write_inequality_constraints(fix_axis_angles=True)
        self.assertEqual(len(s.x), n - (s.axis_points - 2))
        self.assertEqual(A.shape[1], len(s.x))
        self.assertFalse(any("zeta_axis" in name for name in s.dof_names))


class ConversionTests(unittest.TestCase):

    def test_to_RZFourier_matches_volume_and_area(self):
        s = perturbed_surface(p_u=3, p_v=3, M=8, N=8)
        rz = s.to_RZFourier(spec_cond=None)
        self.assertIsInstance(rz, SurfaceRZFourier)
        self.assertEqual((rz.nfp, rz.mpol, rz.ntor, rz.stellsym), (s.nfp, 8, 8, True))
        np.testing.assert_allclose(rz.volume(), s.volume(), rtol=1e-3)
        np.testing.assert_allclose(rz.area(), s.area(), rtol=1e-3)

    def test_spec_cond_wiring(self):
        """to_RZFourier returns the condensed surface, and partial options keep the other defaults."""
        s = SurfaceBSpline(M=4, N=4)
        for spec_cond, method, default_key in (
            ("variational", "variational_spec_cond", "shapetol"),
            ("direct", "condense_spectrum", "method"),
        ):
            with self.subTest(spec_cond=spec_cond):
                condensed = SurfaceRZFourier(nfp=s.nfp, mpol=4, ntor=4)
                ret = condensed if spec_cond == "variational" else (condensed, {})
                with mock.patch.object(SurfaceRZFourier, method, autospec=True, return_value=ret) as m:
                    out = s.to_RZFourier(spec_cond=spec_cond, spec_cond_options={"verbose": True})
                self.assertIs(out, condensed)
                kwargs = m.call_args.kwargs
                self.assertTrue(kwargs["verbose"])
                self.assertIn(default_key, kwargs)


@unittest.skipIf((MPI is None) or (vmec is None), "Valid Python interface to VMEC not found")
class OptimizationTests(unittest.TestCase):

    def test_stage_one_optimization(self):
        """A few steps of the stage_one_splines.py problem: QA quasisymmetry, aspect ratio and iota."""
        from monty.tempfile import ScratchDir
        from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
        from simsopt.objectives import LeastSquaresProblem
        from simsopt.solve import least_squares_mpi_solve
        from simsopt.util import MpiPartition

        mpi = MpiPartition()
        s = SurfaceBSpline(axis_points=3, points_per_cs=4, n_cs=3, nfp=2, M=4, N=4, p_u=3, p_v=3,
                           cs_global_angle_free=True, axis_angles_fixed=True, use_bishop_frame=True,
                           default_r=0.2)
        s.axis.fix("r_axis_0")
        with ScratchDir("."):
            vmec = Vmec.vmec_from_surf(surf=s, mpi=mpi, ns=9, M=4, N=4, ftol=1e-8)
            qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.25), helicity_m=1, helicity_n=0)
            prob = LeastSquaresProblem.from_tuples(
                [(qs.residuals, 0, 1), (vmec.aspect, 6, 1), (vmec.mean_iota, 0.42, 1)]
            )
            f0, aspect0 = prob.objective(), vmec.aspect()
            least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-9, abs_step=5e-6, max_nfev=3)
            f1, aspect1 = prob.objective(), vmec.aspect()

            self.assertLess(f1, 0.5 * f0)
            self.assertLess(abs(aspect1 - 6), abs(aspect0 - 6))
            self.assertTrue(np.all(s.x >= s.lower_bounds) and np.all(s.x <= s.upper_bounds))
            # VMEC's boundary is the optimized spline
            np.testing.assert_allclose(aspect1, s.to_RZFourier().aspect_ratio(), rtol=1e-6)


class SerializationTests(unittest.TestCase):

    def assert_same_surface(self, t, s):
        self.assertIsInstance(t, SurfaceBSpline)
        np.testing.assert_array_equal(t.gamma(), s.gamma())
        np.testing.assert_array_equal(t.full_x, s.full_x)
        np.testing.assert_array_equal(t.dofs_free_status, s.dofs_free_status)
        np.testing.assert_array_equal(t.lower_bounds, s.lower_bounds)
        np.testing.assert_array_equal(t.upper_bounds, s.upper_bounds)
        # children are restored as connected parents, so editing them changes the surface
        self.assertTrue(t.axis in t.ancestors and all(cs in t.ancestors for cs in t.cs_list))

    def test_json_round_trip(self):
        for kw in VARIANTS + [dict(nurbs=True), dict(poloidal_equispaced=True, axis_angles_fixed=True)]:
            with self.subTest(**kw):
                s = perturbed_surface(**kw)
                s.cs_list[1].fix("r_2")
                s.axis.set_upper_bound("r_axis_1", 1.7)
                self.assert_same_surface(json.loads(s.save(fmt="json"), cls=GSONDecoder), s)

    def test_file_round_trip_keeps_shared_references(self):
        s = perturbed_surface(use_bishop_frame=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spline.json"
            save([s, s.axis], path)
            t, axis = load(path)
        self.assert_same_surface(t, s)
        self.assertIs(axis, t.axis)

    def test_mismatched_children_raise(self):
        with self.assertRaises(ValueError):
            SurfaceBSpline(axis_points=4, axis=InteriorAxis(n_ctrl_pts=5, nfp=2))
        with self.assertRaises(ValueError):
            SurfaceBSpline(n_cs=2, cs_list=[CrossSectionFixedZeta(0, z_sym=True)])

    def test_reusing_children(self):
        """Passing existing children reproduces the surface and shares their dofs."""
        s = perturbed_surface(n_cs=3)
        t = SurfaceBSpline(n_cs=3, axis=s.axis, cs_list=s.cs_list)
        np.testing.assert_array_equal(t.gamma(), s.gamma())
        s.axis.set("r_axis_1", s.axis.get("r_axis_1") + 0.05)
        np.testing.assert_array_equal(t.gamma(), s.gamma())


@unittest.skipIf(matplotlib is None, "matplotlib not installed")
class PlotTests(unittest.TestCase):

    def test_matplotlib(self):
        import matplotlib.pyplot as plt
        s = SurfaceBSpline()
        ax = s.plot(show=False)
        self.assertEqual(ax.name, "3d")
        self.assertGreater(len(ax.lines), 0)  # axis and control net
        bare = s.plot(show=False, plot_axis=False, plot_control_net=False)
        self.assertEqual(len(bare.lines), 0)
        plt.close("all")

    @unittest.skipIf(plotly is None, "plotly not installed")
    def test_plotly(self):
        fig = SurfaceBSpline().plot(engine="plotly", show=False)
        self.assertEqual(fig.data[0].type, "surface")
        self.assertGreater(len(fig.data), 1)


if __name__ == "__main__":
    unittest.main()
