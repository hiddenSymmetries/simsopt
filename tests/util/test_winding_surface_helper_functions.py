"""
Unit tests for simsopt.util.winding_surface_helper_functions.

These tests cover current potential evaluation, contour utilities, geometry
mapping, and data loading. Tests use synthetic data where possible to avoid
file dependencies; a few tests require the regcoil_out.hsx.nc test file.
"""

import unittest
import numpy as np
from pathlib import Path

from simsopt.util.winding_surface_helper_functions import (
    load_from_CP_object,
    load_simsopt_regcoil_data,
    load_CP_and_geometries,
    current_potential_at_point,
    _grad_current_potential_at_point,
    genCPvals,
    genKvals,
    is_periodic_lines,
    minDist,
    sortLevels,
    chooseContours_matching_coilType,
    _points_in_polygon,
    map_data_to_more_periods,
    map_data_to_more_periods_3x1,
    _ID_mod_hel,
    ID_and_cut_contour_types,
    ID_halfway_contour,
    _removearray,
    compute_baseline_WP_currents,
    check_and_compute_nested_WP_currents,
    _real_space,
    REGCOIL_line_XYZ_RZ,
    SIMSOPT_line_XYZ_RZ,
    writeToCurve,
    writeToCurve_helical,
    load_regcoil_data,
    set_axes_equal,
    writeContourToFile,
    _load_surface_dofs_properly,
    make_onclick,
    make_onpick,
    make_on_key,
)
# Import private function for testing fallback path
from simsopt.util.winding_surface_helper_functions import _contour_paths

TEST_DIR = Path(__file__).resolve().parent.parent / "test_files"


def _make_args(Ip=0, It=0, nfp=4):
    """Create minimal args tuple for current potential evaluation."""
    xm = np.array([0, 1, 1])
    xn = np.array([0, 0, 1]) * nfp
    phi_cos = np.array([0.0, 0.5, 0.2])
    phi_sin = np.array([0.0, 0.1, -0.1])
    return (Ip, It, xm, xn, phi_cos, phi_sin, np.array([nfp]))


class TestCurrentPotentialAtPoint(unittest.TestCase):
    """Tests for current_potential_at_point."""

    def test_zero_modes(self):
        """With phi_cos=phi_sin=0 and Ip=It=0, potential is zero."""
        args = (0, 0, np.array([0]), np.array([0]), np.array([0.0]), np.array([0.0]), np.array([4]))
        val = current_potential_at_point(np.array([0.5, 0.3]), args)
        self.assertAlmostEqual(val, 0.0)

    def test_multi_valued_part(self):
        """Multi-valued part It*θ/(2π) + Ip*ζ/(2π) is correct."""
        args = (2 * np.pi, 4 * np.pi, np.array([0]), np.array([0]), np.array([0.0]), np.array([0.0]), np.array([4]))
        val = current_potential_at_point(np.array([0.5, 0.25]), args)
        # MV = It*θ/(2π) + Ip*ζ/(2π) = 4π*0.5/(2π) + 2π*0.25/(2π) = 1 + 0.25 = 1.25
        self.assertAlmostEqual(val, 1.25)

    def test_fourier_part(self):
        """Single mode cos(mθ - nζ) gives correct value."""
        args = (0, 0, np.array([1]), np.array([0]), np.array([1.0]), np.array([0.0]), np.array([4]))
        val = current_potential_at_point(np.array([0.0, 0.0]), args)
        self.assertAlmostEqual(val, 1.0)

    def test_scalar_input(self):
        """x can be list or array; returns float."""
        args = _make_args()
        v1 = current_potential_at_point([0.5, 0.3], args)
        v2 = current_potential_at_point(np.array([0.5, 0.3]), args)
        self.assertIsInstance(v1, float)
        self.assertAlmostEqual(v1, v2)


class TestGradCurrentPotentialAtPoint(unittest.TestCase):
    """Tests for _grad_current_potential_at_point."""

    def test_constant_potential_zero_grad(self):
        """Constant potential has zero gradient."""
        args = (0, 0, np.array([0]), np.array([0]), np.array([0.0]), np.array([0.0]), np.array([4]))
        val = _grad_current_potential_at_point(np.array([0.5, 0.3]), args)
        self.assertAlmostEqual(val, 0.0)

    def test_returns_positive(self):
        """|∇φ| is non-negative."""
        args = _make_args()
        val = _grad_current_potential_at_point(np.array([0.5, 0.3]), args)
        self.assertGreaterEqual(val, 0)


class TestGenCPvals(unittest.TestCase):
    """Tests for genCPvals."""

    def test_output_shapes(self):
        """Output arrays have correct shapes."""
        args = _make_args()
        thetas, zetas, phi, phi_SV, phi_NSV, ARGS = genCPvals(
            (0, 2 * np.pi), (0, 2 * np.pi / 4), (8, 4), args
        )
        self.assertEqual(len(thetas), 8)
        self.assertEqual(len(zetas), 4)
        self.assertEqual(phi.shape, (4, 8))  # .T so (nZ, nT)
        self.assertEqual(phi_SV.shape, (4, 8))
        self.assertEqual(phi_NSV.shape, (4, 8))

    def test_single_valued_vs_full(self):
        """With Ip=It=0, phi matches phi_SV."""
        args = _make_args(Ip=0, It=0)
        _, _, phi, phi_SV, _, _ = genCPvals(
            (0, 2 * np.pi), (0, 2 * np.pi / 4), (4, 4), args
        )
        np.testing.assert_allclose(phi, phi_SV)


class TestGenKvals(unittest.TestCase):
    """Tests for genKvals."""

    def test_output_shapes(self):
        """Output arrays have correct shapes."""
        args = _make_args()
        thetas, zetas, K, K_SV, K_NSV, ARGS = genKvals(
            (0, 2 * np.pi), (0, 2 * np.pi / 4), (6, 3), args
        )
        self.assertEqual(K.shape, (3, 6))
        self.assertGreaterEqual(np.min(K), 0)


class TestIsPeriodicLines(unittest.TestCase):
    """Tests for is_periodic_lines."""

    def test_closed_contour(self):
        """Contour that starts and ends at same point is periodic."""
        line = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        self.assertTrue(is_periodic_lines(line, tol=0.1))

    def test_open_contour(self):
        """Contour with different endpoints is not periodic."""
        line = np.array([[0, 0], [1, 0], [1, 1], [0.5, 0.5]])
        self.assertFalse(is_periodic_lines(line, tol=0.01))

    def test_nearly_closed(self):
        """Within tolerance counts as closed."""
        line = np.array([[0, 0], [1, 0], [1, 1], [0, 0.001]])
        self.assertTrue(is_periodic_lines(line, tol=0.1))


class TestMinDist(unittest.TestCase):
    """Tests for minDist."""

    def test_point_on_line(self):
        """Distance from point on line is zero."""
        pt = np.array([1.0, 0.0])
        line = np.array([[0, 0], [1, 0], [2, 0]])
        self.assertAlmostEqual(minDist(pt, line), 0.0)

    def test_point_off_line(self):
        """Distance from point to nearest point in line (discretized path)."""
        pt = np.array([1.0, 1.0])
        # Line from (0,0) to (2,0) - include (1,0) so closest point gives distance 1
        line = np.array([[0, 0], [1, 0], [2, 0]])
        self.assertAlmostEqual(minDist(pt, line), 1.0)


class TestSortLevels(unittest.TestCase):
    """Tests for sortLevels."""

    def test_ascending_order(self):
        """Levels and points are sorted by level ascending."""
        levels = [3, 1, 2]
        points = [np.array([3, 3]), np.array([1, 1]), np.array([2, 2])]
        L, P = sortLevels(levels, points)
        self.assertEqual(L, [1, 2, 3])
        np.testing.assert_array_almost_equal(P[0], [1, 1])
        np.testing.assert_array_almost_equal(P[1], [2, 2])
        np.testing.assert_array_almost_equal(P[2], [3, 3])


class TestChooseContoursMatchingCoilType(unittest.TestCase):
    """Tests for chooseContours_matching_coilType."""

    def test_free_returns_all(self):
        """ctype='free' returns all lines."""
        closed = np.array([[0, 0], [1, 0], [0, 0]])
        open_ = np.array([[0, 0], [1, 1]])
        lines = [closed, open_]
        result = chooseContours_matching_coilType(lines, 'free')
        self.assertEqual(len(result), 2)

    def test_wp_returns_only_closed(self):
        """ctype='wp' returns only closed contours."""
        closed = np.array([[0, 0], [1, 0], [0, 0]])
        open_ = np.array([[0, 0], [1, 1]])
        lines = [closed, open_]
        result = chooseContours_matching_coilType(lines, 'wp')
        self.assertEqual(len(result), 1)
        np.testing.assert_array_almost_equal(result[0], closed)

    def test_mod_returns_only_open(self):
        """ctype='mod' returns only open contours."""
        closed = np.array([[0, 0], [1, 0], [0, 0]])
        open_ = np.array([[0, 0], [1, 1]])
        lines = [closed, open_]
        result = chooseContours_matching_coilType(lines, 'mod')
        self.assertEqual(len(result), 1)
        np.testing.assert_array_almost_equal(result[0], open_)

    def test_hel_returns_only_open(self):
        """ctype='hel' returns only open contours (same as mod)."""
        closed = np.array([[0, 0], [1, 0], [0, 0]])
        open_ = np.array([[0, 0], [1, 1]])
        lines = [closed, open_]
        result = chooseContours_matching_coilType(lines, 'hel')
        self.assertEqual(len(result), 1)
        np.testing.assert_array_almost_equal(result[0], open_)


class TestPointsInPolygon(unittest.TestCase):
    """Tests for _points_in_polygon."""

    def test_triangle_contains_inside(self):
        """Points inside triangle are identified."""
        # Triangle (0,0), (1,0), (0.5,1)
        polygon = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])
        t = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        i1, i2, BOOL = _points_in_polygon((t, z), None, polygon)
        # Center (0.5, 0.33) should be inside
        self.assertGreater(np.sum(BOOL), 0)

    def test_returns_correct_lengths(self):
        """Returned indices match BOOL mask."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        t = np.linspace(0.1, 0.9, 3)
        z = np.linspace(0.1, 0.9, 3)
        i1, i2, BOOL = _points_in_polygon((t, z), None, polygon)
        self.assertEqual(len(i1), np.sum(BOOL))
        self.assertEqual(len(i2), np.sum(BOOL))


class TestMapDataToMorePeriods(unittest.TestCase):
    """Tests for map_data_to_more_periods and map_data_to_more_periods_3x1."""

    def test_3x3_shape(self):
        """3x3 extension has correct shape."""
        args = _make_args()
        x = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        y = np.linspace(0, 2 * np.pi / 4, 3, endpoint=False)
        z = np.random.rand(3, 5)
        xN, yN, ret = map_data_to_more_periods(x, y, z, args)
        self.assertEqual(len(xN), 5 * 3)
        self.assertEqual(len(yN), 3 * 3)
        self.assertEqual(ret.shape, (9, 15))

    def test_3x1_shape(self):
        """3x1 extension has correct shape."""
        args = _make_args()
        x = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        y = np.linspace(0, 2 * np.pi / 4, 3, endpoint=False)
        z = np.random.rand(3, 4)
        xN, yN, ret = map_data_to_more_periods_3x1(x, y, z, args)
        self.assertEqual(len(xN), 4)
        self.assertEqual(len(yN), 9)
        self.assertEqual(ret.shape, (9, 4))


class TestIDModHel(unittest.TestCase):
    """Tests for _ID_mod_hel."""

    def test_modular_contour(self):
        """Contour with same cos(θ) and ζ at ends is modular."""
        # Start (0, 0), end (2π, 0) -> cos(0)=cos(2π)=1, zeta same
        contour = np.array([[0, 0], [np.pi, 0.5], [2 * np.pi, 0]])
        names, ints = _ID_mod_hel([contour], tol=0.1)
        self.assertEqual(names[0], 'mod')
        self.assertEqual(ints[0], 1)

    def test_helical_contour(self):
        """Contour with θ spanning 2π is helical."""
        contour = np.array([[0, 0], [np.pi, 0.25], [2 * np.pi, 0.5]])
        names, ints = _ID_mod_hel([contour], tol=0.05)
        self.assertEqual(names[0], 'hel')
        self.assertEqual(ints[0], 2)

    def test_vacuum_field_contour(self):
        """Contour with neither mod nor hel endpoints is vacuum-field (vf)."""
        # θ spans ~π, ζ differs; neither cos(θ) match nor |θ0-θf|≈2π
        contour = np.array([[0, 0], [np.pi / 2, 0.5], [np.pi, 1.0]])
        names, ints = _ID_mod_hel([contour], tol=0.05)
        self.assertEqual(names[0], 'vf')
        self.assertEqual(ints[0], 3)


class TestIDAndCutContourTypes(unittest.TestCase):
    """Tests for ID_and_cut_contour_types."""

    def test_mixed_contours(self):
        """Closed and open contours are classified correctly."""
        closed = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        open_ = np.array([[0, 0], [1, 1]])
        open_contours, closed_contours, types = ID_and_cut_contour_types([closed, open_])
        self.assertEqual(len(closed_contours), 1)
        self.assertEqual(len(open_contours), 1)
        self.assertEqual(types[0], 0)
        self.assertIn(types[1], (1, 2, 3))


class TestIDHalfwayContour(unittest.TestCase):
    """Tests for ID_halfway_contour."""

    def test_halfway_contours_between_open(self):
        """Halfway contours are found between consecutive open contours."""
        args = _make_args()
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / 4, 15, endpoint=False)
        _, _, cpd, _, _, _ = genCPvals((0, 2 * np.pi), (0, 2 * np.pi / 4), (20, 15), args)
        c1 = np.array([[0.5, 0.3], [1.5, 0.4]])
        c2 = np.array([[1.0, 0.5], [2.0, 0.6]])
        contours = [c1, c2]
        data = (theta, zeta, cpd)
        halfway = ID_halfway_contour(contours, data, do_plot=False, args=args)
        self.assertEqual(len(halfway), 1)
        self.assertGreater(len(halfway[0]), 0)

    def test_halfway_contours_do_plot_true(self):
        """ID_halfway_contour with do_plot=True sets up axes and returns contours."""
        from unittest.mock import patch
        args = _make_args()
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / 4, 15, endpoint=False)
        _, _, cpd, _, _, _ = genCPvals((0, 2 * np.pi), (0, 2 * np.pi / 4), (20, 15), args)
        c1 = np.array([[0.5, 0.3], [1.5, 0.4]])
        c2 = np.array([[1.0, 0.5], [2.0, 0.6]])
        contours = [c1, c2]
        data = (theta, zeta, cpd)
        with patch('simsopt.util.winding_surface_helper_functions.plt.show'):
            halfway = ID_halfway_contour(contours, data, do_plot=True, args=args)
        self.assertEqual(len(halfway), 1)


class TestRemoveArray(unittest.TestCase):
    """Tests for _removearray."""

    def test_remove_existing(self):
        """Removing existing array modifies list in place."""
        a = np.array([1, 2])
        b = np.array([3, 4])
        L = [a, b]
        _removearray(L, a)
        self.assertEqual(len(L), 1)
        np.testing.assert_array_equal(L[0], b)

    def test_remove_raises_if_not_found(self):
        """Removing non-existent array raises ValueError."""
        L = [np.array([1, 2])]
        with self.assertRaises(ValueError):
            _removearray(L, np.array([9, 9]))


class TestComputeBaselineWPCurrents(unittest.TestCase):
    """Tests for compute_baseline_WP_currents."""

    def test_single_contour(self):
        """Single closed contour returns one current."""
        args = _make_args()
        contour = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]])
        wp_currents, max_vals, func_vals = compute_baseline_WP_currents([contour], args, plot=False)
        self.assertEqual(len(wp_currents), 1)
        self.assertGreater(wp_currents[0], 0)

    def test_single_contour_plot_true(self):
        """compute_baseline_WP_currents with plot=True runs without error."""
        from unittest.mock import patch
        args = _make_args()
        contour = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]])
        with patch('simsopt.util.winding_surface_helper_functions.plt.show'):
            wp_currents, _, _ = compute_baseline_WP_currents([contour], args, plot=True)
        self.assertEqual(len(wp_currents), 1)


class TestCheckAndComputeNestedWPCurrents(unittest.TestCase):
    """Tests for check_and_compute_nested_WP_currents."""

    def test_no_nesting(self):
        """Non-nested contours return unchanged currents."""
        args = _make_args()
        c1 = np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0], [0.5, 0.5]])
        c2 = np.array([[2.0, 2.0], [2.5, 2.0], [2.5, 2.5], [2.0, 2.5], [2.0, 2.0]])
        wp = [1.0, 1.0]
        result, nested, fv, nc = check_and_compute_nested_WP_currents([c1, c2], wp, args, plot=False)
        np.testing.assert_array_almost_equal(result, wp)

    def test_nested_contours(self):
        """Nested contours get adjusted currents from potential difference."""
        args = _make_args()
        # Outer square contains inner square
        outer = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]])
        inner = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]])
        wp = [1.0, 1.0]  # Will be overwritten for outer
        result, nested, fv, nc = check_and_compute_nested_WP_currents([outer, inner], wp, args, plot=False)
        self.assertIsNotNone(nested)
        self.assertTrue(nested[0, 1])  # outer contains inner
        self.assertEqual(len(result), 2)

    def test_nested_contours_plot_true(self):
        """check_and_compute_nested_WP_currents with plot=True runs without error."""
        from unittest.mock import patch
        args = _make_args()
        outer = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]])
        inner = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]])
        wp = [1.0, 1.0]
        with patch('simsopt.util.winding_surface_helper_functions.plt.show'):
            result, nested, _, _ = check_and_compute_nested_WP_currents(
                [outer, inner], wp, args, plot=True
            )
        self.assertEqual(len(result), 2)
        self.assertTrue(nested[0, 1])


class TestRealSpace(unittest.TestCase):
    """Tests for _real_space (REGCOIL format)."""

    def test_circular_cross_section(self):
        """Simple m=0, n=0 mode gives circular R."""
        # R = R0 + r*cos(θ), Z = 0 for m=n=0
        # REGCOIL format: nnum=4*n, mnum=m; columns 2,4,5,3 for crc,crs,czc,czs
        nharmonics = 1
        coeff_array = np.array([[0, 0, 1.0, 0, 0, 0]])  # R=1, Z=0
        polAng = np.array([0, np.pi / 2, np.pi])
        torAng = np.array([0, 0, 0])
        X, Y, R, Z = _real_space(nharmonics, coeff_array, polAng, torAng)
        np.testing.assert_allclose(R, [1, 1, 1])
        np.testing.assert_allclose(Z, [0, 0, 0])


class TestREGCOILLineXYZRZ(unittest.TestCase):
    """Tests for REGCOIL_line_XYZ_RZ."""

    def test_output_shape(self):
        """Output arrays have correct length."""
        nharmonics = 1
        coeff_array = np.array([[0, 0, 1.0, 0, 0, 0]])
        polAng = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        torAng = np.zeros(10)
        X, Y, R, Z = REGCOIL_line_XYZ_RZ(nharmonics, coeff_array, polAng, torAng)
        self.assertEqual(len(X), 10)
        self.assertEqual(len(R), 10)


class TestSIMSOPTLineXYZRZ(unittest.TestCase):
    """Tests for SIMSOPT_line_XYZ_RZ."""

    def test_output_shape(self):
        """Output arrays have correct length for SurfaceRZFourier."""
        from simsopt.geo import SurfaceRZFourier
        surf = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        surf.set_dofs(np.zeros(len(surf.get_dofs())))
        npts = 8
        theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / 4, npts, endpoint=False)
        X, Y, R, Z = SIMSOPT_line_XYZ_RZ(surf, (theta, zeta))
        self.assertEqual(len(X), npts)
        self.assertEqual(len(R), npts)

    def test_points_on_surface(self):
        """Points from SIMSOPT_line_XYZ_RZ lie on the surface (uses gamma_lin)."""
        from simsopt.geo import SurfaceRZFourier
        surf = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        npts = 12
        theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / 4, npts, endpoint=False)
        X, Y, R, Z = SIMSOPT_line_XYZ_RZ(surf, (theta, zeta))
        quadpoints_theta = np.mod(theta / (2 * np.pi), 1.0)
        quadpoints_phi = np.mod(zeta / (2 * np.pi), 1.0)
        gamma = np.zeros((npts, 3))
        surf.gamma_lin(gamma, quadpoints_phi, quadpoints_theta)
        np.testing.assert_allclose(X, gamma[:, 0], atol=1e-14)
        np.testing.assert_allclose(Y, gamma[:, 1], atol=1e-14)
        np.testing.assert_allclose(Z, gamma[:, 2], atol=1e-14)


class TestWriteToCurve(unittest.TestCase):
    """Tests for writeToCurve."""

    def test_curve_created(self):
        """writeToCurve creates a CurveXYZFourier that approximates the contour."""
        from simsopt.geo import SurfaceRZFourier
        surf = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        npts = 32
        theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / 4, npts, endpoint=False)
        contour = np.column_stack([theta, zeta])
        curve = writeToCurve(contour, [], fourier_trunc=10, winding_surface=surf)
        gamma = curve.gamma()
        self.assertEqual(gamma.shape[1], 3)
        # Curve should be in a reasonable range (surface has R~1)
        self.assertGreater(np.max(np.linalg.norm(gamma, axis=1)), 0.5)
        self.assertLess(np.max(np.linalg.norm(gamma, axis=1)), 2.0)

    def test_writeToCurve_plotting_and_fix_stellarator_symmetry(self):
        """writeToCurve with plotting_args=(1,) and fix_stellarator_symmetry=True."""
        from simsopt.geo import SurfaceRZFourier
        from unittest.mock import patch
        surf = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        n_pts = 50
        t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        theta, zeta = t, t  # Helical: closes in 3D
        contour = np.column_stack([theta, zeta])
        contour_xyz = SIMSOPT_line_XYZ_RZ(surf, [theta, zeta])
        with patch('simsopt.util.winding_surface_helper_functions.plt.show'):
            curve = writeToCurve(
                contour, contour_xyz, fourier_trunc=10,
                plotting_args=(1,), fix_stellarator_symmetry=True
            )
        gamma = curve.gamma()
        self.assertEqual(gamma.shape[1], 3)


def _max_distance_to_contour(points, contour_xyz):
    """Max distance from any point to nearest point on contour (one-sided Hausdorff)."""
    # contour_xyz is [X, Y, R, Z] or [X, Y, Z]
    X, Y = contour_xyz[0], contour_xyz[1]
    Z = contour_xyz[3] if len(contour_xyz) > 3 else contour_xyz[2]
    xyz = np.column_stack([X, Y, Z])
    dists = np.min(np.linalg.norm(points[:, None, :] - xyz[None, :, :], axis=2), axis=1)
    return np.max(dists)


class TestWriteToCurveHelical(unittest.TestCase):
    """Tests for writeToCurve_helical. Note: cut_coils uses writeToCurve for helical coils."""

    def test_helical_curve_closes_and_reasonable_shape(self):
        """writeToCurve_helical produces a closed curve with reasonable geometry."""
        nfp = 4
        n_pts = 200
        phi = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
        phi[-1] = 2 * np.pi
        R0, a = 1.0, 0.3
        R = R0 + a * np.cos(phi)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        Z = a * np.sin(phi)
        contour_xyz = [X, Y, R, Z]
        curve = writeToCurve_helical(contour_xyz, fourier_order=8, nfp=nfp, stellsym=True, ntor=1)
        gamma = curve.gamma()
        self.assertEqual(gamma.shape[1], 3)
        # Curve is effectively closed (quadpoints may not include t=1)
        self.assertLess(
            np.linalg.norm(gamma[0] - gamma[-1]), 0.02,
            msg="CurveXYZFourierSymmetries should be nearly periodic"
        )
        self.assertGreater(np.max(np.linalg.norm(gamma, axis=1)), 0.5)
        self.assertLess(np.max(np.linalg.norm(gamma, axis=1)), 2.0)

    def test_writeToCurve_helical_ntor_nfp_not_coprime_raises(self):
        """writeToCurve_helical raises when ntor and nfp are not coprime."""
        n = 20
        contour_xyz = [
            np.ones(n), np.zeros(n), np.ones(n), np.zeros(n)
        ]
        with self.assertRaises(ValueError) as cm:
            writeToCurve_helical(contour_xyz, fourier_order=4, nfp=4, ntor=2)
        self.assertIn("coprime", str(cm.exception))

    def test_writeToCurve_helical_open_contour(self):
        """writeToCurve_helical handles open (non-closing) contour."""
        n_pts = 50
        phi = np.linspace(0, 1.5 * np.pi, n_pts, endpoint=False)  # Not full circle
        R0, a = 1.0, 0.3
        R = R0 + a * np.cos(phi)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        Z = a * np.sin(phi)
        contour_xyz = [X, Y, R, Z]
        curve = writeToCurve_helical(contour_xyz, fourier_order=6, nfp=4, stellsym=True, ntor=1)
        gamma = curve.gamma()
        self.assertEqual(gamma.shape[1], 3)


class TestWriteToCurveHelicalClosed(unittest.TestCase):
    """Test writeToCurve with helical contour (3D closure, not θζ closure)."""

    def test_writeToCurve_helical_contour_approximates(self):
        """writeToCurve with 3D-closed helical contour approximates the input."""
        from simsopt.geo import SurfaceRZFourier
        surf = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        n_pts = 150
        # Helical: θ and ζ increase together, contour closes in 3D but not in (θ,ζ)
        t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        theta = t
        zeta = t
        contour = np.column_stack([theta, zeta])
        contour_xyz = SIMSOPT_line_XYZ_RZ(surf, [theta, zeta])
        curve = writeToCurve(contour, contour_xyz, fourier_trunc=15, winding_surface=surf)
        gamma = curve.gamma()
        X, Y, Z = contour_xyz[0], contour_xyz[1], contour_xyz[3]
        err = _max_distance_to_contour(gamma, [X, Y, Z])
        self.assertLess(err, 0.25, msg=f"Curve should approximate helical input; max dist={err:.4f}")


class TestHelicalExtensionCloses(unittest.TestCase):
    """Sanity checks: extended helical contour closes in 3D; curve returns to start."""

    def test_extended_helical_contour_closes_in_3d(self):
        """Helical contour extended by +j*2π/nfp and closing point closes in 3D."""
        from simsopt.geo import SurfaceRZFourier
        surf = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        nfp = 4
        n_pts = 50
        zeta_1fp = np.linspace(0, 2 * np.pi / nfp, n_pts, endpoint=False)
        theta_1fp = 0.5 * zeta_1fp
        contour_1fp = np.column_stack([theta_1fp, zeta_1fp])
        L = len(contour_1fp)
        d = 2 * np.pi / nfp
        Contour = np.zeros((L * nfp, 2))
        for j in range(nfp):
            j1, j2 = j * L, (j + 1) * L
            Contour[j1:j2, 0] = contour_1fp[:, 0]
            Contour[j1:j2, 1] = contour_1fp[:, 1] + j * d
        # Force closure: append first point with ζ+2π*nfp (surface period 2π*nfp in zeta)
        first_pt = np.array([[contour_1fp[0, 0], contour_1fp[0, 1] + 2 * np.pi * nfp]])
        Contour = np.vstack([Contour, first_pt])
        X, Y, R, Z = SIMSOPT_line_XYZ_RZ(surf, [Contour[:, 0], Contour[:, 1]])
        gap = np.linalg.norm(np.array([X[-1], Y[-1], Z[-1]]) - np.array([X[0], Y[0], Z[0]]))
        self.assertLess(gap, 0.01, msg=f"Helical contour must close in 3D; gap={gap:.4f}")

    def test_helical_curve_returns_to_start(self):
        """Curve from closed helical contour has gamma(0) ≈ gamma(1)."""
        from simsopt.geo import SurfaceRZFourier
        surf = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        nfp = 4
        n_pts = 40
        zeta_1fp = np.linspace(0, 2 * np.pi / nfp, n_pts, endpoint=False)
        theta_1fp = 0.5 * zeta_1fp
        contour_1fp = np.column_stack([theta_1fp, zeta_1fp])
        L = len(contour_1fp)
        d = 2 * np.pi / nfp
        Contour = np.zeros((L * nfp, 2))
        for j in range(nfp):
            j1, j2 = j * L, (j + 1) * L
            Contour[j1:j2, 0] = contour_1fp[:, 0]
            Contour[j1:j2, 1] = contour_1fp[:, 1] + j * d
        first_pt = np.array([[contour_1fp[0, 0], contour_1fp[0, 1] + 2 * np.pi * nfp]])
        Contour = np.vstack([Contour, first_pt])
        contour_xyz = list(SIMSOPT_line_XYZ_RZ(surf, [Contour[:, 0], Contour[:, 1]]))
        curve = writeToCurve(Contour, contour_xyz, fourier_trunc=12, winding_surface=surf)
        gamma = curve.gamma()
        closure_err = np.linalg.norm(gamma[0] - gamma[-1])
        self.assertLess(closure_err, 0.05, msg=f"Helical curve must close; |gamma[0]-gamma[-1]|={closure_err:.4f}")


class TestCutCoilsHelicalClosure(unittest.TestCase):
    """Integration test: cut_coils helical coils close (run with regcoil data)."""

    def test_cut_coils_helical_coils_close(self):
        """Helical coils from cut_coils close: first point ≈ last point."""
        fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest(f"Test file not found: {fpath}")
        # Import cut_coils from examples (no package structure)
        import sys
        examples_dir = Path(__file__).resolve().parents[2] / "examples" / "3_Advanced"
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        from cut_coils import run_cut_coils
        coils = run_cut_coils(
            surface_filename=fpath,
            ilambda=6,
            single_valued=False,
            show_final_coilset=False,
            show_plots=False,
            write_coils_to_file=False,
        )
        # Check each coil's curve closes (helical coils are typically last)
        for i, coil in enumerate(coils):
            gamma = coil.curve.gamma()
            closure_err = np.linalg.norm(gamma[0] - gamma[-1])
            self.assertLess(
                closure_err, 0.1,
                msg=f"Coil {i+1} must close; |gamma[0]-gamma[-1]|={closure_err:.4f}"
            )

    def test_cut_coils_no_center_crossing(self):
        """Coils must not cross through the center (R_min > threshold); detects X artifact."""
        fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest(f"Test file not found: {fpath}")
        import sys
        examples_dir = Path(__file__).resolve().parents[2] / "examples" / "3_Advanced"
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        from cut_coils import run_cut_coils
        coils = run_cut_coils(
            surface_filename=fpath,
            ilambda=6,
            single_valued=False,
            show_final_coilset=False,
            show_plots=False,
            write_coils_to_file=False,
            curve_fourier_cutoff=40,
        )
        # Coils lie on winding surface; R = sqrt(x^2+y^2) must stay away from axis
        R_min_acceptable = 0.3  # Winding surface is typically R > 0.5; allow some margin
        for i, coil in enumerate(coils):
            gamma = coil.curve.gamma()
            R = np.sqrt(gamma[:, 0]**2 + gamma[:, 1]**2)
            R_min = np.min(R)
            self.assertGreater(
                R_min, R_min_acceptable,
                msg=f"Coil {i+1} crosses center: R_min={R_min:.4f} (expected >{R_min_acceptable})"
            )

    def test_cut_coils_on_winding_surface(self):
        """Extracted coils must lie on the winding surface within tolerance (helical stitching)."""
        fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest(f"Test file not found: {fpath}")
        import sys
        from scipy.spatial import cKDTree
        examples_dir = Path(__file__).resolve().parents[2] / "examples" / "3_Advanced"
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        from cut_coils import run_cut_coils
        cpst, s_coil_fp, s_coil_full, s_plasma_fp, s_plasma_full = load_CP_and_geometries(
            str(fpath), plot_flags=(0, 0, 0, 0)
        )
        coils = run_cut_coils(
            surface_filename=fpath,
            ilambda=6,
            single_valued=False,
            show_final_coilset=False,
            show_plots=False,
            write_coils_to_file=False,
            curve_fourier_cutoff=40,
        )
        # Dense surface sampling for distance check
        ntheta, nphi = 64, 64 * s_coil_full.nfp
        theta = np.linspace(0, 1, ntheta, endpoint=False)
        phi = np.linspace(0, 1, nphi, endpoint=False)
        data = np.zeros((ntheta * nphi, 3))
        s_coil_full.gamma_lin(data, np.repeat(phi, ntheta), np.tile(theta, nphi))
        surf_pts = data
        tree = cKDTree(surf_pts)
        dist_tol = 0.08  # Coils within 8cm of winding surface (Fourier fit tolerance)
        for i, coil in enumerate(coils):
            gamma = coil.curve.gamma()
            d, _ = tree.query(gamma, k=1)
            max_d = np.max(d)
            argmax_d = np.argmax(d)
            self.assertLess(
                max_d, dist_tol,
                msg=f"Coil {i+1} has points off surface: max_dist={max_d:.4f} at pt {argmax_d} "
                f"(tol={dist_tol}); xyz={gamma[argmax_d]}"
            )


class TestLoadSimsoptRegcoilData(unittest.TestCase):
    """Tests for load_simsopt_regcoil_data."""

    def test_load_simsopt_regcoil_file(self):
        """load_simsopt_regcoil_data loads simsopt-regcoil format."""
        from unittest.mock import MagicMock, patch

        def _mock_var(val):
            m = MagicMock()
            m.__getitem__ = MagicMock(return_value=val)
            return m

        mock_f = MagicMock()
        mock_f.__enter__ = MagicMock(return_value=mock_f)
        mock_f.__exit__ = MagicMock(return_value=False)
        mock_f.variables = {
            'nfp': _mock_var(4),
            'ntheta_coil': _mock_var(32),
            'nzeta_coil': _mock_var(16),
            'theta_coil': _mock_var(np.linspace(0, 2*np.pi, 32, endpoint=False)),
            'zeta_coil': _mock_var(np.linspace(0, 2*np.pi/4, 16, endpoint=False)),
            'r_coil': _mock_var(np.ones(32*16)),
            'xm_coil': _mock_var(np.array([0, 1])),
            'xn_coil': _mock_var(np.array([0, 1])*4),
            'xm_potential': _mock_var(np.array([0, 1])),
            'xn_potential': _mock_var(np.array([0, 1])*4),
            'net_poloidal_current_amperes': _mock_var(np.array([0.0])),
            'net_toroidal_current_amperes': _mock_var(np.array([0.0])),
            'single_valued_current_potential_mn': _mock_var(np.zeros((1, 2))),
            'single_valued_current_potential_thetazeta': _mock_var(np.zeros((1, 16, 32))),
            'lambda': _mock_var(np.array([1e-10])),
            'K2': _mock_var(np.zeros((1, 16, 32))),
            'chi2_B': _mock_var(np.array([0.1])),
            'chi2_K': _mock_var(np.array([0.01])),
        }
        with patch('simsopt.util.winding_surface_helper_functions.netcdf_file', return_value=mock_f):
            data = load_simsopt_regcoil_data('/fake/path.nc', sparse=False)
        self.assertEqual(len(data), 19)
        self.assertEqual(data[0], 32)
        self.assertEqual(data[1], 16)

    def test_load_simsopt_regcoil_sparse(self):
        """load_simsopt_regcoil_data with sparse=True loads L1 variables."""
        from unittest.mock import MagicMock, patch

        def _mock_var(val):
            m = MagicMock()
            m.__getitem__ = MagicMock(return_value=val)
            return m

        mock_f = MagicMock()
        mock_f.__enter__ = MagicMock(return_value=mock_f)
        mock_f.__exit__ = MagicMock(return_value=False)
        mock_f.variables = {
            'nfp': _mock_var(4), 'ntheta_coil': _mock_var(16), 'nzeta_coil': _mock_var(8),
            'theta_coil': _mock_var(np.linspace(0, 2*np.pi, 16, endpoint=False)),
            'zeta_coil': _mock_var(np.linspace(0, 2*np.pi/4, 8, endpoint=False)),
            'r_coil': _mock_var(np.ones(128)), 'xm_coil': _mock_var(np.array([0, 1])),
            'xn_coil': _mock_var(np.array([0, 1])*4), 'xm_potential': _mock_var(np.array([0, 1])),
            'xn_potential': _mock_var(np.array([0, 1])*4),
            'net_poloidal_current_amperes': _mock_var(np.array([0.0])),
            'net_toroidal_current_amperes': _mock_var(np.array([0.0])),
            'single_valued_current_potential_mn_l1': _mock_var(np.zeros((1, 2))),
            'single_valued_current_potential_thetazeta_l1': _mock_var(np.zeros((1, 8, 16))),
            'lambda': _mock_var(np.array([1e-10])), 'K2_l1': _mock_var(np.zeros((1, 8, 16))),
            'chi2_B_l1': _mock_var(np.array([0.1])), 'chi2_K_l1': _mock_var(np.array([0.01])),
        }
        with patch('simsopt.util.winding_surface_helper_functions.netcdf_file', return_value=mock_f):
            data = load_simsopt_regcoil_data('/fake/path.nc', sparse=True)
        self.assertEqual(len(data), 19)
        self.assertEqual(data[0], 16)


class TestLoadFromCPObject(unittest.TestCase):
    """Tests for load_from_CP_object."""

    def test_load_from_CP_object_use_l2(self):
        """load_from_CP_object with use_l1=False returns L2 data."""
        fpath = TEST_DIR / "regcoil_out.w7x_infty.nc"
        if not fpath.exists():
            fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest("No regcoil test file found")
        from simsopt.field import CurrentPotentialSolve
        cpst = CurrentPotentialSolve.from_netcdf(fpath, 1.0, 1.0, 1.0, 1.0)
        data = load_from_CP_object(cpst, use_l1=False)
        self.assertEqual(len(data), 19)
        ntheta, nzeta = data[0], data[1]
        self.assertGreater(ntheta, 0)
        self.assertGreater(nzeta, 0)

    def test_load_from_CP_object_use_l1(self):
        """load_from_CP_object with use_l1=True returns L1 data when available."""
        fpath = TEST_DIR / "regcoil_out.w7x_infty.nc"
        if not fpath.exists():
            fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest("No regcoil test file found")
        from simsopt.field import CurrentPotentialSolve
        cpst = CurrentPotentialSolve.from_netcdf(fpath, 1.0, 1.0, 1.0, 1.0)
        if not hasattr(cpst, 'dofs_l1') or cpst.dofs_l1 is None:
            self.skipTest("CP object has no L1 solution")
        data = load_from_CP_object(cpst, use_l1=True)
        self.assertEqual(len(data), 19)


class TestLoadCPAndGeometries(unittest.TestCase):
    """Tests for load_CP_and_geometries."""

    def test_load_CP_and_geometries_loadDOFsProperly_false(self):
        """load_CP_and_geometries with loadDOFsProperly=False skips DOF copy."""
        fpath = TEST_DIR / "regcoil_out.w7x_infty.nc"
        if not fpath.exists():
            fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest("No regcoil test file found")
        result = load_CP_and_geometries(str(fpath), plot_flags=(0, 0, 0, 0), loadDOFsProperly=False)
        self.assertEqual(len(result), 5)
        cpst, s_coil_fp, s_coil_full, s_plasma_fp, s_plasma_full = result
        self.assertIsNotNone(cpst)
        self.assertIsNotNone(s_coil_fp)

    def test_load_CP_and_geometries_loadDOFsProperly_true(self):
        """load_CP_and_geometries with loadDOFsProperly=True copies surface DOFs."""
        fpath = TEST_DIR / "regcoil_out.w7x_infty.nc"
        if not fpath.exists():
            fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest("No regcoil test file found")
        result = load_CP_and_geometries(str(fpath), plot_flags=(0, 0, 0, 0), loadDOFsProperly=True)
        self.assertEqual(len(result), 5)
        cpst, s_coil_fp, s_coil_full, s_plasma_fp, s_plasma_full = result
        self.assertIsNotNone(s_coil_fp)

    def test_load_CP_and_geometries_with_plot_flags(self):
        """load_CP_and_geometries with plot_flags calls plot (mocked)."""
        fpath = TEST_DIR / "regcoil_out.w7x_infty.nc"
        if not fpath.exists():
            fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest("No regcoil test file found")
        from unittest.mock import patch
        with patch('simsopt.geo.plot') as mock_plot:
            result = load_CP_and_geometries(str(fpath), plot_flags=(1, 1, 1, 1))
        self.assertEqual(len(result), 5)
        mock_plot.assert_called()


class TestLoadRegcoilData(unittest.TestCase):
    """Tests for load_regcoil_data with real file."""

    def test_load_regcoil_file(self):
        """Load legacy REGCOIL file returns expected structure."""
        fpath = TEST_DIR / "regcoil_out.hsx.nc"
        if not fpath.exists():
            self.skipTest(f"Test file not found: {fpath}")
        data = load_regcoil_data(str(fpath))
        self.assertEqual(len(data), 19)
        ntheta, nzeta, theta, zeta, r, xm, xn, xmp, xnp, nfp, Ip, It = data[:12]
        self.assertGreater(ntheta, 0)
        self.assertGreater(nzeta, 0)
        self.assertEqual(len(theta), ntheta)
        self.assertEqual(len(zeta), nzeta)
        lambdas, chi2_B, chi2_K, K2 = data[15:19]
        self.assertGreater(len(lambdas), 0)


class TestSetAxesEqual(unittest.TestCase):
    """Tests for set_axes_equal."""

    def test_set_axes_equal_3d(self):
        """set_axes_equal adjusts 3D axes to equal scale."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot([0, 1], [0, 1], [0, 1])
        set_axes_equal(ax)
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xr = abs(xlim[1] - xlim[0])
        yr = abs(ylim[1] - ylim[0])
        zr = abs(zlim[1] - zlim[0])
        self.assertAlmostEqual(xr, yr)
        self.assertAlmostEqual(yr, zr)
        plt.close()


class TestContourPaths(unittest.TestCase):
    """Tests for _contour_paths (matplotlib version-agnostic)."""

    def test_contour_paths_allsegs(self):
        """_contour_paths returns paths from contour object."""
        import matplotlib.pyplot as plt
        x = np.linspace(0, 2 * np.pi, 20)
        y = np.linspace(0, 2 * np.pi / 4, 15)
        z = np.outer(np.sin(y), np.cos(x))
        cs = plt.contour(x, y, z, levels=[0])
        paths = _contour_paths(cs, 0)
        self.assertIsInstance(paths, list)
        if len(paths) > 0:
            self.assertIsInstance(paths[0], np.ndarray)
        plt.close()

    def test_contour_paths_fallback_collections(self):
        """_contour_paths uses collections.get_paths when allsegs not available."""
        from unittest.mock import MagicMock
        mock_path = MagicMock()
        mock_path.vertices = np.array([[0, 0], [1, 1]])
        mock_collection = MagicMock()
        mock_collection.get_paths.return_value = [mock_path]
        mock_cdata = type('CData', (), {'collections': [mock_collection]})()
        paths = _contour_paths(mock_cdata, 0)
        self.assertEqual(len(paths), 1)
        np.testing.assert_array_equal(paths[0], np.array([[0, 0], [1, 1]]))


class TestWriteContourToFile(unittest.TestCase):
    """Tests for writeContourToFile."""

    def test_write_contour_to_file(self):
        """writeContourToFile writes contour in legacy format."""
        import tempfile
        nharmonics = 1
        coeff_array = np.array([[0, 0, 1.0, 0, 0, 0]])
        args = (nharmonics, coeff_array, 'wp')
        contour = np.array([[0, 0], [np.pi / 2, 0], [np.pi, 0], [0, 0]])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as tmp:
            tmpname = tmp.name
        try:
            with open(tmpname, 'w') as f:
                writeContourToFile(f, contour, 1.0, args)
            with open(tmpname) as rf:
                lines = rf.readlines()
            self.assertGreater(len(lines), 0)
        finally:
            import os
            if os.path.exists(tmpname):
                os.unlink(tmpname)


class TestLoadSurfaceDofsProperly(unittest.TestCase):
    """Tests for _load_surface_dofs_properly."""

    def test_load_surface_dofs_properly(self):
        """_load_surface_dofs_properly copies surface DOFs and returns s_new."""
        from simsopt.geo import SurfaceRZFourier
        s = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        dofs = s.get_dofs()
        s.set_dofs(np.random.randn(len(dofs)) * 0.01)
        s_new = SurfaceRZFourier(nfp=4, mpol=2, ntor=2, stellsym=True)
        s_new = s_new.from_nphi_ntheta(nfp=4, ntheta=8, nphi=16, mpol=2, ntor=2, stellsym=True, range='field period')
        result = _load_surface_dofs_properly(s, s_new)
        self.assertIsNotNone(result)
        self.assertIs(result, s_new)
        # Result should have non-zero DOFs where s has them (at least rc, zs)
        self.assertTrue(np.any(result.get_dofs() != 0))


class TestMakeOnclickOnpickOnKey(unittest.TestCase):
    """Tests for make_onclick, make_onpick, make_on_key."""

    def test_make_onclick_returns_callable(self):
        """make_onclick returns a callable."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        args = _make_args()
        contours = []
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / 4, 8, endpoint=False)
        _, _, cp, _, _, _ = genCPvals((0, 2 * np.pi), (0, 2 * np.pi / 4), (10, 8), args)
        handler = make_onclick(ax, args, contours, theta, zeta, cp)
        self.assertTrue(callable(handler))
        plt.close()

    def test_make_onpick_returns_callable(self):
        """make_onpick returns a callable."""
        contours = []
        handler = make_onpick(contours)
        self.assertTrue(callable(handler))

    def test_make_on_key_returns_callable(self):
        """make_on_key returns a callable."""
        contours = []
        handler = make_on_key(contours)
        self.assertTrue(callable(handler))

    def test_make_onclick_dblclick_appends_contour(self):
        """make_onclick on dblclick finds contour and appends to list."""
        import matplotlib.pyplot as plt
        from unittest.mock import MagicMock
        fig, ax = plt.subplots()
        args = _make_args()
        contours = []
        theta = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / 4, 10, endpoint=False)
        _, _, cp, _, _, _ = genCPvals((0, 2 * np.pi), (0, 2 * np.pi / 4), (15, 10), args)
        handler = make_onclick(ax, args, contours, theta, zeta, cp)
        event = MagicMock()
        event.dblclick = True
        event.xdata, event.ydata = 1.0, 0.3
        handler(event)
        self.assertGreater(len(contours), 0)
        self.assertEqual(contours[0].shape[1], 2)
        plt.close()

    def test_make_onpick_sets_picked_object(self):
        """make_onpick stores picked artist on axes."""
        import matplotlib.pyplot as plt
        from unittest.mock import MagicMock
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        handler = make_onpick([])
        event = MagicMock()
        event.artist = ax.lines[0]
        handler(event)
        self.assertIs(plt.gca().picked_object, event.artist)
        plt.close()

    def test_make_on_key_delete_removes_picked(self):
        """make_on_key on 'delete' removes picked contour from list."""
        import matplotlib.pyplot as plt
        from unittest.mock import MagicMock
        fig, ax = plt.subplots()
        line_data = np.array([[0.5, 0.3], [1.0, 0.4]])
        line, = ax.plot(line_data[:, 0], line_data[:, 1])
        contours = [line_data.copy()]
        handler = make_on_key(contours)
        ax.picked_object = line
        event = MagicMock()
        event.key = 'delete'
        handler(event)
        self.assertEqual(len(contours), 0)
        self.assertIsNone(ax.picked_object)
        plt.close()


if __name__ == "__main__":
    unittest.main()
