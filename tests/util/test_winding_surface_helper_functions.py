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
    load_CP_and_geometries,
    current_potential_at_point,
    grad_current_potential_at_point,
    genCPvals,
    genKvals,
    is_periodic_lines,
    minDist,
    sortLevels,
    chooseContours_matching_coilType,
    points_in_polygon,
    map_data_to_more_periods,
    map_data_to_more_periods_3x1,
    ID_mod_hel,
    ID_and_cut_contour_types,
    removearray,
    gen_parametrization,
    splice_curve_fourier_coeffs,
    get1DFourierSinCosComps,
    getCurveOrder,
    compute_baseline_WP_currents,
    check_and_compute_nested_WP_currents,
    real_space,
    REGCOIL_line_XYZ_RZ,
    SIMSOPT_line_XYZ_RZ,
    writeToCurve,
    writeToCurve_helical,
    generate_sunflower,
    box_field_period,
    load_regcoil_data,
)

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
    """Tests for grad_current_potential_at_point."""

    def test_constant_potential_zero_grad(self):
        """Constant potential has zero gradient."""
        args = (0, 0, np.array([0]), np.array([0]), np.array([0.0]), np.array([0.0]), np.array([4]))
        val = grad_current_potential_at_point(np.array([0.5, 0.3]), args)
        self.assertAlmostEqual(val, 0.0)

    def test_returns_positive(self):
        """|∇φ| is non-negative."""
        args = _make_args()
        val = grad_current_potential_at_point(np.array([0.5, 0.3]), args)
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


class TestPointsInPolygon(unittest.TestCase):
    """Tests for points_in_polygon."""

    def test_triangle_contains_inside(self):
        """Points inside triangle are identified."""
        # Triangle (0,0), (1,0), (0.5,1)
        polygon = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])
        t = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        i1, i2, BOOL = points_in_polygon((t, z), None, polygon)
        # Center (0.5, 0.33) should be inside
        self.assertGreater(np.sum(BOOL), 0)

    def test_returns_correct_lengths(self):
        """Returned indices match BOOL mask."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        t = np.linspace(0.1, 0.9, 3)
        z = np.linspace(0.1, 0.9, 3)
        i1, i2, BOOL = points_in_polygon((t, z), None, polygon)
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
    """Tests for ID_mod_hel."""

    def test_modular_contour(self):
        """Contour with same cos(θ) and ζ at ends is modular."""
        # Start (0, 0), end (2π, 0) -> cos(0)=cos(2π)=1, zeta same
        contour = np.array([[0, 0], [np.pi, 0.5], [2 * np.pi, 0]])
        names, ints = ID_mod_hel([contour], tol=0.1)
        self.assertEqual(names[0], 'mod')
        self.assertEqual(ints[0], 1)

    def test_helical_contour(self):
        """Contour with θ spanning 2π is helical."""
        contour = np.array([[0, 0], [np.pi, 0.25], [2 * np.pi, 0.5]])
        names, ints = ID_mod_hel([contour], tol=0.05)
        self.assertEqual(names[0], 'hel')
        self.assertEqual(ints[0], 2)


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


class TestRemoveArray(unittest.TestCase):
    """Tests for removearray."""

    def test_remove_existing(self):
        """Removing existing array modifies list in place."""
        a = np.array([1, 2])
        b = np.array([3, 4])
        L = [a, b]
        removearray(L, a)
        self.assertEqual(len(L), 1)
        np.testing.assert_array_equal(L[0], b)

    def test_remove_raises_if_not_found(self):
        """Removing non-existent array raises ValueError."""
        L = [np.array([1, 2])]
        with self.assertRaises(ValueError):
            removearray(L, np.array([9, 9]))


class TestGenParametrization(unittest.TestCase):
    """Tests for gen_parametrization."""

    def test_cumulative_length(self):
        """Output is cumulative sum of |dx|."""
        x = np.array([0, 1, 2, 5])
        s = gen_parametrization(x)
        self.assertAlmostEqual(s[0], 0)
        self.assertAlmostEqual(s[1], 1)
        self.assertAlmostEqual(s[2], 2)
        self.assertAlmostEqual(s[3], 5)


class TestSpliceCurveFourierCoeffs(unittest.TestCase):
    """Tests for splice_curve_fourier_coeffs."""

    def test_output_length(self):
        """Output length is len(Ws)+len(Wc)-1."""
        Ws = np.array([0, 0.1, 0.2])
        Wc = np.array([1.0, 0.5, 0.3])
        coeffs = splice_curve_fourier_coeffs(Ws, Wc)
        self.assertEqual(len(coeffs), 3 + 3 - 1)

    def test_interleaving(self):
        """Cos and sin are interleaved correctly."""
        Ws = np.array([0, 1, 2])
        Wc = np.array([10, 20, 30])
        coeffs = splice_curve_fourier_coeffs(Ws, Wc)
        self.assertEqual(coeffs[0], 10)
        self.assertEqual(coeffs[1], 1)
        self.assertEqual(coeffs[2], 20)


class TestGet1DFourierSinCosComps(unittest.TestCase):
    """Tests for get1DFourierSinCosComps."""

    def test_reproduces_signal(self):
        """Recreation matches original for simple signal."""
        n = 32
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        signal = np.cos(t) + 0.5 * np.sin(2 * t)
        Amat, Bmat, eM, recreation = get1DFourierSinCosComps(
            numT=n, signal=signal, plot=(0, 0), forward=True, trunc=5
        )
        np.testing.assert_allclose(recreation, signal, atol=1e-10)

    def test_trunc_affects_modes(self):
        """Truncation limits number of modes."""
        n = 64
        signal = np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False))
        A1, B1, eM1, _ = get1DFourierSinCosComps(n, signal, (0, 0), True, trunc=3)
        A2, B2, eM2, _ = get1DFourierSinCosComps(n, signal, (0, 0), True, trunc=10)
        self.assertEqual(len(A1), 3)
        self.assertEqual(len(A2), 10)


class TestComputeBaselineWPCurrents(unittest.TestCase):
    """Tests for compute_baseline_WP_currents."""

    def test_single_contour(self):
        """Single closed contour returns one current."""
        args = _make_args()
        contour = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]])
        wp_currents, max_vals, func_vals = compute_baseline_WP_currents([contour], args, plot=False)
        self.assertEqual(len(wp_currents), 1)
        self.assertGreater(wp_currents[0], 0)


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


class TestRealSpace(unittest.TestCase):
    """Tests for real_space (REGCOIL format)."""

    def test_circular_cross_section(self):
        """Simple m=0, n=0 mode gives circular R."""
        # R = R0 + r*cos(θ), Z = 0 for m=n=0
        # REGCOIL format: nnum=4*n, mnum=m; columns 2,4,5,3 for crc,crs,czc,czs
        nharmonics = 1
        coeff_array = np.array([[0, 0, 1.0, 0, 0, 0]])  # R=1, Z=0
        polAng = np.array([0, np.pi / 2, np.pi])
        torAng = np.array([0, 0, 0])
        X, Y, R, Z = real_space(nharmonics, coeff_array, polAng, torAng)
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


class TestGetCurveOrder(unittest.TestCase):
    """Tests for getCurveOrder."""

    def test_matches_order(self):
        """getCurveOrder finds order that matches DOF count."""
        xdata = np.linspace(0, 1, 64, endpoint=False)
        # CurveXYZFourier order N has 3*(1 + 2*N) DOFs for stellsym
        order = getCurveOrder(xdata, 15, verb=False)
        self.assertGreaterEqual(order, 0)


class TestGenerateSunflower(unittest.TestCase):
    """Tests for generate_sunflower."""

    def test_shape(self):
        """Output is 2 x N."""
        pts = generate_sunflower(10, 1.0, False)
        self.assertEqual(pts.shape, (2, 10))

    def test_radius(self):
        """Points lie within radius."""
        pts = generate_sunflower(100, 2.0, False)
        r = np.sqrt(pts[0, :]**2 + pts[1, :]**2)
        self.assertLessEqual(np.max(r), 2.0 + 1e-6)


class TestBoxFieldPeriod(unittest.TestCase):
    """Tests for box_field_period."""

    def test_returns_box_when_ret_true(self):
        """When ret=True, returns corner coordinates."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        box = box_field_period(ax, [1, 1], ret=True)
        plt.close()
        self.assertEqual(box.shape, (5, 2))


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


if __name__ == "__main__":
    unittest.main()
