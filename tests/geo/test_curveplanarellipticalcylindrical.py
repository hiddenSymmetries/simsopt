"""
Unit tests for simsopt.geo.curveplanarellipticalcylindrical.
"""
import unittest
import numpy as np

from simsopt.geo.curveplanarellipticalcylindrical import (
    CurvePlanarEllipticalCylindrical,
    create_equally_spaced_cylindrical_curves,
    r_ellipse,
    xyz_cyl,
    convert_to_cyl,
    cylindrical_shift,
    cyl_to_cart,
    gamma_pure,
)


class TestREllipse(unittest.TestCase):
    """Tests for r_ellipse."""

    def test_r_ellipse_circle(self):
        """For a=b, r_ellipse should give constant radius."""
        a = b = 1.0
        l = np.linspace(0, 1, 10, endpoint=False)
        r = r_ellipse(a, b, l)
        np.testing.assert_allclose(r, np.ones_like(l), atol=1e-10)

    def test_r_ellipse_ellipse(self):
        """r_ellipse returns positive values for ellipse."""
        a, b = 1.0, 0.5
        l = np.array([0.0, 0.25, 0.5])
        r = r_ellipse(a, b, l)
        self.assertTrue(np.all(r > 0))
        self.assertEqual(len(r), 3)


class TestXyzCyl(unittest.TestCase):
    """Tests for xyz_cyl."""

    def test_xyz_cyl_shape(self):
        """xyz_cyl returns (n, 3) array."""
        a, b = 1.0, 0.5
        l = np.linspace(0, 1, 8, endpoint=False)
        out = xyz_cyl(a, b, l)
        self.assertEqual(out.shape, (8, 3))

    def test_xyz_cyl_y_zero(self):
        """Curve lies in xz plane (y=0)."""
        a, b = 1.0, 0.5
        l = np.linspace(0, 1, 10, endpoint=False)
        out = xyz_cyl(a, b, l)
        np.testing.assert_allclose(out[:, 1], 0.0)


class TestConvertToCyl(unittest.TestCase):
    """Tests for convert_to_cyl and cyl_to_cart."""

    def test_cyl_roundtrip(self):
        """convert_to_cyl and cyl_to_cart roundtrip."""
        xyz = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        cyl = convert_to_cyl(xyz)
        back = cyl_to_cart(cyl)
        np.testing.assert_allclose(back, xyz, atol=1e-10)

    def test_cylindrical_shift(self):
        """cylindrical_shift adds dphi and dz."""
        a = np.array([[1.0, 0.0, 0.0]])
        shifted = cylindrical_shift(a, dphi=np.pi / 2, dz=1.0)
        np.testing.assert_allclose(shifted[:, 1], np.pi / 2)
        np.testing.assert_allclose(shifted[:, 2], 1.0)


class TestGammaPure(unittest.TestCase):
    """Tests for gamma_pure."""

    def test_gamma_pure_shape(self):
        """gamma_pure returns correct shape."""
        dofs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # R0, phi, Z0, r_rot, phi_rot, z_rot
        points = np.linspace(0, 1, 16, endpoint=False)
        a, b = 0.3, 0.2
        gamma = gamma_pure(dofs, points, a, b)
        self.assertEqual(gamma.shape, (16, 3))


class TestCurvePlanarEllipticalCylindrical(unittest.TestCase):
    """Tests for CurvePlanarEllipticalCylindrical."""

    def test_init_int_quadpoints(self):
        """Quadpoints can be int (converted to linspace)."""
        curve = CurvePlanarEllipticalCylindrical(16, a=0.3, b=0.2)
        self.assertEqual(curve.num_dofs(), 6)
        gamma = curve.gamma()
        self.assertEqual(gamma.shape[0], 16)
        self.assertEqual(gamma.shape[1], 3)

    def test_init_array_quadpoints(self):
        """Quadpoints can be array."""
        quadpoints = np.linspace(0, 1, 32, endpoint=False)
        curve = CurvePlanarEllipticalCylindrical(quadpoints, a=0.3, b=0.2)
        gamma = curve.gamma()
        self.assertEqual(gamma.shape[0], 32)

    def test_init_with_dofs(self):
        """Curve can be initialized and dofs set explicitly."""
        curve = CurvePlanarEllipticalCylindrical(16, a=0.3, b=0.2)
        dofs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        curve.set_dofs(dofs)
        np.testing.assert_allclose(curve.get_dofs(), dofs)

    def test_num_dofs(self):
        """num_dofs returns 6 (R0, phi, Z0, r_rot, phi_rot, z_rot)."""
        curve = CurvePlanarEllipticalCylindrical(16, a=0.3, b=0.2)
        self.assertEqual(curve.num_dofs(), 6)

    def test_get_set_dofs(self):
        """get_dofs and set_dofs roundtrip."""
        curve = CurvePlanarEllipticalCylindrical(16, a=0.3, b=0.2)
        dofs = np.array([1.0, 0.5, 0.1, 0.0, 0.0, 0.0])
        curve.set_dofs(dofs)
        np.testing.assert_allclose(curve.get_dofs(), dofs)

    def test_set_individual_dofs(self):
        """Individual dofs can be set by name."""
        curve = CurvePlanarEllipticalCylindrical(16, a=0.3, b=0.2)
        curve.set("R0", 1.5)
        curve.set("phi", np.pi / 4)
        curve.set("Z0", 0.2)
        curve.set("r_rotation", 0.0)
        curve.set("phi_rotation", 0.0)
        curve.set("z_rotation", 0.0)
        dofs = curve.get_dofs()
        self.assertAlmostEqual(dofs[0], 1.5)
        self.assertAlmostEqual(dofs[1], np.pi / 4)
        self.assertAlmostEqual(dofs[2], 0.2)

    def test_gamma_finite(self):
        """gamma returns finite values."""
        curve = CurvePlanarEllipticalCylindrical(16, a=0.3, b=0.2)
        curve.set("R0", 1.0)
        gamma = curve.gamma()
        self.assertTrue(np.all(np.isfinite(gamma)))


class TestCreateEquallySpacedCylindricalCurves(unittest.TestCase):
    """Tests for create_equally_spaced_cylindrical_curves."""

    def test_create_curves_count(self):
        """Correct number of curves created."""
        curves = create_equally_spaced_cylindrical_curves(
            ncurves=4, nfp=2, stellsym=True, R0=1.0, a=0.3, b=0.2
        )
        self.assertEqual(len(curves), 4)

    def test_create_curves_stellsym(self):
        """Curves have correct phi spacing with stellsym."""
        curves = create_equally_spaced_cylindrical_curves(
            ncurves=2, nfp=2, stellsym=True, R0=1.0, a=0.3, b=0.2
        )
        for i, c in enumerate(curves):
            phi = c.get("phi")
            expected = (i + 0.5) * (2 * np.pi) / (2 * 2 * 2)  # (1+stellsym)*nfp*ncurves
            self.assertAlmostEqual(phi, expected)

    def test_create_curves_numquadpoints(self):
        """numquadpoints is respected."""
        curves = create_equally_spaced_cylindrical_curves(
            ncurves=2, nfp=2, stellsym=True, R0=1.0, a=0.3, b=0.2, numquadpoints=64
        )
        gamma = curves[0].gamma()
        self.assertEqual(gamma.shape[0], 64)
