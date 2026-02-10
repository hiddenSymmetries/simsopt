import unittest
import os
import numpy as np

from simsopt.configs.zoo import get_data
from simsopt.field.tracing import (
    ScipyFieldlineIntegrator,
    compute_xyz_fourier_coefficients_from_points
)
from simsopt.geo import SurfaceRZFourier, CurveXYZFourierSymmetries


# Get path to test files
TEST_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_FILES_DIR = os.path.join(TEST_DIR, "test_files")


class TestComputeXYZFourierCoefficientsFromPoints(unittest.TestCase):
    """
    Test the compute_xyz_fourier_coefficients_from_points function.
    """

    def setUp(self):
        """Set up W7-X configuration for testing."""
        base_curves, base_currents, ma, nfp, bs = get_data('w7x', coil_order=12, points_per_period=4)
        self.base_curves = base_curves
        self.base_currents = base_currents
        self.ma = ma
        self.nfp = nfp
        self.bs = bs

    def test_axis_coefficients_match_curve(self):
        """
        Test that compute_xyz_fourier_coefficients_from_points can reconstruct
        a CurveXYZFourierSymmetries curve from its sampled points.
        """
        nfp = self.nfp
        order = 7
        ntor = 1
        n_points = 200
        
        # Create a curve with known coefficients
        original_curve = CurveXYZFourierSymmetries(
            n_points, order, nfp, stellsym=True, ntor=ntor
        )
        
        # Set some coefficients that make a curve similar to W7-X axis
        test_dofs = np.zeros(3*order + 1)
        test_dofs[0] = 5.5  # xc(0) - major radius
        test_dofs[1] = 0.3  # xc(1) - variation
        test_dofs[order+1] = 0.15  # ys(1)
        test_dofs[2*order+1] = 0.3  # zs(1)
        
        original_curve.x = test_dofs
        xyz_points = original_curve.gamma()
        
        # The quadpoints are from 0 to 1
        varthetas = np.linspace(0, 1, n_points, endpoint=False)
        
        xc, ys, zs, rmse = compute_xyz_fourier_coefficients_from_points(
            xyz_points, nfp=nfp, order=order, ntor=ntor, varthetas=varthetas
        )
        
        # Check that the fitted coefficients match the original
        fitted_xc = xc
        fitted_ys = ys
        fitted_zs = zs
        
        np.testing.assert_allclose(fitted_xc, test_dofs[0:order+1], atol=1e-10)
        np.testing.assert_allclose(fitted_ys, test_dofs[order+1:2*order+1], atol=1e-10)
        np.testing.assert_allclose(fitted_zs, test_dofs[2*order+1:], atol=1e-10)
        
        # RMSE should be essentially zero
        self.assertLess(rmse, 1e-10, f"RMSE too large: {rmse}")

    def test_fitted_curve_gamma_matches_original(self):
        """
        Test that a curve created from fitted coefficients produces the same
        gamma() as the original curve.
        
        This test:
        1. Creates a CurveXYZFourierSymmetries with known coefficients
        2. Samples its gamma() points
        3. Uses compute_xyz_fourier_coefficients_from_points to fit those points
        4. Creates a new CurveXYZFourierSymmetries from the fitted coefficients
        5. Verifies that both curves' gamma() outputs match
        """
        nfp = self.nfp
        order = 6
        ntor = 1
        n_points = 300
        
        # Create a curve with known coefficients
        original_curve = CurveXYZFourierSymmetries(
            n_points, order, nfp, stellsym=True, ntor=ntor
        )
        
        # Set some coefficients that make a realistic curve
        test_dofs = np.zeros(3*order + 1)
        test_dofs[0] = 5.5   # xc(0) - major radius
        test_dofs[1] = 0.35  # xc(1)
        test_dofs[2] = 0.02  # xc(2)
        test_dofs[order+1] = 0.18   # ys(1)
        test_dofs[order+2] = 0.01   # ys(2)
        test_dofs[2*order+1] = 0.32  # zs(1)
        test_dofs[2*order+2] = 0.015  # zs(2)
        
        original_curve.x = test_dofs
        xyz_points = original_curve.gamma()
        
        # The quadpoints are from 0 to 1
        varthetas = np.linspace(0, 1, n_points, endpoint=False)
        
        # Fit the coefficients
        xc, ys, zs, rmse = compute_xyz_fourier_coefficients_from_points(
            xyz_points, nfp=nfp, order=order, ntor=ntor, varthetas=varthetas
        )
        
        # Create a new curve with the fitted coefficients
        fitted_curve = CurveXYZFourierSymmetries(
            n_points, order, nfp, stellsym=True, ntor=ntor
        )
        fitted_dofs = np.concatenate([xc, ys, zs])
        fitted_curve.x = fitted_dofs
        
        # Get gamma from both curves
        original_gamma = original_curve.gamma()
        fitted_gamma = fitted_curve.gamma()
        
        # The curves should have identical gamma (within numerical precision)
        np.testing.assert_allclose(
            fitted_gamma, original_gamma, atol=1e-10,
            err_msg="Fitted curve gamma does not match original curve gamma"
        )
        
        # Also check at a few specific parameter values with different quadpoints
        # Need to create new curves since quadpoints can't be set after construction
        test_quadpoints = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
        original_curve_test = CurveXYZFourierSymmetries(
            test_quadpoints, order, nfp, stellsym=True, ntor=ntor
        )
        original_curve_test.x = test_dofs
        
        fitted_curve_test = CurveXYZFourierSymmetries(
            test_quadpoints, order, nfp, stellsym=True, ntor=ntor
        )
        fitted_curve_test.x = fitted_dofs
        
        original_test_gamma = original_curve_test.gamma()
        fitted_test_gamma = fitted_curve_test.gamma()
        
        np.testing.assert_allclose(
            fitted_test_gamma, original_test_gamma, atol=1e-10,
            err_msg="Fitted curve gamma does not match at test quadpoints"
        )

    def test_returns_correct_shapes(self):
        """
        Test that compute_xyz_fourier_coefficients_from_points returns arrays
        of the correct shape.
        """
        # Use n_points coprime with nfp to avoid Nyquist aliasing
        # (when n_points is a multiple of nfp*order, sin columns become zero)
        n_points = 53  # prime number, coprime with nfp=5
        order = 5
        nfp = 5
        ntor = 1
        
        # Create some dummy xyz points
        varthetas = np.linspace(0, 1, n_points, endpoint=False)
        xyz_points = np.column_stack([
            5.5 * np.cos(2 * np.pi * varthetas),
            5.5 * np.sin(2 * np.pi * varthetas),
            0.3 * np.sin(2 * np.pi * varthetas * nfp)
        ])
        
        xc, ys, zs, rmse = compute_xyz_fourier_coefficients_from_points(
            xyz_points, nfp=nfp, order=order, ntor=ntor, varthetas=varthetas
        )
        
        # xc should have order+1 elements (includes constant term)
        self.assertEqual(len(xc), order + 1)
        # ys and zs should have order elements
        self.assertEqual(len(ys), order)
        self.assertEqual(len(zs), order)
        # rmse should be a scalar
        self.assertIsInstance(float(rmse), float)


class TestGetSurfaceRZFourierFromFieldlineSimple(unittest.TestCase):
    """
    Test the get_SurfaceRZFourier_from_fieldline_simple method.
    """

    @classmethod
    def setUpClass(cls):
        """Set up W7-X configuration for testing."""
        base_curves, base_currents, ma, nfp, bs = get_data('w7x', coil_order=12, points_per_period=4)
        cls.base_curves = base_curves
        cls.base_currents = base_currents
        cls.ma = ma
        cls.nfp = nfp
        cls.bs = bs
        
        # Create an integrator
        cls.intg = ScipyFieldlineIntegrator(
            bs, nfp=nfp, stellsym=True, 
            integrator_type='RK45', 
            integrator_args={'rtol': 1e-9, 'atol': 1e-11}
        )

    def test_surface_from_axis_fieldline(self):
        """
        Test that get_SurfaceRZFourier_from_fieldline_simple produces a surface
        when starting from near the magnetic axis.
        """
        # Start near the magnetic axis
        axis_point = self.ma.gamma()[0]
        start_RZ = np.array([np.sqrt(axis_point[0]**2 + axis_point[1]**2), axis_point[2]])
        
        # Add a small offset to be off-axis
        start_RZ[0] += 0.1
        
        # Get a surface from this field line
        surface = self.intg.get_SurfaceRZFourier_from_fieldline_simple(
            start_RZ, phi0=0, nphi=10, num_fieldlinetransits=5, ntor=3, mpol=3
        )
        
        # Check that we got a SurfaceRZFourier
        self.assertIsInstance(surface, SurfaceRZFourier)
        
        # Check surface properties
        self.assertEqual(surface.nfp, self.nfp)
        self.assertTrue(surface.stellsym)
        self.assertEqual(surface.mpol, 3)
        self.assertEqual(surface.ntor, 3)

    def test_surface_matches_w7x_reference(self):
        """
        Test that the surface from field line tracing resembles the W7-X
        standard configuration.

        # MORE TESTING NEEDED, routine also not perfect.
        
        """
        # Load the reference surface from file
        ref_surface_file = os.path.join(TEST_FILES_DIR, "input.W7-X_standard_configuration")
        if not os.path.exists(ref_surface_file):
            self.skipTest(f"Reference file not found: {ref_surface_file}")
        
        ref_surface = SurfaceRZFourier.from_vmec_input(ref_surface_file)
        
        # Get the major radius and minor radius from reference surface
        ref_gamma = ref_surface.gamma()
        ref_R = np.sqrt(ref_gamma[:, :, 0]**2 + ref_gamma[:, :, 1]**2)
        ref_Z = ref_gamma[:, :, 2]
        ref_major_radius = np.mean(ref_R)
        ref_minor_radius = (np.max(ref_R) - np.min(ref_R)) / 2
        
        # Start from a point on the reference surface
        # gamma()[0] gives the first point (phi=0, theta=0)
        start_point = ref_surface.gamma()[0, 0]  # shape (3,): [x, y, z]
        start_R = np.sqrt(start_point[0]**2 + start_point[1]**2)
        start_Z = start_point[2]
        start_RZ = np.array([start_R, start_Z])
        
        # Create surface from field line
        # Use more points and regularization for numerical stability
        surface = self.intg.get_SurfaceRZFourier_from_fieldline_simple(
            start_RZ, phi0=0, nphi=53, num_fieldlinetransits=53, ntor=4, mpol=4,
            regularization=1e-6
        )
        
        # Compare geometric properties rather than coefficients
        traced_gamma = surface.gamma()
        traced_R = np.sqrt(traced_gamma[:, :, 0]**2 + traced_gamma[:, :, 1]**2)
        traced_major_radius = np.mean(traced_R)
        
        # The major radius should be similar (within 10%)
        self.assertAlmostEqual(
            traced_major_radius, ref_major_radius, delta=ref_major_radius * 0.1,
            msg=f"Major radius mismatch: traced={traced_major_radius}, ref={ref_major_radius}"
        )


class TestCurveXYZFourierSymmetriesFromFieldline(unittest.TestCase):
    """
    Test the get_CurveXYZFourierSymmetries_from_fieldline method.
    """

    @classmethod
    def setUpClass(cls):
        """Set up ncsx configuration for testing."""
        base_curves, base_currents, ma, nfp, bs = get_data('ncsx', coil_order=12, points_per_period=4)
        cls.base_curves = base_curves
        cls.base_currents = base_currents
        cls.ma = ma
        cls.nfp = nfp
        cls.bs = bs
        
        # Create an integrator
        cls.intg = ScipyFieldlineIntegrator(
            bs, nfp=nfp, stellsym=True,
            integrator_type='RK45',
            integrator_args={'rtol': 1e-9, 'atol': 1e-11}
        )

    def test_curve_from_axis_matches_axis(self):
        """
        Test that get_CurveXYZFourierSymmetries_from_fieldline reproduces
        a curve close to the magnetic axis when started on the axis.
        """
        # Start on the magnetic axis
        axis_point = self.ma.gamma()[0]
        start_RZ = np.array([np.sqrt(axis_point[0]**2 + axis_point[1]**2), axis_point[2]])
        
        # Get a curve from this field line 
        curve = self.intg.get_CurveXYZFourierSymmetries_from_fieldline(
            start_RZ, order=10, ntor=1, nfp=self.nfp, num_points=200
        )
        
        # Check that we got a CurveXYZFourierSymmetries
        self.assertIsInstance(curve, CurveXYZFourierSymmetries)

        # though not required for CurveXYZFourierSymmetries, in htis
        # curve the toroidal angle equals the theta parameter of the
        # curve. 
        # We can compare the two by evaluating their gammas. 
        ma_gamma = self.ma.gamma()
        fitcurv_gamma = np.zeros_like(ma_gamma)
        curve.gamma_impl(fitcurv_gamma, self.ma.quadpoints) # sets the fitcurv_gamma
        point_distances = np.linalg.norm(fitcurv_gamma - ma_gamma, axis=1)

        # assert all points less than a centimeter:
        self.assertTrue(
            np.all(point_distances < 1e-2),
            msg=f"Not all points within 1 cm of axis. Max distance: {np.max(point_distances)}"
        )

    def test_curve_island_center(self):
        """
        Test that get_CurveXYZFourierSymmetries_from_fieldline can find the center of the island, and represent the
        field line through it. 
        """
        # Approximate location of an island center in NCSX
        start_RZ = np.array([1.52, 0.])  # R, Z coordinates
        fixp = self.intg.find_fixed_point(start_RZ, repetition_period=7, phi0=0)
        known_location = np.array([1.52288140e+00, 0])

        # Check that the fixed point is close to the known location
        np.testing.assert_allclose(
            fixp, known_location, atol=1e-4,
            err_msg="Found fixed point is not close to known island center location"
        )

        # Get a curve from this field line
        curve = self.intg.get_CurveXYZFourierSymmetries_from_fieldline(
            fixp, order=20, ntor=7, nfp=self.nfp, num_points=3000
        )

        # Check that we got a CurveXYZFourierSymmetries
        self.assertIsInstance(curve, CurveXYZFourierSymmetries)
        
        # Also integrate the field line for one full period to compare
        phis = np.linspace(0, 2*np.pi*7, 256)
        status, fieldline_points = self.intg.integrate_cyl_planes(fixp, phis, return_cartesian=True)

        effective_quadpoints = phis/(2*np.pi*7)

        curve_at_effective_quadpoints = np.zeros((len(phis), 3))
        curve.gamma_impl(curve_at_effective_quadpoints, effective_quadpoints)

        curve_distances = np.linalg.norm(curve_at_effective_quadpoints - fieldline_points, axis=1)

        # Compare the field line points with the curve points
        self.assertTrue(np.all(curve_distances < 1e-2), 
            "Field line points do not match curve points at effective quad points"
            )


        
if __name__ == '__main__':
    unittest.main()
