import unittest
import numpy as np
from pathlib import Path
from monty.tempfile import ScratchDir

from simsopt.util.dipole_array_helper_functions import (
    quaternion_from_axis_angle, quaternion_multiply, rotate_vector, compute_quaternion,
    rho, a_m, b_m, compute_fourier_coeffs, rho_fourier,
    generate_curves, remove_inboard_dipoles, remove_interlinking_dipoles_and_TFs, align_dipoles_with_plasma,
    initialize_coils, dipole_array_optimization_function, save_coil_sets,
    generate_even_arc_angles, generate_windowpane_array, generate_tf_array
)
from simsopt.geo import SurfaceRZFourier
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import CurvePlanarFourier

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'


class TestDipoleArrayHelperFunctions(unittest.TestCase):

    def test_quaternion_from_axis_angle(self):
        axis = np.array([0, 0, 1])
        theta = np.pi / 2
        q = quaternion_from_axis_angle(axis, theta)
        # Should be a 90-degree rotation about z
        np.testing.assert_allclose(q, [np.cos(theta/2), 0, 0, np.sin(theta/2)])

    def test_quaternion_multiply(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        q = quaternion_multiply(q1, q2)
        np.testing.assert_allclose(q, q2)  # Identity times anything is itself

    def test_rotate_vector(self):
        v = np.array([1, 0, 0])
        axis = np.array([0, 0, 1])
        theta = np.pi / 2
        q = quaternion_from_axis_angle(axis, theta)
        v_rot = rotate_vector(v, q)
        np.testing.assert_allclose(v_rot, [0, 1, 0], atol=1e-8)

    def test_compute_quaternion(self):
        normal = np.array([0, 0, 1])
        tangent = np.array([1, 0, 0])
        q = compute_quaternion(normal, tangent)
        # Should be identity quaternion
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-8)

    def test_rho_and_fourier(self):
        a, b, n = 2.0, 1.0, 2.0
        theta = np.linspace(0, 2*np.pi, 100)
        r = rho(theta, a, b, n)
        self.assertTrue(np.all(r > 0))
        # Fourier coefficients
        coeffs = compute_fourier_coeffs(3, a, b, n)
        r_approx = rho_fourier(theta, coeffs, 3)
        # Should be close to the original for low order
        np.testing.assert_allclose(r, r_approx, rtol=0.2, atol=0.2)

    def test_a_m_b_m(self):
        a, b, n = 2.0, 1.0, 2.0
        a0 = a_m(0, a, b, n)
        b1 = b_m(1, a, b, n)
        self.assertIsInstance(a0, float)
        self.assertIsInstance(b1, float)

    def test_generate_curves_and_remove_inboard(self):
        # Use a real surface and vessel for a minimal working test
        nphi, ntheta = 16, 16
        with ScratchDir("."):
            s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            VV = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            VV.extend_via_projected_normal(0.5)
            base_wp_curves, base_tf_curves = generate_curves(s, VV, outdir="", inboard_radius=0.2, wp_fil_spacing=0.1, half_per_spacing=0.1)
            self.assertTrue(len(base_wp_curves) > 0)
            self.assertTrue(len(base_tf_curves) > 0)

            # Compute minimum distance from origin for all dipoles before removal
            dists_before = np.array([
                np.min(np.linalg.norm(curve.gamma(), axis=1))
                for curve in base_wp_curves
            ])
            print(dists_before)

            # Remove inboard dipoles with a high threshold to force removal
            eps = 0.5  # Large enough to remove dipoles close to the plasma
            filtered = remove_inboard_dipoles(s, base_wp_curves, eps=eps)
            dists_after = np.array([
                np.min(np.linalg.norm(curve.gamma(), axis=1))
                for curve in filtered
            ])
            print(dists_after)

            # All remaining dipoles should be farther from the origin than the threshold
            threshold = (1.0 + eps) * s.get_rc(0, 0)
            self.assertTrue(np.all(dists_after >= threshold))

            # If any dipoles were removed, the number should decrease
            self.assertLessEqual(len(filtered), len(base_wp_curves))
            # If at least one was removed, check that the minimum distance increased
            if len(filtered) < len(base_wp_curves):
                self.assertGreater(np.min(dists_after), np.min(dists_before))

            # Remove interlinking dipoles and TFs (should not error, may remove none)
            filtered2 = remove_interlinking_dipoles_and_TFs(base_wp_curves, base_tf_curves, eps=0.05)
            self.assertTrue(isinstance(filtered2, np.ndarray))
            # Align dipoles with plasma (should return two arrays of angles)
            alphas, deltas = align_dipoles_with_plasma(s, base_wp_curves)
            self.assertEqual(alphas.shape, (len(base_wp_curves),))
            self.assertEqual(deltas.shape, (len(base_wp_curves),))

    def test_initialize_coils(self):
        nphi, ntheta = 8, 8
        config_file_map = {
            'LandremanPaulQA': 'input.LandremanPaul2021_QA_reactorScale_lowres',
            'LandremanPaulQH': 'input.LandremanPaul2021_QH_reactorScale_lowres',
            'SchuettHennebergQAnfp2': 'input.schuetthenneberg_nfp2'
        }
        with ScratchDir("."):
            for config, surf_file in config_file_map.items():
                surf_path = (Path(__file__).parent / ".." / "test_files" / surf_file).resolve()
                s = SurfaceRZFourier.from_vmec_input(surf_path, range="half period", nphi=nphi, ntheta=ntheta)
                base_curves, curves, coils, base_currents = initialize_coils(s, TEST_DIR, config)
                self.assertTrue(len(base_curves) > 0)
                self.assertTrue(len(curves) > 0)
                self.assertTrue(len(coils) > 0)
                self.assertTrue(len(base_currents) > 0)
                # Check that all base_curves are unique objects
                self.assertEqual(len(set(map(id, base_curves))), len(base_curves))
                # Check magnetic field at major radius
                bs = BiotSavart(coils)
                R_major = s.get_rc(0, 0)
                # Evaluate at (R_major, 0, 0)
                B = bs.set_points(np.array([[R_major, 0, 0]])).B().flatten()
                B_magnitude = np.linalg.norm(B)
                print(f"B at major radius for {config} is {B_magnitude}, should be 5-6 T")
                self.assertTrue(np.isclose(B_magnitude, 5.7, atol=1), f"B at major radius for {config} is {B_magnitude}, expected ~5.7 T")
            # Test for error on unknown configuration
            with self.assertRaises(ValueError):
                initialize_coils(s, TEST_DIR, 'not_a_real_config')

    def test_generate_even_arc_angles(self):
        a, b, ntheta = 2.0, 1.0, 10
        thetas = generate_even_arc_angles(a, b, ntheta)
        self.assertEqual(len(thetas), ntheta)
        # Should be sorted and between 0 and 2pi
        self.assertTrue(np.all(np.diff(thetas) >= 0))
        self.assertTrue(np.all(thetas >= 0) and np.all(thetas <= 2 * np.pi))

    def test_generate_windowpane_array_and_tf_array(self):
        from simsopt._core import Optimizable
        nphi, ntheta = 16, 16
        with ScratchDir("."):
            s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            VV = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            VV.extend_via_normal(0.5)
            # Windowpane array
            wp_curves = generate_windowpane_array(
                VV, inboard_radius=0.1, wp_fil_spacing=0.05, half_per_spacing=0.05, wp_n=2,
                numquadpoints=32, order=2, verbose=False
            )
            self.assertTrue(len(wp_curves) > 0)
            self.assertIsInstance(wp_curves[0], CurvePlanarFourier)

            # TF array (original test)
            tf_curves = generate_tf_array(
                s, ntf=2, TF_R0=VV.get_rc(0, 0), TF_a=VV.get_rc(1, 0), TF_b=VV.get_rc(1, 0),
                fixed_geo_tfs=True, planar_tfs=True, order=2, numquadpoints=32
            )
            # Check that all curves have the expected number of quadrature points
            for curve in wp_curves + tf_curves:
                self.assertEqual(curve.gamma().shape[0], 32)
            self.assertTrue(len(tf_curves) > 0)
            # Check that all TF curves are unique objects
            self.assertEqual(len(set(map(id, tf_curves))), len(tf_curves))

            # --- New test: Planarity and free dofs for planar TFs ---
            tf_curves_planar = generate_tf_array(
                s, ntf=2, TF_R0=VV.get_rc(0, 0), TF_a=VV.get_rc(1, 0), TF_b=VV.get_rc(1, 0),
                fixed_geo_tfs=True, planar_tfs=True, order=2, numquadpoints=32
            )
            for curve in tf_curves_planar:
                # Check planarity: all points should have the same normal vector (z direction)
                gamma = curve.gamma()
                v1 = gamma[1] - gamma[0]
                v2 = gamma[2] - gamma[0]
                normal = np.cross(v1, v2)
                normal /= np.linalg.norm(normal)
                diffs = gamma - gamma[0]
                dots = np.dot(diffs, normal)
                self.assertTrue(np.allclose(dots, 0, atol=1e-12), "TF coil is not planar")
                # Check for free dofs R0 and r_rotation
                dof_names = getattr(curve, "local_dof_names", None)
                self.assertIsNotNone(dof_names, "Curve does not have free_names attribute")
                self.assertIn("xc(0)", dof_names, "Curve does not have free dof xc(0)")
                self.assertIn("xc(1)", dof_names, "Curve does not have free dof xc(1)")

            # --- Additional test: Unfix fixed_geo_tfs and check for more dofs ---
            # This flag makes it so the TF coils are initialized using
            # create_equally_spaced_cylindrical_curve
            tf_curves_unfixed = generate_tf_array(
                s, ntf=2, TF_R0=VV.get_rc(0, 0), TF_a=VV.get_rc(1, 0), TF_b=VV.get_rc(1, 0),
                fixed_geo_tfs=False, planar_tfs=True, order=2, numquadpoints=32
            )
            dof_names_unfixed = getattr(tf_curves_unfixed[0], "local_dof_names", [])
            self.assertIn("R0", dof_names_unfixed, "Curve does not have free dof R0")
            self.assertIn("phi", dof_names_unfixed, "Curve does not have free dof phi")
            self.assertIn("Z0", dof_names_unfixed, "Curve does not have free dof Z0")
            self.assertIn("r_rotation", dof_names_unfixed, "Curve does not have free dof r_rotation")
            self.assertIn("phi_rotation", dof_names_unfixed, "Curve does not have free dof phi_rotation")
            self.assertIn("z_rotation", dof_names_unfixed, "Curve does not have free dof z_rotation")
            # --- End new test ---

            base_currents = [Current(1.0) for _ in wp_curves]
            base_currents_tf = [Current(1.0) for _ in tf_curves]
            coils = coils_via_symmetries(wp_curves, base_currents, s.nfp, True)
            coils_tf = coils_via_symmetries(tf_curves, base_currents_tf, s.nfp, True)
            bs = BiotSavart(coils)
            bs_tf = BiotSavart(coils_tf)
            btot = bs + bs_tf
            btot.set_points(s.gamma().reshape(-1, 3))

            # Should not raise
            save_coil_sets(btot, ".", "_test")
            # Check that files were created (now saves all coils together)
            files = list(Path(".").glob("*"))
            self.assertTrue(any("coils_test" in f.name for f in files))

            class DummyObj(Optimizable):
                def J(self): return 1.0
                def dJ(self): return np.ones(len(wp_curves[0].x))
                x = np.ones(len(wp_curves[0].x))
                def shortest_distance(self): return 42.0

            obj_dict = {
                "btot": btot,
                "Jcs": [DummyObj()],
                "Jmscs": [DummyObj()],
                "Jls": [DummyObj()],
                "Jls_TF": [DummyObj()],
                "Jccdist": DummyObj(),
                "Jccdist2": DummyObj(),
                "Jcsdist": DummyObj(),
                "linkNum": DummyObj(),
                "Jforce": DummyObj(),
                "Jforce2": DummyObj(),
                "Jtorque": DummyObj(),
                "Jtorque2": DummyObj(),
            }
            weight_dict = {
                "length_weight": 1.0,
                "curvature_weight": 1.0,
                "msc_weight": 1.0,
                "msc_threshold": 1.0,
                "cc_weight": 1.0,
                "cs_weight": 1.0,
                "link_weight": 1.0,
                "force_weight": 1.0,
                "net_force_weight": 1.0,
                "torque_weight": 1.0,
                "net_torque_weight": 1.0,
            }
            dofs = np.ones(len(wp_curves[0].x))
            J, grad = dipole_array_optimization_function(dofs, obj_dict, weight_dict)
            self.assertIsInstance(J, float)
            self.assertTrue(np.allclose(grad, np.ones_like(grad)))
            # Check that the gradient has the correct shape
            self.assertEqual(grad.shape, dofs.shape)
            # Check that the objective is equal to 1
            self.assertEqual(J, 1.0)


if __name__ == "__main__":
    unittest.main()
