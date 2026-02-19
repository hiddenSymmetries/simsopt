import unittest
import numpy as np

from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.field.tracing import Integrator, SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator
from simsopt.configs.zoo import get_data, configurations
from simsopt._core.util import ObjectiveFailure


class TestIntegratorBase(unittest.TestCase):
    def setUp(self):
        # Simple, fast analytic field suitable for determinism in tests
        self.R0 = 1.3
        self.B0 = 0.8
        self.field = ToroidalField(self.R0, self.B0)

    def test_coordinate_roundtrip(self):
        # Use a handful of random points away from pathological locations
        rng = np.random.default_rng(0)
        pts_rphiz = np.column_stack([
            rng.uniform(self.R0 * 0.8, self.R0 * 1.2, size=10),
            rng.uniform(-np.pi, np.pi, size=10),
            rng.uniform(-0.5, 0.5, size=10),
        ])
        xyz = Integrator._rphiz_to_xyz(pts_rphiz)
        rphiz_back = Integrator._xyz_to_rphiz(xyz)
        # Phi has 2*pi periodicity, compare via unit circle embedding
        self.assertTrue(np.allclose(pts_rphiz[:, 0], rphiz_back[:, 0], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(pts_rphiz[:, 2], rphiz_back[:, 2], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(np.cos(pts_rphiz[:, 1]), np.cos(rphiz_back[:, 1]), atol=1e-12))
        self.assertTrue(np.allclose(np.sin(pts_rphiz[:, 1]), np.sin(rphiz_back[:, 1]), atol=1e-12))

    def test_incorrect_staticmethods(self):
        # Test incorrect static method calls
        with self.assertRaises(ValueError):
            Integrator._rphiz_to_xyz(1)  # Invalid input
        with self.assertRaises(ValueError):
            Integrator._rphiz_to_xyz(np.random.random(4))  # Invalid input

        with self.assertRaises(ValueError):
            Integrator._xyz_to_rphiz(1)  # Invalid input
        with self.assertRaises(ValueError):
            Integrator._rphiz_to_xyz(np.random.random(2))  # Invalid input

class TestIntegratorsCoordinateHandling(unittest.TestCase):
    def setUp(self):
        self.R0 = 1.2
        self.B0 = 1.0
        self.field = ToroidalField(self.R0, self.B0)
        self.simsopt_intg = SimsoptFieldlineIntegrator(self.field, nfp=1)
        self.scipy_intg = ScipyFieldlineIntegrator(self.field, nfp=1)

    def test_invalid_coordinate_inputs(self):
        start_xyz = np.array([self.R0, 0.0, 0.0])
        start_RZ = np.array([self.R0, 0.0])
        for intg in [self.simsopt_intg, self.scipy_intg]:
            with self.assertRaises(ValueError):
                intg.integrate_toroidally(start_xyz, delta_phi=np.pi/2, input_coordinates='invalid', output_coordinates='cartesian')
            with self.assertRaises(ValueError):
                intg.integrate_toroidally(start_xyz, delta_phi=np.pi/2, input_coordinates='cartesian', output_coordinates='invalid')
            with self.assertRaises(ValueError):
                intg.integrate_toroidally(start_RZ, phi0=None, delta_phi=np.pi/2, input_coordinates='cylindrical', output_coordinates='cartesian')
        
        for intg in [self.simsopt_intg, self.scipy_intg]:
            with self.assertRaises(ValueError):
                intg.integrate_fieldlinepoints(start_xyz, delta_phi=np.pi/2, input_coordinates='invalid', output_coordinates='cartesian')
            with self.assertRaises(ValueError):
                intg.integrate_fieldlinepoints(start_xyz, delta_phi=np.pi/2, input_coordinates='cartesian', output_coordinates='invalid')
            with self.assertRaises(ValueError):
                intg.integrate_fieldlinepoints(start_RZ, phi0=None, delta_phi=np.pi/2, input_coordinates='cylindrical', output_coordinates='cartesian')
        with self.assertRaises(ValueError): 
            self.scipy_intg.integrate_3d_fieldlinepoints(start_xyz, l_total=1.0, n_points=10, input_coordinates='invalid', output_coordinates='cartesian')
        with self.assertRaises(ValueError):
            self.scipy_intg.integrate_3d_fieldlinepoints(start_xyz, l_total=1.0, n_points=10, input_coordinates='cartesian', output_coordinates='invalid')
        with self.assertRaises(ValueError):
            self.scipy_intg.integrate_3d_fieldlinepoints(start_RZ, l_total=1.0, phi0=None, n_points=10, input_coordinates='cylindrical', output_coordinates='cartesian')

class TestSimsoptFieldlineIntegrator(unittest.TestCase):
    def setUp(self):
        self.R0 = 1.2
        self.B0 = 1.0
        self.field = ToroidalField(self.R0, self.B0)
        # Keep tmax modest so tests are quick
        self.intg = SimsoptFieldlineIntegrator(self.field, nfp=1, stellsym=True, tmax=100.0, tol=1e-9)

    def test_poincare_hits_basic(self):
        # Two starting radii on midplane
        RZ = np.array([[self.R0 + 0.05, 0.0], [self.R0 + 0.10, 0.0]])
        phis = np.linspace(0.1, 2*np.pi, 8, endpoint=False)
        res_tys, res_phi_hits = self.intg.compute_poincare_hits(RZ, n_transits=3, phis=phis, phi0=0.0)
        # For a purely toroidal field: Z stays 0, R stays constant; and we visit planes cyclically
        for i, hits in enumerate(res_phi_hits):
            R_const = RZ[i, 0]
            Z_const = RZ[i, 1]
            # Consider only actual plane hits; the last row may be a stopping-criterion (idx < 0)
            mask = hits[:, 1] >= 0
            r = np.sqrt(hits[mask, 2]**2 + hits[mask, 3]**2)
            z = hits[mask, 4]
            # Allow small numerical deviation from exact circle
            self.assertTrue(np.allclose(r, R_const, rtol=1e-7, atol=1e-7))
            self.assertTrue(np.allclose(z, Z_const, rtol=1e-9, atol=1e-9))
            # Plane indices should increase modulo len(phis)
            idx = hits[mask, 1].astype(int)
            self.assertEqual(len(idx), len(phis) * 3)
            self.assertTrue(np.all((idx[1:] - idx[:-1]) % len(phis) == 1))
        # res_tys should be a list with same length as inputs
        self.assertEqual(len(res_tys), len(RZ))
        for ty in res_tys:
            # Columns: t, x, y, z
            self.assertEqual(ty.shape[1], 4)

    def test_integrate_in_phi_cart_rotation(self):
        # Start at phi=0 on midplane, rotate by pi/2
        start_xyz = np.array([self.R0, 0.0, 0.0])
        end_xyz = self.intg.integrate_toroidally(start_xyz, delta_phi=np.pi/2, input_coordinates='cartesian', output_coordinates='cartesian')
        expected = np.array([0.0, self.R0, 0.0])
        self.assertTrue(np.allclose(end_xyz, expected, atol=5e-6))

        end_RZ = self.intg.integrate_toroidally(start_xyz, delta_phi=np.pi/2, input_coordinates='cartesian', output_coordinates='cylindrical')
        
        self.assertEqual(end_RZ.shape[0], 2)
        self.assertTrue(np.allclose(end_RZ[0], self.R0, atol=5e-6))
        self.assertTrue(np.allclose(end_RZ[1], 0.0, atol=5e-9))
        # Reconstruct xyz using start phi + delta
        phi_start = 0.0
        phi_end = phi_start + np.pi/2
        recon_xyz = Integrator._rphiz_to_xyz(np.array([end_RZ[0], phi_end, end_RZ[1]])[None, :])[0]
        self.assertTrue(np.allclose(recon_xyz, expected, atol=5e-6))

    def test_integrate_in_phi_cyl_rotation(self):
        # Start at phi=pi/4 on midplane, rotate by pi/2 using cylindrical input
        start_RZ = np.array([self.R0, 0.0])
        start_phi = np.pi/4
        end_xyz = self.intg.integrate_toroidally(start_RZ, delta_phi=np.pi/2, phi0=start_phi, input_coordinates='cylindrical', output_coordinates='cartesian')
        # Expected at phi=3pi/4
        phi_end = start_phi + np.pi/2
        expected = np.array([self.R0*np.cos(phi_end), self.R0*np.sin(phi_end), 0.0])
        self.assertTrue(np.allclose(end_xyz, expected, atol=5e-6))

        # Also test return_cartesian=False path: should give R,Z only
        end_RZ = self.intg.integrate_toroidally(start_RZ, delta_phi=np.pi/2, phi0=start_phi, input_coordinates='cylindrical', output_coordinates='cylindrical')
        self.assertEqual(end_RZ.shape[0], 2)
        self.assertTrue(np.allclose(end_RZ[0], self.R0, atol=5e-6))
        self.assertTrue(np.allclose(end_RZ[1], 0.0, atol=5e-9))
        recon_xyz = Integrator._rphiz_to_xyz(np.array([end_RZ[0], phi_end, end_RZ[1]])[None, :])[0]
        self.assertTrue(np.allclose(recon_xyz, expected, atol=5e-6))

    def test_integrate_fieldlinepoints_cart_and_cyl(self):
        # One transit around torus; points should lie on circle R=R0, Z=0
        start_xyz = np.array([self.R0, 0.0, 0.0])
        pts_cart = self.intg.integrate_fieldlinepoints(start_xyz, n_transits=1, input_coordinates='cartesian', output_coordinates='cartesian')
        self.assertEqual(pts_cart.shape[1], 3)
        r = np.sqrt(pts_cart[:, 0]**2 + pts_cart[:, 1]**2)
        z = pts_cart[:, 2]
        self.assertTrue(np.allclose(r, self.R0, rtol=1e-7, atol=1e-7))
        self.assertTrue(np.allclose(z, 0.0, atol=1e-9))

        # Cylindrical variant should be consistent
        start_RZ = np.array([self.R0, 0.0])
        start_phi = 0.0
        pts_cyl = self.intg.integrate_fieldlinepoints(start_RZ, start_phi, n_transits=1, input_coordinates='cylindrical', output_coordinates='cylindrical')
        self.assertEqual(pts_cyl.shape[1], 3)
        self.assertTrue(np.allclose(pts_cyl[:, 0], self.R0, atol=1e-7))
        self.assertTrue(np.allclose(pts_cyl[:, 2], 0.0, atol=1e-9))

    def test_integrate_right_direction(self):
        # W7X has B_phi in the negative phi direction; verify that field is flipped. 
        name = "w7x"
        base_curves, base_currents, ma, nfp, bs = get_data(name)
        # confirm that B_phi is negative:
        bs.set_points(ma.gamma()[0: 1])
        self.assertTrue(bs.B_cyl()[0, 1] < 0, msg="Expected B_phi < 0 for W7X configuration")
                        
        gamma = ma.gamma()
        start_xyz = gamma[0, :]
        intg = SimsoptFieldlineIntegrator(bs, nfp=nfp, tmax=1e3)
        axispoints = intg.integrate_fieldlinepoints(start_xyz, n_transits=0.5, input_coordinates='cartesian', output_coordinates='cylindrical')
        # check that phi is nevertheless strictly increasing:
        phis = axispoints[:, 1]
        dphis = np.diff(phis)
        self.assertTrue(np.all(dphis > 0), msg="Expected strictly increasing phi along integrated fieldline in W7X configuration")



class TestScipyFieldlineIntegrator(unittest.TestCase):
    def setUp(self):
        self.R0 = 1.1
        self.B0 = 0.7
        self.field = ToroidalField(self.R0, self.B0)
        self.intg = ScipyFieldlineIntegrator(self.field, nfp=1, stellsym=True, 
                                             integrator_type='RK45', integrator_args={'rtol': 1e-9, 'atol': 1e-11})

    def test_poincare_hits_and_trajectories(self):
        RZ = np.array([[self.R0 + 0.02, 0.0]])
        phis = np.linspace(0, 2*np.pi, 6, endpoint=False)
        hits = self.intg.compute_poincare_hits(RZ, n_transits=2, phis=phis, phi0=0.0)
        print(hits)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        # Expect one row per plane per transit
        self.assertEqual(h.shape[0], len(phis) * 2)
        # Radii constant and Z constant
        r = np.sqrt(h[:, 2]**2 + h[:, 3]**2)
        z = h[:, 4]
        self.assertTrue(np.allclose(r, RZ[0, 0], atol=1e-10))
        self.assertTrue(np.allclose(z, RZ[0, 1], atol=1e-10))
        # Plane index sequence
        idx = h[:, 1].astype(int)
        self.assertTrue(np.all((idx[1:] - idx[:-1]) % len(phis) == 1))

        # Trajectory sampling in phi (no plane filtering)
        tys = self.intg.compute_poincare_trajectories(RZ, n_transits=1, phi0=0.0)
        self.assertEqual(len(tys), 1)
        ty = tys[0]
        # Columns: phi, x, y, z (phi is the integrate variable here)
        self.assertEqual(ty.shape[1], 4)
        # Z equals initial Z, radius equals initial R
        r_traj = np.sqrt(ty[:, 1]**2 + ty[:, 2]**2)
        z_traj = ty[:, 3]
        self.assertTrue(np.allclose(r_traj, RZ[0, 0], atol=1e-9))
        self.assertTrue(np.allclose(z_traj, RZ[0, 1], atol=1e-12))

    def test_defaults(self):
        # Defaults should be reasonable and work
        intg2 = ScipyFieldlineIntegrator(self.field)
        self.assertEqual(intg2._integrator_args['rtol'], 1e-7)
        self.assertEqual(intg2._integrator_args['atol'], 1e-9)
        self.assertEqual(intg2._integrator_type, 'RK45')
        self.assertEqual(intg2.nfp, 1)

    def test_integrate_in_phi_cyl_rotation(self):
        # Start at phi=pi/6, rotate by pi/3
        RZ0 = np.array([self.R0, 0.0])
        phi_start = np.pi/6
        delta_phi = np.pi/3
        # Existing API uses start_phi positional parameter name
        RZ_end = self.intg.integrate_toroidally(RZ0, phi_start, delta_phi, input_coordinates='cylindrical', output_coordinates='cylindrical')
        end_xyz = Integrator._rphiz_to_xyz(np.array([RZ_end[0], phi_start + delta_phi, RZ_end[1]]))[-1]
        expected = np.array([self.R0*np.cos(phi_start + delta_phi), self.R0*np.sin(phi_start + delta_phi), 0.0])
        self.assertTrue(np.allclose(end_xyz, expected, atol=1e-6))

        # Also verify return_cartesian=True branch directly
        end_xyz_direct = self.intg.integrate_toroidally(RZ0, phi_start, delta_phi, input_coordinates='cylindrical', output_coordinates='cartesian')
        self.assertTrue(np.allclose(end_xyz_direct, expected, atol=1e-6))
        # And return_cartesian=False -> R,Z then reconstruct xyz
        end_RZ = self.intg.integrate_toroidally(RZ0, phi_start, delta_phi, input_coordinates='cylindrical', output_coordinates='cylindrical')
        # Scipy path should already return (2,), but be defensive:
        if end_RZ.shape[0] == 3:
            rphiz = Integrator._xyz_to_rphiz(end_RZ[None, :])[0]
            end_RZ = np.array([rphiz[0], rphiz[2]])
        self.assertEqual(end_RZ.shape[0], 2)
        self.assertTrue(np.allclose(end_RZ[0], self.R0, atol=1e-6))
        self.assertTrue(np.allclose(end_RZ[1], 0.0, atol=1e-9))
        recon_xyz = Integrator._rphiz_to_xyz(np.array([end_RZ[0], phi_start + delta_phi, end_RZ[1]])[None, :])[0]
        self.assertTrue(np.allclose(recon_xyz, expected, atol=1e-6))

    def test_compare_cart_cyl_rotation(self, start_RZ, start_phi, delta_phi):
        #compare cylindrical and cartesian integration paths
        end_xyz_from_cyl = self.intg.integrate_toroidally(start_RZ, start_phi, delta_phi, input_coordinates='cylindrical', output_coordinates='cartesian')
        start_xyz = Integrator._rphiz_to_xyz(np.array([start_RZ[0], start_phi, start_RZ[1]]))[0]
        end_xyz_from_cart = self.intg.integrate_toroidally(start_xyz, delta_phi, input_coordinates='cartesian', output_coordinates='cartesian')
        self.assertTrue(np.allclose(end_xyz_from_cyl, end_xyz_from_cart, atol=1e-6))

    def test_integrate_cyl_planes_and_fieldlinepoints(self):
        # Evaluate at specific phis and via fieldlinepoints helpers
        RZ0 = np.array([self.R0, 0.0])
        phis = np.linspace(0, 2*np.pi, 9, endpoint=True)
        status, rphiz = self.intg.integrate_cyl_planes(RZ0, phis, output_coordinates='cylindrical')
        self.assertEqual(status, 0)
        self.assertEqual(rphiz.shape, (len(phis), 3))
        self.assertTrue(np.allclose(rphiz[:, 0], self.R0, atol=1e-7))
        self.assertTrue(np.allclose(rphiz[:, 2], 0.0, atol=1e-9))

        # Fieldline points by RZ
        # Existing API: (start_RZ, start_phi, delta_phi, n_points, ...)
        pts_xyz = self.intg.integrate_fieldlinepoints(RZ0, 2*np.pi, 50, phi0=0.0, endpoint=True, input_coordinates='cylindrical', output_coordinates='cartesian')
        self.assertEqual(pts_xyz.shape, (50, 3))
        r = np.sqrt(pts_xyz[:, 0]**2 + pts_xyz[:, 1]**2)
        self.assertTrue(np.allclose(r, self.R0, atol=1e-6))
        self.assertTrue(np.allclose(pts_xyz[:, 2], 0.0, atol=1e-9))

        # Fieldline points by xyz convenience wrapper
        start_xyz = np.array([self.R0, 0.0, 0.0])
        pts2 = self.intg.integrate_fieldlinepoints(start_xyz, 2*np.pi, 60, endpoint=True, input_coordinates='cartesian', output_coordinates='cartesian')
        self.assertEqual(pts2.shape, (60, 3))
        r2 = np.sqrt(pts2[:, 0]**2 + pts2[:, 1]**2)
        self.assertTrue(np.allclose(r2, self.R0, atol=1e-6))
        self.assertTrue(np.allclose(pts2[:, 2], 0.0, atol=1e-9))


    def test_integrate_3d_fieldlinepoints_cart(self):
        # Integrate 3D arc length: quarter circle
        start_xyz = np.array([self.R0, 0.0, 0.0])
        l_total = self.R0 * (np.pi/2)
        pts = self.intg.integrate_3d_fieldlinepoints(start_xyz, l_total=l_total, n_points=40, input_coordinates='cartesian', output_coordinates='cartesian')
        self.assertEqual(pts.shape, (40, 3))
        # End point should be around phi=pi/2
        end_phi = np.arctan2(pts[-1, 1], pts[-1, 0])
        self.assertTrue(np.isfinite(end_phi))
        # Allow some tolerance due to adaptive stepping and solver
        self.assertTrue(abs(((end_phi - (np.pi/2) + np.pi) % (2*np.pi)) - np.pi) < 5e-3)
        # Radius and Z constant
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        self.assertTrue(np.allclose(r, self.R0, atol=1e-6))
        self.assertTrue(np.allclose(pts[:, 2], 0.0, atol=1e-9))

        start_RZ = start_xyz[:2]
        pts = self.intg.integrate_3d_fieldlinepoints(start_RZ, l_total=l_total, phi0=0, n_points=40, input_coordinates='cylindrical', output_coordinates='cylindrical')

    def test_lost_poincare(self):
        # integration should fail if toroidal field returns nans. Overload B_cyl to simulate this.
        R0 = 1.0
        B0 = 1.0
        field = ToroidalField(R0, B0)
        b_hidden = field.B_cyl
        global global_counter
        global_counter = 0
        def failing_field():
            global global_counter
            if global_counter < 100:
                global_counter += 1
                return b_hidden()
            else:
                return np.array([np.nan, np.nan, np.nan])
        field.B_cyl = failing_field
        intg = ScipyFieldlineIntegrator(field, nfp=1) #, integrator_args={'max_step':1e3})
        RZ = np.array([[R0 + 0.05, 0.0], [R0 + 0.10, 0.0]])
        phis = np.linspace(0, 2*np.pi, 8, endpoint=False)
        res_phi_hits = intg.compute_poincare_hits(RZ, n_transits=35, phis=phis, phi0=0.0)
        # the second integration failed and should have -1 as first index, indicating failure
        self.assertEqual(res_phi_hits[-1][-1, 1], -1)
        
        start_RZ = np.array([R0 + 0.05, 0.0])
        global_counter = 90
        endpoint_RZ = intg.integrate_toroidally(start_RZ, 0.0, 2*np.pi, input_coordinates='cylindrical', output_coordinates='cylindrical')
        #should be nans
        self.assertTrue(np.isnan(endpoint_RZ).all())

        # test integrate in cyl failure: 
        global_counter = 90
        with self.assertRaises(ObjectiveFailure):
            _ = intg.integrate_fieldlinepoints(start_RZ, 4*np.pi, n_points=50, phi0=0.0, input_coordinates='cylindrical', output_coordinates='cartesian')

        


class TestIntegratorAgreement(unittest.TestCase):
    def test_biotsavart_axis_endpoints_match_and_agree(self):
        # Compare both integrators on stellarator fields for all named configurations.
        # Start at the first magnetic axis point and integrate in phi to the last axis point.
        # Check: (a) each integrator hits the target axis point, (b) both agree with each other.
        # This is also a test of the configurations. 
        for name in configurations:
            if name == 'quasr':
                break   # do not clobber the external database, which also does not provide axes so this test cannot be performed
            with self.subTest(config=name):
                base_curves, base_currents, ma, nfp, bs = get_data(name)
                gamma = ma.gamma()
                start_xyz = gamma[0, :]
                target_xyz = gamma[-1, :]

                # Compute phi start/end directly from endpoints
                phi_start = np.arctan2(start_xyz[1], start_xyz[0])
                phi_end = np.arctan2(target_xyz[1], target_xyz[0])
                delta_phi = phi_end - phi_start

                # Simsopt integrator: integrate over delta_phi in Cartesian space
                so = SimsoptFieldlineIntegrator(
                    bs, nfp=nfp, tmax=5e4, tol=1e-10
                )
                end_xyz_so = so.integrate_toroidally(start_xyz, delta_phi=delta_phi, input_coordinates='cartesian', output_coordinates='cartesian')

                # Scipy integrator (cylindrical) over phi
                rphiz0 = Integrator._xyz_to_rphiz(start_xyz)[-1]
                RZ0 = np.array([rphiz0[0], rphiz0[2]])
                sc = ScipyFieldlineIntegrator(
                    bs, nfp=nfp, integrator_type='RK45', integrator_args={'rtol': 1e-10, 'atol': 1e-12}
                )
                # Adapt to existing signature: (start_RZ, start_phi, delta_phi)
                RZ_end = sc.integrate_toroidally(RZ0, phi_start, phi_end - phi_start, input_coordinates='cylindrical', output_coordinates='cylindrical')
                self.assertTrue(np.all(np.isfinite(RZ_end)), msg=f"scipy integrator produced non-finite result for config {name}")
                end_xyz_sc = Integrator._rphiz_to_xyz(np.array([RZ_end[0], phi_end, RZ_end[1]]))[-1]

                # Tolerances: strict agreement in meters
                # (the w7x axis is not really great...)
                tol_abs = 2e-3 # 2 mm distance tolerance
                err_so = np.linalg.norm(end_xyz_so - target_xyz)
                err_sc = np.linalg.norm(end_xyz_sc - target_xyz)
                agree = np.linalg.norm(end_xyz_so - end_xyz_sc)
                self.assertLess(
                    err_so,
                    tol_abs,
                    msg=f"[{name}] simsopt->target dist={err_so:.3e} > tol={tol_abs:.1e}; |scipy-target|={err_sc:.3e}; |so-scipy|={agree:.3e}",
                )
                self.assertLess(
                    err_sc,
                    tol_abs,
                    msg=f"[{name}] scipy->target dist={err_sc:.3e} > tol={tol_abs:.1e}; |simsopt-target|={err_so:.3e}; |so-scipy|={agree:.3e}",
                )
                self.assertLess(
                    agree,
                    tol_abs,
                    msg=f"[{name}] simsopt vs scipy dist={agree:.3e} > tol={tol_abs:.1e}; |simsopt-target|={err_so:.3e}; |scipy-target|={err_sc:.3e}",
                )
    

if __name__ == '__main__':
    unittest.main()
