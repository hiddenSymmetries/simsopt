import unittest
import numpy as np

from simsopt.field.magneticfieldclasses import ToroidalField, PoloidalField
from simsopt.field.integrator import Integrator, SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator
from simsopt.field.tracing import MinRStoppingCriterion
from simsopt.configs.zoo import get_data, configurations
from simsopt._core.util import ObjectiveFailure


class TestIntegratorBase(unittest.TestCase):
    def setUp(self):
        self.R0 = 1.3
        self.B0 = 0.8
        self.field = ToroidalField(self.R0, self.B0)

    def test_coordinate_roundtrip(self):
        rng = np.random.default_rng(0)
        pts_rphiz = np.column_stack([
            rng.uniform(self.R0 * 0.8, self.R0 * 1.2, size=10),
            rng.uniform(-np.pi, np.pi, size=10),
            rng.uniform(-0.5, 0.5, size=10),
        ])
        xyz = Integrator._rphiz_to_xyz(pts_rphiz)
        rphiz_back = Integrator._xyz_to_rphiz(xyz)
        self.assertTrue(np.allclose(pts_rphiz[:, 0], rphiz_back[:, 0], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(pts_rphiz[:, 2], rphiz_back[:, 2], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(np.cos(pts_rphiz[:, 1]), np.cos(rphiz_back[:, 1]), atol=1e-12))
        self.assertTrue(np.allclose(np.sin(pts_rphiz[:, 1]), np.sin(rphiz_back[:, 1]), atol=1e-12))

    def test_incorrect_staticmethods(self):
        with self.assertRaises(ValueError):
            Integrator._rphiz_to_xyz(1)
        with self.assertRaises(ValueError):
            Integrator._rphiz_to_xyz(np.random.random(4))
        with self.assertRaises(ValueError):
            Integrator._xyz_to_rphiz(1)
        with self.assertRaises(ValueError):
            Integrator._xyz_to_rphiz(np.random.random(2))

    def test_base_class_hooks_not_implemented(self):
        intg = Integrator(self.field)
        start_xyz = np.array([self.R0, 0.0, 0.0])
        with self.assertRaises(NotImplementedError):
            intg.integrate_toroidally(start_xyz, np.pi)
        with self.assertRaises(NotImplementedError):
            intg.integrate_fieldlinepoints(start_xyz, np.pi)
        with self.assertRaises(NotImplementedError):
            intg.compute_poincare_hits(np.array([[self.R0, 0.0]]), 1, phis=[0.0])


class TestIntegratorsCommonInterface(unittest.TestCase):
    """Tests that apply to both backends identically."""

    def setUp(self):
        self.R0 = 1.2
        self.B0 = 1.0
        self.field = ToroidalField(self.R0, self.B0)
        self.integrators = [SimsoptFieldlineIntegrator(self.field, tmax=100.0, tol=1e-10),
                            ScipyFieldlineIntegrator(self.field, integrator_args={'rtol': 1e-10, 'atol': 1e-12})]

    def test_invalid_inputs(self):
        start_xyz = np.array([self.R0, 0.0, 0.0])
        start_RZ = np.array([self.R0, 0.0])
        for intg in self.integrators:
            with self.subTest(integrator=type(intg).__name__):
                for method in [intg.integrate_toroidally, intg.integrate_fieldlinepoints]:
                    with self.assertRaises(ValueError):
                        method(start_xyz, np.pi/2, input_coordinates='invalid')
                    with self.assertRaises(ValueError):
                        method(start_xyz, np.pi/2, output_coordinates='invalid')
                    with self.assertRaises(ValueError):
                        method(start_RZ, np.pi/2, phi0=None, input_coordinates='cylindrical')
                    with self.assertRaises(ValueError):
                        method(start_xyz, np.pi/2, phi0=0.0, input_coordinates='cylindrical')
                    with self.assertRaises(ValueError):
                        method(start_RZ, np.pi/2, input_coordinates='cartesian')
                    with self.assertRaises(ValueError):
                        method(start_xyz, -np.pi/2)
                with self.assertRaises(ValueError):
                    intg.compute_poincare_hits(np.array([1.0, 0.0, 0.0]), 1)
        scipy_intg = self.integrators[1]
        with self.assertRaises(ValueError):
            scipy_intg.integrate_3d_fieldlinepoints(start_xyz, l_total=1.0, n_points=10, input_coordinates='invalid')
        with self.assertRaises(ValueError):
            scipy_intg.integrate_3d_fieldlinepoints(start_xyz, l_total=1.0, n_points=10, output_coordinates='invalid')
        with self.assertRaises(ValueError):
            scipy_intg.integrate_3d_fieldlinepoints(start_RZ, l_total=1.0, n_points=10, phi0=None, input_coordinates='cylindrical')

    def test_integrate_toroidally_rotation(self):
        # in a purely toroidal field, the end point is the start point rotated by delta_phi
        start_RZ = np.array([self.R0, 0.1])
        phi0 = np.pi/4
        for intg in self.integrators:
            for delta_phi in [0.0, np.pi/2, 3*np.pi]:
                with self.subTest(integrator=type(intg).__name__, delta_phi=delta_phi):
                    phi_end = phi0 + delta_phi
                    expected = np.array([self.R0*np.cos(phi_end), self.R0*np.sin(phi_end), 0.1])
                    end_xyz = intg.integrate_toroidally(start_RZ, delta_phi, phi0=phi0, input_coordinates='cylindrical')
                    np.testing.assert_allclose(end_xyz, expected, atol=1e-7)
                    end_RZ = intg.integrate_toroidally(start_RZ, delta_phi, phi0=phi0, input_coordinates='cylindrical',
                                                       output_coordinates='cylindrical')
                    np.testing.assert_allclose(end_RZ, start_RZ, atol=1e-7)
                    # cartesian input gives the same result
                    start_xyz = Integrator._rphiz_to_xyz(np.array([start_RZ[0], phi0, start_RZ[1]]))[0]
                    np.testing.assert_allclose(intg.integrate_toroidally(start_xyz, delta_phi), expected, atol=1e-7)

    def test_integrate_fieldlinepoints(self):
        start_RZ = np.array([self.R0, 0.0])
        for intg in self.integrators:
            with self.subTest(integrator=type(intg).__name__):
                # equally spaced points, over more than one transit
                pts = intg.integrate_fieldlinepoints(start_RZ, 3*np.pi, n_points=13, phi0=0.0, endpoint=True,
                                                     input_coordinates='cylindrical', output_coordinates='cylindrical')
                self.assertEqual(pts.shape, (13, 3))
                np.testing.assert_allclose(pts[:, 1], np.linspace(0, 3*np.pi, 13), atol=1e-12)
                np.testing.assert_allclose(pts[:, 0], self.R0, atol=1e-7)
                np.testing.assert_allclose(pts[:, 2], 0.0, atol=1e-9)
                pts = intg.integrate_fieldlinepoints(start_RZ, 2*np.pi, n_points=10, phi0=0.0,
                                                     input_coordinates='cylindrical', output_coordinates='cylindrical')
                np.testing.assert_allclose(pts[:, 1], np.linspace(0, 2*np.pi, 10, endpoint=False), atol=1e-12)
                # adaptive points, phi increasing and bounded by the end angle
                for endpoint in [True, False]:
                    pts = intg.integrate_fieldlinepoints(np.array([self.R0, 0.0, 0.0]), 2*np.pi, endpoint=endpoint,
                                                         output_coordinates='cylindrical')
                    self.assertTrue(np.all(np.diff(pts[:, 1]) > 0))
                    self.assertEqual(pts[0, 1], 0.0)
                    if endpoint:
                        self.assertAlmostEqual(pts[-1, 1], 2*np.pi)
                    else:
                        self.assertLess(pts[-1, 1], 2*np.pi)
                    xyz = intg.integrate_fieldlinepoints(np.array([self.R0, 0.0, 0.0]), 2*np.pi, endpoint=endpoint)
                    np.testing.assert_allclose(np.linalg.norm(xyz[:, :2], axis=1), self.R0, atol=1e-7)

    def test_poincare_hits(self):
        RZ = np.array([[self.R0 + 0.05, 0.0], [self.R0 + 0.10, 0.02]])
        phis = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for intg in self.integrators:
            with self.subTest(integrator=type(intg).__name__):
                res_tys, res_phi_hits = intg.compute_poincare_hits(RZ, n_transits=3, phis=phis, phi0=0.1)
                self.assertEqual(len(res_tys), len(RZ))
                self.assertEqual(len(res_phi_hits), len(RZ))
                for i, hits in enumerate(res_phi_hits):
                    self.assertEqual(hits.shape[1], 5)
                    self.assertEqual(res_tys[i].shape[1], 4)
                    # terminating row: n_transits completed
                    self.assertEqual(hits[-1, 1], -1)
                    planes = hits[:-1]
                    self.assertEqual(len(planes), len(phis) * 3)
                    np.testing.assert_allclose(np.linalg.norm(planes[:, 2:4], axis=1), RZ[i, 0], atol=1e-7)
                    np.testing.assert_allclose(planes[:, 4], RZ[i, 1], atol=1e-9)
                    # the plane index is consistent with the position of the hit
                    idx = planes[:, 1].astype(int)
                    hit_phi = np.arctan2(planes[:, 3], planes[:, 2])
                    np.testing.assert_allclose(np.cos(hit_phi), np.cos(phis[idx]), atol=1e-7)
                    np.testing.assert_allclose(np.sin(hit_phi), np.sin(phis[idx]), atol=1e-7)
                    # the first plane crossed is the first plane after phi0
                    self.assertEqual(idx[0], 1)
                    self.assertTrue(np.all((idx[1:] - idx[:-1]) % len(phis) == 1))

    def test_poincare_hits_no_planes(self):
        RZ = np.array([[self.R0 + 0.05, 0.0]])
        for intg in self.integrators:
            with self.subTest(integrator=type(intg).__name__):
                _, res_phi_hits = intg.compute_poincare_hits(RZ, n_transits=1, phis=[])
                self.assertEqual(res_phi_hits[0].shape, (1, 5))
                self.assertEqual(res_phi_hits[0][0, 1], -1)


class TestSimsoptFieldlineIntegrator(unittest.TestCase):
    def test_defaults(self):
        field = ToroidalField(1.0, 1.0)
        intg = SimsoptFieldlineIntegrator(field)
        self.assertEqual(intg.stopping_criteria, [])
        self.assertEqual(intg.tol, 1e-9)
        self.assertEqual(intg.tmax, 1e4)

    def test_stopping_criterion(self):
        # a field line at R=1.1 is stopped by MinRStoppingCriterion(1.2) immediately
        field = ToroidalField(1.0, 1.0)
        intg = SimsoptFieldlineIntegrator(field, stopping_criteria=[MinRStoppingCriterion(1.2)], tmax=100)
        _, res_phi_hits = intg.compute_poincare_hits(np.array([[1.1, 0.0], [1.3, 0.0]]), 2, phis=[0.0])
        self.assertEqual(res_phi_hits[0][-1, 1], -2)
        self.assertEqual(res_phi_hits[1][-1, 1], -1)

    def test_integrate_right_direction(self):
        # W7X has B_phi in the negative phi direction; verify that field is flipped.
        base_curves, base_currents, ma, nfp, bs = get_data("w7x")
        bs.set_points(ma.gamma()[0:1])
        self.assertTrue(bs.B_cyl()[0, 1] < 0, msg="Expected B_phi < 0 for W7X configuration")
        start_xyz = ma.gamma()[0, :]
        intg = SimsoptFieldlineIntegrator(bs, tmax=1e3)
        axispoints = intg.integrate_fieldlinepoints(start_xyz, np.pi, output_coordinates='cylindrical')
        self.assertTrue(np.all(np.diff(axispoints[:, 1]) > 0),
                        msg="Expected strictly increasing phi along integrated fieldline in W7X configuration")

    def test_failure(self):
        # tmax too short to reach the end angle
        field = ToroidalField(1.0, 1.0)
        intg = SimsoptFieldlineIntegrator(field, tmax=1.0)
        start_RZ = np.array([1.0, 0.0])
        self.assertTrue(np.all(np.isnan(intg.integrate_toroidally(start_RZ, np.pi, phi0=0, input_coordinates='cylindrical',
                                                                  output_coordinates='cylindrical'))))
        with self.assertRaises(ObjectiveFailure):
            intg.integrate_fieldlinepoints(start_RZ, np.pi, n_points=5, phi0=0, input_coordinates='cylindrical')
        with self.assertRaises(ObjectiveFailure):
            intg.integrate_fieldlinepoints(start_RZ, np.pi, phi0=0, endpoint=True, input_coordinates='cylindrical')


class TestScipyFieldlineIntegrator(unittest.TestCase):
    def setUp(self):
        self.R0 = 1.1
        self.field = ToroidalField(self.R0, 0.7)

    def test_defaults(self):
        intg = ScipyFieldlineIntegrator(self.field)
        self.assertEqual(intg._integrator_args['rtol'], 1e-7)
        self.assertEqual(intg._integrator_args['atol'], 1e-9)
        self.assertEqual(intg._integrator_type, 'RK45')

    def test_integrator_args_not_shared(self):
        args = {'rtol': 1e-5}
        intg1 = ScipyFieldlineIntegrator(self.field, integrator_args=args)
        intg2 = ScipyFieldlineIntegrator(self.field)
        self.assertEqual(args, {'rtol': 1e-5})
        self.assertEqual(intg1._integrator_args['rtol'], 1e-5)
        self.assertEqual(intg2._integrator_args['rtol'], 1e-7)

    def test_trajectories(self):
        intg = ScipyFieldlineIntegrator(self.field, trajectory_points_per_transit=20)
        RZ = np.array([[self.R0 + 0.02, 0.0]])
        res_tys, res_phi_hits = intg.compute_poincare_hits(RZ, n_transits=2, phis=[0.0], phi0=0.0)
        self.assertEqual(res_tys[0].shape, (41, 4))
        np.testing.assert_allclose(res_tys[0][:, 0], np.linspace(0, 4*np.pi, 41))
        np.testing.assert_allclose(np.linalg.norm(res_tys[0][:, 1:3], axis=1), RZ[0, 0], atol=1e-6)
        res_tys, res_phi_hits_2 = intg.compute_poincare_hits(RZ, n_transits=2, phis=[0.0], phi0=0.0, return_trajectories=False)
        self.assertIsNone(res_tys)
        np.testing.assert_allclose(res_phi_hits[0], res_phi_hits_2[0])

    def test_integrate_3d_fieldlinepoints(self):
        intg = ScipyFieldlineIntegrator(self.field)
        start_xyz = np.array([self.R0, 0.0, 0.0])
        l_total = self.R0 * (np.pi/2)
        pts = intg.integrate_3d_fieldlinepoints(start_xyz, l_total=l_total, n_points=40)
        self.assertEqual(pts.shape, (40, 3))
        end_phi = np.arctan2(pts[-1, 1], pts[-1, 0])
        self.assertLess(abs(end_phi - np.pi/2), 5e-3)
        np.testing.assert_allclose(np.linalg.norm(pts[:, :2], axis=1), self.R0, atol=1e-6)
        np.testing.assert_allclose(pts[:, 2], 0.0, atol=1e-9)
        pts_cyl = intg.integrate_3d_fieldlinepoints(start_xyz[[0, 2]], l_total=l_total, phi0=0, n_points=40,
                                                    input_coordinates='cylindrical', output_coordinates='cylindrical')
        np.testing.assert_allclose(pts_cyl[:, 0], self.R0, atol=1e-6)

    def test_lost_fieldline(self):
        # integration fails if the field returns nans. Overload B_cyl to simulate this.
        R0 = 1.0
        field = ToroidalField(R0, 1.0)
        b_hidden = field.B_cyl
        counter = {'n': 0}

        def failing_field():
            counter['n'] += 1
            if counter['n'] <= 100:
                return b_hidden()
            return np.array([[np.nan, np.nan, np.nan]])
        field.B_cyl = failing_field
        intg = ScipyFieldlineIntegrator(field)
        RZ = np.array([[R0 + 0.05, 0.0], [R0 + 0.10, 0.0]])
        phis = np.linspace(0, 2*np.pi, 8, endpoint=False)
        res_tys, res_phi_hits = intg.compute_poincare_hits(RZ, n_transits=35, phis=phis, phi0=0.0)
        # the second integration failed and is marked with idx=-2
        self.assertEqual(res_phi_hits[-1][-1, 1], -2)
        self.assertTrue(res_tys[-1][-1, 0] < 35*2*np.pi)

        start_RZ = np.array([R0 + 0.05, 0.0])
        self.assertTrue(np.isnan(intg.integrate_toroidally(start_RZ, 2*np.pi, phi0=0.0, input_coordinates='cylindrical',
                                                           output_coordinates='cylindrical')).all())
        with self.assertRaises(ObjectiveFailure):
            intg.integrate_fieldlinepoints(start_RZ, 4*np.pi, n_points=50, phi0=0.0, input_coordinates='cylindrical')

    def test_start_failures(self):
        # a field line cannot be followed in phi where B_phi vanishes or is too small
        start_RZ = np.array([1.2, 0.0])
        for field, status in [(PoloidalField(1.0, 1.0, 1.0), -1),
                              (ToroidalField(1.0, 1e-5) + PoloidalField(1.0, 1.0, 1.0), 1)]:
            intg = ScipyFieldlineIntegrator(field)
            self.assertEqual(intg._solve(start_RZ, [0, 1.0]).status, status)
            self.assertTrue(np.all(np.isnan(intg.integrate_toroidally(start_RZ, 1.0, phi0=0, input_coordinates='cylindrical'))))
            res_tys, res_phi_hits = intg.compute_poincare_hits(start_RZ[None, :], 1, phis=[1.0])
            self.assertEqual(res_phi_hits[0].shape, (1, 5))
            self.assertEqual(res_phi_hits[0][0, 1], -2)
            self.assertEqual(res_tys[0].shape, (1, 4))


class TestIntegratorAgreement(unittest.TestCase):
    def test_biotsavart_axis_endpoints_match_and_agree(self):
        # Compare both integrators on stellarator fields for all named configurations.
        # Start at the first magnetic axis point and integrate in phi to the last axis point.
        # Check: (a) each integrator hits the target axis point, (b) both agree with each other.
        # This is also a test of the configurations.
        for name in configurations:
            if name == 'quasr':
                continue  # the external database does not provide axes
            with self.subTest(config=name):
                base_curves, base_currents, ma, nfp, bs = get_data(name)
                gamma = ma.gamma()
                start_xyz = gamma[0, :]
                target_xyz = gamma[-1, :]
                phi_start = np.arctan2(start_xyz[1], start_xyz[0])
                phi_end = np.arctan2(target_xyz[1], target_xyz[0])
                delta_phi = np.mod(phi_end - phi_start, 2*np.pi)

                so = SimsoptFieldlineIntegrator(bs, tmax=5e4, tol=1e-10)
                sc = ScipyFieldlineIntegrator(bs, integrator_args={'rtol': 1e-10, 'atol': 1e-12})
                end_xyz_so = so.integrate_toroidally(start_xyz, delta_phi)
                end_xyz_sc = sc.integrate_toroidally(start_xyz, delta_phi)
                self.assertTrue(np.all(np.isfinite(end_xyz_sc)), msg=f"scipy integrator produced non-finite result for config {name}")

                tol_abs = 2e-3  # the w7x axis is not very accurate
                err_so = np.linalg.norm(end_xyz_so - target_xyz)
                err_sc = np.linalg.norm(end_xyz_sc - target_xyz)
                agree = np.linalg.norm(end_xyz_so - end_xyz_sc)
                self.assertLess(err_so, tol_abs, msg=f"[{name}] |simsopt-target|={err_so:.3e}")
                self.assertLess(err_sc, tol_abs, msg=f"[{name}] |scipy-target|={err_sc:.3e}")
                self.assertLess(agree, tol_abs, msg=f"[{name}] |simsopt-scipy|={agree:.3e}")

    def test_off_axis_agreement(self):
        # off the axis, consecutive transits differ, so this checks that the
        # correct crossing is selected when delta_phi exceeds 2*pi.
        base_curves, base_currents, ma, nfp, bs = get_data('ncsx')
        start_xyz = ma.gamma()[0, :] + np.array([0.05, 0.0, 0.0])
        so = SimsoptFieldlineIntegrator(bs, tol=1e-10)
        sc = ScipyFieldlineIntegrator(bs, integrator_args={'rtol': 1e-10, 'atol': 1e-12})
        for delta_phi in [np.pi/3, 3*np.pi]:
            with self.subTest(delta_phi=delta_phi):
                np.testing.assert_allclose(so.integrate_toroidally(start_xyz, delta_phi),
                                           sc.integrate_toroidally(start_xyz, delta_phi), atol=1e-7)
        pts_so = so.integrate_fieldlinepoints(start_xyz, 3*np.pi, n_points=7, endpoint=True)
        pts_sc = sc.integrate_fieldlinepoints(start_xyz, 3*np.pi, n_points=7, endpoint=True)
        np.testing.assert_allclose(pts_so, pts_sc, atol=1e-7)
        RZ = np.array([[np.linalg.norm(start_xyz[:2]), start_xyz[2]]])
        phis = np.linspace(0, 2*np.pi/nfp, 4, endpoint=False)
        _, hits_so = so.compute_poincare_hits(RZ, 3, phis=phis, phi0=0.3)
        _, hits_sc = sc.compute_poincare_hits(RZ, 3, phis=phis, phi0=0.3)
        # plane crossings agree; the terminating row is where the last step ended, which differs between backends
        np.testing.assert_allclose(hits_so[0][:-1, 1:], hits_sc[0][:-1, 1:], atol=1e-7)
        self.assertEqual(hits_so[0][-1, 1], hits_sc[0][-1, 1])


if __name__ == '__main__':
    unittest.main()
