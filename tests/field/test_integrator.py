"""
Tests for :mod:`simsopt.field.integrator`.

Most tests use :class:`~simsopt.field.magneticfieldclasses.ToroidalField`,
whose field B = B0*R0/R points in the toroidal direction, so that its field
lines are circles of constant R and Z. Every integration result is then known
exactly: following a field line over delta_phi rotates the start point by
delta_phi about the z axis. Tests on stellarator coil sets (NCSX, W7-X, ...)
instead compare the two backends with each other and with known periodic field
lines (magnetic axes and island X- and O-points).
"""
import unittest
import numpy as np

from simsopt.field.magneticfieldclasses import ToroidalField, PoloidalField
from simsopt.field.integrator import Integrator, SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator
from simsopt.field.tracing import MinRStoppingCriterion, MaxRStoppingCriterion, IterationStoppingCriterion
from simsopt.configs.zoo import get_data, configurations
from simsopt.geo import PeriodicFieldLine
from simsopt._core.util import ObjectiveFailure


class TestIntegratorBase(unittest.TestCase):
    """Backend-independent parts of the :class:`Integrator` base class."""

    def setUp(self):
        self.R0 = 1.3
        self.B0 = 0.8
        self.field = ToroidalField(self.R0, self.B0)

    def test_coordinate_roundtrip(self):
        """
        Converting cylindrical (R, phi, Z) to Cartesian and back recovers the
        points. phi is compared through cos and sin, since it is only defined
        modulo 2*pi.
        """
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
        """
        The coordinate conversions accept only (3,) or (n,3) arrays, and raise
        a ValueError for anything else.
        """
        with self.assertRaises(ValueError):
            Integrator._rphiz_to_xyz(1)
        with self.assertRaises(ValueError):
            Integrator._rphiz_to_xyz(np.random.random(4))
        with self.assertRaises(ValueError):
            Integrator._xyz_to_rphiz(1)
        with self.assertRaises(ValueError):
            Integrator._xyz_to_rphiz(np.random.random(2))

    def test_base_class_hooks_not_implemented(self):
        """
        The base class only implements the public interface. The integration
        itself is delegated to private hooks that each backend must implement,
        so calling the public methods on a bare Integrator raises
        NotImplementedError.
        """
        intg = Integrator(self.field)
        start_xyz = np.array([self.R0, 0.0, 0.0])
        with self.assertRaises(NotImplementedError):
            intg.integrate_toroidally(start_xyz, np.pi)
        with self.assertRaises(NotImplementedError):
            intg.integrate_fieldlinepoints(start_xyz, np.pi)
        with self.assertRaises(NotImplementedError):
            intg.compute_poincare_hits(np.array([[self.R0, 0.0]]), 1, phis=[0.0])


class TestIntegratorsCommonInterface(unittest.TestCase):
    """
    Tests that apply to both backends identically, in a purely toroidal field.
    Its field lines are circles at constant R and Z, so every result is known
    exactly, and both backends must reproduce it through the same interface.
    """

    def setUp(self):
        self.R0 = 1.2
        self.B0 = 1.0
        self.field = ToroidalField(self.R0, self.B0)
        self.integrators = [SimsoptFieldlineIntegrator(self.field, tmax=100.0, tol=1e-10),
                            ScipyFieldlineIntegrator(self.field, integrator_args={'rtol': 1e-10, 'atol': 1e-12})]

    def test_invalid_inputs(self):
        """
        Inconsistent input is rejected with a ValueError by both backends:
        unknown coordinate systems, cylindrical input without phi0, start
        points of the wrong length for the coordinate system, and negative
        delta_phi (field lines are always traced towards increasing phi).
        """
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
        """
        In a purely toroidal field, following a field line over delta_phi
        rotates the start point by delta_phi about the z axis, keeping R and Z
        fixed. This holds for delta_phi = 0, for less than a transit, and for
        more than one transit (3*pi), and is independent of whether the start
        point is given in Cartesian or cylindrical coordinates.
        """
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
        """
        Points along a field line of the toroidal field lie on the circle
        R=R0, Z=0. With n_points they are equally spaced in phi, also over more
        than one transit; without, they are the adaptive solver steps, with phi
        strictly increasing and bounded by the end angle, which is included
        only if endpoint=True.
        """
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
        """
        Poincare sections of the toroidal field. Each field line crosses every
        plane once per transit, at its own R and Z, so 3 transits through 8
        planes give 24 crossings, followed by a terminating row with idx=-1
        (transits completed). The plane index of each crossing matches its
        position, and the planes are crossed in order, starting with the first
        plane after phi0.
        """
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
        """
        Without planes, no crossings are recorded, and res_phi_hits contains
        only the terminating row.
        """
        RZ = np.array([[self.R0 + 0.05, 0.0]])
        for intg in self.integrators:
            with self.subTest(integrator=type(intg).__name__):
                _, res_phi_hits = intg.compute_poincare_hits(RZ, n_transits=1, phis=[])
                self.assertEqual(res_phi_hits[0].shape, (1, 5))
                self.assertEqual(res_phi_hits[0][0, 1], -1)


class TestSimsoptFieldlineIntegrator(unittest.TestCase):
    """Behaviour specific to the backend that wraps the C++ tracing routines."""

    def test_defaults(self):
        """
        Default settings: no stopping criteria, tolerance 1e-9, and an
        integration time of 1e4, roughly 10 km of field line.
        """
        field = ToroidalField(1.0, 1.0)
        intg = SimsoptFieldlineIntegrator(field)
        self.assertEqual(intg.stopping_criteria, [])
        self.assertEqual(intg.tol, 1e-9)
        self.assertEqual(intg.tmax, 1e4)

    def test_integrate_right_direction(self):
        """
        In W7-X the toroidal field points towards decreasing phi. The C++
        routine follows B, so the integrator must reverse the field to keep
        tracing towards increasing phi, as all integrators do. phi must
        therefore increase strictly along the traced axis.
        """
        base_curves, base_currents, ma, nfp, bs = get_data("w7x")
        bs.set_points(ma.gamma()[0:1])
        self.assertTrue(bs.B_cyl()[0, 1] < 0, msg="Expected B_phi < 0 for W7X configuration")
        start_xyz = ma.gamma()[0, :]
        intg = SimsoptFieldlineIntegrator(bs, tmax=1e3)
        axispoints = intg.integrate_fieldlinepoints(start_xyz, np.pi, output_coordinates='cylindrical')
        self.assertTrue(np.all(np.diff(axispoints[:, 1]) > 0),
                        msg="Expected strictly increasing phi along integrated fieldline in W7X configuration")

    def test_failure(self):
        """
        If the integration time tmax runs out before the requested angle is
        reached (here 1 m of field line for half a transit at R=1 m),
        integrate_toroidally returns NaNs, and integrate_fieldlinepoints raises
        ObjectiveFailure.
        """
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
    """
    Behaviour specific to the backend that solves dR/dphi = R B_R/B_phi,
    dZ/dphi = R B_Z/B_phi with scipy, using phi as the independent variable.
    """

    def setUp(self):
        self.R0 = 1.1
        self.field = ToroidalField(self.R0, 0.7)

    def test_defaults(self):
        """
        Default solver settings: RK45 with rtol=1e-7 and atol=1e-9.
        """
        intg = ScipyFieldlineIntegrator(self.field)
        self.assertEqual(intg._integrator_args['rtol'], 1e-7)
        self.assertEqual(intg._integrator_args['atol'], 1e-9)
        self.assertEqual(intg._integrator_type, 'RK45')

    def test_integrator_args_not_shared(self):
        """
        Solver settings are copied on construction: the defaults that are
        filled in neither modify the dictionary passed by the user, nor leak
        into other integrators.
        """
        args = {'rtol': 1e-5}
        intg1 = ScipyFieldlineIntegrator(self.field, integrator_args=args)
        intg2 = ScipyFieldlineIntegrator(self.field)
        self.assertEqual(args, {'rtol': 1e-5})
        self.assertEqual(intg1._integrator_args['rtol'], 1e-5)
        self.assertEqual(intg2._integrator_args['rtol'], 1e-7)

    def test_trajectories(self):
        """
        Trajectories and plane crossings come from the same solve. Trajectories
        are sampled at trajectory_points_per_transit points per transit, equally
        spaced in phi, and lie on the circle R=R0+0.02. Skipping the
        trajectories gives the same plane crossings.
        """
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
        """
        Integration in arc length, dx/ds = B/|B|, which unlike the phi
        formulation also works where B_phi changes sign. A quarter of the circle
        R=R0 has length R0*pi/2, so it ends at phi=pi/2, and all points stay at
        R=R0, Z=0.
        """
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
        """
        A field that starts returning NaN partway through the integration, as
        a field may outside its domain of validity. The field line still being traced is marked
        lost (terminating row idx=-2, before the requested transits), and the
        single field line methods return NaNs or raise ObjectiveFailure.
        """
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
        """
        The phi formulation is singular where B_phi vanishes. In a purely
        poloidal field the right hand side is not finite at the start point
        (status -1); in a field with a tiny toroidal component, |B_phi|/|B| is
        already below the threshold (status 1). In both cases integration is
        not started at all, since scipy's initial step selection does not
        terminate on non-finite input, and the field line is reported lost.
        """
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
    """
    The two backends solve different ODEs (3D Cartesian with the C++ routines,
    and R, Z as functions of phi with scipy). Agreement between them, and with
    known field lines of stellarator coil sets, verifies both.
    """

    def test_stopping_criteria(self):
        """
        Stopping criteria behave the same in both backends. A field line at
        R=1.1 is stopped by MinRStoppingCriterion(1.2), the second criterion,
        so its terminating row has idx=-2-1=-3; a field line at R=1.3 completes
        its transits (idx=-1). An iteration limit stops a field line early.
        Stopping criteria only apply to Poincare sections, not to single field
        line methods.
        """
        field = ToroidalField(1.0, 1.0)
        for cls, kwargs in [(SimsoptFieldlineIntegrator, {'tmax': 100}), (ScipyFieldlineIntegrator, {})]:
            with self.subTest(integrator=cls.__name__):
                intg = cls(field, stopping_criteria=[IterationStoppingCriterion(10**6), MinRStoppingCriterion(1.2)], **kwargs)
                _, res_phi_hits = intg.compute_poincare_hits(np.array([[1.1, 0.0], [1.3, 0.0]]), 2, phis=[0.0])
                self.assertEqual(res_phi_hits[0][-1, 1], -3)
                self.assertEqual(res_phi_hits[1][-1, 1], -1)
                # an iteration limit stops the field line before the transits are completed
                intg = cls(field, stopping_criteria=[IterationStoppingCriterion(3)], **kwargs)
                _, res_phi_hits = intg.compute_poincare_hits(np.array([[1.3, 0.0]]), 2, phis=[0.0])
                self.assertEqual(res_phi_hits[0][-1, 1], -2)
                # stopping criteria are ignored outside compute_poincare_hits
                intg = cls(field, stopping_criteria=[MinRStoppingCriterion(1.2)], **kwargs)
                end_RZ = intg.integrate_toroidally(np.array([1.1, 0.0]), np.pi, phi0=0.0, input_coordinates='cylindrical',
                                                   output_coordinates='cylindrical')
                np.testing.assert_allclose(end_RZ, [1.1, 0.0], atol=1e-7)

    def test_stopping_criteria_location(self):
        """
        A field line outside the NCSX plasma moves outward and is stopped when
        it crosses R=R_max. The C++ backend checks criteria after each step, so
        it stops just beyond R_max; the scipy backend locates the crossing by
        root finding, so it stops on R_max. Both record the same plane crossings
        before that.
        """
        _, _, ma, nfp, bs = get_data('ncsx')
        R_axis = np.linalg.norm(ma.gamma()[0, :2])
        R_max = R_axis + 0.2
        RZ = np.array([[R_axis + 0.18, 0.0]])
        so = SimsoptFieldlineIntegrator(bs, stopping_criteria=[MaxRStoppingCriterion(R_max)], tol=1e-10)
        sc = ScipyFieldlineIntegrator(bs, stopping_criteria=[MaxRStoppingCriterion(R_max)],
                                      integrator_args={'rtol': 1e-10, 'atol': 1e-12})
        phis = np.linspace(0, 2*np.pi, 32, endpoint=False)
        _, hits_so = so.compute_poincare_hits(RZ, 5, phis=phis)
        _, hits_sc = sc.compute_poincare_hits(RZ, 5, phis=phis)
        self.assertGreater(len(hits_sc[0]), 1)
        self.assertEqual(hits_so[0][-1, 1], -2)
        self.assertEqual(hits_sc[0][-1, 1], -2)
        R_stop_sc = np.linalg.norm(hits_sc[0][-1, 2:4])
        R_stop_so = np.linalg.norm(hits_so[0][-1, 2:4])
        self.assertAlmostEqual(R_stop_sc, R_max, places=6)
        self.assertGreaterEqual(R_stop_so, R_max)
        # both record the same plane crossings before stopping
        np.testing.assert_allclose(hits_so[0][:-1, 1:], hits_sc[0][:-1, 1:], atol=1e-7)

    def test_biotsavart_axis_endpoints_match_and_agree(self):
        """
        For every zoo configuration, the field line starting on the magnetic
        axis stays on it. Following it from the first to the last sample point
        of the axis must end on that point, for both backends, which then also
        agree with each other. This also tests the axis data of the
        configurations, accurate to about 2 mm (W7-X).
        """
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
        """
        Off the axis, a field line rotates around the axis, so consecutive
        transits end in different places. Both backends must agree on the end
        point after delta_phi, also for delta_phi > 2*pi where the C++ backend
        has to select the crossing on the right transit, and on points along the
        field line and Poincare sections.
        """
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


class TestPeriodicFieldline(unittest.TestCase):
    """
    Periodic field lines: the magnetic axis, and the X- and O-points of island
    chains, which close on themselves after a number of toroidal transits.
    """

    def test_magnetic_axis_from_offset(self):
        """
        Starting 1% of the major radius outward from the magnetic axis of each
        zoo configuration, periodic_fieldline traces one field period, fits a
        curve, and converges to the axis, the periodic field line nearby.
        Converted to a CurveRZFourier, it matches the axis of the configuration,
        which is accurate to about 5e-4 m (W7-X) and 1e-4 m (others). This
        includes W7-X and the LHD-like configuration, where B_phi < 0 and the
        curve must run towards decreasing phi.
        """
        for name in configurations:
            if name == 'quasr':
                continue  # the external database does not provide axes
            _, _, ma, nfp, bs = get_data(name)
            start_xyz = ma.gamma()[0] + np.array([0.01*ma.x[0], 0.0, 0.0])
            for cls in [SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator]:
                with self.subTest(config=name, integrator=cls.__name__):
                    fieldline = cls(bs).periodic_fieldline(start_xyz, order=ma.order, field_nfp=nfp)
                    self.assertIsInstance(fieldline, PeriodicFieldLine)
                    self.assertTrue(fieldline.res['success'])
                    rz = fieldline.curve.to_RZFourier(order=ma.order, quadpoints=ma.quadpoints, nfp=ma.nfp)
                    np.testing.assert_allclose(rz.gamma(), ma.gamma(), atol=1e-3)
                    np.testing.assert_allclose(rz.x, ma.x, atol=1e-3)

    def test_fieldline_symmetry(self):
        """
        Symmetry of a periodic field line on the rational surface iota = n/m,
        in a field with nfp field periods. It closes after m/gcd(n, m) transits,
        and the island chain consists of gcd(n, m) distinct field lines, each
        with nfp/gcd(n, m) field periods if the field periods map them onto
        each other. If that is not possible, or the field periods and transits
        are not coprime (as CurveXYZFourierSymmetries requires), fewer field
        periods are used. The sign of m only sets the winding direction.
        """
        cases = [((3, None), (3, 1)),    # magnetic axis
                 ((3, (3, 4)), (3, 4)),  # one field line, closes after 4 transits
                 ((5, (5, 5)), (1, 1)),  # five field lines, each closes after 1 transit
                 ((4, (2, 6)), (2, 3)),  # two field lines, each with 2 field periods
                 ((3, (2, 4)), (1, 2)),  # two field lines cannot be a single orbit of 3 periods
                 ((2, (1, 2)), (1, 2)),  # 2 field periods and 2 transits are not coprime
                 ((3, (3, -7)), (3, 7))]  # counter-clockwise winding
        for (nfp, iota), expected in cases:
            with self.subTest(nfp=nfp, iota=iota):
                self.assertEqual(Integrator._fieldline_symmetry(nfp, iota), expected)
        with self.assertRaises(ValueError):
            Integrator._fieldline_symmetry(2, (1, 0))

    def test_ncsx_island(self):
        """
        O- and X-point of the iota=3/7 island chain of NCSX. Their field line
        closes after 7 toroidal transits, making 3 poloidal turns in the
        counter-clockwise direction (m=-7), and has the 3 field periods of NCSX.
        One curve period therefore spans 7 field periods, so the curve needs
        roughly 7 times the Fourier order of the axis. The solved curve must
        pass through the fixed point on the phi=0 plane.
        """
        _, _, ma, nfp, bs = get_data('ncsx', coil_order=12, points_per_period=4)
        fixed_points = {'O': np.array([1.52288140, 0.0]), 'X': np.array([1.69779218, 0.0])}
        for kind, RZ in fixed_points.items():
            for cls in [SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator]:
                with self.subTest(point=kind, integrator=cls.__name__):
                    fieldline = cls(bs).periodic_fieldline(RZ + np.array([1e-3, 0.0]), order=80, field_nfp=nfp, iota=(3, -7),
                                                           phi0=0.0, input_coordinates='cylindrical')
                    self.assertTrue(fieldline.res['success'])
                    self.assertEqual((fieldline.curve.nfp, abs(fieldline.curve.ntor)), (3, 7))
                    # the curve passes through the fixed point on the phi=0 plane
                    start = fieldline.curve.gamma()[0]
                    np.testing.assert_allclose([np.linalg.norm(start[:2]), start[2]], RZ, atol=1e-5)

    def test_find_periodic_point(self):
        """
        Root find for the crossing of a periodic field line with phi=0: the
        magnetic axis (period 1 field period) and the O- and X-point of the
        iota=3/7 island chain (period 7 transits). Both backends find the known
        fixed points, which return to themselves after their period, and agree
        with each other. With the island period, a start near the axis converges
        to the axis, which already returns after one field period; this point of
        lower periodicity must be rejected.
        """
        _, _, ma, nfp, bs = get_data('ncsx', coil_order=12, points_per_period=4)
        axis_guess = ma.gamma()[0][[0, 2]]  # the zoo axis is not exact for this coil order
        cases = {'axis': (None, axis_guess, None),
                 'O': ((3, -7), np.array([1.52288140, 0.0]), np.array([1.52288140, 0.0])),
                 'X': ((3, -7), np.array([1.69779218, 0.0]), np.array([1.69779218, 0.0]))}
        for kind, (iota, guess, expected) in cases.items():
            found = []
            for cls in [SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator]:
                with self.subTest(point=kind, integrator=cls.__name__):
                    intg = cls(bs, tol=1e-11) if cls is SimsoptFieldlineIntegrator else \
                        cls(bs, integrator_args={'rtol': 1e-10, 'atol': 1e-12})
                    RZ = intg.find_periodic_point(guess + np.array([1e-3, 1e-3]), field_nfp=nfp, iota=iota)
                    found.append(RZ)
                    if expected is not None:
                        np.testing.assert_allclose(RZ, expected, atol=1e-7)
                    # the point is periodic
                    n_transits = 1 if iota is None else 7
                    RZ_end = intg.integrate_toroidally(RZ, 2*np.pi*n_transits, phi0=0.0, input_coordinates='cylindrical',
                                                       output_coordinates='cylindrical')
                    np.testing.assert_allclose(RZ_end, RZ, atol=1e-7)
            np.testing.assert_allclose(found[0], found[1], atol=1e-7)
        # with the period of the island chain, a start near the axis converges
        # to the axis, which returns to itself after a single field period
        with self.assertRaises(ObjectiveFailure):
            ScipyFieldlineIntegrator(bs).find_periodic_point(axis_guess, field_nfp=nfp, iota=(3, -7))

    def test_cylindrical_input(self):
        """
        periodic_fieldline accepts a start point in cylindrical coordinates,
        here 1 cm outward from the NCSX axis on the phi=0 plane, and converges
        to the axis.
        """
        _, _, ma, nfp, bs = get_data('ncsx')
        R_axis = ma.x[0] + np.sum(ma.x[1:ma.order+1])  # R at phi=0
        fieldline = ScipyFieldlineIntegrator(bs).periodic_fieldline(
            np.array([R_axis + 0.01, 0.0]), order=ma.order, field_nfp=nfp, phi0=0.0, input_coordinates='cylindrical')
        self.assertTrue(fieldline.res['success'])
        np.testing.assert_allclose(fieldline.curve.gamma()[0], ma.gamma()[0], atol=1e-3)


if __name__ == '__main__':
    unittest.main()
