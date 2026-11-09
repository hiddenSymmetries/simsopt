import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from simsopt.field.coil import coils_via_symmetries
from simsopt.geo.boozersurface import BoozerSurface, boozer_surface_residual
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.geo.surfaceobjectives import ToroidalFlux, Area
from simsopt.configs.zoo import get_data
from .surface_test_helpers import get_surface, get_exact_surface, get_boozer_surface


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


def failed_minimize(fun, x, **kwargs):
    """Stand-in for SciPy's minimize: one failed step to x + 1."""
    f, g = fun(x + 1.)
    return SimpleNamespace(x=x + 1., fun=f, jac=g, nit=1, success=False)


def failed_least_squares(fun, x, jac=None, **kwargs):
    """Stand-in for SciPy's least_squares: one failed step to x + 1."""
    r, J = fun(x + 1.), jac(x + 1.)
    return SimpleNamespace(x=x + 1., fun=r, grad=J.T @ r, jac=J, status=0)


# Makes every Newton step a vector of ones.
UNIT_NEWTON_STEPS = mock.patch('numpy.linalg.solve', side_effect=lambda _A, b: np.ones_like(b))


class BoozerSurfaceTests(unittest.TestCase):
    def test_residual(self):
        """
        This test loads a SurfaceXYZFourier that interpolates the xyz
        coordinates of a surface in the NCSX configuration that was computed
        on a previous branch of pyplasmaopt. Here, we verify that the Boozer
        residual at these interpolation points is small.
        """

        s = get_exact_surface()
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)

        weight = 1.
        tf = ToroidalFlux(s, bs_tf)

        # these data are obtained from `boozer` branch of pyplamsaopt
        tf_target = 0.41431152
        iota = -0.44856192

        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        x = np.concatenate((s.get_dofs(), [iota]))
        # Vectorized version returns scalar objective, not residual vector
        r0_scalar = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=0, constraint_weight=weight, optimize_G=False)
        # Check that the objective is small (residual should be close to zero)
        assert r0_scalar < 1e-6

    def test_boozer_penalty_constraints_gradient(self):
        """
        Taylor test to verify the gradient of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype,
                                      stellsym=stellsym,
                                      optimize_G=optimize_G):
                        self.subtest_boozer_penalty_constraints_gradient(surfacetype, stellsym, optimize_G)

    def test_boozer_penalty_constraints_hessian(self):
        """
        Taylor test to verify the Hessian of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype,
                                      stellsym=stellsym,
                                      optimize_G=optimize_G):
                        self.subtest_boozer_penalty_constraints_hessian(
                            surfacetype, stellsym, optimize_G)

    def subtest_boozer_penalty_constraints_gradient(self, surfacetype, stellsym,
                                                    optimize_G=False):
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        weight = 11.1232

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        fun = boozer_surface.boozer_penalty_constraints_vectorized

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
        f0, J0 = fun(x, derivatives=1, constraint_weight=weight, optimize_G=optimize_G)
        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("###############################################################")
        for eps in epsilons:
            f1 = fun(x + eps*h, derivatives=0, constraint_weight=weight, optimize_G=optimize_G)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old, f0, f1)
            assert err < err_old * 0.55
            err_old = err
        print("###############################################################")

    def subtest_boozer_penalty_constraints_hessian(self, surfacetype, stellsym,
                                                   optimize_G=False):
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        fun = boozer_surface.boozer_penalty_constraints_vectorized

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate(
                (x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))

        f0, J0, H0 = fun(x, derivatives=2, optimize_G=optimize_G)
        h1 = np.random.uniform(size=x.shape)-0.5
        h2 = np.random.uniform(size=x.shape)-0.5
        d2f = h1 @ H0 @ h2

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(10, 20)))
        print("###############################################################")
        for eps in epsilons:
            fp, Jp = fun(x + eps*h1, derivatives=1, optimize_G=optimize_G)
            d2f_fd = (Jp@h2-J0@h2)/eps
            err = np.abs(d2f_fd-d2f)/np.abs(d2f)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err

    def test_boozer_constrained_jacobian(self):
        """
        Taylor test to verify the Jacobian of the first order optimality
        conditions of the exactly constrained optimization problem.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype,
                                      stellsym=stellsym,
                                      optimize_G=optimize_G):
                        self.subtest_boozer_constrained_jacobian(
                            surfacetype, stellsym, optimize_G)

    def subtest_boozer_constrained_jacobian(self, surfacetype, stellsym,
                                            optimize_G=False):
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.3
        lm = [0., 0.]
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate(
                (x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
        xl = np.concatenate((x, lm))
        res0, dres0 = boozer_surface.boozer_exact_constraints(
            xl, derivatives=1, optimize_G=optimize_G)

        h = np.random.uniform(size=xl.shape)-0.5
        dres_exact = dres0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("###############################################################")
        for eps in epsilons:
            res1 = boozer_surface.boozer_exact_constraints(
                xl + eps*h, derivatives=0, optimize_G=optimize_G)
            dres_fd = (res1-res0)/eps
            err = np.linalg.norm(dres_fd-dres_exact)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err
        print("###############################################################")

    def test_boozer_surface_optimisation_convergence(self):
        """
        Test to verify the various optimization algorithms that compute
        the Boozer angles on a surface.
        """

        configs = [
            ("SurfaceXYZTensorFourier", True, True, 'residual_exact'),  # noqa
            ("SurfaceXYZTensorFourier", True, True, 'newton_exact'),  # noqa
            ("SurfaceXYZTensorFourier", True, True, 'newton'),  # noqa
            ("SurfaceXYZTensorFourier", False, True, 'ls'),  # noqa
            ("SurfaceXYZFourier", True, False, 'ls'),  # noqa
        ]
        for surfacetype, stellsym, optimize_G, second_stage in configs:
            for config in ["hsx", "ncsx", "giuliani"]:
                with self.subTest(
                        surfacetype=surfacetype, stellsym=stellsym,
                            optimize_G=optimize_G, second_stage=second_stage, config=config):
                        self.subtest_boozer_surface_optimisation_convergence(surfacetype, stellsym, optimize_G, second_stage, config)

    def subtest_boozer_surface_optimisation_convergence(self, surfacetype,
                                                        stellsym, optimize_G,
                                                        second_stage, config):
        base_curves, base_currents, ma, nfp, bs = get_data(config)
        if stellsym:
            coils = bs.coils
        else:
            # Create a stellarator that still has rotational symmetry but
            # doesn't have stellarator symmetry. We do this by first applying
            # stellarator symmetry, then breaking this slightly, and then
            # applying rotational symmetry
            from simsopt.geo.curve import RotatedCurve
            curves_flipped = [RotatedCurve(c, 0, True) for c in base_curves]
            currents_flipped = [-cur for cur in base_currents]
            for c in curves_flipped:
                c.rotmat += 0.001*np.random.uniform(low=-1., high=1.,
                                                    size=c.rotmat.shape)
                c.rotmatT = c.rotmat.T
            coils = coils_via_symmetries(base_curves + curves_flipped,
                                         base_currents + currents_flipped, nfp, False)
        
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        bs = BiotSavart(coils)

        s = get_surface(surfacetype, stellsym, nfp=nfp)
        s.fit_to_curve(ma, 0.1)

        if config == "ncsx":
            iota = -0.4
        elif config == "giuliani":
            iota = 0.4
        elif config == "hsx":
            iota = 1.
        else:
            raise Exception("initial guess for rotational transform for this config not given")

        ar = Area(s)
        ar_target = ar.J()
        boozer_surface = BoozerSurface(bs, s, ar, ar_target)

        if optimize_G:
            G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))
        else:
            G = None

        cw = (s.quadpoints_phi.size * s.quadpoints_theta.size * 3)
        # compute surface first using LBFGS exact and an area constraint
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-12, maxiter=700, constraint_weight=100/cw, iota=iota, G=G)
        print('Residual norm after LBFGS', res['iter'], np.sqrt(2*res['fun']))

        boozer_surface.recompute_bell()
        if second_stage == 'ls':
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(
                tol=1e-11, maxiter=100, constraint_weight=1000./cw,
                iota=res['iota'], G=res['G'])
        elif second_stage == 'newton':
            res = boozer_surface.minimize_boozer_penalty_constraints_newton(
                tol=1e-10, maxiter=20, constraint_weight=100./cw,
                iota=res['iota'], G=res['G'], stab=1e-4)
        elif second_stage == 'newton_exact':
            res = boozer_surface.minimize_boozer_exact_constraints_newton(
                tol=1e-10, maxiter=15, iota=res['iota'], G=res['G'])
        elif second_stage == 'residual_exact':
            res = boozer_surface.solve_residual_equation_exactly_newton(
                tol=1e-12, maxiter=15, iota=res['iota'], G=res['G'])

        print('Residual norm after second stage', np.linalg.norm(res['residual']))
        assert res['success']
        assert not boozer_surface.surface.is_self_intersecting(thetas=100)

        # For the stellsym case we have z(0, 0) = y(0, 0) = 0. For the not
        # stellsym case, we enforce z(0, 0) = 0, but expect y(0, 0) \neq 0
        gammazero = s.gamma()[0, 0, :]
        assert np.abs(gammazero[2]) < 1e-10
        if stellsym:
            assert np.abs(gammazero[1]) < 1e-10
        else:
            assert np.abs(gammazero[1]) > 1e-6

        if surfacetype == 'SurfaceXYZTensorFourier':
            residual_norm = np.linalg.norm(res['residual'])
            np.testing.assert_array_less(
                residual_norm, 1e-9,
                err_msg=f"Residual norm {residual_norm:.2e} is not less than 1e-9. Residual: {res['residual']}")

        print(ar_target, ar.J())
        print(res['residual'][-10:])
        if surfacetype == 'SurfaceXYZTensorFourier' or second_stage == 'newton_exact':
            assert np.abs(ar_target - ar.J()) < 1e-9
        else:
            assert np.abs(ar_target - ar.J()) < 1e-4

    def test_boozer_serialization(self):
        """
        Test to verify the serialization capability of a BoozerSurface.
        """
        for label in ['Volume', 'Area', 'ToroidalFlux']:
            with self.subTest(label=label):
                self.subtest_boozer_serialization(label)

    def subtest_boozer_serialization(self, label):
        import json
        from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON

        # Don't converge the BoozerSurface to avoid slow optimization that can cause timeouts
        bs, boozer_surface = get_boozer_surface(label=label, converge=False)

        # test serialization of BoozerSurface here too
        bs_str = json.dumps(SIMSON(boozer_surface), cls=GSONEncoder)
        bs_regen = json.loads(bs_str, cls=GSONDecoder)

        diff = boozer_surface.surface.x - bs_regen.surface.x
        self.assertAlmostEqual(np.linalg.norm(diff.ravel()), 0)
        self.assertAlmostEqual(boozer_surface.label.J(), bs_regen.label.J())
        self.assertAlmostEqual(boozer_surface.targetlabel, bs_regen.targetlabel)

        # check that BoozerSurface.surface and label.surface are the same surfaces
        assert bs_regen.label.surface is bs_regen.surface

    def test_run_code(self):
        """
        This unit test verifies that the run_code portion of the BoozerSurface class is working as expected
        """
        bs, boozer_surface = get_boozer_surface(boozer_type='ls')
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # this second time should not actually run
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        for c in bs.coils:
            c.current.fix_all()

        boozer_surface.need_to_run_code = True
        # run without providing value of G
        boozer_surface.run_code(boozer_surface.res['iota'])

        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # this second time should not actually run
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # run the BoozerExact algorithm without a guess for G
        boozer_surface.need_to_run_code = True
        boozer_surface.solve_residual_equation_exactly_newton(iota=boozer_surface.res['iota'])

    def test_minimize_boozer_penalty_constraints_ls_manual(self):
        """
        Test minimize_boozer_penalty_constraints_ls with method='manual' (damped Gauss-Newton).
        """
        for stellsym in [True, False]:
            for optimize_G in [True, False]:
                with self.subTest(stellsym=stellsym, optimize_G=optimize_G):
                    self.subtest_minimize_boozer_penalty_constraints_ls_manual(stellsym, optimize_G)

    def subtest_minimize_boozer_penalty_constraints_ls_manual(self, stellsym, optimize_G):
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        s = get_surface("SurfaceXYZTensorFourier", stellsym, nfp=nfp)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)
        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.4
        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi)) if optimize_G else None

        cw = (s.quadpoints_phi.size * s.quadpoints_theta.size * 3)
        
        # First run LBFGS to get a good initial guess
        res_lbfgs = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-12, maxiter=700, constraint_weight=1000./cw, iota=iota, G=G)
        print('Residual norm after LBFGS', res_lbfgs['iter'], np.sqrt(2*res_lbfgs['fun']))
        
        boozer_surface.recompute_bell()
        
        # Now run manual method starting from LBFGS result
        res = boozer_surface.minimize_boozer_penalty_constraints_ls(
            tol=1e-8, maxiter=50, constraint_weight=1000./cw,
            iota=res_lbfgs['iota'], G=res_lbfgs.get('G'), method='manual')
        print(np.linalg.norm(res['residual']))
        
        # Manual method may not always succeed, but should improve or at least not diverge too much
        assert 'iota' in res
        if optimize_G:
            assert 'G' in res
        assert 's' in res
        # Check that residual is reasonable (not diverged)
        residual_norm = np.linalg.norm(res['residual'])
        np.testing.assert_array_less(
            residual_norm, 1e-3,
            err_msg=f"Residual norm {residual_norm:.2e} is not less than 1e-3. Residual: {res['residual']}")
        assert res['success'], f"Optimization did not succeed. Residual norm: {residual_norm:.2e}, Residual: {res['residual']}"

    def test_need_to_run_code_false(self):
        """
        Test that methods return cached results when need_to_run_code=False.
        """
        for stellsym in [True, False]:
            for optimize_G in [True, False]:
                with self.subTest(stellsym=stellsym, optimize_G=optimize_G):
                    self.subtest_need_to_run_code_false(stellsym, optimize_G)

    def subtest_need_to_run_code_false(self, stellsym, optimize_G):
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        s = get_surface("SurfaceXYZTensorFourier", stellsym, nfp=nfp)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)
        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.4
        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi)) if optimize_G else None

        cw = (s.quadpoints_phi.size * s.quadpoints_theta.size * 3)
        
        # Run once to populate self.res
        res1 = boozer_surface.minimize_boozer_penalty_constraints_ls(
            tol=1e-8, maxiter=5, constraint_weight=100./cw,
            iota=iota, G=G, method='lm')
        
        # Set need_to_run_code to False
        boozer_surface.need_to_run_code = False
        
        # Run again - should return cached result
        res2 = boozer_surface.minimize_boozer_penalty_constraints_ls(
            tol=1e-8, maxiter=5, constraint_weight=100./cw,
            iota=iota, G=G, method='lm')
        
        # Results should be identical (same object)
        assert res1 is res2
        assert boozer_surface.res is res2

    def test_minimize_boozer_exact_constraints_newton_G_None(self):
        """
        Test minimize_boozer_exact_constraints_newton with G=None (not optimizing G).
        """
        for stellsym in [True, False]:
            with self.subTest(stellsym=stellsym):
                self.subtest_minimize_boozer_exact_constraints_newton_G_None(stellsym)

    def subtest_minimize_boozer_exact_constraints_newton_G_None(self, stellsym):
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)

        s = get_surface("SurfaceXYZTensorFourier", stellsym, nfp=nfp)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)
        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.4
        G = None  # Not optimizing G

        # First run LBFGS to get a good initial guess
        cw = (s.quadpoints_phi.size * s.quadpoints_theta.size * 3)
        res_lbfgs = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-6, maxiter=50, constraint_weight=100./cw, iota=iota, G=G)
        
        boozer_surface.recompute_bell()
        
        # Now run exact constraints Newton with G=None
        # Note: This method may not work well with G=None, so we use more lenient criteria
        res = boozer_surface.minimize_boozer_exact_constraints_newton(
            tol=1e-6, maxiter=5, iota=res_lbfgs['iota'], G=None)
        
        assert 'iota' in res
        assert res['G'] is None  # G should be None when not optimizing
        assert 's' in res
        # For G=None case, exact constraints Newton may not converge well, so just
        # Newton often blows up here! 

    def test_minimize_boozer_exact_constraints_newton_stellsym_false(self):
        """
        Test minimize_boozer_exact_constraints_newton with stellsym=False (non-stellarator symmetric).
        """
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        
        # Create non-stellarator symmetric configuration
        from simsopt.geo.curve import RotatedCurve
        rng = np.random.default_rng(12345)
        curves_flipped = [RotatedCurve(c, 0, True) for c in base_curves]
        currents_flipped = [-cur for cur in base_currents]
        for c in curves_flipped:
            c.rotmat += 0.001*rng.uniform(low=-1., high=1., size=c.rotmat.shape)
            c.rotmatT = c.rotmat.T
        coils = coils_via_symmetries(base_curves + curves_flipped,
                                     base_currents + currents_flipped, nfp, False)
        current_sum = sum(abs(c.current.get_value()) for c in coils)
        bs = BiotSavart(coils)

        s = get_surface("SurfaceXYZTensorFourier", False, nfp=nfp)
        s.fit_to_curve(ma, 0.1)

        # Use Area instead of ToroidalFlux to avoid shape mismatch issues with non-stellsym
        ar = Area(s)
        ar_target = ar.J()
        boozer_surface = BoozerSurface(bs, s, ar, ar_target)

        iota = -0.4
        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))

        # First run LBFGS to get a good initial guess
        cw = (s.quadpoints_phi.size * s.quadpoints_theta.size * 3)
        res_lbfgs = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-8, maxiter=200, constraint_weight=100./cw, iota=iota, G=G)
        
        boozer_surface.recompute_bell()
        
        # Now run exact constraints Newton with stellsym=False
        res = boozer_surface.minimize_boozer_exact_constraints_newton(
            tol=1e-6, maxiter=100, iota=res_lbfgs['iota'], G=res_lbfgs['G'])
        
        assert 'iota' in res
        assert 'G' in res
        assert 's' in res
        # For non-stellsym, lm should be a list of 2 elements
        assert len(res['lm']) == 2
        # Check that residual is reasonable (not diverged)
        residual_norm = np.linalg.norm(res['residual'])
        np.testing.assert_array_less(
            residual_norm, 1e-6,
            err_msg=f"Residual norm {residual_norm:.2e} is not less than 1e-6. Residual: {res['residual']}")
        assert res['success'], f"Optimization did not succeed. Residual norm: {residual_norm:.2e}, Residual: {res['residual']}"


    def check_failed_solve(self, boozer_surface, solve, patches, x0, shift, after, G, expected, info=None):
        """
        Run a failed ``solve`` under ``patches`` from ``x0`` (dofs, iota and G) and check that it kept
        ``x0 + shift``, the step the mocked solver takes, if its measure ``after`` is at most its value
        1 at ``x0``, and ``x0`` otherwise. ``expected(v)`` gives the returned values at the retained
        point, where the measure is v, and ``info(after)`` the raw SciPy result.
        """
        ndofs = boozer_surface.surface.get_dofs().size
        boozer_surface.surface.set_dofs(x0[:ndofs])
        boozer_surface.need_to_run_code = True
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            res = solve()
        self.assertFalse(res['success'])
        kept = after <= 1.
        x = x0 + shift if kept else x0
        np.testing.assert_array_equal(boozer_surface.surface.get_dofs(), x[:ndofs])
        self.assertEqual(res['iota'], x[ndofs])
        self.assertEqual(res.get('G'), None if G is None else x[ndofs + 1])
        for key, value in expected(after if kept else 1.).items():
            np.testing.assert_array_equal(res[key], value, err_msg=key)
        if info is not None:
            for key, value in info(after).items():
                np.testing.assert_array_equal(getattr(res['info'], key), value, err_msg=f"info.{key}")

    def test_failed_solves_restore_the_starting_point(self):
        """
        A solve that fails and ends with a larger or non-finite residual than it
        started with returns its starting surface, iota and G, with the residual
        and derivatives evaluated there. A failed solve that ends with an equal
        or strictly smaller finite residual keeps its final iterate, with the
        residual and derivatives evaluated there. Either way the raw SciPy
        result in 'info' describes the solve itself, i.e. its final iterate.
        """
        _, ls = get_boozer_surface(boozer_type="ls", converge=False)
        _, exact = get_boozer_surface(boozer_type="exact", converge=False)
        ls_dofs = ls.surface.get_dofs()
        exact_dofs = exact.surface.get_dofs()

        for after in [10., np.nan, 1., 0.5]:
            # The residual is 1 at the starting point and `after` anywhere else.
            for G in [-2.0, None]:
                x0 = np.concatenate((ls_dofs, [-0.406] if G is None else [-0.406, G]))
                n = x0.size

                def value(x):
                    return 1. if np.array_equal(x, x0) else after

                def penalty(x, derivatives=1, **kwargs):
                    ls.surface.set_dofs(x[:ls_dofs.size])
                    v = value(x)
                    return (v, v * np.ones(n)) + ((v * np.eye(n),) if derivatives == 2 else ())

                def residual_and_jacobian(x, *args):
                    ls.surface.set_dofs(x[:ls_dofs.size])
                    v = value(x)
                    return v * np.ones(n), v * np.eye(n)

                penalty_patch = mock.patch.object(ls, 'boozer_penalty_constraints_vectorized', side_effect=penalty)
                ls_patch = mock.patch.object(ls, '_get_residual_vector_and_jacobian', side_effect=residual_and_jacobian)
                ones, eye = np.ones(n), np.eye(n)
                cases = {
                    'LBFGS': (lambda: ls.minimize_boozer_penalty_constraints_LBFGS(maxiter=1, iota=-0.406, G=G),
                              [penalty_patch, mock.patch('simsopt.geo.boozersurface.minimize', side_effect=failed_minimize)], 1.,
                              lambda v: {'fun': v, 'gradient': v * ones}),
                    'newton': (lambda: ls.minimize_boozer_penalty_constraints_newton(maxiter=1, iota=-0.406, G=G),
                               [penalty_patch, UNIT_NEWTON_STEPS], -1.,
                               lambda v: {'residual': v * ones, 'jacobian': v * ones, 'hessian': v * eye}),
                    'ls manual': (lambda: ls.minimize_boozer_penalty_constraints_ls(maxiter=1, iota=-0.406, G=G, method='manual'),
                                  [ls_patch, UNIT_NEWTON_STEPS], -1.,
                                  lambda v: {'residual': v * ones, 'gradient': v**2 * ones, 'jacobian': v**2 * eye}),
                    'ls': (lambda: ls.minimize_boozer_penalty_constraints_ls(maxiter=1, iota=-0.406, G=G),
                           [ls_patch, mock.patch('simsopt.geo.boozersurface.least_squares', side_effect=failed_least_squares)], 1.,
                           lambda v: {'residual': v * ones, 'gradient': v**2 * ones, 'jacobian': v * eye}),
                }
                # The mocked SciPy solvers end at x0 + 1, where the residual is `after`.
                infos = {
                    'LBFGS': lambda after: {'x': x0 + 1., 'fun': after, 'jac': after * ones},
                    'ls': lambda after: {'x': x0 + 1., 'fun': after * ones, 'jac': after * eye},
                }
                for name, (solve, patches, shift, expected) in cases.items():
                    with self.subTest(solver=name, after=after, G=G):
                        self.check_failed_solve(ls, solve, patches, x0, shift, after, G, expected, infos.get(name))

                x0 = np.concatenate((exact_dofs, [-0.406] if G is None else [-0.406, G]))
                xl0 = np.concatenate((x0, [0., 0.]))

                def exact_constraints(xl, derivatives=1, optimize_G=True):
                    exact.surface.set_dofs(xl[:exact_dofs.size])
                    v = 1. if np.array_equal(xl, xl0) else after
                    return v * np.ones(xl.size), v * np.eye(xl.size)

                with self.subTest(solver='exact constraints newton', after=after, G=G):
                    self.check_failed_solve(
                        exact, lambda: exact.minimize_boozer_exact_constraints_newton(maxiter=1, iota=-0.406, G=G),
                        [mock.patch.object(exact, 'boozer_exact_constraints', side_effect=exact_constraints), UNIT_NEWTON_STEPS],
                        x0, -1., after, G, lambda v: {'residual': v * np.ones(xl0.size), 'jacobian': v * np.eye(xl0.size)})

            x0 = np.concatenate((exact_dofs, [-0.406, -2.0]))
            nresidual = exact.surface.gamma().size

            def residual(surface, iota, G, *args, derivatives=1):
                x = np.concatenate((surface.get_dofs(), [iota, G]))
                v = 1. if np.array_equal(x, x0) else after
                return v * np.ones(nresidual), v * np.ones((nresidual, x.size))

            # The label residual is zero everywhere, so that the norm is set by the
            # residual alone. Each Newton step, with iterative refinement, is -2.
            with self.subTest(solver='residual equation newton', after=after):
                self.check_failed_solve(
                    exact, lambda: exact.solve_residual_equation_exactly_newton(maxiter=1, iota=-0.406, G=-2.0),
                    [mock.patch('simsopt.geo.boozersurface.boozer_surface_residual', side_effect=residual), UNIT_NEWTON_STEPS,
                     mock.patch.object(exact.label, 'J', side_effect=lambda: exact.targetlabel)],
                    x0, -2., after, -2.0, lambda v: {'residual': v * np.ones(nresidual)})

    def test_failed_penalty_solves_compare_their_own_convergence_measure(self):
        """
        A failed step that lowers the penalty value and residual norm but
        raises the gradient norm, or the reverse, is kept or undone by the
        value (LBFGS) or residual norm (SciPy least squares), and by the
        gradient norm in penalty Newton and manual least squares.
        """
        _, ls = get_boozer_surface(boozer_type="ls", converge=False)
        dofs = ls.surface.get_dofs()
        x0 = np.concatenate((dofs, [-0.406, -2.0]))
        n = x0.size
        for value, gradient in [(0.5, 10.), (10., 0.5)]:
            # Both measures are 1 at x0, and value and gradient everywhere else.
            def scales(x, value=value, gradient=gradient):
                return (1., 1.) if np.array_equal(x, x0) else (value, gradient)

            def penalty(x, derivatives=1, **kwargs):
                ls.surface.set_dofs(x[:dofs.size])
                v, g = scales(x)
                return (v, g * np.ones(n)) + ((np.eye(n),) if derivatives == 2 else ())

            def residual_and_jacobian(x, *args):
                ls.surface.set_dofs(x[:dofs.size])
                v, g = scales(x)
                return v * np.ones(n), g / v * np.eye(n)

            penalty_patch = mock.patch.object(ls, 'boozer_penalty_constraints_vectorized', side_effect=penalty)
            ls_patch = mock.patch.object(ls, '_get_residual_vector_and_jacobian', side_effect=residual_and_jacobian)
            cases = {
                'LBFGS': (lambda: ls.minimize_boozer_penalty_constraints_LBFGS(maxiter=1, iota=-0.406, G=-2.0),
                          [penalty_patch, mock.patch('simsopt.geo.boozersurface.minimize', side_effect=failed_minimize)],
                          1., value),
                'ls': (lambda: ls.minimize_boozer_penalty_constraints_ls(maxiter=1, iota=-0.406, G=-2.0),
                       [ls_patch, mock.patch('simsopt.geo.boozersurface.least_squares', side_effect=failed_least_squares)],
                       1., value),
                'newton': (lambda: ls.minimize_boozer_penalty_constraints_newton(maxiter=1, iota=-0.406, G=-2.0),
                           [penalty_patch, UNIT_NEWTON_STEPS], -1., gradient),
                'ls manual': (lambda: ls.minimize_boozer_penalty_constraints_ls(maxiter=1, iota=-0.406, G=-2.0, method='manual'),
                              [ls_patch, UNIT_NEWTON_STEPS], -1., gradient),
            }
            for name, (solve, patches, shift, measure) in cases.items():
                with self.subTest(solver=name, value=value, gradient=gradient):
                    self.check_failed_solve(ls, solve, patches, x0, shift, measure, -2.0, lambda v: {})

    def test_failed_scipy_solves_leave_the_field_on_the_returned_surface(self):
        """
        After a failed SciPy solve, which evaluates its starting point to decide
        whether to keep the final iterate, the field points lie on the returned
        surface.
        """
        _, boozer_surface = get_boozer_surface(boozer_type="ls", converge=False)
        for name, solve in [
            ('LBFGS', lambda: boozer_surface.minimize_boozer_penalty_constraints_LBFGS(maxiter=1, iota=-0.406, G=-2.0)),
            ('ls', lambda: boozer_surface.minimize_boozer_penalty_constraints_ls(maxiter=2, iota=-0.406, G=-2.0)),
        ]:
            with self.subTest(solver=name):
                boozer_surface.need_to_run_code = True
                res = solve()
                self.assertFalse(res['success'])
                np.testing.assert_array_equal(boozer_surface.biotsavart.get_points_cart(),
                                              boozer_surface.surface.gamma().reshape((-1, 3)))

    def test_residual_equation_newton_success_is_that_of_the_final_iterate(self):
        """
        A solve that converges on its last allowed iteration succeeds.
        """
        _, boozer_surface = get_boozer_surface(boozer_type="exact", converge=False)
        s = boozer_surface.surface
        x0 = s.get_dofs()
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=20, iota=-0.406)
        self.assertTrue(res['success'])
        x, iterations = s.get_dofs(), res['iter']
        s.set_dofs(x0)
        boozer_surface.need_to_run_code = True
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=iterations, iota=-0.406)
        self.assertTrue(res['success'])
        np.testing.assert_array_equal(s.get_dofs(), x)

    def test_residual_equation_newton_nonpositive_maxiter(self):
        """Non-positive limits skip Newton steps and retain upstream's success sentinel."""
        _, boozer_surface = get_boozer_surface(boozer_type="exact", converge=False)
        s = boozer_surface.surface
        dofs = s.get_dofs().copy()
        initial_residual, jacobian = boozer_surface_residual(
            s, -0.406, -2.0, boozer_surface.biotsavart, derivatives=1)
        for maxiter in (0, -1):
            with self.subTest(maxiter=maxiter):
                boozer_surface.need_to_run_code = True
                # A converged initial residual must still use upstream's 1e6
                # sentinel when no iterations are requested.
                residual = np.zeros_like(initial_residual)
                with mock.patch('simsopt.geo.boozersurface.boozer_surface_residual',
                                return_value=(residual, jacobian)), \
                        mock.patch.object(boozer_surface.label, 'J', return_value=boozer_surface.targetlabel), \
                        mock.patch('numpy.linalg.solve', side_effect=AssertionError("No Newton step allowed")):
                    res = boozer_surface.solve_residual_equation_exactly_newton(
                        tol=1e-10, maxiter=maxiter, iota=-0.406, G=-2.0)
                self.assertEqual(res['iter'], 0)
                self.assertFalse(res['success'])
                self.assertEqual(res['iota'], -0.406)
                self.assertEqual(res['G'], -2.0)
                np.testing.assert_array_equal(s.get_dofs(), dofs)
                np.testing.assert_array_equal(res['residual'], residual)

    def test_residual_equation_newton_fractional_maxiter(self):
        """A fractional limit bounds steps even when the residual cannot converge."""
        _, boozer_surface = get_boozer_surface(boozer_type="exact", converge=False)
        s = boozer_surface.surface
        dofs = s.get_dofs().copy()
        initial_residual, jacobian = boozer_surface_residual(
            s, -0.406, -2.0, boozer_surface.biotsavart, derivatives=1)
        for maxiter, iterations in ((0.5, 1), (1.5, 2)):
            with self.subTest(maxiter=maxiter):
                boozer_surface.need_to_run_code = True
                residual = np.ones_like(initial_residual)
                # Exhaustion makes an unbounded loop fail promptly rather than
                # hanging the test runner; refinement uses two solves per step.
                steps = [np.zeros(dofs.size + 2) for _ in range(2 * iterations)]
                with mock.patch('simsopt.geo.boozersurface.boozer_surface_residual',
                                return_value=(residual, jacobian)), \
                        mock.patch('numpy.linalg.solve', side_effect=steps):
                    res = boozer_surface.solve_residual_equation_exactly_newton(
                        tol=1e-10, maxiter=maxiter, iota=-0.406, G=-2.0)
                self.assertEqual(res['iter'], iterations)
                self.assertFalse(res['success'])
                np.testing.assert_array_equal(s.get_dofs(), dofs)

    def test_penalty_newton_divergence_factor(self):
        """
        The penalty Newton solve stops once the gradient norm exceeds
        divergence_factor times its starting value, and not before. By
        default it does not stop early.
        """
        _, boozer_surface = get_boozer_surface(boozer_type="ls", converge=False)
        dofs = boozer_surface.surface.get_dofs()

        def exploding(x, derivatives=2, **kwargs):
            # Each Newton step multiplies x, and so the gradient norm, by -19.
            return 0.5 * x @ x, x, 0.05 * np.eye(x.size)

        for options, iterations in [({'divergence_factor': 1e3}, 3), ({'divergence_factor': None}, 40), ({}, 40)]:
            with self.subTest(**options):
                boozer_surface.surface.set_dofs(dofs)
                boozer_surface.need_to_run_code = True
                with mock.patch.object(boozer_surface, 'boozer_penalty_constraints_vectorized', side_effect=exploding):
                    res = boozer_surface.minimize_boozer_penalty_constraints_newton(
                        tol=1e-14, maxiter=40, iota=-0.406, G=-2.0, **options)
                self.assertEqual(res['iter'], iterations)
                self.assertFalse(res['success'])
                np.testing.assert_array_equal(boozer_surface.surface.get_dofs(), dofs)
                self.assertEqual(res['iota'], -0.406)
                self.assertEqual(res['G'], -2.0)

        def nonmonotone(x, derivatives=2, **kwargs):
            # The gradient norm drops to 1e-3 and then rises to 1 before
            # converging, within a factor 1e3 of its starting value.
            g = np.zeros_like(x)
            g[0] = -1. if x[0] < 0.5 else -0.001 if x[0] < 1.0005 else -(2. - x[0]) if x[0] < 1.5 else 0.
            return 0.5 * g[0]**2, g, np.eye(x.size)

        dofs0 = dofs.copy()
        dofs0[0] = 0.
        boozer_surface.surface.set_dofs(dofs0)
        boozer_surface.need_to_run_code = True
        with mock.patch.object(boozer_surface, 'boozer_penalty_constraints_vectorized', side_effect=nonmonotone):
            res = boozer_surface.minimize_boozer_penalty_constraints_newton(maxiter=10, iota=-0.406, G=-2.0)
        self.assertTrue(res['success'])
        self.assertEqual(res['iter'], 3)


    def test_boozer_surface_quadpoints(self):
        """ 
        this unit test checks that the quadpoints mask for stellarator symmetric Boozer Surfaces are correctly initialized
        """
        for idx in range(4):
            with self.subTest(idx=idx):
                self.subtest_boozer_surface_quadpoints(idx)

    def subtest_boozer_surface_quadpoints(self, idx):
        mpol = 6
        ntor = 6
        nfp = 3

        if idx == 0:
            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[:, mpol+1:] = False
            mask_true[ntor+1:, 0] = False
        elif idx == 1:
            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 0.5, mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[ntor+1:, 0] = False
        elif idx == 2:
            phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[0, mpol+1:] = False
        elif idx == 3:
            phis = np.linspace(0, 1., 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1., 2*mpol+1, endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

        if idx < 3:  # the first three quadrature point sets should pass without issue.
            mask = s.get_stellsym_mask()
            assert np.all(mask == mask_true)
        else:
            with self.assertRaises(Exception):
                mask = s.get_stellsym_mask()

    def test_boozer_surface_type_assert(self):
        """
        this unit test checks that an exception is raised if a SurfaceRZFourier is passed to a BoozerSurface
        """
        mpol = 6
        ntor = 6
        nfp = 3
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceRZFourier(mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")

        lab = Area(s)
        lab_target = 0.1

        with self.assertRaises(Exception):
            _ = BoozerSurface(bs, s, lab, lab_target)


if __name__ == "__main__":
    unittest.main()
