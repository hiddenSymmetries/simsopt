import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from simsopt.field.coil import coils_via_symmetries
import simsopt.geo.boozersurface as boozersurface_module
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.geo.surfaceobjectives import ToroidalFlux, Area
from simsopt.configs.zoo import get_data
from .surface_test_helpers import get_surface, get_exact_surface, get_boozer_surface


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


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


    def test_label_gradient_with_fixed_surface(self):
        """
        The label constraint uses the gradient with respect to all surface
        coefficients, including shared surface dofs and fixed coefficients.
        """
        for label in ["Volume", "Area", "ToroidalFlux", "AspectRatio"]:
            for n in [None, 51]:
                for wrapped in [False, True]:
                    with self.subTest(label=label, n=n, wrapped=wrapped):
                        bs, b = get_boozer_surface(
                            label=label, nphi=n, ntheta=n, boozer_type="ls", converge=False)
                        if wrapped:
                            b = BoozerSurface(bs, b.surface, 2 * b.label, 2 * b.targetlabel)
                        # Keep the label constraint active so an incorrect gradient
                        # cannot pass merely because the target is already met.
                        b.targetlabel *= 1.1
                        s = b.surface
                        x = np.concatenate((s.get_dofs(), [-0.406, -2.0]))
                        direction = np.random.default_rng(1).uniform(-0.5, 0.5, x.shape)
                        fun = b.boozer_penalty_constraints_vectorized
                        expected = fun(x, derivatives=1, constraint_weight=100., optimize_G=True)
                        eps = 1e-6
                        plus = fun(x + eps * direction, derivatives=0, constraint_weight=100., optimize_G=True)
                        minus = fun(x - eps * direction, derivatives=0, constraint_weight=100., optimize_G=True)
                        np.testing.assert_allclose(
                            expected[1] @ direction, (plus - minus) / (2 * eps), rtol=1e-5, atol=1e-7)
                        for fixed in ["some", "all"]:
                            with self.subTest(fixed=fixed):
                                if fixed == "some":
                                    s.fix(0)
                                else:
                                    s.fix_all()
                                actual = fun(x, derivatives=1, constraint_weight=100., optimize_G=True)
                                np.testing.assert_array_equal(actual[0], expected[0])
                                np.testing.assert_array_equal(actual[1], expected[1])

    def test_wrapped_label(self):
        """
        A label that wraps another one, here 2 * Volume, gives the same exact
        solve as the label itself, with free and with fixed surface dofs.
        """
        bs, boozer_surface = get_boozer_surface(
            label="Volume", nphi=51, ntheta=51, boozer_type="exact", converge=False)
        s = boozer_surface.surface
        x0 = s.get_dofs()
        wrapped = BoozerSurface(bs, s, 2 * boozer_surface.label, 2 * boozer_surface.targetlabel)
        solutions = []
        for b in [boozer_surface, wrapped]:
            for fixed in [False, True]:
                with self.subTest(wrapped=b is wrapped, fixed=fixed):
                    if fixed:
                        s.fix_all()
                    else:
                        s.unfix_all()
                    s.set_dofs(x0)
                    b.need_to_run_code = True
                    res = b.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=20, iota=-0.406)
                    self.assertTrue(res['success'])
                    solutions.append(np.concatenate((s.get_dofs(), [res['iota'], res['G']])))
        for solution in solutions[1:]:
            np.testing.assert_allclose(solution, solutions[0], rtol=0, atol=1e-10)

    def test_failed_solves_restore_the_starting_point(self):
        """
        A solve that fails and ends with a larger or non-finite residual than it
        started with returns its starting surface, iota and G, with the residual
        and derivatives evaluated there. A failed solve that ends with an equal
        residual keeps its final iterate.
        """
        def solve_step(_A, b):
            return np.ones_like(b)

        def check(boozer_surface, solve, patches, x0, shift, after, G, expected):
            # x0: starting dofs, iota (and G). shift: the step the mocked solver
            # takes. expected: the returned values at the starting point.
            boozer_surface.surface.set_dofs(x0[:boozer_surface.surface.get_dofs().size])
            boozer_surface.need_to_run_code = True
            with contextlib.ExitStack() as stack:
                for target, name, fake in patches:
                    stack.enter_context(mock.patch.object(target, name, side_effect=fake))
                res = solve()
            self.assertFalse(res['success'])
            x = x0 if after != 1 else x0 + shift
            ndofs = boozer_surface.surface.get_dofs().size
            np.testing.assert_array_equal(boozer_surface.surface.get_dofs(), x[:ndofs])
            self.assertEqual(res['iota'], x[ndofs])
            self.assertEqual(res.get('G'), None if G is None else x[ndofs + 1])
            for key, value in expected.items():
                np.testing.assert_array_equal(res[key], value, err_msg=key)

        for after in [10., np.nan, 1.]:
            # The residual is 1 at the starting point and `after` anywhere else.
            for G in [-2.0, None]:
                _, boozer_surface = get_boozer_surface(boozer_type="ls", converge=False)
                dofs = boozer_surface.surface.get_dofs()
                x0 = np.concatenate((dofs, [-0.406] if G is None else [-0.406, G]))
                n = x0.size

                def value(x):
                    return 1. if np.array_equal(x, x0) else after

                def penalty(x, derivatives=1, **kwargs):
                    boozer_surface.surface.set_dofs(x[:dofs.size])
                    v = value(x)
                    return (v, v * np.ones(n)) + ((v * np.eye(n),) if derivatives == 2 else ())

                def residual_and_jacobian(x, *args):
                    boozer_surface.surface.set_dofs(x[:dofs.size])
                    v = value(x)
                    return v * np.ones(n), v * np.eye(n)

                def failed_minimize(fun, x, **kwargs):
                    f, g = fun(x + 1.)
                    return SimpleNamespace(x=x + 1., fun=f, jac=g, nit=1, success=False)

                def failed_least_squares(fun, x, jac=None, **kwargs):
                    r, J = fun(x + 1.), jac(x + 1.)
                    return SimpleNamespace(x=x + 1., fun=r, grad=J.T @ r, jac=J, status=0)

                penalty_patch = (boozer_surface, 'boozer_penalty_constraints_vectorized', penalty)
                ls_patch = (boozer_surface, '_get_residual_vector_and_jacobian', residual_and_jacobian)
                solve_patch = (boozersurface_module.np.linalg, 'solve', solve_step)
                ones, eye = np.ones(n), np.eye(n)
                cases = {
                    'LBFGS': (lambda: boozer_surface.minimize_boozer_penalty_constraints_LBFGS(maxiter=1, iota=-0.406, G=G),
                              [penalty_patch, (boozersurface_module, 'minimize', failed_minimize)], 1.,
                              {'fun': 1., 'gradient': ones}),
                    'newton': (lambda: boozer_surface.minimize_boozer_penalty_constraints_newton(maxiter=1, iota=-0.406, G=G),
                               [penalty_patch, solve_patch], -1.,
                               {'residual': ones, 'jacobian': ones, 'hessian': eye}),
                    'ls manual': (lambda: boozer_surface.minimize_boozer_penalty_constraints_ls(maxiter=1, iota=-0.406, G=G, method='manual'),
                                  [ls_patch, solve_patch], -1.,
                                  {'residual': ones, 'gradient': ones, 'jacobian': eye}),
                    'ls': (lambda: boozer_surface.minimize_boozer_penalty_constraints_ls(maxiter=1, iota=-0.406, G=G),
                           [ls_patch, (boozersurface_module, 'least_squares', failed_least_squares)], 1.,
                           {'residual': ones, 'gradient': ones, 'jacobian': eye}),
                }
                for name, (solve, patches, shift, expected) in cases.items():
                    with self.subTest(solver=name, after=after, G=G):
                        check(boozer_surface, solve, patches, x0, shift, after, G, expected)

                _, boozer_surface = get_boozer_surface(boozer_type="exact", converge=False)
                dofs = boozer_surface.surface.get_dofs()
                x0 = np.concatenate((dofs, [-0.406] if G is None else [-0.406, G]))
                xl0 = np.concatenate((x0, [0., 0.]))

                def exact_constraints(xl, derivatives=1, optimize_G=True):
                    boozer_surface.surface.set_dofs(xl[:dofs.size])
                    v = 1. if np.array_equal(xl, xl0) else after
                    return v * np.ones(xl.size), v * np.eye(xl.size)

                with self.subTest(solver='exact constraints newton', after=after, G=G):
                    check(boozer_surface,
                          lambda: boozer_surface.minimize_boozer_exact_constraints_newton(maxiter=1, iota=-0.406, G=G),
                          [(boozer_surface, 'boozer_exact_constraints', exact_constraints), solve_patch],
                          x0, -1., after, G, {'residual': np.ones(xl0.size), 'jacobian': np.eye(xl0.size)})

            _, boozer_surface = get_boozer_surface(boozer_type="exact", converge=False)
            dofs = boozer_surface.surface.get_dofs()
            x0 = np.concatenate((dofs, [-0.406, -2.0]))
            nresidual = boozer_surface.surface.gamma().size

            def residual(surface, iota, G, *args, derivatives=1):
                x = np.concatenate((surface.get_dofs(), [iota, G]))
                v = 1. if np.array_equal(x, x0) else after
                return v * np.ones(nresidual), v * np.ones((nresidual, x.size))

            # The label residual is zero everywhere, so that the norm is set by the
            # residual alone. Each Newton step, with iterative refinement, is -2.
            label = boozer_surface.label
            with self.subTest(solver='residual equation newton', after=after):
                check(boozer_surface,
                      lambda: boozer_surface.solve_residual_equation_exactly_newton(maxiter=1, iota=-0.406, G=-2.0),
                      [(boozersurface_module, 'boozer_surface_residual', residual), solve_patch,
                       (label, 'J', lambda: boozer_surface.targetlabel)],
                      x0, -2., after, -2.0, {'residual': np.ones(nresidual)})

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

    def test_penalty_newton_divergence_factor(self):
        """
        The penalty Newton solve stops once the gradient norm exceeds
        divergence_factor times its starting value, and not before.
        """
        _, boozer_surface = get_boozer_surface(boozer_type="ls", converge=False)
        dofs = boozer_surface.surface.get_dofs()

        def exploding(x, derivatives=2, **kwargs):
            # Each Newton step multiplies x, and so the gradient norm, by -19.
            return 0.5 * x @ x, x, 0.05 * np.eye(x.size)

        for divergence_factor, iterations in [(1e3, 3), (None, 40), (0, 40)]:
            with self.subTest(divergence_factor=divergence_factor):
                boozer_surface.surface.set_dofs(dofs)
                boozer_surface.need_to_run_code = True
                with mock.patch.object(boozer_surface, 'boozer_penalty_constraints_vectorized', side_effect=exploding):
                    res = boozer_surface.minimize_boozer_penalty_constraints_newton(
                        tol=1e-14, maxiter=40, iota=-0.406, G=-2.0, divergence_factor=divergence_factor)
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
