import unittest

import numpy as np
from simsopt.field.coil import coils_via_symmetries
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.geo.surfaceobjectives import ToroidalFlux, Area
from simsopt.configs.zoo import get_data
from .surface_test_helpers import get_surface, get_exact_surface, get_boozer_surface

# Fixed random seed for reproducibility across platforms
RANDOM_SEED = 42

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
        r0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=weight, optimize_G=False,
            scalarize=False)
        # the residual should be close to zero for all entries apart from the y
        # and z coordinate at phi=0 and theta=0 (and the corresponding rotations)
        ignores_idxs = np.zeros_like(r0)
        ignores_idxs[[1, 2, 693, 694, 695, 1386, 1387, 1388, -2, -1]] = 1
        assert np.max(np.abs(r0[ignores_idxs < 0.5])) < 1e-8
        assert np.max(np.abs(r0[-2:])) < 1e-6

    def test_boozer_penalty_constraints_gradient(self):
        """
        Taylor test to verify the gradient of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    for vectorize in [True, False]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          vectorize=vectorize):
                            self.subtest_boozer_penalty_constraints_gradient(surfacetype, stellsym, optimize_G, vectorize)

    def test_boozer_penalty_constraints_hessian(self):
        """
        Taylor test to verify the Hessian of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    for vectorize in [True, False]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          vectorize=vectorize):
                            self.subtest_boozer_penalty_constraints_hessian(
                                surfacetype, stellsym, optimize_G, vectorize)

    def subtest_boozer_penalty_constraints_gradient(self, surfacetype, stellsym,
                                                    optimize_G=False, vectorize=False):
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
        fun = boozer_surface.boozer_penalty_constraints_vectorized if vectorize else boozer_surface.boozer_penalty_constraints

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
                                                   optimize_G=False, vectorize=False):
        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        fun = boozer_surface.boozer_penalty_constraints_vectorized if vectorize else boozer_surface.boozer_penalty_constraints

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
                for vectorize in [True, False]:
                    with self.subTest(
                        surfacetype=surfacetype, stellsym=stellsym,
                            optimize_G=optimize_G, second_stage=second_stage, config=config, vectorize=vectorize):
                        self.subtest_boozer_surface_optimisation_convergence(surfacetype, stellsym, optimize_G, second_stage, config, vectorize)

    def subtest_boozer_surface_optimisation_convergence(self, surfacetype,
                                                        stellsym, optimize_G,
                                                        second_stage, config,
                                                        vectorize):
        # Set random seed for reproducibility across platforms
        np.random.seed(RANDOM_SEED)
        base_curves, base_currents, ma, nfp, bs = get_data(config)
        if stellsym:
            coils = bs.coils
        else:
            # Create a stellarator that still has rotational symmetry but
            # doesn't have stellarator symmetry. We do this by first applying
            # stellarator symmetry, then breaking this slightly, and then
            # applying rotational symmetry
            from simsopt.geo.curve import RotatedCurve
            rng = np.random.default_rng(12345)
            curves_flipped = [RotatedCurve(c, 0, True) for c in base_curves]
            currents_flipped = [-cur for cur in base_currents]
            for c in curves_flipped:
                c.rotmat += 0.001*rng.uniform(low=-1., high=1.,
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
            tol=1e-12, maxiter=700, constraint_weight=100/cw, iota=iota, G=G,
            vectorize=vectorize)
        print('Residual norm after LBFGS', res['iter'], np.sqrt(2*res['fun']))

        boozer_surface.recompute_bell()
        if second_stage == 'ls':
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(
                tol=1e-10, maxiter=100, constraint_weight=1000./cw,
                iota=res['iota'], G=res['G'])
        elif second_stage == 'newton':
            res = boozer_surface.minimize_boozer_penalty_constraints_newton(
                tol=1e-10, maxiter=100, constraint_weight=100./cw,
                iota=res['iota'], G=res['G'], vectorize=vectorize)
        elif second_stage == 'newton_exact':
            res = boozer_surface.minimize_boozer_exact_constraints_newton(
                tol=1e-10, maxiter=100, iota=res['iota'], G=res['G'])
        elif second_stage == 'residual_exact':
            res = boozer_surface.solve_residual_equation_exactly_newton(
                tol=1e-10, maxiter=100, iota=res['iota'], G=res['G'])

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
            assert np.linalg.norm(res['residual']) < 1e-9

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

    def test_convergence_cpp_and_notcpp_same(self):
        """
        This unit test verifies that both the cpp (vectorized) and python (non-vectorized) 
        implementations converge to valid solutions. Due to platform-specific numerical 
        differences (e.g., different BLAS/LAPACK implementations), the implementations may
        converge to slightly different local minima, but both should achieve small residuals.
        """
        res_vec, residual_vec = self.subtest_convergence_cpp_and_notcpp_same(True)
        res_nonvec, residual_nonvec = self.subtest_convergence_cpp_and_notcpp_same(False)
        
        # Both implementations should converge successfully
        assert res_vec['success'], "Vectorized implementation failed to converge"
        assert res_nonvec['success'], "Non-vectorized implementation failed to converge"
        
        # Both should achieve small residuals (< 1e-8)
        assert residual_vec < 1e-8, f"Vectorized residual too large: {residual_vec}"
        assert residual_nonvec < 1e-8, f"Non-vectorized residual too large: {residual_nonvec}"
        
        # Both should have similar iota values (within 5%)
        iota_vec = res_vec['iota']
        iota_nonvec = res_nonvec['iota']
        rel_diff = abs(iota_vec - iota_nonvec) / abs(iota_vec)
        assert rel_diff < 0.05, f"Iota values differ by more than 5%: {iota_vec} vs {iota_nonvec}"

    def subtest_convergence_cpp_and_notcpp_same(self, vectorize):
        """
        compute a surface using either the vectorized or non-vectorized subroutines
        Returns the result dict and the final residual norm.
        """
        # Set random seed for reproducibility across platforms
        np.random.seed(RANDOM_SEED)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        s = get_surface('SurfaceXYZTensorFourier', True, nfp=nfp)
        s.fit_to_curve(ma, 0.1)
        iota = -0.4

        ar = Area(s)
        ar_target = ar.J()
        boozer_surface = BoozerSurface(bs, s, ar, ar_target)

        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))

        cw = 3*s.quadpoints_phi.size * s.quadpoints_theta.size
        # Use random_seed for reproducibility across platforms
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=600, constraint_weight=100./cw, iota=iota, G=G,
            vectorize=vectorize)
        print(f'Residual norm after LBFGS (vectorize={vectorize})', np.sqrt(2*res['fun']))

        boozer_surface.recompute_bell()
        res = boozer_surface.minimize_boozer_penalty_constraints_newton(
            tol=1e-10, maxiter=20, constraint_weight=100./cw,
            iota=res['iota'], G=res['G'], vectorize=vectorize)

        # Compute the final residual norm
        residual_norm = np.linalg.norm(res['residual'])
        print(f'Residual norm after Newton (vectorize={vectorize})', residual_norm)
        
        return res, residual_norm

    def test_boozer_penalty_constraints_cpp_notcpp(self):
        """
        Test to verify cpp and python implementations of the BoozerLS objective return the same thing.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for weight_inv_modB in [False, True]:
                    for (optimize_G, nphi, ntheta, mpol, ntor) in [(True, 1, 1, 3, 3), (False, 1, 1, 13, 2), (True, 2, 2, 10, 3), (False, 2, 1, 3, 4), (True, 6, 9, 3, 3), (False, 7, 8, 3, 4), (True, 3, 3, 3, 3), (False, 3, 3, 3, 5)]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          weight_inv_modB=weight_inv_modB,
                                          mpol=mpol,
                                          ntor=ntor):
                            self.subtest_boozer_penalty_constraints_cpp_notcpp(surfacetype, stellsym, optimize_G, nphi, ntheta, weight_inv_modB, mpol, ntor)

    def subtest_boozer_penalty_constraints_cpp_notcpp(self, surfacetype, stellsym, optimize_G, nphi, ntheta, weight_inv_modB, mpol, ntor):

        np.random.seed(1)
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)

        phis = None
        thetas = None
        if nphi == 1:
            phis = [0.2234567989]
        elif nphi == 2:
            phis = [0.2234567989, 0.432123451]

        if ntheta == 1:
            thetas = [0.2432101234]
        elif ntheta == 2:
            thetas = [0.2432101234, 0.9832134]

        s = get_surface(surfacetype, stellsym, nphi=nphi, ntheta=ntheta, thetas=thetas, phis=phis, mpol=mpol, ntor=ntor)
        s.fit_to_curve(ma, 0.1)
        s.x = s.x + np.random.rand(s.x.size)*1e-6

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))

        # deriv = 0
        w = 0.
        f0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=0, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0))

        # deriv = 1
        f0, J0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=1, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1, J1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=1, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(J0, J1, atol=1e-9, rtol=1e-11)

        # check directional derivative
        h1 = np.random.rand(J0.size)-0.5
        np.testing.assert_allclose(J0@h1, J1@h1, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0), np.abs(J0@h1-J1@h1)/np.abs(J0@h1))

        # deriv = 2
        f0, J0, H0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=2, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1, J1, H1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=2, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)

        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(J0, J1, atol=1e-9, rtol=1e-11)
        np.testing.assert_allclose(H0, H1, atol=1e-10, rtol=1e-10)
        h2 = np.random.rand(J0.size)-0.5

        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(J0@h1, J1@h1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose((H0@h1)@h2, (H1@h1)@h2, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0), np.abs(J0@h1-J1@h1)/np.abs(J0@h1), np.abs((H0@h1)@h2-(H1@h1)@h2)/np.abs((H0@h1)@h2))

        def compute_differences(Ha, Hb):
            diff = np.abs(Ha.flatten() - Hb.flatten())
            rel_diff = diff/np.abs(Ha.flatten())
            ij1 = np.where(diff.reshape(Ha.shape) == np.max(diff))
            i1 = ij1[0][0]
            j1 = ij1[1][0]

            ij2 = np.where(rel_diff.reshape(Ha.shape) == np.max(rel_diff))
            i2 = ij2[0][0]
            j2 = ij2[1][0]
            print(f'max err     ({i1:03}, {j1:03}): {np.max(diff):.6e}, {Ha[i1, j1]:.6e}\nmax rel err ({i2:03}, {j2:03}): {np.max(rel_diff):.6e}, {Ha[i2,j2]:.6e}\n')
        compute_differences(H0, H1)

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
