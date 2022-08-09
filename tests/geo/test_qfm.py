import unittest
import numpy as np
from simsopt.field.coil import coils_via_symmetries
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.geo.surfaceobjectives import Area, Volume
from simsopt.configs.zoo import get_ncsx_data, get_hsx_data
from .surface_test_helpers import get_surface, get_exact_surface

surfacetypes_list = ["SurfaceXYZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


class QfmSurfaceTests(unittest.TestCase):

    def test_residual(self):
        """
        This test loads a SurfaceXYZFourier that interpolates the xyz coordinates
        of a surface in the NCSX configuration that was computed on a previous
        branch of pyplasmaopt.  Here, we verify that the QFM residual at these
        interpolation points is small. This test has been copied from
        test_boozersurface.py since Boozer coordinates are well defined when
        a magnetic surface exists, and thus the QFM residual should be small.
        """

        s = get_exact_surface()
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        weight = 1.
        tf = ToroidalFlux(s, bs_tf)

        # these data are obtained from `boozer` branch of pyplamsaopt
        tf_target = 0.41431152

        qfm_surface = QfmSurface(bs, s, tf, tf_target)
        x = s.get_dofs()
        r0 = qfm_surface.qfm_penalty_constraints(x, derivatives=0,
                                                 constraint_weight=weight)
        assert (r0 < 1e-10)

    def test_qfm_objective_gradient(self):
        """
        Taylor test to verify the gradient of the qfm objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for config in [get_ncsx_data, get_hsx_data]:
                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, config=config):
                        self.subtest_qfm_objective_gradient(surfacetype, stellsym, config)

    def subtest_qfm_objective_gradient(self, surfacetype, stellsym, get_data):
        np.random.seed(1)
        curves, currents, ma = get_data()
        nfp = ma.nfp
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf)

        tf_target = 0.1
        qfm_surface = QfmSurface(bs, s, tf, tf_target)

        x = s.get_dofs()
        f0, J0 = qfm_surface.qfm_objective(
            x, derivatives=1)

        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(13, 20)))
        print("###############################################################")
        for eps in epsilons:
            f1 = qfm_surface.qfm_objective(
                x + eps*h, derivatives=0)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old)
            assert err < err_old * 0.6
            err_old = err
        print("###############################################################")

    def test_qfm_label_constraint_gradient(self):
        """
        Taylor test to verify the gradient of the qfm objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_qfm_label_constraint_gradient(surfacetype, stellsym)

    def subtest_qfm_label_constraint_gradient(self, surfacetype, stellsym):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf)

        tf_target = 0.1
        qfm_surface = QfmSurface(bs, s, tf, tf_target)

        x = s.get_dofs()
        f0, J0 = qfm_surface.qfm_label_constraint(x, derivatives=1)

        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 17)))
        print("###############################################################")
        for eps in epsilons:
            f1 = qfm_surface.qfm_label_constraint(
                x + eps*h, derivatives=0)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old)
            assert err < err_old * 0.6
            err_old = err
        print("###############################################################")

    def test_qfm_penalty_constraints_gradient(self):
        """
        Taylor test to verify the gradient of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for get_data in [get_ncsx_data, get_hsx_data]:
                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, get_data=get_data):
                        self.subtest_qfm_penalty_constraints_gradient(surfacetype, stellsym, get_data)

    def subtest_qfm_penalty_constraints_gradient(self, surfacetype, stellsym, get_data):
        np.random.seed(1)
        curves, currents, ma = get_data()
        nfp = ma.nfp
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        weight = 11.1232

        tf = ToroidalFlux(s, bs_tf)

        tf_target = 0.1
        qfm_surface = QfmSurface(bs, s, tf, tf_target)

        x = s.get_dofs()
        f0, J0 = qfm_surface.qfm_penalty_constraints(
            x, derivatives=1, constraint_weight=weight)

        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(12, 17)))
        print("###############################################################")
        for eps in epsilons:
            f1 = qfm_surface.qfm_penalty_constraints(
                x + eps*h, derivatives=0, constraint_weight=weight)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old)
            assert err < err_old * 0.6
            err_old = err
        print("###############################################################")

    def test_qfm_surface_optimization_convergence(self):
        """
        Test to verify the various optimization algorithms that find a QFM
        surface
        """

        configs = [
            ("SurfaceXYZTensorFourier", False),
            ("SurfaceXYZTensorFourier", True),
            ("SurfaceXYZFourier", True),
            ("SurfaceXYZFourier", False)
        ]
        for surfacetype, stellsym in configs:
            with self.subTest(
                    surfacetype=surfacetype, stellsym=stellsym):
                self.subtest_qfm_surface_optimization_convergence(surfacetype,
                                                                  stellsym)

    def subtest_qfm_surface_optimization_convergence(self, surfacetype,
                                                     stellsym):
        """
        For each configuration, first reduce penalty objective using LBFGS at
        fixed volume. Then solve constrained problem using SLSQP. Repeat
        both steps for fixed area. Check that volume is preserved.
        """
        curves, currents, ma = get_ncsx_data()
        nfp = 3

        if stellsym:
            coils = coils_via_symmetries(curves, currents, nfp, True)
        else:
            # Create a stellarator that still has rotational symmetry but
            # doesn't have stellarator symmetry. We do this by first applying
            # stellarator symmetry, then breaking this slightly, and then
            # applying rotational symmetry
            from simsopt.geo.curve import RotatedCurve
            curves_flipped = [RotatedCurve(c, 0, True) for c in curves]
            currents_flipped = [-cur for cur in currents]
            for c in curves_flipped:
                c.rotmat += 0.001*np.random.uniform(low=-1., high=1.,
                                                    size=c.rotmat.shape)
                c.rotmatT = c.rotmat.T.copy()
            coils = coils_via_symmetries(curves + curves_flipped, currents + currents_flipped, nfp, False)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        phis = np.linspace(0, 1/nfp, 20, endpoint=False)
        thetas = np.linspace(0, 1, 20, endpoint=False)
        constraint_weight = 1e0

        s = get_surface(surfacetype, stellsym, phis=phis, thetas=thetas, ntor=4,
                        mpol=4)
        s.fit_to_curve(ma, 0.2)

        vol = Volume(s)
        vol_target = vol.J()
        qfm_surface = QfmSurface(bs, s, vol, vol_target)

        # Compute surface first using LBFGS and a volume constraint
        res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=1e-8, maxiter=1000, constraint_weight=constraint_weight)

        assert res['success']
        assert np.linalg.norm(res['gradient']) < 1e-2
        assert res['fun'] < 1e-5
        assert np.abs(vol_target - vol.J()) < 1e-4

        # As a second step, optimize with SLSQP

        res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-9,
                                                               maxiter=1000)

        assert res['success']
        assert np.linalg.norm(res['gradient']) < 1e-3
        assert res['fun'] < 1e-5
        assert np.abs(vol_target - vol.J()) < 1e-5

        vol_opt1 = vol.J()

        # Now optimize with area constraint

        ar = Area(s)
        ar_target = ar.J()
        qfm_surface = QfmSurface(bs, s, ar, ar_target)

        res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=1e-9, maxiter=1000, constraint_weight=constraint_weight)

        assert res['success']
        assert res['fun'] < 1e-5
        assert np.linalg.norm(res['gradient']) < 1e-2
        assert np.abs(ar_target - ar.J()) < 1e-5

        res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-9,
                                                               maxiter=1000)

        assert res['success']
        assert res['fun'] < 1e-5
        assert np.linalg.norm(res['gradient']) < 1e-3
        assert np.abs(ar_target - ar.J()) < 1e-4

        vol_opt2 = vol.J()

        # Check that volume after second opt does not change

        assert np.abs(vol_opt2 - vol_opt1) < 1e-3

    def test_minimize_qfm(self):
        """
        Test to verify that minimize_qfm() yields same result as
        minimize_qfm_exact_constraints_SLSQP or
        minimize_qfm_penalty_constraints_LBFGS separately.
        """

        configs = [
            ("SurfaceXYZTensorFourier", False),
            ("SurfaceXYZTensorFourier", True),
            ("SurfaceXYZFourier", True),
            ("SurfaceXYZFourier", False)
        ]
        for surfacetype, stellsym in configs:
            with self.subTest(
                    surfacetype=surfacetype, stellsym=stellsym):
                self.subtest_minimize_qfm(surfacetype, stellsym)

    def subtest_minimize_qfm(self, surfacetype, stellsym):
        """
        For each configuration, test to verify that minimize_qfm() yields same
        result as minimize_qfm_exact_constraints_SLSQP or
        minimize_qfm_penalty_constraints_LBFGS separately. Test that InputError
        is raised if 'LBFGS' or 'SLSQP' is passed.
        """
        curves, currents, ma = get_ncsx_data()
        nfp = 3

        if stellsym:
            coils = coils_via_symmetries(curves, currents, nfp, True)
        else:
            # Create a stellarator that still has rotational symmetry but
            # doesn't have stellarator symmetry. We do this by first applying
            # stellarator symmetry, then breaking this slightly, and then
            # applying rotational symmetry
            from simsopt.geo.curve import RotatedCurve
            curves_flipped = [RotatedCurve(c, 0, True) for c in curves]
            currents_flipped = [-cur for cur in currents]
            for c in curves_flipped:
                c.rotmat += 0.001*np.random.uniform(low=-1., high=1.,
                                                    size=c.rotmat.shape)
                c.rotmatT = c.rotmat.T.copy()
            coils = coils_via_symmetries(curves + curves_flipped, currents + currents_flipped, nfp, False)

        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        phis = np.linspace(0, 1/nfp, 20, endpoint=False)
        thetas = np.linspace(0, 1, 20, endpoint=False)
        constraint_weight = 1e0

        s = get_surface(surfacetype, stellsym, phis=phis, thetas=thetas, ntor=3,
                        mpol=3)
        s.fit_to_curve(ma, 0.2)

        vol = Volume(s)
        vol_target = vol.J()
        qfm_surface = QfmSurface(bs, s, vol, vol_target)

        # Compute surface first using LBFGS and a volume constraint
        res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=1e-8, maxiter=1000, constraint_weight=constraint_weight)

        grad1 = np.linalg.norm(res['gradient'])
        fun1 = res['fun']
        vol1 = vol.J()

        # Perform same calculation by calling qfm_minimize
        s = get_surface(surfacetype, stellsym, phis=phis, thetas=thetas, ntor=3,
                        mpol=3)
        s.fit_to_curve(ma, 0.2)

        res = qfm_surface.minimize_qfm(method='LBFGS',
                                       tol=1e-9, maxiter=1000, constraint_weight=constraint_weight)

        grad2 = np.linalg.norm(res['gradient'])
        fun2 = res['fun']
        vol2 = vol.J()

        # Test for preservation of results
        np.allclose(grad1, grad2)
        np.allclose(fun1, fun2)
        np.allclose(vol1, vol2)

        # Perform calculation with SLSQP
        s = get_surface(surfacetype, stellsym, phis=phis, thetas=thetas, ntor=3,
                        mpol=3)
        s.fit_to_curve(ma, 0.2)
        res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-9,
                                                               maxiter=1000)

        grad1 = np.linalg.norm(res['gradient'])
        fun1 = res['fun']
        vol1 = vol.J()

        # Perform same calculation by calling qfm_minimize
        s = get_surface(surfacetype, stellsym, phis=phis, thetas=thetas, ntor=3,
                        mpol=3)
        s.fit_to_curve(ma, 0.2)
        res = qfm_surface.minimize_qfm(method='SLSQP',
                                       tol=1e-9, maxiter=1000)

        grad2 = np.linalg.norm(res['gradient'])
        fun2 = res['fun']
        vol2 = vol.J()

        # Test for preservation of results
        np.allclose(grad1, grad2)
        np.allclose(fun1, fun2)
        np.allclose(vol1, vol2)

        # Test that InputError raised
        with self.assertRaises(ValueError):
            res = qfm_surface.minimize_qfm(method='SLSQPP',
                                           tol=1e-9, maxiter=1000)


if __name__ == "__main__":
    unittest.main()
