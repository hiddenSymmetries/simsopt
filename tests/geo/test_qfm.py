import unittest
import numpy as np
from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.geo.surfaceobjectives import Area, Volume
from .surface_test_helpers import get_ncsx_data, get_surface, get_exact_surface

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
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

        weight = 1.
        tf = ToroidalFlux(s, bs_tf)

        # these data are obtained from `boozer` branch of pyplamsaopt
        tf_target = 0.41431152

        qfm_surface = QfmSurface(bs, s, tf, tf_target)
        x = s.get_dofs()
        r0 = qfm_surface.qfm_penalty_constraints(x, derivatives=0,
                                                 constraint_weight=weight)
        assert(r0 < 1e-10)

    def test_qfm_objective_gradient(self):
        """
        Taylor test to verify the gradient of the qfm objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_qfm_objective_gradient(surfacetype, stellsym)

    def subtest_qfm_objective_gradient(self, surfacetype, stellsym):
        np.random.seed(1)
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

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
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("################################################################################")
        for eps in epsilons:
            f1 = qfm_surface.qfm_objective(
                x + eps*h, derivatives=0)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old)
            assert err < err_old * 0.6
            err_old = err
        print("################################################################################")

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
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

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
        print("################################################################################")
        for eps in epsilons:
            f1 = qfm_surface.qfm_label_constraint(
                x + eps*h, derivatives=0)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old)
            assert err < err_old * 0.6
            err_old = err
        print("################################################################################")

    def test_qfm_penalty_constraints_gradient(self):
        """
        Taylor test to verify the gradient of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_qfm_penalty_constraints_gradient(surfacetype, stellsym)

    def test_qfm_penalty_constraints_hessian(self):
        """
        Taylor test to verify the Hessian of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_qfm_penalty_constraints_hessian(surfacetype, stellsym)

    def subtest_qfm_penalty_constraints_gradient(self, surfacetype, stellsym):
        np.random.seed(1)
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

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
        epsilons = np.power(2., -np.asarray(range(7, 17)))
        print("################################################################################")
        for eps in epsilons:
            f1 = qfm_surface.qfm_penalty_constraints(
                x + eps*h, derivatives=0, constraint_weight=weight)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old)
            assert err < err_old * 0.6
            err_old = err
        print("################################################################################")

    def subtest_qfm_penalty_constraints_hessian(self, surfacetype, stellsym):
        np.random.seed(1)
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf)

        tf_target = 0.5
        qfm_surface = QfmSurface(bs, s, tf, tf_target)

        x = s.get_dofs()
        f0, J0, H0 = qfm_surface.qfm_penalty_constraints(x, derivatives=2)

        h1 = np.random.uniform(size=x.shape)-0.5
        h2 = np.random.uniform(size=x.shape)-0.5
        d2f = h1@H0@h2

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(9, 18)))
        print("################################################################################")
        for eps in epsilons:
            fp, Jp = qfm_surface.qfm_penalty_constraints(x + eps*h1, derivatives=1)
            d2f_fd = (Jp@h2-J0@h2)/eps
            err = np.abs(d2f_fd-d2f)/np.abs(d2f)
            print(err/err_old)
            assert err < err_old * 0.6
            err_old = err

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
        coils, currents, ma = get_ncsx_data()

        if stellsym:
            stellarator = CoilCollection(coils, currents, 3, True)
        else:
            # Create a stellarator that still has rotational symmetry but
            # doesn't have stellarator symmetry. We do this by first applying
            # stellarator symmetry, then breaking this slightly, and then
            # applying rotational symmetry
            from simsopt.geo.curve import RotatedCurve
            coils_flipped = [RotatedCurve(c, 0, True) for c in coils]
            currents_flipped = [-cur for cur in currents]
            for c in coils_flipped:
                c.rotmat += 0.001*np.random.uniform(low=-1., high=1.,
                                                    size=c.rotmat.shape)
                c.rotmatT = c.rotmat.T
            stellarator = CoilCollection(coils + coils_flipped,
                                         currents + currents_flipped, 3, False)

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

        nfp = 3
        phis = np.linspace(0, 1/nfp, 50, endpoint=False)
        thetas = np.linspace(0, 1, 50, endpoint=False)

        s = get_surface(surfacetype, stellsym, phis=phis, thetas=thetas)
        s.fit_to_curve(ma, 0.1)

        vol = Volume(s)
        vol_target = vol.J()
        qfm_surface = QfmSurface(bs, s, vol, vol_target)

        # Compute surface first using LBFGS and a volume constraint
        res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=1e-9, maxiter=1000, constraint_weight=1e4)

        assert res['success']
        assert np.linalg.norm(res['gradient']) < 1e-2
        assert res['fun'] < 1e-5
        assert np.abs(vol_target - vol.J()) < 1e-5

        # As a second step, optimize with SLSQP

        res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-9, maxiter=1000)

        assert res['success']
        assert np.linalg.norm(res['gradient']) < 1e-2
        assert res['fun'] < 1e-5
        assert np.abs(vol_target - vol.J()) < 1e-5

        vol_opt1 = vol.J()

        # Now optimize with area constraint

        ar = Area(s)
        ar_target = ar.J()
        qfm_surface = QfmSurface(bs, s, ar, ar_target)

        res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=1e-9, maxiter=1000, constraint_weight=1e4)

        assert res['success']
        assert res['fun'] < 1e-5
        assert np.linalg.norm(res['gradient']) < 1e-2
        assert np.abs(ar_target - ar.J()) < 1e-7

        res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-9, maxiter=1000)

        assert res['success']
        assert res['fun'] < 1e-5
        assert np.linalg.norm(res['gradient']) < 1e-2
        assert np.abs(ar_target - ar.J()) < 1e-4

        vol_opt2 = vol.J()

        # Check that volume after second opt does not change

        assert np.abs(vol_opt2 - vol_opt1) < 1e-4


if __name__ == "__main__":
    unittest.main()
