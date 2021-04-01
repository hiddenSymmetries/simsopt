import unittest
import numpy as np
from math import pi
from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curverzfourier import CurveRZFourier 
from simsopt.geo.curvexyzfourier import CurveXYZFourier 
from simsopt.geo.surfaceobjectives import ToroidalFlux 
from simsopt.geo.surfaceobjectives import Area 
from .surface_test_helpers import get_ncsx_data, get_surface, get_exact_surface 


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


class BoozerSurfaceTests(unittest.TestCase):
    def test_residual(self):
        s = get_exact_surface()
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

        weight = 1.
        tf = ToroidalFlux(s, bs_tf)
        
        ## these data are obtained from `boozer` branch of pyplamsaopt
        tf_target = 0.41431152
        iota =  -0.44856192

        boozerSurface = BoozerSurface(bs, s, tf, tf_target) 
        x = np.concatenate((s.get_dofs(), [iota]))
        r0 = boozerSurface.boozer_penalty_constraints(x, derivatives=0, constraint_weight=weight, optimize_G=False, scalarize=False)
        # the residual should be close to zero for all entries apart from the y
        # and z coordinate at phi=0 and theta=0 (and the corresponding rotations)
        ignores_idxs = np.zeros_like(r0)
        ignores_idxs[[1, 2, 693, 694, 695, 1386, 1387, 1388, -2, -1]] = 1
        assert np.max(np.abs(r0[ignores_idxs<0.5])) < 1e-8
        assert np.max(np.abs(r0[-2:])) < 1e-6

        
    def test_boozer_penalty_constraints_gradient(self):
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, optimize_G=optimize_G):
                        self.subtest_boozer_penalty_constraints_gradient(surfacetype, stellsym, optimize_G)

    def test_boozer_penalty_constraints_hessian(self):
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, optimize_G=optimize_G):
                        self.subtest_boozer_penalty_constraints_hessian(surfacetype, stellsym, optimize_G)

    def subtest_boozer_penalty_constraints_gradient(self, surfacetype, stellsym, optimize_G=False):
        np.random.seed(1)
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
         
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        
        s = get_surface(surfacetype,stellsym)
        s.fit_to_curve(ma, 0.1)
        
        weight = 11.1232


        tf = ToroidalFlux(s, bs_tf)

        tf_target = 0.1
        boozerSurface = BoozerSurface(bs, s, tf, tf_target) 

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))]))
        f0, J0 = boozerSurface.boozer_penalty_constraints(x, derivatives=1, constraint_weight=weight, optimize_G=optimize_G)

        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("################################################################################")
        for eps in epsilons:
            f1 = boozerSurface.boozer_penalty_constraints(x + eps*h, derivatives=0, constraint_weight=weight, optimize_G=optimize_G)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old, f0, f1)
            assert err < err_old * 0.55
            err_old = err
        print("################################################################################")

    def subtest_boozer_penalty_constraints_hessian(self, surfacetype, stellsym, optimize_G=False):
        np.random.seed(1)
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        
        s = get_surface(surfacetype,stellsym)
        s.fit_to_curve(ma, 0.1)
        
        tf = ToroidalFlux(s, bs_tf)

        tf_target = 0.1
        boozerSurface = BoozerSurface(bs, s, tf, tf_target) 

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))]))
        f0, J0, H0 = boozerSurface.boozer_penalty_constraints(x, derivatives=2, optimize_G=optimize_G)

        h1 = np.random.uniform(size=x.shape)-0.5
        h2 = np.random.uniform(size=x.shape)-0.5
        d2f = h1@H0@h2

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(10, 20)))
        print("################################################################################")
        for eps in epsilons:
            fp, Jp = boozerSurface.boozer_penalty_constraints(x + eps*h1, derivatives=1, optimize_G=optimize_G)
            d2f_fd = (Jp@h2-J0@h2)/eps
            err = np.abs(d2f_fd-d2f)/np.abs(d2f)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err

    def test_boozer_constrained_jacobian(self):
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, optimize_G=optimize_G):
                        self.subtest_boozer_constrained_jacobian(surfacetype, stellsym, optimize_G)

    def subtest_boozer_constrained_jacobian(self, surfacetype, stellsym, optimize_G=False):
        np.random.seed(1)
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        
        s = get_surface(surfacetype,stellsym)
        s.fit_to_curve(ma, 0.1)
        
        tf = ToroidalFlux(s, bs_tf)

        tf_target = 0.1
        boozerSurface = BoozerSurface(bs, s, tf, tf_target) 

        iota = -0.3
        lm = [0.,0.,0.]
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))]))
        xl = np.concatenate((x, lm ))
        res0, dres0 = boozerSurface.boozer_exact_constraints(xl, derivatives=1, optimize_G=optimize_G)
        
        h = np.random.uniform(size=xl.shape)-0.5
        dres_exact = dres0@h
        

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("################################################################################")
        for eps in epsilons:
            res1 = boozerSurface.boozer_exact_constraints(xl + eps*h, derivatives=0, optimize_G=optimize_G)
            dres_fd = (res1-res0)/eps
            err = np.linalg.norm(dres_fd-dres_exact)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err
        print("################################################################################")

    
    def test_BoozerSurface(self):
        configs = [
            ("SurfaceXYZTensorFourier", False, True,  'ls'),
            ("SurfaceXYZTensorFourier", True,  True,  'newton'),
            ("SurfaceXYZTensorFourier", True,  True,  'newton_exact'),
            ("SurfaceXYZTensorFourier", True,  True,  'ls'),
            ("SurfaceXYZFourier",       True,  False, 'ls'),
        ]
        for surfacetype, stellsym, optimize_G, second_stage in configs:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym, optimize_G=optimize_G, second_stage=second_stage):
                    self.subtest_BoozerSurface(surfacetype, stellsym, optimize_G, second_stage)

    def subtest_BoozerSurface(self, surfacetype, stellsym, optimize_G, second_stage):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        
        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)
        iota = -0.3
        
        ar = Area(s)
        ar_target = ar.J()
        boozerSurface = BoozerSurface(bs, s, ar, ar_target) 

        if optimize_G:
            G = 2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))
        else:
            G = None
       
        # compute surface first using LBFGS exact and an area constraint
        res = boozerSurface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-9, maxiter=400, constraint_weight=100., iota=iota, G=G)
        print('Squared residual after LBFGS', res['fun'])
        if second_stage == 'ls':
            res = boozerSurface.minimize_boozer_penalty_constraints_ls(tol=1e-9, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'])
        elif second_stage == 'newton':
            res = boozerSurface.minimize_boozer_penalty_constraints_newton(tol=1e-9, maxiter=10, constraint_weight=100., iota=res['iota'], G=res['G'], stab=1e-4)
        elif second_stage == 'newton_exact':
            res = boozerSurface.minimize_boozer_exact_constraints_newton(tol=1e-9, maxiter=10, iota=res['iota'], G=res['G'])

        assert res['success']

        if surfacetype == 'SurfaceXYZTensorFourier':
            assert np.linalg.norm(res['residual']) < 1e-9

        if second_stage == 'newton_exact' or surfacetype == 'SurfaceXYZTensorFourier':
            assert np.abs(ar_target - ar.J()) < 1e-9

if __name__ == "__main__":
    unittest.main()
