import unittest
import numpy as np
from math import pi
from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curverzfourier import CurveRZFourier 
from simsopt.geo.curvexyzfourier import CurveXYZFourier 
from simsopt.geo.surfaceobjectives import ToroidalFlux 
from simsopt.geo.surfaceobjectives import Area 
from surface_test_helpers import get_ncsx_data,get_surface, get_exact_surface 


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceRZFourier"]
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
        f0 = boozerSurface.boozer_penalty_constraints(x, derivatives = 0, constraint_weight = weight)
        # f0 should be close to 0, but it's not within machine precision because tf_target and iota
        # are only known to 8 significant figures
        assert f0 < 1e-5

        
    def test_boozer_penalty_constraints_gradient(self):
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype = surfacetype, stellsym=stellsym):
                    self.subtest_boozer_penalty_constraints_gradient(surfacetype,stellsym)

    def test_boozer_penalty_constraints_hessian(self):
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype = surfacetype, stellsym=stellsym):
                    self.subtest_boozer_penalty_constraints_hessian(surfacetype,stellsym)

    def subtest_boozer_penalty_constraints_gradient(self, surfacetype, stellsym):
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
        f0, J0 = boozerSurface.boozer_penalty_constraints(x, derivatives = 1, constraint_weight = weight)

        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("################################################################################")
        for eps in epsilons:
            f1 = boozerSurface.boozer_penalty_constraints(x + eps*h, derivatives = 0, constraint_weight = weight)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old, f0, f1)
            assert err < err_old * 0.55
            err_old = err
        print("################################################################################")

    def subtest_boozer_penalty_constraints_hessian(self, surfacetype, stellsym):
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
        f0, J0, H0 = boozerSurface.boozer_penalty_constraints(x, derivatives=2)

        h1 = np.random.uniform(size=x.shape)-0.5
        h2 = np.random.uniform(size=x.shape)-0.5
        d2f = h1@H0@h2

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(10, 20)))
        print("################################################################################")
        for eps in epsilons:
            fp, Jp = boozerSurface.boozer_penalty_constraints(x + eps*h1, derivatives = 1)
            d2f_fd = (Jp@h2-J0@h2)/eps
            err = np.abs(d2f_fd-d2f)/np.abs(d2f)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err

    def test_boozer_constrained_jacobian(self):
        for surfacetype in surfacetypes_list:
                    for stellsym in stellsym_list:
                        with self.subTest(surfacetype = surfacetype, stellsym=stellsym):
                            self.subtest_boozer_constrained_jacobian(surfacetype,stellsym)

    def subtest_boozer_constrained_jacobian(self, surfacetype, stellsym):
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
        xl = np.concatenate((s.get_dofs(), [iota], lm ))
        res0, dres0 = boozerSurface.boozer_exact_constraints(xl, derivatives = 1)
        
        h = np.random.uniform(size=xl.shape)-0.5
        dres_exact = dres0@h
        

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("################################################################################")
        for eps in epsilons:
            res1 = boozerSurface.boozer_exact_constraints(xl + eps*h, derivatives = 0)
            dres_fd = (res1-res0)/eps
            err = np.linalg.norm(dres_fd-dres_exact)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err
        print("################################################################################")

    def test_BoozerSurface(self):
        stellsym = True
        surfacetype = "SurfaceXYZFourier"
        
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        
        s = get_surface(surfacetype,stellsym)
        s.fit_to_curve(ma, 0.1)
        iota = -0.3
        
        ar = Area(s)
        ar_target = ar.J()
        boozerSurface = BoozerSurface(bs, s, ar, ar_target) 
       
        # compute surface first using LBFGS exact and an area constraint
        s,iota = boozerSurface.minimize_boozer_penalty_constraints_LBFGS(tol = 1e-12, maxiter = 100, constraint_weight = 100., iota = iota)
        s,iota = boozerSurface.minimize_boozer_penalty_constraints_ls(tol = 1e-12, maxiter = 100, constraint_weight = 100., iota = iota)

        tf = ToroidalFlux(s, bs_tf)
        tf_target = 0.1
        boozerSurface = BoozerSurface(bs, s, tf, tf_target) 
        print("Initial toroidal flux is :", tf.J(), "Target toroidal flux is: ", tf_target)
        print("Surface computed using LBFGS and penalised toroidal flux constraint") 
        s,iota = boozerSurface.minimize_boozer_penalty_constraints_LBFGS(tol = 1e-12, maxiter = 100, constraint_weight = 100., iota = iota)
        s,iota = boozerSurface.minimize_boozer_penalty_constraints_ls(tol = 1e-12, maxiter = 100, constraint_weight = 100., iota = iota)
        print("Toroidal flux is :", tf.J(), "Target toroidal flux is: ", tf_target, "\n")
        
        print("Surface computed using Newton and penalised toroidal flux constraint") 
        s, iota = boozerSurface.minimize_boozer_penalty_constraints_newton(tol = 1e-11, maxiter = 10, constraint_weight = 100., iota = iota)
        print("Toroidal flux is :", tf.J(), "Target toroidal flux is: ", tf_target, "\n")
        

        print("Surface computed using Newton and exact toroidal flux constraint") 
        s, iota, lm = boozerSurface.minimize_boozer_exact_constraints_newton(tol = 1e-11, maxiter = 10,iota = iota)
        print("Toroidal flux is :", tf.J(), "Target toroidal flux is: ", tf_target, "\n")
        assert np.abs(tf_target - tf.J()) < 1e-11

if __name__ == "__main__":
    unittest.main()
