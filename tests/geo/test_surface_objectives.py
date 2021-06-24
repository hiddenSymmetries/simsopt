import unittest
import numpy as np
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curverzfourier import CurveRZFourier 
from simsopt.geo.curvexyzfourier import CurveXYZFourier 
from simsopt.geo.surfaceobjectives import NonQuasiAxisymmetricComponentPenalty, ToroidalFlux, Area, MajorRadius
from simsopt.geo.surfaceobjectives import boozer_surface_residual
from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.boozersurface import BoozerSurface
from .surface_test_helpers import get_ncsx_data, get_surface, get_exact_surface



surfacetypes_list = ["SurfaceXYZFourier", "SurfaceRZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


def taylor_test1(f, df, x, epsilons=None, direction=None):
    np.random.seed(1)
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(10, 20)))
    print("################################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = f(x + eps * direction)
        dfest = (fpluseps-f0)/eps
        err = np.linalg.norm(dfest - dfx)
        print(err/err_old)
        assert err < 0.55 * err_old
        err_old = err
    print("################################################################################")

def taylor_test2(f, df, d2f, x, epsilons=None, direction1=None, direction2 = None):
    np.random.seed(1)
    if direction1 is None:
        direction1 = np.random.rand(*(x.shape))-0.5
    if direction2 is None:
        direction2 = np.random.rand(*(x.shape))-0.5

    f0 = f(x)
    df0 = df(x) @ direction1
    d2fval = direction2.T @ d2f(x) @ direction1
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(7, 20)))
    print("################################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = df(x + eps * direction2) @ direction1
        d2fest = (fpluseps-df0)/eps
        err = np.abs(d2fest - d2fval)
        
        print(err/err_old)
        assert err < 0.6 * err_old
        err_old = err
    print("################################################################################")

class ToroidalFluxTests(unittest.TestCase):
    def test_toroidal_flux_is_constant(self):
        # this test ensures that the toroidal flux does not change, regardless
        # of the cross section (varphi = constant) across which it is computed
        s = get_exact_surface()
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        
        import ipdb;ipdb.set_trace()

        gamma = s.gamma()
        num_phi = gamma.shape[0]
        
        tf_list = np.zeros( (num_phi,) )
        for idx in range(num_phi):
            tf = ToroidalFlux(s, bs_tf, stellarator, idx = idx)
            tf_list[idx] = tf.J()
        mean_tf = np.mean(tf_list)

        max_err = np.max( np.abs(mean_tf - tf_list) ) / mean_tf
        assert max_err < 1e-2
    
    def test_toroidal_flux_dcoilcoeff(self):
        s = get_exact_surface()
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        
        tf = ToroidalFlux(s, bs, stellarator)
        coeffs = stellarator.get_dofs()
        def f(dofs):
            stellarator.set_dofs(dofs)
            return tf.J()
        def df(dofs):
            stellarator.set_dofs(dofs)
            dJ_by_dcoils = tf.dJ_by_dcoilcoefficients()
            dJ_by_dcoils = stellarator.reduce_coefficient_derivatives(dJ_by_dcoils)
            return dJ_by_dcoils 
        taylor_test1(f, df, coeffs )


    def test_toroidal_flux_first_derivative(self):
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype = surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux1(surfacetype,stellsym)
    def test_toroidal_flux_second_derivative(self):
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype = surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux2(surfacetype,stellsym)

    def subtest_toroidal_flux1(self, surfacetype, stellsym):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        s = get_surface(surfacetype, stellsym)
        
        tf = ToroidalFlux(s, bs_tf, stellarator)
        coeffs = s.get_dofs()
        
        def f(dofs):
            s.set_dofs(dofs)
            return tf.J()
        def df(dofs):
            s.set_dofs(dofs)
            return tf.dJ_by_dsurfacecoefficients() 
        taylor_test1(f, df, coeffs)
 
    def subtest_toroidal_flux2(self, surfacetype, stellsym):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs, stellarator)
        coeffs = s.get_dofs()
        
        def f(dofs):
            s.set_dofs(dofs)
            return tf.J()
        def df(dofs):
            s.set_dofs(dofs)
            return tf.dJ_by_dsurfacecoefficients() 
        def d2f(dofs):
            s.set_dofs(dofs)
            return tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        taylor_test2(f, df, d2f, coeffs) 


class NonQuasiAxiSymmetricComponentPenaltyTests(unittest.TestCase):
    def test_NonQuasiSymmetricComponentPenalty_by_coilcurrents(self):
        current_fak = 1./(4 * np.pi * 1e-7)
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)

        mpol = 5  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 5  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota = -0.4
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Area(s, stellarator)
        label_target = label.J()
 
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        boozer_s = BoozerSurface(bs, s, label, label_target)
        res = boozer_s.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=300,iota=iota,G=G0)

        if res['success'] == False:
            raise Exception('Surface computation did not converge')

        iota_target = 2. * res['iota']
        iota_weight = 3.1213
        non_qs = NonQuasiAxisymmetricComponentPenalty(boozer_s, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=1., target_major_radius=0.1)
        coeffs = stellarator.get_currents()/current_fak
        def f(dofs):
            stellarator.set_currents(current_fak *dofs )
            bs.clear_cached_properties()
            return non_qs.J()
        def df(dofs):
            stellarator.set_currents(current_fak *dofs )
            bs.clear_cached_properties()
            dJ_by_dcurrents = stellarator.reduce_current_derivatives(non_qs.dJ_by_dcoilcurrents()) * current_fak
            return dJ_by_dcurrents
        taylor_test1(f, df, coeffs )




    def test_NonQuasiAxiSymmetricComponentPenalty_by_surfacecoefficients(self):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)

        mpol = 5  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 5  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota = -0.4
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Area(s, stellarator)
        label_target = label.J()
 
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        boozer_s = BoozerSurface(bs, s, label, label_target)
        res = boozer_s.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=300,iota=iota,G=G0)

        if res['success'] == False:
            raise Exception('Surface computation did not converge')

        iota_target = 2. * res['iota']
        iota_weight = 3.1213
        non_qs = NonQuasiAxisymmetricComponentPenalty(boozer_s, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=1., target_major_radius=0.1) 
        coeffs = s.get_dofs()
        def f(dofs):
            s.set_dofs(dofs)
            return non_qs.J()
        def df(dofs):
            s.set_dofs(dofs)
            return non_qs.dJ_by_dsurfacecoefficients() 
        taylor_test1(f, df, coeffs )
    def test_NonQuasiSymmetricComponentPenalty_by_coilcoefficients(self):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)

        mpol = 5  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 5  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota = -0.4
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Area(s, stellarator)
        label_target = label.J()
 
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        boozer_s = BoozerSurface(bs, s, label, label_target)
        res = boozer_s.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=300,iota=iota,G=G0)

        if res['success'] == False:
            raise Exception('Surface computation did not converge')

        iota_target = 2. * res['iota']
        iota_weight = 3.1213
        non_qs = NonQuasiAxisymmetricComponentPenalty(boozer_s, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=1., target_major_radius=0.1)
        coeffs = stellarator.get_dofs()
        def f(dofs):
            stellarator.set_dofs(dofs)
            bs.clear_cached_properties()
            return non_qs.J()
        def df(dofs):
            stellarator.set_dofs(dofs)
            bs.clear_cached_properties()
            dJ_by_dcoils = non_qs.dJ_by_dcoilcoefficients()
            dJ_by_dcoils = stellarator.reduce_coefficient_derivatives(dJ_by_dcoils)
            return dJ_by_dcoils
        taylor_test1(f, df, coeffs )

    def test_NonQuasiSymmetricComponentPenalty_exact(self):
        """
        This test verifies that the non-QS penalty term returns what we expect it to when
        we know what the non-quasisymmetric component of the field is zero.
        """
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)

        mpol = 5  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 5  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota = -0.4
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Area(s, stellarator)
        label_target = label.J()
 
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        boozer_s = BoozerSurface(bs, s, label, label_target)
        res = boozer_s.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=300,iota=iota,G=G0)

        if res['success'] == False:
            raise Exception('Surface computation did not converge')

        iota_target = res['iota']
        iota_weight = 3.1213
        non_qs = NonQuasiAxisymmetricComponentPenalty(boozer_s, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=1., target_major_radius=0.1)






        quadpoints_phi   = s.quadpoints_phi
        quadpoints_theta = s.quadpoints_theta

        grid_phi,grid_theta = np.meshgrid(quadpoints_phi, quadpoints_theta)
        grid_phi = grid_phi.T
        grid_theta = grid_theta.T
        class BiotSavartTestField:
            def __init__(self,coils, currents):
                self.coils = coils
                self.currents = currents
            def set_points(self,x):
                self.points = x
                phi = quadpoints_phi
                theta = quadpoints_theta
                self.phi,self.theta = np.meshgrid(phi,theta)
                self.phi = self.phi.T
                self.theta = self.theta.T
                self.clear_cached_properties()
            def clear_cached_properties(self):
                self._B = None
            def B(self, compute_derivatives=0):
                if self._B is None:
                    assert compute_derivatives >= 0
                    self._B = np.cos(self.theta).reshape((-1,1)) * np.ones( self.points.shape )+np.sin(self.theta).reshape((-1,1)) * np.ones( self.points.shape ) 
                return self._B



        bs = BiotSavartTestField(coils,currents)
        boozer_surface = BoozerSurface(bs, s, label, label_target)
        boozer_surface.res = boozer_s.res
        non_qs = NonQuasiAxisymmetricComponentPenalty(boozer_surface, stellarator, target_iota=iota_target, iota_weight=iota_weight)
        
        assert np.abs(non_qs.J()) < 1e-14
    def test_objective_exact_constrained_gradient(self):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        coils = stellarator.coils
        currents = stellarator.currents
        current_fak = 1./(4 * np.pi * 1e-7)
        
       
        mpol = 8  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 8  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota = -0.4
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        label = ToroidalFlux(s, bs_tf, stellarator)
        label_target = 0.05

 
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        boozer_surface = BoozerSurface(bs, s, label, label_target)
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=300,iota=iota,G=G0)
        iota0 = res['iota']
        print(f"iota={res['iota']:.3f}, label={label.J():.3f}, area={s.area():.3f}, |label error|={np.abs(label.J()-label_target):.3e}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
        
        iota_target = 2. * res['iota']
        iota_weight = 3.1213
        problem = NonQuasiAxisymmetricComponentPenalty(boozer_surface, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=1., target_major_radius=0.1) 
        coeffs_curr = stellarator.get_currents()/current_fak
        coeffs_coil = stellarator.get_dofs()
        coeffs = np.concatenate([coeffs_curr, coeffs_coil])
        idx_curr = (0,len(stellarator.get_currents()))
        idx_coil = (idx_curr[1], idx_curr[1] + len(stellarator.get_dofs()))
        

        def f(dofs):
            stellarator.set_currents(dofs[idx_curr[0]:idx_curr[1]]*current_fak)
            stellarator.set_dofs(dofs[idx_coil[0]:idx_coil[1]])
            bs.clear_cached_properties()
            boozer_surface.res = problem.boozer_surface.solve_residual_equation_exactly_newton( tol=1e-10, maxiter=10, iota=iota0, G=G0)
            if not boozer_surface.res['success']:
                raise Exception('Did not converge')
            J = problem.J()
            return J

        def df(dofs):
            stellarator.set_currents(dofs[idx_curr[0]:idx_curr[1]]*current_fak)
            stellarator.set_dofs(dofs[idx_coil[0]:idx_coil[1]])
            bs.clear_cached_properties()
            boozer_surface.res = problem.boozer_surface.solve_residual_equation_exactly_newton( tol=1e-10, maxiter=10, iota=iota0, G=G0)
            if not boozer_surface.res['success']:
                raise Exception('Did not converge')
            dJ_curr, dJ_coil = problem.dJ()
            return np.concatenate([dJ_curr, dJ_coil])
        
        h = np.zeros(coeffs.shape)
        h[-1]=1.
        taylor_test1(f, df, coeffs, direction=h)



    def test_objective_exact_constrained_gradient(self):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        coils = stellarator.coils
        currents = stellarator.currents
        current_fak = 1./(4 * np.pi * 1e-7)
        
       
        mpol = 8  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 8  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota = -0.4
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        label = ToroidalFlux(s,bs)
#        label_target = 0.025

        label = Area(s, stellarator)
        label_target = 5
 
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        boozer_surface = BoozerSurface(bs, s, label, label_target)
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=300,iota=iota,G=G0)
        iota0 = res['iota']
        print(f"iota={res['iota']:.3f}, label={label.J():.3f}, area={s.area():.3f}, |label error|={np.abs(label.J()-label_target):.3e}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
        
        iota_target = 2. * res['iota']
        iota_weight = 3.1213
        problem = NonQuasiAxisymmetricComponentPenalty(boozer_surface, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=1., target_major_radius=0.1) 
        coeffs_curr = stellarator.get_currents()/current_fak
        coeffs_coil = stellarator.get_dofs()
        coeffs = np.concatenate([coeffs_curr, coeffs_coil])
        idx_curr = (0,len(stellarator.get_currents()))
        idx_coil = (idx_curr[1], idx_curr[1] + len(stellarator.get_dofs()))
        

        def f(dofs):
            stellarator.set_currents(dofs[idx_curr[0]:idx_curr[1]]*current_fak)
            stellarator.set_dofs(dofs[idx_coil[0]:idx_coil[1]])
            bs.clear_cached_properties()
            boozer_surface.res = problem.boozer_surface.solve_residual_equation_exactly_newton( tol=1e-10, maxiter=10, iota=iota0, G=G0)
            if not boozer_surface.res['success']:
                raise Exception('Did not converge')
            J = problem.J()
            return J

        def df(dofs):
            stellarator.set_currents(dofs[idx_curr[0]:idx_curr[1]]*current_fak)
            stellarator.set_dofs(dofs[idx_coil[0]:idx_coil[1]])
            bs.clear_cached_properties()
            boozer_surface.res = problem.boozer_surface.solve_residual_equation_exactly_newton( tol=1e-10, maxiter=10, iota=iota0, G=G0)
            if not boozer_surface.res['success']:
                raise Exception('Did not converge')
            dJ_curr, dJ_coil = problem.dJ()
            return np.concatenate([dJ_curr, dJ_coil])
        taylor_test1(f, df, coeffs)

    def test_objective_penalty_constrained_gradient(self):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        coils = stellarator.coils
        currents = stellarator.currents
        
        mpol = 5  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 5  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        
        phis = np.linspace(0, 1/(2*nfp), ntor+5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+5, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota = -0.4
        
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Area(s, stellarator)
        label_target = 5

        weight = 100.
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        boozer_surface = BoozerSurface(bs, s, label, label_target)
        res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=300, constraint_weight = weight, iota=iota,G=G0, method='manual')
        print(f"iota={res['iota']:.3f}, label={label.J():.3f}, area={s.area():.3f}, |label error|={np.abs(label.J()-label_target):.3e}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
        
        iota_target = 2. * res['iota']
        iota_weight = 3.1213
        problem = NonQuasiAxisymmetricComponentPenalty(boozer_surface, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=1., target_major_radius=0.1) 
        coeffs = stellarator.get_dofs()
        
        def f(dofs):
            stellarator.set_dofs(dofs)
            bs.clear_cached_properties()
            boozer_surface.res = problem.boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=300, constraint_weight = weight, iota=iota,G=G0, method='manual')
            if not boozer_surface.res['success']:
                raise Exception('Did not converge')
            J = problem.J()
            return J

        def df(dofs):
            stellarator.set_dofs(dofs)
            bs.clear_cached_properties()
            boozer_surface.res = problem.boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=300, constraint_weight = weight, iota=iota,G=G0, method='manual')
            if not boozer_surface.res['success']:
                raise Exception('Did not converge')
            dJ = problem.dJ()
            return dJ

        taylor_test1(f, df, coeffs)


class MajorRadiusTests(unittest.TestCase):
#     def test_average_area(self):
#        s = get_exact_surface()
#        mr = MajorRadius(s)
#        print(mr.J())
    def test_first_derivative(self):
        s = get_exact_surface()
        mr = MajorRadius(s)

        coeffs = s.get_dofs()
        def f(dofs):
            s.set_dofs(dofs)
            return mr.J()
        def df(dofs):
            s.set_dofs(dofs)
            return mr.dJ() 
        taylor_test1(f, df, coeffs)


