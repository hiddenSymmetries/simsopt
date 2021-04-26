import unittest
import numpy as np
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curverzfourier import CurveRZFourier 
from simsopt.geo.curvexyzfourier import CurveXYZFourier 
from simsopt.geo.surfaceobjectives import NonQuasiAxisymmetricComponentPenalty, ToroidalFlux 
from simsopt.geo.coilcollection import CoilCollection
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

#class ToroidalFluxTests(unittest.TestCase):
#    def test_toroidal_flux_is_constant(self):
#        # this test ensures that the toroidal flux does not change, regardless
#        # of the cross section (varphi = constant) across which it is computed
#        s = get_exact_surface()
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#        
#        gamma = s.gamma()
#        num_phi = gamma.shape[0]
#        
#        tf_list = np.zeros( (num_phi,) )
#        for idx in range(num_phi):
#            tf = ToroidalFlux(s, bs_tf,idx = idx)
#            tf_list[idx] = tf.J()
#        mean_tf = np.mean(tf_list)
#
#        max_err = np.max( np.abs(mean_tf - tf_list) ) / mean_tf
#        assert max_err < 1e-2
#
#    def test_toroidal_flux_first_derivative(self):
#        for surfacetype in surfacetypes_list:
#            for stellsym in stellsym_list:
#                with self.subTest(surfacetype = surfacetype, stellsym=stellsym):
#                    self.subtest_toroidal_flux1(surfacetype,stellsym)
#    def test_toroidal_flux_second_derivative(self):
#        for surfacetype in surfacetypes_list:
#            for stellsym in stellsym_list:
#                with self.subTest(surfacetype = surfacetype, stellsym=stellsym):
#                    self.subtest_toroidal_flux2(surfacetype,stellsym)
#
#    def subtest_toroidal_flux1(self, surfacetype, stellsym):
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#        s = get_surface(surfacetype, stellsym)
#        
#        tf = ToroidalFlux(s, bs_tf)
#        coeffs = s.get_dofs()
#        
#        def f(dofs):
#            s.set_dofs(dofs)
#            return tf.J()
#        def df(dofs):
#            s.set_dofs(dofs)
#            return tf.dJ_by_dsurfacecoefficients() 
#        taylor_test1(f, df, coeffs)
# 
#    def subtest_toroidal_flux2(self, surfacetype, stellsym):
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        s = get_surface(surfacetype, stellsym)
#
#        tf = ToroidalFlux(s, bs)
#        coeffs = s.get_dofs()
#        
#        def f(dofs):
#            s.set_dofs(dofs)
#            return tf.J()
#        def df(dofs):
#            s.set_dofs(dofs)
#            return tf.dJ_by_dsurfacecoefficients() 
#        def d2f(dofs):
#            s.set_dofs(dofs)
#            return tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
#
#        taylor_test2(f, df, d2f, coeffs) 


class NonQuasiSymmetricComponentPenaltyTests(unittest.TestCase):
#    def test_NonQuasiSymmetricComponentPenalty_by_surfacecoefficients(self):
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        s = get_exact_surface()
#        
#        non_qs = NonQuasiAxisymmetricComponentPenalty(s, bs)
#        coeffs = s.get_dofs()
#        def f(dofs):
#            s.set_dofs(dofs)
#            return non_qs.J()
#        def df(dofs):
#            s.set_dofs(dofs)
#            return non_qs.dJ_by_dsurfacecoefficients() 
#        taylor_test1(f, df, coeffs )
    def test_NonQuasiAxiSymmetricComponentPenalty_by_coilcoefficients(self):
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        s = get_exact_surface()

        non_qs = NonQuasiAxisymmetricComponentPenalty(s, bs)
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
#    def test_NonQuasiSymmetricComponentPenalty_exact(self):
#        """
#        This test verifies that the non-QS penalty term returns what we expect it to when
#        we know what the quasisymmetric and non-quasisymmetric components of the field strength
#        are.
#        """
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#        s = get_exact_surface()
#        nphi = s.gamma().shape[0] 
#        ntheta= s.gamma().shape[1] 
#        quadpoints_phi   = s.quadpoints_phi
#        quadpoints_theta = s.quadpoints_theta
#
#        grid_phi,grid_theta = np.meshgrid(quadpoints_phi, quadpoints_theta)
#        grid_phi = grid_phi.T
#        grid_theta = grid_theta.T
#
#        modes_phi   = 1 + (nphi-1)/2
#        modes_theta = 1 + (ntheta-1)/2
#
#        bf_phi   = np.ones( (nphi,  nphi,ntheta) )
#        bf_theta = np.ones( (ntheta,nphi,ntheta) )
#        
#        bf_phi[1:(nphi+1),:,:] = np.sin(2*np.pi * (1:(nphi+1)) ) 
#        bf_phi[(nphi+1):,:,:]  = np.cos(2*np.pi * (1:(nphi+1)) ) 
#        bf_theta[1:(ntheta+1),:,:] = np.sin(2*np.pi * (1:(ntheta+1)) ) 
#        bf_theta[(ntheta+1):,:,:]  = np.cos(2*np.pi * (1:(ntheta+1)) ) 
#       
#        qs_idx1 = N*np.arange( min(modes_phi,modes_theta) )
#        qs_idx2 =   np.arange( min(modes_phi,modes_theta) )
# 
#        amps = np.random.rand(nphi,ntheta)
#        amps_non_qs = np.copy(amps)
#        amps_non_qs[qs_idx1              ,qs_idx2                ] = 0.
#        
#        if N>0:
#            amps_non_qs[qs_idx1[1:]+modes_phi,qs_idx2[1:]            ] = 0.
#            amps_non_qs[qs_idx1[1:]          ,qs_idx2[1:]+modes_theta] = 0.
#            amps_non_qs[qs_idx1[1:]+modes_phi,qs_idx2[1:]+modes_theta] = 0.
#
#        func = np.sum(amps[:,None,None] * bf_phi[:,None,...] * bf_theta[None,:,...],axis = 0)
#        func_nonqs = np.sum(amps_non_qs[:,:,None,None] * bf_phi[:,None,...] * bf_theta[None,:,...],axis = (0,1))
#        class BiotSavartTestField:
#            def __init__(self,coils, currents):
#                self.coils = coils
#                self.currents = currents
#            def set_points(self,x):
#                self.points = points
#                phi = np.linspace(0,1,nphi)
#                theta = np.linspace(0,1,ntheta)
#                self.phi,self.theta = np.meshgrid(phi,theta)
#                self.phi = self.phi.T
#                self.theta = self.theta.T
#            def clear_cached_properties(self):
#                self._B = None
#            def B(self, compute_derivatives=0):
#                self._B = 
#
#
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#
#        non_qs = NonQuasiSymmetricComponentPenalty(s, bs, N = 0)
#        coeffs = stellarator.get_dofs()
       

