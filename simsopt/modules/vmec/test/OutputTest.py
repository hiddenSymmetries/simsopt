import unittest
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')

from output import VmecOutput
from finite_differences import finiteDifferenceDerivative

class Test(unittest.TestCase):
    
    vmecOutput = VmecOutput('wout.nc',ntheta=70,nzeta=70)
        
    def test_compute_N_derivatives(self):
        """
            Checks that derivatives of Nx, Ny, Nz integrate to zero
            over the full grid
        """
        [dNxdtheta, dNxdzeta, dNydtheta, dNydzeta, dNzdtheta, dNzdzeta] = \
            self.vmecOutput.compute_N_derivatives()
        # Test that each component integrates to 0
        dNxdtheta_sum = np.sum(dNxdtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dNydtheta_sum = np.sum(dNydtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dNzdtheta_sum = np.sum(dNzdtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        self.assertAlmostEqual(dNxdtheta_sum, 0, places=4)
        self.assertAlmostEqual(dNydtheta_sum, 0, places=4)
        self.assertAlmostEqual(dNzdtheta_sum, 0, places=4)
        
    def test_compute_n_derivatives(self):        
        """
            Checks that derivatives of nx, ny, nz integrate to zero
            over the full grid
        """
        [dnxdtheta, dnxdzeta, dnydtheta, dnydzeta, dnzdtheta, dnzdzeta] = \
            self.vmecOutput.compute_n_derivatives()
        # Test that each component integrates to 0
        dnxdtheta_sum = np.sum(dnxdtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dnydtheta_sum = np.sum(dnydtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dnzdtheta_sum = np.sum(dnzdtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        self.assertAlmostEqual(dnxdtheta_sum, 0, places=3)
        self.assertAlmostEqual(dnydtheta_sum, 0, places=3)
        self.assertAlmostEqual(dnzdtheta_sum, 0, places=3)      
        
    def test_normalized_jacobian(self):
        """
            Checks that normalized jacobian integrates to 1
        """
        normalized_jacobian = self.vmecOutput.normalized_jacobian()
        normalized_jacobian_sum = np.sum(normalized_jacobian) \
            / np.sum(np.size(normalized_jacobian))
        self.assertAlmostEqual(normalized_jacobian_sum, 1)
        
    def test_position_first_derivatives(self):
        """
            Checks that derivatives of position vector integrate to zero
            over full grid and performs finite difference testing.
        """
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta] = \
            self.vmecOutput.position_first_derivatives(\
                theta=self.vmecOutput.thetas_2d_full,
                zeta=self.vmecOutput.zetas_2d_full)
        dxdtheta_sum = np.sum(dxdtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dxdzeta_sum = np.sum(dxdzeta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dydtheta_sum = np.sum(dydtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dydzeta_sum = np.sum(dydzeta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dzdtheta_sum = np.sum(dzdtheta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        dzdzeta_sum = np.sum(dzdzeta) * self.vmecOutput.dtheta \
            * self.vmecOutput.dzeta * self.vmecOutput.nfp
        self.assertAlmostEqual(dxdtheta_sum, 0)
        self.assertAlmostEqual(dxdzeta_sum, 0)
        self.assertAlmostEqual(dydtheta_sum, 0)
        self.assertAlmostEqual(dydzeta_sum, 0)
        self.assertAlmostEqual(dzdtheta_sum, 0)
        self.assertAlmostEqual(dzdzeta_sum, 0)
        
        for itheta in range(0,self.vmecOutput.ntheta,5):
            for izeta in range(0,self.vmecOutput.nzeta,5):
                this_zeta = self.vmecOutput.zetas[izeta]
                this_theta = self.vmecOutput.thetas[itheta]
                dxdzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.compute_position(-1, \
                    this_theta,zeta)[0], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dxdzeta_fd, dxdzeta[izeta,itheta], places=5)
                dydzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.compute_position(-1, \
                    this_theta,zeta)[1], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dydzeta_fd, dydzeta[izeta,itheta], places=5)
                dzdzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.compute_position(-1, \
                    this_theta,zeta)[2], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dzdzeta_fd, dzdzeta[izeta,itheta], places=5)
                dxdtheta_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecOutput.compute_position(-1, \
                    theta,this_zeta,)[0], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dxdtheta_fd, dxdtheta[izeta,itheta], places=5)
                dydtheta_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecOutput.compute_position(-1, \
                    theta,this_zeta)[1], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dydtheta_fd, dydtheta[izeta,itheta], places=5)
                dzdtheta_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecOutput.compute_position(-1, \
                    theta,this_zeta)[2], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dzdtheta_fd, dzdtheta[izeta,itheta], places=5)
                
    def test_position_second_derivatives(self):
        """
            Performs finite difference testing of second derivatives of position
            vector.
        """
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
            d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta] = \
            self.vmecOutput.position_second_derivatives()
        
        for itheta in range(0,self.vmecOutput.ntheta,5):
            for izeta in range(0,self.vmecOutput.nzeta,5):
                this_zeta = self.vmecOutput.zetas[izeta]
                this_theta = self.vmecOutput.thetas[itheta]
                d2xdthetadzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.position_first_derivatives(-1, \
                    this_theta,zeta)[0], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2xdthetadzeta_fd, \
                                       d2xdthetadzeta[izeta,itheta], places=5)
                d2ydthetadzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.position_first_derivatives(-1, \
                    this_theta,zeta)[2], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2ydthetadzeta_fd, \
                                       d2ydthetadzeta[izeta,itheta], places=5)
                d2zdthetadzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.position_first_derivatives(-1, \
                    this_theta,zeta)[4], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2zdthetadzeta_fd, \
                                       d2zdthetadzeta[izeta,itheta], places=5)
                d2xdtheta2_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecOutput.position_first_derivatives(-1, \
                    theta,this_zeta)[0], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2xdtheta2_fd, \
                                       d2xdtheta2[izeta,itheta], places=5)
                d2ydtheta2_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecOutput.position_first_derivatives(-1, \
                    theta,this_zeta)[2], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2ydtheta2_fd, \
                                       d2ydtheta2[izeta,itheta], places=5)
                d2zdtheta2_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecOutput.position_first_derivatives(-1, \
                    theta,this_zeta)[4], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2zdtheta2_fd, \
                                       d2zdtheta2[izeta,itheta], places=5)
                d2xdzeta2_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.position_first_derivatives(-1, \
                    this_theta,zeta)[1], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2xdzeta2_fd, \
                                       d2xdzeta2[izeta,itheta], places=5)
                d2ydzeta2_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.position_first_derivatives(-1, \
                    this_theta,zeta)[3], epsilon=1e-5, method='centered')
                self.assertAlmostEqual(d2ydzeta2_fd, \
                                       d2ydzeta2[izeta,itheta], places=5)
                d2zdzeta2_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.position_first_derivatives(-1, \
                    this_theta,zeta)[5], epsilon=1e-5, method='centered')
                self.assertAlmostEqual(d2zdzeta2_fd, \
                                       d2zdzeta2[izeta,itheta], places=5)
                
    def test_jacobian_derivatives(self):
        """
            Performs finite difference testing of derivative of jacobian with 
            respect to boundary harmonics
        """
        
        def temp_fun_rmnc(im, rmnc):
            rmnc_init = self.vmecOutput.rmnc[-1,im]
            self.vmecOutput.rmnc[-1,im] = rmnc
            jacobian = self.vmecOutput.jacobian()
            self.vmecOutput.rmnc[-1,im] = rmnc_init
            return jacobian
        
        def temp_fun_zmns(im, zmns):
            zmns_init = self.vmecOutput.zmns[-1,im]
            self.vmecOutput.zmns[-1,im] = zmns
            jacobian = self.vmecOutput.jacobian()
            self.vmecOutput.zmns[-1,im] = zmns_init
            return jacobian
        
        # Check first 10 modes
        xm_sensitivity = self.vmecOutput.xm[0:10]
        xn_sensitivity = self.vmecOutput.xn[0:10]/self.vmecOutput.nfp

        [dNdrmnc, dNdzmns] = self.vmecOutput.jacobian_derivatives(\
                xm_sensitivity, xn_sensitivity)

        for im in range(len(xm_sensitivity)):
            this_rmnc = self.vmecOutput.rmnc[-1,im]
            this_zmns = self.vmecOutput.zmns[-1,im]
            djacobiandrmnc_fd = finiteDifferenceDerivative(this_rmnc,\
                lambda rmnc : temp_fun_rmnc(im,rmnc), epsilon=1e-5, \
                method='centered')
            djacobiandzmns_fd = finiteDifferenceDerivative(this_zmns,\
                lambda zmns : temp_fun_zmns(im,zmns), epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecOutput.nzeta):
                for itheta in range(self.vmecOutput.ntheta):
                    self.assertAlmostEqual(djacobiandrmnc_fd[izeta,itheta], \
                        dNdrmnc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(djacobiandzmns_fd[izeta,itheta], \
                        dNdzmns[im,izeta,itheta], places=5)
                    
    def test_radius_derivatives(self):
        """
            Performs finite difference testing of derivative of radius with 
            respect to boundary harmonics
        """
        
        def temp_fun_rmnc(im, rmnc):
            rmnc_init = self.vmecOutput.rmnc[-1, im]
            self.vmecOutput.rmnc[-1, im] = rmnc
            [X, Y, Z, R] = self.vmecOutput.compute_position()
            self.vmecOutput.rmnc[-1, im] = rmnc_init
            return R
        
        # Check first 10 modes
        xm_sensitivity = self.vmecOutput.xm[0:10]
        xn_sensitivity = self.vmecOutput.xn[0:10]/self.vmecOutput.nfp

        [dRdrmnc, dRdzmns] = self.vmecOutput.radius_derivatives(\
                xm_sensitivity, xn_sensitivity)

        for im in range(len(xm_sensitivity)):
            this_rmnc = self.vmecOutput.rmnc[-1, im]
            dRdrmnc_fd = finiteDifferenceDerivative(this_rmnc,\
                lambda rmnc : temp_fun_rmnc(im, rmnc), epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecOutput.nzeta):
                for itheta in range(self.vmecOutput.ntheta):
                    self.assertAlmostEqual(dRdrmnc_fd[izeta, itheta], \
                        dRdrmnc[im, izeta, itheta], places=5)
                    self.assertEqual(0, dRdzmns[im, izeta, itheta])
                    
    def area_derivatives(self):
        """
            Performs finite difference testing of derivative of area with 
            respect to boundary harmonics
        """
       
        def temp_fun_rmnc(im, rmnc):
            rmnc_init = self.vmecOutput.rmnc[-1, im]
            self.vmecOutput.rmnc[-1, im] = rmnc
            area = self.vmecOutput.area()
            self.vmecOutput.rmnc[-1, im] = rmnc_init
            return area
        
        def temp_fun_zmns(im, rmnc):
            zmns_init = self.vmecOutput.zmns[-1, im]
            self.vmecOutput.zmns[-1, im] = zmns
            area = self.vmecOutput.area()
            self.vmecOutput.zmns[-1, im] = zmns_init
            return area
        
        # Check first 10 modes
        xm_sensitivity = self.vmecOutput.xm[0:10]
        xn_sensitivity = self.vmecOutput.xn[0:10]/self.vmecOutput.nfp

        [dareadrmnc, dareadzmns] = self.vmecOutput.area_derivatives(\
                xm_sensitivity, xn_sensitivity)
   
        for im in range(len(xm_sensitivity)):
            this_rmnc = self.vmecOutput.rmnc[-1, im]
            this_zmns = self.vmecOutput.zmns[-1, im]
            dareadrmnc_fd = finiteDifferenceDerivative(this_rmnc,\
                lambda rmnc : temp_fun_rmnc(im, rmnc), epsilon=1e-5, \
                method='centered')
            dareadzmns_fd = finiteDifferenceDerivative(this_zmns,\
                lambda zmns : temp_fun_zmns(im, zmns), epsilon=1e-5, \
                method='centered')
            self.assertAlmostEqual(dareadrmnc_fd[izeta, itheta], \
                dareadrmnc[im, izeta, itheta], places=5)
            
    def test_normalized_jacobian_derivatives(self):
        """
            Performs finite difference testing of derivative of normalized
            jacobian with respect to boundary harmonics
        """        
    
        def temp_fun_rmnc(im, rmnc):
            rmnc_init = self.vmecOutput.rmnc[-1, im]
            self.vmecOutput.rmnc[-1, im] = rmnc
            normalized_jacobian = self.vmecOutput.normalized_jacobian()
            self.vmecOutput.rmnc[-1, im] = rmnc_init
            return normalized_jacobian
        
        def temp_fun_zmns(im, zmns):
            zmns_init = self.vmecOutput.zmns[-1, im]
            self.vmecOutput.zmns[-1, im] = zmns
            normalized_jacobian = self.vmecOutput.normalized_jacobian()
            self.vmecOutput.zmns[-1, im] = zmns_init
            return normalized_jacobian
        
        # Check first 10 modes
        xm_sensitivity = self.vmecOutput.xm[0:10]
        xn_sensitivity = self.vmecOutput.xn[0:10]/self.vmecOutput.nfp

        [dnormalized_jacobiandrmnc, dnormalized_jacobiandzmns] = \
            self.vmecOutput.normalized_jacobian_derivatives(xm_sensitivity, \
                                                            xn_sensitivity)

        for im in range(len(xm_sensitivity)):
            this_rmnc = self.vmecOutput.rmnc[-1,im]
            this_zmns = self.vmecOutput.zmns[-1,im]
            dnormalized_jacobiandrmnc_fd = finiteDifferenceDerivative(this_rmnc,\
                lambda rmnc : temp_fun_rmnc(im,rmnc), epsilon=1e-5, \
                method='centered')
            dnormalized_jacobiandzmns_fd = finiteDifferenceDerivative(this_zmns,\
                lambda zmns : temp_fun_zmns(im,zmns), epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecOutput.nzeta):
                for itheta in range(self.vmecOutput.ntheta):
                    self.assertAlmostEqual(dnormalized_jacobiandrmnc_fd[izeta,itheta], \
                        dnormalized_jacobiandrmnc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dnormalized_jacobiandzmns_fd[izeta,itheta], \
                        dnormalized_jacobiandzmns[im,izeta,itheta], places=5)
                    
    def test_B_on_arclength_grid(self):
        """
            Check that min/max of field strength on arclength grid matches with
            that computed on the VMEC grid
        """        
        [Bx_end, By_end, Bz_end, theta_arclength] = \
            self.vmecOutput.B_on_arclength_grid()
        modB2_arclength = Bx_end ** 2 + By_end ** 2 + Bz_end ** 2
        
        modB2 = self.vmecOutput.compute_modB(isurf=self.vmecOutput.ns_half - 1, \
                                            full=False) ** 2 
    
        for izeta in range(self.vmecOutput.nzeta):
            self.assertAlmostEqual(np.min(modB2[izeta,:]), \
                                   np.min(modB2_arclength[izeta,:]), places=1)
            self.assertAlmostEqual(np.max(modB2[izeta,:]), \
                                   np.max(modB2_arclength[izeta,:]), places=1)
            
    def test_compute_position(self):
        """
            Test that X and Y are consistent with R and that Z averages to 0
        """        
        [X, Y, Z, R] = self.vmecOutput.compute_position()
        
        Z_int = np.sum(Z) * self.vmecOutput.dzeta * self.vmecOutput.dtheta \
            * self.vmecOutput.nfp
        self.assertAlmostEqual(Z_int, 0, places=5)
        
        for itheta in range(self.vmecOutput.ntheta):
            for izeta in range(self.vmecOutput.nzeta):
                self.assertAlmostEqual(X[izeta,itheta] ** 2 + \
                                       Y[izeta,itheta] ** 2, \
                                       R[izeta,itheta] ** 2, places = 10)
        
    def test_compute_modB(self):
        """
            Test that ValueError is raised if incorrect grids are passed.
        """                
        thetas = self.vmecOutput.thetas_2d
        zetas = self.vmecOutput.zetas_2d
        zetas = np.delete(zetas, 0)
        self.assertRaises(ValueError, self.vmecOutput.compute_modB, \
                          theta = thetas, zeta = zetas)
        
    def test_compute_current(self):
        """
            Test that the current is computed on the half flux grid.
        """           
        It_half = self.vmecOutput.compute_current()
        self.assertEqual(len(It_half), self.vmecOutput.ns_half)
        
    def test_compute_N(self):
        """
            Test that ValueError is raised if incorrect grid or isurf is passed.
        """           
        thetas = self.vmecOutput.thetas_2d
        zetas = self.vmecOutput.zetas_2d
        zetas = np.delete(zetas, 0)

        self.assertRaises(ValueError, self.vmecOutput.compute_N, \
                         theta = thetas, zeta = zetas)
        self.assertRaises(ValueError, self.vmecOutput.compute_N, \
                         isurf = self.vmecOutput.ns)
        
    def test_evaluate_iota_objective(self):
        """
            Test that ValueError is raised if incorrect weight is passed.
        """                   
        weight = self.vmecOutput.s_half
        weight = np.delete(weight,0)
        
        self.assertRaises(TypeError, self.vmecOutput.evaluate_iota_objective, \
                         weight = weight)
        
    def test_evaluate_well_objective(self):
        """
            Test that ValueError is raised if incorrect weight is passed.
        """                   
        weight = self.vmecOutput.s_half
        weight = np.delete(weight,0)
        
        self.assertRaises(TypeError, self.vmecOutput.evaluate_well_objective, \
                         weight = weight)
        
    def test_jacobian(self):
        """
            Test that ValueError is raised if incorrect grid or isurf is passed.
        """                   
        thetas = self.vmecOutput.thetas_2d
        zetas = self.vmecOutput.zetas_2d
        zetas = np.delete(zetas, 0)

        self.assertRaises(ValueError, self.vmecOutput.jacobian, \
                         theta = thetas, zeta = zetas)
        self.assertRaises(ValueError, self.vmecOutput.jacobian, \
                         isurf = self.vmecOutput.ns)
        
    def test_area(self):
        """
            Test that ValueError is raised if incorrect grid or isurf is passed.
        """                   
        thetas = self.vmecOutput.thetas_2d
        zetas = self.vmecOutput.zetas_2d
        zetas = np.delete(zetas, 0)

        self.assertRaises(ValueError, self.vmecOutput.area, \
                         theta = thetas, zeta = zetas)
        self.assertRaises(ValueError, self.vmecOutput.area, \
                         isurf = self.vmecOutput.ns)
        
    def test_mean_curvature(self):
        """
            Test that ValueError is raised if incorrect isurf is passed.
        """                   
        self.assertRaises(ValueError, self.vmecOutput.mean_curvature, \
                          isurf = self.vmecOutput.ns)

    def test_evaluate_modB_objective(self):
        """
            Test that ValueError is raised if incorrect isurf is passed.
        """  
        self.assertRaises(ValueError, self.vmecOutput.evaluate_modB_objective, \
                          isurf = self.vmecOutput.ns)

    def test_evaluate_modB_objective_volume(self):
        """
            Test that a positive float is returned.
        """  
        output = self.vmecOutput.evaluate_modB_objective_volume()
        self.assertIsInstance(output, float)
        self.assertGreater(output, 0)
                      
if __name__ == '__main__':
    unittest.main()
        
