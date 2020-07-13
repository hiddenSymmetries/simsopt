import unittest
import sys
import numpy as np

sys.path.append('..')

from output import VmecOutput
from finite_differences import finiteDifferenceDerivative

class Test(unittest.TestCase):
    
    vmecOutput = VmecOutput('wout_QAS.nc',ntheta=100,nzeta=100)
        
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
        self.assertAlmostEqual(dNxdtheta_sum,0)
        self.assertAlmostEqual(dNydtheta_sum,0)
        self.assertAlmostEqual(dNzdtheta_sum,0)
        
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
        self.assertAlmostEqual(dnxdtheta_sum,0,places=5)
        self.assertAlmostEqual(dnydtheta_sum,0,places=5)
        self.assertAlmostEqual(dnzdtheta_sum,0,places=5)      
        
    def test_normalize_jacobian(self):
        """
            Checks that normalized jacobian integrates to 1
        """
        normalized_jacobian = self.vmecOutput.normalized_jacobian()
        normalized_jacobian_sum = np.sum(normalized_jacobian) \
            / np.sum(np.size(normalized_jacobian))
        self.assertAlmostEqual(normalized_jacobian_sum,1)
        
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
        self.assertAlmostEqual(dxdtheta_sum,0)
        self.assertAlmostEqual(dxdzeta_sum,0)
        self.assertAlmostEqual(dydtheta_sum,0)
        self.assertAlmostEqual(dydzeta_sum,0)
        self.assertAlmostEqual(dzdtheta_sum,0)
        self.assertAlmostEqual(dzdzeta_sum,0)
        
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
                    this_theta,zeta)[3], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2ydzeta2_fd, \
                                       d2ydzeta2[izeta,itheta], places=5)
                d2zdzeta2_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecOutput.position_first_derivatives(-1, \
                    this_theta,zeta)[5], epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2zdzeta2_fd, \
                                       d2zdzeta2[izeta,itheta], places=5)

        
if __name__ == '__main__':
    unittest.main()
        
