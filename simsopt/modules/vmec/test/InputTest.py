import unittest
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')

from input import VmecInput
from finite_differences import finiteDifferenceDerivative

class Test(unittest.TestCase):

    vmecInput = VmecInput('input.w7x',ntheta=70,nzeta=70)
    
    def test_update_namelist(self):
        """
        Test that nfp is correctly updated in the namelist
        """
        nfp_init = self.vmecInput.nfp
        self.vmecInput.nfp = 20
        self.vmecInput.update_namelist()
        self.assertEqual(self.vmecInput.nfp, self.vmecInput.namelist['nfp'])
        self.vmecInput.nfp = nfp_init
        self.vmecInput.update_namelist()
        self.assertEqual(self.vmecInput.nfp, self.vmecInput.namelist['nfp'])
        self.assertEqual(nfp_init, self.vmecInput.nfp)
        
    def test_update_modes(self):
        """
        Test that number of modes, arrays (xm, xn, rbc, zbs), and length
        (mnmax) are updated correctly
        """        
        mpol = self.vmecInput.mpol
        ntor = self.vmecInput.ntor
        mnmax = self.vmecInput.mnmax
        xm = np.copy(self.vmecInput.xm)
        xn = np.copy(self.vmecInput.xn)
        rbc = np.copy(self.vmecInput.rbc)
        zbs = np.copy(self.vmecInput.zbs)
        # Adjust modes 
        self.vmecInput.update_modes(2 * mpol, 2 * ntor)
        self.assertEqual(np.max(self.vmecInput.xm), 2 * mpol - 1)
        self.assertEqual(np.max(self.vmecInput.xn), 2 * ntor)
        self.assertEqual(len(self.vmecInput.rbc), self.vmecInput.mnmax)
        self.assertEqual(len(self.vmecInput.zbs), self.vmecInput.mnmax)
        self.assertEqual(len(self.vmecInput.xm), self.vmecInput.mnmax)
        self.assertEqual(len(self.vmecInput.xn), self.vmecInput.mnmax)
        # Return to initial values
        self.vmecInput.update_modes(mpol, ntor)
        self.assertEqual(mnmax, self.vmecInput.mnmax)
        self.assertTrue(np.all(xm == self.vmecInput.xm))
        self.assertTrue(np.all(xn == self.vmecInput.xn))
        self.assertTrue(np.all(rbc == self.vmecInput.rbc))
        self.assertTrue(np.all(zbs == self.vmecInput.zbs))
        
    def test_update_grids(self):
        """
        Check that grids update correctly and reset back to initial size. 
        """
        # Save original items
        ntheta = self.vmecInput.ntheta
        nzeta = self.vmecInput.nzeta
        thetas = np.copy(self.vmecInput.thetas)
        zetas = np.copy(self.vmecInput.zetas)
        thetas_2d = np.copy(self.vmecInput.thetas_2d)
        zetas_2d = np.copy(self.vmecInput.zetas_2d)
        dtheta = np.copy(self.vmecInput.dtheta)
        dzeta = np.copy(self.vmecInput.dzeta)
        zetas_full = np.copy(self.vmecInput.zetas_full)
        thetas_2d_full = np.copy(self.vmecInput.thetas_2d_full)
        zetas_2d_full = np.copy(self.vmecInput.zetas_2d_full)
        nfp = self.vmecInput.nfp
        # Update grids
        self.vmecInput.update_grids(2*ntheta, 2*nzeta)
        self.assertEqual(self.vmecInput.ntheta, 2*ntheta)
        self.assertEqual(self.vmecInput.nzeta, 2*nzeta)
        self.assertEqual(len(self.vmecInput.thetas), 2*ntheta)
        self.assertEqual(len(self.vmecInput.zetas), 2*nzeta)
        self.assertEqual(dtheta*ntheta, 2*self.vmecInput.dtheta*ntheta)
        self.assertEqual(dzeta*nzeta, 2*self.vmecInput.dzeta*nzeta)
        self.assertEqual(len(self.vmecInput.zetas_full), \
                         nfp * nzeta * 2)
        self.assertEqual(len(self.vmecInput.thetas_2d_full[:,0]), nzeta * 2 * nfp)
        self.assertEqual(len(self.vmecInput.zetas_2d_full[:,0]), nzeta * 2 * nfp)
        self.assertEqual(len(self.vmecInput.thetas_2d_full[0,:]), ntheta * 2)
        self.assertEqual(len(self.vmecInput.zetas_2d_full[0,:]), ntheta * 2)
        # Reset
        self.vmecInput.update_grids(ntheta, nzeta)
        self.assertEqual(self.vmecInput.ntheta, ntheta)
        self.assertEqual(self.vmecInput.nzeta, nzeta)
        self.assertTrue(np.all(self.vmecInput.thetas == thetas))
        self.assertTrue(np.all(self.vmecInput.zetas == zetas))
        self.assertEqual(dtheta, self.vmecInput.dtheta)
        self.assertEqual(dzeta, self.vmecInput.dzeta)
        self.assertTrue(np.all(self.vmecInput.zetas_full == zetas_full))
        self.assertTrue(np.all(self.vmecInput.thetas_2d_full == thetas_2d_full))
        self.assertTrue(np.all(self.vmecInput.zetas_2d_full == zetas_2d_full))
        
    def test_read_boundary_input(self):
        """
        Check that every non-zero element in rbc_array is in rbc (and for zbs)
        """
        rbc_array = np.array(self.vmecInput.namelist["rbc"]).T
        zbs_array = np.array(self.vmecInput.namelist["zbs"]).T
        
        [rbc, zbs] = self.vmecInput.read_boundary_input()
        for im in range(len(rbc_array[0,:])):
            for jn in range(len(rbc_array[:,0])):
                if (rbc_array[jn,im] == 0):
                    self.assertTrue(np.any(rbc == rbc_array[jn,im]))
        for im in range(len(zbs_array[0,:])):
            for jn in range(len(zbs_array[:,0])):
                if (zbs_array[jn,im] == 0):
                    self.assertTrue(np.any(zbs == zbs_array[jn,im]))
                    
    def test_area(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Test that a positive float is returned.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.area, \
            theta = thetas, zeta = zetas)
        area = self.vmecInput.area()
        self.assertIsInstance(area, float)
        self.assertGreater(area, 0)
        
    def test_volume(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Test that a positive float is returned.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.volume, \
            theta = thetas, zeta = zetas)
        volume = self.vmecInput.volume()
        self.assertIsInstance(volume, float)
        self.assertGreater(volume, 0)
        
    def test_N(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.N, \
            theta = thetas, zeta = zetas)
        [Nx, Ny, Nz] = self.vmecInput.N()
        self.assertTrue(np.all(np.shape(Nx) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Ny) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Nz) == np.shape(thetas)))
        
    def test_position_first_derivatives(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Perform finite-difference testing of derivatives.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.position_first_derivatives, \
            theta = thetas, zeta = zetas)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta] = self.vmecInput.position_first_derivatives()
        self.assertTrue(np.all(np.shape(dxdtheta) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(dxdzeta) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(dydtheta) == np.shape(thetas)))   
        self.assertTrue(np.all(np.shape(dydzeta == np.shape(thetas))))
        self.assertTrue(np.all(np.shape(dzdtheta == np.shape(thetas))))
        self.assertTrue(np.all(np.shape(dzdzeta == np.shape(thetas))))
        self.assertTrue(np.all(np.shape(dRdtheta == np.shape(thetas))))
        self.assertTrue(np.all(np.shape(dRdzeta == np.shape(thetas))))
                
        for itheta in range(0,self.vmecInput.ntheta,5):
            for izeta in range(0,self.vmecInput.nzeta,5):
                this_zeta = self.vmecInput.zetas[izeta]
                this_theta = self.vmecInput.thetas[itheta]
                dxdzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[0], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dxdzeta_fd, dxdzeta[izeta,itheta], 
                                       places=5)
                dydzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[1], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dydzeta_fd, dydzeta[izeta,itheta], 
                                       places=5)
                dzdzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[2], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dzdzeta_fd, dzdzeta[izeta,itheta], 
                                       places=5)
                dRdzeta_fd = finiteDifferenceDerivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[3], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dRdzeta_fd, dRdzeta[izeta,itheta], 
                                       places=5)
                dxdtheta_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[0], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dxdtheta_fd, dxdtheta[izeta,itheta], 
                                       places=5)
                dydtheta_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[1], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dydtheta_fd, dydtheta[izeta,itheta], 
                                       places=5)
                dzdtheta_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[2], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dzdtheta_fd, dzdtheta[izeta,itheta], 
                                       places=5)
                dRdtheta_fd = finiteDifferenceDerivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[3], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dRdtheta_fd, dRdtheta[izeta,itheta], 
                                       places=5)     
    
    def test_jacobian(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.jacobian, \
            theta = thetas, zeta = zetas)
        norm_normal = self.vmecInput.jacobian()
        self.assertTrue(np.all(np.shape(thetas) == np.shape(norm_normal)))
        
    def test_normalized_jacobian(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Checks that normalized jacobian integrates to 1
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.normalized_jacobian, \
            theta = thetas, zeta = zetas)
        normalized_jacobian = self.vmecInput.normalized_jacobian()
        self.assertTrue(np.all(np.shape(thetas) == \
                               np.shape(normalized_jacobian)))

        normalized_jacobian_sum = np.sum(normalized_jacobian) \
            / np.sum(np.size(normalized_jacobian))
        self.assertAlmostEqual(normalized_jacobian_sum, 1)
        
    def test_normalized_jacobian_derivatives(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Performs finite difference testing of derivatives.
        """ 
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        self.assertRaises(ValueError, \
                          self.vmecInput.normalized_jacobian_derivatives, 
                          xm_sensitivity, xn_sensitivity, \
                          theta = thetas, zeta = zetas)
        [dnormalized_jacobiandrbc, dnormalized_jacobiandzbs] = \
            self.vmecInput.normalized_jacobian_derivatives(xm_sensitivity, \
                                                           xn_sensitivity)

        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dnormalized_jacobiandrbc[1,:,:]))))
        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dnormalized_jacobiandzbs[1,:,:]))))

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            normalized_jacobian = self.vmecInput.normalized_jacobian()
            self.vmecInput.rbc[im] = rbc_init
            return normalized_jacobian
        
        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            normalized_jacobian = self.vmecInput.normalized_jacobian()
            self.vmecInput.zbs[im] = zbs_init
            return normalized_jacobian
        
        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dnormalized_jacobiandrbc_fd = finiteDifferenceDerivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-5, \
                method='centered')
            dnormalized_jacobiandzbs_fd = finiteDifferenceDerivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(dnormalized_jacobiandrbc_fd[izeta, itheta], \
                        dnormalized_jacobiandrbc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dnormalized_jacobiandzbs_fd[izeta, itheta], \
                        dnormalized_jacobiandzbs[im,izeta,itheta], places=5)
                    
    def test_position(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Test that X and Y are consistent with R and that Z averages to 0
        """         
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
                
        self.assertRaises(ValueError, \
                          self.vmecInput.position, 
                          theta = thetas, zeta = zetas)
        
        thetas = self.vmecInput.thetas_2d_full
        zetas = self.vmecInput.zetas_2d_full

        [X, Y, Z, R] = self.vmecInput.position(theta = thetas,zeta = zetas)
        self.assertTrue(np.all(np.shape(X) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Y) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Z) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(R) == np.shape(thetas)))

        Z_int = np.sum(Z) * self.vmecInput.dzeta * self.vmecInput.dtheta 
        self.assertAlmostEqual(Z_int, 0, places=5)
        
        for itheta in range(self.vmecInput.ntheta):
            for izeta in range(self.vmecInput.nzeta):
                self.assertAlmostEqual(X[izeta,itheta] ** 2 + \
                                       Y[izeta,itheta] ** 2, \
                                       R[izeta,itheta] ** 2, places = 10)        
        
if __name__ == '__main__':
    unittest.main()