import numpy as np
import ipdb
from simsopt.core.optimizable import Optimizable
class Surface(Optimizable):
    """
    Surface is a base class for various representations of toroidal
    surfaces in simsopt.
    """

    def __init__(self):
        Optimizable.__init__(self)
        self.dependencies = []
        self.fixed = np.full(len(self.get_dofs()), False)
                             
    
    def plot(self, ax=None, show=True, plot_normal=False, plot_derivative=False, scalars=None, wireframe=True):
        gamma = self.gamma()
        
        from mayavi import mlab
        mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], scalars=scalars)
        if wireframe:
            mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], representation='wireframe', color = (0,0,0))
        

        if plot_derivative:
            dg1 = 0.05 * self.gammadash1()
            dg2 = 0.05 * self.gammadash2()
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], dg1[:,:,0], dg1[:,:,1], dg1[:,:,2])
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], dg2[:,:,0], dg2[:,:,1], dg2[:,:,2])
        if plot_normal:
            n = 0.005 * self.normal()
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], n[:,:,0], n[:,:,1], n[:,:,2])
        if show:
            mlab.show()


    def aspect_ratio(self):
        """
        Note: cylindrical coordinates are (R, phi, Z)
               Boozer coordinates are      (varphi, theta)
       

        For a given surface, this function computes its aspect ratio using the VMEC
        definition:

        AR = R_major / R_minor

        where R_major = (volume enclosed by surface) / (2 pi^2 * R_minor^2)
              R_minor = sqrt[ (mean cross sectional area) / pi ]

        The main difficult part of this calculation is the (mean cross sectional
        area).  This is given by the integral

        (mean cross sectional area) = 1/(2*pi)\int^{2 pi}_{0} \int_{S_phi} dS dphi
        where S_phi is the cross section of the surface at the cylindrical angle phi.

        Note that \int_{S_\phi} dS can be rewritten as a line integral using the divergence
        theorem

        \int_{S_phi}dS = \int_{S_phi} dR dZ 
                       = \int_{\partial S_phi} nabla_{R,Z} \cdot [R,0] \cdot n dl 
                       where n = [n_R, n_Z] is the outward pointing normal
                       = \int_{\partial S_phi} R * n_R dl

        Consider the surface written in cylindrical coordinates in terms of its Boozer angles
        [R(varphi,theta), phi(varphi,theta), Z(varphi,theta)].  \partial S_phi is given by the
        points theta->[R(varphi(phi,theta),theta), Z(varphi(phi,theta),theta)] for fixed
        phi.  The cross sectional area of S_phi becomes

                       = \int^{2pi}_{0} R(varphi(phi,theta),theta)
                                        d/dtheta[Z(varphi(phi,theta),theta)] dtheta

        Now, substituting this into the formula for the mean cross sectional area, we have
        1/(2*pi)\int^{2 pi}_{0}\int^{2pi}_{0} R(varphi(phi,theta),theta)
                                        d/dtheta[Z(varphi(phi,theta),theta)] dtheta dphi

        Instead of integrating over cylindrical phi, let's complete the change of variables and
        integrate over Boozer varphi using the mapping:
        
        [phi,theta] <- [atan2(y(varphi,theta), x(varphi,theta)), theta]

        After the change of variables, the integral becomes:
        1/(2*pi)\int^{2 pi}_{0}\int^{2pi}_{0} R(varphi,theta) [dZ_dvarphi dvarphi_dtheta 
                                                               + dZ_dtheta ] detJ dtheta dvarphi
        where detJ is the determinant of the mapping's Jacobian.
        
        """

        xyz = self.gamma()
        x2y2 = xyz[:,:,0]**2 + xyz[:,:,1]**2
        dgamma1 = self.gammadash1()
        dgamma2 = self.gammadash2()
        
        # compute the average cross sectional area
        J = np.zeros( (xyz.shape[0], xyz.shape[1], 2,2) )
        J[:,:,0,0] = (1./ (2. * np.pi))*(xyz[:,:,0] * dgamma1[:,:,1] - xyz[:,:,1] * dgamma1[:,:,0] )/x2y2
        J[:,:,0,1] = (1./ (2. * np.pi))*(xyz[:,:,0] * dgamma2[:,:,1] - xyz[:,:,1] * dgamma2[:,:,0] )/x2y2
        J[:,:,1,0] = 0.
        J[:,:,1,1] = 1.

        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)
        
        dZ_dtheta = dgamma1[:,:,2] * Jinv[:,:,0,1] + dgamma2[:,:,2] * Jinv[:,:,1,1]
        mean_cross_sectional_area = np.abs(np.mean( np.sqrt(x2y2) * dZ_dtheta * detJ )) 
        
        R_minor = np.sqrt( mean_cross_sectional_area / np.pi )
        R_major = np.abs(self.volume()) / (2. * np.pi**2 * R_minor**2)
        
        AR =  R_major/R_minor
        return AR

    
    def volume(self):
        xyz = self.gamma()
        x = xyz[:,:,0]
        dgamma1 = self.gammadash1()
        dgamma2 = self.gammadash2()
        nor =  np.cross( dgamma1.reshape( (-1,3) ), dgamma2.reshape( (-1,3) ) ).reshape( xyz.shape ) 
        prod = x * nor[:,:,0]
        return np.mean( prod )

    def __repr__(self):
        return "Surface " + str(hex(id(self)))

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.  All subclasses should implement this abstract
        method.
        """
        raise NotImplementedError
