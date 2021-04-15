import numpy as np
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
        mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], scalars=scalars)
        if wireframe:
            mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], representation='wireframe', color=(0, 0, 0), opacity=0.5)
        

        if plot_derivative:
            dg1 = 0.05 * self.gammadash1()
            dg2 = 0.05 * self.gammadash2()
            mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], dg1[:, :, 0], dg1[:, :, 1], dg1[:, :, 2])
            mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], dg2[:, :, 0], dg2[:, :, 1], dg2[:, :, 2])
        if plot_normal:
            n = 0.005 * self.normal()
            mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], n[:, :, 0], n[:, :, 1], n[:, :, 2])
        if show:
            mlab.show()


    def __repr__(self):
        return "Surface " + str(hex(id(self)))

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.  All subclasses should implement this abstract
        method.
        """
        raise NotImplementedError

    def cross_section(self, phi, varphi_resolution=None, thetas=None):
        """
        This function takes in a cylindrical angle phi and returns the cross
        section of the surface in that plane. This is done using the method of bisection.

        This function assumes that the surface intersection with the plane is a single
        curve.
        """

        # phi is assumed to be between [-pi, pi], so if it does not lie on that interval
        # we shift it by multiples of 2pi until it does
        phi = phi - np.sign(phi) * np.floor(np.abs(phi) / (2*np.pi)) * (2. * np.pi)
        if phi > np.pi:
            phi = phi - 2. * np.pi
        if phi < -np.pi:
            phi = phi + 2. * np.pi
        
        if varphi_resolution is None:
            varphi_resolution = self.gamma().shape[0]

        # varphi are the search intervals on which we look for the cross section in 
        # at constant cylindrical phi
        # The cross section is sampled at a number of points (theta_resolution) poloidally.
        varphi = np.linspace(0., 1., varphi_resolution * self.nfp, endpoint=False)

        if thetas is None:
            theta = np.asarray(self.quadpoints_theta)
        elif isinstance(thetas, np.ndarray):
            theta = thetas
        elif isinstance(thetas, int):
            theta = np.linspace(0, 1, thetas, endpoint=False)
        else:
            raise NotImplementedError('Need to pass int or 1d np.array to thetas')

        varphigrid, thetagrid = np.meshgrid(varphi, theta)
        varphigrid = varphigrid.T
        thetagrid = thetagrid.T

        # sample the surface at the varphi and theta points
        gamma = np.zeros((varphigrid.shape[0], varphigrid.shape[1], 3))
        self.gamma_impl(gamma, varphi, theta)

        # compute the cylindrical phi coordinate of each sampled point on the surface
        cyl_phi = np.arctan2(gamma[:, :, 1], gamma[:, :, 0])
        
        # reorder varphi, theta with respect to increasing cylindrical phi
        idx = np.argsort(cyl_phi, axis=0)
        cyl_phi = np.take_along_axis(cyl_phi, idx, axis=0)
        varphigrid = np.take_along_axis(varphigrid, idx, axis=0)
        
        # In case the target cylindrical angle "phi" lies above the first row or below the last row,
        # we must concatenate the lower row above the top row and the top row below the lower row.
        # This is allowable since the data in the matrices are periodic
        cyl_phi = np.concatenate((cyl_phi[-1, :][None, :]-2.*np.pi, cyl_phi, cyl_phi[0, :][None, :]+2.*np.pi), axis=0)
        varphigrid = np.concatenate((varphigrid[-1, :][None, :]-1., varphigrid, varphigrid[0, :][None, :]+1.), axis=0)
        
        # ensure that varphi does not have massive jumps.
        diff = varphigrid[1:]-varphigrid[:-1]
        pinc = np.abs(diff+1) < np.abs(diff)
        minc = np.abs(diff-1) < np.abs(diff)
        inc = pinc.astype(int) - minc.astype(int)
        prefix_sum = np.cumsum(inc, axis=0)
        varphigrid[1:] = varphigrid[1:] + prefix_sum
        
        # find the subintervals in varphi on which the desired cross section lies.
        # if idx_right == 0, then the subinterval must be idx_left = 0 and idx_right = 1
        idx_right = np.argmax(phi <= cyl_phi, axis=0)
        idx_right = np.where(idx_right == 0, 1, idx_right)
        idx_left = idx_right-1 
        


        varphi_left = varphigrid[idx_left, np.arange(idx_left.size)]
        varphi_right = varphigrid[idx_right, np.arange(idx_right.size)]
        cyl_phi_left = cyl_phi[idx_left, np.arange(idx_left.size)]
        cyl_phi_right = cyl_phi[idx_right, np.arange(idx_right.size)]
       
        # this function converts varphi to cylindrical phi, ensuring that the returned angle
        # lies between left_bound and right_bound.
        def varphi2phi(varphi_in, left_bound, right_bound):
            gamma = np.zeros((varphi_in.size, 3))
            self.gamma_lin(gamma, varphi_in, theta)
            phi = np.arctan2(gamma[:, 1], gamma[:, 0])
            pinc = (phi < left_bound).astype(int) 
            minc = (phi > right_bound).astype(int)
            phi = phi + 2.*np.pi * (pinc - minc)
            return phi
        def bisection(phia, a, phic, c):
            err = 1.
            while err > 1e-13:
                b = (a + c)/2.
                phib = varphi2phi(b, phia, phic)
                
                flag = (phib - phi) * (phic - phi) > 0
                # if flag is true,  then root lies on interval [a,b)
                # if flag is false, then root lies on interval [b,c]
                phia = np.where(flag, phia, phib)
                phic = np.where(flag, phib, phic)
                a = np.where(flag, a, b)
                c = np.where(flag, b, c)
                err = np.max(np.abs(a-c))
            b = (a + c)/2.
            return b          
        # bisect cyl_phi to compute the cross section
        sol = bisection(cyl_phi_left, varphi_left, cyl_phi_right, varphi_right)
        cross_section = np.zeros((sol.size, 3))
        self.gamma_lin(cross_section, sol, theta) 
        return cross_section

    def aspect_ratio(self):
        """
        Note: cylindrical coordinates are                  (R, phi, Z)
              angles that parametrize the surface are      (varphi, theta)
       
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
        Consider the surface written in cylindrical coordinates in terms of its angles
        [R(varphi,theta), phi(varphi,theta), Z(varphi,theta)].  \partial S_phi is given by the
        points theta->[R(varphi(phi,theta),theta), Z(varphi(phi,theta),theta)] for fixed
        phi.  The cross sectional area of S_phi becomes
                       = \int^{2pi}_{0} R(varphi(phi,theta),theta)
                                        d/dtheta[Z(varphi(phi,theta),theta)] dtheta
        Now, substituting this into the formula for the mean cross sectional area, we have
        1/(2*pi)\int^{2 pi}_{0}\int^{2pi}_{0} R(varphi(phi,theta),theta)
                                        d/dtheta[Z(varphi(phi,theta),theta)] dtheta dphi
        Instead of integrating over cylindrical phi, let's complete the change of variables and
        integrate over varphi using the mapping:
        
        [phi,theta] <- [atan2(y(varphi,theta), x(varphi,theta)), theta]
        After the change of variables, the integral becomes:
        1/(2*pi)\int^{2 pi}_{0}\int^{2pi}_{0} R(varphi,theta) [dZ_dvarphi dvarphi_dtheta 
                                                               + dZ_dtheta ] detJ dtheta dvarphi
        where detJ is the determinant of the mapping's Jacobian.
        
        """

        xyz = self.gamma()
        x2y2 = xyz[:, :, 0]**2 + xyz[:, :, 1]**2
        dgamma1 = self.gammadash1()
        dgamma2 = self.gammadash2()

        # compute the average cross sectional area
        J = np.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
        J[:, :, 0, 0] = (1. / (2. * np.pi))*(xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0])/x2y2
        J[:, :, 0, 1] = (1. / (2. * np.pi))*(xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0])/x2y2
        J[:, :, 1, 0] = 0.
        J[:, :, 1, 1] = 1.

        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)

        dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
        mean_cross_sectional_area = np.abs(np.mean(np.sqrt(x2y2) * dZ_dtheta * detJ)) 

        R_minor = np.sqrt(mean_cross_sectional_area / np.pi)
        R_major = np.abs(self.volume()) / (2. * np.pi**2 * R_minor**2)

        AR = R_major/R_minor
        return AR
