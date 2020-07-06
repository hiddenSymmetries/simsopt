import os
import logging
import numpy as np
from scipy.io import netcdf

class VmecOutput:

    def __init__(self, wout_filename, ntheta=100, nzeta=100):
        self.wout_filename = wout_filename
        self.input_filename = 'input.' + wout_filename.split('.')[0][5::]
        self.directory = os.getcwd()
           
        f = netcdf.netcdf_file(self.wout_filename, 'r', mmap=False)
        self.rmnc = f.variables["rmnc"][()]
        self.zmns = f.variables["zmns"][()]
        self.bsubumnc = f.variables["bsubumnc"][()]
        self.bsubvmnc = f.variables["bsubvmnc"][()]
        self.bsupumnc = f.variables["bsupumnc"][()]
        self.bsupvmnc = f.variables["bsupvmnc"][()]
        self.xm = f.variables["xm"][()]
        self.xn = f.variables["xn"][()]
        self.mnmax = f.variables["mnmax"][()]
        self.gmnc = f.variables["gmnc"][()]
        self.psi = f.variables["phi"][()]/(2*np.pi)
        self.mpol = f.variables["mpol"][()]
        self.ntor = f.variables["ntor"][()]
        self.iota = f.variables["iotas"][()]
        self.vp = f.variables["vp"][()]
        self.pres = f.variables["pres"][()]
        self.volume = f.variables["volume_p"][()]
        
        # Remove axis point from half grid quantities
        self.bsubumnc = np.delete(self.bsubumnc, 0, 0)
        self.bsubvmnc = np.delete(self.bsubvmnc, 0, 0)
        self.bsupumnc = np.delete(self.bsupumnc, 0, 0)
        self.bsupvmnc = np.delete(self.bsupvmnc, 0, 0)
        self.gmnc = np.delete(self.gmnc, 0, 0)
        self.iota = np.delete(self.iota, 0)
        self.vp = np.delete(self.vp, 0)
        self.pres = np.delete(self.pres, 0)
        
        self.s_full = self.psi / self.psi[-1]
        self.ds = self.s_full[1] - self.s_full[0]
        self.s_half = self.s_full - 0.5*self.ds
        self.s_half = np.delete(self.s_half, 0)
        self.ns = len(self.s_full)
        self.ns_half = len(self.s_half)
        
        self.nfp = f.variables["nfp"][()]
        self.sign_jac = f.variables["signgs"][()]
        
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.thetas = np.linspace(0, 2*np.pi, ntheta+1)
        self.zetas = np.linspace(0, 2*np.pi/self.nfp, nzeta+1)
        self.thetas = np.delete(self.thetas, -1)
        self.zetas = np.delete(self.zetas, -1)
        [self.thetas_2d, self.zetas_2d] = np.meshgrid(self.thetas, self.zetas)
        self.dtheta = self.thetas[1] - self.thetas[0]
        self.dzeta = self.zetas[1] - self.zetas[0]
        
        self.zetas_full = np.linspace(0, 2*np.pi, self.nfp * nzeta + 1)
        self.zetas_full = np.delete(self.zetas_full, -1)
        [self.thetas_2d_full, self.zetas_2d_full] = np.meshgrid(
                self.thetas, self.zetas_full)
        
        self.mu0 = 4*np.pi*1.0e-7
        
    def d_r_d_u(self, isurf=-1):
        this_rmnc = self.rmnc[isurf,:]
        this_zmns = self.zmns[isurf,:]
        
        dRdu = np.zeros(np.shape(self.thetas_2d))
        dZdu = np.zeros(np.shape(self.thetas_2d))
        for im in range(self.mnmax):
            angle = self.xm[im] * self.thetas_2d - self.xn[im] * self.zetas_2d
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            dRdu -= self.xm[im] * this_rmnc[im] * sin_angle
            dZdu += self.xm[im] * this_zmns[im] * cos_angle
        dXdu = dRdu*np.cos(self.zetas_2d)
        dYdu = dRdu*np.sin(self.zetas_2d)
        return dXdu, dYdu, dZdu
  
    def d_r_d_v(self, isurf=-1):
        this_rmnc = self.rmnc[isurf,:]
        this_zmns = self.zmns[isurf,:]
        
        dRdv = np.zeros(np.shape(self.thetas_2d))
        dZdv = np.zeros(np.shape(self.thetas_2d))
        R = np.zeros(np.shape(self.thetas_2d))
        for im in range(self.mnmax):
            angle = self.xm[im] * self.thetas_2d - self.xn[im] * self.zetas_2d
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            dRdv = dRdv + self.xn[im] * this_rmnc[im] * sin_angle
            dZdv = dZdv - self.xn[im] * this_zmns[im] * cos_angle
            R = R + this_rmnc[im] * cos_angle
        dXdv = dRdv*np.cos(self.zetas_2d) - R*np.sin(self.zetas_2d)
        dYdv = dRdv*np.sin(self.zetas_2d) + R*np.cos(self.zetas_2d)
        return dXdv, dYdv, dZdv
  
    def B_on_arclength_grid(self):
        # Extrapolate to last full mesh grid point
        bsupumnc_end = 1.5*self.bsupumnc[-1,:] - 0.5*self.bsupumnc[-2,:]
        bsupvmnc_end = 1.5*self.bsupvmnc[-1,:] - 0.5*self.bsupvmnc[-2,:]
        rmnc_end = self.rmnc[-1,:]
        zmns_end = self.zmns[-1,:]
          
        # Evaluate at last full mesh grid point
        [dXdu_end, dYdu_end, dZdu_end] = self.d_r_d_u(-1)
        [dXdv_end, dYdv_end, dZdv_end] = self.d_r_d_v(-1)
        
        Bsupu_end = np.zeros(np.shape(self.thetas_2d))
        Bsupv_end = np.zeros(np.shape(self.thetas_2d))
        R_end = np.zeros(np.shape(self.thetas_2d))
        Z_end = np.zeros(np.shape(self.thetas_2d))
        for im in range(self.mnmax):
            angle = self.xm[im] * self.thetas_2d - self.xn[im] * self.zetas_2d
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            Bsupu_end = Bsupu_end + bsupumnc_end[im] * cos_angle
            Bsupv_end = Bsupv_end + bsupvmnc_end[im] * cos_angle
            R_end = R_end + rmnc_end[im] * cos_angle
            Z_end = Z_end + zmns_end[im] * sin_angle
        Bx_end = Bsupu_end * dXdu_end + Bsupv_end * dXdv_end
        By_end = Bsupu_end * dYdu_end + Bsupv_end * dYdv_end
        Bz_end = Bsupu_end * dZdu_end + Bsupv_end * dZdv_end

        theta_arclength = np.zeros(np.shape(self.zetas_2d))
        for izeta in range(self.nzeta):
            for itheta in range(1, self.ntheta):
                dr = np.sqrt((R_end[izeta, itheta] - R_end[izeta, itheta-1])**2 \
                        + (Z_end[izeta, itheta] - Z_end[izeta, itheta-1])**2)
                theta_arclength[izeta, itheta] = \
                        theta_arclength[izeta, itheta-1] + dr
        
        return Bx_end, By_end, Bz_end, theta_arclength

    def compute_current(self):
        It_half = self.sign_jac * 2*np.pi * self.bsubumnc[:,0] / self.mu0
        return It_half

    def compute_N(self, isurf=-1, theta=None, zeta=None):  
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta] = \
          self.position_first_derivatives(isurf, theta, zeta)
      
        Nx = -dydzeta*dzdtheta + dydtheta*dzdzeta
        Ny = -dzdzeta*dxdtheta + dzdtheta*dxdzeta
        Nz = -dxdzeta*dydtheta + dxdtheta*dydzeta
        return Nx, Ny, Nz
  
    def compute_N_derivatives(self, isurf=-1, theta=None, zeta=None):
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta] = \
                self.position_first_derivatives(isurf, theta, zeta)
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta] = \
                self.position_second_derivatives(isurf, theta, zeta)

        dNxdtheta = -d2ydthetadzeta * dzdtheta - dydzeta * d2zdtheta2 \
                + d2ydtheta2 * dzdzeta + dydtheta * d2zdthetadzeta
        dNxdzeta = -d2ydzeta2 * dzdtheta - dydzeta*d2zdthetadzeta \
                + d2ydthetadzeta*dzdzeta + dydtheta*d2zdzeta2
        dNydtheta = -d2zdthetadzeta * dxdtheta - dzdzeta * d2xdtheta2 \
                + d2zdtheta2 * dxdzeta + dzdtheta * d2xdthetadzeta 
        dNydzeta = -d2zdzeta2 * dxdtheta - dzdzeta * d2xdthetadzeta \
                + d2zdthetadzeta * dxdzeta + dzdtheta * d2xdzeta2 
        dNzdtheta = -d2xdthetadzeta * dydtheta - dxdzeta * d2ydtheta2 \
                + d2xdtheta2 * dydzeta + dxdtheta * d2ydthetadzeta
        dNzdzeta = -d2xdzeta2 * dydtheta - dxdzeta * d2ydthetadzeta \
                + d2xdthetadzeta * dydzeta + dxdtheta * d2ydzeta2
        return dNxdtheta, dNxdzeta, dNydtheta, dNydzeta, dNzdtheta, dNzdzeta
  
    def compute_n_derivatives(self, isurf=-1):
        [Nx,Ny,Nz] = self.compute_N(isurf)
        [dNxdtheta, dNxdzeta, dNydtheta, dNydzeta, dNzdtheta, dNzdzeta] = \
          self.compute_N_derivatives(isurf)
        norm_normal = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz)
        dnorm_normaldtheta = (Nx*dNxdtheta + Ny*dNydtheta + Nz*dNzdtheta)/norm_normal
        dnorm_normaldzeta = (Nx*dNxdzeta + Ny*dNydzeta + Nz*dNzdzeta)/norm_normal
        dnxdtheta = dNxdtheta/norm_normal - Nx*dnorm_normaldtheta/(norm_normal**2)
        dnxdzeta = dNxdzeta/norm_normal - Nx*dnorm_normaldzeta/(norm_normal**2)
        dnydtheta = dNydtheta/norm_normal - Ny*dnorm_normaldtheta/(norm_normal**2)
        dnydzeta = dNydzeta/norm_normal - Ny*dnorm_normaldzeta/(norm_normal**2)
        dnzdtheta = dNzdtheta/norm_normal - Nz*dnorm_normaldtheta/(norm_normal**2)
        dnzdzeta = dNzdzeta/norm_normal - Nz*dnorm_normaldzeta/(norm_normal**2)
        return dnxdtheta, dnxdzeta, dnydtheta, dnydzeta, dnzdtheta, dnzdzeta
  
    # Returns X, Y, Z at isurf point on full mesh 
    def compute_position(self, isurf=-1, full=False):
        this_rmnc = self.rmnc[isurf,:]
        this_zmns = self.zmns[isurf,:] 

        if (full):
          theta = self.thetas_2d_full
          zeta = self.zetas_2d_full
        else:
          theta = self.thetas_2d
          zeta = self.zetas_2d
        R = np.zeros(np.shape(theta))
        Z = np.zeros(np.shape(zeta))
        for im in range(self.mnmax):
          angle = self.xm[im] * theta - self.xn[im] * zeta
          cos_angle = np.cos(angle)
          sin_angle = np.sin(angle)
          R += this_rmnc[im] * cos_angle
          Z += this_zmns[im] * sin_angle
        X = R * np.cos(zeta)
        Y = R * np.sin(zeta)
        return X, Y, Z, R

    # Integrated rotational transform with weight function
    def evaluate_iota_objective(self, weight):
        return np.sum(weight(self.s_half) * self.iota) * self.ds * \
                self.psi[-1] * self.sign_jac
  
    # Integrated differential volume with weight function
    def evaluate_well_objective(self, weight):
        return 4*np.pi*np.pi*np.sum(weight(self.s_half) * self.vp) * self.ds
  
    def jacobian(self, isurf=-1, theta=None, zeta=None):
        [Nx,Ny,Nz] = self.compute_N(isurf,theta=theta, zeta=zeta)
        norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        return norm_normal
  
    def normalized_jacobian(self, isurf=-1, theta=None, zeta=None):
        norm_normal = self.jacobian(isurf, theta, zeta)
        area = self.area(isurf, theta, zeta)
        return norm_normal * 4*np.pi*np.pi / area
  
    def area(self, isurf=-1, theta=None, zeta=None):
        [Nx, Ny, Nz] = self.compute_N(isurf, theta, zeta)
        norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        area = np.sum(norm_normal) * self.dtheta * self.dzeta * self.nfp
        return area
  
    def position_first_derivatives(self, isurf=-1, theta=None, zeta=None):
        logger = logging.getLogger(__name__)
        this_rmnc = self.rmnc[isurf,:]
        this_zmns = self.zmns[isurf,:] 
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            logger.error('Incorrect shape of theta and zeta in '
                         'position_first_derivatives')
            sys.exit(0)
        dRdtheta = np.zeros(np.shape(theta))
        dzdtheta = np.zeros(np.shape(theta))
        dRdzeta = np.zeros(np.shape(theta))
        dzdzeta = np.zeros(np.shape(theta))
        R = np.zeros(np.shape(theta))
        for im in range(self.mnmax):
            angle = self.xm[im] * theta - self.xn[im] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            dRdtheta -= self.xm[im] * this_rmnc[im] * sin_angle
            dzdtheta += self.xm[im] * this_zmns[im] * cos_angle
            dRdzeta += self.xn[im] * this_rmnc[im] * sin_angle
            dzdzeta -= self.xn[im] * this_zmns[im] * cos_angle
            R += this_rmnc[im] * cos_angle
        dxdtheta = dRdtheta * np.cos(zeta)
        dydtheta = dRdtheta * np.sin(zeta)
        dxdzeta = dRdzeta * np.cos(zeta) - R * np.sin(zeta)
        dydzeta = dRdzeta * np.sin(zeta) + R * np.cos(zeta)
        return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta
  
    def position_second_derivatives(self, isurf=-1, theta=None, zeta=None):
        logger = logging.getLogger(__name__)
        this_rmnc = self.rmnc[isurf,:]
        this_zmns = self.zmns[isurf,:] 
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            logger.error('Incorrect shape of theta and zeta in '
                         'position_first_derivatives')
            sys.exit(0)
       
        d2Rdtheta2 = np.zeros(np.shape(theta))
        d2Rdzeta2 = np.zeros(np.shape(theta))
        d2Zdtheta2 = np.zeros(np.shape(theta))
        d2Zdzeta2 = np.zeros(np.shape(theta))
        d2Rdthetadzeta = np.zeros(np.shape(theta))
        d2Zdthetadzeta = np.zeros(np.shape(theta))
        dRdtheta = np.zeros(np.shape(theta))
        dzdtheta = np.zeros(np.shape(theta))
        dRdzeta = np.zeros(np.shape(theta))
        dzdzeta = np.zeros(np.shape(theta))
        R = np.zeros(np.shape(theta))

        for im in range(self.mnmax):
            angle = self.xm[im] * theta - self.xn[im] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            R = R + this_rmnc[im] * cos_angle
            d2Rdtheta2 -= self.xm[im] * self.xm[im] * this_rmnc[im] * cos_angle
            d2Zdtheta2 -= self.xm[im] * self.xm[im] * this_zmns[im] * sin_angle
            d2Rdzeta2 -= self.xn[im] * self.xn[im] * this_rmnc[im] * cos_angle
            d2Zdzeta2 -= self.xn[im] * self.xn[im] * this_zmns[im] * sin_angle
            d2Rdthetadzeta += self.xm[im] * self.xn[im] * this_rmnc[im] * cos_angle
            d2Zdthetadzeta += self.xm[im] * self.xn[im] * this_zmns[im] * sin_angle
            dRdtheta -= self.xm[im] * this_rmnc[im] * sin_angle
            dzdtheta += self.xm[im] * this_zmns[im] * cos_angle
            dRdzeta += self.xn[im] * this_rmnc[im] * sin_angle
            dzdzeta -= self.xn[im] * this_zmns[im] * cos_angle

        d2xdtheta2 = d2Rdtheta2 * np.cos(zeta)
        d2ydtheta2 = d2Rdtheta2 * np.sin(zeta)
        d2xdzeta2 = d2Rdzeta2 * np.cos(zeta) \
                - 2*dRdzeta * np.sin(zeta) - R * np.cos(zeta)
        d2ydzeta2 = d2Rdzeta2 * np.sin(zeta) \
                + 2*dRdzeta * np.cos(zeta) - R * np.sin(zeta)
        d2xdthetadzeta = d2Rdthetadzeta * np.cos(zeta) - dRdtheta * np.sin(zeta)
        d2ydthetadzeta = d2Rdthetadzeta * np.sin(zeta) + dRdtheta * np.cos(zeta)
        return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta
    
    def mean_curvature(self, isurf=-1):
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta] = \
                self.position_first_derivatives(isurf)
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta] = \
                self.position_second_derivatives(isurf)
        
        norm_x = dydtheta * dZdzeta - dydzeta * dZdtheta
        norm_y = dZdtheta * dxdzeta - dZdzeta * dxdtheta
        norm_z = dxdtheta * dydzeta - dxdzeta * dydtheta
        norm_normal = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
        nx = norm_x / norm_normal
        ny = norm_y / norm_normal
        nz = norm_z / norm_normal
        E = dxdtheta * dxdtheta + dydtheta * dydtheta + dZdtheta * dZdtheta
        F = dxdtheta * dxdzeta + dydtheta * dydzeta + dZdtheta * dZdzeta
        G = dxdzeta * dxdzeta + dydzeta * dydzeta + dZdzeta * dZdzeta
        e = nx * d2xdtheta2 + ny * d2ydtheta2 + nz * d2Zdtheta2
        f = nx * d2xdthetadzeta + ny * d2ydthetadzeta + nz * d2Zdthetadzeta
        g = nx * d2xdzeta2 + ny * d2ydzeta2 + nz * d2Zdzeta2
        H = (e*G - 2*f*F + g*E) / (E*G - F*F)
        return H
  
    # Check that theta and zeta are of correct size
    # Shape of theta and zeta must be the same
    def jacobian_derivatives(self, xm_sensitivity, xn_sensitivity, \
                             theta=None, zeta=None):
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta '
                         'and zeta in jacobian_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta '
                         'and zeta in jacobian_derivatives.')
            sys.exit(0)

        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta] = \
                self.position_first_derivatives(-1, theta, zeta)
        [Nx, Ny, Nz] = self.compute_N(-1, theta, zeta)
        N = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz)
        nx = Nx / N
        ny = Ny / N
        nz = Nz / N
        
        mnmax_sensitivity = len(xm_sensitivity)
        
        d2rdthetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdthetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))

        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn] * theta \
                    - self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            cos_zeta = np.cos(zeta)
            sin_zeta = np.sin(zeta)
            d2rdthetadrmnc[0,imn,:,:] = -xm_sensitivity[imn] \
                    * sin_angle * cos_zeta
            d2rdthetadrmnc[1,imn,:,:] = -xm_sensitivity[imn] \
                    * sin_angle * sin_zeta
            d2rdzetadrmnc[0,imn,:,:] = self.nfp * xn_sensitivity[imn] \
                    * sin_angle * cos_zeta - cos_angle*sin_zeta
            d2rdzetadrmnc[1,imn,:,:] = self.nfp * xn_sensitivity[imn] \
                    * sin_angle * sin_zeta + cos_angle * cos_zeta
            d2rdthetadzmns[2,imn,:,:] = xm_sensitivity[imn] * cos_angle
            d2rdzetadzmns[2,imn,:,:] = -self.nfp * xn_sensitivity[imn] \
                    * cos_angle

        dNdrmnc = (d2rdthetadrmnc[1,:,:,:]*dzdzeta - d2rdthetadrmnc[2,:,:,:]*dydzeta)*nx \
                + (d2rdthetadrmnc[2,:,:,:]*dxdzeta - d2rdthetadrmnc[0,:,:,:]*dzdzeta)*ny \
                + (d2rdthetadrmnc[0,:,:,:]*dydzeta - d2rdthetadrmnc[1,:,:,:]*dxdzeta)*nz \
                + (dydtheta*d2rdzetadrmnc[2,:,:,:] - dzdtheta*d2rdzetadrmnc[1,:,:,:])*nx \
                + (dzdtheta*d2rdzetadrmnc[0,:,:,:] - dxdtheta*d2rdzetadrmnc[2,:,:,:])*ny \
                + (dxdtheta*d2rdzetadrmnc[1,:,:,:] - dydtheta*d2rdzetadrmnc[0,:,:,:])*nz
        dNdzmns = (d2rdthetadzmns[1,:,:,:]*dzdzeta - d2rdthetadzmns[2,:,:,:]*dydzeta)*nx \
                + (d2rdthetadzmns[2,:,:,:]*dxdzeta - d2rdthetadzmns[0,:,:,:]*dzdzeta)*ny \
                + (d2rdthetadzmns[0,:,:,:]*dydzeta - d2rdthetadzmns[1,:,:,:]*dxdzeta)*nz \
                + (dydtheta*d2rdzetadzmns[2,:,:,:] - dzdtheta*d2rdzetadzmns[1,:,:,:])*nx \
                + (dzdtheta*d2rdzetadzmns[0,:,:,:] - dxdtheta*d2rdzetadzmns[2,:,:,:])*ny \
                + (dxdtheta*d2rdzetadzmns[1,:,:,:] - dydtheta*d2rdzetadzmns[0,:,:,:])*nz
        return dNdrmnc, dNdzmns

    def radius_derivatives(self, xm_sensitivity, xn_sensitivity, 
                           theta=None, zeta=None):
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta '
                         'and zeta in radius_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta '
                         'and zeta in radius_derivatives.')
            sys.exit(0)
          
        mnmax_sensitivity = len(xm_sensitivity)
        
        dRdrmnc = np.zeros((mnmax_sensitivity, dim1, dim2))
        dRdzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn] * theta \
                    - self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            dRdrmnc[imn,:,:] = cos_angle
        return dRdrmnc, dRdzmns
  
    def area_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, \
                         zeta=None):
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                xm_sensitivity, xn_sensitivity, theta, zeta)
        dareadrmnc = np.sum(dNdrmnc, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        dareadzmns = np.sum(dNdzmns, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        return dareadrmnc, dareadzmns
  
    def normalized_jacobian_derivatives(self, xm_sensitivity, xn_sensitivity, \
                                        theta=None, zeta=None):
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta '
                         'and zeta in normalized_jacobian_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta '
                         'and zeta in normalized_jacobian_derivatives.')
            sys.exit(0)

        [dareadrmnc, dareadzmns] = self.area_derivatives(
                xm_sensitivity, xn_sensitivity, theta=theta, zeta=zeta)
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                xm_sensitivity, xn_sensitivity, theta=theta, zeta=zeta)
        area = self.area(theta=theta, zeta=zeta)
        N = self.jacobian(theta=theta, zeta=zeta)
        
        mnmax_sensitivity = len(xm_sensitivity)
        dnormalized_jacobiandrmnc = np.zeros((mnmax_sensitivity, dim1, dim2))
        dnormalized_jacobiandzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            dnormalized_jacobiandrmnc[imn,:,:] = dNdrmnc[imn,:,:] / area \
                    - N * dareadrmnc[imn] / (area * area)
            dnormalized_jacobiandzmns[imn,:,:] = dNdzmns[imn,:,:] / area \
                    - N * dareadzmns[imn] / (area * area)
        
        return 4*np.pi*np.pi*dnormalized_jacobiandrmnc, \
                4*np.pi*np.pi*dnormalized_jacobiandzmns
      
    
    
