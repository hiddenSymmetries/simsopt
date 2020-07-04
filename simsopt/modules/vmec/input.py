# coding: utf-8

import os
import sys
import logging

import numpy as np
import scipy
from scipy.io import netcdf
import optimization_utils
import f90nml
from FortranNamelistTools import *  # TODO: Change this import
from optimization_utils import *    # TODO: Change this import

__author__ = "Elizabeth Paul, Bharat Medasani"
#__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "0.0.1"
__maintainer__ = "Bharat Medasani"
__email__ = "mbkumar@gmail.com"
__status__ = "Preliminary"
__date__ = "Jul 03, 2020"


class VmecInput:
  
    #def __init__(self, ntheta=100, nzeta=100, **kwargs):
    #    self.__dict__.update(kwargs)
    #    self.update_grids(ntheta, nzeta)
    #    self.min_curvature_radius = 0.3

    #@classmethod
    #def from_file(cls, filename):
    def __init__(self, input_filename, ntheta=100, nzeta=100):
        """
        Reads VMEC input parameters from a VMEC input file. 

        Args:
            input_filename (str): The filename to read from.
            ntheta (int): Theta intervals
            nzeta (int): Zeta intervals

        Returns:
            VMECInput.
        """
        logger = logging.getLogger(__name__)
        self.input_filename = input_filename
        self.directory = os.getcwd()

        # Read items from Fortran namelist
        nml = f90nml.read(filename)
        nml = nml.get("indata")
        self.nfp = nml.get("nfp")
        mpol_input = nml.get("mpol")
        if (mpol_input < 2):
            logger.error('Error! mpol must be > 1.')
            sys.exit(1)
        self.namelist = nml
        
        self.raxis = nml.get("raxis")
        self.zaxis = nml.get("zaxis")
        self.curtor = nml.get("curtor")
        self.ac_aux_f = nml.get("ac_aux_f")
        self.ac_aux_s = nml.get("ac_aux_s")
        self.pcurr_type = nml.get("pcurr_type")
        self.am_aux_f = nml.get("am_aux_f")
        self.am_aux_s = nml.get("am_aux_s")
        self.pmass_type = nml.get("pmass_type")
        
        self.update_modes(nml.get("mpol"), nml.get("ntor"))

        self.update_grids(ntheta, nzeta)
        self.min_curvature_radius = 0.3

    # updates self.namelist with attributes (except rbc/zbs)
    def update_namelist(self):
        """
        TODO: doc string
        """
        self.namelist['nfp'] = self.nfp
        self.namelist['raxis'] = self.raxis
        self.namelist['zaxis'] = self.zaxis
        self.namelist['curtor'] = self.curtor
        self.namelist['ac_aux_f'] = self.ac_aux_f
        self.namelist['ac_aux_s'] = self.ac_aux_s
        self.namelist['pcurr_type'] = self.pcurr_type
        self.namelist['am_aux_f'] = self.am_aux_f
        self.namelist['am_aux_s'] = self.am_aux_s
        self.namelist['pmass_type'] = self.pmass_type
        self.namelist['mpol'] = self.mpol
        self.namelist['ntor'] = self.ntor

    def update_modes(self, mpol, ntor):
        """
        TODO: doc string
        """
        self.mpol = mpol
        self.ntor = ntor
        [self.mnmax, self.xm, self.xn] = optimization_utils.init_modes(
                self.mpol - 1, self.ntor)
        [self.rbc,self.zbs] = self.read_boundary_input()
        
    def update_grids(self,ntheta,nzeta):
        """
        TODO: doc string
        """
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.thetas = np.linspace(0, 2 * np.pi, ntheta + 1)
        self.zetas = np.linspace(0,2 * np.pi / self.nfp, nzeta + 1)
        self.thetas = np.delete(self.thetas, -1)
        self.zetas = np.delete(self.zetas, -1)
        [self.thetas_2d, self.zetas_2d] = np.meshgrid(self.thetas, self.zetas)
        self.dtheta = self.thetas[1] - self.thetas[0]
        self.dzeta = self.zetas[1] - self.zetas[0]
        
        self.zetas_full = np.linspace(0, 2 * np.pi, self.nfp * nzeta + 1)
        self.zetas_full = np.delete(self.zetas_full, -1)
        [self.thetas_2d_full, self.zetas_2d_full] = np.meshgrid(
                self.thetas, self.zetas_full)
        
    def read_boundary_input(self):
        """
        Read in boundary harmonics (rbc, zbs) from Fortran namelist

        Parameters
        ----------
        xm (int array) : poloidal mode numbers to read
        xn (int array) : toroidal mode numbers to read

        Returns
        ----------
        rbc (float array) : boundary harmonics for radial coordinate
                    (same size as xm and xn)
        zbs (float array) : boundary harmonics for height coordinate
                    (same size as xm and xn)

        """
        rbc_array = np.array(self.namelist["rbc"]).T
        zbs_array = np.array(self.namelist["zbs"]).T

        rbc_array[rbc_array == None] = 0
        zbs_array[zbs_array == None] = 0
        [dim1_rbc,dim2_rbc] = np.shape(rbc_array)
        [dim1_zbs,dim2_zbs] = np.shape(zbs_array)

        [nmin_rbc, mmin_rbc, nmax_rbc, mmax_rbc] = min_max_indices_2d(
                'rbc', self.input_filename)
        nmax_rbc = max(nmax_rbc, -nmin_rbc)
        [nmin_zbs, mmin_zbs, nmax_zbs, mmax_zbs] = min_max_indices_2d(
                'zbs', self.input_filename)
        nmax_zbs = max(nmax_zbs, -nmin_zbs)

        rbc = np.zeros(self.mnmax)
        zbs = np.zeros(self.mnmax)
        for imn in range(self.mnmax):
            if (((self.xn[imn] - nmin_rbc) < dim1_rbc) and \
                    ((self.xm[imn] - mmin_rbc) < dim2_rbc) and \
                    ((self.xn[imn] - nmin_rbc) > -1) and \
                    ((self.xm[imn] - mmin_rbc) > -1)):
                rbc[imn] = rbc_array[int(self.xn[imn] - nmin_rbc), \
                                     int(self.xm[imn] - mmin_rbc)]
            if (((self.xn[imn] - nmin_zbs) < dim1_zbs) and \
                    ((self.xm[imn] - mmin_zbs) < dim2_zbs) and \
                    ((self.xn[imn] - nmin_rbc) > -1) and \
                    ((self.xm[imn] - mmin_rbc) > -1)):
                zbs[imn] = zbs_array[int(self.xn[imn] - nmin_zbs), \
                                     int(self.xm[imn]-mmin_zbs)]

        return rbc, zbs

    def area(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        norm_normal = self.jacobian(theta, zeta)
        area = np.sum(norm_normal) * self.dtheta * self.dzeta * self.nfp
        return area
      
    def volume(self,theta=None,zeta=None):
        """
        TODO: doc string
        """
        [Nx, Ny, Nz] = self.N(theta, zeta)
        [X, Y, Z, R] = self.position(theta, zeta)
        return abs(np.sum(Z * Nz)) * self.dtheta * self.dzeta * self.nfp
      
    def N(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                    dRdzeta] = self.position_first_derivatives(theta,zeta)
      
        Nx = -dydzeta * dzdtheta + dydtheta * dzdzeta
        Ny = -dzdzeta * dxdtheta + dzdtheta * dxdzeta
        Nz = -dxdzeta * dydtheta + dxdtheta * dydzeta
        return Nx, Ny, Nz
      
    def position_first_derivatives(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            logger.error("Incorrect shape of theta and zeta in "
                         "position_first_derivatives")
            sys.exit(0)
        dRdtheta = np.zeros(np.shape(theta))
        dzdtheta = np.zeros(np.shape(theta))
        dRdzeta = np.zeros(np.shape(theta))
        dzdzeta = np.zeros(np.shape(theta))
        R = np.zeros(np.shape(theta))
        for im in range(self.mnmax):
            angle = self.xm[im] * theta - self.nfp * self.xn[im] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            dRdtheta = dRdtheta - self.xm[im] * self.rbc[im] * sin_angle
            dzdtheta = dzdtheta + self.xm[im] * self.zbs[im] * cos_angle
            dRdzeta = dRdzeta + self.nfp * self.xn[im] * self.rbc[im] * sin_angle
            dzdzeta = dzdzeta - self.nfp * self.xn[im] * self.zbs[im] * cos_angle
            R = R + self.rbc[im] * cos_angle
        dxdtheta = dRdtheta * np.cos(zeta)
        dydtheta = dRdtheta * np.sin(zeta)
        dxdzeta = dRdzeta * np.cos(zeta) - R * np.sin(zeta)
        dydzeta = dRdzeta * np.sin(zeta) + R * np.cos(zeta)
        return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta

    def jacobian(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        [Nx, Ny, Nz] = self.N(theta, zeta)
        norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        return norm_normal
      
    def normalized_jacobian(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        norm_normal = self.jacobian(theta, zeta)
        area = self.area(theta, zeta)
        return norm_normal * 4*np.pi * np.pi / area
      
    def normalized_jacobian_derivatives(self, xm_sensitivity, xn_sensitivity,
                                        theta=None, zeta=None):
        """
        TODO: doc string
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:, 0])
            dim2 = len(theta[0, :])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
            sys.exit(0)
        [dareadrmnc, dareadzmns] = self.area_derivatives(
                    xm_sensitivity, xn_sensitivity, theta, zeta)
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                    xm_sensitivity, xn_sensitivity, theta, zeta)
        area = self.area(theta, zeta)
        N = self.jacobian(theta, zeta)
        
        mnmax_sensitivity = len(xm_sensitivity)
        dnormalized_jacobiandrmnc = np.zeros((mnmax_sensitivity, dim1, dim2))
        dnormalized_jacobiandzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            dnormalized_jacobiandrmnc[imn, :, :] = dNdrmnc[imn]/area - \
                    N * dareadrmnc[imn] / (area * area)
            dnormalized_jacobiandzmns[imn, :, :] = dNdzmns[imn]/area - \
                    N * dareadzmns[imn] / (area * area)
        
        return 4 *np.pi * np.pi * dnormalized_jacobiandrmnc, \
                    4 * np.pi * np.pi * dnormalized_jacobiandzmns
        
    # Returns X, Y, Z at isurf point on full mesh 
    def position(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            logger.error("Incorrect shape of theta and zeta in "
                         "position_first_derivatives")
            sys.exit(0)
      
        R = np.zeros(np.shape(theta))
        Z = np.zeros(np.shape(zeta))
        for im in range(self.mnmax):
            angle = self.xm[im] * theta - self.nfp * self.xn[im] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            R = R + self.rbc[im] * cos_angle
            Z = Z + self.zbs[im] * sin_angle
        X = R * np.cos(zeta)
        Y = R * np.sin(zeta)
        return X, Y, Z, R
      
    def position_second_derivatives(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        logger = logging.getLogger(__name__)
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
            angle = self.xm[im]*theta - self.nfp*self.xn[im]*zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            R = R + this_rmnc[im]*cos_angle
            d2Rdtheta2 -= self.xm[im] * self.xm[im] * self.rbc[im] * cos_angle
            d2Zdtheta2 -= self.xm[im] * self.xm[im] * self.zbs[im] * sin_angle
            d2Rdzeta2 -= self.nfp * self.nfp * self.xn[im] * self.xn[im] * \
                    self.rbc[im] * cos_angle
            d2Zdzeta2 -= self.nfp * self.nfp * self.xn[im] * self.xn[im] * \
                    self.zbs[im] * sin_angle
            d2Rdthetadzeta += self.nfp * self.xm[im] * self.xn[im] * \
                    self.rbc[im] * cos_angle
            d2Zdthetadzeta += self.nfp * self.xm[im] * self.xn[im] * \
                    self.zbs[im] * sin_angle
            dRdtheta -= self.xm[im] * self.rbc[im] * sin_angle
            dzdtheta += self.xm[im] * self.zbs[im] * cos_angle
            dRdzeta += self.nfp * self.xn[im] * self.rbc[im] * sin_angle
            dzdzeta -= self.nfp * self.xn[im] * self.zbs[im] * cos_angle
        d2xdtheta2 = d2Rdtheta2 * np.cos(zeta)
        d2ydtheta2 = d2Rdtheta2 * np.sin(zeta)
        d2xdzeta2 = d2Rdzeta2 * np.cos(zeta) - 2 * dRdzeta * np.sin(zeta) - \
                R * np.cos(zeta)
        d2ydzeta2 = d2Rdzeta2 * np.sin(zeta) + 2 * dRdzeta * np.cos(zeta) - \
                R * np.sin(zeta)
        d2xdthetadzeta = d2Rdthetadzeta * np.cos(zeta) - dRdtheta * np.sin(zeta)
        d2ydthetadzeta = d2Rdthetadzeta * np.sin(zeta) + dRdtheta * np.cos(zeta)
        return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta
        
    def mean_curvature(self, theta=None, zeta=None):
        """
        TODO: doc string
        """
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta] = \
                self.position_second_derivatives(theta, zeta)
        
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
        H = (e*G -2*f*F + g*E)/(E*G - F*F)
        return H    
      
    def print_namelist(self, input_filename=None, namelist=None):
        """
        TODO: doc string
        """
        self.update_namelist()
        if (namelist is None):
            namelist = self.namelist
        if (input_filename is None):
            input_filename = self.input_filename
        f = open(input_filename, 'w')
        f.write("&INDATA\n")
        for item in namelist.keys():
            if namelist[item] is not None:
                if (not isinstance(namelist[item], 
                                   (list, tuple, np.ndarray, str))):
                    f.write(item + "=" + str(namelist[item]) + "\n")
                elif (isinstance(namelist[item], str)):
                    f.write(item + "=" + "'" + str(namelist[item]) + "'" + "\n")
                elif (item not in ['rbc', 'zbs']):
                    f.write(item + "=")
                    f.writelines(map(lambda x: str(x) + " ", namelist[item]))
                    f.write("\n")
        # Write RBC
        for imn in range(self.mnmax):
          if (self.rbc[imn] != 0):
            f.write('RBC(' + str(int(self.xn[imn])) + "," \
                     + str(int(self.xm[imn])) + ") = " \
                     + str(self.rbc[imn]) + '\n')
        # Write ZBS
        for imn in range(self.mnmax):
          if (self.zbs[imn] != 0):
            f.write('ZBS(' + str(int(self.xn[imn])) + "," \
                     + str(int(self.xm[imn])) + ") = " \
                     + str(self.zbs[imn]) + '\n')

        f.write("/\n")
        f.close()
        
    def axis_position(self):
        """
        TODO: doc string
        """
        R0 = np.zeros(self.nzeta)
        Z0 = np.zeros(self.nzeta)
        
        if isinstance(self.raxis, (np.ndarray, list)):
            length = len(self.raxis)
        else:
            length = 1
            self.raxis = [self.raxis]
        for im in range(length):
            angle = -self.nfp * im * self.zetas
            R0 += self.raxis[im] * np.cos(angle)

        if isinstance(self.zaxis, (np.ndarray, list)):
            length = len(self.zaxis)
        else:
            length = 1
            self.zaxis = [self.zaxis]

        for im in range(length):
            angle = -self.nfp * im * self.zetas
            Z0 += self.zaxis[im] * np.sin(angle)
     
        return R0, Z0
      
      # Given (R,Z) defining boundary shape, determine if (R0,Z0) lies inside boundary
    def test_axis(self):
        """
        TODO: doc string
        """
        [X, Y, Z, R] = self.position()
        [R0, Z0] = self.axis_position()
        
        inSurface = np.zeros(self.nzeta)
        for izeta in range(self.nzeta):
            inSurface[izeta] = point_in_polygon(R[izeta, :], Z[izeta, :], 
                                                R0[izeta], Z0[izeta])
        return np.all(inSurface)
      
    def modify_axis(self):
        logger = logging.getLogger(__name__)
        # if (boundary is None):
        #     boundary = self.boundary
        [X, Y, Z, R] = self.position()
        # RBC = boundary[0:self.mnmax]
        # ZBS = boundary[self.mnmax::]
        # if (self.vmecOutputObject is None):
        #    zetas = np.linspace(0,2*np.pi/self.nfp,self.nzeta+1)
        #    zetas = np.delete(zetas,-1)
        # else:
        #    zetas = self.vmecOutputObject.zetas
        zetas = self.zetas
          
        angle_1s = np.linspace(0, 2*np.pi, 20)
        angle_2s = np.linspace(0, 2*np.pi, 20)
        angle_1s = np.delete(angle_1s, -1)
        angle_2s = np.delete(angle_2s, -1)
        inSurface = False
        for angle_1 in angle_1s:
            for angle_2 in angle_2s:
                if (angle_1 != angle_2):
                    R0_angle1 = np.zeros(np.shape(zetas))
                    Z0_angle1 = np.zeros(np.shape(zetas))
                    R0_angle2 = np.zeros(np.shape(zetas))
                    Z0_angle2 = np.zeros(np.shape(zetas))
                    for im in range(self.mnmax):
                        angle = self.xm[im] * angle_1 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle1 = R0_angle1 + self.rbc[im] * np.cos(angle)
                        Z0_angle1 = Z0_angle1 + self.zbs[im] * np.sin(angle)
                        angle = self.xm[im] * angle_2 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle2 = R0_angle2 + self.rbc[im] * np.cos(angle)
                        Z0_angle2 = Z0_angle2 + self.zbs[im] * np.sin(angle)
                    Raxis = 0.5*(R0_angle1 + R0_angle2)
                    Zaxis = 0.5*(Z0_angle1 + Z0_angle2)
                    # Check new axis shape
                    inSurfaces = np.zeros(self.nzeta)
                    for izeta in range(self.nzeta):
                        inSurfaces[izeta] = point_in_polygon(
                                R[izeta, :], Z[izeta, :], Raxis[izeta], 
                                Zaxis[izeta])
                        # if (not inSurfaces[izeta]):
                        # plt.plot(R[izeta,:],Z[izeta,:])
                        # plt.plot(Raxis[izeta],Zaxis[izeta],marker='*')
                        # plt.title('angle_1 = '+str(angle_1)+', angle_2 ='+str(angle_2))
                        # plt.show()
                    inSurface = np.all(inSurfaces)
                    if (inSurface):
                        break
        if (not inSurface):
            logger.warning('Unable to find suitable axis shape in modify_axis.')
            return -1
        
        # Now Fourier transform 
        raxis_cc = np.zeros(self.ntor)
        zaxis_cs = np.zeros(self.ntor)
        for n in range(self.ntor):
            angle = -n * self.nfp * zetas
            raxis_cc[n] = np.sum(Raxis * np.cos(angle)) / \
                    np.sum(np.cos(angle)**2)
            if (n > 0):
                zaxis_cs[n] = np.sum(Zaxis * np.sin(angle)) / \
                        np.sum(np.sin(angle)**2)
        self.raxis = raxis_cc
        self.zaxis = zaxis_cs
        return 1
            
    def radius_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                           zeta=None):
        """
        TODO: doc string
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in radius_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:, 0])
            dim2 = len(theta[0, :])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in radius_derivatives.')
            sys.exit(0)
          
        mnmax_sensitivity = len(xm_sensitivity)
        
        dRdrmnc = np.zeros((mnmax_sensitivity, dim1, dim2))
        dRdzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn] * theta - \
                    self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            dRdrmnc[imn, :, :] = cos_angle
        return dRdrmnc, dRdzmns
      
    def volume_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                           zeta=None):
        """
        TODO: doc string
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
            sys.exit(0)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [X, Y, Z, R] = self.position(theta, zeta)
        [Nx, Ny, Nz] = self.N(theta, zeta)
        
        mnmax_sensitivity = len(xm_sensitivity)
        
        d2rdthetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdthetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        dzdzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn] * theta - \
                    self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            cos_zeta = np.cos(zeta)
            sin_zeta = np.sin(zeta)
            d2rdthetadrmnc[0,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * cos_zeta
            d2rdthetadrmnc[1,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * sin_zeta
            d2rdzetadrmnc[0,imn,:,:] = self.nfp*xn_sensitivity[imn] * \
                    sin_angle * cos_zeta - cos_angle * sin_zeta
            d2rdzetadrmnc[1,imn,:,:] = self.nfp * xn_sensitivity[imn] * \
                    sin_angle * sin_zeta + cos_angle * cos_zeta
            dzdzmns[imn,:,:] = dzdzmns[imn,:,:] = sin_angle
          
        dNzdrmnc = -d2rdzetadrmnc[0,:,:,:] * dydtheta - \
                dxdzeta * d2rdthetadrmnc[1,:,:,:] + \
                d2rdthetadrmnc[0,:,:,:] * dydzeta + \
                dxdtheta * d2rdzetadrmnc[1,:,:,:] 
        dNzdzmns = -d2rdzetadzmns[0,:,:,:] * dydtheta - \
                dxdzeta * d2rdthetadzmns[1,:,:,:] + \
                d2rdthetadzmns[0,:,:,:] * dydzeta + \
                dxdtheta * d2rdzetadzmns[1,:,:,:]
        dvolumedrmnc = -np.tensordot(dNzdrmnc, Z,axes=2) * self.dtheta * \
                self.dzeta * self.nfp
        dvolumedzmns = -(np.tensordot(dNzdzmns, Z,axes=2) + \
                np.tensordot(dzdzmns, Nz, axes=2)) * self.dtheta * \
                self.dzeta * self.nfp
        return dvolumedrmnc, dvolumedzmns
      
    def area_derivatives(self,xm_sensitivity,xn_sensitivity,theta=None,zeta=None):
        """
        TODO: doc string
        """
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                xm_sensitivity, xn_sensitivity, theta, zeta)
        dareadrmnc = np.sum(dNdrmnc, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        dareadzmns = np.sum(dNdzmns, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        return dareadrmnc, dareadzmns
        # The above 3 statements could be rewritten as 
        # return map(lambda x: np.sum(x, axis=(1, 2)) * self.dtheta * \
        #        self.dzeta * self.nfp, [dNdrmnc, dNdzmns])
        # or for readability
        # def some_fn(x):  # Choose appropriate name for some_fn
        #    return np.sum(x, axis=(1, 2)) * self.dtheta * self.dzeta * self.nfp
        # return map(some_fn, [dNdrmnc, dNdzmns])
      
    # Check that theta and zeta are of correct size
    # Shape of theta and zeta must be the same
    def jacobian_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                             zeta=None):
        """
        TODO: doc string
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in jacobian_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in jacobian_derivatives.')
            sys.exit(0)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [Nx, Ny, Nz] = self.N(theta, zeta)
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
            angle = xm_sensitivity[imn] * theta - \
                    self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            cos_zeta = np.cos(zeta)
            sin_zeta = np.sin(zeta)
            d2rdthetadrmnc[0,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * cos_zeta
            d2rdthetadrmnc[1,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * sin_zeta
            d2rdzetadrmnc[0,imn,:,:] = self.nfp * xn_sensitivity[imn] * \
                    sin_angle * cos_zeta - cos_angle * sin_zeta
            d2rdzetadrmnc[1,imn,:,:] = self.nfp * xn_sensitivity[imn] * \
                    sin_angle * sin_zeta + cos_angle * cos_zeta
            d2rdthetadzmns[2,imn,:,:] = xm_sensitivity[imn] * cos_angle
            d2rdzetadzmns[2,imn,:,:] = -self.nfp * xn_sensitivity[imn] * \
                    cos_angle

        dNdrmnc = (d2rdthetadrmnc[1,:,:,:] * dzdzeta - d2rdthetadrmnc[2,:,:,:]*dydzeta)*nx \
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
      
    def effectiveDistance(self, l1=2, l2=0.5, s1=100, s2=100):
        """
        TODO: doc string
        """
        [X, Y, Z, R] = self.position()
        effectiveDistance = np.zeros((self.nzeta, self.ntheta))
        for izeta in range(self.nzeta):
            effectiveDistance[izeta,:] = tanhCoulombPotential(
                    R[izeta,:], Z[izeta,:], l1, l2, s1, s2)
        return np.sum(effectiveDistance) * self.dtheta * self.dzeta

    def summed_proximity(self, min_curvature_radius=None):
        """
        TODO: doc string
        """
        proximity = self.proximity(min_curvature_radius=min_curvature_radius)
        return np.sum(proximity) * self.dtheta * self.dzeta 
      
    def summed_proximity_derivatives(self, xm_sensitivity, xn_sensitivity, 
                                     min_curvature_radius=None):
        """
        TODO: doc string
        """
        [dQpdrmnc, dQpdzmns] = self.proximity_derivatives(
                xm_sensitivity, xn_sensitivity,
                min_curvature_radius=min_curvature_radius)
        return np.sum(dQpdrmnc, axis=(1, 2)) * self.dtheta * self.dzeta, \
                np.sum(dQpdzmns, axis=(1,2)) * self.dtheta * self.dzeta

    def proximity(self, derivatives=False, min_curvature_radius=None):
        """
        TODO: doc string
        """
        if min_curvature_radius is None:
            min_curvature_radius = self.min_curvature_radius
        
        [X,Y,Z,R] = self.position()
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives()
        dldtheta = np.sqrt(dRdtheta**2 + dzdtheta**2)
        tR = dRdtheta/dldtheta
        tz = dzdtheta/dldtheta
        Qp = np.zeros((self.nzeta, self.ntheta))
        if derivatives:
            dQpdp1 = np.zeros((self.nzeta, self.ntheta, 2))
            dQpdp2 = np.zeros((self.nzeta, self.ntheta, self.ntheta, 2))
            dQpdtau2 = np.zeros((self.nzeta, self.ntheta, self.ntheta, 2))
            dQpdlprime = np.zeros((self.nzeta, self.ntheta, self.ntheta))
        for izeta in range(self.nzeta):
            if derivatives:
                [Qp[izeta,:], dQpdp1[izeta, ...], dQpdp2[izeta, ...], \
                        dQpdtau2[izeta, ...], dQpdlprime[izeta, ...]] = \
                        proximity_slice(R[izeta, :], Z[izeta, :], \
                        tR[izeta, :], tz[izeta, :], dldtheta[izeta, :], \
                        derivatives=derivatives,\
                        min_curvature_radius=min_curvature_radius)
            else:
                Qp[izeta, :] = proximity_slice(
                        R[izeta, :], Z[izeta, :], tR[izeta, :], tz[izeta, :], 
                        dldtheta[izeta, :], derivatives=derivatives,    
                        min_curvature_radius=min_curvature_radius)
        if derivatives:
            return self.dtheta * Qp, self.dtheta * dQpdp1, \
                    self.dtheta * dQpdp2, self.dtheta * dQpdtau2, \
                    self.dtheta * dQpdlprime 
        else:
            return self.dtheta * Qp
      
    def proximity_derivatives(self, xm_sensitivity, xn_sensitivity, 
                              min_curvature_radius=None):
        """
        TODO: Function doc here
        """
        [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = self.proximity(
                derivatives=True, min_curvature_radius=min_curvature_radius)
        [X,Y,Z,R] = self.position() 
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives()
        lprime = np.sqrt(dRdtheta**2 + dzdtheta**2)
        mnmax_sensitivity = len(xm_sensitivity)
        dQpdrmnc = np.zeros((mnmax_sensitivity, self.nzeta, self.ntheta))
        dQpdzmns = np.zeros((mnmax_sensitivity, self.nzeta, self.ntheta))
        for izeta in range(self.nzeta):
            for imn in range(mnmax_sensitivity):
                angle = xm_sensitivity[imn] * self.thetas_2d[izeta, :] - \
                        self.nfp*xn_sensitivity[imn] * self.zetas_2d[izeta, :]
                dRdrmnc = np.cos(angle)
                dZdzmns = np.sin(angle)
                d2Rdthetadrmnc = -xm_sensitivity[imn] * np.sin(angle)
                d2Zdthetadzmns = xm_sensitivity[imn] * np.cos(angle)
                dlprimedrmnc = dRdtheta[izeta, :] * d2Rdthetadrmnc / \
                        lprime[izeta, :]
                dlprimedzmns = dzdtheta[izeta, :] * d2Zdthetadzmns / \
                        lprime[izeta, :]
                dtauRdrmnc = d2Rdthetadrmnc / lprime[izeta,:] - \
                        dRdtheta[izeta, :] * dlprimedrmnc / lprime[izeta, :]**2
                dtauRdzmns = -dRdtheta[izeta,:] * dlprimedzmns / lprime[izeta,:]**2
                dtauZdrmnc = -dzdtheta[izeta,:] * dlprimedrmnc / lprime[izeta,:]**2
                dtauZdzmns = d2Zdthetadzmns / lprime[izeta, :] - \
                        dzdtheta[izeta, :] *dlprimedzmns / lprime[izeta, :]**2
                for itheta in range(self.ntheta):
                    dQpdrmnc[imn, izeta, itheta] = \
                            dQpdp1[izeta, itheta, 0] * dRdrmnc[itheta] \
                            + np.dot(dQpdp2[izeta, itheta, :, 0], dRdrmnc) \
                            + np.dot(dQpdtau2[izeta, itheta, :, 0], dtauRdrmnc) \
                            + np.dot(dQpdtau2[izeta, itheta, :, 1], dtauZdrmnc) \
                            + np.dot(dQpdlprime[izeta, itheta, :], dlprimedrmnc)
                    dQpdzmns[imn, izeta, itheta] = \
                            dQpdp1[izeta, itheta, 1] * dZdzmns[itheta] \
                            + np.dot(dQpdp2[izeta, itheta, :, 1], dZdzmns) \
                            + np.dot(dQpdtau2[izeta, itheta, :, 0], dtauRdzmns) \
                            + np.dot(dQpdtau2[izeta, itheta, :, 1], dtauZdzmns) \
                            + np.dot(dQpdlprime[izeta, itheta, :], dlprimedzmns)
        return dQpdrmnc, dQpdzmns
      
def proximity_slice(R, Z, tR, tZ, dldtheta, min_curvature_radius=1.0, 
                    derivatives=False):
    assert(np.shape(R) == np.shape(Z)) 
    assert(np.shape(R) == np.shape(tR)) 
    assert(np.shape(R) == np.shape(tZ)) 
    assert(np.shape(R) == np.shape(dldtheta))
  
    ntheta = len(R)
    Qp = np.zeros((ntheta))
    if derivatives:
        dQpdp1 = np.zeros((ntheta, 2))
        dQpdp2 = np.zeros((ntheta, ntheta, 2))
        dQpdtau2 = np.zeros((ntheta, ntheta, 2))
        dQpdlprime = np.zeros((ntheta, ntheta))

    for itheta in range(ntheta):
        psi = np.zeros(ntheta)
        if derivatives:
            dpsidp1 = np.zeros((ntheta, 2))
        for ithetap in range(ntheta):
            p1 = [R[itheta], Z[itheta]]
            p2 = [R[ithetap], Z[ithetap]]
            tau2 = [tR[ithetap], tZ[ithetap]]
            if (p1 != p2):
                Sc = self_contact_function(p1, p2, tau2) - \
                        2 * min_curvature_radius
                if derivatives:
                    [dScdp1, dScdp2, dScdtau2] = self_contact_function_gradient(
                            p1, p2, tau2)
            else:
                Sc = 0
            if (Sc >= 0):
                psi[ithetap] = 0
            else:
                psi[ithetap] = 0.5 * Sc**2
                if derivatives:
                    dpsidp1[ithetap, :] = Sc * dScdp1
                    dpsidp2 = Sc * dScdp2
                    dpsidtau2 = Sc * dScdtau2
                    dQpdp2[itheta, ithetap, :] = dpsidp2 * dldtheta[ithetap]
                    dQpdtau2[itheta, ithetap, :] = dpsidtau2 * dldtheta[ithetap]
                    dQpdlprime[itheta,ithetap] = psi[ithetap]
            Qp[itheta] = np.dot(psi, dldtheta)
        if derivatives:
            dQpdp1[itheta, :] = np.dot(dpsidp1.T, dldtheta)
    if derivatives:
        return Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime
    else:
        return Qp
  
def normalDistanceRatio(p1, p2, tau2):
    # assert(np.isclose(scipy.linalg.norm(tau2),1))
    p1 = np.array(p1)
    p2 = np.array(p2)
    tau2 = np.array(tau2)
    norm = scipy.linalg.norm(p1 - p2)
    return 1 - ((p2 - p1).dot(tau2))**2 / norm**2

def normalDistanceRatio_gradient(p1, p2, tau2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    tau2 = np.array(tau2)
    norm = scipy.linalg.norm(p1 - p2)
    dnormalDistanceRatiodp1 = -2*((p2 - p1).dot(tau2) / norm) \
            * (-tau2 / norm + ((p2 - p1).dot(tau2) / norm**3) * (p2 - p1))
    dnormalDistanceRatiodp2 = -2*((p2 - p1).dot(tau2) / norm) \
            * (tau2 / norm + ((p2 - p1).dot(tau2) / norm**3) * (p1 - p2))
    dnormalDistanceRatiodtau2 = -2*((p2 - p1).dot(tau2) / norm) \
            * ((p2 - p1) / norm)
    return dnormalDistanceRatiodp1, dnormalDistanceRatiodp2, \
            dnormalDistanceRatiodtau2

# Given 2 points p1 and p2 and tangent at p2, compute self-contact 
# function [(26) in Walker]
def self_contact_function(p1, p2, tau2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    norm = scipy.linalg.norm(p1 - p2)
    normal_distance_ratio = normalDistanceRatio(p1, p2, tau2)
    # assert(normal_distance_ratio>=0 and normal_distance_ratio<=1)
    return norm / normal_distance_ratio

def self_contact_function_gradient(p1, p2, tau2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    norm = scipy.linalg.norm(p1 - p2)
    assert(norm!=0)
    N = normalDistanceRatio(p1, p2, tau2) # norm distance ratio squared
    assert(N!=0)
    [dnormalDistanceRatiodp1, dnormalDistanceRatiodp2, \
            dnormalDistanceRatiodtau2] = normalDistanceRatio_gradient(
            p1, p2, tau2)
    dScdp1 = (p1 - p2) / (norm * N) - norm * dnormalDistanceRatiodp1 / (N * N)
    dScdp2 = (p2 - p1) / (norm * N) - norm * dnormalDistanceRatiodp2 / (N * N)
    dScdtau2 = - norm * dnormalDistanceRatiodtau2 / (N * N)
    return dScdp1, dScdp2, dScdtau2

# Determine if (R0,Z0) lies in boundary defined by (R,Z) in toroidal plane
#TODO : test for correct sizes
def point_in_polygon(R, Z, R0, Z0):
    ntheta = len(R)
    oddNodes = False
    j = ntheta-1
    for i in range(ntheta):
        if ((Z[i] < Z0 and Z[j] >= Z0) or (Z[j] < Z0 and Z[i] >= Z0)):
            if (R[i] + (Z0 - Z[i]) / (Z[j] - Z[i]) * (R[j] - R[i]) < R0):
                oddNodes = not oddNodes
        j = i
    return oddNodes

def tanhCoulombPotential(R, Z, l1, l2, s1, s2):
    assert(len(R) == len(Z))
    assert(R[0] != R[-1] and R[0] != R[-1])
    ntheta = len(R)
    dR = np.roll(R, -1) -R
    dZ = np.roll(Z, -1) -Z
    dl = np.sqrt(dR**2 + dZ**2)
    l = np.cumsum(dl)
    effective_distance = np.zeros((ntheta, ntheta))
    effective_distance_integrated = np.zeros(ntheta)
    activation_length = np.zeros((ntheta, ntheta))
    activation_function1 = np.zeros((ntheta, ntheta))
    for itheta in range(ntheta):
        for ithetap in range(ntheta):
            if (ithetap != itheta):
                activation_length[itheta, ithetap] = (l[itheta] - l[ithetap])
                # Limit activation_legnth to -L/2, L/2
                if activation_length[itheta, ithetap] > l[-1]/2:
                    activation_length[itheta, ithetap] = \
                            l[-1] - activation_length[itheta, ithetap]
                if activation_length[itheta,ithetap] < -l[-1]/2:
                    activation_length[itheta,ithetap] = \
                            -l[-1] - activation_length[itheta, ithetap]
                assert(activation_length[itheta, ithetap] <= l[-1]/2) 
                assert(activation_length[itheta, ithetap] >= -l[-1]/2)
                activation_function1[itheta, ithetap] = (1 + np.tanh(s1 * \
                        (np.abs(activation_length[itheta, ithetap]) - l1)))/2
                assert(activation_function1[itheta, ithetap] >= 0)
                activation_distance = np.sqrt((R[itheta] - R[ithetap])**2 + \
                        (Z[itheta] - Z[ithetap])**2)
                activation_function2 = (1 - np.tanh(s2 * \
                        (activation_distance - l2)))/2
                assert(activation_function2 >= 0)
                effective_distance[itheta,ithetap] = activation_function1[\
                        itheta, ithetap] * activation_function2
                assert(effective_distance[itheta, ithetap] >= 0)
    effective_distance_integrated[itheta] = np.dot(
            effective_distance[itheta, :], dl)
    return effective_distance_integrated, effective_distance
#, activation_length, activation_function1
    
      


