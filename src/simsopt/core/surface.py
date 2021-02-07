# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides several classes for representing toroidal
surfaces.  There is a base class Surface, and several child classes
corresponding to different discrete representations.
"""

# These next 2 lines Use double precision:
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit

import numpy as np
import logging
from mpi4py import MPI
from .util import isbool
from .optimizable import Optimizable

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

#@jit(static_argnums=(4, 5, 6, 7, 8, 9))
def area_volume_pure(rc, rs, zc, zs, stelsym, nfp, mpol, ntor, ntheta, nphi):
    """
    Compute the area and volume of a surface. This pure function is
    designed for automatic differentiation.
    """
    mdim = mpol + 1
    ndim = 2 * ntor + 1
    theta1d = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
    phi1d = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    dtheta = theta1d[1] - theta1d[0]
    dphi = phi1d[1] - phi1d[0]
    phi, theta = jnp.meshgrid(phi1d, theta1d)

    r = jnp.zeros((ntheta, nphi))
    x = jnp.zeros((ntheta, nphi))
    y = jnp.zeros((ntheta, nphi))
    z = jnp.zeros((ntheta, nphi))
    dxdtheta = jnp.zeros((ntheta, nphi))
    dydtheta = jnp.zeros((ntheta, nphi))
    dzdtheta = jnp.zeros((ntheta, nphi))
    dxdphi = jnp.zeros((ntheta, nphi))
    dydphi = jnp.zeros((ntheta, nphi))
    dzdphi = jnp.zeros((ntheta, nphi))
    sinphi = jnp.sin(phi)
    cosphi = jnp.cos(phi)
    for m in range(mdim):
        for jn in range(ndim):
            # Presently this loop includes negative n when m=0.
            # This is unnecesary but doesn't hurt I think.
            n_without_nfp = jn - ntor
            n = n_without_nfp * nfp
            angle = m * theta - n * phi
            sinangle = jnp.sin(angle)
            cosangle = jnp.cos(angle)
            rmnc = rc[m, jn]
            zmns = zs[m, jn]
            r += rmnc * cosangle
            x += rmnc * cosangle * cosphi
            y += rmnc * cosangle * sinphi
            z += zmns * sinangle

            dxdtheta += rmnc * (-m * sinangle) * cosphi
            dydtheta += rmnc * (-m * sinangle) * sinphi
            dzdtheta += zmns * m * cosangle

            dxdphi += rmnc * (n * sinangle * cosphi + cosangle * (-sinphi))
            dydphi += rmnc * (n * sinangle * sinphi + cosangle * cosphi)
            dzdphi += zmns * (-n * cosangle)
            if not stelsym:
                rmns = rs[m, jn]
                zmnc = zc[m, jn]
                r += rmns * sinangle
                x += rmns * sinangle * cosphi
                y += rmns * sinangle * sinphi
                z += zmnc * cosangle

                dxdtheta += rmns * (m * cosangle) * cosphi
                dydtheta += rmns * (m * cosangle) * sinphi
                dzdtheta += zmnc * (-m * sinangle)

                dxdphi += rmns * (-n * cosangle * cosphi + sinangle * (-sinphi))
                dydphi += rmns * (-n * cosangle * sinphi + sinangle * cosphi)
                dzdphi += zmnc * (n * sinangle)

    normalx = dydphi * dzdtheta - dzdphi * dydtheta
    normaly = dzdphi * dxdtheta - dxdphi * dzdtheta
    normalz = dxdphi * dydtheta - dydphi * dxdtheta
    norm_normal = jnp.sqrt(normalx * normalx + normaly * normaly + normalz * normalz)
    area = nfp * dtheta * dphi * jnp.sum(norm_normal)
    # Compute plasma volume using \int (1/2) R^2 dZ dphi
    # = \int (1/2) R^2 (dZ/dtheta) dtheta dphi
    volume = 0.5 * nfp * dtheta * dphi * jnp.sum(r * r * dzdtheta)
    return jnp.array([area, volume])

#area_volume_pure(rc, rs, zc, zs, stelsym, nfp, mpol, ntor, ntheta, nphi)
#jit_area_volume_pure = jit(area_volume_pure, static_argnums=(4, 5, 6, 7, 8, 9))
#jit_area_volume_pure = jit(area_volume_pure, static_argnums=(8, 9))
jit_area_volume_pure = area_volume_pure
darea_volume_pure = jacrev(area_volume_pure, argnums=(0, 1, 2, 3))

# Here I have Surface subclass Optimizable, which is convenient while
# surface.py is part of simsopt instead of being in a separate simsgeo
# repo. If surface.py is moved to simsgeo, we would no longer have
# Surface subclass Optimizable. As a result we might need to add
# optimizable() in a few places, such as vmec.py.
class Surface(Optimizable):
    """
    Surface is a base class for various representations of toroidal
    surfaces in simsopt.
    """

    def __init__(self, nfp=1, stelsym=True):
        if not isinstance(nfp, int):
            raise TypeError('nfp must be an integer')
        if not isbool(stelsym):
            raise TypeError('stelsym must be a bool')
        self.nfp = nfp
        self.stelsym = stelsym

    def __repr__(self):
        return "Surface " + str(hex(id(self))) + " (nfp=" + str(self.nfp) \
            + ", stelsym=" + str(self.stelsym) + ")"

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.  All subclasses should implement this abstract
        method.
        """
        raise NotImplementedError

class SurfaceRZFourier(Surface):
    """
    SurfaceRZFourier is a surface that is represented in cylindrical
    coordinates using the following Fourier series: 

    r(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
                     r_{c,m,n} \cos(m \theta - n nfp \phi)
                     + r_{s,m,n} \sin(m \theta - n nfp \phi) ]

    and the same for z(theta, phi).

    Here, (r, phi, z) are standard cylindrical coordinates, and theta
    is any poloidal angle.
    """
    def __init__(self, nfp=1, stelsym=True, mpol=1, ntor=0):
        mpol = int(mpol)
        ntor = int(ntor)
        # Perform some validation.
        if mpol < 1:
            raise ValueError("mpol must be at least 1")
        if ntor < 0:
            raise ValueError("ntor must be at least 0")
        Surface.__init__(self, nfp=nfp, stelsym=stelsym)
        self.mpol = mpol
        self.ntor = ntor
        self.allocate()
        self.recalculate = True
        self.recalculate_derivs = True

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        #self.get_rc(0,0) = 1.0
        #self.get_rc(1,0) = 0.1
        #self.get_zs(1,0) = 0.1
        self.rc[0, ntor] = 1.0
        self.rc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        
        # Resolution for computing area, volume, etc:
        self.ntheta = 63
        self.nphi = 62

    def allocate(self):
        """
        Create the arrays for the rc, rs, zc, and zs coefficients.
        """
        logger.info("Allocating SurfaceRZFourier")
        self.mdim = self.mpol + 1
        self.ndim = 2 * self.ntor + 1
        myshape = (self.mdim, self.ndim)

        self.rc = np.zeros(myshape)
        self.zs = np.zeros(myshape)
        self.names = self.make_names('rc', True) + self.make_names('zs', False)
        
        if not self.stelsym:
            self.rs = np.zeros(myshape)
            self.zc = np.zeros(myshape)
            self.names += self.make_names('rs', False) + self.make_names('zc', True)

    def change_resolution(self, mpol, ntor):
        """
        Change the values of mpol and ntor. Any new Fourier amplitudes
        will have a magnitude of zero.  Any previous nonzero Fourier
        amplitudes that are not within the new range will be
        discarded.
        """
        old_mpol = self.mpol
        old_ntor = self.ntor
        old_rc = self.rc
        old_zs = self.zs
        if not self.stelsym:
            old_rs = self.rs
            old_zc = self.zc
        self.mpol = mpol
        self.ntor = ntor
        self.allocate()
        if mpol < old_mpol or ntor < old_ntor:
            # Don't need to recalculate if we only add zeros
            self.recalculate = True
            self.recalculate_derivs = True
            
        min_mpol = np.min((mpol, old_mpol))
        min_ntor = np.min((ntor, old_ntor))
        for m in range(min_mpol + 1):
            for n in range(-min_ntor, min_ntor + 1):
                self.rc[m, n + ntor] = old_rc[m, n + old_ntor]
                self.zs[m, n + ntor] = old_zs[m, n + old_ntor]
                if not self.stelsym:
                    self.rs[m, n + ntor] = old_rs[m, n + old_ntor]
                    self.zc[m, n + ntor] = old_zc[m, n + old_ntor]
        
    def make_names(self, prefix, include0):
        """
        Form a list of names of the rc, zs, rs, or zc array elements.
        """
        if include0:
            names = [prefix + "(0,0)"]
        else:
            names = []

        names += [prefix + '(0,' + str(n) + ')' for n in range(1, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names
            
    def __repr__(self):
        return "SurfaceRZFourier " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp) + ", stelsym=" + str(self.stelsym) + \
            ", mpol=" + str(self.mpol) + ", ntor=" + str(self.ntor) \
            + ")"

    def _validate_mn(self, m, n):
        """
        Check whether m and n are in the allowed range.
        """
        if m < 0:
            raise ValueError('m must be >= 0')
        if m > self.mpol:
            raise ValueError('m must be <= mpol')
        if n > self.ntor:
            raise ValueError('n must be <= ntor')
        if n < -self.ntor:
            raise ValueError('n must be >= -ntor')
    
    def get_rc(self, m, n):
        """
        Return a particular rc Parameter.
        """
        self._validate_mn(m, n)
        return self.rc[m, n + self.ntor]

    def get_rs(self, m, n):
        """
        Return a particular rs Parameter.
        """
        if self.stelsym:
            return ValueError( \
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.rs[m, n + self.ntor]

    def get_zc(self, m, n):
        """
        Return a particular zc Parameter.
        """
        if self.stelsym:
            return ValueError( \
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.zc[m, n + self.ntor]

    def get_zs(self, m, n):
        """
        Return a particular zs Parameter.
        """
        self._validate_mn(m, n)
        return self.zs[m, n + self.ntor]

    def set_rc(self, m, n, val):
        """
        Set a particular rc Parameter.
        """
        self._validate_mn(m, n)
        self.rc[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def set_rs(self, m, n, val):
        """
        Set a particular rs Parameter.
        """
        if self.stelsym:
            return ValueError( \
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.rs[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def set_zc(self, m, n, val):
        """
        Set a particular zc Parameter.
        """
        if self.stelsym:
            return ValueError( \
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.zc[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def set_zs(self, m, n, val):
        """
        Set a particular zs Parameter.
        """
        self._validate_mn(m, n)
        self.zs[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def area_volume(self):
        """
        Compute the surface area and the volume enclosed by the surface.
        """
        if self.recalculate:
            logger.info('Running calculation of area and volume')
        else:
            logger.info('area_volume called, but no need to recalculate')
            return

        self.recalculate = False

        if self.stelsym:
            rs = None
            zc = None
        else:
            rs = self.rs
            zc = self.zc

        results = jit_area_volume_pure(self.rc, rs, zc, self.zs,
                                   self.stelsym, self.nfp, self.mpol,
                                   self.ntor, self.ntheta, self.nphi)

        self._area = float(results[0])
        self._volume = float(results[1])
        """
        ntheta = self.ntheta # Shorthand
        nphi = self.nphi
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / self.nfp, nphi, \
                                endpoint=False)
        dtheta = theta1d[1] - theta1d[0]
        dphi = phi1d[1] - phi1d[0]
        phi, theta = np.meshgrid(phi1d, theta1d)
        r = np.zeros((ntheta, nphi))
        x = np.zeros((ntheta, nphi))
        y = np.zeros((ntheta, nphi))
        z = np.zeros((ntheta, nphi))
        dxdtheta = np.zeros((ntheta, nphi))
        dydtheta = np.zeros((ntheta, nphi))
        dzdtheta = np.zeros((ntheta, nphi))
        dxdphi = np.zeros((ntheta, nphi))
        dydphi = np.zeros((ntheta, nphi))
        dzdphi = np.zeros((ntheta, nphi))
        mdim = self.mpol + 1
        ndim = 2 * self.ntor + 1
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        nfp = self.nfp
        for m in range(mdim):
            for jn in range(ndim):
                # Presently this loop includes negative n when m=0.
                # This is unnecesary but doesn't hurt I think.
                n_without_nfp = jn - self.ntor
                n = n_without_nfp * nfp
                angle = m * theta - n * phi
                sinangle = np.sin(angle)
                cosangle = np.cos(angle)
                rmnc = self.get_rc(m, n_without_nfp)
                zmns = self.get_zs(m, n_without_nfp)
                r += rmnc * cosangle
                x += rmnc * cosangle * cosphi
                y += rmnc * cosangle * sinphi
                z += zmns * sinangle

                dxdtheta += rmnc * (-m * sinangle) * cosphi
                dydtheta += rmnc * (-m * sinangle) * sinphi
                dzdtheta += zmns * m * cosangle

                dxdphi += rmnc * (n * sinangle * cosphi \
                                      + cosangle * (-sinphi))
                dydphi += rmnc * (n * sinangle * sinphi \
                                      + cosangle * cosphi)
                dzdphi += zmns * (-n * cosangle)
                if not self.stelsym:
                    rmns = self.get_rs(m, n_without_nfp)
                    zmnc = self.get_zc(m, n_without_nfp)
                    r += rmns * sinangle
                    x += rmns * sinangle * cosphi
                    y += rmns * sinangle * sinphi
                    z += zmnc * cosangle

                    dxdtheta += rmns * (m * cosangle) * cosphi
                    dydtheta += rmns * (m * cosangle) * sinphi
                    dzdtheta += zmnc * (-m * sinangle)
                    
                    dxdphi += rmns * (-n * cosangle * cosphi \
                                           + sinangle * (-sinphi))
                    dydphi += rmns * (-n * cosangle * sinphi \
                                           + sinangle * cosphi)
                    dzdphi += zmnc * (n * sinangle)

        normalx = dydphi * dzdtheta - dzdphi * dydtheta
        normaly = dzdphi * dxdtheta - dxdphi * dzdtheta
        normalz = dxdphi * dydtheta - dydphi * dxdtheta
        norm_normal = np.sqrt(normalx * normalx + normaly * normaly \
                                  + normalz * normalz)
        self._area = nfp * dtheta * dphi * np.sum(np.sum(norm_normal))
        # Compute plasma volume using \int (1/2) R^2 dZ dphi
        # = \int (1/2) R^2 (dZ/dtheta) dtheta dphi
        self._volume = 0.5 * nfp * dtheta * dphi * np.sum(np.sum(r * r * dzdtheta))
        """
        
    def area(self):
        """
        Return the area of the surface.
        """
        self.area_volume()
        return self._area

    def volume(self):
        """
        Return the volume of the surface.
        """
        self.area_volume()
        return self._volume

    def darea_volume(self):
        """
        Compute the derivative of the surface area and the volume enclosed
        by the surface.
        """
        if self.recalculate_derivs:
            logger.info('Running calculation of derivative of area and volume')
        else:
            logger.info('darea_volume called, but no need to recalculate')
            return

        self.recalculate_derivs = False

        if self.stelsym:
            rs = None
            zc = None
        else:
            rs = self.rs
            zc = self.zc

        results = darea_volume_pure(self.rc, rs, zc, self.zs,
                                   self.stelsym, self.nfp, self.mpol,
                                   self.ntor, self.ntheta, self.nphi)

        self._darea_drc = np.array(results[0][0, :, :])
        self._dvolume_drc = np.array(results[0][1, :, :])
        self._darea_dzs = np.array(results[3][0, :, :])
        self._dvolume_dzs = np.array(results[3][1, :, :])
        if not self.stelsym:
            self._darea_drs = np.array(results[1][0, :, :])
            self._dvolume_drs = np.array(results[1][1, :, :])
            self._darea_dzc = np.array(results[2][0, :, :])
            self._dvolume_dzc = np.array(results[2][1, :, :])

    def darea(self):
        """
        Return the area of the surface.
        """
        self.darea_volume()
        mpol = self.mpol # Shorthand
        ntor = self.ntor
        if self.stelsym:
            return np.concatenate( \
                (self._darea_drc[0, ntor:], \
                 self._darea_drc[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._darea_dzs[0, ntor + 1:], \
                 self._darea_dzs[1:, :].reshape(mpol * (ntor * 2 + 1))))
        else:
            return np.concatenate( \
                (self._darea_drc[0, ntor:], \
                 self._darea_drc[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._darea_dzs[0, ntor + 1:], \
                 self._darea_dzs[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._darea_drs[0, ntor + 1:], \
                 self._darea_drs[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._darea_dzc[0, ntor:], \
                 self._darea_dzc[1:, :].reshape(mpol * (ntor * 2 + 1))))

    def dvolume(self):
        """
        Return the volume of the surface.
        """
        self.darea_volume()
        mpol = self.mpol # Shorthand
        ntor = self.ntor
        if self.stelsym:
            return np.concatenate( \
                (self._dvolume_drc[0, ntor:], \
                 self._dvolume_drc[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._dvolume_dzs[0, ntor + 1:], \
                 self._dvolume_dzs[1:, :].reshape(mpol * (ntor * 2 + 1))))
        else:
            return np.concatenate( \
                (self._dvolume_drc[0, ntor:], \
                 self._dvolume_drc[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._dvolume_dzs[0, ntor + 1:], \
                 self._dvolume_dzs[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._dvolume_drs[0, ntor + 1:], \
                 self._dvolume_drs[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self._dvolume_dzc[0, ntor:], \
                 self._dvolume_dzc[1:, :].reshape(mpol * (ntor * 2 + 1))))

    @classmethod
    def from_focus(cls, filename):
        """
        Read in a surface from a FOCUS-format file.
        """
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()

        # Read the line containing Nfou and nfp:
        splitline = lines[1].split()
        errmsg = "This does not appear to be a FOCUS-format file."
        assert len(splitline) == 3, errmsg
        Nfou = int(splitline[0])
        nfp = int(splitline[1])

        # Now read the Fourier amplitudes:
        n = np.full(Nfou, 0)
        m = np.full(Nfou, 0)
        rc = np.zeros(Nfou)
        rs = np.zeros(Nfou)
        zc = np.zeros(Nfou)
        zs = np.zeros(Nfou)
        for j in range(Nfou):
            splitline = lines[j + 4].split()
            n[j] = int(splitline[0])
            m[j] = int(splitline[1])
            rc[j] = float(splitline[2])
            rs[j] = float(splitline[3])
            zc[j] = float(splitline[4])
            zs[j] = float(splitline[5])
        assert np.min(m) == 0
        stelsym = np.max(np.abs(rs)) == 0 and np.max(np.abs(zc)) == 0
        mpol = int(np.max(m))
        ntor = int(np.max(np.abs(n)))

        surf = cls(nfp=nfp, stelsym=stelsym, mpol=mpol, ntor=ntor)
        for j in range(Nfou):
            surf.rc[m[j], n[j] + ntor] = rc[j]
            surf.zs[m[j], n[j] + ntor] = zs[j]
            if not stelsym:
                surf.rs[m[j], n[j] + ntor] = rs[j]
                surf.zc[m[j], n[j] + ntor] = zc[j]

        return surf

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        mpol = self.mpol # Shorthand
        ntor = self.ntor
        if self.stelsym:
            return np.concatenate( \
                (self.rc[0, ntor:], \
                 self.rc[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self.zs[0, ntor + 1:], \
                 self.zs[1:, :].reshape(mpol * (ntor * 2 + 1))))
        else:
            return np.concatenate( \
                (self.rc[0, ntor:], \
                 self.rc[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self.zs[0, ntor + 1:], \
                 self.zs[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self.rs[0, ntor + 1:], \
                 self.rs[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self.zc[0, ntor:], \
                 self.zc[1:, :].reshape(mpol * (ntor * 2 + 1))))

    def set_dofs(self, v):
        """
        Set the shape coefficients from a 1D list/array
        """

        n = len(self.get_dofs())
        if len(v) != n:
            raise ValueError('Input vector should have ' + str(n) + \
                             ' elements but instead has ' + str(len(v)))
        
        # Check whether any elements actually change:
        if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
            logger.info('set_dofs called, but no dofs actually changed')
            return

        logger.info('set_dofs called, and at least one dof changed')
        self.recalculate = True
        self.recalculate_derivs = True
        
        mpol = self.mpol # Shorthand
        ntor = self.ntor
        self.rc[0, ntor:] = v[0:ntor + 1]
        self.rc[1:, :] = np.array(v[ntor + 1:ntor + 1 + mpol * (ntor * 2 + 1)]).reshape(mpol, ntor * 2 + 1)
        self.zs[0, ntor + 1:] = v[ntor + 1 + mpol * (ntor * 2 + 1): \
                                           ntor * 2 + 1 + mpol * (ntor * 2 + 1)]
        self.zs[1:, :] = np.array(v[ntor * 2 + 1 + mpol * (ntor * 2 + 1): \
                                           ntor * 2 + 1 + 2 * mpol * (ntor * 2 + 1)]).reshape(mpol, ntor * 2 + 1)

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of m and n values.

        All modes with m in the interval [mmin, mmax] and n in the
        interval [nmin, nmax] will have their fixed property set to
        the value of the 'fixed' parameter. Note that mmax and nmax
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        for m in range(mmin, mmax + 1):
            for n in range(nmin, nmax + 1):
                self.set_fixed('rc({},{})'.format(m, n), fixed)
                self.set_fixed('zs({},{})'.format(m, n), fixed)
                if not self.stelsym:
                    self.set_fixed('rs({},{})'.format(m, n), fixed)
                    self.set_fixed('zc({},{})'.format(m, n), fixed)
        
    def to_RZFourier(self):
        """
        No conversion necessary.
        """
        return self
        
    def to_Garabedian(self):
        """
        Return a SurfaceGarabedian object with the identical shape.

        For a derivation of the transformation here, see 
        https://terpconnect.umd.edu/~mattland/assets/notes/toroidal_surface_parameterizations.pdf
        """
        if not self.stelsym:
            raise RuntimeError('Non-stellarator-symmetric SurfaceGarabedian objects have not been implemented')
        mmax = self.mpol + 1
        mmin = np.min((0, 1 - self.mpol))
        s = SurfaceGarabedian(nfp=self.nfp, mmin=mmin, mmax=mmax, nmin=-self.ntor, nmax=self.ntor)
        for n in range(-self.ntor, self.ntor + 1):
            for m in range(mmin, mmax + 1):
                Delta = 0
                if m - 1 >= 0:
                    Delta = 0.5 * (self.get_rc(m - 1, n) - self.get_zs(m - 1, n))
                if 1 - m >= 0:
                    Delta += 0.5 * (self.get_rc(1 - m, -n) + self.get_zs(1 - m, -n))
                s.set_Delta(m, n, Delta)
                
        return s

    
class SurfaceGarabedian(Surface):
    """
    SurfaceGarabedian represents a toroidal surface for which the
    shape is parameterized using Garabedian's Delta_{m,n}
    coefficients.

    The present implementation assumes stellarator symmetry. Note that
    non-stellarator-symmetric surfaces require that the Delta_{m,n}
    coefficients be imaginary.
    """
    def __init__(self, nfp=1, mmax=1, mmin=0, nmax=0, nmin=None):
        if nmin is None:
            nmin = -nmax
        # Perform some validation.
        if mmax < mmin:
            raise ValueError("mmin must be >= mmax")
        if nmax < nmin:
            raise ValueError("nmin must be >= nmax")
        if mmax < 1:
            raise ValueError("mmax must be >= 1")
        if mmin > 0:
            raise ValueError("mmin must be <= 0")
        Surface.__init__(self, nfp=nfp, stelsym=True)
        self.mmin = mmin
        self.mmax = mmax
        self.nmin = nmin
        self.nmax = nmax
        self.allocate()
        self.recalculate = True
        self.recalculate_derivs = True

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.set_Delta(1, 0, 1.0)
        self.set_Delta(0, 0, 0.1)

    def __repr__(self):
        return "SurfaceGarabedian " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp) + ", mmin=" + str(self.mmin) + ", mmax=" + str(self.mmax) \
            + ", nmin=" + str(self.nmin) + ", nmax=" + str(self.nmax) \
            + ")"

    def allocate(self):
        """
        Create the array for the Delta_{m,n} coefficients.
        """
        logger.info("Allocating SurfaceGarabedian")
        self.mdim = self.mmax - self.mmin + 1
        self.ndim = self.nmax - self.nmin + 1
        myshape = (self.mdim, self.ndim)
        self.Delta = np.zeros(myshape)
        self.names = []
        for n in range(self.nmin, self.nmax + 1):
            for m in range(self.mmin, self.mmax + 1):
                self.names.append('Delta(' + str(m) + ',' + str(n) + ')')

    def get_Delta(self, m, n):
        """
        Return a particular Delta_{m,n} coefficient.
        """
        return self.Delta[m - self.mmin, n - self.nmin]

    def set_Delta(self, m, n, val):
        """
        Set a particular Delta_{m,n} coefficient.
        """
        self.Delta[m - self.mmin, n - self.nmin] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        num_dofs = (self.mmax - self.mmin + 1) * (self.nmax - self.nmin + 1)
        return np.reshape(self.Delta, (num_dofs,), order='F')

    def set_dofs(self, v):
        """
        Set the shape coefficients from a 1D list/array
        """

        n = len(self.get_dofs())
        if len(v) != n:
            raise ValueError('Input vector should have ' + str(n) + \
                             ' elements but instead has ' + str(len(v)))
        
        # Check whether any elements actually change:
        if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
            logger.info('set_dofs called, but no dofs actually changed')
            return

        logger.info('set_dofs called, and at least one dof changed')
        self.recalculate = True
        self.recalculate_derivs = True

        self.Delta = v.reshape((self.mmax - self.mmin + 1, self.nmax - self.nmin + 1), order='F')

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of m and n values.

        All modes with m in the interval [mmin, mmax] and n in the
        interval [nmin, nmax] will have their fixed property set to
        the value of the 'fixed' parameter. Note that mmax and nmax
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        for m in range(mmin, mmax + 1):
            for n in range(nmin, nmax + 1):
                self.set_fixed('Delta({},{})'.format(m, n), fixed)
        
    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier object with the identical shape.

        For a derivation of the transformation here, see 
        https://terpconnect.umd.edu/~mattland/assets/notes/toroidal_surface_parameterizations.pdf
        """
        mpol = int(np.max((1, self.mmax - 1, 1 - self.mmin)))
        ntor = int(np.max((self.nmax, -self.nmin)))
        s = SurfaceRZFourier(nfp=self.nfp, stelsym=True, mpol=mpol, ntor=ntor)
        s.set_rc(0, 0, self.get_Delta(1, 0))
        for m in range(mpol + 1):
            nmin = -ntor
            if m == 0:
                nmin = 1
            for n in range(nmin, ntor + 1):
                Delta1 = 0
                Delta2 = 0
                if 1 - m >= self.mmin and -n >= self.nmin and -n <= self.nmax:
                    Delta1 = self.get_Delta(1 - m, -n)
                if 1 + m <= self.mmax and n >= self.nmin and n <= self.nmax:
                    Delta2 = self.get_Delta(1 + m, n)
                s.set_rc(m, n, Delta1 + Delta2)
                s.set_zs(m, n, Delta1 - Delta2)

        return s

    def area_volume(self):
        """
        Compute the surface area and the volume enclosed by the surface.
        """
        if self.recalculate:
            logger.info('Running calculation of area and volume')
        else:
            logger.info('area_volume called, but no need to recalculate')
            return

        self.recalculate = False

        # Delegate to the area and volume calculations of SurfaceRZFourier():
        s = self.to_RZFourier()
        self._area = s.area()
        self._volume = s.volume()

    def area(self):
        """
        Return the area of the surface.
        """
        self.area_volume()
        return self._area

    def volume(self):
        """
        Return the volume of the surface.
        """
        self.area_volume()
        return self._volume

