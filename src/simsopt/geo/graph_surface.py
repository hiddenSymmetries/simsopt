# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides several classes for representing toroidal
surfaces.  There is a base class Surface, and several child classes
corresponding to different discrete representations.
"""
import numpy as np
import logging
import abc
from jax.config import config
config.update("jax_enable_x64", True)  # Use double precision:

import jax.numpy as jnp
from jax import jacrev
#import xarray as xr

from .._core.util import isbool
from .._core.graph_optimizable import Optimizable

logger = logging.getLogger(__name__)


# @jit(static_argnums=(4, 5, 6, 7, 8, 9))
def area_volume_pure(rc, rs, zc, zs, stellsym, nfp, mpol, ntor, ntheta, nphi):
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
            if not stellsym:
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


def dataset_area_volume(surface_rz_fourier,
                        stellsym,
                        nfp,
                        mpol,
                        ntor,
                        ntheta,
                        nphi):
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

    ds = surface_rz_fourier  # For simple access
    rc = ds["rc"]
    zs = ds["zs"]
    rs = ds.get("rs", None)
    zc = ds.get("zc", None)
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
            if not stellsym:
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


# area_volume_pure(rc, rs, zc, zs, stellsym, nfp, mpol, ntor, ntheta, nphi)
# jit_area_volume_pure = jit(area_volume_pure, static_argnums=(4, 5, 6, 7, 8, 9))
# jit_area_volume_pure = jit(area_volume_pure, static_argnums=(8, 9))
jit_area_volume_pure = area_volume_pure
darea_volume_pure = jacrev(area_volume_pure, argnums=(0, 1, 2, 3))

jit_dataset_area_volume = dataset_area_volume


class Surface(abc.ABC):
    """
    Surface is a base class for various representations of toroidal
    surfaces in simsopt.
    """

    def __init__(self, nfp=1, stellsym=True):
        #if not isinstance(nfp, int):
        #    raise TypeError('nfp must be an integer')
        #if not isbool(stellsym):
        #    raise TypeError('stellsym must be a bool')
        self.nfp = nfp
        self.stellsym = stellsym

    def __repr__(self):
        return "Surface " + str(hex(id(self))) + " (nfp=" + str(self.nfp) \
               + ", stellsym=" + str(self.stellsym) + ")"

    @abc.abstractmethod
    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.  All subclasses should implement this abstract
        method.
        """
        #raise NotImplementedError


class SurfaceRZFourier(Surface, Optimizable):
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

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0):

        Surface.__init__(self, nfp=nfp, stellsym=stellsym)
        if mpol < 1:
            raise ValueError("mpol must be at least 1")
        if ntor < 0:
            raise ValueError("ntor must be at least 0")
        self.mpol = mpol
        self.ntor = ntor

        logger.info("Allocating SurfaceRZFourier")
        self.mdim = self.mpol + 1
        self.ndim = 2 * self.ntor + 1
        self.size = self.mdim * self.ndim
        self.shape = (self.mdim, self.ndim)

        rc = np.zeros(self.shape)
        zs = np.zeros(self.shape)
        dof_names = self.make_names('rc') + self.make_names('zs')

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        rc[0, ntor] = 1.0
        rc[1, ntor] = 0.1
        zs[1, ntor] = 0.1

        dofs_x = np.concatenate([rc.flatten(), zs.flatten()])
        rc_fixed = [True] * self.ntor + [False] * (self.ntor + 1)
        zs_fixed = [True] * (self.ntor + 1) + [False] * self.ntor
        for m in range(self.mpol):
            rc_fixed += [False] * (2*self.ntor + 1)
            zs_fixed += [False] * (2*self.ntor + 1)
        dofs_fixed = np.concatenate([np.array(rc_fixed), np.array(zs_fixed)])

        if not self.stellsym:
            rs = np.zeros(self.shape)
            zc = np.zeros(self.shape)
            dofs_x = np.concatenate([dofs_x, rs.flatten(), zc.flatten()])

            zc_fixed = [True] * self.ntor + [False] * (self.ntor + 1)
            rs_fixed = [True] * (self.ntor + 1) + [False] * self.ntor
            for m in range(self.mpol):
                zc_fixed += [False] * (2 * self.ntor + 1)
                rs_fixed += [False] * (2 * self.ntor + 1)
            dofs_fixed = np.concatenate([dofs_fixed, np.array(rs_fixed),
                                        np.array(zc_fixed)])
            dof_names += self.make_names('rs') + self.make_names('zc')

        Optimizable.__init__(self, x0=dofs_x, names=dof_names, fixed=dofs_fixed)

        #self.recalculate = True
        self.recalculate_derivs = True

        # Resolution for computing area, volume, etc:
        self.ntheta = 63
        self.nphi = 62

    def make_names(self, prefix):
        """
        Create the arrays for the rc, rs, zc, and zs coefficients.
        """
        names = []
        for m in range(self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' \
                      for n in range(-self.ntor, self.ntor + 1)]
        return names

    def __repr__(self):
        return "SurfaceRZFourier " + str(hex(id(self))) + " (nfp=" + \
               str(self.nfp) + ", stellsym=" + str(self.stellsym) + \
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

    @property
    def rc(self):
        return self.local_full_x[0:self.size].reshape(self.shape)

    @rc.setter
    def rc(self, rc):
        rc = np.array(rc)
        self._dofs.iloc[0:self.size, 0] = rc.flatten()
        self.new_x = True

    @property
    def zs(self):
        return self.local_full_x[self.size:2*self.size].reshape(self.shape)

    @zs.setter
    def zs(self, zs):
        zs = np.array(zs)
        self._dofs.iloc[self.size:2*self.size, 0] = zs.flatten()
        self.new_x = True

    @property
    def rs(self):
        if self.stellsym:
            return None
        return self.local_full_x[2*self.size:3*self.size].reshape(self.shape)

    @rs.setter
    def rs(self, rs):
        rs = np.array(rs)
        self._dofs.iloc[2*self.size:3*self.size, 0] = rs.flatten()
        self.new_x = True

    @property
    def zc(self):
        if self.stellsym:
            return None
        return self.local_full_x[3*self.size:].reshape(self.shape)

    @zc.setter
    def zc(self, zc):
        zc = np.array(zc)
        self._dofs.iloc[3*self.size:, 0] = zc.flatten()
        self.new_x = True

    def get_rc(self, m, n):
        """
        Return a particular rc Parameter.
        """
        self._validate_mn(m, n)
        rc = self.local_full_x[0:self.size].reshape(self.shape)
        return rc[m, n + self.ntor]

    def get_rs(self, m, n):
        """
        Return a particular rs Parameter.
        """
        if self.stellsym:
            return ValueError(
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        rs = self.local_full_x[2*self.size:3*self.size].reshape(self.shape)
        return rs[m, n + self.ntor]

    def get_zc(self, m, n):
        """
        Return a particular zc Parameter.
        """
        if self.stellsym:
            return ValueError(
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        zc = self.local_full_x[3*self.size:].reshape(self.shape)
        return zc[m, n + self.ntor]

    def get_zs(self, m, n):
        """
        Return a particular zs Parameter.
        """
        self._validate_mn(m, n)
        zs = self.local_full_x[self.size:2*self.size].reshape(self.shape)
        return zs[m, n + self.ntor]

    def set_rc(self, m, n, val):
        """
        Set a particular rc Parameter.
        """
        self._validate_mn(m, n)
        i = m * self.ndim + n + self.ntor
        self._dofs.iloc[i, 0] = val
        #self.rc[m, n + self.ntor] = val
        #self.recalculate = True
        self.new_x = True
        self.recalculate_derivs = True

    def set_rs(self, m, n, val):
        """
        Set a particular rs Parameter.
        """
        if self.stellsym:
            return ValueError(
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        i = m * self.ndim + n + self.ntor + 2 * self.size
        #self.rs[m, n + self.ntor] = val
        self._dofs.iloc[i, 0] = val
        #self.recalculate = True
        self.new_x = True
        self.recalculate_derivs = True

    def set_zc(self, m, n, val):
        """
        Set a particular zc Parameter.
        """
        if self.stellsym:
            return ValueError(
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        #self.zc[m, n + self.ntor] = val
        i = m * self.ndim + n + self.ntor + 3 * self.size
        self._dofs.iloc[i, 0] = val
        #self.recalculate = True
        self.new_x = True
        self.recalculate_derivs = True

    def set_zs(self, m, n, val):
        """
        Set a particular zs Parameter.
        """
        self._validate_mn(m, n)
        i = m * self.ndim + n + self.ntor + self.size
        self._dofs.iloc[i, 0] = val
        self.new_x = True
        self.recalculate_derivs = True

    def area_volume(self):
        """
        Compute the surface area and the volume enclosed by the surface.
        """
        if self.new_x:
            logger.info('Running calculation of area and volume')
        else:
            logger.info('area_volume called, but no need to recalculate')
            return

        rc = self.local_full_x[0:self.size].reshape(self.shape)
        zs = self.local_full_x[self.size:2*self.size].reshape(self.shape)

        if self.stellsym:
            rs = None
            zc = None
        else:
            #rs = self.rs
            #zc = self.zc
            rs = self.local_full_x[2*self.size:3*self.size].reshape(self.shape)
            zc = self.local_full_x[3*self.size:].reshape(self.shape)

        results = jit_area_volume_pure(rc, rs, zc, zs,
                                       self.stellsym, self.nfp, self.mpol,
                                       self.ntor, self.ntheta, self.nphi)

        self._area = float(results[0])
        self._volume = float(results[1])
        self.new_x = False
        return results

    def area(self):
        """
        Return the area of the surface.
        """
        self.area_volume()
        return self._area

    def f(self):
        return self.area_volume()

    def volume(self):
        """
        Return the volume of the surface.
        """
        self.area_volume()
        return self._volume

    return_fn_map = {'area': area, 'volume': volume}

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

        rc = self.local_full_x[0:self.size].reshape(self.shape)
        zs = self.local_full_x[self.size:2*self.size].reshape(self.shape)
        if self.stellsym:
            rs = None
            zc = None
        else:
            rs = self.local_full_x[2*self.size:3*self.size].reshape(self.shape)
            zc = self.local_full_x[3*self.size:].reshape(self.shape)

        results = darea_volume_pure(rc, rs, zc, zs,
                                    self.stellsym, self.nfp, self.mpol,
                                    self.ntor, self.ntheta, self.nphi)

        self._darea_drc = np.array(results[0][0, :, :])
        self._dvolume_drc = np.array(results[0][1, :, :])
        self._darea_dzs = np.array(results[3][0, :, :])
        self._dvolume_dzs = np.array(results[3][1, :, :])
        if not self.stellsym:
            self._darea_drs = np.array(results[1][0, :, :])
            self._dvolume_drs = np.array(results[1][1, :, :])
            self._darea_dzc = np.array(results[2][0, :, :])
            self._dvolume_dzc = np.array(results[2][1, :, :])

    def darea(self):
        """
        Return the area of the surface.
        """
        self.darea_volume()
        mpol = self.mpol  # Shorthand
        ntor = self.ntor
        if self.stellsym:
            return np.concatenate(
                (self._darea_drc[0, ntor:],
                 self._darea_drc[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._darea_dzs[0, ntor + 1:],
                 self._darea_dzs[1:, :].reshape(mpol * (ntor * 2 + 1))))
        else:
            return np.concatenate(
                (self._darea_drc[0, ntor:],
                 self._darea_drc[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._darea_dzs[0, ntor + 1:],
                 self._darea_dzs[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._darea_drs[0, ntor + 1:],
                 self._darea_drs[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._darea_dzc[0, ntor:],
                 self._darea_dzc[1:, :].reshape(mpol * (ntor * 2 + 1))))

    def dvolume(self):
        """
        Return the volume of the surface.
        """
        self.darea_volume()
        mpol = self.mpol  # Shorthand
        ntor = self.ntor
        if self.stellsym:
            return np.concatenate(
                (self._dvolume_drc[0, ntor:],
                 self._dvolume_drc[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._dvolume_dzs[0, ntor + 1:],
                 self._dvolume_dzs[1:, :].reshape(mpol * (ntor * 2 + 1))))
        else:
            return np.concatenate(
                (self._dvolume_drc[0, ntor:],
                 self._dvolume_drc[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._dvolume_dzs[0, ntor + 1:],
                 self._dvolume_dzs[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._dvolume_drs[0, ntor + 1:],
                 self._dvolume_drs[1:, :].reshape(mpol * (ntor * 2 + 1)),
                 self._dvolume_dzc[0, ntor:],
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
        stellsym = np.max(np.abs(rs)) == 0 and np.max(np.abs(zc)) == 0
        mpol = int(np.max(m))
        ntor = int(np.max(np.abs(n)))

        surf = cls(nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor)
        for j in range(Nfou):
            surf.set_rc(m[j], n[j], rc[j])
            surf.set_zs(m[j], n[j], zs[j])
            if not stellsym:
                surf.set_rs(m[j], n[j], rs[j])
                surf.set_zc(m[j], n[j], zc[j])

        return surf

    #def get_dofs(self):
    #    """
    #    Return a 1D numpy array with all the degrees of freedom.
    #    """
    #    mpol = self.mpol  # Shorthand
    #    ntor = self.ntor
    #    if self.stellsym:
    #        return np.concatenate(
    #            (self.rc[0, ntor:],
    #             self.rc[1:, :].reshape(mpol * (ntor * 2 + 1)),
    #             self.zs[0, ntor + 1:],
    #             self.zs[1:, :].reshape(mpol * (ntor * 2 + 1))))
    #    else:
    #        return np.concatenate(
    #            (self.rc[0, ntor:],
    #             self.rc[1:, :].reshape(mpol * (ntor * 2 + 1)),
    #             self.zs[0, ntor + 1:],
    #             self.zs[1:, :].reshape(mpol * (ntor * 2 + 1)),
    #             self.rs[0, ntor + 1:],
    #             self.rs[1:, :].reshape(mpol * (ntor * 2 + 1)),
    #             self.zc[0, ntor:],
    #             self.zc[1:, :].reshape(mpol * (ntor * 2 + 1))))

    #def set_dofs(self, v):
    #    """
    #    Set the shape coefficients from a 1D list/array
    #    """

    #    n = len(self.get_dofs())
    #    if len(v) != n:
    #        raise ValueError('Input vector should have ' + str(n) + \
    #                         ' elements but instead has ' + str(len(v)))

    #    # Check whether any elements actually change:
    #    if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
    #        logger.info('set_dofs called, but no dofs actually changed')
    #        return

    #    logger.info('set_dofs called, and at least one dof changed')
    #    self.recalculate = True
    #    self.recalculate_derivs = True

    #    mpol = self.mpol  # Shorthand
    #    ntor = self.ntor
    #    self.rc[0, ntor:] = v[0:ntor + 1]
    #    self.rc[1:, :] = np.array(v[ntor + 1:ntor + 1 + mpol * (ntor * 2 + 1)]).reshape(mpol, ntor * 2 + 1)
    #    self.zs[0, ntor + 1:] = v[ntor + 1 + mpol * (ntor * 2 + 1): \
    #                              ntor * 2 + 1 + mpol * (ntor * 2 + 1)]
    #    self.zs[1:, :] = np.array(v[ntor * 2 + 1 + mpol * (ntor * 2 + 1): \
    #                                       ntor * 2 + 1 + 2 * mpol * (ntor * 2 + 1)]).reshape(mpol, ntor * 2 + 1)

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
        if not self.stellsym:
            raise RuntimeError('Non-stellarator-symmetric SurfaceGarabedian '
                               'objects have not been implemented')
        mmax = self.mpol + 1
        mmin = np.min((0, 1 - self.mpol))
        s = SurfaceGarabedian(nfp=self.nfp, mmin=mmin, mmax=mmax,
                              nmin=-self.ntor, nmax=self.ntor)
        for n in range(-self.ntor, self.ntor + 1):
            for m in range(mmin, mmax + 1):
                Delta = 0
                if m - 1 >= 0:
                    Delta = 0.5 * (self.get_rc(m - 1, n) - self.get_zs(m - 1, n))
                if 1 - m >= 0:
                    Delta += 0.5 * (self.get_rc(1 - m, -n) + self.get_zs(1 - m, -n))
                s.set_Delta(m, n, Delta)

        return s


class SurfaceGarabedian(Surface, Optimizable):
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
        Surface.__init__(self, nfp=nfp, stellsym=True)
        self.mmin = mmin
        self.mmax = mmax
        self.nmin = nmin
        self.nmax = nmax
        #self.allocate()

        # Create the array for the Delta_{m,n} coefficients.
        logger.info("Allocating SurfaceGarabedian")
        self.mdim = self.mmax - self.mmin + 1
        self.ndim = self.nmax - self.nmin + 1
        self.shape = (self.mdim, self.ndim)

        Delta = np.zeros(self.shape)
        dof_names = []
        for m in range(self.mmin, self.mmax + 1):
            for n in range(self.nmin, self.nmax + 1):
                dof_names.append('Delta(' + str(m) + ',' + str(n) + ')')

        #self.new_x = True
        self.recalculate_derivs = True

        Optimizable.__init__(self, Delta.flatten(), names=dof_names)
        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.set_Delta(1, 0, 1.0)
        self.set_Delta(0, 0, 0.1)

    def __repr__(self):
        return "SurfaceGarabedian " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp) + ", mmin=" + str(self.mmin) + ", mmax=" + str(self.mmax) \
            + ", nmin=" + str(self.nmin) + ", nmax=" + str(self.nmax) \
            + ")"

    @property
    def Delta(self):
        return self.local_full_x.reshape(self.shape)

    @Delta.setter
    def Delta(self, Delta):
        assert(self.shape == Delta.shape)
        self._dofs.loc[:, 'x'] = Delta.flatten()

    def get_Delta(self, m, n):
        """
        Return a particular Delta_{m,n} coefficient.
        """
        return self.Delta[m - self.mmin, n - self.nmin]

    def set_Delta(self, m, n, val):
        """
        Set a particular Delta_{m,n} coefficient.
        """
        # Not sure if the below statement will not lead to a copy under all
        # circumstances. TODO: Investigate this
        #self.Delta[m - self.mmin, n - self.nmin] = val

        # Using a safer approach
        i = self.ndim * (m - self.mmin) + n - self.nmin
        self._dofs.iloc[i, 0] = val
        self.new_x = True
        self.recalculate_derivs = True

    #def get_dofs(self):
    #    """
    #    Return a 1D numpy array with all the degrees of freedom.
    #    """
    #    num_dofs = (self.mmax - self.mmin + 1) * (self.nmax - self.nmin + 1)
    #    return np.reshape(self.Delta, (num_dofs,), order='F')

    #def set_dofs(self, v):
    #    """
    #    Set the shape coefficients from a 1D list/array
    #    """

    #    n = len(self.get_dofs())
    #    if len(v) != n:
    #        raise ValueError('Input vector should have ' + str(n) + \
    #                         ' elements but instead has ' + str(len(v)))

    #    # Check whether any elements actually change:
    #    if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
    #        logger.info('set_dofs called, but no dofs actually changed')
    #        return

    #    logger.info('set_dofs called, and at least one dof changed')
    #    self.recalculate = True
    #    self.recalculate_derivs = True

    #    self.Delta = v.reshape((self.mmax - self.mmin + 1, self.nmax - self.nmin + 1), order='F')

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier object with the identical shape.

        For a derivation of the transformation here, see 
        https://terpconnect.umd.edu/~mattland/assets/notes/toroidal_surface_parameterizations.pdf
        """
        mpol = int(np.max((1, self.mmax - 1, 1 - self.mmin)))
        ntor = int(np.max((self.nmax, -self.nmin)))
        s = SurfaceRZFourier(nfp=self.nfp, stellsym=True, mpol=mpol, ntor=ntor)
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
        if self.new_x:
            logger.info('Running calculation of area and volume')
        else:
            logger.info('area_volume called, but no need to recalculate')
            return

        self.new_x = False

        # Delegate to the area and volume calculations of SurfaceRZFourier():
        s = self.to_RZFourier()
        #self._area = s.area()
        #self._volume = s.volume()
        return s.area_volume()

    def f(self):
        return self.area_volume()

    def area(self):
        """
        Return the area of the surface.
        """
        return self.area_volume()[0]
        #return self._area

    def volume(self):
        """
        Return the volume of the surface.
        """
        return self.area_volume()[1]
        #return self._volume

    return_fn_map = {'area': area, 'volume': volume}

