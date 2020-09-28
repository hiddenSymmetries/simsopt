"""
This module provides several classes for representing toroidal
surfaces.  There is a base class Surface, and several child classes
corresponding to different discrete representations.
"""

import numpy as np
import logging
from .util import isbool

class Surface:
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
        self.logger = logging.getLogger(__name__)

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
        # Perform some validation.
        if not isinstance(mpol, int):
            raise TypeError("mpol must have type int")
        if not isinstance(ntor, int):
            raise TypeError("ntor must have type int")
        if mpol < 1:
            raise ValueError("mpol must be at least 1")
        if ntor < 0:
            raise ValueError("ntor must be at least 0")
        self.mpol = mpol
        self.ntor = ntor
        Surface.__init__(self, nfp=nfp, stelsym=stelsym)
        self.logger = logging.getLogger(__name__)
        self.allocate()
        self.recalculate = True

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
        self.logger.info("Allocating SurfaceRZFourier")
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

    def area_volume(self):
        """
        Compute the surface area and the volume enclosed by the surface.
        """
        if self.recalculate:
            self.logger.info('Running calculation of area and volume')
        else:
            self.logger.info('area_volume called, but no need to recalculate')
            return

        self.recalculate = False
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

    @property
    def area(self):
        """
        Return the area of the surface.
        """
        self.area_volume()
        return self._area

    @property
    def volume(self):
        """
        Return the volume of the surface.
        """
        self.area_volume()
        return self._volume

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
            self.logger.info('set_dofs called, but no dofs actually changed')
            return

        self.logger.info('set_dofs called, and at least one dof changed')
        self.recalculate = True
        
        mpol = self.mpol # Shorthand
        ntor = self.ntor
        self.rc[0, ntor:] = v[0:ntor + 1]
        self.rc[1:, :] = np.array(v[ntor + 1:ntor + 1 + mpol * (ntor * 2 + 1)]).reshape(mpol, ntor * 2 + 1)
        self.zs[0, ntor + 1:] = v[ntor + 1 + mpol * (ntor * 2 + 1): \
                                           ntor * 2 + 1 + mpol * (ntor * 2 + 1)]
        self.zs[1:, :] = np.array(v[ntor * 2 + 1 + mpol * (ntor * 2 + 1): \
                                           ntor * 2 + 1 + 2 * mpol * (ntor * 2 + 1)]).reshape(mpol, ntor * 2 + 1)
