"""
This module provides several classes for representing toroidal
surfaces.  There is a base class Surface, and several child classes
corresponding to different discrete representations.
"""

import numpy as np
from .parameter import Parameter, ParameterArray
from .shape import Shape
from .target import Target
import logging

class Surface(Shape):
    """
    Surface is a base class for various representations of toroidal
    surfaces in simsopt.
    """

    def __init__(self, nfp=1, stelsym=True):
        # Shape handles validation of the arguments
        Shape.__init__(self, nfp, stelsym)

    def __repr__(self):
        return "Surface " + str(hex(id(self))) + " (nfp=" + str(self._nfp.val) \
            + ", stelsym=" + str(self._stelsym.val) + ")"

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
            raise RuntimeError("mpol must have type int")
        if not isinstance(ntor, int):
            raise RuntimeError("ntor must have type int")
        if mpol < 1:
            raise RuntimeError("mpol must be at least 1")
        if ntor < 0:
            raise RuntimeError("ntor must be at least 0")
        self.mpol = Parameter(mpol, min=1, name="mpol for SurfaceRZFourier " \
                                  + str(hex(id(self))))
        self.ntor = Parameter(ntor, min=0, name="ntor for SurfaceRZFourier " \
                                  + str(hex(id(self))))
        Surface.__init__(self, nfp=nfp, stelsym=stelsym)
        self.allocate()

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.get_rc(0,0).val = 1.0
        self.get_rc(1,0).val = 0.1
        self.get_zs(1,0).val = 0.1

        # Resolution for computing area, volume, etc:
        self.ntheta = 63
        self.nphi = 62

    def _generate_names(self, prefix):
        """
        Generate the names for the Parameter objects.
        """
        assert(type(prefix) is str)
        self.mdim = self.mpol.val + 1
        self.ndim = 2 * self.ntor.val + 1
        objstr = " for SurfaceRZFourier " + str(hex(id(self)))
        names = []
        for m in range(self.mdim):
            namess = []
            for jn in range(self.ndim):
                newstr = prefix + "(m={: 04d},n={: 04d})".format(\
                    m,jn - self.ntor.val) + objstr
                namess.append(newstr)
            names.append(namess)
        return np.array(names)

    def allocate(self):
        """
        Create the ParameterArrays for the rc, rs, zc, and zs coefficients.
        """
        logger = logging.getLogger(__name__)
        logger.info("Allocating SurfaceRZFourier")
        self.mdim = self.mpol.val + 1
        self.ndim = 2 * self.ntor.val + 1
        myshape = (self.mdim, self.ndim)

        self.rc = ParameterArray(np.zeros(myshape), \
                                     name=self._generate_names("rc"))
        self.zs = ParameterArray(np.zeros(myshape), \
                                     name=self._generate_names("zs"))

        if not self.stelsym.val:
            self.rs = ParameterArray(np.zeros(myshape), \
                                         name=self._generate_names("rs"))
            self.zc = ParameterArray(np.zeros(myshape), \
                                         name=self._generate_names("zc"))

        # Create a set of all the surface Parameters, which will be
        # used for Targets that depend on this surface.
        params = {self.nfp, self.stelsym, self.mpol, self.ntor}
        params = params.union(set(self.rc.data.flat))
        params = params.union(set(self.zs.data.flat))
        if not self.stelsym.val:
            params = params.union(set(self.rs.data.flat))
            params = params.union(set(self.zc.data.flat))

        self.area = Target(params, self.compute_area)
        self.volume = Target(params, self.compute_volume)

    def __repr__(self):
        return "SurfaceRZFourier " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp.val) + ", stelsym=" + str(self.stelsym.val) + \
            ", mpol=" + str(self.mpol.val) + ", ntor=" + str(self.ntor.val) \
            + ")"

    def _validate_mn(self, m, n):
        """
        Check whether m and n are in the allowed range.
        """
        if m < 0:
            raise ValueError('m must be >= 0')
        if m > self.mpol.val:
            raise ValueError('m must be <= mpol')
        if n > self.ntor.val:
            raise ValueError('n must be <= ntor')
        if n < -self.ntor.val:
            raise ValueError('n must be >= -ntor')
    
    def get_rc(self, m, n):
        """
        Return a particular rc Parameter.
        """
        self._validate_mn(m, n)
        return self.rc.data[m, n + self.ntor.val]

    def get_rs(self, m, n):
        """
        Return a particular rs Parameter.
        """
        if self.stelsym.val:
            return ValueError( \
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.rs.data[m, n + self.ntor.val]

    def get_zc(self, m, n):
        """
        Return a particular zc Parameter.
        """
        if self.stelsym.val:
            return ValueError( \
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.zc.data[m, n + self.ntor.val]

    def get_zs(self, m, n):
        """
        Return a particular zs Parameter.
        """
        self._validate_mn(m, n)
        return self.zs.data[m, n + self.ntor.val]

    def area_volume(self):
        """
        Compute the surface area and the volume enclosed by the surface.
        """
        ntheta = self.ntheta # Shorthand
        nphi = self.nphi
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / self.nfp.val, nphi, \
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
        mdim = self.mpol.val + 1
        ndim = 2 * self.ntor.val + 1
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        nfp = self.nfp.val
        for m in range(mdim):
            for jn in range(ndim):
                # Presently this loop includes negative n when m=0.
                # This is unnecesary but doesn't hurt I think.
                n_without_nfp = jn - self.ntor.val
                n = n_without_nfp * nfp
                angle = m * theta - n * phi
                sinangle = np.sin(angle)
                cosangle = np.cos(angle)
                rmnc = self.get_rc(m, n_without_nfp).val
                zmns = self.get_zs(m, n_without_nfp).val
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
                if not self.stelsym.val:
                    rmns = self.get_rs(m, n_without_nfp).val
                    zmnc = self.get_zc(m, n_without_nfp).val
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
        area = nfp * dtheta * dphi * np.sum(np.sum(norm_normal))
        # Compute plasma volume using \int (1/2) R^2 dZ dphi
        # = \int (1/2) R^2 (dZ/dtheta) dtheta dphi
        volume = 0.5 * nfp * dtheta * dphi * np.sum(np.sum(r * r * dzdtheta))
        return (area, volume)

    def compute_area(self):
        """
        Return the area of the surface.
        """
        area, volume = self.area_volume()
        return area

    def compute_volume(self):
        """
        Return the volume of the surface.
        """
        area, volume = self.area_volume()
        return volume

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
            surf.rc.data[m[j], n[j] + ntor].val = rc[j]
            surf.zs.data[m[j], n[j] + ntor].val = zs[j]
            if not stelsym:
                surf.rs.data[m[j], n[j] + ntor].val = rs[j]
                surf.zc.data[m[j], n[j] + ntor].val = zc[j]

        return surf
