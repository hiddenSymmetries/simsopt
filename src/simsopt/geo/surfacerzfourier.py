import numpy as np
import simsgeopp as sgpp
from .surface import Surface


class SurfaceRZFourier(sgpp.SurfaceRZFourier, Surface):
    r"""`SurfaceRZFourier` is a surface that is represented in cylindrical
    coordinates using the following Fourier series:
    
    .. math::
           r(\theta, \phi) = \sum_{m=0}^{m_{\text{pol}}} \sum_{n=-n_{\text{tor}}}^{n_\text{tor}} [
               r_{c,m,n} \cos(m \theta - n_{\text{fp}} n \phi)
               + r_{s,m,n} \sin(m \theta - n_{\text{fp}} n \phi) ]

    and the same for :math:`z(\theta, \phi)`.
    
    Here, :math:`(r,\phi, z)` are standard cylindrical coordinates, and theta
    is any poloidal angle.
    
    Note that for :math:`m=0` we skip the :math:`n<0` term for the cos terms, and the :math:`n \leq 0`
    for the sin terms.
    
    In addition, in the ``stellsym=True`` case, we skip the sin terms for :math:`r`, and
    the cos terms for :math:`z`.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0, quadpoints_phi=63, quadpoints_theta=62):
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
            quadpoints_theta = list(quadpoints_theta)
        sgpp.SurfaceRZFourier.__init__(self, mpol, ntor, nfp, stellsym, quadpoints_phi, quadpoints_theta)
        self.rc[0, ntor] = 1.0
        self.rc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        Surface.__init__(self)
        self.make_names()

    def get_dofs(self):
        """
        Return the dofs associated to this surface.
        """
        return np.asarray(sgpp.SurfaceRZFourier.get_dofs(self))

    def make_names(self):
        """
        Form a list of names of the `rc`, `zs`, `rs`, or `zc` array elements.
        """
        self.names = self.make_names_helper('rc', True) + self.make_names_helper('zs', False)
        if not self.stellsym:
            self.names += self.make_names_helper('rs', False) + self.make_names_helper('zc', True)

    def make_names_helper(self, prefix, include0):
        if include0:
            names = [prefix + "(0,0)"]
        else:
            names = []

        names += [prefix + '(0,' + str(n) + ')' for n in range(1, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names

    @classmethod
    def from_focus(cls, filename, nphi=32, ntheta=32):
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

        surf = cls(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, quadpoints_phi=nphi, quadpoints_theta=ntheta)
        for j in range(Nfou):
            surf.rc[m[j], n[j] + ntor] = rc[j]
            surf.zs[m[j], n[j] + ntor] = zs[j]
            if not stellsym:
                surf.rs[m[j], n[j] + ntor] = rs[j]
                surf.zc[m[j], n[j] + ntor] = zc[j]

        return surf

    def change_resolution(self, mpol, ntor):
        """
        Change the values of `mpol` and `ntor`. Any new Fourier amplitudes
        will have a magnitude of zero.  Any previous nonzero Fourier
        amplitudes that are not within the new range will be
        discarded.
        """
        old_mpol = self.mpol
        old_ntor = self.ntor
        old_rc = self.rc
        old_zs = self.zs
        if not self.stellsym:
            old_rs = self.rs
            old_zc = self.zc
        self.mpol = mpol
        self.ntor = ntor
        self.allocate()
        if mpol < old_mpol or ntor < old_ntor:
            self.invalidate_cache()

        min_mpol = np.min((mpol, old_mpol))
        min_ntor = np.min((ntor, old_ntor))
        for m in range(min_mpol + 1):
            for n in range(-min_ntor, min_ntor + 1):
                self.rc[m, n + ntor] = old_rc[m, n + old_ntor]
                self.zs[m, n + ntor] = old_zs[m, n + old_ntor]
                if not self.stellsym:
                    self.rs[m, n + ntor] = old_rs[m, n + old_ntor]
                    self.zc[m, n + ntor] = old_zc[m, n + old_ntor]
        self.make_names()

    def to_RZFourier(self):
        """
        No conversion necessary.
        """
        return self

    def __repr__(self):
        return "SurfaceRZFourier " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp) + ", stellsym=" + str(self.stellsym) + \
            ", mpol=" + str(self.mpol) + ", ntor=" + str(self.ntor) \
            + ")"

    def _validate_mn(self, m, n):
        """
        Check whether `m` and `n` are in the allowed range.
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
        Return a particular `rc` Parameter.
        """
        self._validate_mn(m, n)
        return self.rc[m, n + self.ntor]

    def get_rs(self, m, n):
        """
        Return a particular `rs` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.rs[m, n + self.ntor]

    def get_zc(self, m, n):
        """
        Return a particular `zc` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.zc[m, n + self.ntor]

    def get_zs(self, m, n):
        """
        Return a particular `zs` Parameter.
        """
        self._validate_mn(m, n)
        return self.zs[m, n + self.ntor]

    def set_rc(self, m, n, val):
        """
        Set a particular `rc` Parameter.
        """
        self._validate_mn(m, n)
        self.rc[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def set_rs(self, m, n, val):
        """
        Set a particular `rs` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.rs[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def set_zc(self, m, n, val):
        """
        Set a particular `zc` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.zc[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def set_zs(self, m, n, val):
        """
        Set a particular `zs` Parameter.
        """
        self._validate_mn(m, n)
        self.zs[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of `m` and `n` values.

        All modes with `m` in the interval [`mmin`, `mmax`] and `n` in the
        interval [`nmin`, `nmax`] will have their fixed property set to
        the value of the `fixed` parameter. Note that `mmax` and `nmax`
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        for m in range(mmin, mmax + 1):
            this_nmin = nmin
            if m == 0 and nmin < 0:
                this_nmin = 0
            for n in range(this_nmin, nmax + 1):
                self.set_fixed('rc({},{})'.format(m, n), fixed)
                if m > 0 or n != 0:
                    self.set_fixed('zs({},{})'.format(m, n), fixed)
                if not self.stellsym:
                    self.set_fixed('zc({},{})'.format(m, n), fixed)
                    if m > 0 or n != 0:
                        self.set_fixed('rs({},{})'.format(m, n), fixed)

    def to_Garabedian(self):
        """
        Return a `SurfaceGarabedian` object with the identical shape.

        For a derivation of the transformation here, see 
        https://terpconnect.umd.edu/~mattland/assets/notes/toroidal_surface_parameterizations.pdf
        """
        if not self.stellsym:
            raise RuntimeError('Non-stellarator-symmetric SurfaceGarabedian objects have not been implemented')
        from simsopt.geo.surfacegarabedian import SurfaceGarabedian
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

    def set_dofs(self, dofs):
        sgpp.SurfaceRZFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()

    def darea(self):
        """ 
        Short hand for `Surface.darea_by_dcoeff()`
        """
        return self.darea_by_dcoeff()

    def dvolume(self):
        """
        Short hand for `Surface.dvolume_by_dcoeff()`
        """
        return self.dvolume_by_dcoeff()

    def set_dofs(self, dofs):
        sgpp.SurfaceRZFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()
