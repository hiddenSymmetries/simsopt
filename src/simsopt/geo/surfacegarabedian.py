import numpy as np
import logging

from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)


class SurfaceGarabedian(Surface):
    """
    `SurfaceGarabedian` represents a toroidal surface for which the
    shape is parameterized using Garabedian's :math:`\Delta_{m,n}`
    coefficients.

    The present implementation assumes stellarator symmetry. Note that
    non-stellarator-symmetric surfaces require that the :math:`\Delta_{m,n}`
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
        self.mmin = mmin
        self.mmax = mmax
        self.nmin = nmin
        self.nmax = nmax
        self.nfp = nfp
        self.stellsym = True
        self.allocate()
        self.recalculate = True
        self.recalculate_derivs = True

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.set_Delta(1, 0, 1.0)
        self.set_Delta(0, 0, 0.1)
        Surface.__init__(self)

    def __repr__(self):
        return "SurfaceGarabedian " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp) + ", mmin=" + str(self.mmin) + ", mmax=" + str(self.mmax) \
            + ", nmin=" + str(self.nmin) + ", nmax=" + str(self.nmax) \
            + ")"

    def allocate(self):
        """
        Create the array for the :math:`\Delta_{m,n}` coefficients.
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
        Return a particular :math:`\Delta_{m,n}` coefficient.
        """
        return self.Delta[m - self.mmin, n - self.nmin]

    def set_Delta(self, m, n, val):
        """
        Set a particular :math:`\Delta_{m,n}` coefficient.
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

