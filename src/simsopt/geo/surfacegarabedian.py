import numpy as np
import logging

import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)

__all__ = ['SurfaceGarabedian']


class SurfaceGarabedian(sopp.Surface, Surface):
    r"""
    ``SurfaceGarabedian`` represents a toroidal surface for which the
    shape is parameterized using Garabedian's :math:`\Delta_{m,n}`
    coefficients:

    .. math::
      R + i Z = e^{i u} \sum_{m = m_\min}^{m_\max} \sum_{n = n_\min}^{n_\max} \Delta_{m,n} e^{-i m u + i n v}

    where :math:`u = 2 \pi \theta` is a poloidal angle on :math:`[0, 2\pi]`, and
    :math:`v` is the standard toroidal angle on :math:`[0, 2\pi]`.

    The present implementation assumes stellarator symmetry. Note that
    non-stellarator-symmetric surfaces require that the :math:`\Delta_{m,n}`
    coefficients be imaginary.

    For more information about the arguments ``quadpoints_phi``, and
    ``quadpoints_theta``, see the general documentation on :ref:`surfaces`.
    Instead of supplying the quadrature point arrays along :math:`\phi` and
    :math:`\theta` directions, one could also specify the number of
    quadrature points for :math:`\phi` and :math:`\theta` using the
    class method :py:meth:`~simsopt.geo.surface.Surface.from_nphi_ntheta`.

    Args:
        nfp: The number of field periods.
        mmin: Minimum poloidal mode number :math:`m` included (usually 0 or negative).
        mmax: Maximum poloidal mode number :math:`m` included.
        nmin: Minimum toroidal mode number :math:`n` included (usually negative).
          If ``None``, ``nmin = -nmax`` will be used.
        nmax: Maximum toroidal mode number :math:`n` included.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, nfp=1, mmax=1, mmin=0, nmax=0, nmin=None,
                 quadpoints_phi=None, quadpoints_theta=None):
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

        self.mdim = self.mmax - self.mmin + 1
        self.ndim = self.nmax - self.nmin + 1
        self.shape = (self.mdim, self.ndim)

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        Delta = np.zeros(self.shape)
        sopp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        Surface.__init__(self, x0=Delta.ravel(),
                         names=self._make_dof_names())

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.set_Delta(1, 0, 1.0)
        self.set_Delta(0, 0, 0.1)

    def _make_dof_names(self):
        names = []
        for m in range(self.mmin, self.mmax + 1):
            for n in range(self.nmin, self.nmax + 1):
                names.append(f'Delta({m},{n})')
        return names

    def __repr__(self):
        return self.name + f" (nfp={self.nfp}, " + \
            f"mmin={self.mmin}, mmax={self.mmax}" + \
            f", nmin={self.nmin}, nmax={self.nmax})"

    @property
    def Delta(self):
        return self.local_full_x.reshape(self.shape)

    @Delta.setter
    def Delta(self, Delta):
        assert (self.shape == Delta.shape)
        self.local_full_x = Delta.flatten()

    def get_Delta(self, m, n):
        """
        Return a particular :math:`\Delta_{m,n}` coefficient.
        """
        return self.Delta[m - self.mmin, n - self.nmin]

    def set_Delta(self, m, n, val):
        """
        Set a particular :math:`\Delta_{m,n}` coefficient.
        """
        i = self.ndim * (m - self.mmin) + n - self.nmin
        self.set(i, val)

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        self.local_full_x

    def set_dofs(self, x):
        """
        Set the shape coefficients from a 1D list/array
        """
        # Check whether any elements actually change:
        if np.all(np.abs(self.get_dofs() - np.array(x)) == 0):
            logger.info('set_dofs called, but no dofs actually changed')
            return

        logger.info('set_dofs called, and at least one dof changed')

        self.local_full_x = x

    def fix_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Fix the DOFs for a range of m and n values.

        All modes with m in the interval [mmin, mmax] and n in the
        interval [nmin, nmax] will have their fixed property set to
        the value of the 'fixed' parameter. Note that mmax and nmax
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        fn = self.fix if fixed else self.unfix
        for m in range(mmin, mmax + 1):
            for n in range(nmin, nmax + 1):
                fn(f'Delta({m},{n})')

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

    # TODO: Reimplement by passing all Delta values once
    @classmethod
    def from_RZFourier(cls, surf):
        """
        Create a `SurfaceGarabedian` from a `SurfaceRZFourier` object of the identical shape.

        For a derivation of the transformation here, see
        https://terpconnect.umd.edu/~mattland/assets/notes/toroidal_surface_parameterizations.pdf
        """
        if not surf.stellsym:
            raise RuntimeError('Non-stellarator-symmetric SurfaceGarabedian '
                               'objects have not been implemented')
        mmax = surf.mpol + 1
        mmin = np.min((0, 1 - surf.mpol))
        s = cls(nfp=surf.nfp, mmin=mmin, mmax=mmax,
                nmin=-surf.ntor, nmax=surf.ntor)
        for n in range(-surf.ntor, surf.ntor + 1):
            for m in range(mmin, mmax + 1):
                Delta = 0
                if m - 1 >= 0:
                    Delta = 0.5 * (surf.get_rc(m - 1, n) - surf.get_zs(m - 1, n))
                if 1 - m >= 0:
                    Delta += 0.5 * (surf.get_rc(1 - m, -n) + surf.get_zs(1 - m, -n))
                s.set_Delta(m, n, Delta)

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

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["mmax"] = self.mmax
        d["mmin"] = self.mmin
        d["nmax"] = self.nmax
        d["nmin"] = self.nmin
        return d

    @classmethod
    def from_dict(cls, d):
        surf = cls(nfp=d["nfp"], mmax=d["mmax"], mmin=d["mmin"],
                   nmax=d["nmax"], nmin=d["nmin"],
                   quadpoints_phi=d["quadpoints_phi"],
                   quadpoints_theta=d["quadpoints_theta"])
        surf.set_dofs(d["x0"])
        return surf

    return_fn_map = {'area': area,
                     'volume': volume,
                     'aspect-ratio': Surface.aspect_ratio}
