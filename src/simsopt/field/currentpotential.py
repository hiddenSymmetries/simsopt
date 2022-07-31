import numpy as np
from .._core.optimizable import DOFs, Optimizable
import simsoptpp as sopp
from simsopt.geo.surface import Surface

__all__ = ['CurrentPotentialFourier', 'CurrentPotential']

class CurrentPotential(Optimizable):

    def set_points(self, points):
        return self.set_points(points)

    def __init__(self, **kwargs):
        Optimizable.__init__(self, **kwargs)

class CurrentPotentialFourier(sopp.CurrentPotentialFourier, CurrentPotential):

    def __init__(self, winding_surface, nfp=1, stellsym=True, mpol=1, ntor=0, nphi=None,
                    ntheta=None, range="full torus",
                    quadpoints_phi=None, quadpoints_theta=None):

        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(nfp=nfp,
                                                                  nphi=nphi, ntheta=ntheta, range=range,
                                                                  quadpoints_phi=quadpoints_phi,
                                                                  quadpoints_theta=quadpoints_theta)

        sopp.CurrentPotentialFourier.__init__(self, winding_surface, mpol, ntor, nfp, stellsym,
                                       quadpoints_phi, quadpoints_theta)
        CurrentPotential.__init__(self, x0=self.get_dofs(),
                         external_dof_setter=CurrentPotentialFourier.set_dofs_impl,
                         names=self._make_names())
        self._make_mn()

    def _make_names(self):
        if self.stellsym:
            names = self._make_names_helper('Phis', False)
        else:
            names = self._make_names_helper('Phis', False) \
                  + self._make_names_helper('Phic', True)
        return names

    def _make_names_helper(self, prefix, include0):
        if include0:
            names = [prefix + "(0,0)"]
        else:
            names = []

        names += [prefix + '(0,' + str(n) + ')' for n in range(1, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        fn = self.fix if fixed else self.unfix
        for m in range(mmin, mmax + 1):
            this_nmin = nmin
            if m == 0 and nmin < 0:
                this_nmin = 0
            for n in range(this_nmin, nmax + 1):
                if not self.stellsym:
                    fn(f'Phic({m},{n})')
                if m > 0 or n != 0:
                    fn(f'Phis({m},{n})')

    def get_dofs(self):
        return np.asarray(sopp.CurrentPotentialFourier.get_dofs(self))

    def set_dofs(self, dofs):
        self.local_full_x = dofs

    def change_resolution(self, mpol, ntor):
        """
        Modeled after SurfaceRZFourier
        Change the values of `mpol` and `ntor`. Any new Fourier amplitudes
        will have a magnitude of zero.  Any previous nonzero Fourier
        amplitudes that are not within the new range will be
        discarded.
        """
        old_mpol = self.mpol
        old_ntor = self.ntor
        old_phis = self.phis
        if not self.stellsym:
            old_phic = self.phic
        self.mpol = mpol
        self.ntor = ntor
        self.allocate()
        if mpol < old_mpol or ntor < old_ntor:
            self.invalidate_cache()

        min_mpol = np.min((mpol, old_mpol))
        min_ntor = np.min((ntor, old_ntor))
        for m in range(min_mpol + 1):
            for n in range(-min_ntor, min_ntor + 1):
                self.phis[m, n + ntor] = old_phis[m, n + old_ntor]
                if not self.stellsym:
                    self.phic[m, n + ntor] = old_phic[m, n + old_ntor]

        # Update the dofs object
        self._dofs = DOFs(self.get_dofs(), self._make_names())
        # The following methods of graph Optimizable framework need to be called
        Optimizable._update_free_dof_size_indices(self)
        Optimizable._update_full_dof_size_indices(self)
        Optimizable._set_new_x(self)

    def get_phic(self, m, n):
        """
        Return a particular `phic` parameter.
        """
        if self.stellsym:
            return ValueError(
                'phic does not exist for this stellarator-symmetric current potential.')
        self._validate_mn(m, n)
        return self.phic[m, n + self.ntor]

    def get_phis(self, m, n):
        """
        Return a particular `phis` parameter.
        """
        self._validate_mn(m, n)
        return self.phis[m, n + self.ntor]

    def set_phic(self, m, n, val):
        """
        Set a particular `phic` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'phic does not exist for this stellarator-symmetric current potential.')
        self._validate_mn(m, n)
        self.phic[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def set_phis(self, m, n, val):
        """
        Set a particular `phis` Parameter.
        """
        self._validate_mn(m, n)
        self.phis[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Modeled after SurfaceRZFourier
        Set the 'fixed' property for a range of `m` and `n` values.

        All modes with `m` in the interval [`mmin`, `mmax`] and `n` in the
        interval [`nmin`, `nmax`] will have their fixed property set to
        the value of the `fixed` parameter. Note that `mmax` and `nmax`
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        # TODO: This will be slow because free dof indices are evaluated all
        # TODO: the time in the loop
        fn = self.fix if fixed else self.unfix
        for m in range(mmin, mmax + 1):
            this_nmin = nmin
            if m == 0 and nmin < 0:
                this_nmin = 0
            for n in range(this_nmin, nmax + 1):
                if not self.stellsym:
                    fn(f'phic({m},{n})')
                if m > 0 or n != 0:
                    fn(f'phis({m},{n})')

    def _validate_mn(self, m, n):
        """
        Copied from SurfaceRZFourier
        Check whether `m` and `n` are in the allowed range.
        """
        if m < 0:
            raise IndexError('m must be >= 0')
        if m > self.mpol:
            raise IndexError('m must be <= mpol')
        if n > self.ntor:
            raise IndexError('n must be <= ntor')
        if n < -self.ntor:
            raise IndexError('n must be >= -ntor')

    def _make_mn(self):
        """
        Make the list of m and n values.
        """
        m1d = np.arange(self.mpol + 1)
        n1d = np.arange(-self.ntor, self.ntor + 1)
        n2d, m2d = np.meshgrid(n1d, m1d)
        m0 = m2d.flatten()[self.ntor:]
        n0 = n2d.flatten()[self.ntor:]
        if not self.stellsym:
            m = np.concatenate((m0[1:], m0))
            n = np.concatenate((n0[1:], n0))
        else:
            m = m0[1::]
            n = n0[1::]
        self.m = m
        self.n = n
