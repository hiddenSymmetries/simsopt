import numpy as np
from .._core.optimizable import Optimizable
import simsoptpp as sopp

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

    def _make_mn(self):
        """
        Make the list of m and n values.
        """
        m1d = np.arange(self.mpol + 1)
        n1d = np.arange(-self.ntor, self.ntor + 1)
        n2d, m2d = np.meshgrid(n1d, m1d)
        m0 = m2d.flatten()[self.ntor:]
        n0 = n2d.flatten()[self.ntor:]
        m = np.concatenate((m0, m0[1:]))
        n = np.concatenate((n0, n0[1:]))
        if not self.stellsym:
            m = np.concatenate((m, m))
            n = np.concatenate((n, n))
        self.m = m
        self.n = n

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
