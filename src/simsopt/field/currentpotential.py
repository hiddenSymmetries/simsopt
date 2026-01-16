import numpy as np
from .._core.optimizable import DOFs, Optimizable
import simsoptpp as sopp
from simsopt.geo import Surface, SurfaceRZFourier
from scipy.io import netcdf_file
import warnings

__all__ = ['CurrentPotentialFourier', 'CurrentPotential']


class CurrentPotential(Optimizable):
    """
    Current Potential base object, not necessarily assuming
    that the current potential will be represented by a
    Fourier expansion in the toroidal and poloidal modes.
    Args:
        winding_surface: SurfaceRZFourier object representing the coil surface.
    """

    def set_points(self, points):
        return self.set_points(points)

    def __init__(self, winding_surface, **kwargs):
        super().__init__(**kwargs)
        self.winding_surface = winding_surface

    def K(self):
        data = np.zeros((len(self.quadpoints_phi), len(self.quadpoints_theta), 3))
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.K_impl_helper(data, dg1, dg2, normal)
        return data

    def K_matrix(self):
        data = np.zeros((len(self.num_dofs), len(self.num_dofs)))
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.K_matrix_impl_helper(data, dg1, dg2, normal)
        return data

    def num_dofs(self):
        return len(self.get_dofs())


class CurrentPotentialFourier(sopp.CurrentPotentialFourier, CurrentPotential):
    """
    Current Potential Fourier object is designed for initializing
    the winding surface problem, assuming that the current potential
    will be represented by a Fourier expansion in the toroidal
    and poloidal modes.

    Args:
        winding_surface: SurfaceRZFourier object representing the coil surface.
        net_poloidal_current_amperes: Net poloidal current in amperes, needed
            to compute the B_GI contributions to the Bnormal part of the optimization.
        net_toroidal_current_amperes: Net toroidal current in amperes, needed
            to compute the B_GI contributions to the Bnormal part of the optimization.
        nfp: The number of field periods.
        stellsym: Whether the surface is stellarator-symmetric, i.e.
          symmetry under rotation by :math:`\pi` about the x-axis.
        mpol: Maximum poloidal mode number included.
        ntor: Maximum toroidal mode number included, divided by ``nfp``.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, winding_surface, net_poloidal_current_amperes=1,
                 net_toroidal_current_amperes=0, nfp=None, stellsym=None,
                 mpol=None, ntor=None,
                #  quadpoints_phi=None, quadpoints_theta=None
                 ):

        if nfp is None:
            nfp = winding_surface.nfp
        if stellsym is None:
            stellsym = winding_surface.stellsym
        if mpol is None:
            mpol = winding_surface.mpol
        if ntor is None:
            ntor = winding_surface.ntor

        if np.max(winding_surface.quadpoints_phi)<=1/nfp:
            raise AttributeError('winding_surface must contain all field periods.')

        # quadpoints_phi or quadpoints_theta has to be the same as those 
        # in the winding surface. Otherwise, it can cause issues during 
        # CurrentPotential.K(), CurrentPotential.K_matrix and 
        # in WindingSurfaceField.__init__().
        # if quadpoints_theta is None:
        #     quadpoints_theta = winding_surface.quadpoints_theta
        # if quadpoints_phi is None:
        #     quadpoints_phi = winding_surface.quadpoints_phi

        sopp.CurrentPotentialFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                            #   quadpoints_phi, quadpoints_theta, 
                                              winding_surface.quadpoints_phi, winding_surface.quadpoints_theta, 
                                              net_poloidal_current_amperes,
                                              net_toroidal_current_amperes)

        CurrentPotential.__init__(self, winding_surface, x0=self.get_dofs(),
                                  external_dof_setter=CurrentPotentialFourier.set_dofs_impl,
                                  names=self._make_names())

        self._make_mn()
        phi_secular, theta_secular = np.meshgrid(
            self.winding_surface.quadpoints_phi,
            self.winding_surface.quadpoints_theta,
            indexing='ij'
        )
        self.current_potential_secular = (
            phi_secular * net_poloidal_current_amperes + theta_secular * net_toroidal_current_amperes
        )

    def _make_names(self):
        if self.stellsym:
            names = self._make_names_helper('Phis')
        else:
            names = self._make_names_helper('Phis') \
                + self._make_names_helper('Phic')
        return names

    def _make_names_helper(self, prefix):
        names = []

        start = 1
        names += [prefix + '(0,' + str(n) + ')' for n in range(start, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names

    def get_dofs(self):
        return np.asarray(sopp.CurrentPotentialFourier.get_dofs(self))

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
        Optimizable.update_free_dof_size_indices(self)
        Optimizable._update_full_dof_size_indices(self)
        Optimizable.set_recompute_flag(self)

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
        self.invalidate_cache()

    def set_phis(self, m, n, val):
        """
        Set a particular `phis` Parameter.
        """
        self._validate_mn(m, n)
        self.phis[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()
        self.invalidate_cache()

    def set_net_toroidal_current_amperes(self, val):
        self.net_toroidal_current_amperes = val
        self.invalidate_cache()

    def set_net_poloidal_current_amperes(self, val):
        self.net_poloidal_current_amperes = val
        self.invalidate_cache()

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
                this_nmin = 1
            for n in range(this_nmin, nmax + 1):
                if m > 0 or n != 0:
                    fn(f'phis({m},{n})')
                    if not self.stellsym:
                        fn(f'phic({m},{n})')

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
        self.m = m0[1::]
        self.n = n0[1::]

        if not self.stellsym:
            self.m = np.append(self.m, self.m)
            self.n = np.append(self.n, self.n)

    def set_current_potential_from_regcoil(self, filename: str, ilambda: int):
        """
        Set phic and phis based on a regcoil netcdf file.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
            ilambda: 0-based index for the lambda array, indicating which current
                potential solution to use
        """
        f = netcdf_file(filename, 'r')
        nfp = f.variables['nfp'][()]
        mpol_potential = f.variables['mpol_potential'][()]
        ntor_potential = f.variables['ntor_potential'][()]
        xm_potential = f.variables['xm_potential'][()]
        xn_potential = f.variables['xn_potential'][()]
        symmetry_option = f.variables['symmetry_option'][()]
        single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][ilambda, :]
        f.close()

        # Check that correct shape of arrays are being used
        if mpol_potential != self.mpol:
            raise ValueError('Incorrect mpol_potential')
        if ntor_potential != self.ntor:
            raise ValueError('Incorrect ntor_potential')
        if nfp != self.nfp:
            raise ValueError('Incorrect nfp')
        if symmetry_option == 1:
            stellsym = True
        else:
            stellsym = False
        if stellsym != self.stellsym:
            raise ValueError('Incorrect stellsym')

        self.set_dofs(single_valued_current_potential_mn)

    @classmethod
    def from_netcdf(cls, filename: str, coil_ntheta_res=1.0, coil_nzeta_res=1.0):
        """
        Initialize a CurrentPotentialFourier object from a regcoil netcdf output file.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
        """
        f = netcdf_file(filename, 'r')
        nfp = f.variables['nfp'][()]
        mpol_potential = f.variables['mpol_potential'][()]
        ntor_potential = f.variables['ntor_potential'][()]
        net_poloidal_current_amperes = f.variables['net_poloidal_current_Amperes'][()]
        net_toroidal_current_amperes = f.variables['net_toroidal_current_Amperes'][()]
        xm_potential = f.variables['xm_potential'][()]
        xn_potential = f.variables['xn_potential'][()]
        symmetry_option = f.variables['symmetry_option'][()]
        if symmetry_option == 1:
            stellsym = True
        else:
            stellsym = False
        rmnc_coil = f.variables['rmnc_coil'][()]
        zmns_coil = f.variables['zmns_coil'][()]
        if ('rmns_coil' in f.variables and 'zmnc_coil' in f.variables):
            rmns_coil = f.variables['rmns_coil'][()]
            zmnc_coil = f.variables['zmnc_coil'][()]
            if np.all(zmnc_coil == 0) and np.all(rmns_coil == 0):
                stellsym_surf = True
            else:
                stellsym_surf = False
        else:
            rmns_coil = np.zeros_like(rmnc_coil)
            zmnc_coil = np.zeros_like(zmns_coil)
            stellsym_surf = True
        xm_coil = f.variables['xm_coil'][()]
        xn_coil = f.variables['xn_coil'][()]
        ntheta_coil = int(f.variables['ntheta_coil'][()] * coil_ntheta_res)
        nzeta_coil = int(f.variables['nzeta_coil'][()] * coil_nzeta_res)
        f.close()
        mpol_coil = int(np.max(xm_coil))
        ntor_coil = int(np.max(xn_coil)/nfp)

        # quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        #                 ntheta=ntheta_coil, nphi=nzeta_coil, nfp=nfp, range='full torus')

        s_coil = SurfaceRZFourier(nfp=nfp, mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym_surf)
        s_coil = s_coil.from_nphi_ntheta(nfp=nfp, ntheta=ntheta_coil, nphi=nzeta_coil * nfp,
                                         mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym_surf, range='full torus')

        # s_coil = SurfaceRZFourier(nfp=nfp, mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym_surf,
        #         quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)
        s_coil.set_dofs(0*s_coil.get_dofs())

        # s_coil.rc = 0*s_coil.rc
        # s_coil.zs = 0*s_coil.zs
        for im in range(len(xm_coil)):
            s_coil.set_rc(xm_coil[im], int(xn_coil[im]/nfp), rmnc_coil[im])
            s_coil.set_zs(xm_coil[im], int(xn_coil[im]/nfp), zmns_coil[im])
            m = int(xm_coil[im])
            n = int(xn_coil[im] / nfp)
            # if (m == 0 and n<0):
            #     s_coil.set_rc(m, -n, rmnc_coil[im])
            #     s_coil.set_zs(m, -n, -zmns_coil[im])
            # else:
            #     s_coil.set_rc(m, n, rmnc_coil[im])
            #     s_coil.set_zs(m, n, zmns_coil[im])
            if not stellsym_surf:
                s_coil.set_rs(xm_coil[im], int(xn_coil[im]/nfp), rmns_coil[im])
                s_coil.set_zc(xm_coil[im], int(xn_coil[im]/nfp), zmnc_coil[im])
                # if (m == 0 and n<0):
                #     s_coil.set_rs(m, -n, -rmns_coil[im])
                #     s_coil.set_zc(m, -n, zmnc_coil[im])
                # else:
                #     s_coil.set_rs(m, n, rmns_coil[im])
                #     s_coil.set_zc(m, n, zmnc_coil[im])

        s_coil.local_full_x = s_coil.get_dofs()
        # for im in range(len(xm_coil)):
        #     m = int(xm_coil[im])
        #     n = int(xn_coil[im] / nfp)
        #     if (m == 0 and n<0):
        #         s_coil.set_rc(m, -n, rmnc_coil[im])
        #         s_coil.set_zs(m, -n, -zmns_coil[im])
        #     else:
        #         s_coil.set_rc(m, n, rmnc_coil[im])
        #         s_coil.set_zs(m, n, zmns_coil[im])
        #     if not stellsym_surf:
        #         if (m == 0 and n<0):
        #             s_coil.set_rs(m, -n, -rmns_coil[im])
        #             s_coil.set_zc(m, -n, zmnc_coil[im])
        #         else:
        #             s_coil.set_rs(m, n, rmns_coil[im])
        #             s_coil.set_zc(m, n, zmnc_coil[im])

        # s_coil.local_full_x = s_coil.get_dofs()

        cp = cls(s_coil, mpol=mpol_potential, ntor=ntor_potential,
                 net_poloidal_current_amperes=net_poloidal_current_amperes,
                 net_toroidal_current_amperes=net_toroidal_current_amperes,
                 stellsym=stellsym)

        return cp
