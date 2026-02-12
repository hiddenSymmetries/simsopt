from __future__ import annotations

from typing import Optional, Union, List
import numpy as np
from .._core.optimizable import DOFs, Optimizable
from .._core.json import GSONDecoder
import simsoptpp as sopp
from simsopt.geo import SurfaceRZFourier
from scipy.io import netcdf_file

__all__ = ['CurrentPotentialFourier', 'CurrentPotential']


class CurrentPotential(Optimizable):
    """
    Current Potential base object, not necessarily assuming
    that the current potential will be represented by a
    Fourier expansion in the toroidal and poloidal modes.

    Args:
        winding_surface: SurfaceRZFourier object representing the coil surface.
        kwargs: Additional keyword arguments passed to the Optimizable base class.
    """

    def __init__(self, winding_surface: SurfaceRZFourier, **kwargs) -> None:
        super().__init__(**kwargs)
        self.winding_surface = winding_surface

    def K(self) -> np.ndarray:
        """
        Get the K vector of the CurrentPotential object.

        Returns:
            np.ndarray: The K vector of the CurrentPotential object.
        """
        data = np.zeros((len(self.quadpoints_phi), len(self.quadpoints_theta), 3))
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.K_impl_helper(data, dg1, dg2, normal)
        return data

    def K_matrix(self) -> np.ndarray:
        """
        Get the K matrix of the CurrentPotential object.

        Returns:
            np.ndarray: The K matrix of the CurrentPotential object.
        """
        data = np.zeros((self.num_dofs(), self.num_dofs()))
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.K_matrix_impl_helper(data, dg1, dg2, normal)
        return data

    def num_dofs(self) -> int:
        """
        Get the number of dofs of the CurrentPotential object.

        Returns:
            int: The number of dofs of the CurrentPotential object.
        """
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

    def __init__(
        self,
        winding_surface: SurfaceRZFourier,
        net_poloidal_current_amperes: float = 1,
        net_toroidal_current_amperes: float = 0,
        nfp: Optional[int] = None,
        stellsym: Optional[bool] = None,
        mpol: Optional[int] = None,
        ntor: Optional[int] = None,
        quadpoints_phi: Optional[Union[np.ndarray, List[float]]] = None,
        quadpoints_theta: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:

        if nfp is None:
            nfp = winding_surface.nfp
        if stellsym is None:
            stellsym = winding_surface.stellsym
        if mpol is None:
            mpol = winding_surface.mpol
        if ntor is None:
            ntor = winding_surface.ntor

        if nfp > 1 and np.max(winding_surface.quadpoints_phi) <= 1/nfp:
            raise AttributeError('winding_surface must contain all field periods.')

        # quadpoints_phi or quadpoints_theta has to be the same as those 
        # in the winding surface. Otherwise, it can cause issues during 
        # CurrentPotential.K(), CurrentPotential.K_matrix and 
        # in WindingSurfaceField.__init__().
        if quadpoints_theta is None:
            quadpoints_theta = winding_surface.quadpoints_theta
        if quadpoints_phi is None:
            quadpoints_phi = winding_surface.quadpoints_phi

        sopp.CurrentPotentialFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                              quadpoints_phi, quadpoints_theta,
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

    def _make_names(self) -> List[str]:
        """
        Create the dof names for the CurrentPotentialFourier object.

        Returns:
            List[str]: The names of the coefficients.
        """
        if self.stellsym:
            names = self._make_names_helper('Phis')
        else:
            names = self._make_names_helper('Phis') \
                + self._make_names_helper('Phic')
        return names

    def _make_names_helper(self, prefix: str) -> List[str]:
        """
        Helper function for _make_names() method to format the strings.

        Args:
            prefix: The prefix for the name of the coefficients.

        Returns:
            List[str]: The names of the coefficients.
        """
        names = []

        start = 1
        names += [prefix + '(0,' + str(n) + ')' for n in range(start, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names

    def get_dofs(self) -> np.ndarray:
        """
        Get the dofs of the CurrentPotentialFourier object.

        Returns:
            np.ndarray: The dofs of the CurrentPotentialFourier object.
        """
        return np.asarray(sopp.CurrentPotentialFourier.get_dofs(self))

    def change_resolution(self, mpol: int, ntor: int) -> None:
        """
        Modeled after SurfaceRZFourier
        Change the values of `mpol` and `ntor`. Any new Fourier amplitudes
        will have a magnitude of zero.  Any previous nonzero Fourier
        amplitudes that are not within the new range will be
        discarded.

        Args:
            mpol: New poloidal mode number.
            ntor: New toroidal mode number.
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

    def get_phic(self, m: int, n: int) -> float:
        """
        Return a particular `phic` parameter.

        Args:
            m: Poloidal mode number.
            n: Toroidal mode number.

        Raises:
            ValueError: If `stellsym` is True (phic does not exist for this stellarator-symmetric current potential).
            IndexError: If `m` is less than 0, `m` is greater than `mpol`, `n` is greater than `ntor`, or `n` is less than -`ntor`.
        """
        if self.stellsym:
            raise ValueError(
                'phic does not exist for this stellarator-symmetric current potential.')
        self._validate_mn(m, n)
        return self.phic[m, n + self.ntor]

    def get_phis(self, m: int, n: int) -> float:
        """
        Return a particular `phis` parameter.

        Args:
            m: Poloidal mode number.
            n: Toroidal mode number.

        Raises:
            IndexError: If `m` is less than 0, `m` is greater than `mpol`, `n` is greater than `ntor`, or `n` is less than -`ntor`.
        """
        self._validate_mn(m, n)
        return self.phis[m, n + self.ntor]

    def set_phic(self, m: int, n: int, val: float) -> None:
        """
        Set a particular `phic` Parameter.

        Args:
            m: Poloidal mode number.
            n: Toroidal mode number.
            val: Value to set the `phic` parameter to.

        Raises:
            ValueError: If `stellsym` is True (phic does not exist for this stellarator-symmetric current potential).
            IndexError: If `m` is less than 0, `m` is greater than `mpol`, `n` is greater than `ntor`, or `n` is less than -`ntor`.
        """
        if self.stellsym:
            raise ValueError(
                'phic does not exist for this stellarator-symmetric current potential.')
        self._validate_mn(m, n)
        self.phic[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()
        self.invalidate_cache()

    def set_phis(self, m: int, n: int, val: float) -> None:
        """
        Set a particular `phis` Parameter.

        Args:
            m: Poloidal mode number.
            n: Toroidal mode number.
            val: Value to set the `phis` parameter to.

        Raises:
            IndexError: If `m` is less than 0, `m` is greater than `mpol`, `n` is greater than `ntor`, or `n` is less than -`ntor`.
        """
        self._validate_mn(m, n)
        self.phis[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()
        self.invalidate_cache()

    def set_net_toroidal_current_amperes(self, val: float) -> None:
        """
        Set the net toroidal current in Amperes.

        Args:
            val: Value to set the net toroidal current to.
        """
        self.net_toroidal_current_amperes = val
        self.invalidate_cache()

    def set_net_poloidal_current_amperes(self, val: float) -> None: 
        """
        Set the net poloidal current in Amperes.

        Args:
            val: Value to set the net poloidal current to.
        """
        self.net_poloidal_current_amperes = val
        self.invalidate_cache()

    def fixed_range(
        self, mmin: int, mmax: int, nmin: int, nmax: int, fixed: bool = True
    ) -> None:
        """
        Modeled after SurfaceRZFourier
        Set the 'fixed' property for a range of `m` and `n` values.

        All modes with `m` in the interval [`mmin`, `mmax`] and `n` in the
        interval [`nmin`, `nmax`] will have their fixed property set to
        the value of the `fixed` parameter. Note that `mmax` and `nmax`
        are included (unlike the upper bound in python's range(min,
        max).)

        Args:
            mmin: Minimum poloidal mode number.
            mmax: Maximum poloidal mode number.
            nmin: Minimum toroidal mode number.
            nmax: Maximum toroidal mode number.
            fixed: Whether to fix the modes.

        Raises:
            IndexError: If `mmin` is less than 0, `mmax` is greater than `mpol`, `nmin` is less than -`ntor`, or `nmax` is greater than `ntor`.
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
                    fn(f'Phis({m},{n})')
                    if not self.stellsym:
                        fn(f'Phic({m},{n})')

    def _validate_mn(self, m: int, n: int) -> None:
        """
        Copied from SurfaceRZFourier
        Check whether `m` and `n` are in the allowed range.

        Args:
            m: Poloidal mode number.
            n: Toroidal mode number.

        Raises:
            IndexError: If `m` is less than 0, `m` is greater than `mpol`, `n` is greater than `ntor`, or `n` is less than -`ntor`.
        """
        if m < 0:
            raise IndexError('m must be >= 0')
        if m > self.mpol:
            raise IndexError('m must be <= mpol')
        if n > self.ntor:
            raise IndexError('n must be <= ntor')
        if n < -self.ntor:
            raise IndexError('n must be >= -ntor')

    def _make_mn(self) -> None:
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
        f = netcdf_file(filename, 'r', mmap=False)
        nfp = f.variables['nfp'][()]
        mpol_potential = f.variables['mpol_potential'][()]
        ntor_potential = f.variables['ntor_potential'][()]
        _xm_potential = f.variables['xm_potential'][()]
        _xn_potential = f.variables['xn_potential'][()]
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

    def as_dict(self, serial_objs_dict=None) -> dict:
        """Sync Python _dofs with C++ state before serialization (set_dofs is C++-only)."""
        if len(self.local_full_x):
            self.local_full_x = self.get_dofs()
        return super().as_dict(serial_objs_dict)

    @classmethod
    def from_netcdf(
        cls,
        filename: str,
        coil_ntheta_res: float = 1.0,
        coil_nzeta_res: float = 1.0,
    ) -> "CurrentPotentialFourier":
        """
        Initialize a CurrentPotentialFourier object from a regcoil netcdf output file.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
            coil_ntheta_res: The resolution of the coil surface in the theta direction.
            coil_nzeta_res: The resolution of the coil surface in the zeta direction.

        Returns:
            cls: The CurrentPotentialFourier object.
        """
        f = netcdf_file(filename, 'r', mmap=False)
        nfp = f.variables['nfp'][()]
        mpol_potential = f.variables['mpol_potential'][()]
        ntor_potential = f.variables['ntor_potential'][()]
        net_poloidal_current_amperes = f.variables['net_poloidal_current_Amperes'][()]
        net_toroidal_current_amperes = f.variables['net_toroidal_current_Amperes'][()]
        _xm_potential = f.variables['xm_potential'][()]
        _xn_potential = f.variables['xn_potential'][()]
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
        s_coil = SurfaceRZFourier(nfp=nfp, mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym_surf)
        s_coil = s_coil.from_nphi_ntheta(nfp=nfp, ntheta=ntheta_coil, nphi=nzeta_coil * nfp,
                                         mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym_surf, range='full torus')

        s_coil.set_dofs(0*s_coil.get_dofs())

        for im in range(len(xm_coil)):
            s_coil.set_rc(xm_coil[im], int(xn_coil[im]/nfp), rmnc_coil[im])
            s_coil.set_zs(xm_coil[im], int(xn_coil[im]/nfp), zmns_coil[im])
            _m = int(xm_coil[im])
            _n = int(xn_coil[im] / nfp)

            if not stellsym_surf:
                s_coil.set_rs(xm_coil[im], int(xn_coil[im]/nfp), rmns_coil[im])
                s_coil.set_zc(xm_coil[im], int(xn_coil[im]/nfp), zmnc_coil[im])


        s_coil.local_full_x = s_coil.get_dofs()

        cp = cls(s_coil, mpol=mpol_potential, ntor=ntor_potential,
                 net_poloidal_current_amperes=net_poloidal_current_amperes,
                 net_toroidal_current_amperes=net_toroidal_current_amperes,
                 stellsym=stellsym)

        return cp

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        """
        Reconstruct CurrentPotentialFourier from serialized dict.
        Excludes dofs from __init__ and sets them after construction.
        """
        decoder = GSONDecoder()
        init_kwargs = dict(d)
        dofs_raw = init_kwargs.pop("dofs", None)
        decoded = {k: decoder.process_decoded(v, serial_objs_dict, recon_objs)
                  for k, v in init_kwargs.items()}
        obj = cls(**decoded)
        if dofs_raw is not None:
            dofs = decoder.process_decoded(dofs_raw, serial_objs_dict, recon_objs)
            obj.set_dofs(np.asarray(dofs.full_x))
        return obj
