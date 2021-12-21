import logging

import numpy as np
from scipy.io import netcdf
from scipy.interpolate import interp1d
import f90nml

import simsoptpp as sopp
from .surface import Surface
from .._core.graph_optimizable import DOFs, Optimizable
from .._core.util import nested_lists_to_array

logger = logging.getLogger(__name__)


class SurfaceRZFourier(sopp.SurfaceRZFourier, Surface):
    r"""
    ``SurfaceRZFourier`` is a surface that is represented in
    cylindrical coordinates using the following Fourier series:

    .. math::
           r(\theta, \phi) = \sum_{m=0}^{m_{\text{pol}}}
               \sum_{n=-n_{\text{tor}}}^{n_\text{tor}} [
               r_{c,m,n} \cos(m \theta - n_{\text{fp}} n \phi)
               + r_{s,m,n} \sin(m \theta - n_{\text{fp}} n \phi) ]

    and the same for :math:`z(\theta, \phi)`.

    Here, :math:`(r,\phi, z)` are standard cylindrical coordinates, and theta
    is any poloidal angle.

    Note that for :math:`m=0` we skip the :math:`n<0` term for the cos terms,
    and the :math:`n \leq 0` for the sin terms.

    In addition, in the ``stellsym=True`` case, we skip the sin terms for
    :math:`r`, and the cos terms for :math:`z`.

    For more information about the arguments ``nphi``, ``ntheta``,
    ``range``, ``quadpoints_phi``, and ``quadpoints_theta``, see the
    general documentation on :ref:`surfaces`.

    Args:
        nfp: The number of field periods.
        stellsym: Whether the surface is stellarator-symmetric, i.e.
          symmetry under rotation by :math:`\pi` about the x-axis.
        mpol: Maximum poloidal mode number included.
        ntor: Maximum toroidal mode number included, divided by ``nfp``.
        nphi: Number of grid points :math:`\phi_j` in the toroidal angle :math:`\phi`.
        ntheta: Number of grid points :math:`\theta_j` in the toroidal angle :math:`\theta`.
        range: Toroidal extent of the :math:`\phi` grid.
          Set to ``"full torus"`` (or equivalently ``SurfaceRZFourier.RANGE_FULL_TORUS``)
          to generate points up to 1 (with no point at 1).
          Set to ``"field period"`` (or equivalently ``SurfaceRZFourier.RANGE_FIELD_PERIOD``)
          to generate points up to :math:`1/n_{fp}` (with no point at :math:`1/n_{fp}`).
          Set to ``"half period"`` (or equivalently ``SurfaceRZFourier.RANGE_HALF_PERIOD``)
          to generate points up to :math:`1/(2 n_{fp})`, with all grid points shifted by half
          of the grid spacing in order to provide spectral convergence of integrals.
          If ``quadpoints_phi`` is specified, ``range`` is irrelevant.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0,
                 nphi=None, ntheta=None, range="full torus",
                 quadpoints_phi=None, quadpoints_theta=None):

        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(nfp=nfp,
                                                                  nphi=nphi, ntheta=ntheta, range=range,
                                                                  quadpoints_phi=quadpoints_phi,
                                                                  quadpoints_theta=quadpoints_theta)
        sopp.SurfaceRZFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                       quadpoints_phi, quadpoints_theta)
        self.rc[0, ntor] = 1.0
        self.rc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        Surface.__init__(self, x0=self.get_dofs(),
                         external_dof_setter=SurfaceRZFourier.set_dofs_impl,
                         names=self._make_names())

    def get_dofs(self):
        """
        Return the dofs associated to this surface.
        """
        return np.asarray(sopp.SurfaceRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        self.local_full_x = dofs

    def _make_names(self):
        """
        Form a list of names of the `rc`, `zs`, `rs`, or `zc` array elements.
        """
        names = self._make_names_helper('rc', True) + self._make_names_helper('zs', False)
        if not self.stellsym:
            names += self._make_names_helper('rs', False) + self._make_names_helper('zc', True)
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

    @classmethod
    def from_wout(cls,
                  filename: str,
                  s: float = 1.0,
                  interp_kind: str = 'linear',
                  **kwargs):
        """
        Read in a surface from a VMEC wout output file. Note that this
        function does not require the VMEC python module.

        Args:
            filename: Name of the ``wout_*.nc`` file to read.
            s: Value of normalized toroidal flux to use for the surface.
              The default value of 1.0 corresponds to the VMEC plasma boundary.
              Must lie in the interval [0, 1].
            interp_kind: Interpolation method in s. The available options correspond to
              the ``kind`` argument of
              `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy-interpolate-interp1d>`_.
            kwargs: Any other arguments to pass to the ``SurfaceRZFourier`` constructor.
              You can specify ``quadpoints_theta`` and ``quadpoints_phi`` here.
        """

        if s < 0 or s > 1:
            raise ValueError('s must lie in the interval [0, 1]')

        f = netcdf.netcdf_file(filename, mmap=False)
        nfp = f.variables['nfp'][()]
        ns = f.variables['ns'][()]
        xm = f.variables['xm'][()]
        xn = f.variables['xn'][()]
        rmnc = f.variables['rmnc'][()]
        zmns = f.variables['zmns'][()]
        lasym = bool(f.variables['lasym__logical__'][()])
        stellsym = not lasym
        if lasym:
            rmns = f.variables['rmns'][()]
            zmnc = f.variables['zmnc'][()]
        f.close()

        # Interpolate in s:
        s_full_grid = np.linspace(0.0, 1.0, ns)

        interp = interp1d(s_full_grid, rmnc, kind=interp_kind, axis=0)
        rbc = interp(s)

        interp = interp1d(s_full_grid, zmns, kind=interp_kind, axis=0)
        zbs = interp(s)

        if lasym:
            interp = interp1d(s_full_grid, rmns, kind=interp_kind, axis=0)
            rbs = interp(s)

            interp = interp1d(s_full_grid, zmnc, kind=interp_kind, axis=0)
            zbc = interp(s)

        mpol = int(np.max(xm))
        ntor = int(np.max(np.abs(xn)) / nfp)

        surf = cls(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                   **kwargs)

        for j in range(len(xm)):
            m = int(xm[j])
            n = int(xn[j] / nfp)
            surf.rc[m, n + ntor] = rbc[j]
            surf.zs[m, n + ntor] = zbs[j]
            if not stellsym:
                surf.rs[m, n + ntor] = rbs[j]
                surf.zc[m, n + ntor] = zbc[j]

        return surf

    @classmethod
    def from_vmec_input(cls,
                        filename: str,
                        **kwargs):
        """
        Read in a surface from a VMEC input file. The ``INDATA`` namelist
        of this file will be read using `f90nml
        <https://f90nml.readthedocs.io/en/latest/index.html>`_. Note
        that this function does not require the VMEC python module.

        Args:
            filename: Name of the ``input.*`` file to read.
            kwargs: Any other arguments to pass to the ``SurfaceRZFourier`` constructor.
              You can specify ``quadpoints_theta`` and ``quadpoints_phi`` here.
        """

        all_namelists = f90nml.read(filename)
        # We only care about the 'indata' namelist
        nml = all_namelists['indata']
        if 'nfp' in nml:
            nfp = nml['nfp']
        else:
            nfp = 1

        if 'lasym' in nml:
            lasym = nml['lasym']
        else:
            lasym = False
        stellsym = not lasym

        # We can assume rbc and zbs are specified in the namelist.
        # f90nml returns rbc and zbs as a list of lists where the
        # inner lists do not necessarily all have the same
        # dimension. Hence we need to be careful when converting to
        # numpy arrays.
        rc = nested_lists_to_array(nml['rbc'])
        zs = nested_lists_to_array(nml['zbs'])
        if lasym:
            rs = nested_lists_to_array(nml['rbs'])
            zc = nested_lists_to_array(nml['zbc'])

        rbc_first_n = nml.start_index['rbc'][0]
        rbc_last_n = rbc_first_n + rc.shape[1] - 1
        zbs_first_n = nml.start_index['zbs'][0]
        zbs_last_n = zbs_first_n + zs.shape[1] - 1
        if lasym:
            rbs_first_n = nml.start_index['rbs'][0]
            rbs_last_n = rbs_first_n + rs.shape[1] - 1
            zbc_first_n = nml.start_index['zbc'][0]
            zbc_last_n = zbc_first_n + zc.shape[1] - 1
        else:
            rbs_first_n = 0
            rbs_last_n = 0
            zbc_first_n = 0
            zbc_last_n = 0
        ntor_boundary = np.max(np.abs(np.array([rbc_first_n, rbc_last_n,
                                                zbs_first_n, zbs_last_n,
                                                rbs_first_n, rbs_last_n,
                                                zbc_first_n, zbc_last_n], dtype='i')))

        rbc_first_m = nml.start_index['rbc'][1]
        rbc_last_m = rbc_first_m + rc.shape[0] - 1
        zbs_first_m = nml.start_index['zbs'][1]
        zbs_last_m = zbs_first_m + zs.shape[0] - 1
        if lasym:
            rbs_first_m = nml.start_index['rbs'][1]
            rbs_last_m = rbs_first_m + rs.shape[0] - 1
            zbc_first_m = nml.start_index['zbc'][1]
            zbc_last_m = zbc_first_m + zc.shape[0] - 1
        else:
            rbs_first_m = 0
            rbs_last_m = 0
            zbc_first_m = 0
            zbc_last_m = 0
        mpol_boundary = np.max((rbc_last_m, zbs_last_m, rbs_last_m, zbc_last_m))
        logger.debug('Input file has ntor_boundary={} mpol_boundary={}' \
                     .format(ntor_boundary, mpol_boundary))

        surf = cls(mpol=mpol_boundary, ntor=ntor_boundary, nfp=nfp, stellsym=stellsym,
                   **kwargs)

        # Transfer boundary shape data from the namelist to the surface object:
        for jm in range(rc.shape[0]):
            m = jm + nml.start_index['rbc'][1]
            for jn in range(rc.shape[1]):
                n = jn + nml.start_index['rbc'][0]
                surf.set_rc(m, n, rc[jm, jn])

        for jm in range(zs.shape[0]):
            m = jm + nml.start_index['zbs'][1]
            for jn in range(zs.shape[1]):
                n = jn + nml.start_index['zbs'][0]
                surf.set_zs(m, n, zs[jm, jn])

        if lasym:
            for jm in range(rs.shape[0]):
                m = jm + nml.start_index['rbs'][1]
                for jn in range(rs.shape[1]):
                    n = jn + nml.start_index['rbs'][0]
                    surf.set_rs(m, n, rs[jm, jn])

            for jm in range(zc.shape[0]):
                m = jm + nml.start_index['zbc'][1]
                for jn in range(zc.shape[1]):
                    n = jn + nml.start_index['zbc'][0]
                    surf.set_zc(m, n, zc[jm, jn])

        return surf

    @classmethod
    def from_focus(cls, filename, **kwargs):
        """
        Read in a surface from a FOCUS-format file.
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

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

        surf = cls(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, **kwargs)

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

        # Update the dofs object
        self._dofs = DOFs(self.get_dofs(), self._make_names())
        # The following methods of graph Optimizable framework need to be called
        Optimizable._update_free_dof_size_indices(self)
        Optimizable._update_full_dof_size_indices(self)
        Optimizable._set_new_x(self)

    def to_RZFourier(self):
        """
        No conversion necessary.
        """
        return self

    def __repr__(self):
        return self.name + f" (nfp={self.nfp}, stellsym={self.stellsym}, " + \
            f"mpol={self.mpol}, ntor={self.ntor})"

    def _validate_mn(self, m, n):
        """
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
        self.local_full_x = self.get_dofs()

    def set_rs(self, m, n, val):
        """
        Set a particular `rs` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.rs[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def set_zc(self, m, n, val):
        """
        Set a particular `zc` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.zc[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def set_zs(self, m, n, val):
        """
        Set a particular `zs` Parameter.
        """
        self._validate_mn(m, n)
        self.zs[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
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
                fn(f'rc({m},{n})')
                if m > 0 or n != 0:
                    fn(f'zs({m},{n})')
                if not self.stellsym:
                    fn(f'zc({m},{n})')
                    if m > 0 or n != 0:
                        fn(f'rs({m},{n})')

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

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

    def write_nml(self, filename: str = 'boundary'):
        """
        Writes a fortran namelist file containing the RBC/RBS/ZBS/ZBS
        coefficients, in the form used in VMEC and SPEC input files.

        Args:
            filename: Name of the file to write.
        """
        with open(filename, 'w') as f:
            f.write('&INDATA\n')
            if self.stellsym:
                f.write('LASYM = .FALSE.\n')
            else:
                f.write('LASYM = .TRUE.\n')
            f.write('NFP = ' + str(self.nfp) + '\n')

            for m in range(self.mpol + 1):
                nmin = -self.ntor
                if m == 0:
                    nmin = 0
                for n in range(nmin, self.ntor + 1):
                    rc = self.get_rc(m, n)
                    zs = self.get_zs(m, n)
                    if np.abs(rc) > 0 or np.abs(zs) > 0:
                        f.write("RBC({:4d},{:4d}) ={:23.15e},    ZBS({:4d},{:4d}) ={:23.15e}\n" \
                                .format(n, m, rc, n, m, zs))
                    if (not self.stellsym):
                        rs = self.get_rs(m, n)
                        zc = self.get_zc(m, n)
                        if np.abs(rs) > 0 or np.abs(zc) > 0:
                            f.write("RBS({:4d},{:4d}) ={:23.15e},    ZBC({:4d},{:4d}) ={:23.15e}\n" \
                                    .format(n, m, rs, n, m, zc))
            f.write('/\n')

    return_fn_map = {'area': sopp.SurfaceRZFourier.area,
                     'volume': sopp.SurfaceRZFourier.volume,
                     'aspect-ratio': Surface.aspect_ratio}
