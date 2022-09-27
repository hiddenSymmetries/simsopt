import logging

import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import interp1d
import f90nml
from monty.json import MSONable

import simsoptpp as sopp
from .surface import Surface
from .._core.optimizable import DOFs, Optimizable
from .._core.util import nested_lists_to_array

logger = logging.getLogger(__name__)

__all__ = ['SurfaceRZFourier', 'SurfaceRZPseudospectral']


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

    For more information about the arguments ``quadpoints_phi``, and
    ``quadpoints_theta``, see the general documentation on :ref:`surfaces`.
    Instead of supplying the quadrature point arrays along :math:`\phi` and
    :math:`\theta` directions, one could also specify the number of
    quadrature points for :math:`\phi` and :math:`\theta` using the
    class method :py:meth:`~simsopt.geo.surface.Surface.from_nphi_ntheta`.

    Args:
        nfp: The number of field periods.
        stellsym: Whether the surface is stellarator-symmetric, i.e.
          symmetry under rotation by :math:`\pi` about the x-axis.
        mpol: Maximum poloidal mode number included.
        ntor: Maximum toroidal mode number included, divided by ``nfp``.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0,
                 quadpoints_phi=None, quadpoints_theta=None):

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        sopp.SurfaceRZFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                       quadpoints_phi, quadpoints_theta)
        self.rc[0, ntor] = 1.0
        self.rc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        Surface.__init__(self, x0=self.get_dofs(),
                         external_dof_setter=SurfaceRZFourier.set_dofs_impl,
                         names=self._make_names())
        self._make_mn()

    def get_dofs(self):
        """
        Return the dofs associated to this surface.
        """
        return np.asarray(sopp.SurfaceRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        self.local_full_x = dofs

    def _make_names(self):
        """
        Form a list of names of the ``rc``, ``zs``, ``rs``, or ``zc``
        array elements.  The order of these four arrays here must
        match the order in ``set_dofs_impl()`` and ``get_dofs()`` in
        ``src/simsoptpp/surfacerzfourier.h``.
        """
        if self.stellsym:
            names = self._make_names_helper('rc', True) + self._make_names_helper('zs', False)
        else:
            names = self._make_names_helper('rc', True) \
                + self._make_names_helper('rs', False) \
                + self._make_names_helper('zc', True) \
                + self._make_names_helper('zs', False)
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

        f = netcdf_file(filename, mmap=False)
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

        ntheta = kwargs.pop("ntheta", None)
        nphi = kwargs.pop("nphi", None)
        grid_range = kwargs.pop("range", None)

        if ntheta is not None or nphi is not None:
            kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] = Surface.get_quadpoints(
                ntheta=ntheta, nphi=nphi, nfp=nfp, range=grid_range)

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

        surf.local_full_x = surf.get_dofs()
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

        ntheta = kwargs.pop("ntheta", None)
        nphi = kwargs.pop("nphi", None)
        grid_range = kwargs.pop("range", None)

        if ntheta is not None or nphi is not None:
            kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] = Surface.get_quadpoints(
                ntheta=ntheta, nphi=nphi, nfp=nfp, range=grid_range)

        surf = cls(mpol=mpol_boundary, ntor=ntor_boundary, nfp=nfp, stellsym=stellsym,
                   **kwargs)

        # Transfer boundary shape data from the namelist to the surface object:
        # In these loops, we set surf.rc/zs rather than call surf.set_rc() for speed.
        for jm in range(rc.shape[0]):
            m = jm + nml.start_index['rbc'][1]
            for jn in range(rc.shape[1]):
                n = jn + nml.start_index['rbc'][0]
                surf.rc[m, n + ntor_boundary] = rc[jm, jn]

        for jm in range(zs.shape[0]):
            m = jm + nml.start_index['zbs'][1]
            for jn in range(zs.shape[1]):
                n = jn + nml.start_index['zbs'][0]
                surf.zs[m, n + ntor_boundary] = zs[jm, jn]

        if lasym:
            for jm in range(rs.shape[0]):
                m = jm + nml.start_index['rbs'][1]
                for jn in range(rs.shape[1]):
                    n = jn + nml.start_index['rbs'][0]
                    surf.rs[m, n + ntor_boundary] = rs[jm, jn]

            for jm in range(zc.shape[0]):
                m = jm + nml.start_index['zbc'][1]
                for jn in range(zc.shape[1]):
                    n = jn + nml.start_index['zbc'][0]
                    surf.zc[m, n + ntor_boundary] = zc[jm, jn]

        # Sync the dofs:
        surf.local_full_x = surf.get_dofs()
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

        surf.local_full_x = surf.get_dofs()
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
        self._make_mn()

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

    def get_nml(self):
        """
        Generates a fortran namelist file containing the RBC/RBS/ZBC/ZBS
        coefficients, in the form used in VMEC and SPEC input
        files. The result will be returned as a string. For saving a
        file, see the ``write_nml()`` function.
        """
        nml = ''
        nml += '&INDATA\n'
        if self.stellsym:
            nml += 'LASYM = .FALSE.\n'
        else:
            nml += 'LASYM = .TRUE.\n'
        nml += f'NFP = {self.nfp}\n'

        for m in range(self.mpol + 1):
            nmin = -self.ntor
            if m == 0:
                nmin = 0
            for n in range(nmin, self.ntor + 1):
                rc = self.get_rc(m, n)
                zs = self.get_zs(m, n)
                if np.abs(rc) > 0 or np.abs(zs) > 0:
                    nml += f"RBC({n:4d},{m:4d}) ={rc:23.15e},    ZBS({n:4d},{m:4d}) ={zs:23.15e}\n"
                if (not self.stellsym):
                    rs = self.get_rs(m, n)
                    zc = self.get_zc(m, n)
                    if np.abs(rs) > 0 or np.abs(zc) > 0:
                        nml += f"RBS({n:4d},{m:4d}) ={rs:23.15e},    ZBC({n:4d},{m:4d}) ={zc:23.15e}\n"
        nml += '/\n'
        return nml

    def write_nml(self, filename: str):
        """
        Writes a fortran namelist file containing the RBC/RBS/ZBC/ZBS
        coefficients, in the form used in VMEC and SPEC input
        files. To just generate the namelist as a string without
        saving a file, see the ``get_nml()`` function.

        Args:
            filename: Name of the file to write.
        """
        with open(filename, 'w') as f:
            f.write(self.get_nml())

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["stellsym"] = self.stellsym
        d["mpol"] = self.mpol
        d["ntor"] = self.ntor
        return d

    @classmethod
    def from_dict(cls, d):
        surf = cls(nfp=d["nfp"], stellsym=d["stellsym"],
                   mpol=d["mpol"], ntor=d["ntor"],
                   quadpoints_phi=d["quadpoints_phi"],
                   quadpoints_theta=d["quadpoints_theta"])
        surf.local_full_x = d["x0"]
        return surf

    return_fn_map = {'area': sopp.SurfaceRZFourier.area,
                     'volume': sopp.SurfaceRZFourier.volume,
                     'aspect-ratio': Surface.aspect_ratio}


class SurfaceRZPseudospectral(Optimizable):
    """
    This class is used to replace the Fourier-space dofs of
    :obj:`SurfaceRZFourier` with real-space dofs, corresponding to the
    position of the surface on grid points.  The advantage of having
    the dofs in real-space is that they are all of the same magnitude,
    so it is easier to know what reasonable box constraints are. This
    class may therefore be useful for stage-1 optimization using
    algorithms that require box constraints.

    Presently, ``SurfaceRZPseudospectral`` assumes stellarator
    symmetry.

    In this class, the position vector on the surface is specified on
    a tensor product grid of ``ntheta * nphi`` points per half period,
    where ``ntheta`` and ``nphi`` are both odd, ``phi`` is the
    standard toroidal angle, and ``theta`` is any poloidal angle. The
    maximum Fourier mode numbers that can be represented by this grid
    are ``mpol`` in the poloidal angle and ``ntor * nfp`` in the
    toroidal angle, where ``ntheta = 1 + 2 * mpol`` and ``nphi = 1 + 2
    * ntor``. However, due to stellarator symmetry, roughly half of
    the grid points are redundant. Therefore the dofs only correspond
    to the non-redundant points, and the remaining points are computed
    from the dofs using symmetry.

    A ``SurfaceRZPseudospectral`` object with resolution parameters
    ``mpol`` and ``ntor`` has exactly the same number of dofs as a
    :obj:`SurfaceRZFourier` object with the same ``mpol`` and
    ``ntor``.  Specifically,

    .. code-block::

        ndofs = 1 + 2 * (mpol + ntor + 2 * mpol * ntor)

    This class also allows the coordinates ``r`` and ``z`` to be
    shifted and scaled, which may help to keep the dofs all of order
    1. Letting ``r_dofs`` and ``z_dofs`` denote the dofs in this
    class, the actual ``r`` and ``z`` coordinates are determined via

    .. code-block::

        r = r_dofs * a_scale + r_shift
        z = z_dofs * a_scale

    where ``r_shift`` and ``a_scale`` are optional arguments to the
    constructor, which would be set to roughly the major and minor
    radius.

    Typical usage::

        vmec = Vmec("input.your_filename_here")
        vmec.boundary = SurfaceRZPseudospectral.from_RZFourier(vmec.boundary)

    The dofs in this class are named ``r(jphi,jtheta)`` and
    ``z(jphi,jtheta)``, where ``jphi`` and ``jtheta`` are integer
    indices into the ``phi`` and ``theta`` grids.

    This class does not presently implement the
    :obj:`simsopt.geo.surface.Surface` interface, e.g. there is not a
    ``gamma()`` function.

    Args:
        mpol: Maximum poloidal Fourier mode number represented.
        ntor: The maximum toroidal Fourier mode number represented, divided by ``nfp``.
        nfp: Number of field periods.
        r_shift: Constant added to the ``r(jphi,jtheta)`` dofs to get the actual major radius.
        a_scale: Dofs are multiplied by this factor to get the actual cylindrical coordinates.
    """

    def __init__(self, mpol, ntor, nfp, r_shift=1.0, a_scale=1.0):
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.r_shift = r_shift
        self.a_scale = a_scale
        ndofs = 1 + 2 * (ntor + mpol * (2 * ntor + 1))
        super().__init__(x0=np.zeros(ndofs), names=self._make_names())

    def _make_names(self):
        """
        Create the list of names for the dofs.
        """
        names = ['r(0,0)']
        for dimension in ['r', 'z']:
            for jtheta in range(1, self.mpol + 1):
                names.append(dimension + f'(0,{jtheta})')
            for jphi in range(1, self.ntor + 1):
                for jtheta in range(2 * self.mpol + 1):
                    names.append(dimension + f'({jphi},{jtheta})')
        return names

    @classmethod
    def from_RZFourier(cls, surff, **kwargs):
        """
        Convert a :obj:`SurfaceRZFourier` object to a
        ``SurfaceRZPseudospectral`` object.

        Args:
            surff: The :obj:`SurfaceRZFourier` object to convert.
            kwargs: You can optionally provide the ``r_shift`` or ``a_scale`` arguments
              to the ``SurfaceRZPseudospectral`` constructor here.
        """
        if not surff.stellsym:
            raise RuntimeError('SurfaceRZPseudospectral presently only '
                               'supports stellarator-symmetric surfaces')

        # shorthand:
        mpol = surff.mpol
        ntor = surff.ntor
        ntheta = 2 * mpol + 1
        nphi = 2 * ntor + 1

        # Make a copy of surff with the desired theta and phi points.
        surf_copy = SurfaceRZFourier.from_nphi_ntheta(
            mpol=mpol, ntor=ntor, nfp=surff.nfp,
            range='field period', ntheta=ntheta, nphi=nphi)
        surf_copy.x = surff.local_full_x

        surf_new = cls(mpol=mpol, ntor=ntor, nfp=surff.nfp, **kwargs)
        gamma = surf_copy.gamma()
        r0 = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
        r = (r0 - surf_new.r_shift) / surf_new.a_scale
        z = gamma[:, :, 2] / surf_new.a_scale

        dofs = np.zeros_like(surf_new.full_x)
        ndofs = len(dofs)
        index = 0
        for jtheta in range(mpol + 1):
            dofs[index] = r[0, jtheta]
            index += 1
        for jphi in range(1, ntor + 1):
            for jtheta in range(ntheta):
                dofs[index] = r[jphi, jtheta]
                index += 1
        for jtheta in range(1, mpol + 1):
            dofs[index] = z[0, jtheta]
            index += 1
        for jphi in range(1, ntor + 1):
            for jtheta in range(ntheta):
                dofs[index] = z[jphi, jtheta]
                index += 1
        assert index == ndofs
        surf_new.x = dofs
        return surf_new

    def _complete_grid(self):
        """
        Using stellarator symmetry, copy the real-space dofs to cover a
        full 2d ``(theta, phi)`` grid.
        """

        # shorthand:
        mpol = self.mpol
        ntor = self.ntor
        ntheta = 2 * mpol + 1
        nphi = 2 * ntor + 1

        r = np.zeros((ntheta, nphi))
        z = np.zeros((ntheta, nphi))
        r[0, 0] = self.x[0]
        shift = mpol + ntor * (2 * mpol + 1)  # = mpol + ntor + 2 * mpol * ntor
        assert 2 * shift + 1 == len(self.x)
        for jtheta in range(1, mpol + 1):
            r[jtheta, 0] = self.x[jtheta]
            r[ntheta - jtheta, 0] = self.x[jtheta]
            assert self.local_dof_names[jtheta + shift] == f'z(0,{jtheta})'
            z[jtheta, 0] = self.x[jtheta + shift]
            z[ntheta - jtheta, 0] = -self.x[jtheta + shift]
        for jphi in range(1, ntor + 1):
            for jtheta in range(ntheta):
                index = (jphi - 1) * ntheta + jtheta + mpol + 1
                assert self.local_dof_names[index] == f'r({jphi},{jtheta})'
                assert self.local_dof_names[index + shift] == f'z({jphi},{jtheta})'
                r[jtheta, jphi] = self.x[index]
                z[jtheta, jphi] = self.x[index + shift]
                if jtheta == 0:
                    r[0, nphi - jphi] = self.x[index]
                    z[0, nphi - jphi] = -self.x[index + shift]
                else:
                    r[ntheta - jtheta, nphi - jphi] = self.x[index]
                    z[ntheta - jtheta, nphi - jphi] = -self.x[index + shift]

        r2 = self.r_shift + self.a_scale * r
        z2 = self.a_scale * z
        return r2, z2

    def to_RZFourier(self, **kwargs):
        """
        Convert to a :obj:`SurfaceRZFourier` describing the same shape.

        Args:
            kwargs: You can optionally provide the ``range``, ``nphi``,
              and/or ``ntheta`` arguments to the :obj:`SurfaceRZFourier` constructor here.
        """
        # shorthand:
        mpol = self.mpol
        ntor = self.ntor

        r, z = self._complete_grid()
        # What follows is a Fourier transform. We could use an FFT,
        # but since speed is not a concern here for now, the Fourier
        # transform is just done "by hand" so there is no uncertainty
        # about normalizations etc.

        ntheta = kwargs.pop("ntheta", None)
        nphi = kwargs.pop("nphi", None)
        grid_range = kwargs.pop("range", None)

        if ntheta is not None or nphi is not None:
            kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] = Surface.get_quadpoints(
                ntheta=ntheta, nphi=nphi, nfp=self.nfp, range=grid_range)

        surf = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=self.nfp, **kwargs)
        surf.set_rc(0, 0, np.mean(r))
        ntheta = 2 * mpol + 1
        nphi = 2 * ntor + 1
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        phi, theta = np.meshgrid(phi1d, theta1d)
        for n in range(1, ntor + 1):
            surf.set_rc(0, n, 2 * np.mean(r * np.cos(-n * phi)))
            surf.set_zs(0, n, 2 * np.mean(z * np.sin(-n * phi)))
        for m in range(1, mpol + 1):
            for n in range(-ntor, ntor + 1):
                surf.set_rc(m, n, 2 * np.mean(r * np.cos(m * theta - n * phi)))
                surf.set_zs(m, n, 2 * np.mean(z * np.sin(m * theta - n * phi)))

        return surf

    def change_resolution(self, mpol, ntor):
        """
        Increase or decrease the number of degrees of freedom.  The new
        real-space dofs are obtained using Fourier interpolation. This
        function is useful for increasing the size of the parameter
        space during stage-1 optimization. If ``mpol`` and ``ntor``
        are increased or unchanged, there is no loss of information.
        If ``mpol`` or ``ntor`` are decreased, information is lost.

        Args:
            mpol: The new maximum poloidal mode number.
            ntor: The new maximum toroidal mode number, divided by ``nfp``.
        """
        # Map to Fourier space:
        surf2 = self.to_RZFourier()
        # Change the resolution in Fourier space, by truncating the modes or padding 0s:
        surf2.change_resolution(mpol=mpol, ntor=ntor)
        # Map from Fourier space back to real space:
        surf3 = SurfaceRZPseudospectral.from_RZFourier(surf2,
                                                       r_shift=self.r_shift,
                                                       a_scale=self.a_scale)
        return surf3

    def as_dict(self) -> dict:
        d = MSONable.as_dict(self)
        d["x0"] = list(self.local_full_x)

    @classmethod
    def from_dict(cls, d):
        surf = cls(d["mpol"], d["ntor"], d["nfp"], d["r_shift"], d["a_scale"])
        surf.local_full_x = d["x0"]
        return surf
