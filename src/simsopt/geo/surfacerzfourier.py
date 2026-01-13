import logging
import time

import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar, least_squares, NonlinearConstraint
import f90nml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mpl_colors

import simsoptpp as sopp
from .surface import Surface
from ..objectives.jaccard import jaccard_index
from .._core.optimizable import Optimizable
from .._core.util import nested_lists_to_array
from .._core.dev import SimsoptRequires

try:
    from qsc import Qsc
    from qsc.util import to_Fourier
except ImportError:
    Qsc = None

logger = logging.getLogger(__name__)

__all__ = ['SurfaceRZFourier', 'SurfaceRZPseudospectral', 'plot_spectral_condensation']


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
                 quadpoints_phi=None, quadpoints_theta=None,
                 dofs=None):

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        sopp.SurfaceRZFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                       quadpoints_phi, quadpoints_theta)
        self.rc[0, ntor] = 1.0
        self.rc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        if dofs is None:
            Surface.__init__(self, x0=self.get_dofs(),
                             external_dof_setter=SurfaceRZFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            Surface.__init__(self, dofs=dofs,
                             external_dof_setter=SurfaceRZFourier.set_dofs_impl)
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
    def from_wout(cls, filename: str, s: float = 1.0,
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
    def from_vmec_input(cls, filename: str, **kwargs):
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
        logger.debug('Input file has ntor_boundary={} mpol_boundary={}'
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

        ntheta = kwargs.pop("ntheta", None)
        nphi = kwargs.pop("nphi", None)
        grid_range = kwargs.pop("range", None)

        if ntheta is not None or nphi is not None:
            kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] = Surface.get_quadpoints(
                ntheta=ntheta, nphi=nphi, nfp=nfp, range=grid_range)

        surf = cls(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, **kwargs)

        for j in range(Nfou):
            surf.rc[m[j], n[j] + ntor] = rc[j]
            surf.zs[m[j], n[j] + ntor] = zs[j]
            if not stellsym:
                surf.rs[m[j], n[j] + ntor] = rs[j]
                surf.zc[m[j], n[j] + ntor] = zc[j]

        surf.local_full_x = surf.get_dofs()
        return surf

    @classmethod
    def from_nescoil_input(cls, filename, which_surf, **kwargs):
        """
        Read in a surface from a NESCOIL input file, as generated by the
        BNORM code.

        Args:
            filename: Name of the ``nescin.*`` file to read.
            which_surf: either ``plasma`` or ``current``, will select whether
                to import the plasma boundary or the ``current surface`` 
                (i.e. winding surface) from the file
            kwargs: Any other arguments to pass to the ``SurfaceRZFourier`` 
                constructor. You can specify ``quadpoints_theta`` and 
                ``quadpoints_phi`` here.
        """

        if which_surf not in ['plasma', 'current']:
            raise ValueError('Parameter which_surf must be `plasma` or '
                             + '`current`')

        with open(filename, 'r') as f:
            lines = f.readlines()

        j_line = 0
        nfp = 0
        errmsg = "This does not appear to be a nescin-format file."

        # Scan through file until nfp is found and desired surface is reached
        while True:
            if 'Plasma information from VMEC' in lines[j_line]:
                j_line += 2
                nfp = int(lines[j_line].split()[0])
                continue
            elif which_surf == 'plasma' and 'Plasma Surface' in lines[j_line]:
                assert nfp != 0, errmsg
                break
            elif which_surf == 'current' and 'Current Surface' in lines[j_line]:
                assert nfp != 0, errmsg
                break
            j_line += 1
            assert j_line < len(lines), errmsg

        # Retrieve the number of Fourier harmonics
        j_line += 2
        n_Fourier = int(lines[j_line].split()[0])
        j_line += 3

        # Now read the Fourier amplitudes:
        n = np.full(n_Fourier, 0)
        m = np.full(n_Fourier, 0)
        rc = np.zeros(n_Fourier)
        rs = np.zeros(n_Fourier)
        zc = np.zeros(n_Fourier)
        zs = np.zeros(n_Fourier)
        for j in range(n_Fourier):
            splitline = lines[j + j_line].split()
            m[j] = int(splitline[0])
            n[j] = -int(splitline[1])    # Note the different sign convention
            rc[j] = float(splitline[2])
            zs[j] = float(splitline[3])
            rs[j] = float(splitline[4])
            zc[j] = float(splitline[5])
        assert np.min(m) == 0
        stellsym = np.max(np.abs(rs)) == 0 and np.max(np.abs(zc)) == 0
        mpol = int(np.max(m))
        ntor = int(np.max(np.abs(n)))

        ntheta = kwargs.pop("ntheta", None)
        nphi = kwargs.pop("nphi", None)
        grid_range = kwargs.pop("range", None)

        if ntheta is not None or nphi is not None:
            kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] \
                = Surface.get_quadpoints(ntheta=ntheta, nphi=nphi, nfp=nfp,
                                         range=grid_range)

        surf = cls(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, **kwargs)

        for j in range(n_Fourier):
            surf.rc[m[j], n[j] + ntor] = rc[j]
            surf.zs[m[j], n[j] + ntor] = zs[j]
            if not stellsym:
                surf.rs[m[j], n[j] + ntor] = rs[j]
                surf.zc[m[j], n[j] + ntor] = zc[j]

        surf.local_full_x = surf.get_dofs()
        return surf

    @classmethod
    @SimsoptRequires(Qsc is not None, "from_pyQSC method requires pyQSC module")
    def from_pyQSC(cls, stel: Qsc, r: float = 0.1, ntheta=20, mpol=10, ntor=20, **kwargs):
        """
        Initialize the surface from a pyQSC object. This creates a surface
        from a near-axis equilibrium with a specified minor radius `r` (in meters).

        Args:
            stel: Qsc object with a near-axis equilibrium.
            r: the near-axis coordinate radius (in meters).
            ntheta: number of points in the theta direction for the Fourier transform.
            mpol: number of poloidal Fourier modes for the surface.
            ntor: number of toroidal Fourier modes for the surface.
            kwargs: Any other arguments to pass to the ``SurfaceRZFourier`` constructor.
              You can specify ``quadpoints_theta`` and ``quadpoints_phi`` here.
        """
        # Get surface shape at fixed off-axis toroidal angle phi
        R_2D, Z_2D, _ = stel.Frenet_to_cylindrical(r, ntheta)

        # Fourier transform the result.
        RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, stel.nfp, mpol, ntor, stel.lasym)

        surf = cls(mpol=mpol, ntor=ntor, nfp=stel.nfp, stellsym=not stel.lasym, **kwargs)

        surf.rc[:, :] = RBC.transpose()
        surf.zs[:, :] = ZBS.transpose()
        if stel.lasym:
            surf.rs[:, :] = RBS.transpose()
            surf.zc[:, :] = ZBC.transpose()

        surf.local_full_x = surf.get_dofs()
        return surf

    def copy(self, **kwargs):
        """
        Return a copy of the ``SurfaceRZFourier`` object. 
        A range of relevant parameters of the surface can be passed to this function
        as keyword arguments in order to modify the properties of the returned copy. 
        attributes changed. Keyword arguments accepted:

        Kwargs: 
         ntheta (int): number of quadrature points in the theta direction
         nphi (int): number of quadrature points in the phi direction
         mpol (int): number of poloidal Fourier modes for the surface
         ntor (int): number of toroidal Fourier modes for the surface
         nfp (int): number of field periods
         stellsym (bool): whether the surface is stellarator-symmetric
         quadpoints_theta (NdArray[float]): theta grid points
         quadpoints_phi (NdArray[float]): phi grid points
         range (str): range of the gridpoints either 'full torus', 'field period' or 'half period'. Ignored if quadponts are provided.

        Returns:
            surf: A new SurfaceRZFourier object, with properties specified by kwargs changed.


        """
        otherntheta = self.quadpoints_theta.size
        othernphi = self.quadpoints_phi.size

        ntheta = kwargs.pop("ntheta", otherntheta)
        nphi = kwargs.pop("nphi", othernphi)
        mpol = kwargs.pop("mpol", self.mpol)
        ntor = kwargs.pop("ntor", self.ntor)
        nfp = kwargs.pop("nfp", self.nfp)
        stellsym = kwargs.pop("stellsym", self.stellsym)
        quadpoints_theta = kwargs.pop("quadpoints_theta", None)
        quadpoints_phi = kwargs.pop("quadpoints_phi", None)
        grid_range = kwargs.pop("range", None)

        # recalculate the quadpoints if necessary (grid_range is not stored in the
        # surface object, so assume that if it is given, the gridpoints should be
        # recalculated to the specified size)
        if quadpoints_theta is None and quadpoints_phi is None:
            if ntheta is not otherntheta or nphi is not othernphi or grid_range is not None:
                kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] = Surface.get_quadpoints(
                    ntheta=ntheta, nphi=nphi, nfp=self.nfp, range=grid_range)
            else:
                kwargs["quadpoints_phi"] = self.quadpoints_phi
                kwargs["quadpoints_theta"] = self.quadpoints_theta
        else:
            if quadpoints_theta is None:
                if ntheta is not otherntheta or grid_range is not None:
                    kwargs["quadpoints_theta"] = Surface.get_theta_quadpoints(ntheta)
                else:
                    kwargs["quadpoints_theta"] = self.quadpoints_theta
            else:
                kwargs["quadpoints_theta"] = quadpoints_theta
            if quadpoints_phi is None:
                if nphi is not othernphi or grid_range is not None:
                    kwargs["quadpoints_phi"] = Surface.get_phi_quadpoints(nphi, range=grid_range, nfp=nfp)
                else:
                    kwargs["quadpoints_phi"] = self.quadpoints_phi
            else:
                kwargs["quadpoints_phi"] = quadpoints_phi
        # create new surface in old resolution
        surf = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                                **kwargs)
        surf.x[:] = 0 

        # copy coefficients to the new surface
        for m in range(0, min(mpol, self.mpol)+1):
            this_nmax = min(ntor, self.ntor)
            if m == 0:
                this_nmin = 0
            else: 
                this_nmin = -this_nmax
            for n in range(this_nmin, this_nmax+1):
                surf.set_rc(m, n, self.get_rc(m, n))
                surf.set_zs(m, n, self.get_zs(m, n))
                if not surf.stellsym and not self.stellsym:
                    surf.set_zc(m, n, self.get_zc(m, n))
                    surf.set_rs(m, n, self.get_rs(m, n))

        surf.local_full_x = surf.get_dofs()
        return surf

    def change_resolution(self, mpol, ntor):
        """
        return a new surface with Fourier resolution mpol, ntor
        Args: 
            mpol: new poloidal mode number
            ntor: new toroidal mode number
        Returns: 
            surf: A new SurfaceRZFourier object with the specified resolution.
        """
        return self.copy(mpol=mpol, ntor=ntor)

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
        Derivative of the area with respect to the surface Fourier coefficients.
        Short hand for `Surface.darea_by_dcoeff()`
        """
        return self.darea_by_dcoeff()

    def dvolume(self):
        """
        Derivative of the volume with respect to the surface Fourier coefficients.
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

    def extend_via_normal(self, distance):
        """
        Extend the surface in the normal direction by a uniform distance.

        *NOTE* this modifies the surface in place. use the surface copy
        method if you want to keep the original surface.

        Args:
            distance: The distance to extend the surface.
        """
        if len(self.quadpoints_phi) < 2 * self.ntor + 1:
            raise RuntimeError("Number of phi quadrature points should be at least 2 * ntor + 1")
        if len(self.quadpoints_theta) < 2 * self.mpol + 1:
            raise RuntimeError("Number of theta quadrature points should be at least 2 * mpol + 1")

        # Generate points that are a uniform distance from the surface, though
        # at irregular phi values:
        points = (self.gamma() + self.unitnormal() * distance).reshape((-1, 3))

        R = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        phi = np.arctan2(points[:, 1], points[:, 0])
        Z = points[:, 2]
        n_phi = len(self.quadpoints_phi)
        n_theta = len(self.quadpoints_theta)
        theta_1d = np.linspace(0, 2 * np.pi, len(self.quadpoints_theta), endpoint=False)
        theta = (theta_1d[None, :] * np.ones((n_phi, n_theta))).flatten()

        # Evaluate the basis functions at the new (phi, theta) points:
        n_cos_dofs = (2 * self.ntor + 1) * self.mpol + self.ntor + 1
        cos_terms = np.cos(self.m[None, :n_cos_dofs] * theta[:, None] - self.nfp * self.n[None, :n_cos_dofs] * phi[:, None])
        sin_terms = np.sin(self.m[None, 1:n_cos_dofs] * theta[:, None] - self.nfp * self.n[None, 1:n_cos_dofs] * phi[:, None])

        if self.stellsym:
            R_basis_funcs = cos_terms
            Z_basis_funcs = sin_terms
        else:
            R_basis_funcs = np.concatenate((cos_terms, sin_terms), axis=1)
            Z_basis_funcs = R_basis_funcs

        # Fit the mode amplitudes to the new points.
        # For some os + python + numpy versions there are errors with numpy linear algebra.
        # This causes a test to fail in the Github Actions CI, as of 3/15/2025.
        # A solution suggested by Ken Hammond is to call the function a 2nd time
        # if it fails the first time. Ref:
        # https://github.com/hiddenSymmetries/simsopt/pull/467#issuecomment-2691164408
        try:
            R_dofs = np.linalg.lstsq(R_basis_funcs, R, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Try a second time
            R_dofs = np.linalg.lstsq(R_basis_funcs, R, rcond=None)[0]

        try:
            Z_dofs = np.linalg.lstsq(Z_basis_funcs, Z, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Try a second time
            Z_dofs = np.linalg.lstsq(Z_basis_funcs, Z, rcond=None)[0]

        self.x = np.concatenate((R_dofs, Z_dofs))

    def fourier_transform_scalar(self, scalar, mpol=None, ntor=None, normalization=None, **kwargs):
        r"""
        Compute the Fourier components of a scalar on the surface. The scalar
        is evaluated at the quadrature points on the surface. 
        The Fourier uses the conventions of the ``SurfaceRZFourier`` series, 
        with ``npol`` going from ``-ntor`` to ``ntor`` and ``mpol`` from 0 to ``mpol``
        i.e.: 

        .. math::
            f(\theta, \phi) = \sum_{m=0}^{mpol} \sum_{n=-npol}^{npol} A^{mn}_s \sin(m\theta - n N_{fp} \phi)
            + A^{mn}_c \cos(m\theta - n N_{fp} \phi)

        Where the cosine series is only evaluated if the surface is not stellarator
        symmetric (if the scalar does not adhere to the symmetry of the surface, 
        request the cosine series by setting the kwarg ``stellsym=False``)
        By default, the poloidal and toroidal resolution are the same as those
        of the surface, but different quantities can be specified in the kwargs. 

        Args:
            scalar: 2D array of shape ``(numquadpoints_phi, numquadpoints_theta)``.
            mpol: maximum poloidal mode number of the transform, if ``None``,
                the mpol attribute of the surface is used.
            ntor: maximum toroidal mode number of the transform if ``None``, 
                the ntor attribute of the surface is used.
            normalization: (optional) Fourier transform normalization. Can be: 
              ``None``: forward and back transform are not normalized.
              ``float``: forward transform is divided by this number.
            stellsym: (optional) boolean to override the stellsym attribute 
                of the surface if you want to force the calculation of the
                cosine series

        Returns:
            2-element tuple ``(A_mns, A_mnc)``, where ``A_mns`` is a 2D array of shape ``(mpol+1, 2*ntor+1)`` containing the sine
            coefficients, and ``A_mnc`` is a  2D array of shape ``(mpol+1, 2*ntor+1)`` containing the cosine coefficients 
            (these are zero if the surface is stellarator symmetric).
        """
        assert scalar.shape[0] == self.quadpoints_phi.size, "scalar must be evaluated at the quadrature points on the surface.\n the scalar you passed in has shape {}".format(scalar.shape)
        assert scalar.shape[1] == self.quadpoints_theta.size, "scalar must be evaluated at the quadrature points on the surface.\n the scalar you passed in has shape {}".format(scalar.shape)
        stellsym = kwargs.pop('stellsym', self.stellsym)
        if mpol is None:
            try:
                mpol = self.mpol
            except AttributeError:
                raise ValueError("mpol must be specified")
        if ntor is None:
            try:
                ntor = self.ntor
            except AttributeError:
                raise ValueError("ntor must be specified")
        A_mns = np.zeros((int(mpol + 1), int(2 * ntor + 1)))  # sine coefficients
        A_mnc = np.zeros((int(mpol + 1), int(2 * ntor + 1)))  # cosine coefficients
        ntheta_grid = len(self.quadpoints_theta)
        nphi_grid = len(self.quadpoints_phi)

        factor = 2.0 / (ntheta_grid * nphi_grid)

        phi2d, theta2d = np.meshgrid(2 * np.pi * self.quadpoints_phi,
                                     2 * np.pi * self.quadpoints_theta,
                                     indexing="ij")

        for m in range(mpol + 1):
            nmin = -ntor
            if m == 0:
                nmin = 1
            for n in range(nmin, ntor+1):
                angle = m * theta2d - n * self.nfp * phi2d
                sinangle = np.sin(angle)
                factor2 = factor
                # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
                if np.mod(ntheta_grid, 2) == 0 and m == (ntheta_grid/2):
                    factor2 = factor2 / 2
                if np.mod(nphi_grid, 2) == 0 and abs(n) == (nphi_grid/2):
                    factor2 = factor2 / 2
                A_mns[m, n + ntor] = np.sum(scalar * sinangle * factor2)
                if not stellsym:
                    cosangle = np.cos(angle)
                    A_mnc[m, n + ntor] = np.sum(scalar * cosangle * factor2)

        if not stellsym:
            A_mnc[0, ntor] = np.sum(scalar) / (ntheta_grid * nphi_grid)
        if normalization is not None:
            if not isinstance(normalization, float):
                raise ValueError("normalization must be a float")
            A_mns = A_mns / normalization
            A_mnc = A_mnc / normalization

        return A_mns, A_mnc

    def inverse_fourier_transform_scalar(self, A_mns, A_mnc, normalization=None, **kwargs):
        r"""
        Compute the inverse Fourier transform of a scalar on the surface, specified by the Fourier coefficients. The quantity must be
        is evaluated at the quadrature points on the surface. The Fourier
        transform is defined as
        :math:`f(\theta, \phi) = \Sum_{m=0}^{mpol} \Sum_{n=-npol}^{npol} A^{mn}_s \sin(m\theta - n*Nfp*\phi) + A^{mn}_c \cos(m\theta - n*Nfp*\phi)`
        Where the cosine series is only evaluated if the surface is not stellarator symmetric.
        *Arguments*:

        - A_mns: 2D array of shape (mpol+1, 2*ntor+1) containing the sine coefficients
        - A_mnc: 2D array of shape (mpol+1, 2*ntor+1) containing the cosine coefficients 
            (these are zero if the surface is stellarator symmetric)

        *Optional keyword arguments*:

        - normalization: Fourier transform normalization. Can be:
            None: forward and back transform are not normalized
            float: inverse transform is multiplied by this number
        - stellsym: boolean to override the stellsym attribute of the surface

        """
        mpol = A_mns.shape[0] - 1
        ntor = int((A_mns.shape[1] - 1) / 2)
        stellsym = kwargs.pop('stellsym', self.stellsym)
        ntheta_grid = len(self.quadpoints_theta)
        nphi_grid = len(self.quadpoints_phi)

        phi2d, theta2d = np.meshgrid(2 * np.pi * self.quadpoints_phi,
                                     2 * np.pi * self.quadpoints_theta,
                                     indexing="ij")

        scalars = np.zeros((nphi_grid, ntheta_grid))
        for m in range(mpol + 1):
            nmin = -ntor
            if m == 0:
                nmin = 1
            for n in range(nmin, ntor+1):
                angle = m * theta2d - n * self.nfp * phi2d
                sinangle = np.sin(angle)
                scalars = scalars + A_mns[m, n + ntor] * sinangle
                if not stellsym:
                    cosangle = np.cos(angle)
                    scalars = scalars + A_mnc[m, n + ntor] * cosangle

        if not stellsym:
            scalars = scalars + A_mnc[0, ntor]
        if normalization is not None:
            if not isinstance(normalization, float):
                raise ValueError("normalization must be a float")
            scalars = scalars * normalization
        return scalars

    def make_rotating_ellipse(self, major_radius, minor_radius, elongation, torsion=0):
        r"""
        Set the surface shape to be a rotating ellipse with the given
        parameters.

        Values of ``elongation`` larger than 1 will result in the elliptical
        cross-section at :math:`\phi=0` being taller than it is wide.
        Values of ``elongation`` less than 1 will result in the elliptical
        cross-section at :math:`\phi=0` being wider than it is tall.

        The sign convention is such that both the rotating elongation and
        positive ``torsion`` will contribute positively to iota according to
        VMEC's sign convention.

        Args:
            major_radius: Average major radius of the surface.
            minor_radius: Average minor radius of the surface.
            elongation: Elongation of the elliptical cross-section.
            torsion: Value to use for the (m,n)=(0,1) mode of RC and -ZS, which
                controls the torsion of the magnetic axis.
        """

        self.local_full_x = np.zeros_like(self.local_full_x)
        self.set_rc(0, 0, major_radius)
        self.set_rc(0, 1, torsion)
        self.set_zs(0, 1, -torsion)

        sqrt_elong = np.sqrt(elongation)
        amplitude = 0.5 * minor_radius * (1 / sqrt_elong - sqrt_elong)
        self.set_rc(1, 1, amplitude)
        self.set_zs(1, 1, -amplitude)

        amplitude = 0.5 * minor_radius * (1 / sqrt_elong + sqrt_elong)
        self.set_rc(1, 0, amplitude)
        self.set_zs(1, 0, amplitude)

    def flip_z(self):
        """
        Flip the sign of the z coordinate. This will flip the sign of the
        rotational transform of a plasma bounded by this surface. Note that vmec
        requires θ to increase as you move from the outboard to inboard side
        over the top of the surface. This z-flip transformation will reverse
        that direction.
        """
        self.zs = -self.zs
        if not self.stellsym:
            self.zc = -self.zc
        self.local_full_x = self.get_dofs()

    def flip_phi(self):
        """
        Flip the sign of the toroidal angle ϕ, i.e. mirror-reflect the surface
        about the x-z plane. This will reverse the sign of the rotational
        transform of a plasma bounded by this surface, without reversing the
        direction in which θ increases. This is the best way to flip the sign of
        the rotational transform for a vmec calculation.
        """
        # Handle m=0 modes, where there are no modes with negative n.
        # cos(-nϕ) → cos(nϕ) = cos(-nϕ)
        # sin(-nϕ) → sin(nϕ) = -sin(-nϕ)
        for n in range(1, self.ntor + 1):
            self.zs[0, n + self.ntor] = -self.zs[0, n + self.ntor]
            if not self.stellsym:
                self.rs[0, n + self.ntor] = -self.rs[0, n + self.ntor]

        # Handle m>0 modes: swap the positive and negative n modes
        for m in range(1, self.mpol + 1):
            for n in range(1, self.ntor + 1):
                temp = self.rc[m, n + self.ntor]
                self.rc[m, n + self.ntor] = self.rc[m, -n + self.ntor]
                self.rc[m, -n + self.ntor] = temp

                temp = self.zs[m, n + self.ntor]
                self.zs[m, n + self.ntor] = self.zs[m, -n + self.ntor]
                self.zs[m, -n + self.ntor] = temp

                if not self.stellsym:
                    temp = self.rs[m, n + self.ntor]
                    self.rs[m, n + self.ntor] = self.rs[m, -n + self.ntor]
                    self.rs[m, -n + self.ntor] = temp

                    temp = self.zc[m, n + self.ntor]
                    self.zc[m, n + self.ntor] = self.zc[m, -n + self.ntor]
                    self.zc[m, -n + self.ntor] = temp

        self.local_full_x = self.get_dofs()

    def flip_theta(self):
        """
        Flip the direction in which the poloidal angle θ increases. The physical
        shape of the surface in 3D will not change, only its parameterization.
        Note that vmec requires θ to increase as you move from the outboard to
        inboard side over the top of the surface. This transformation will
        reverse that direction.
        """
        # We don't change the m=0 modes since they are independent of θ.
        for m in range(1, self.mpol + 1):
            # For m>0 modes with n=0:
            # cos(mθ) → cos(-mθ) =  cos(mθ)
            # sin(mθ) → sin(-mθ) = -sin(mθ)
            # So, flip the sign of the sin terms
            self.zs[m, self.ntor] = -self.zs[m, self.ntor]
            if not self.stellsym:
                self.rs[m, self.ntor] = -self.rs[m, self.ntor]

            # For m>0 modes with nonzero n:
            # cos(mθ-nϕ) → cos(-mθ-nϕ) =  cos(mθ+nϕ)
            # sin(mθ-nϕ) → sin(-mθ-nϕ) = -sin(mθ+nϕ)
            # So, swap the positive and negative n modes,
            # with a sign flip for the sin terms only.
            for n in range(1, self.ntor + 1):
                temp = self.rc[m, n + self.ntor]
                self.rc[m, n + self.ntor] = self.rc[m, -n + self.ntor]
                self.rc[m, -n + self.ntor] = temp

                temp = self.zs[m, n + self.ntor]
                self.zs[m, n + self.ntor] = -self.zs[m, -n + self.ntor]
                self.zs[m, -n + self.ntor] = -temp

                if not self.stellsym:
                    temp = self.rs[m, n + self.ntor]
                    self.rs[m, n + self.ntor] = -self.rs[m, -n + self.ntor]
                    self.rs[m, -n + self.ntor] = -temp

                    temp = self.zc[m, n + self.ntor]
                    self.zc[m, n + self.ntor] = self.zc[m, -n + self.ntor]
                    self.zc[m, -n + self.ntor] = temp

        self.local_full_x = self.get_dofs()

    def rotate_half_field_period(self):
        """
        Rotate the surface toroidally by half a field period.

        This operation is useful when you have a surface with the bean
        cross-section at ϕ = π / nfp, and you want to rotate it so that the bean
        is at ϕ = 0.
        """
        x = self.local_full_x
        # Flip the sign of all modes with odd n:
        odd_ns = (self.n % 2 == 1)
        x[odd_ns] = -x[odd_ns]
        self.local_full_x = x

    def shift_theta_by_half(self):
        """
        Shift the origin of the poloidal angle θ by 1/2.

        This operation is useful when you have a surface with θ=0 at the inboard
        side instead of the usual outboard side.
        """
        x = self.local_full_x
        # Flip the sign of all modes with odd m:
        odd_ms = (self.m % 2 == 1)
        x[odd_ms] = -x[odd_ms]
        self.local_full_x = x


    def spectral_width(self, power=2):
        r"""
        Compute the spectral width of the surface Fourier coefficients.

        For this function, the spectral width is defined as:

        .. math::
            W = \frac{1}{2} \sum_{m,n} \left( \frac{ (r_{m,n}^c)^2 + (r_{m,n}^s)^2 + (z_{m,n}^c)^2 + (z_{m,n}^s)^2 }{ a^2 } \right) (m^2 + n^2)^{p}

        where :math:`r_{m,n}^c`, :math:`r_{m,n}^s`, :math:`z_{m,n}^c`, and
        :math:`z_{m,n}^s` are the Fourier coefficients of the surface shape,
        :math:`a` is the minor radius of the surface, and :math:`p` is a
        constant corresponding to the ``power`` argument.

        This quantity is similar to the spectral width discussed in various
        papers related to VMEC, such as Hirshman and Meier, Physics of Fluids
        28, 1387 (1985). However the definition here is not exactly the same,
        due to the 1/2 factor, different normalization, and inclusion of :math:`n`
        in the weighting factor.

        See also the method :func:`~simsopt.geo.SurfaceRZFourier.condense_spectrum`,
        which minimizes this quantity by adjusting the poloidal angle.

        Args:
            power: The power to which to raise the Fourier coefficients
                when computing the spectral width. Default is 2.

        Returns:
            The spectral width as a float.
        """
        return 0.5 * np.sum((self.x / self.minor_radius())**2 * (self.m**2 + self.n**2)**power)

    def condense_spectrum(
        self,
        n_theta=None,
        n_phi=None,
        power=2,
        maxiter=None,
        method="SLSQP",
        epsilon=1e-3,
        Fourier_continuation=True,
        verbose=True,
    ):
        r"""Apply spectral condensation so the Fourier amplitudes ``rc`` and ``zs`` decay
        more rapidly with ``m`` and ``n``.

        This function has similarities to the spectral condensation
        used in VMEC, see e.g. Hirshman and Meier, Physics of Fluids 28, 1387
        (1985), although the algorithm used here is different. The idea is to
        reparameterize the poloidal angle so that the shape can be represented
        with as few Fourier modes as possible.

        Let :math:`\theta_1` be the original poloidal angle, and let
        :math:`\theta_2` be a new poloidal angle with :math:`\theta_2 = \theta_1
        + \lambda(\theta_1, \phi)` for some function :math:`\lambda`. This
        routine optimizes :math:`\lambda` to minimize the spectral width while
        preserving the surface shape in real space as well as possible. The
        objective function minimized is the same function shown in the
        documentation for :func:`~simsopt.geo.SurfaceRZFourier.spectral_width`.
        At each iteration, i.e. for any specific choice of :math:`\lambda`,
        updated values of ``rc`` and ``zs`` are computed by Fourier-transforming
        points on the original surface, with the results used to evaluate the
        objective. Gradients are evaluated using JAX. Optionally, a constrained
        optimization method can be used to ensure that the maximum change to the
        surface shape is below a specified fraction of the minor radius, given
        by the ``epsilon`` argument.

        If ``n_theta`` and ``n_phi`` are not specified, they default to ``3*mpol+1``
        and ``3*ntor+1``, respectively.

        The optimization algorithm is specified by ``method``. The recommended
        settings are either ``SLSQP`` or ``trust-constr``. This argument accepts
        any method supported by `scipy.optimize.minimize
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__
        (``L-BFGS-B``, ``BFGS``, etc.) or any of the methods supported by
        `scipy.optimize.least_squares
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`__
        (``trf``, ``dogleg``, or ``lm``). The constrained algorithms ``SLSQP``
        or ``trust-constr`` are recommended to ensure that the surface shape
        does not change too much. You can try the unconstrained algorithms,
        which may be faster and result in a smaller spectral width, but not
        always, and sometimes the surface shape will not be preserved well.
        Based on experience, the least-squares methods achieve slightly lower
        values of the spectral width, but take more time per iteration. Any of
        the algorithms ``trf``, ``BFGS``, and ``lm`` are likely to work.

        If ``maxiter`` is not specified, it defaults to 25 for the least-squares
        methods, and 100 for other methods.

        If ``Fourier_continuation`` is True, first the Fourier modes of
        :math:`\lambda` with :math:`|m|` and :math:`|n|` up to 1 are optimized,
        then modes with :math:`|m|` and :math:`|n|` up to 2, and so on. If
        ``False``, all modes of :math:`\lambda` are optimized at once. Using
        Fourier continuation generally results in better spectral condensation,
        but it requires more time.

        The degrees of freedom of the original surface are not modified.
        Instead, a copy of the surface is returned with the new optimized
        degrees of freedom.

        Spectral condensation always results in some finite change to the
        surface shape. Ideally this change is as small as possible. To measure
        this change, the return value ``max_RZ_error`` gives the maximum
        absolute change to ``R`` and ``Z`` on the :math:`\theta_1` grid points,
        normalized to the minor radius. If this value is approximately equal to
        ``epsilon`` for the constrained algorithms, or unacceptably large for
        the unconstrained algorithms, you can use the
        :func:`~simsopt.geo.SurfaceRZFourier.change_resolution` method to
        increase ``mpol`` and ``ntor`` of the surface before calling
        ``condense_spectrum``, so both the Fourier and grid resolutions used in
        ``condense_spectrum`` are increased. You may then be able to call
        :func:`~simsopt.geo.SurfaceRZFourier.change_resolution` again to reduce
        ``mpol`` and ``ntor`` after the condensation.

        The ``data`` dictionary that is returned contains information about
        the optimization. The most noteworthy entries are ``max_RZ_error`` and
        ``spectral_width_reduction``. The former indicates how well the surface
        shape was preserved, and the latter indicates how much the spectral
        width was reduced by the optimization.
        
        The function
        :func:`~simsopt.geo.plot_spectral_condensation` is
        useful for visualizing the results of spectral condensation.

        Parameters
        ----------
            n_theta: int or None, optional
                Number of theta grid points to use for fitting.
            n_phi: int or None, optional
                Number of phi grid points to use for fitting.
            power: float, optional
                Power to use in the spectral width objective function.
            maxiter: int or None, optional
                Maximum number of iterations for the optimization.
            method: str or None, optional
                Optimization method to use. See above for details.
            epsilon: float, optional
                Maximum allowed change to R and Z on the θ₁ grid points,
                normalized to the minor radius. Only matters if
                ``method=="SLSQP"`` or ``method=="trust-constr"``.
            Fourier_continuation: bool, optional
                If True, increase the Fourier resolution of λ in steps (slower).
                If False, optimize all Fourier modes of λ at once (faster).
            verbose: bool, optional
                Whether to print progress messages.

        Returns
        -------
            surface: SurfaceRZFourier
                A new SurfaceRZFourier object with the condensed spectrum.
            data: dict
                A dictionary containing the following data from the
                optimization:
                
                - "method" (str):
                    The optimization algorithm used (value of the ``method`` argument).
                - "maxiter" (int):
                    The maximum number of iterations for the optimizer.
                - "n_theta" (int):
                    Number of theta grid points used for fitting.
                - "n_phi" (int):
                    Number of phi grid points used for fitting.
                - "power" (float):
                    The exponent used in the definition of spectral width.
                - "epsilon" (float):
                    The constraint tolerance on pointwise R and Z changes, normalized
                    to the minor radius (only relevant for constrained optimizers).
                - "Fourier_continuation" (bool):
                    Whether Fourier continuation (progressively increasing mode count)
                    was used.
                - "initial_objective" (float):
                    Value of the scalar objective (spectral width) before optimization.
                - "final_objective" (float):
                    Value of the scalar objective after the final optimization step.
                - "spectral_width_reduction" (float):
                    The ratio final_objective / initial_objective
                - "max_RZ_error" (float):
                    The maximum absolute change in R or Z on the θ₁ grid points,
                    normalized by the minor radius, for the optimized
                    poloidal angle.
                - "m" (ndarray of int):
                    The array of Fourier poloidal mode numbers used for the
                    angle difference λ.
                - "n" (ndarray of int):
                    The array of Fourier toroidal mode numbers used for the
                    angle difference λ.
                - "lambda_mn" (ndarray of float):
                    The optimized Fourier coefficients for the reparameterization
                    function λ (sin coefficients only for the
                    stellarator-symmetric case implemented here). These coefficients
                    are in the same ordering as "m" and "n".
                - "x_scale" (ndarray of float):
                    The exponential spectral scaling applied to λ modes.
                - "elapsed_time" (float):
                    Wall-clock time in seconds spent in the condensation routine.
                - "minor_radius" (float):
                    The minor radius used to normalize R and Z errors and 
                    the objective.
                - "theta1_1d" (ndarray, shape (n_theta*n_phi,)):
                    The flattened original poloidal angles (θ₁) corresponding to the
                    quadrature points on the surface. Values are in
                    radians in the interval [0, 2π).
                - "phi_1d" (ndarray, shape (n_theta*n_phi,)):
                    The flattened toroidal angles corresponding to the quadrature
                    points on the surface. Values are in radians.
                - "theta1_2d" (ndarray, shape (n_phi, n_theta)):
                    The 2-D grid of original poloidal angles used to evaluate the
                    surface (meshgrid of quadpoints_theta scaled by 2π).
                - "phi_2d" (ndarray, shape (n_phi, n_theta)):
                    The 2-D grid of toroidal angles used to evaluate the surface
                    (meshgrid of quadpoints_phi scaled by 2π).
                - "theta_optimized" (ndarray, shape (n_theta*n_phi,)):
                    The optimized poloidal angles θ₂ = θ₁ + λ(θ₁, φ) evaluated at the
                    quadrature points (flattened).
                - "original_x" (ndarray):
                    The Fourier amplitudes of the original surface prior to
                    condensation.
                - "condensed_x" (ndarray):
                    The Fourier amplitudes of the surface after condensation.
        """
        if not self.stellsym:
            raise NotImplementedError("condense_spectrum is only implemented for stellarator-symmetric surfaces")
        
        mpol = self.mpol
        ntor = self.ntor
        nfp = self.nfp
        minor_radius = self.minor_radius()
        if verbose:
            print("minor radius before condensation:", minor_radius)
        original_x = self.local_full_x.copy()

        if n_theta is None:
            n_theta = 3 * mpol + 1
        if n_phi is None:
            n_phi = 3 * ntor + 1

        # Avoid error about finding dphi when n_phi=1:
        n_phi = max(n_phi, 2)

        method_is_lstsq = (method.lower() in ["trf", "lm", "dogleg"])
        if maxiter is None:
            if method_is_lstsq:
                maxiter = 25
            else:
                maxiter = 100

        # Make a copy of the surface with the desired grid resolution:
        surf = self.copy(range="half period", nphi=n_phi, ntheta=n_theta)

        # Index in .x where Rmnc and Zmns are separated:
        n_Rmn = (len(surf.x) + 1) // 2
        m_for_R = surf.m[:n_Rmn]
        n_for_R = surf.n[:n_Rmn]
        m_for_Z = surf.m[n_Rmn:]
        n_for_Z = surf.n[n_Rmn:]

        # lambda could in principle have different mpol and ntor than the original surface.
        # But it works reasonably well to use the same mpol and ntor:
        mpol_for_lambda = mpol
        ntor_for_lambda = ntor
        # surf_dummy is just a convenient way to get the m and n arrays for lambda
        surf_dummy = SurfaceRZFourier(
            nfp=nfp,
            mpol=mpol_for_lambda,
            ntor=ntor_for_lambda,
        )
        n_Rmn_for_lambda = (len(surf_dummy.x) + 1) // 2
        m_for_lambda_full = surf_dummy.m[n_Rmn_for_lambda:]
        n_for_lambda_full = surf_dummy.n[n_Rmn_for_lambda:]
        max_mn_lambda = max(max(m_for_lambda_full), max(n_for_lambda_full))

        def compute_x_scale(m_for_lambda, n_for_lambda):
            # Exponential spectral scaling, which will be applied to lambda. See
            # Jang, Conlin, & Landreman, (2025)
            # https://arxiv.org/abs/2509.16320
            x_scale = np.exp(-1.0 * np.sqrt(m_for_lambda**2 + n_for_lambda**2))
            return x_scale

        x_scale_full = compute_x_scale(m_for_lambda_full, n_for_lambda_full)

        # Evaluate the surface shape on uniformly spaced points in theta1 and phi:
        gamma = surf.gamma()
        R = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
        Z = gamma[:, :, 2]
        R_to_fit = R.flatten()
        Z_to_fit = Z.flatten()

        theta1_2d, phi_2d = np.meshgrid(2 * np.pi * surf.quadpoints_theta, 2 * np.pi * surf.quadpoints_phi)
        assert theta1_2d.shape == (n_phi, n_theta)
        phi_1d = phi_2d.flatten()
        theta1_1d = theta1_2d.flatten()

        def lambda_Fourier_to_grid(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            scaled_lambda_dofs = lambda_dofs * x_scale
            lambd = jnp.sum(scaled_lambda_dofs[None, :] * jnp.sin(
                m_for_lambda[None, :] * theta1_1d[:, None] - nfp * n_for_lambda[None, :] * phi_1d[:, None]
            ), axis=1)
            theta2_1d = theta1_1d + lambd
            return theta2_1d
        
        # Same as compute_r2mn_and_z2mn, but also returns theta2_1d for use in computing R and Z errors
        def _compute_r2mn_and_z2mn(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            theta2_1d = lambda_Fourier_to_grid(lambda_dofs, m_for_lambda, n_for_lambda, x_scale)

            scaled_lambda_dofs = lambda_dofs * x_scale
            d_theta2_d_theta1 = 1 + jnp.sum(scaled_lambda_dofs[None, :] * m_for_lambda[None, :] * jnp.cos(
                m_for_lambda[None, :] * theta1_1d[:, None] - nfp * n_for_lambda[None, :] * phi_1d[:, None]
            ), axis=1)
            # Order of indices: (point index, mode index)
            Rmnc_new = 2 * jnp.mean(d_theta2_d_theta1[:, None] * R_to_fit[:, None] * jnp.cos(
                m_for_R[None, :] * theta2_1d[:, None] - nfp * n_for_R[None, :] * phi_1d[:, None]
            ), axis=0)
            Rmnc_new = Rmnc_new.at[0].set(Rmnc_new[0] * 0.5)
            Zmns_new = 2 * jnp.mean(d_theta2_d_theta1[:, None] * Z_to_fit[:, None] * jnp.sin(
                m_for_Z[None, :] * theta2_1d[:, None] - nfp * n_for_Z[None, :] * phi_1d[:, None]
            ), axis=0)

            return Rmnc_new, Zmns_new, theta2_1d
        
        def compute_r2mn_and_z2mn(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            Rmnc_new, Zmns_new, _ = _compute_r2mn_and_z2mn(lambda_dofs, m_for_lambda, n_for_lambda, x_scale)
            return jnp.concatenate([Rmnc_new, Zmns_new])

        @jax.jit
        def compute_RZ_errors(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            Rmnc_new, Zmns_new, theta2_1d = _compute_r2mn_and_z2mn(lambda_dofs, m_for_lambda, n_for_lambda, x_scale)
            R_new = jnp.sum(Rmnc_new[None, :] * jnp.cos(m_for_R[None, :] * theta2_1d[:, None] - nfp * n_for_R[None, :] * phi_1d[:, None]), axis=1)
            Z_new = jnp.sum(Zmns_new[None, :] * jnp.sin(m_for_Z[None, :] * theta2_1d[:, None] - nfp * n_for_Z[None, :] * phi_1d[:, None]), axis=1)
            # Scale errors so the constraints are at +/- 1:
            R_error = (R_new - R_to_fit) / (minor_radius * epsilon)
            Z_error = (Z_new - Z_to_fit) / (minor_radius * epsilon)
            return jnp.concatenate([R_error, Z_error])

        def compute_max_RZ_error(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            RZ_errors = compute_RZ_errors(lambda_dofs, m_for_lambda, n_for_lambda, x_scale)
            max_error = np.max(np.abs(RZ_errors)) * epsilon
            if verbose:
                print(f"(Max error in fitting R or Z of points) / minor_radius: {max_error:.3e}", flush=True)
            return max_error

        @jax.jit
        def residuals_func(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            """Objective function for spectral width."""
            x = compute_r2mn_and_z2mn(lambda_dofs, m_for_lambda, n_for_lambda, x_scale)
            residuals = (x / minor_radius) * (surf.m**2 + surf.n**2)**(power * 0.5)
            # The first residual is always zero (since m=n=0) so there is no need to include it:
            return residuals[1:]

        @jax.jit
        def scalar_objective(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            residuals = residuals_func(lambda_dofs, m_for_lambda, n_for_lambda, x_scale)
            return 0.5 * jnp.sum(residuals**2)

        iteration_counter = {"count": 0}

        def scalar_objective_with_printing(lambda_dofs, m_for_lambda, n_for_lambda, x_scale):
            obj_value = scalar_objective(lambda_dofs, m_for_lambda, n_for_lambda, x_scale)
            iteration_counter["count"] += 1
            obj_value_float = obj_value.astype(float)
            print(f"Iteration {iteration_counter['count']:5}: objective = {obj_value_float:.6e}")
            return obj_value

        n_constraints = R_to_fit.size + Z_to_fit.size

        jac_constraints = jax.jit(jax.jacfwd(compute_RZ_errors))

        initial_objective = scalar_objective(
            np.zeros_like(m_for_lambda_full),
            m_for_lambda_full,
            n_for_lambda_full,
            x_scale_full,
        )
        start_time = time.time()

        if Fourier_continuation:
            mnmax_steps = np.arange(1, max_mn_lambda + 1)
        else:
            mnmax_steps = [max_mn_lambda]

        previous_m_for_lambda = None
        previous_n_for_lambda = None
        lambda_dofs_optimized = None
        # Fourier continuation loop:
        for mnmax in mnmax_steps:
            if Fourier_continuation and verbose:
                line_width = 120
                print("*" * line_width)
                print(f"Beginning Fourier continuation step with mnmax = {mnmax}")
                print("*" * line_width, flush=True)

            # Select only lambda dofs with |m|, |n| <= mnmax:
            indices_to_keep = np.where((np.abs(m_for_lambda_full) <= mnmax) & (np.abs(n_for_lambda_full) <= mnmax))[0]
            m_for_lambda = m_for_lambda_full[indices_to_keep]
            n_for_lambda = n_for_lambda_full[indices_to_keep]
            x_scale = compute_x_scale(m_for_lambda, n_for_lambda)

            lambda_dofs = np.zeros(len(m_for_lambda))
            if Fourier_continuation and mnmax > 1:
                # Copy lambda_dofs_optimized from the previous Fourier
                # continuation step to the new set of dofs:
                for j_new, (m_new, n_new) in enumerate(zip(m_for_lambda, n_for_lambda)):
                    for j_old, (m_old, n_old) in enumerate(zip(previous_m_for_lambda, previous_n_for_lambda)):
                        if (m_new == m_old) and (n_new == n_old):
                            lambda_dofs[j_new] = lambda_dofs_optimized[j_old]

            if method_is_lstsq:
                jac_fn = jax.jit(jax.jacfwd(residuals_func))
                if verbose:
                    verbose_for_least_squares = 2
                else:
                    verbose_for_least_squares = 0

                res = least_squares(
                    residuals_func,
                    lambda_dofs,
                    jac=jac_fn,
                    method=method,
                    verbose=verbose_for_least_squares,
                    max_nfev=maxiter,
                    args=(m_for_lambda, n_for_lambda, x_scale)
                )
            elif method in ["SLSQP", "trust-constr"]:
                # Constrained optimization methods:
                constraints = NonlinearConstraint(
                    lambda x: compute_RZ_errors(x, m_for_lambda, n_for_lambda, x_scale),
                    -jnp.ones(n_constraints),  # Lower bounds
                    jnp.ones(n_constraints),  # Upper bounds
                    jac=lambda x: jac_constraints(x, m_for_lambda, n_for_lambda, x_scale),
                )

                options = {"maxiter": maxiter}
                if verbose:
                    if method == "trust-constr":
                        options["verbose"] = 3
                        objective_to_use = scalar_objective
                    else:
                        # SLSQP does not have a 'verbose' option
                        objective_to_use = scalar_objective_with_printing
                else:
                    objective_to_use = scalar_objective

                grad_objective = jax.jit(jax.grad(scalar_objective))
                res = minimize(
                    objective_to_use,
                    lambda_dofs,
                    jac=grad_objective,
                    method=method,
                    constraints=constraints,
                    options=options,
                    args=(m_for_lambda, n_for_lambda, x_scale)
                )
            else:
                # Other non-least-squares methods:
                grad_objective = jax.jit(jax.grad(scalar_objective))
                if verbose:
                    objective_to_use = scalar_objective_with_printing
                else:
                    objective_to_use = scalar_objective

                res = minimize(
                    objective_to_use,
                    lambda_dofs,
                    jac=grad_objective,
                    method=method,
                    options={"maxiter": maxiter, "disp": verbose},
                    args=(m_for_lambda, n_for_lambda, x_scale)
                )

            lambda_dofs_optimized = res.x
            previous_m_for_lambda = m_for_lambda
            previous_n_for_lambda = n_for_lambda
            max_RZ_error = compute_max_RZ_error(lambda_dofs_optimized, m_for_lambda, n_for_lambda, x_scale)

        elapsed_time = time.time() - start_time
        if verbose:
            print(res)
            print("Time taken (s):", elapsed_time, flush=True)

        # Update surface with optimized parameters
        lambda_dofs_optimized = res.x
        theta_optimized = lambda_Fourier_to_grid(lambda_dofs_optimized, m_for_lambda_full, n_for_lambda_full, x_scale_full)
        final_objective = scalar_objective(lambda_dofs_optimized, m_for_lambda_full, n_for_lambda_full, x_scale_full)
        surf_to_return = self.copy()
        surf_to_return.local_full_x = compute_r2mn_and_z2mn(lambda_dofs_optimized, m_for_lambda_full, n_for_lambda_full, x_scale_full)

        data = {
            "method": method,
            "maxiter": maxiter,
            "n_theta": n_theta,
            "n_phi": n_phi,
            "power": power,
            "epsilon": epsilon,
            "Fourier_continuation": Fourier_continuation,
            "initial_objective": float(initial_objective),
            "final_objective": float(final_objective),
            "spectral_width_reduction": float(final_objective / initial_objective),
            "max_RZ_error": float(max_RZ_error),
            "m": m_for_lambda_full,
            "n": n_for_lambda_full,
            "lambda_mn": lambda_dofs_optimized,
            "x_scale": x_scale_full,
            "elapsed_time": elapsed_time,
            "minor_radius": minor_radius,
            "theta1_1d": theta1_1d,
            "phi_1d": phi_1d,
            "theta1_2d": theta1_2d,
            "phi_2d": phi_2d,
            "theta_optimized": theta_optimized,
            "original_x": original_x,
            "condensed_x": surf_to_return.local_full_x,
        }
        return surf_to_return, data

    return_fn_map = {'area': sopp.SurfaceRZFourier.area,
                     'volume': sopp.SurfaceRZFourier.volume,
                     'aspect-ratio': Surface.aspect_ratio}

    def variational_spec_cond(
            self,
            p=4,
            q=1,
            plot=False,
            ftol=1e-4,
            Mtol=1.1,
            shapetol=None,
            niters=5000,
            verbose=False,
            cutoff=1e-6
        ):
        '''
        Variational spectral condensation à la Hirshman, Meier 1985. 
        '''
        M = self.mpol
        N = self.ntor
        nfp = self.nfp

        m_arr = np.arange(0, M+1)
        n_arr = np.arange(-N, N+1)
        ntheta = 32
        nzeta = 32
        t_1d = np.linspace(0,2*np.pi,num=ntheta)
        z_1d = np.linspace(0,2*np.pi,num=nzeta)# + 2*np.pi/nfp
        z_grid, t_grid = np.meshgrid(z_1d,t_1d)

        # fourier basis functions
        cosnmtz = np.array(
            [[np.cos(m*t_grid - n*z_grid*nfp) for m in range(0, M+1)] for n in range(-N, N+1)],
        )
        sinnmtz = np.array(
            [[np.sin(m*t_grid - n*z_grid*nfp) for m in range(0, M+1)] for n in range(-N, N+1)],
        )

        def x_t_y_t(
                rbc,
                zbs
        ):
            '''
            Computes Fourier representation of derivative of the surface wrt. theta
            '''
            # coefficients of derivatives of the r, z wrt. theta
            rnm_t = np.einsum('nm,m->nm', rbc, m_arr)
            znm_t = np.einsum('nm,m->nm', zbs, m_arr)

            x_t = np.einsum('nm,nmtz->tz', rnm_t, -sinnmtz)
            y_t = np.einsum('nm,nmtz->tz', znm_t, cosnmtz)

            return x_t, y_t

        def hwM_pq(
                rbc,
                zbs,
                p=p,
                q=q
        ):
            num = np.einsum('m,nm->nm', m_arr**(p+q), rbc**2 + zbs**2)
            denom = np.einsum('m,nm->nm', m_arr**(p), rbc**2 + zbs**2)
            return np.sum(num)/np.sum(denom)
        
        def hw_I_callable(
                rbc,
                zbs,
        ):
            M_pq = hwM_pq(rbc, zbs)
            f_m = (m_arr**p) * (m_arr**q - M_pq)
            
            xfm = np.einsum('nm,m->nm', rbc, f_m)
            yfm = np.einsum('nm,m->nm', zbs, f_m)

            X = np.einsum('nmtz,nm->tz', cosnmtz[:,1:,:,:], xfm[:,1:])
            Y = np.einsum('nmtz,nm->tz', sinnmtz[:,1:,:,:], yfm[:,1:])

            x_t, y_t = x_t_y_t(rbc, zbs)

            return X*x_t + Y*y_t

        def naive_spec_cond(
                surf,
                verbose=verbose
            ):
            '''
            iterate R_mn, Z_mn as X_mn,[n+1] = X_mn,[n] + aδx_mn 
            Includes a line-search for the step size a
            '''
            newsurf = surf.copy()
            rbc = newsurf.rc.T
            zbs = newsurf.zs.T

            dtdz = ((2*np.pi)/(nzeta-1)) * ((2*np.pi)/(ntheta-1))
            I = hw_I_callable(rbc, zbs)
            integral_I2 = np.einsum('tz,tz',I**2,dtdz*np.ones_like(I))
            niter = 0
            flast = hwM_pq(rbc, zbs)
            df = 1
            if verbose:
                print(f'∫I^2(t,z)dtdz: {integral_I2}')
                print(f'Initial M: {flast}')

            success=False

            while success == False:
                niter += 1
                x_t, y_t = x_t_y_t(rbc, zbs)
                x_mn_integrand = np.einsum('nmtz,tz->nmtz', cosnmtz,-I*x_t)
                y_mn_integrand = np.einsum('nmtz,tz->nmtz', sinnmtz,-I*y_t)
                drbc = np.einsum('nmtz,tz->nm',x_mn_integrand,dtdz*np.ones_like(I))
                dzbs = np.einsum('nmtz,tz->nm',y_mn_integrand,dtdz*np.ones_like(I))

                drbc[:N,0] = 0
                dzbs[:N,0] = 0

                def f(alpha, rbc, zbs, drbc, dzbs):
                    _rbc = np.copy(rbc)
                    _zbs = np.copy(zbs)
                    _rbc += alpha * drbc
                    _zbs += alpha * dzbs
                    return hwM_pq(_rbc, _zbs)

                res = minimize_scalar(f, bracket = (-1e-4, 1e-4), args = (rbc, zbs, drbc, dzbs), method='golden', options={'disp':False})
                alpha = res.x

                rbc += alpha * drbc
                zbs += alpha * dzbs

                newsurf.rc = rbc.T
                newsurf.zs = zbs.T

                I = hw_I_callable(rbc, zbs)
                integral_I2 = np.einsum('tz,tz',I**2,dtdz*np.ones_like(I))
                fnew = hwM_pq(rbc, zbs)
                df = np.abs(fnew - flast)
                flast = fnew 

                if hwM_pq(rbc, zbs) <= Mtol:
                    success = True
                    message = 'Terminated due to sufficiently low M'
                if niter > niters:#5000:
                    success = True
                    message = 'Maxiter reached'
                if df <= ftol:#1e-3:
                    success = True
                    message = 'dM < ftol reached'
                if shapetol is not None:
                    shape_error_arr = jaccard_index(
                                        newsurf, 
                                        surf
                                    )
                    shape_error = (np.average(np.abs(shape_error_arr)))
                    if shape_error >= shapetol:
                        success = True
                        message = f'Shape error {shapetol} reached'

            if verbose:
                print(message)
                print(f'Final ∫I^2(t,z)dtdz: {integral_I2}')
                print(f'Final M: {hwM_pq(rbc, zbs)}')

            return newsurf, rbc, zbs

        surf_to_return, rbc_f, zbs_f = naive_spec_cond(self)

        rbc_f *= (np.abs(rbc_f) > cutoff)
        zbs_f *= (np.abs(zbs_f) > cutoff)

        if plot:
            def boundary_poincare_plot(
                    surf,
                    phi,
                    ntheta=200,
                    ):
                M, N, nfp = surf.mpol, surf.ntor, surf.nfp
                rbc, zbs = surf.rc.T, surf.zs.T
                xn = np.arange(-N,N+1,1)
                xm = np.arange(0,M+1,1)

                ntheta = 200
                theta = np.linspace(0,2*np.pi,num=ntheta)

                R = np.zeros((ntheta,1))
                Z = np.zeros((ntheta,1))

                for i in range(rbc.shape[0]):
                    for j in range(rbc.shape[1]):
                        if rbc[i,j] !=0 or zbs[i,j] != 0:
                            angle = xm[j]*theta - xn[i]*phi*nfp
                            R = R + rbc[i,j]*np.cos(angle)#/(np.abs(i) + np.abs(j))
                            Z = Z + zbs[i,j]*np.sin(angle)#/(np.abs(i) + np.abs(j))
                return R.flatten(), Z.flatten()
        
            fig, ax = plt.subplots()
            pwr_init = np.einsum('nm->m', self.rc.T**2 + self.zs.T**2)
            pwr_final = np.einsum('nm->m', rbc_f**2 + zbs_f**2)
            ax.semilogy(m_arr, pwr_init, label = 'equal arc length')
            ax.semilogy(m_arr, pwr_final, label ='condensed')
            ax.set_xlabel('m')
            ax.set_ylabel('$\sum_n R_{mn}^2 + Z_{mn}^2$')
            ax.legend()

            fig1, ax1 = plt.subplots()
            phi_array = np.linspace(np.pi/2, np.pi, 5)
            for phi in phi_array[:-1]:
                R_init, Z_init = boundary_poincare_plot(self, phi)
                R_cond, Z_cond = boundary_poincare_plot(surf_to_return, phi)
                ax1.plot(R_init, Z_init, color='r')
                ax1.plot(R_cond, Z_cond, color='b')
            R_init, Z_init = boundary_poincare_plot(self, phi_array[-1])
            R_cond, Z_cond = boundary_poincare_plot(surf_to_return, phi_array[-1])
            ax1.plot(R_init, Z_init, color='r', label='initial')
            ax1.plot(R_cond, Z_cond, color='b', label='condensed')
            ax1.legend()
            ax1.set_aspect('equal')

        surf_to_return = self.copy()
        surf_to_return.rc, surf_to_return.zs = rbc_f.T, zbs_f.T 

        return surf_to_return

def plot_spectral_condensation(surf1, surf2, data, show=True):
    """Plot results from :func:`~simsopt.geo.SurfaceRZFourier.condense_spectrum`.

    Three figures are generated.

    Figure 1 shows λ as a function of θ₁ at various ϕ values (left), and the
    Fourier amplitudes of λ are plotted on the right.

    Figure 2 shows the m- and n-dependence of rc and zs with respect to θ₁ and
    θ₂ in two ways: heatmaps on the left, and a scatter plot on the right.

    Figure 3 shows eight cross-sections of the surface, including points that
    are uniformly spaced with respect to θ₁ and θ₂.

    Parameters
    ----------
    surf1 : SurfaceRZFourier
        The original surface before spectral condensation.
    surf2 : SurfaceRZFourier
        The condensed surface after spectral condensation.
    data : dict
        The data dictionary returned by :func:`~simsopt.geo.SurfaceRZFourier.condense_spectrum`.
    show : bool, optional
        Whether to call matplotlib's ``show()`` function. Default is True.    

    Returns
    -------
    fig1, fig2, fig3
        Matplotlib handles for the three figures
    """
    assert surf1.nfp == surf2.nfp
    assert surf1.mpol == surf2.mpol
    assert surf1.ntor == surf2.ntor
    n_theta = data["n_theta"]
    n_phi = data["n_phi"]
    minor_radius = data["minor_radius"]
    nfp = surf1.nfp
    title_str = f"n_theta: {n_theta}, n_phi: {n_phi}, power: {data['power']}, method: {data['method']}, maxiter: {data['maxiter']}\n"

    n_theta_points_for_plot = 20
    decimate = 6

    quadpoints_theta_for_plotting = np.linspace(0, 1, n_theta_points_for_plot * decimate + 1)
    quadpoints_phi_for_plotting = np.linspace(0, 1 / nfp, 8, endpoint=False)
    surf_theta1_for_plotting = SurfaceRZFourier(
        mpol=surf1.mpol,
        ntor=surf1.ntor,
        nfp=nfp,
        quadpoints_theta=quadpoints_theta_for_plotting,
        quadpoints_phi=quadpoints_phi_for_plotting,
    )
    surf_theta1_for_plotting.local_full_x = surf1.local_full_x
    surf_theta2_for_plotting = SurfaceRZFourier(
        mpol=surf1.mpol,
        ntor=surf1.ntor,
        nfp=nfp,
        quadpoints_theta=quadpoints_theta_for_plotting,
        quadpoints_phi=quadpoints_phi_for_plotting,
    )
    surf_theta2_for_plotting.local_full_x = surf2.local_full_x

    figsize = (14.5, 8.1)
    fig1 = plt.figure(figsize=figsize)

    n_rows = 1
    n_cols = 2

    plt.subplot(n_rows, n_cols, 1)
    cmap = plt.get_cmap("jet")
    for j_phi in range(n_phi):
        color = cmap(float(j_phi) / n_phi)
        plt.plot(
            data["theta1_2d"][j_phi, :],
            (data["theta_optimized"] - data["theta1_1d"]).reshape((n_phi, n_theta))[j_phi, :],
            '.-',
            color=color,
        )

    plt.xlabel('theta1')
    plt.ylabel('lambda')

    plt.subplot(n_rows, n_cols, 2)
    plt.semilogy(
        np.sqrt(data["m"]**2 + data["n"]**2),
        np.abs(data["lambda_mn"] * data["x_scale"]),
        '.g',
    )
    plt.xlabel('sqrt(m^2 + n^2)')
    plt.ylabel('Mode amplitudes of lambda')
    plt.ylim(1e-16, 2e1)

    plt.tight_layout()

    # Create a layout with the left half split into a 2x2 grid and a
    # single subplot occupying the entire right half.
    fig = plt.figure(figsize=figsize)
    fig2 = fig
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 2])

    # Left half: 2x2 small subplots
    axes_small = [
        fig.add_subplot(gs[0, 0], aspect='equal'),
        fig.add_subplot(gs[0, 1], aspect='equal'),
        fig.add_subplot(gs[1, 0], aspect='equal'),
        fig.add_subplot(gs[1, 1], aspect='equal'),
    ]

    colors = ["b", "r"]

    for j_surf in range(2):
        if j_surf == 0:
            surf_to_plot = surf_theta1_for_plotting
            short_name = "theta1"
        else:
            surf_to_plot = surf_theta2_for_plotting
            short_name = "theta2"

        for j_rz in range(2):
            if j_rz == 0:
                data_to_plot = surf_to_plot.rc
                data_name = 'Rmnc'
            else:
                data_to_plot = surf_to_plot.zs
                data_name = 'Zmns'

            ax = axes_small[j_surf + j_rz * 2]
            extent = (-surf_to_plot.ntor - 0.5, surf_to_plot.ntor + 0.5, surf_to_plot.mpol + 0.5, -0.5)
            im = ax.imshow(np.abs(data_to_plot / minor_radius), extent=extent, norm=mpl_colors.LogNorm(vmin=1e-6, vmax=10))
            fig.colorbar(im, ax=ax)
            ax.set_title(data_name + " / minor_radius, " + short_name)
            ax.set_xlabel('n / nfp')
            ax.set_ylabel('m')

    # Right half: one large subplot spanning both rows
    ax_big = fig.add_subplot(gs[:, -1])

    plt.semilogy(
        np.sqrt(surf_theta1_for_plotting.m**2 + surf_theta1_for_plotting.n**2),
        np.abs(surf_theta1_for_plotting.x / minor_radius),
        '+',
        color=colors[0],
        label='theta1'
    )
    plt.semilogy(
        np.sqrt(surf_theta2_for_plotting.m**2 + surf_theta2_for_plotting.n**2),
        np.abs(surf_theta2_for_plotting.x / minor_radius),
        'x',
        color=colors[1],
        label='theta2'
    )
    plt.legend(loc=0)
    ax_big.set_xlabel('sqrt(m^2 + n^2)')
    ax_big.set_ylabel('(rc or zs) / minor radius')
    ax_big.set_ylim(1e-16, 2e1)

    plt.suptitle(title_str, fontsize=12)
    plt.tight_layout()

    ##########################################################################################################
    ##########################################################################################################

    n_rows = 2
    n_cols = 4
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=figsize, subplot_kw={'aspect': 'equal'})

    axes = axes.flatten()

    for j_phi in range(8):
        for j_surf in range(2):
            if j_surf == 0:
                surf_to_plot = surf_theta1_for_plotting
                linespec = '-'
                marker = '+'
                theta0_marker = 's'
            else:
                surf_to_plot = surf_theta2_for_plotting
                linespec = ':'
                marker = 'x'
                theta0_marker = 'o'

            gamma = surf_to_plot.gamma()
            R = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
            Z = gamma[:, :, 2]
            color = colors[j_surf]
            axes[j_phi].plot(R[j_phi, :], Z[j_phi, :], linespec, color=color)
            axes[j_phi].plot(R[j_phi, 0], Z[j_phi, 0], theta0_marker, color=color)
            axes[j_phi].plot(R[j_phi, ::decimate], Z[j_phi, ::decimate], marker, color=color, label=f"theta{j_surf+1}")

        axes[j_phi].set_title(f"phi = {j_phi/(8*nfp):.3f} * 2pi")
        axes[j_phi].set_xlabel("R")
        axes[j_phi].set_ylabel("Z")

    axes[0].legend(loc=0)

    plt.suptitle(title_str, fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()

    return fig1, fig2, fig3

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

    def __init__(self, mpol, ntor, nfp, r_shift=1.0, a_scale=1.0, **kwargs):
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.r_shift = r_shift
        self.a_scale = a_scale
        ndofs = 1 + 2 * (ntor + mpol * (2 * ntor + 1))
        if "dofs" not in kwargs:
            if "x0" not in kwargs:
                kwargs["x0"] = np.zeros(ndofs)
            else:
                assert (len(kwargs["x0"]) == ndofs)
            if "names" not in kwargs:
                kwargs["names"] = self._make_names()
            else:
                assert (len(kwargs["names"]) == ndofs)
        else:
            assert (len(kwargs["dofs"]) == ndofs)
        super().__init__(**kwargs)

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
        # Map to Fourier space and return a surface with changed resolution
        surf2 = self.to_RZFourier().change_resolution(mpol=mpol, ntor=ntor)
        # Map from Fourier space back to real space:
        surf3 = SurfaceRZPseudospectral.from_RZFourier(surf2,
                                                       r_shift=self.r_shift,
                                                       a_scale=self.a_scale)
        return surf3
