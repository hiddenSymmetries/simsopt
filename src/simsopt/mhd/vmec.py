# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
from scipy.integrate import quad

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

try:
    import vmec
except ImportError as e:
    vmec = None
    logger.debug(str(e))

from .._core.optimizable import Optimizable
from .._core.util import Struct, ObjectiveFailure  # noqa: F401
from ..geo.surfacerzfourier import SurfaceRZFourier
from .profiles import ProfilePolynomial, ProfileSpline
# Re-imported here so that e.g. ``from simsopt.mhd.vmec import array_to_namelist``
# keeps working:
from .vmec_solver import (Vmec2000Solver, load_wout_file, to_namelist_bool,
                          array_to_namelist, restart_flag, readin_flag,
                          timestep_flag, output_flag, cleanup_flag,
                          reset_jacdt_flag)  # noqa: F401

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None

__all__ = ["Vmec", "SurfaceRZFourierProtocol", "ProfileProtocol",
           "VmecSolverProtocol", "VmecBoundary", "VmecProfile"]


@runtime_checkable
class SurfaceRZFourierProtocol(Protocol):
    """
    Boundary surface passed from simsopt to a Vmec solver.

    Fourier modes are sparse dicts keyed by the physical mode numbers
    ``(m, n)``, so no solver's index-offset convention leaks into the
    interface. Only nonzero modes need be present, and ``rbs``/``zbc``
    are empty when ``stellsym`` is True.

    ``mpol``/``ntor`` truncate the boundary representation and size the
    surface's dof vector; they are not the solver's internal resolution,
    which is a solver setting.
    """
    nfp: int
    stellsym: bool
    mpol: int
    ntor: int
    rbc: dict  # {(m, n): value}
    zbs: dict
    rbs: dict
    zbc: dict


@runtime_checkable
class ProfileProtocol(Protocol):
    """
    Radial profile (pressure, current or iota) passed from simsopt to a
    Vmec solver, together with its VMEC parametrization.

    - ``power_series`` (also ``gauss_trunc``, ``two_power``): ``coeffs``
      are the polynomial coefficients (``am``/``ac``/``ai``), ``knots``
      is None.
    - ``cubic_spline``, ``akima_spline``, ``line_segment``: ``coeffs``
      are the spline values (``*_aux_f``) at ``knots`` (``*_aux_s``).

    Current profiles may append ``_i`` or ``_ip``, passed through
    verbatim.
    """
    profile_type: str
    coeffs: Any
    knots: Any


@runtime_checkable
class VmecSolverProtocol(Protocol):
    """
    Interface a VMEC backend must satisfy to be driven by :obj:`Vmec`.

    Covers the physics only: the boundary, the profiles, the three
    scalar dofs and the converged output. Solver settings (``ns_array``,
    ``ftol_array``, ``delt``, ``mgrid_file``, ...) are reached through
    :attr:`indata`, whose type is chosen by the backend.

    ``phiedge``, ``curtor`` and ``pres_scale`` must read and write
    straight through to :attr:`indata` rather than being cached, so that
    writing e.g. ``vmec.indata.curtor`` directly still takes effect.

    Conformance is structural, so implementations need not import
    simsopt. Note that :func:`isinstance` checks only that the names are
    present, not their types.
    """

    boundary: SurfaceRZFourierProtocol
    pressure: Optional[ProfileProtocol]
    current: Optional[ProfileProtocol]
    iota: Optional[ProfileProtocol]

    phiedge: float
    curtor: float
    pres_scale: float

    indata: Any
    wout: Any
    output_file: Any

    def solve(self) -> None:
        """Run the solver and populate :attr:`wout`."""
        ...


@dataclass
class VmecBoundary:
    """
    Concrete :obj:`SurfaceRZFourierProtocol`, built by :obj:`Vmec` from
    its boundary surface.

    ``surface`` additionally carries the originating
    :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier` so backends
    that write fortran namelists can reuse its ``get_nml()``. It is not
    part of the protocol; other solvers should ignore it.
    """
    nfp: int = 1
    stellsym: bool = True
    mpol: int = 1
    ntor: int = 0
    rbc: dict = field(default_factory=dict)
    zbs: dict = field(default_factory=dict)
    rbs: dict = field(default_factory=dict)
    zbc: dict = field(default_factory=dict)
    surface: Any = None


@dataclass
class VmecProfile:
    """Concrete :obj:`ProfileProtocol`."""
    profile_type: str = "power_series"
    coeffs: Any = None
    knots: Any = None


#: Fields downstream simsopt code reads from ``Vmec.wout``. A backend's
#: output object must provide these using wout file conventions: 2D
#: fourier arrays indexed ``[mode, radius]``, and half-grid quantities
#: carrying a dummy entry at index 0.
REQUIRED_WOUT_FIELDS = (
    # Scalars and metadata
    'aspect', 'Aminor_p', 'Rmajor_p', 'betatotal', 'ctor', 'ier_flag',
    'lasym', 'mnmax', 'mnmax_nyq', 'mpol', 'nfp', 'ns', 'ntor', 'signgs',
    'volavgB', 'volume_p', 'fsqr', 'fsql', 'fsqz',
    # Profile type tags
    'pmass_type', 'pcurr_type', 'piota_type',
    # Mode numbers
    'xm', 'xn', 'xm_nyq', 'xn_nyq',
    # Radial profiles
    'iotaf', 'iotas', 'pres', 'phi', 'chi', 'vp', 'buco', 'bvco',
    'jcurv', 'jdotb',
    # Fourier arrays, stellarator-symmetric
    'rmnc', 'zmns', 'lmns', 'gmnc', 'bmnc', 'bsupumnc', 'bsupvmnc',
    'bsubumnc', 'bsubvmnc', 'bsubsmns',
)

#: Additional ``wout`` fields required only when ``lasym`` is True.
REQUIRED_WOUT_FIELDS_ASYM = (
    'rmns', 'zmnc', 'lmnc', 'gmns', 'bmns', 'bsupumns', 'bsupvmns',
    'bsubumns', 'bsubvmns', 'bsubsmnc',
)


class Vmec(Optimizable):
    r"""
    This class represents the VMEC equilibrium code.

    You can initialize this class either from a VMEC
    ``input.<extension>`` file or from a ``wout_<extension>.nc`` output
    file. If neither is provided, a default input file is used. When
    this class is initialized from an input file, it is possible to
    modify the input parameters and run the VMEC code. When this class
    is initialized from a ``wout`` file, all the data from the
    ``wout`` file is available in memory but the VMEC code cannot be
    re-run, since some of the input data (e.g. radial multigrid
    parameters) is not available in the wout file.

    The input parameters to VMEC are all accessible as attributes of
    the ``indata`` attribute. For example, if ``vmec`` is an instance
    of ``Vmec``, then you can read or write the input resolution
    parameters using ``vmec.indata.mpol``, ``vmec.indata.ntor``,
    ``vmec.indata.ns_array``, etc. However, the boundary surface is
    different: ``rbc``, ``rbs``, ``zbc``, and ``zbs`` from the
    ``indata`` attribute are always ignored, and these arrays are
    instead taken from the simsopt surface object associated to the
    ``boundary`` attribute. If ``boundary`` is a surface based on some
    other representation than VMEC's Fourier representation, the
    surface will automatically be converted to VMEC's representation
    (:obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`) before
    each run of VMEC. You can replace ``boundary`` with a new surface
    object, of any type that implements the conversion function
    ``to_RZFourier()``.

    VMEC is run either when the :meth:`run()` function is called, or when
    any of the output functions like :meth:`aspect()` or :meth:`iota_axis()`
    are called.

    A caching mechanism is implemented, using the attribute
    ``need_to_run_code``. Whenever VMEC is run, or if the class is
    initialized from a ``wout`` file, this attribute is set to
    ``False``. Subsequent calls to :meth:`run()` or output functions
    like :meth:`aspect()` will not actually run VMEC again, until
    ``need_to_run_code`` is changed to ``True``. The attribute
    ``need_to_run_code`` is automatically set to ``True`` whenever the
    state vector ``.x`` is changed, and when dofs of the ``boundary``
    are changed. However, ``need_to_run_code`` is not automatically
    set to ``True`` when entries of ``indata`` are modified.

    Once VMEC has run at least once, or if the class is initialized
    from a ``wout`` file, all of the quantities in the ``wout`` output
    file are available as attributes of the ``wout`` attribute.  For
    example, if ``vmec`` is an instance of ``Vmec``, then the flux
    surface shapes can be obtained from ``vmec.wout.rmnc`` and
    ``vmec.wout.zmns``.

    Since the underlying fortran implementation of VMEC uses global
    module variables, it is not possible to have more than one python
    Vmec object with different parameters; changing the parameters of
    one would change the parameters of the other.

    An instance of this class owns three optimizable degrees of
    freedom: ``phiedge``, ``curtor``, and ``pres_scale``. The optimizable
    degrees of freedom associated with the boundary surface are owned
    by that surface object.

    To run VMEC, two input profiles must be specified: pressure and
    either iota or toroidal current.  Each of these profiles can be
    specified in several ways. One way is to specify the profile in
    the input file used to initialize the ``Vmec`` object. For
    instance, the pressure profile is determined by the variables
    ``pmass_type``, ``am``, ``am_aux_s``, and ``am_aux_f``. You can
    also modify these variables from python via the ``indata``
    attribute, e.g. ``vmec.indata.am = [1.0e5, -1.0e5]``. Another
    option is to assign a :obj:`simsopt.mhd.profiles.Profile` object
    to the attributes ``pressure_profile``, ``current_profile``, or
    ``iota_profile``. This approach allows for the profiles to be
    optimized, and it allows you to use profile shapes defined in
    python that are not available in the fortran VMEC code. To explain
    this approach we focus here on the pressure profile; the iota and
    current profiles are analogous. If the ``pressure_profile``
    attribute of a ``Vmec`` object is ``None`` (the default), then a
    simsopt :obj:`~simsopt.mhd.profiles.Profile` object is not used,
    and instead the settings from ``Vmec.indata`` (initialized from
    the input file) are used. If a
    :obj:`~simsopt.mhd.profiles.Profile` object is assigned to the
    ``pressure_profile`` attribute, then an :ref:`edge in the
    dependency graph <dependecies>` is introduced, so the ``Vmec``
    object then depends on the dofs of the
    :obj:`~simsopt.mhd.profiles.Profile` object. Whenever VMEC is run,
    the simsopt :obj:`~simsopt.mhd.profiles.Profile` is converted to
    either a polynomial (power series) or cubic spline in the
    normalized toroidal flux :math:`s`, depending on whether
    ``indata.pmass_type`` is ``"power_series"`` or
    ``"cubic_spline"``. (The current profile is different in that
    either ``"cubic_spline_ip"`` or ``"cubic_spline_i"`` is specified
    instead of ``"cubic_spline"``, where ``cubic_spline_ip`` sets I'(s) while ``cubic_spline_i`` sets I(s).) The number of terms in the power
    series or number of spline nodes is determined by the attributes
    ``n_pressure``, ``n_current``, and ``n_iota``.  If a cubic spline
    is used, the spline nodes are uniformly spaced from :math:`s=0` to
    1. Note that the choice of whether a polynomial or spline is used
    for the VMEC calculation is independent of the subclass of
    :obj:`~simsopt.mhd.profiles.Profile` used. Also, whether the iota
    or current profile is used is always determined by the
    ``indata.ncurr`` attribute: 0 for iota, 1 for current. Example::

        from sismopt.mhd.profiles import ProfilePolynomial, ProfileSpline, ProfilePressure, ProfileScaled
        from simsopt.util.constants import ELEMENTARY_CHARGE

        ne = ProfilePolynomial(1.0e20 * np.array([1, 0, 0, 0, -0.9]))
        Te = ProfilePolynomial(8.0e3 * np.array([1, -0.9]))
        Ti = ProfileSpline([0, 0.5, 0.8, 1], 7.0e3 * np.array([1, 0.9, 0.8, 0.1]))
        ni = ne
        pressure = ProfilePressure(ne, Te, ni, Ti)  # p = ne * Te + ni * Ti
        pressure_Pa = ProfileScaled(pressure, ELEMENTARY_CHARGE)  # Te and Ti profiles were in eV, so convert to SI here.
        vmec = Vmec(filename)
        vmec.pressure_profile = pressure_Pa
        vmec.indata.pmass_type = "cubic_spline"
        vmec.n_pressure = 8  # Use 8 spline nodes

    When a current profile is used, the ``VMEC`` object automatically updates ``curtor`` so that the total toroidal current I(s=1) matches that of the specified profile.

    When VMEC is run multiple times, the default behavior is that all
    ``wout`` output files will be deleted except for the first and
    most recent iteration on worker group 0. If you wish to keep all
    the ``wout`` files, you can set ``keep_all_files = True``. If you
    want to save the ``wout`` file for a certain intermediate
    iteration, you can set the ``files_to_delete`` attribute to ``[]``
    after that run of VMEC.

    Args:
        filename: Name of a VMEC ``input.<extension>`` file or ``wout_<extension>.nc``
          output file to use for loading the
          initial parameters. If ``None``, default parameters will be used.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which
          the worker groups will be used for VMEC calculations. If ``None``,
          each MPI process will run VMEC independently.
        keep_all_files: If ``False``, all ``wout`` output files will be deleted
          except for the first and most recent ones from worker group 0. If
          ``True``, all ``wout`` files will be kept.
        verbose: Whether to print to stdout when running vmec.

    Attributes:
        iter: Number of times VMEC has run.
        s_full_grid: The "full" grid in the radial coordinate s (normalized
          toroidal flux), including points at s=0 and s=1. Used for the output
          arrays and ``zmns``.
        s_half_grid: The "half" grid in the radial coordinate s, used for
          ``bmnc``, ``lmns``, and other output arrays. In contrast to
          wout files, this array has only ns-1 entries, so there is no
          leading 0.
        ds: The spacing between grid points for the radial coordinate s.
    """

    def __init__(self,
                 filename: Optional[str] = None,
                 mpi: Optional[MpiPartition] = None,
                 keep_all_files: bool = False,
                 verbose: bool = True,
                 ntheta=50,
                 nphi=50,
                 range_surface='full torus',
                 solver=None):

        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'input.default')
            logger.info(f"Initializing a VMEC object from defaults in {filename}")

        basename = os.path.basename(filename)
        if basename[:5] == 'input':
            logger.info(f"Initializing a VMEC object from input file: {filename}")
            self.runnable = True
        elif basename[:4] == 'wout':
            logger.info(f"Initializing a VMEC object from wout file: {filename}")
            self.runnable = False
        else:
            raise ValueError('Invalid filename')

        self._solver = None
        self._wout = Struct()
        self._output_file = None
        self._verbose = verbose

        # Get MPI communicator:
        if (mpi is None and MPI is not None):
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi

        self._pressure_profile = None
        self._current_profile = None
        self._iota_profile = None
        self.n_pressure = 10
        self.n_current = 10
        self.n_iota = 10

        if self.runnable:
            if MPI is None:
                raise RuntimeError("mpi4py needs to be installed for running VMEC")
            if solver is None:
                solver = Vmec2000Solver
            if isinstance(solver, VmecSolverProtocol):
                self._solver = solver
            else:
                self._solver = solver(filename, self.mpi,
                                      keep_all_files=keep_all_files,
                                      verbose=verbose)
            if not isinstance(self._solver, VmecSolverProtocol):
                raise TypeError(
                    f"{type(self._solver).__name__} does not satisfy "
                    "VmecSolverProtocol. It must provide: boundary, pressure, "
                    "current, iota, phiedge, curtor, pres_scale, indata, wout, "
                    "output_file, solve.")

            # A vmec object has mpol and ntor attributes independent of
            # the boundary. The boundary surface object is initialized
            # with mpol and ntor values that match those of the vmec
            # object, but the mpol/ntor values of either the vmec object
            # or the boundary surface object can be changed independently
            # by the user.
            solver_boundary = self._solver.get_boundary()
            self._boundary = SurfaceRZFourier.from_nphi_ntheta(nfp=solver_boundary.nfp,
                                                               stellsym=solver_boundary.stellsym,
                                                               mpol=solver_boundary.mpol,
                                                               ntor=solver_boundary.ntor,
                                                               ntheta=ntheta,
                                                               nphi=nphi,
                                                               range=range_surface)

            # Transfer boundary shape data from the solver to the ParameterArray:
            ntor = solver_boundary.ntor
            for (m, n), value in solver_boundary.rbc.items():
                self._boundary.rc[m, n + ntor] = value
            for (m, n), value in solver_boundary.zbs.items():
                self._boundary.zs[m, n + ntor] = value
            for (m, n), value in solver_boundary.rbs.items():
                self._boundary.rs[m, n + ntor] = value
            for (m, n), value in solver_boundary.zbc.items():
                self._boundary.zc[m, n + ntor] = value
            self._boundary.local_full_x = self._boundary.get_dofs()

            self.need_to_run_code = True
        else:
            # Initialized from a wout file, so not runnable.
            self._boundary = SurfaceRZFourier.from_wout(filename, nphi=nphi, ntheta=ntheta, range=range_surface)
            self.output_file = filename
            self.load_wout()

        # Handle a few variables that are not Parameters:
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['phiedge', 'curtor', 'pres_scale']
        super().__init__(x0=x0, fixed=fixed, names=names,
                         depends_on=[self._boundary],
                         external_dof_setter=Vmec.set_dofs)

        if not self.runnable:
            # This next line must come after Optimizable.__init__
            # since that calls recompute_bell()
            self.need_to_run_code = False

    def _solver_attribute(self, name):
        """ Return the solver, or raise if this object was created from a wout file. """
        if self._solver is None:
            raise AttributeError(f"'Vmec' object has no attribute '{name}', because it "
                                 "was initialized from a wout file.")
        return self._solver

    @property
    def indata(self):
        """ The input parameters of the solver. """
        if self._solver is None:
            raise RuntimeError('Cannot access indata for a Vmec object that was initialized from a wout file.')
        return self._solver.indata

    @property
    def wout(self):
        """ The data from the VMEC ``wout`` output file. """
        return self._wout if self._solver is None else self._solver.wout

    @wout.setter
    def wout(self, wout):
        if self._solver is None:
            self._wout = wout
        else:
            self._solver.wout = wout

    @property
    def output_file(self):
        """ Name of the ``wout`` file most recently loaded or written. """
        return self._output_file if self._solver is None else self._solver.output_file

    @output_file.setter
    def output_file(self, output_file):
        if self._solver is None:
            self._output_file = output_file
        else:
            self._solver.output_file = output_file

    @property
    def verbose(self):
        """ Whether to print to stdout when running vmec. """
        return self._verbose if self._solver is None else self._solver.verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        if self._solver is not None:
            self._solver.verbose = verbose

    @property
    def input_file(self):
        """ Name of the VMEC input file this object was initialized from. """
        return self._solver_attribute("input_file").input_file

    @input_file.setter
    def input_file(self, input_file):
        self._solver_attribute("input_file").input_file = input_file

    @property
    def iter(self):
        """ Number of times VMEC has run. """
        return self._solver_attribute("iter").iter

    @iter.setter
    def iter(self, iter):
        self._solver_attribute("iter").iter = iter

    @property
    def keep_all_files(self):
        """ If ``False``, all but the first and most recent ``wout`` files are deleted. """
        return self._solver_attribute("keep_all_files").keep_all_files

    @keep_all_files.setter
    def keep_all_files(self, keep_all_files):
        self._solver_attribute("keep_all_files").keep_all_files = keep_all_files

    @property
    def files_to_delete(self):
        """ Files that will be deleted after the next run of VMEC. """
        return self._solver_attribute("files_to_delete").files_to_delete

    @files_to_delete.setter
    def files_to_delete(self, files_to_delete):
        self._solver_attribute("files_to_delete").files_to_delete = files_to_delete

    @property
    def free_boundary(self):
        """ Whether VMEC is run in free-boundary mode. """
        return self._solver_attribute("free_boundary").free_boundary

    @free_boundary.setter
    def free_boundary(self, free_boundary):
        self._solver_attribute("free_boundary").free_boundary = free_boundary

    @property
    def fcomm(self):
        """ Fortran handle for the MPI communicator of the worker group. """
        return self._solver_attribute("fcomm").fcomm

    @fcomm.setter
    def fcomm(self, fcomm):
        self._solver_attribute("fcomm").fcomm = fcomm

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary is not self._boundary:
            logging.debug('Replacing surface in boundary setter')
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)
            self.need_to_run_code = True

    @property
    def pressure_profile(self):
        return self._pressure_profile

    @pressure_profile.setter
    def pressure_profile(self, pressure_profile):
        if pressure_profile is not self._pressure_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._pressure_profile is not None:
                self.remove_parent(self._pressure_profile)
            self._pressure_profile = pressure_profile
            if pressure_profile is not None:
                self.append_parent(pressure_profile)
                self.need_to_run_code = True

    @property
    def current_profile(self):
        return self._current_profile

    @current_profile.setter
    def current_profile(self, current_profile):
        if current_profile is not self._current_profile:
            logging.debug('Replacing current_profile in setter')
            if self._current_profile is not None:
                self.remove_parent(self._current_profile)
            self._current_profile = current_profile
            if current_profile is not None:
                self.append_parent(current_profile)
                self.need_to_run_code = True

    @property
    def iota_profile(self):
        return self._iota_profile

    @iota_profile.setter
    def iota_profile(self, iota_profile):
        if iota_profile is not self._iota_profile:
            logging.debug('Replacing iota_profile in setter')
            if self._iota_profile is not None:
                self.remove_parent(self._iota_profile)
            self._iota_profile = iota_profile
            if iota_profile is not None:
                self.append_parent(iota_profile)
                self.need_to_run_code = True

    def get_dofs(self):
        if not self.runnable:
            # Use default values from vmec_input
            return np.array([1.0, 0.0, 1.0])
        else:
            return np.array([self._solver.phiedge, self._solver.curtor,
                             self._solver.pres_scale])

    def set_dofs(self, x):
        if self.runnable:
            self.need_to_run_code = True
            self._solver.phiedge = x[0]
            self._solver.curtor = x[1]
            self._solver.pres_scale = x[2]

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def set_profile(self, longname, shortname, letter):
        """
        This function is used to set the pressure, current, and/or iota
        profiles. The simsopt :obj:`~simsopt.mhd.profiles.Profile` object
        is converted to VMEC's parametrization and transferred to the
        solver.

        Args:
            longname: ``"pressure"``, ``"current"``, or ``"iota"``.
            shortname: ``"mass"``, ``"curr"``, or ``"iota"``, as in VMEC's
              ``pmass_type``, ``pcurr_type``, and ``piota_type``.
            letter: ``"m"``, ``"c"``, or ``"i"``, as in VMEC's ``am``,
              ``ac``, and ``ai``.

        Returns:
            The :obj:`VmecProfile` handed to the solver, or ``None`` if no
            simsopt ``Profile`` object is attached.
        """
        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return None

        n = self.__getattribute__("n_" + longname)
        profile_type = self.indata.__getattribute__("p" + shortname + "_type")
        if isinstance(profile_type, bytes):
            profile_type = profile_type.decode()
        profile_type = profile_type.lower().strip()

        vmec_profile = self._to_vmec_profile(profile, n, profile_type)
        setattr(self._solver, longname, vmec_profile)
        self._solver.set_profile(vmec_profile, letter)
        return vmec_profile

    @staticmethod
    def _to_vmec_profile(profile, n, profile_type):
        """
        Convert a simsopt :obj:`~simsopt.mhd.profiles.Profile` to a
        :obj:`VmecProfile` with the parametrization ``profile_type``. If
        the simsopt profile already uses that parametrization, its dofs
        are passed through unchanged; otherwise the profile is sampled
        and refit, using ``n`` polynomial coefficients or spline nodes.
        """
        if profile_type[:12] == 'power_series':
            if isinstance(profile, ProfilePolynomial):
                return VmecProfile(profile_type, np.array(profile.local_full_x))

            # Evaluate the new Profile on a Gauss-Legendre grid in s,
            # so the polynomial fit is well conditioned.
            nodes, weights = np.polynomial.legendre.leggauss(n)
            x = nodes * 0.5 + 0.5  # So x is in (0, 1)
            y = profile(x)
            poly = np.polynomial.polynomial.Polynomial.fit(x, y, n - 1, domain=[0, 1]).convert().coef
            logger.debug(f'Fitting a power series to a profile.  x: {x}  y: {y}  poly: {poly}')
            return VmecProfile(profile_type, poly)

        elif profile_type[:12] == 'cubic_spline' \
                or profile_type[:12] == 'akima_spline' \
                or profile_type[:12] == 'line_segment':
            # A ProfileSpline is a spline of the same kind that VMEC uses
            # only for the cubic_spline (degree 3) and line_segment
            # (degree 1) types. Akima splines have no simsopt equivalent,
            # so they are sampled below.
            matching_degree = {'cubic_spline': 3, 'line_segment': 1}.get(profile_type[:12])
            if isinstance(profile, ProfileSpline) and profile.degree == matching_degree:
                return VmecProfile(profile_type, np.array(profile.local_full_x),
                                   np.array(profile.s))

            x = np.linspace(0, 1, n)
            y = profile(x)
            logger.debug(f'Sampling a profile for splines. x: {x}  y: {y}')
            return VmecProfile(profile_type, y, x)

        else:
            raise RuntimeError('To use a simsopt Profile class with vmec, vmec profile type must be power_series, '
                               'cubic_spline, akima_spline, or line_segment. For current profiles, _i or _ip can be appended.')

    def set_indata(self):
        """
        Transfer data from simsopt objects to the solver.  Presently,
        this function sets the boundary shape, the magnetic axis shape,
        and the input profiles. This data transfer is performed before
        writing a Vmec input file or running Vmec. The boundary surface
        object converted to ``SurfaceRZFourier`` is returned.
        """
        if not self.runnable:
            raise RuntimeError('Cannot access indata for a Vmec object that was initialized from a wout file.')
        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()
        self._solver.boundary = self._to_vmec_boundary(boundary_RZFourier)

        # Set profiles, if they are not None:
        self.set_profile("pressure", "mass", "m")
        current = self.set_profile("current", "curr", "c")
        self.set_profile("iota", "iota", "i")
        if self.pressure_profile is not None:
            self._solver.pres_scale = 1.0
        if self.current_profile is not None:
            # The total current is obtained from the simsopt Profile,
            # which is callable, rather than from the coefficients
            # transferred to the solver:
            if current.profile_type in ['power_series', 'gauss_trunc', 'two_power',
                                        'cubic_spline_ip', 'akima_spline_ip']:
                integral, _ = quad(self.current_profile, 0, 1)
                self._solver.curtor = integral
            else:
                self._solver.curtor = self.current_profile(1.0)

        self._solver.set_indata()
        return boundary_RZFourier

    @staticmethod
    def _to_vmec_boundary(surface):
        """
        Convert a :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier` to
        a :obj:`VmecBoundary` for the solver.
        """
        boundary = VmecBoundary(nfp=surface.nfp, stellsym=surface.stellsym,
                                mpol=surface.mpol, ntor=surface.ntor,
                                surface=surface)
        for m in range(surface.mpol + 1):
            for n in range(-surface.ntor, surface.ntor + 1):
                boundary.rbc[(m, n)] = surface.get_rc(m, n)
                boundary.zbs[(m, n)] = surface.get_zs(m, n)
                if not surface.stellsym:
                    boundary.rbs[(m, n)] = surface.get_rs(m, n)
                    boundary.zbc[(m, n)] = surface.get_zc(m, n)
        return boundary

    def get_input(self):
        """
        Generate a VMEC input file. The result will be returned as a
        string. To save a file, see the ``write_input()`` function.
        """
        self.set_indata()  # Transfer the boundary and profiles to the solver.
        return self._solver.get_input()

    def write_input(self, filename):
        """
        Write a VMEC input file. To just get the result as a string
        without saving a file, see the ``get_input()`` function.

        Args:
            filename: Name of the file to write. Selected MPI processes can pass
              ``None`` if you wish for these processes to not write a file.
        """
        # All procs should call self.set_indata(), even procs that do
        # not directly write the file:
        self.set_indata()
        self._solver.write_input(filename)

    def run(self):
        """
        Run VMEC, if ``need_to_run_code`` is ``True``.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run VMEC.")
            return

        if not self.runnable:
            raise RuntimeError('Cannot run a Vmec object that was initialized from a wout file.')

        # Transfer the boundary and profiles from simsopt to the solver:
        self.set_indata()

        self._solver.solve()
        self._set_grids()

        self.need_to_run_code = False

    def _set_grids(self):
        """
        Set the radial grids from the ``wout`` data. In contrast to wout
        files, ``s_half_grid`` has only ns-1 entries, so there is no
        leading 0.
        """
        self.s_full_grid = np.linspace(0, 1, self.wout.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds

    def load_wout(self):
        """
        Read in the most recent ``wout`` file created, and store all the
        data in a ``wout`` attribute of this Vmec object.
        """
        if self._solver is None:
            ierr = load_wout_file(self.output_file, self._wout)
        else:
            ierr = self._solver.load_wout()
        self._set_grids()
        return ierr

    def update_mpi(self, new_mpi):
        """
        Replace the :obj:`~simsopt.util.mpi.MpiPartition` with a new one.

        Args:
            new_mpi: A new :obj:`simsopt.util.mpi.MpiPartition` object.
        """
        self.mpi = new_mpi
        if self._solver is not None:
            self._solver.update_mpi(new_mpi)

    def aspect(self):
        """
        Return the plasma aspect ratio.
        """
        self.run()
        return self.wout.aspect

    def volume(self):
        """
        Return the volume inside the VMEC last closed flux surface.
        """
        self.run()
        return self.wout.volume

    def iota_axis(self):
        """
        Return the rotational transform on axis
        """
        self.run()
        return self.wout.iotaf[0]

    def iota_edge(self):
        """
        Return the rotational transform at the boundary
        """
        self.run()
        return self.wout.iotaf[-1]

    def mean_iota(self):
        """
        Return the mean rotational transform. The average is taken over
        the normalized toroidal flux s.
        """
        self.run()
        return np.mean(self.wout.iotas[1:])

    def mean_shear(self):
        """
        Return an average magnetic shear, d(iota)/ds, where s is the
        normalized toroidal flux. This is computed by fitting the
        rotational transform to a linear (plus constant) function in
        s. The slope of this fit function is returned.
        """
        self.run()

        # Fit a linear polynomial:
        poly = np.polynomial.Polynomial.fit(self.s_half_grid,
                                            self.wout.iotas[1:], deg=1)
        # Return the slope:
        return poly.deriv()(0)

    def get_max_mn(self):
        """
        Look through the rbc and zbs data in the solver to determine the
        largest m and n for which rbc or zbs is nonzero.
        """
        return self._solver_attribute("get_max_mn").get_max_mn()

    def __repr__(self):
        """
        Print the object in an informative way.
        """
        return f"{self.name} (nfp={self.indata.nfp} mpol={self.indata.mpol}" + \
               f" ntor={self.indata.ntor})"

    def external_current(self):
        """
        Return the total electric current associated with external
        currents, i.e. the current through the "doughnut hole". This
        number is useful for coil optimization, to know what the sum
        of the coil currents must be.

        Returns:
            float with the total external electric current in Amperes.
        """
        self.run()
        bvco = self.wout.bvco[-1] * 1.5 - self.wout.bvco[-2] * 0.5
        mu0 = 4 * np.pi * (1.0e-7)
        # The formula in the next line follows from Ampere's law:
        # \int \vec{B} dot (d\vec{r} / d phi) d phi = mu_0 I.
        return 2 * np.pi * bvco / mu0

    def vacuum_well(self):
        """
        Compute a single number W that summarizes the vacuum magnetic well,
        given by the formula

        W = (dV/ds(s=0) - dV/ds(s=1)) / (dV/ds(s=0)

        where dVds is the derivative of the flux surface volume with
        respect to the radial coordinate s. Positive values of W are
        favorable for stability to interchange modes. This formula for
        W is motivated by the fact that

        d^2 V / d s^2 < 0

        is favorable for stability. Integrating over s from 0 to 1
        and normalizing gives the above formula for W. Notice that W
        is dimensionless, and it scales as the square of the minor
        radius. To compute dV/ds, we use

        dV/ds = 4 * pi**2 * abs(sqrt(g)_{0,0})

        where sqrt(g) is the Jacobian of (s, theta, phi) coordinates,
        computed by VMEC in the gmnc array, and _{0,0} indicates the
        m=n=0 Fourier component. Since gmnc is reported by VMEC on the
        half mesh, we extrapolate by half of a radial grid point to s
        = 0 and 1.
        """
        self.run()

        # gmnc is on the half mesh, so drop the 0th radial entry:
        dVds = 4 * np.pi * np.pi * np.abs(self.wout.gmnc[0, 1:])

        # To get from the half grid to s=0 and s=1, we must
        # extrapolate by 1/2 of a radial grid point:
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]

        well = (dVds_s0 - dVds_s1) / dVds_s0
        return well

    return_fn_map = {'aspect': aspect, 'volume': volume, 'iota_axis': iota_axis,
                     'iota_edge': iota_edge, 'mean_iota': mean_iota,
                     'mean_shear': mean_shear, 'vacuum_well': vacuum_well}
