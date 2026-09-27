# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
from dataclasses import dataclass, field
from typing import Any, Generic, NamedTuple, Optional, Protocol, TypeVar, runtime_checkable

import numpy as np

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
# Re-exported for backwards compatibility:
from .vmec_solver import (Vmec2000Solver, load_wout_file,  # noqa: F401
                          to_namelist_bool, array_to_namelist,
                          restart_flag, readin_flag, timestep_flag,
                          output_flag, cleanup_flag, reset_jacdt_flag)

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None

__all__ = ["FourierMode", "ProfileProtocol", "SurfaceRZFourierProtocol", "Vmec",
           "VmecBoundary", "VmecSolverProtocol"]


class FourierMode(NamedTuple):
    """ Key of a boundary Fourier coefficient. """
    m: int
    n: int


@runtime_checkable
class SurfaceRZFourierProtocol(Protocol):
    """ Boundary passed to a Vmec solver, with coefficients keyed by :obj:`FourierMode`. """
    nfp: int
    stellsym: bool
    mpol: int
    ntor: int
    rbc: dict
    zbs: dict
    rbs: dict
    zbc: dict


@runtime_checkable
class ProfileProtocol(Protocol):
    """ Radial profile passed to a Vmec solver: a callable of ``s``. """
    def __call__(self, s): ...


@runtime_checkable
class VmecSolverProtocol(Protocol[IndataT, WoutT]):
    """
    Interface of a VMEC backend driven by :obj:`Vmec`.

    The constructor takes backend options only; :obj:`Vmec` then calls
    ``initialize()`` once, with its input file and settings.

    A new backend needs no change in simsopt: this protocol is structural,
    so any class with these members works, whether or not it inherits from
    anything here. Implement it in your own package and pass an instance::

        class MyVmec:
            def __init__(self, **backend_options): ...
            def initialize(self, filename, mpi, keep_all_files=False, verbose=True): ...
            # ... plus the properties and methods below

        vmec = Vmec("input.my_config", solver=MyVmec())

    ``initialize()`` must set up ``indata``, ``boundary`` and the other
    members. ``get_input()``, ``write_input()`` and ``get_max_mn()`` are
    optional; :obj:`Vmec` forwards them if the backend has them.

    ``phiedge``, ``curtor`` and ``pres_scale`` are views onto ``indata``.
    If both ``current`` and ``iota`` are set, ``indata.ncurr`` selects one.
    """

    # Properties, so implementations may define them as properties too:
    @property
    def pressure(self) -> Optional[ProfileProtocol]: ...
    @pressure.setter
    def pressure(self, profile: Optional[ProfileProtocol], /) -> None: ...

    @property
    def current(self) -> Optional[ProfileProtocol]: ...
    @current.setter
    def current(self, profile: Optional[ProfileProtocol], /) -> None: ...

    @property
    def iota(self) -> Optional[ProfileProtocol]: ...
    @iota.setter
    def iota(self, profile: Optional[ProfileProtocol], /) -> None: ...

    @property
    def boundary(self) -> SurfaceRZFourierProtocol: ...
    @boundary.setter
    def boundary(self, boundary: SurfaceRZFourierProtocol, /) -> None: ...

    @property
    def phiedge(self) -> float: ...
    @phiedge.setter
    def phiedge(self, phiedge: float, /) -> None: ...

    @property
    def curtor(self) -> float: ...
    @curtor.setter
    def curtor(self, curtor: float, /) -> None: ...

    @property
    def pres_scale(self) -> float: ...
    @pres_scale.setter
    def pres_scale(self, pres_scale: float, /) -> None: ...

    indata: IndataT
    wout: WoutT
    output_file: Any
    verbose: bool

    def initialize(self, filename: str, mpi, keep_all_files: bool = False,
                   verbose: bool = True) -> None: ...

    def solve(self) -> None: ...

    def load_wout(self) -> int: ...

    def save_wout(self, filename: str) -> None: ...

    def update_mpi(self, new_mpi) -> None: ...


@dataclass
class VmecBoundary:
    """ :obj:`SurfaceRZFourierProtocol` built by :obj:`Vmec` from its boundary ``surface``. """
    nfp: int = 1
    stellsym: bool = True
    mpol: int = 1
    ntor: int = 0
    rbc: dict = field(default_factory=dict)
    zbs: dict = field(default_factory=dict)
    rbs: dict = field(default_factory=dict)
    zbc: dict = field(default_factory=dict)
    surface: Any = None


REQUIRED_WOUT_FIELDS = (
    'aspect', 'Aminor_p', 'Rmajor_p', 'betatotal', 'ctor', 'ier_flag',
    'lasym', 'mnmax', 'mnmax_nyq', 'mpol', 'nfp', 'ns', 'ntor', 'signgs',
    'volavgB', 'volume_p', 'fsqr', 'fsql', 'fsqz',
    'pmass_type', 'pcurr_type', 'piota_type',
    'xm', 'xn', 'xm_nyq', 'xn_nyq',
    'iotaf', 'iotas', 'pres', 'phi', 'chi', 'vp', 'buco', 'bvco',
    'jcurv', 'jdotb',
    'rmnc', 'zmns', 'lmns', 'gmnc', 'bmnc', 'bsupumnc', 'bsupvmnc',
    'bsubumnc', 'bsubvmnc', 'bsubsmns',
)

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
            # A class is a factory, even if its class attributes satisfy the protocol:
            if isinstance(solver, VmecSolverProtocol) and not isinstance(solver, type):
                self._solver = solver
            else:
                self._solver = solver(filename, self.mpi,
                                      keep_all_files=keep_all_files,
                                      verbose=verbose)
            if not isinstance(self._solver, VmecSolverProtocol):
                raise TypeError(f"{type(self._solver).__name__} does not satisfy VmecSolverProtocol")

            # A vmec object has mpol and ntor attributes independent of
            # the boundary. The boundary surface object is initialized
            # with mpol and ntor values that match those of the vmec
            # object, but the mpol/ntor values of either the vmec object
            # or the boundary surface object can be changed independently
            # by the user.
            solver_boundary = self._solver.boundary
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

    @property
    def _require_solver(self):
        if self._solver is None:
            raise AttributeError("This Vmec object was initialized from a wout file, "
                                 "so it has no solver.")
        return self._solver

    @property
    def indata(self):
        if self._solver is None:
            raise AttributeError('Cannot access indata for a Vmec object that was initialized from a wout file.')
        return self._solver.indata

    @property
    def wout(self):
        return self._wout if self._solver is None else self._solver.wout

    @wout.setter
    def wout(self, wout):
        if self._solver is None:
            self._wout = wout
        else:
            self._solver.wout = wout

    @property
    def output_file(self):
        return self._output_file if self._solver is None else self._solver.output_file

    @output_file.setter
    def output_file(self, output_file):
        if self._solver is None:
            self._output_file = output_file
        else:
            self._solver.output_file = output_file

    @property
    def verbose(self):
        return self._verbose if self._solver is None else self._solver.verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        if self._solver is not None:
            self._solver.verbose = verbose

    @property
    def input_file(self):
        return self._require_solver.input_file

    @input_file.setter
    def input_file(self, input_file):
        self._require_solver.input_file = input_file

    @property
    def iter(self):
        return self._require_solver.iter

    @iter.setter
    def iter(self, iter):
        self._require_solver.iter = iter

    @property
    def keep_all_files(self):
        return self._require_solver.keep_all_files

    @keep_all_files.setter
    def keep_all_files(self, keep_all_files):
        self._require_solver.keep_all_files = keep_all_files

    @property
    def files_to_delete(self):
        return self._require_solver.files_to_delete

    @files_to_delete.setter
    def files_to_delete(self, files_to_delete):
        self._require_solver.files_to_delete = files_to_delete

    @property
    def free_boundary(self):
        return self._require_solver.free_boundary

    @free_boundary.setter
    def free_boundary(self, free_boundary):
        self._require_solver.free_boundary = free_boundary

    @property
    def ictrl(self):
        return self._require_solver.ictrl

    @ictrl.setter
    def ictrl(self, ictrl):
        self._require_solver.ictrl = ictrl

    @property
    def fcomm(self):
        return self._require_solver.fcomm

    @fcomm.setter
    def fcomm(self, fcomm):
        self._require_solver.fcomm = fcomm

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
        Hand the profile ``longname`` (``"pressure"``, ``"current"`` or
        ``"iota"``) and its ``n_*`` to the solver. ``shortname`` and
        ``letter`` are unused.
        """
        size = "n_" + longname
        if hasattr(self._solver, size):
            setattr(self._solver, size, getattr(self, size))
        setattr(self._solver, longname, getattr(self, longname + "_profile"))

    def set_indata(self):
        """
        Hand the boundary and profiles to the solver. Returns the boundary
        as a ``SurfaceRZFourier``.
        """
        if not self.runnable:
            raise RuntimeError('Cannot access indata for a Vmec object that was initialized from a wout file.')
        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()
        self._solver.boundary = self._to_vmec_boundary(boundary_RZFourier)

        self.set_profile("pressure", "mass", "m")
        self.set_profile("current", "curr", "c")
        self.set_profile("iota", "iota", "i")
        if self.pressure_profile is not None:
            self._solver.pres_scale = 1.0

        return boundary_RZFourier

    @staticmethod
    def _to_vmec_boundary(surface):
        boundary = VmecBoundary(nfp=surface.nfp, stellsym=surface.stellsym,
                                mpol=surface.mpol, ntor=surface.ntor,
                                surface=surface)
        for m in range(surface.mpol + 1):
            for n in range(-surface.ntor, surface.ntor + 1):
                boundary.rbc[FourierMode(m, n)] = surface.get_rc(m, n)
                boundary.zbs[FourierMode(m, n)] = surface.get_zs(m, n)
                if not surface.stellsym:
                    boundary.rbs[FourierMode(m, n)] = surface.get_rs(m, n)
                    boundary.zbc[FourierMode(m, n)] = surface.get_zc(m, n)
        return boundary

    def get_input(self):
        """
        Generate a VMEC input file. The result will be returned as a
        string. To save a file, see the ``write_input()`` function.
        """
        self.set_indata()
        return self._require_solver.get_input()

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
        self._require_solver.write_input(filename)

    def run(self):
        """
        Run VMEC, if ``need_to_run_code`` is ``True``.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run VMEC.")
            return

        if not self.runnable:
            raise RuntimeError('Cannot run a Vmec object that was initialized from a wout file.')

        self.set_indata()

        self._solver.solve()
        self._set_grids()

        self.need_to_run_code = False

    def _set_grids(self):
        """ Radial grids from ``wout.ns``; ``s_half_grid`` has no leading 0. """
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
        return self._require_solver.get_max_mn()

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
