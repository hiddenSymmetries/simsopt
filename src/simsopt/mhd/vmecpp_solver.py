# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides the VMEC++ backend, i.e. the class that owns all
interaction with the ``vmecpp`` python package.

``vmecpp`` is a hard dependency of this module, but this module is not
imported by ``simsopt.mhd.vmec``, so ``import simsopt`` stays free of it::

    from simsopt.mhd.vmec import Vmec
    from simsopt.mhd.vmecpp_solver import VmecppSolver

    v = Vmec("input.li383_low_res", solver=VmecppSolver)
"""

import logging
import os.path
from pathlib import Path

import numpy as np
import pydantic
import vmecpp

from .._core.util import ObjectiveFailure

logger = logging.getLogger(__name__)

__all__ = ["COMMON_INDATA_FIELDS", "VmecppIndata", "VmecppSolver"]


#: ``indata`` field names that mean the same thing on both the VMEC2000
#: and the VMEC++ backend, so that code poking at ``Vmec.indata`` works
#: regardless of which backend is in use.
COMMON_INDATA_FIELDS = (
    # Resolution and iteration control
    'ns_array', 'ftol_array', 'niter_array', 'delt', 'nstep', 'tcon0',
    'gamma', 'mpol', 'ntor', 'ntheta', 'nzeta',
    # Scalars, some of which are dofs
    'phiedge', 'curtor', 'pres_scale', 'ncurr', 'nfp', 'lasym',
    # Free boundary
    'lfreeb', 'mgrid_file', 'extcur',
    # Profiles
    'am', 'ac', 'ai',
    'am_aux_s', 'am_aux_f', 'ac_aux_s', 'ac_aux_f', 'ai_aux_s', 'ai_aux_f',
    'pmass_type', 'pcurr_type', 'piota_type',
    # Magnetic axis, in fortran's spelling
    'raxis_cc', 'raxis_cs', 'zaxis_cc', 'zaxis_cs',
)

#: fortran ``indata`` axis array names mapped to VMEC++'s names.
AXIS_ALIASES = {
    'raxis_cc': 'raxis_c',
    'raxis_cs': 'raxis_s',
    'zaxis_cc': 'zaxis_c',
    'zaxis_cs': 'zaxis_s',
}

#: ``indata`` fields that fortran-facing code may assign ``bytes`` to.
_BYTES_FIELDS = ('mgrid_file', 'pmass_type', 'pcurr_type', 'piota_type')

#: VMEC profile parametrizations simsopt knows how to hand to a solver.
_PROFILE_FAMILIES = ('power_series', 'cubic_spline', 'akima_spline', 'line_segment')

#: ``indata`` field holding the profile type tag, per profile letter.
_PROFILE_TYPE_FIELD = {'m': 'pmass_type', 'c': 'pcurr_type', 'i': 'piota_type'}

#: ``ier_flag`` value meaning "converged", i.e. successful_term_flag.
SUCCESSFUL_TERM_FLAG = 11


def final_resolution(value):
    """ The target Fourier resolution: ``value`` itself, or the last entry of a schedule. """
    return int(value) if np.ndim(value) == 0 else int(value[-1])


class VmecppIndata(vmecpp.VmecInput):
    """
    :obj:`vmecpp.VmecInput` with two conveniences for code written
    against fortran VMEC: the axis arrays are also reachable under their
    fortran names ``raxis_cc``/``raxis_cs``/``zaxis_cc``/``zaxis_cs``,
    and ``bytes`` assigned to ``mgrid_file`` or to the ``p*_type`` tags
    is decoded.

    All other fields, along with their type hints and docstrings, are
    inherited from :obj:`vmecpp.VmecInput`.
    """

    @classmethod
    def from_vmec_input(cls, vmec_input):
        """ Re-type an already validated :obj:`vmecpp.VmecInput` as this class. """
        return cls.model_construct(
            _fields_set=set(vmec_input.__pydantic_fields_set__),
            **dict(vmec_input.__dict__))

    def __getattr__(self, name):
        if name in AXIS_ALIASES:
            return getattr(self, AXIS_ALIASES[name])
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        name = AXIS_ALIASES.get(name, name)
        if name in _BYTES_FIELDS and isinstance(value, bytes):
            value = value.decode().strip()
        super().__setattr__(name, value)


class VmecppSolver:
    """
    The VMEC++ backend, implementing
    :obj:`~simsopt.mhd.vmec.VmecSolverProtocol`.

    All of VMEC++'s input parameters are available through the ``indata``
    attribute, which is a :obj:`VmecppIndata`, i.e. a real
    :obj:`vmecpp.VmecInput`. The boundary shape and the profiles are
    instead taken from the ``boundary``, ``pressure``, ``current`` and
    ``iota`` attributes, which are assigned by the caller and pushed into
    ``indata`` by :meth:`solve`.

    ``indata.rbc`` and friends only have room for ``m = 0, ..., mpol - 1``,
    which is all VMEC uses, whereas a simsopt surface with the same
    ``mpol`` also carries an ``m == mpol`` row. That row is reported as
    part of the boundary, so dof vectors match the VMEC2000 backend, but
    it is silently dropped when pushing to ``indata``.

    Consequently, for input files that do specify boundary coefficients
    at ``m == mpol`` (``input.li383_low_res`` among them) that row reads
    back as zero here while the VMEC2000 backend reports the file
    values. VMEC ignores those modes, so equilibria agree, but
    ``vmec.boundary`` itself does not. Tracked in simsopt PR #437.

    VMEC++ is OpenMP- rather than MPI-parallel, so ``mpi`` is only used
    for output file naming and may be ``None``. Additional VMEC++
    specific settings are plain attributes: ``max_threads``,
    ``restart_from`` and ``magnetic_field``.

    Args:
        filename: Name of a VMEC ``input.<extension>`` file or of a VMEC++
          ``<name>.json`` input file.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, or ``None``.
        keep_all_files: If ``False``, all ``wout`` output files will be deleted
          except for the first and most recent ones from worker group 0. If
          ``True``, all ``wout`` files will be kept.
        verbose: Whether to print to stdout when running VMEC++.
    """

    def __init__(self, filename, mpi, keep_all_files: bool = False, verbose: bool = True):
        basename = os.path.basename(filename)
        if not (basename.startswith('input') or basename.endswith('.json')):
            raise ValueError(f"Invalid filename {filename}: VmecppSolver needs an "
                             "'input.<extension>' or '<name>.json' input file")

        self.mpi = mpi
        self.verbose = verbose
        self.wout = None
        self.output_file = None

        # Physics inputs, assigned by the caller before each solve:
        self._boundary = None
        self.pressure = None
        self.current = None
        self.iota = None

        self.iter = -1
        self.keep_all_files = keep_all_files
        self.files_to_delete = []
        self.input_file = filename

        # vmecpp.VmecInput.from_file() reads either a classic INDATA file
        # or a VMEC++ JSON file:
        self.indata = VmecppIndata.from_vmec_input(vmecpp.VmecInput.from_file(filename))
        self.free_boundary = bool(self.indata.lfreeb)

        #: Number of OpenMP threads. The default of 1 avoids
        #: oversubscribing the machine when simsopt runs one VMEC per
        #: process for finite differencing. Pass ``None`` for all cores.
        self.max_threads = 1
        #: A :obj:`vmecpp.VmecOutput` to hot restart from. Hot restarting
        #: makes the result depend on the history of previous
        #: evaluations, so it must not be used under finite differencing.
        self.restart_from = None
        #: A :obj:`vmecpp.MagneticFieldResponseTable` for in-memory free
        #: boundary runs, instead of ``indata.mgrid_file``.
        self.magnetic_field = None
        #: The full :obj:`vmecpp.VmecOutput` of the most recent solve.
        self.output_quantities = None

    @property
    def phiedge(self):
        """ Toroidal flux at the boundary, a view onto ``indata.phiedge``. """
        return self.indata.phiedge

    @phiedge.setter
    def phiedge(self, phiedge):
        self.indata.phiedge = phiedge

    @property
    def curtor(self):
        """ Total toroidal current, a view onto ``indata.curtor``. """
        return self.indata.curtor

    @curtor.setter
    def curtor(self, curtor):
        self.indata.curtor = curtor

    @property
    def pres_scale(self):
        """ Pressure scale factor, a view onto ``indata.pres_scale``. """
        return self.indata.pres_scale

    @pres_scale.setter
    def pres_scale(self, pres_scale):
        self.indata.pres_scale = pres_scale

    @property
    def resolution(self):
        """ ``(mpol, ntor)`` of ``indata``, resolving a continuation schedule. """
        return (final_resolution(self.indata.mpol), final_resolution(self.indata.ntor))

    @property
    def boundary(self):
        """
        The boundary shape this solver is configured with, as a
        :obj:`~simsopt.mhd.vmec.VmecBoundary`. Before one has been
        assigned this is read back from ``indata``, i.e. from the input
        file. An assigned boundary reaches ``indata`` at the next solve.
        """
        if self._boundary is not None:
            return self._boundary
        return self._boundary_from_indata()

    @boundary.setter
    def boundary(self, boundary):
        self._boundary = boundary

    def _boundary_from_indata(self):
        """ Build a :obj:`~simsopt.mhd.vmec.VmecBoundary` from ``indata``. """
        # Imported here rather than at module scope so that importing
        # simsopt.mhd.vmec never pulls in vmecpp.
        from .vmec import VmecBoundary

        vi = self.indata  # Shorthand
        mpol, ntor = self.resolution
        # mpol is reported unchanged, not mpol - 1 as in
        # vmecpp.simsopt_compat, so that the boundary dof vector does
        # not change length with the backend. See
        # https://github.com/hiddenSymmetries/simsopt/pull/437
        boundary = VmecBoundary(nfp=vi.nfp, stellsym=not vi.lasym, mpol=mpol, ntor=ntor)
        for m in range(mpol):
            for n in range(-ntor, ntor + 1):
                boundary.rbc[(m, n)] = vi.rbc[m, n + ntor]
                boundary.zbs[(m, n)] = vi.zbs[m, n + ntor]
                if vi.lasym:
                    boundary.rbs[(m, n)] = vi.rbs[m, n + ntor]
                    boundary.zbc[(m, n)] = vi.zbc[m, n + ntor]
        return boundary

    def set_profile(self, profile, letter):
        """
        Write a profile into ``indata``'s arrays for the pressure,
        current, or iota profile.

        Args:
            profile: A :obj:`~simsopt.mhd.vmec.VmecProfile`, or ``None``
              to leave ``indata`` unchanged.
            letter: ``"m"`` for pressure, ``"c"`` for current, or ``"i"``
              for iota.
        """
        if profile is None:
            return

        profile_type = profile.profile_type
        coeffs = np.asarray(profile.coeffs, dtype=float)
        family = profile_type[:12]
        # Fresh arrays rather than slice assignment, since VMEC++'s
        # profile arrays are variable length.
        if family == 'power_series':
            logger.debug(f'Setting vmec a{letter} profile using power series: {coeffs}')
            setattr(self.indata, 'a' + letter, np.array(coeffs))
        elif family in ('cubic_spline', 'akima_spline', 'line_segment'):
            knots = np.asarray(profile.knots, dtype=float)
            logger.debug(f'Setting vmec a{letter} profile using splines. '
                         f'knots: {knots}  values: {coeffs}')
            setattr(self.indata, f'a{letter}_aux_s', np.array(knots))
            setattr(self.indata, f'a{letter}_aux_f', np.array(coeffs))
        else:
            raise ValueError(self._profile_type_error(profile_type))

        # vmecpp.set_profile() is deliberately not used: it returns a
        # copy and forces line_segment, which would both break the
        # identity of the object Vmec.indata returns and override the
        # parametrization the user asked for.
        try:
            setattr(self.indata, _PROFILE_TYPE_FIELD[letter], profile_type)
        except pydantic.ValidationError as e:
            raise ValueError(self._profile_type_error(profile_type)) from e

    @staticmethod
    def _profile_type_error(profile_type):
        return (f"The VMEC++ backend cannot use the profile type '{profile_type}'. "
                f"It must be one of {', '.join(_PROFILE_FAMILIES)}, with '_i' or "
                "'_ip' optionally appended for current profiles.")

    def _push_to_indata(self):
        """
        Transfer the boundary shape and the profiles from this object's
        attributes into ``indata``. Performed before running VMEC++.
        """
        vi = self.indata  # Shorthand
        boundary = self._boundary
        if boundary is None:
            raise RuntimeError("No boundary has been assigned to the solver.")
        mpol, ntor = self.resolution
        vi.rbc.fill(0.0)
        vi.zbs.fill(0.0)
        if vi.lasym:
            vi.rbs.fill(0.0)
            vi.zbc.fill(0.0)

        # VMEC++ holds m = 0, ..., mpol - 1, all VMEC uses, so the
        # inert m == mpol row is dropped. Modes a lower-resolution
        # surface lacks stay zero; the surface is never resized, so
        # run() cannot alter the caller's boundary or its dofs.
        mpol_capped = min(boundary.mpol + 1, mpol)
        ntor_capped = min(boundary.ntor, ntor)
        for m in range(mpol_capped):
            for n in range(-ntor_capped, ntor_capped + 1):
                vi.rbc[m, n + ntor] = boundary.rbc.get((m, n), 0.0)
                vi.zbs[m, n + ntor] = boundary.zbs.get((m, n), 0.0)
                if vi.lasym:
                    vi.rbs[m, n + ntor] = boundary.rbs.get((m, n), 0.0)
                    vi.zbc[m, n + ntor] = boundary.zbc.get((m, n), 0.0)

        # Set axis shape to something that is obviously wrong (R=0) to
        # trigger vmec's internal guess_axis.f to run. Otherwise the
        # initial axis shape for run N will be the final axis shape
        # from run N-1, which makes VMEC results depend slightly on
        # the history of previous evaluations, confusing the finite
        # differencing.
        vi.raxis_c.fill(0.0)
        vi.zaxis_s.fill(0.0)
        if vi.lasym:
            vi.raxis_s.fill(0.0)
            vi.zaxis_c.fill(0.0)

        # Set profiles, if they are not None:
        self.set_profile(self.pressure, "m")
        self.set_profile(self.current, "c")
        self.set_profile(self.iota, "i")

    @property
    def group(self):
        """ Index of this process's worker group, or 0 without MPI. """
        return 0 if self.mpi is None else self.mpi.group

    def _base_filename(self):
        """
        ``input.<extension>_<group>_<iter>``, the naming scheme
        :obj:`~simsopt.mhd.vmec_solver.Vmec2000Solver` uses. A ``.json``
        input file is renamed into that scheme.
        """
        name = os.path.basename(self.input_file)
        if name.endswith('.json'):
            name = name[:-len('.json')]
            if not name.startswith('input.'):
                name = 'input.' + name
        return name + f'_{self.group:03d}_{self.iter:06d}'

    def solve(self):
        """
        Run VMEC++ and store the resulting ``wout`` data.
        """
        logger.info("Preparing to run VMEC++.")

        self.iter += 1
        self._push_to_indata()
        self.output_file = os.path.join(
            os.getcwd(),
            self._base_filename().replace('input.', 'wout_') + '.nc')

        logger.info("Running VMEC++.")
        kwargs = {}
        if self.magnetic_field is not None:
            kwargs["magnetic_field"] = self.magnetic_field
        try:
            self.output_quantities = vmecpp.run(
                self.indata,
                max_threads=self.max_threads,
                # Never let vmecpp's default animated progress bar through,
                # since it would pollute optimizer logs.
                verbose=1 if self.verbose else 0,
                restart_from=self.restart_from,
                **kwargs)
        except RuntimeError as e:
            wout = getattr(e, "wout", None)
            reason = "" if wout is None else f" {wout.reason}."
            raise ObjectiveFailure(f"VMEC++ failed: {e}{reason}") from e
        self.wout = self.output_quantities.wout

        logger.info("VMEC++ run complete. Now saving output.")
        # Every process in a worker group runs its own VMEC++, so only
        # the group leader writes the file. Consumers such as
        # virtual_casing.from_vmec and Boozer read vmec.output_file, so
        # this is written whether or not keep_all_files is set.
        if self.mpi is None or self.mpi.proc0_groups:
            self.wout.save(Path(self.output_file))

        if self.mpi is not None:
            # Make sure all procs are done with the file before deleting:
            self.mpi.comm_groups.barrier()

        # Group leaders handle deletion of files:
        if self.mpi is None or self.mpi.proc0_groups:
            # If the worker group is not 0, delete all wout files, unless
            # keep_all_files is True:
            if (not self.keep_all_files) and (self.group > 0):
                os.remove(self.output_file)

            # Delete the previous output file, if desired:
            for filename in self.files_to_delete:
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    logger.debug(f"Tried to delete the file {filename} but it was not found")

            self.files_to_delete = []

            # Record the latest output file to delete if we run again:
            if (self.group == 0) and (self.iter > 0) and (not self.keep_all_files):
                self.files_to_delete += [self.output_file]

    def load_wout(self):
        """
        Read in the most recent ``wout`` file created, and store the data
        in the ``wout`` attribute of this object.
        """
        logger.info(f"Attempting to read file {self.output_file}")
        self.wout = vmecpp.VmecWOut.from_wout_file(self.output_file)
        if self.wout.ier_flag not in (0, SUCCESSFUL_TERM_FLAG):
            raise ObjectiveFailure(f"VMEC++ did not succeed. {self.wout.reason}")
        return 0

    def update_mpi(self, new_mpi):
        """
        Adopt a new :obj:`~simsopt.util.mpi.MpiPartition`. VMEC++ is
        OpenMP-only, so the partition only affects output file names.

        Args:
            new_mpi: A new :obj:`simsopt.util.mpi.MpiPartition` object, or ``None``.
        """
        self.mpi = new_mpi
        if new_mpi is not None:
            # Synchronize iteration counters, so that all procs within a
            # group agree on which wout file to write.
            self.iter = new_mpi.comm_world.bcast(self.iter)

    def __repr__(self):
        mpol, ntor = self.resolution
        return f"VmecppSolver (nfp={self.indata.nfp} mpol={mpol} ntor={ntor})"
