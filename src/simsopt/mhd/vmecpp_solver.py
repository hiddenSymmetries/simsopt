# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
The VMEC++ backend::

    v = Vmec("input.li383_low_res", solver=VmecppSolver)
"""

import logging
import os.path
import tempfile
from pathlib import Path

import numpy as np
import vmecpp

from .._core.util import ObjectiveFailure
from .vmec_solver import (
    PROFILE_SIZE_FIELD,
    PROFILE_TYPE_FIELD,
    fit_profile,
    profile_curtor,
    profile_type_tag,
)

logger = logging.getLogger(__name__)

__all__ = ["COMMON_INDATA_FIELDS", "VmecppIndata", "VmecppSolver"]


#: ``indata`` fields with the same meaning on the VMEC2000 and VMEC++ backends.
COMMON_INDATA_FIELDS = (
    'ns_array', 'ftol_array', 'niter_array', 'delt', 'nstep', 'tcon0',
    'gamma', 'mpol', 'ntor', 'ntheta', 'nzeta',
    'phiedge', 'curtor', 'pres_scale', 'ncurr', 'nfp', 'lasym',
    'lfreeb', 'mgrid_file', 'extcur',
    'am', 'ac', 'ai',
    'am_aux_s', 'am_aux_f', 'ac_aux_s', 'ac_aux_f', 'ai_aux_s', 'ai_aux_f',
    'pmass_type', 'pcurr_type', 'piota_type',
    'raxis_cc', 'raxis_cs', 'zaxis_cc', 'zaxis_cs',
)

AXIS_ALIASES = {
    'raxis_cc': 'raxis_c',
    'raxis_cs': 'raxis_s',
    'zaxis_cc': 'zaxis_c',
    'zaxis_cs': 'zaxis_s',
}

_BYTES_FIELDS = ('mgrid_file', 'pmass_type', 'pcurr_type', 'piota_type')

SUCCESSFUL_TERM_FLAG = 11


def final_resolution(value):
    return int(value) if np.ndim(value) == 0 else int(value[-1])


class VmecppIndata(vmecpp.VmecInput):
    """ :obj:`vmecpp.VmecInput` that also accepts the fortran axis names and ``bytes`` tags. """

    @classmethod
    def from_vmec_input(cls, vmec_input):
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
    :obj:`~simsopt.mhd.vmec.VmecSolverProtocol` for VMEC++.

    ``indata`` has no ``m == mpol`` boundary row, so that row of the
    boundary is dropped. Every rank runs VMEC++ serially without communicating;
    ``mpi`` only decides which rank writes files and names them. It may be ``None``.
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

        self._boundary = None
        self.pressure = None
        self.current = None
        self.iota = None
        self.n_pressure = 10
        self.n_current = 10
        self.n_iota = 10

        self.iter = -1
        self.keep_all_files = keep_all_files
        self.files_to_delete = []
        self.input_file = filename

        self.indata = VmecppIndata.from_vmec_input(vmecpp.VmecInput.from_file(filename))
        self.free_boundary = bool(self.indata.lfreeb)

        #: OpenMP threads; 1 avoids oversubscription under finite differencing.
        self.max_threads = 1
        #: :obj:`vmecpp.VmecOutput` to hot restart the next solve from, once.
        self.restart_from = None
        #: :obj:`vmecpp.MagneticFieldResponseTable` for free boundary, instead of ``mgrid_file``.
        self.magnetic_field = None
        #: :obj:`vmecpp.VmecOutput` of the most recent solve.
        self.output_quantities = None

    @property
    def phiedge(self):
        return self.indata.phiedge

    @phiedge.setter
    def phiedge(self, phiedge):
        self.indata.phiedge = phiedge

    @property
    def curtor(self):
        return self.indata.curtor

    @curtor.setter
    def curtor(self, curtor):
        self.indata.curtor = curtor

    @property
    def pres_scale(self):
        return self.indata.pres_scale

    @pres_scale.setter
    def pres_scale(self, pres_scale):
        self.indata.pres_scale = pres_scale

    @property
    def resolution(self):
        """ Final ``(mpol, ntor)`` of ``indata``. """
        return (final_resolution(self.indata.mpol), final_resolution(self.indata.ntor))

    def _resize_indata(self, new_mpol, new_ntor):
        vi = self.indata  # Shorthand
        requested_mpol, requested_ntor = vi.mpol, vi.ntor
        # In place, so references to indata stay live:
        vars(vi).update(vars(vi.resize(new_mpol, new_ntor)))

        # Keep a continuation schedule:
        for name, requested in (('mpol', requested_mpol), ('ntor', requested_ntor)):
            if np.ndim(requested) != 0 and \
                    final_resolution(requested) == getattr(self.indata, name):
                setattr(self.indata, name, requested)

    def _ensure_indata_resolution(self):
        mpol, ntor = self.resolution
        if self.indata.rbc.shape != (mpol, 2 * ntor + 1):
            self._resize_indata(mpol, ntor)

    def set_mpol_ntor(self, new_mpol, new_ntor):
        self._resize_indata(new_mpol, new_ntor)
        self.indata.mpol = new_mpol
        self.indata.ntor = new_ntor

    def _check_lasym_arrays(self):
        vi = self.indata  # Shorthand
        if not vi.lasym:
            return
        missing = [name for name in ('rbs', 'zbc', 'raxis_s', 'zaxis_c')
                   if getattr(vi, name) is None]
        if missing:
            raise ValueError(f"indata.lasym is True but {', '.join(missing)} missing; "
                             "a lasym run needs an input file with LASYM = T")

    @property
    def boundary(self):
        """ Read back from ``indata`` until one is assigned. """
        if self._boundary is not None:
            return self._boundary
        return self._boundary_from_indata()

    @boundary.setter
    def boundary(self, boundary):
        self._boundary = boundary

    def _boundary_from_indata(self):
        from .vmec import VmecBoundary

        self._check_lasym_arrays()
        self._ensure_indata_resolution()
        vi = self.indata  # Shorthand
        mpol, ntor = self.resolution
        # mpol as on VMEC2000, not mpol - 1, so the dof count matches:
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
        """ Fit ``profile`` into ``indata`` for ``letter`` ``"m"``, ``"c"`` or ``"i"``. ``None`` leaves it unchanged. """
        if profile is None:
            return

        profile_type = profile_type_tag(getattr(self.indata, PROFILE_TYPE_FIELD[letter]))
        coeffs, knots = fit_profile(profile, getattr(self, PROFILE_SIZE_FIELD[letter]),
                                    profile_type)
        # Fresh arrays, since VMEC++'s profile arrays are variable length:
        if knots is None:
            logger.debug(f'Setting vmec a{letter} profile using power series: {coeffs}')
            setattr(self.indata, 'a' + letter, np.array(coeffs, dtype=float))
        else:
            logger.debug(f'Setting vmec a{letter} profile using splines. '
                         f'knots: {knots}  values: {coeffs}')
            setattr(self.indata, f'a{letter}_aux_s', np.array(knots, dtype=float))
            setattr(self.indata, f'a{letter}_aux_f', np.array(coeffs, dtype=float))

        if letter == "c":
            self.curtor = profile_curtor(profile, profile_type)

    def _push_to_indata(self):
        boundary = self._boundary
        if boundary is None:
            raise RuntimeError("No boundary has been assigned to the solver.")
        self._check_lasym_arrays()
        self._ensure_indata_resolution()
        vi = self.indata  # Shorthand
        mpol, ntor = self.resolution
        vi.rbc.fill(0.0)
        vi.zbs.fill(0.0)
        if vi.lasym:
            vi.rbs.fill(0.0)
            vi.zbc.fill(0.0)

        # indata has no m == mpol row, so the boundary's is dropped:
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

    def get_input(self):
        """ The input as VMEC++ JSON. """
        self._push_to_indata()
        return self.indata.model_dump_json()

    def write_input(self, filename):
        """ Write JSON, or an INDATA namelist for an ``input.*`` ``filename``. ``None`` writes nothing. """
        # All procs should call self.get_input() so _push_to_indata()
        # gets called, even procs that do not directly write the file:
        indata_json = self.get_input()
        if filename is None or not (self.mpi is None or self.mpi.proc0_groups):
            return

        filename = Path(filename)
        if filename.name.startswith('input.') and filename.suffix != '.json':
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = Path(tmpdir) / (filename.name + '.json')
                json_path.write_text(indata_json)
                with vmecpp.ensure_vmec2000_input(json_path) as indata_path:
                    filename.write_text(indata_path.read_text())
        else:
            filename.write_text(indata_json)

    @property
    def group(self):
        return 0 if self.mpi is None else self.mpi.group

    def _base_filename(self):
        """ ``input.<extension>_<group>_<iter>``, as on VMEC2000. """
        assert self.input_file is not None
        name = os.path.basename(self.input_file)
        if name.endswith('.json'):
            name = name[:-len('.json')]
            if not name.startswith('input.'):
                name = 'input.' + name
        return name + f'_{self.group:03d}_{self.iter:06d}'

    def solve(self):
        """ Run VMEC++ and store ``wout``. """
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

        # A hot restart is used once:
        restart_from, self.restart_from = self.restart_from, None
        indata = self.indata
        if restart_from is not None:
            # A hot restart runs only the last multigrid step:
            indata = indata.model_copy(deep=True)
            indata.ns_array = indata.ns_array[-1:]
            indata.ftol_array = indata.ftol_array[-1:]
            indata.niter_array = indata.niter_array[-1:]

        try:
            self.output_quantities = vmecpp.run(
                indata,
                max_threads=self.max_threads,
                verbose=1 if self.verbose else 0,
                restart_from=restart_from,
                **kwargs)
        except (RuntimeError, AttributeError) as e:
            # vmecpp reports a mismatched hot restart as AttributeError:
            if isinstance(e, AttributeError) and "hot restart" not in str(e):
                raise
            wout = getattr(e, "wout", None)
            reason = "" if wout is None else f" {wout.reason}."
            raise ObjectiveFailure(f"VMEC++ failed: {e}{reason}") from e
        self._set_wout(self.output_quantities.wout)

        logger.info("VMEC++ run complete. Now saving output.")
        if self.mpi is None or self.mpi.proc0_groups:
            self.wout.save(Path(self.output_file))

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
        """ Read ``output_file`` into ``wout``. """
        assert self.output_file is not None
        logger.info(f"Attempting to read file {self.output_file}")
        self._set_wout(vmecpp.VmecWOut.from_wout_file(self.output_file))
        if self.wout.ier_flag not in (0, SUCCESSFUL_TERM_FLAG):
            raise ObjectiveFailure(f"VMEC++ did not succeed. {self.wout.reason}")
        return 0

    def _set_wout(self, wout):
        # In place after the first run, so references to wout stay live:
        if self.wout is None:
            self.wout = wout.model_copy()
        else:
            vars(self.wout).update(vars(wout))

    def update_mpi(self, new_mpi):
        self.mpi = new_mpi

    def __repr__(self):
        mpol, ntor = self.resolution
        return f"VmecppSolver (nfp={self.indata.nfp} mpol={mpol} ntor={ntor})"
