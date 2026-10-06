# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
The VMEC++ backend::

    v = Vmec("input.li383_low_res", solver=VmecppSolver(max_threads=4))
"""

import logging
import os.path
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import vmecpp

from .._core.util import ObjectiveFailure
from ..field.mgrid import MGrid
from .vmec_solver import (
    PROFILE_SIZE_FIELD,
    PROFILE_TYPE_FIELD,
    fit_profile,
    profile_curtor,
    profile_type_tag,
)

if TYPE_CHECKING:
    from .vmec import ProfileProtocol, SurfaceRZFourierProtocol

logger = logging.getLogger(__name__)

__all__ = ["VmecppSolver"]


SUCCESSFUL_TERM_FLAG = 11


def mgrid_response_table(mgrid: MGrid) -> vmecpp.MagneticFieldResponseTable:
    """ The fields of ``mgrid``, per coil group, as VMEC++ reads them from an mgrid file. """
    parameters = vmecpp.MakegridParameters(
        # Both only affect how vmecpp computes a table, and the wout's mgrid_mode:
        normalize_by_currents=False,
        assume_stellarator_symmetry=False,
        number_of_field_periods=int(mgrid.nfp),
        r_grid_minimum=float(mgrid.rmin),
        r_grid_maximum=float(mgrid.rmax),
        number_of_r_grid_points=int(mgrid.nr),
        z_grid_minimum=float(mgrid.zmin),
        z_grid_maximum=float(mgrid.zmax),
        number_of_z_grid_points=int(mgrid.nz),
        number_of_phi_grid_points=int(mgrid.nphi))

    def flatten(fields):
        # (nphi, nz, nr) per coil group, flattened in the order of the mgrid file:
        return np.array([np.ravel(field) for field in fields], dtype=float)

    return vmecpp.MagneticFieldResponseTable(parameters=parameters,
                                             b_r=flatten(mgrid.br_arr),
                                             b_p=flatten(mgrid.bp_arr),
                                             b_z=flatten(mgrid.bz_arr))


class VmecppSolver:
    """
    :obj:`~simsopt.mhd.vmec.VmecSolverProtocol` for VMEC++.

    ``indata`` has no ``m == mpol`` boundary row, so that row of the
    boundary is dropped. Every rank runs VMEC++ serially without communicating;
    ``mpi`` only decides which rank writes files and names them. It may be ``None``.
    """

    #: Input parameters, read from the input file.
    indata: vmecpp.VmecInput
    #: Output of the most recent run, ``None`` before the first one.
    wout: vmecpp.VmecWOut

    def __init__(self, max_threads: int = 1,
                 magnetic_field: "MGrid | vmecpp.MagneticFieldResponseTable | None" = None):
        #: OpenMP threads; 1 avoids oversubscription under finite differencing.
        self.max_threads = max_threads
        if isinstance(magnetic_field, MGrid):
            magnetic_field = mgrid_response_table(magnetic_field)
        #: :obj:`vmecpp.MagneticFieldResponseTable` for free boundary, instead of ``mgrid_file``.
        #: An :obj:`~simsopt.field.mgrid.MGrid` is converted once, here.
        self.magnetic_field = magnetic_field
        #: :obj:`vmecpp.VmecOutput` to hot restart the next solve from, once.
        self.restart_from = None
        #: :obj:`vmecpp.VmecOutput` of the most recent solve.
        self.output_quantities = None
        self.input_file: str | None = None

    def initialize(self, filename, mpi, keep_all_files: bool = False, verbose: bool = True):
        """ Read ``filename``. Called once, by :obj:`~simsopt.mhd.vmec.Vmec`. """
        if self.input_file is not None:
            raise RuntimeError(f"This solver was already initialized from {self.input_file}; "
                               "give each Vmec a new solver")
        basename = os.path.basename(filename)
        if not (basename.startswith('input') or basename.endswith('.json')):
            raise ValueError(f"Invalid filename {filename}: VmecppSolver needs an "
                             "'input.<extension>' or '<name>.json' input file")

        self.mpi = mpi
        self.verbose = verbose
        self.wout = None  # type: ignore[assignment]
        self.output_file: str | None = None

        self._boundary: Optional[SurfaceRZFourierProtocol] = None
        self.pressure: Optional[ProfileProtocol] = None
        self.current: Optional[ProfileProtocol] = None
        self.iota: Optional[ProfileProtocol] = None
        self.n_pressure = 10
        self.n_current = 10
        self.n_iota = 10

        self.iter = -1
        self.keep_all_files = keep_all_files
        self.files_to_delete = []

        self.indata = vmecpp.VmecInput.from_file(filename)
        self.input_file = filename

    @property
    def phiedge(self) -> float:
        return self.indata.phiedge

    @phiedge.setter
    def phiedge(self, phiedge: float):
        self.indata.phiedge = phiedge

    @property
    def curtor(self) -> float:
        return self.indata.curtor

    @curtor.setter
    def curtor(self, curtor: float):
        self.indata.curtor = curtor

    @property
    def pres_scale(self) -> float:
        return self.indata.pres_scale

    @pres_scale.setter
    def pres_scale(self, pres_scale: float):
        self.indata.pres_scale = pres_scale

    @property
    def resolution(self):
        """ Final ``(mpol, ntor)`` of ``indata``. """
        return (self.indata.mpol_max, self.indata.ntor_max)

    def _resize_indata(self, new_mpol, new_ntor):
        self.indata.resize(new_mpol, new_ntor)

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
    def boundary(self) -> "SurfaceRZFourierProtocol":
        """ Read back from ``indata`` until one is assigned. """
        if self._boundary is not None:
            return self._boundary
        return self._boundary_from_indata()

    @boundary.setter
    def boundary(self, boundary: "SurfaceRZFourierProtocol"):
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
                    assert vi.rbs is not None
                    assert vi.zbc is not None
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
        vi.rbc[:, :] = 0.0
        vi.zbs[:, :] = 0.0
        if vi.lasym:
            assert vi.rbs is not None
            assert vi.zbc is not None
            vi.rbs[:, :] = 0.0
            vi.zbc[:, :] = 0.0

        # indata has no m == mpol row, so the boundary's is dropped:
        mpol_capped = min(boundary.mpol + 1, mpol)
        ntor_capped = min(boundary.ntor, ntor)
        for m in range(mpol_capped):
            for n in range(-ntor_capped, ntor_capped + 1):
                vi.rbc[m, n + ntor] = boundary.rbc.get((m, n), 0.0)
                vi.zbs[m, n + ntor] = boundary.zbs.get((m, n), 0.0)
                if vi.lasym:
                    assert vi.rbs is not None
                    assert vi.zbc is not None
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
            assert vi.raxis_s is not None
            assert vi.zaxis_c is not None
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
        except (RuntimeError, ValueError) as e:
            # vmecpp rejects a mismatched hot restart with a ValueError:
            if isinstance(e, ValueError) and restart_from is None:
                raise
            wout = getattr(e, "wout", None)
            reason = "" if wout is None else f" {wout.reason}."
            raise ObjectiveFailure(f"VMEC++ failed: {e}{reason}") from e
        self.wout = self.output_quantities.wout

        logger.info("VMEC++ run complete. Now saving output.")
        # Group leaders handle files. Unless keep_all_files is True, only
        # worker group 0 saves the wout file:
        if self.mpi is None or self.mpi.proc0_groups:
            if self.keep_all_files or self.group == 0:
                self.wout.save(Path(self.output_file))

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
        self.wout = vmecpp.VmecWOut.from_wout_file(self.output_file)
        if self.wout.ier_flag not in (0, SUCCESSFUL_TERM_FLAG):
            raise ObjectiveFailure(f"VMEC++ did not succeed. {self.wout.reason}")
        return 0

    def update_mpi(self, new_mpi):
        self.mpi = new_mpi

    def __repr__(self):
        mpol, ntor = self.resolution
        return f"VmecppSolver (nfp={self.indata.nfp} mpol={mpol} ntor={ntor})"
