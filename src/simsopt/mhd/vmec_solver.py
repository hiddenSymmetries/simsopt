# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides the VMEC2000 backend, i.e. the class that owns all
interaction with the fortran ``vmec`` python extension.
"""

import logging
import os.path
from datetime import datetime

import numpy as np
from scipy.io import netcdf_file

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

from .._core.util import Struct, ObjectiveFailure

__all__ = ["Vmec2000Solver"]


# Flags used by runvmec():
restart_flag = 1
readin_flag = 2
timestep_flag = 4
output_flag = 8
cleanup_flag = 16
reset_jacdt_flag = 32


def to_namelist_bool(bool_in):
    """ Convert a boolean to a format suitable for fortran namelist input """
    return "T" if bool_in else "F"


def array_to_namelist(arr, aux_s=False):
    """
    This routine writes an array to a string, stopping after the last
    nonzero or nonnegative entry.  This is used for writing the array
    data in vmec input files.
    """
    if aux_s:
        if np.all(arr < 0):
            index = 0
        else:
            index = np.max(np.where(arr >= 0))
    else:
        if np.all(arr == 0):
            index = 0
        else:
            index = np.max(np.nonzero(arr))
    nml = ''
    for j in range(index + 1):
        nml += f'{arr[j]} '
    nml += '\n'
    return nml

# Documentation of flags for runvmec() from the VMEC source code:
#
#value flag-name         calls routines to...
#----- ---------         ---------------------
#  1   restart_flag      reset internal run-control parameters
#                        (for example, if jacobian was bad, to try a smaller
#                        time-step)
#  2   readin_flag       read in data from input_file and initialize parameters
#                        or arrays which do not dependent on radial grid size
#                        allocate internal grid-dependent arrays used by vmec;
#                        initialize internal grid-dependent vmec profiles (xc,
#                        iota, etc);
#                        setup loop for radial multi-grid meshes or, if
#                        ns_index = ictrl_array(4) is > 0, use radial grid
#                        points specified by ns_array[ns_index]
#  4   timestep_flag     iterate vmec either by "niter" time steps or until ftol
#                        satisfied, whichever comes first.
#                        If numsteps (see below) > 0, vmec will return
#                        to caller after numsteps, rather than niter, steps.
#  8   output_flag       write out output files (wout, jxbout)
# 16   cleanup_flag      cleanup (deallocate arrays) - this terminates present
#                        run of the sequence
#                        This flag will be ignored if the run might be continued.
#                        For example, if ier_flag (see below) returns the value
#                        more_iter_flag, the cleanup code will be skipped even if
#                        cleanup_flag is set, so that the run could be continued
#                        on the next call to runvmec.
# 32   reset_jacdt_flag  Resets ijacobian flag and time step to delt0
#                        thus, setting ictrl_flag = 1+2+4+8+16 will perform ALL
#                        the tasks thru cleanup_flag in addition,
#                        if ns_index = 0 and numsteps = 0 (see below), vmec will
#                        control its own run history


def load_wout_file(output_file, wout):
    """
    Read a VMEC ``wout`` netCDF file, storing the data as attributes of
    the ``wout`` object provided. Attributes are set on the existing
    object rather than on a fresh one, so that references held by user
    code remain valid.

    Args:
        output_file: Name of the ``wout_<extension>.nc`` file to read.
        wout: Object (typically a :obj:`~simsopt._core.util.Struct`) to
          store the data in.
    """
    ierr = 0
    logger.info(f"Attempting to read file {output_file}")

    with netcdf_file(output_file, mmap=False) as f:
        for key, val in f.variables.items():
            # 2D arrays need to be transposed.
            val2 = val[()]  # Convert to numpy array
            val3 = val2.T if len(val2.shape) == 2 else val2
            wout.__setattr__(key, val3)

        if wout.ier_flag != 0:
            logger.info("VMEC did not succeed!")
            raise ObjectiveFailure("VMEC did not succeed")

        # Shorthand for a long variable name:
        wout.lasym = f.variables['lasym__logical__'][()]
        wout.volume = wout.volume_p

    return ierr


class Vmec2000Solver:
    """
    The fortran VMEC2000 backend, implementing
    :obj:`~simsopt.mhd.vmec.VmecSolverProtocol`.

    This class owns all interaction with the ``vmec`` python extension.
    All of VMEC's input parameters are available through the ``indata``
    attribute, which is the fortran ``vmec.vmec_input`` module. The
    boundary shape and the profiles are instead taken from the
    ``boundary``, ``pressure``, ``current`` and ``iota`` attributes,
    which are assigned by the caller and pushed into ``indata`` by
    :meth:`solve` and :meth:`get_input`.

    Since the fortran implementation of VMEC uses global module
    variables, it is not possible to have more than one solver object
    with different parameters.

    Args:
        filename: Name of a VMEC ``input.<extension>`` file.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which
          the worker groups will be used for VMEC calculations.
        keep_all_files: If ``False``, all ``wout`` output files will be deleted
          except for the first and most recent ones from worker group 0. If
          ``True``, all ``wout`` files will be kept.
        verbose: Whether to print to stdout when running vmec.
    """

    def __init__(self, filename, mpi, keep_all_files: bool = False, verbose: bool = True):
        if MPI is None:
            raise RuntimeError("mpi4py needs to be installed for running VMEC")
        if vmec is None:
            raise RuntimeError(
                "Running VMEC from simsopt requires VMEC python extension. "
                "Install the VMEC python extension from "
                "https://github.com/hiddenSymmetries/VMEC2000")

        self.mpi = mpi
        self.verbose = verbose
        self.wout = Struct()
        self.output_file = None

        # Physics inputs, assigned by the caller before each solve:
        self._boundary = None
        self.pressure = None
        self.current = None
        self.iota = None

        comm = self.mpi.comm_groups
        self.fcomm = comm.py2f()

        self.ictrl = np.zeros(5, dtype=np.int32)
        self.iter = -1
        self.keep_all_files = keep_all_files
        self.files_to_delete = []
        self.input_file = filename

        self.indata = vmec.vmec_input  # Shorthand
        vi = vmec.vmec_input  # Shorthand

        self.ictrl[0] = restart_flag + readin_flag
        self.ictrl[1] = 0  # ierr
        self.ictrl[2] = 0  # numsteps
        self.ictrl[3] = 0  # ns_index
        self.ictrl[4] = 0  # iseq
        reset_file = ''
        logger.info('About to call runvmec to readin')
        vmec.runvmec(self.ictrl, filename, self.verbose, self.fcomm, reset_file)
        ierr = self.ictrl[1]
        logger.info(f'Done with runvmec. ierr={ierr}. Calling cleanup next.')
        # Deallocate arrays allocated by VMEC's fixaray():
        vmec.cleanup(False)
        if ierr != 0:
            raise RuntimeError(f"Failed to initialize VMEC from input file {filename}. Error code: {ierr}.")

        self.free_boundary = bool(vi.lfreeb)

    @property
    def phiedge(self):
        """ Toroidal flux at the boundary, a view onto ``indata.phiedge``. """
        return vmec.vmec_input.phiedge

    @phiedge.setter
    def phiedge(self, phiedge):
        vmec.vmec_input.phiedge = phiedge

    @property
    def curtor(self):
        """ Total toroidal current, a view onto ``indata.curtor``. """
        return vmec.vmec_input.curtor

    @curtor.setter
    def curtor(self, curtor):
        vmec.vmec_input.curtor = curtor

    @property
    def pres_scale(self):
        """ Pressure scale factor, a view onto ``indata.pres_scale``. """
        return vmec.vmec_input.pres_scale

    @pres_scale.setter
    def pres_scale(self, pres_scale):
        vmec.vmec_input.pres_scale = pres_scale

    @property
    def boundary(self):
        """
        The boundary shape presently in the fortran ``indata``, as a
        :obj:`~simsopt.mhd.vmec.VmecBoundary`. Assigning a boundary
        stores it; it reaches ``indata`` at the next solve or
        ``get_input()``.
        """
        return self._boundary_from_indata()

    @boundary.setter
    def boundary(self, boundary):
        self._boundary = boundary

    def _boundary_from_indata(self):
        """ Build a :obj:`~simsopt.mhd.vmec.VmecBoundary` from the fortran ``indata``. """
        # Imported here rather than at module scope because vmec.py
        # imports this module.
        from .vmec import VmecBoundary

        vi = vmec.vmec_input  # Shorthand
        boundary = VmecBoundary(nfp=vi.nfp, stellsym=not bool(vi.lasym),
                                mpol=vi.mpol, ntor=vi.ntor)
        for m in range(vi.mpol + 1):
            for n in range(-vi.ntor, vi.ntor + 1):
                boundary.rbc[(m, n)] = vi.rbc[101 + n, m]
                boundary.zbs[(m, n)] = vi.zbs[101 + n, m]
                if vi.lasym:
                    boundary.rbs[(m, n)] = vi.rbs[101 + n, m]
                    boundary.zbc[(m, n)] = vi.zbc[101 + n, m]
        return boundary

    def set_profile(self, profile, letter):
        """
        Write a profile into the fortran arrays for the pressure,
        current, or iota profile.

        Args:
            profile: A :obj:`~simsopt.mhd.vmec.VmecProfile`, or ``None``
              to leave the fortran data unchanged.
            letter: ``"m"`` for pressure, ``"c"`` for current, or ``"i"``
              for iota.
        """
        if profile is None:
            return

        profile_type = profile.profile_type
        coeffs = np.asarray(profile.coeffs)
        n = len(coeffs)
        if profile_type[:12] == 'power_series':
            logger.debug(f'Setting vmec a{letter} profile using power series: {coeffs}')
            ax = self.indata.__getattribute__("a" + letter)
            ax[:] = 0.0
            ax[:n] = coeffs

        elif profile_type[:12] == 'cubic_spline' \
                or profile_type[:12] == 'akima_spline' \
                or profile_type[:12] == 'line_segment':
            knots = np.asarray(profile.knots)
            logger.debug(f'Setting vmec a{letter} profile using splines. '
                         f'knots: {knots}  values: {coeffs}')
            aux_s = self.indata.__getattribute__("a" + letter + "_aux_s")
            aux_f = self.indata.__getattribute__("a" + letter + "_aux_f")
            aux_s[:] = 0.0
            aux_f[:] = 0.0
            aux_s[:n] = knots
            aux_f[:n] = coeffs

        else:
            raise RuntimeError('To use a simsopt Profile class with vmec, vmec profile type must be power_series, '
                               'cubic_spline, akima_spline, or line_segment. For current profiles, _i or _ip can be appended.')

    def _push_to_indata(self):
        """
        Transfer the boundary shape and the profiles from this object's
        attributes to VMEC's fortran module data. This is performed
        before writing a Vmec input file or running Vmec.
        """
        vi = vmec.vmec_input  # Shorthand
        boundary = self._boundary
        if boundary is None:
            raise RuntimeError("No boundary has been assigned to the solver.")
        # VMEC does not allow mpol or ntor above 101:
        if vi.mpol > 101:
            raise ValueError("VMEC does not allow mpol > 101")
        if vi.ntor > 101:
            raise ValueError("VMEC does not allow ntor > 101")
        vi.rbc[:, :] = 0
        vi.zbs[:, :] = 0
        if vi.lasym:
            vi.rbs[:, :] = 0
            vi.zbc[:, :] = 0
        mpol_capped = np.min([boundary.mpol, 101])
        ntor_capped = np.min([boundary.ntor, 101])
        # Transfer boundary shape data from the boundary object to VMEC:
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                vi.rbc[101 + n, m] = boundary.rbc.get((m, n), 0.0)
                vi.zbs[101 + n, m] = boundary.zbs.get((m, n), 0.0)
                if vi.lasym:
                    vi.rbs[101 + n, m] = boundary.rbs.get((m, n), 0.0)
                    vi.zbc[101 + n, m] = boundary.zbc.get((m, n), 0.0)

        # Set axis shape to something that is obviously wrong (R=0) to
        # trigger vmec's internal guess_axis.f to run. Otherwise the
        # initial axis shape for run N will be the final axis shape
        # from run N-1, which makes VMEC results depend slightly on
        # the history of previous evaluations, confusing the finite
        # differencing.
        vi.raxis_cc[:] = 0
        vi.raxis_cs[:] = 0
        vi.zaxis_cc[:] = 0
        vi.zaxis_cs[:] = 0

        # Set profiles, if they are not None:
        self.set_profile(self.pressure, "m")
        self.set_profile(self.current, "c")
        self.set_profile(self.iota, "i")

    def get_input(self):
        """
        Generate a VMEC input file. The result will be returned as a
        string. To save a file, see the ``write_input()`` function.
        """
        self._push_to_indata()  # Transfer the boundary and profiles to fortran.
        vi = vmec.vmec_input  # Shorthand
        nml = '&INDATA\n'
        nml += '! This file created by simsopt on ' + datetime.now().strftime("%B %d %Y, %H:%M:%S") + '\n\n'
        nml += '! ---- Geometric parameters ----\n'
        nml += f'NFP = {vi.nfp}\n'
        nml += f'LASYM = {to_namelist_bool(vi.lasym)}\n'

        if vi.lfreeb:
            nml += '\n! ---- Free-boundary parameters ----\n'
            nml += 'LFREEB = T\n'
            nml += f"MGRID_FILE = '{vi.mgrid_file.decode('utf-8')}'\n"
            nml += 'EXTCUR = ' + array_to_namelist(vi.extcur)
            nml += '\n'

        nml += '\n! ---- Resolution parameters ----\n'
        nml += f'MPOL = {vi.mpol}\n'
        nml += f'NTOR = {vi.ntor}\n'
        if vi.ntheta != 0:
            nml += f'NTHETA = {vi.ntheta}\n'
        if vi.nzeta != 0:
            nml += f'NZETA = {vi.nzeta}\n'
        index = np.max(np.nonzero(vi.ns_array))
        nml += 'NS_ARRAY    ='
        for j in range(index + 1):
            nml += f'{vi.ns_array[j]:7}'
        nml += '\n'
        index = np.max(np.where(vi.niter_array > 0))
        nml += 'NITER_ARRAY ='
        for j in range(index + 1):
            nml += f'{vi.niter_array[j]:7}'
        nml += '\n'
        index = np.max(np.nonzero(vi.ftol_array))
        nml += 'FTOL_ARRAY  ='
        for j in range(index + 1):
            nml += f'{vi.ftol_array[j]:7}'
        nml += '\n'

        nml += '\n! ---- Boundary toroidal flux ----\n'
        nml += f'PHIEDGE = {vi.phiedge}\n'

        nml += '\n! ---- Pressure profile specification ----\n'
        profile_type = vi.pmass_type.decode().strip()
        nml += f'PMASS_TYPE = "{profile_type}"\n'
        nml += 'AM = ' + array_to_namelist(vi.am)
        if np.any(vi.am_aux_s >= 0):
            nml += 'AM_AUX_S = ' + array_to_namelist(vi.am_aux_s, True)
            nml += 'AM_AUX_F = ' + array_to_namelist(vi.am_aux_f)
        nml += f'PRES_SCALE = {vi.pres_scale}\n'

        nml += '\n! ---- Profile specification of iota or current ----\n'
        nml += f'NCURR = {vi.ncurr}\n'
        if vi.ncurr == 0:
            # Iota profile specified
            profile_type = vi.piota_type.decode().strip()
            nml += f'PIOTA_TYPE = "{profile_type}"\n'
            nml += 'AI = ' + array_to_namelist(vi.ai)
            if np.any(vi.ai_aux_s >= 0):
                nml += 'AI_AUX_S = ' + array_to_namelist(vi.ai_aux_s, True)
                nml += 'AI_AUX_F = ' + array_to_namelist(vi.ai_aux_f)
        else:
            # Current profile specified
            nml += f'CURTOR = {vi.curtor}\n'
            profile_type = vi.pcurr_type.decode().strip()
            nml += f'PCURR_TYPE = "{profile_type}"\n'
            nml += 'AC = ' + array_to_namelist(vi.ac)
            if np.any(vi.ac_aux_s >= 0):
                nml += 'AC_AUX_S = ' + array_to_namelist(vi.ac_aux_s, True)
                nml += 'AC_AUX_F = ' + array_to_namelist(vi.ac_aux_f)

        nml += '\n! ---- Other numerical parameters ----\n'
        nml += f'DELT = {vi.delt}\n'
        nml += f'NSTEP = {vi.nstep}\n'

        nml += '\n! ---- Boundary shape. Array index order is (n, m) ----\n'
        surf_str = self._boundary.surface.get_nml().split('\n')
        for j in range(3, len(surf_str)):
            nml += surf_str[j] + '\n'

        return nml

    def write_input(self, filename):
        """
        Write a VMEC input file. To just get the result as a string
        without saving a file, see the ``get_input()`` function.

        Args:
            filename: Name of the file to write. Selected MPI processes can pass
              ``None`` if you wish for these processes to not write a file.
        """
        # All procs should call self.get_input() so _push_to_indata()
        # gets called, even procs that do not directly write the file:
        input_namelist = self.get_input()
        if self.mpi.proc0_groups and (filename is not None):
            with open(filename, 'w') as f:
                f.write(input_namelist)

    def solve(self):
        """
        Run VMEC and load the resulting ``wout`` data.
        """
        logger.info("Preparing to run VMEC.")

        self.iter += 1
        base_filename = self.input_file + '_{:03d}_{:06d}'.format(
            self.mpi.group, self.iter)
        input_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename))
        self.output_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename).replace('input.', 'wout_') + '.nc')
        mercier_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename).replace('input.', 'mercier.'))
        jxbout_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename).replace('input.', 'jxbout_') + '.nc')

        file_to_write = input_file if (self.mpi.proc0_world or self.keep_all_files) else None
        # This next line also calls _push_to_indata():
        self.write_input(file_to_write)

        logger.info("Calling VMEC reinit().")
        vmec.reinit()

        logger.info("Calling runvmec().")
        self.ictrl[0] = restart_flag + reset_jacdt_flag \
            + timestep_flag + output_flag
        self.ictrl[1] = 0  # ierr
        self.ictrl[2] = 0  # numsteps
        self.ictrl[3] = 0  # ns_index
        self.ictrl[4] = 0  # iseq
        reset_file = ''
        vmec.runvmec(self.ictrl, input_file, self.verbose, self.fcomm, reset_file)
        ierr = self.ictrl[1]

        # Deallocate arrays, even if vmec did not converge:
        logger.info("Calling VMEC cleanup().")
        vmec.cleanup(True)

        # See VMEC2000/Sources/General/vmec_params.f for ierr codes.
        # 11 = successful_term_flag.
        # Error codes that are expected to occur due to lack of
        # convergence cause ObjectiveFailure, which the optimizer
        # handles gracefully by treating the point as bad. But the
        # user/developer should know if an error codes arises that
        # should logically never occur, so these codes raise a
        # different exception.
        if ierr in [0, 5]:
            raise RuntimeError(f"runvmec returned an error code that should never occur: ierr={ierr}")
        if ierr != 11:
            raise ObjectiveFailure(f"VMEC did not converge. ierr={ierr}")

        logger.info("VMEC run complete. Now loading output.")
        self.load_wout()
        # Make sure all procs have finished loading the wout file before we delete it:
        self.mpi.comm_groups.barrier()
        logger.info("Done loading VMEC output.")

        # Group leaders handle deletion of files:
        if self.mpi.proc0_groups:
            # Delete some files produced by VMEC that we never care
            # about. For some reason the os.remove statements give a 'file
            # not found' error in the CI, hence the try-except blocks.
            try:
                os.remove(mercier_file)
            except FileNotFoundError:
                logger.debug(f'Tried to delete the file {mercier_file} but it was not found')
                raise

            try:
                os.remove(jxbout_file)
            except FileNotFoundError:
                logger.debug(f'Tried to delete the file {jxbout_file} but it was not found')
                raise

            try:
                os.remove("fort.9")
            except FileNotFoundError:
                logger.debug('Tried to delete the file fort.9 but it was not found')

            # If the worker group is not 0, delete all wout files, unless
            # keep_all_files is True:
            if (not self.keep_all_files) and (self.mpi.group > 0):
                os.remove(self.output_file)

            # Delete the previous output file, if desired:
            for filename in self.files_to_delete:
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    logger.debug(f"Tried to delete the file {filename} but it was not found")

            self.files_to_delete = []

            # Record the latest output file to delete if we run again:
            if (self.mpi.group == 0) and (self.iter > 0) and (not self.keep_all_files):
                self.files_to_delete += [input_file, self.output_file]

    def load_wout(self):
        """
        Read in the most recent ``wout`` file created, and store all the
        data in the ``wout`` attribute of this object.
        """
        return load_wout_file(self.output_file, self.wout)

    def update_mpi(self, new_mpi):
        """
        Replace the :obj:`~simsopt.util.mpi.MpiPartition` with a new one.

        Args:
            new_mpi: A new :obj:`simsopt.util.mpi.MpiPartition` object.
        """
        self.mpi = new_mpi
        self.fcomm = self.mpi.comm_groups.py2f()
        # Synchronize iteration counters. If we don't do this,
        # different procs within a group may have different values of
        # ``iter``, causing them to look for different wout files.
        self.iter = self.mpi.comm_world.bcast(self.iter)

    def get_max_mn(self):
        """
        Look through the rbc and zbs data in fortran to determine the
        largest m and n for which rbc or zbs is nonzero.
        """
        max_m = 0
        max_n = 0
        for m in range(1, 101):
            for n in range(1, 101):
                if np.abs(vmec.vmec_input.rbc[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbs[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbs[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbc[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbc[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbs[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbs[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbc[101 - n, m]) > 0:
                    max_m = np.max((max_m, m))
                    max_n = np.max((max_n, n))
        # It may happen that mpol or ntor exceed the max_m or max_n
        # according to rbc/zbs. In this case, go with the larger
        # value.
        max_m = np.max((max_m, vmec.vmec_input.mpol))
        max_n = np.max((max_n, vmec.vmec_input.ntor))
        return (max_m, max_n)

    def __repr__(self):
        return f"Vmec2000Solver (nfp={self.indata.nfp} mpol={self.indata.mpol}" + \
               f" ntor={self.indata.ntor})"
