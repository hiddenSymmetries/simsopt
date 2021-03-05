# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path

import numpy as np
from scipy.io import netcdf
from mpi4py import MPI
from monty.dev import requires
try:
    import vmec
except ImportError as err:
    vmec = None

from simsopt.core.optimizable import Optimizable, optimizable
from simsopt.core.surface import SurfaceRZFourier
from simsopt.core.util import Struct
from simsopt.util.mpi import MpiPartition

logger = logging.getLogger(__name__)

# Flags used by runvmec():
restart_flag = 1
readin_flag = 2
timestep_flag = 4
output_flag = 8
cleanup_flag = 16
reset_jacdt_flag = 32

"""
value flag-name         calls routines to...
----- ---------         ---------------------
  1   restart_flag      reset internal run-control parameters
                        (for example, if jacobian was bad, to try a smaller 
                        time-step)
  2   readin_flag       read in data from input_file and initialize parameters
                        or arrays which do not dependent on radial grid size
                        allocate internal grid-dependent arrays used by vmec;
                        initialize internal grid-dependent vmec profiles (xc,
                        iota, etc);
                        setup loop for radial multi-grid meshes or, if
                        ns_index = ictrl_array(4) is > 0, use radial grid
                        points specified by ns_array[ns_index]
  4   timestep_flag     iterate vmec either by "niter" time steps or until ftol
                        satisfied, whichever comes first.
                        If numsteps (see below) > 0, vmec will return
                        to caller after numsteps, rather than niter, steps.
  8   output_flag       write out output files (wout, jxbout)
 16   cleanup_flag      cleanup (deallocate arrays) - this terminates present
                        run of the sequence
                        This flag will be ignored if the run might be continued.
                        For example, if ier_flag (see below) returns the value
                        more_iter_flag, the cleanup code will be skipped even if
                        cleanup_flag is set, so that the run could be continued
                        on the next call to runvmec.
 32   reset_jacdt_flag  Resets ijacobian flag and time step to delt0
                        thus, setting ictrl_flag = 1+2+4+8+16 will perform ALL
                        the tasks thru cleanup_flag in addition,
                        if ns_index = 0 and numsteps = 0 (see below), vmec will
                        control its own run history
"""


@requires(vmec is not None,
          "Running VMEC from simsopt requires VMEC python extension. "
          "Install the VMEC python extension from "
          "https://https://github.com/hiddenSymmetries/VMEC2000")
class Vmec(Optimizable):
    """
    This class represents the VMEC equilibrium code.
    """

    def __init__(self, filename=None, mpi=None):
        """
        Constructor
        """
        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'input.default')
            logger.info("Initializing a VMEC object from defaults in "
                        + filename)
        else:
            logger.info("Initializing a VMEC object from file: " + filename)
        self.input_file = filename

        # Get MPI communicator:
        if mpi is None:
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi
        comm = self.mpi.comm_groups
        self.fcomm = comm.py2f()

        self.ictrl = np.zeros(5, dtype=np.int32)
        self.iter = 0
        self.wout = Struct()

        self.ictrl[0] = restart_flag + readin_flag
        self.ictrl[1] = 0  # ierr
        self.ictrl[2] = 0  # numsteps
        self.ictrl[3] = 0  # ns_index
        self.ictrl[4] = 0  # iseq
        verbose = True
        reset_file = ''
        print('About to call runvmec to readin')
        vmec.runvmec(self.ictrl, filename, verbose, self.fcomm, reset_file)
        ierr = self.ictrl[1]
        print('Done with runvmec. ierr={}. Calling cleanup next.'.format(ierr))
        # Deallocate arrays allocated by VMEC's fixaray():
        vmec.cleanup(False)
        if ierr != 0:
            raise RuntimeError("Failed to initialize VMEC from input file {}. "
                               "error code {}".format(filename, ierr))

        objstr = " for Vmec " + str(hex(id(self)))
        # nfp and stelsym are initialized by the Equilibrium constructor:
        # Equilibrium.__init__(self)

        # Create an attribute for each VMEC input parameter in VMEC's fortran
        # modules,
        vi = vmec.vmec_input  # Shorthand
        self.nfp = vi.nfp
        self.stelsym = not vi.lasym
        # It probably makes sense for a vmec object to have mpol and
        # ntor attributes independent of the boundary, since the
        # boundary may be a kind of surface that does not use the same
        # Fourier representation. But if the surface is a
        # SurfaceRZFourier, how then should the mpol and ntor of this
        # surface be coordinated with the mpol and ntor of the Vmec
        # object?
        self.mpol = vi.mpol
        self.ntor = vi.ntor
        self.delt = vi.delt
        self.tcon0 = vi.tcon0
        self.phiedge = vi.phiedge
        self.curtor = vi.curtor
        self.gamma = vi.gamma
        self.boundary = optimizable(SurfaceRZFourier(nfp=self.nfp,
                                                     stelsym=self.stelsym,
                                                     mpol=self.mpol,
                                                     ntor=self.ntor))
        self.ncurr = vi.ncurr
        self.free_boundary = bool(vi.lfreeb)

        # Transfer boundary shape data from fortran to the ParameterArray:
        for m in range(vi.mpol + 1):
            for n in range(-vi.ntor, vi.ntor + 1):
                self.boundary.rc[m, n + vi.ntor] = vi.rbc[101 + n, m]
                self.boundary.zs[m, n + vi.ntor] = vi.zbs[101 + n, m]
        # Handle a few variables that are not Parameters:
        self.depends_on = ["boundary"]
        self.need_to_run_code = True

        self.fixed = np.full(len(self.get_dofs()), True)
        self.names = ['delt', 'tcon0', 'phiedge', 'curtor', 'gamma']

    def get_dofs(self):
        return np.array(
            [self.delt, self.tcon0, self.phiedge, self.curtor, self.gamma])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.delt = x[0]
        self.tcon0 = x[1]
        self.phiedge = x[2]
        self.curtor = x[3]
        self.gamma = x[4]

    def run(self):
        """
        Run VMEC, if needed.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run VMEC.")
            return
        logger.info("Preparing to run VMEC.")
        # Transfer values from Parameters to VMEC's fortran modules:
        vi = vmec.vmec_input  # Shorthand
        vi.nfp = self.nfp
        vi.lasym = int(not self.stelsym)
        vi.delt = self.delt
        vi.phiedge = self.phiedge
        vi.curtor = self.curtor
        vi.gamma = self.gamma
        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()
        # VMEC does not allow mpol or ntor above 101:
        mpol_capped = np.min((boundary_RZFourier.mpol, 101))
        ntor_capped = np.min((boundary_RZFourier.ntor, 101))
        vi.mpol = mpol_capped
        vi.ntor = ntor_capped
        vi.rbc[:, :] = 0
        vi.zbs[:, :] = 0
        # Transfer boundary shape data from the surface object to VMEC:
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                vi.rbc[101 + n, m] = boundary_RZFourier.get_rc(m, n)
                vi.zbs[101 + n, m] = boundary_RZFourier.get_zs(m, n)

        # Set axis shape to something that is obvious wrong (R=0) to
        # trigger vmec's internal guess_axis.f to run. Otherwise the
        # initial axis shape for run N will be the final axis shape
        # from run N-1, which makes VMEC results depend slightly on
        # the history of previous evaluations, confusing the finite
        # differencing.
        vi.raxis_cc[:] = 0
        vi.raxis_cs[:] = 0
        vi.zaxis_cc[:] = 0
        vi.zaxis_cs[:] = 0

        self.iter += 1
        input_file = self.input_file + '_{:03d}_{:06d}'.format(
            self.mpi.group, self.iter)
        self.output_file = os.path.join(
            os.getcwd(),
            os.path.basename(input_file).replace('input.', 'wout_') + '.nc')

        # I should write an input file here.
        logger.info("Calling VMEC reinit().")
        vmec.reinit()

        logger.info("Calling runvmec().")
        self.ictrl[0] = restart_flag + reset_jacdt_flag \
                        + timestep_flag + output_flag
        self.ictrl[1] = 0  # ierr
        self.ictrl[2] = 0  # numsteps
        self.ictrl[3] = 0  # ns_index
        self.ictrl[4] = 0  # iseq
        verbose = True
        reset_file = ''
        vmec.runvmec(self.ictrl, input_file, verbose, self.fcomm, reset_file)
        ierr = self.ictrl[1]

        # Deallocate arrays, even if vmec did not converge:
        logger.info("Calling VMEC cleanup().")
        vmec.cleanup(True)

        if ierr != 11:  # 11 = successful_term_flag, defined in General/vmec_params.f
            raise RuntimeError("VMEC did not converge. "
                               "error code {}".format(ierr))
        logger.info("VMEC run complete. Now loading output.")
        self.load_wout()
        logger.info("Done loading VMEC output.")
        self.need_to_run_code = False

    def load_wout(self):
        ierr = 0
        logger.info("Attempting to read file " + self.output_file)
        # vmec.read_wout_mod.read_wout_file(self.output_file, ierr)
        #if ierr == 0:
        #    logger.info('Successufully load VMEC results from ' + \
        #                self.output_file)
        # else:
        #    print('Load VMEC results from {:} failed!'.format(
        #        self.output_file))
        #wout = vmec.read_wout_mod
        #print('xm:', wout.xm)
        #print('xn:', wout.xn)
        #print('xm_nyq:', wout.xm_nyq)
        #print('xn_nyq:', wout.xn_nyq)
        #print('type(xm):', type(wout.xm), ' type(xn):', type(wout.xn), ' type(xm_nyq):', type(wout.xm_nyq), ' type(xn_nyq):', type(wout.xn_nyq))
        #print('ierr:', ierr)
        #print('mnmax:', wout.mnmax, ' len(xm):', len(wout.xm), ' len(xn):', len(wout.xn))
        #print('mnmax_nyq:', wout.mnmax_nyq, ' len(xm_nyq):', len(wout.xm_nyq), ' len(xn_nyq):', len(wout.xn_nyq))
        #assert len(wout.xm) == wout.mnmax
        #assert len(wout.xn) == wout.mnmax
        #assert len(wout.xm_nyq) == wout.mnmax_nyq
        #assert len(wout.xn_nyq) == wout.mnmax_nyq

        f = netcdf.netcdf_file(self.output_file, mmap=False)
        self.wout.ier_flag = f.variables['ier_flag'][()]
        if self.wout.ier_flag != 0:
            logger.info("VMEC did not succeed!")
            raise RuntimeError("VMEC did not succeed")
        self.wout.nfp = f.variables['nfp'][()]
        self.wout.lasym = f.variables['lasym__logical__'][()]
        self.wout.ns = f.variables['ns'][()]
        self.wout.mnmax = f.variables['mnmax'][()]
        self.wout.mnmax_nyq = f.variables['mnmax_nyq'][()]
        self.wout.xm = f.variables['xm'][()]
        self.wout.xn = f.variables['xn'][()]
        self.wout.xm_nyq = f.variables['xm_nyq'][()]
        self.wout.xn_nyq = f.variables['xn_nyq'][()]
        self.wout.mpol = f.variables['mpol'][()]
        self.wout.ntor = f.variables['ntor'][()]
        self.wout.bmnc = f.variables['bmnc'][()].transpose()
        self.wout.rmnc = f.variables['rmnc'][()].transpose()
        self.wout.zmns = f.variables['zmns'][()].transpose()
        self.wout.lmns = f.variables['lmns'][()].transpose()
        self.wout.bsubumnc = f.variables['bsubumnc'][()].transpose()
        self.wout.bsubvmnc = f.variables['bsubvmnc'][()].transpose()
        self.wout.iotas = f.variables['iotas'][()]
        self.wout.iotaf = f.variables['iotaf'][()]
        self.wout.aspect = f.variables['aspect'][()]
        self.wout.volume = f.variables['volume_p'][()]
        f.close()

        return ierr

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
        """
        Print the object in an informative way.
        """
        return "Vmec instance " + str(hex(id(self))) + " (nfp=" + \
               str(self.nfp) + " mpol=" + \
               str(self.mpol) + " ntor=" + str(self.ntor) + ")"
