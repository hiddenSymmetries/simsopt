"""
VMEC python wrapper
Author: Caoxiang Zhu (caoxiangzhu@gmail.com)
This file has a few modifications compared to core.py in the VMEC2000 module.
"""
from __future__ import print_function, absolute_import, division
import numpy as np
import os
import logging
import time
from mpi4py import MPI
from scipy.io import netcdf

import vmec

# Empty mutable object
class Struct():
    pass

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

run_modes =  {'all': 63,
              'input': 35,  # STELLOPT uses 35; V3FIT uses 7
              'output': 8,
              'main': 45}   # STELLOPT uses 61; V3FIT uses 45
# 35 = 1 + 2 + 32 = restart + readin + reset_jacdt
# 45 = 1 + 4 + 8 + 32 = restart + timestep + output + reset_jacdt
"""
value     flag-name             calls routines to...
-----     ---------             ---------------------
  1       restart_flag          reset internal run-control parameters
                                (for example, if jacobian was bad, to try a smaller time-step)
  2       readin_flag           read in data from input_file and initialize parameters/arrays
                                which do not dependent on radial grid size
                                allocate internal grid-dependent arrays used by vmec;
                                initialize internal grid-dependent vmec profiles (xc, iota, etc);
                                setup loop for radial multi-grid meshes or, if ns_index = ictrl_array(4)
                                is > 0, use radial grid points specified by ns_array[ns_index]
  4       timestep_flag         iterate vmec either by "niter" time steps or until ftol satisfied,
                                whichever comes first. If numsteps (see below) > 0, vmec will return
                                to caller after numsteps, rather than niter, steps.
  8       output_flag           write out output files (wout, jxbout)
 16       cleanup_flag          cleanup (deallocate arrays) - this terminates present run of the sequence
                                This flag will be ignored if the run might be continued. For example,
                                if ier_flag (see below) returns the value more_iter_flag, the cleanup
                                code will be skipped even if cleanup_flag is set, so that the run
                                could be continued on the next call to runvmec.
 32       reset_jacdt_flag      Resets ijacobian flag and time step to delt0

      thus, setting ictrl_flag = 1+2+4+8+16 will perform ALL the tasks thru cleanup_flag
      in addition, if ns_index = 0 and numsteps = 0 (see below), vmec will control its own run history
"""

class VMEC(object):
    def __init__(self, input_file='', verbose=False, comm=0, group=0, **kwargs):
        """Initialization of VMEC runs

        Args:
            input_file (str) : Filename for VMEC input namelist. (default: '').
            verbose (bool): If wants scree outputs. (default: True).
            comm (int): MPI_Communicater, should be converted to Fortran
                        format using MPI.py2f(). (default: 0).
        Returns:
            None
        """
        # pass arguments and check
        assert isinstance(input_file, str), \
            "input_file should the input filename in str."
        #if not input_file.startswith('input.'): # This causes problems if the filename starts with a drectory!
        #    input_file = 'input.' + input_file
        self.input_file = input_file
        assert isinstance(comm, int), \
            "MPI communicator should be converted using MPI.py2f()."
        self.comm = comm
        assert isinstance(verbose, bool), "verbose is either True or False."
        self.verbose = verbose
        self.group = group

        # re-usable attributs
        self.iter = 0
        self.success_code = (0, 11) # successful return codes
        self.ictrl = np.zeros(5, dtype=np.int32)

        # initialize VMEC
        #self.reset()
        self.linit = self.run(mode='input', input_file=self.input_file,
                              numsteps=1, verbose=False, comm=self.comm)
        # if failed, run again with verbose
        if not self.linit:
            self.run(mode='input', input_file=self.input_file, numsteps=1,
                     verbose=True, comm=self.comm)
            raise ValueError(
                "VMEC initialization error, code:{:d}".format(self.ictrl[1]))
        # create link for indata
        self.indata = vmec.vmec_input
        # create link for read_wout_mod
        #self.wout = vmec.read_wout_mod # Removed MJL 2021-02-05
        self.wout = Struct()

        return

    def reset(self):
        ier = 0
        numsteps = 0
        ns_index = -1
        #iseq = MPI.COMM_WORLD.Get_rank()
        iseq = 0
        input_file = ''
        reset_file = ''
        # prepare Fortran arguments
        self.ictrl[0] = 16
        self.ictrl[1] = ier
        self.ictrl[2] = numsteps
        self.ictrl[3] = ns_index
        self.ictrl[4] = iseq
        # run VMEC
        print("In core.reset")
        verbose = True
        vmec.runvmec(self.ictrl, input_file, verbose, self.comm, reset_file)

    def finalize(self):
        ier = 0
        numsteps = 0
        ns_index = -1
        #iseq = MPI.COMM_WORLD.Get_rank()
        iseq = 0
        input_file = ''
        reset_file = ''
        # prepare Fortran arguments
        self.ictrl[0] = 16 + 32
        self.ictrl[1] = ier
        self.ictrl[2] = numsteps
        self.ictrl[3] = ns_index
        self.ictrl[4] = iseq
        # run VMEC
        print("In core.finalize")
        #vmec.parallel_vmec_module.parvmec = True
        #vmec.parallel_vmec_module.ns_resltn = 0 # Sam says "Need to do this otherwise situations arrise which cause problems."
        verbose = True
        vmec.runvmec(self.ictrl, input_file, verbose, self.comm, reset_file)
        #vmec.parallel_vmec_module.finalizesurfacecomm(vmec.parallel_vmec_module.ns_comm)
        #vmec.parallel_vmec_module.finalizerunvmec(vmec.parallel_vmec_module.runvmec_comm_world)


    def reinit(self, **kwargs):
        """Re-initialize VMEC run from indata."""
        vmec.reinit()
        return

    def run(self, mode='main', ier=0, numsteps=-1, ns_index=-1, iseq=0,
            input_file=None, verbose=None, comm=None, reset_file='', **kwargs):
        """Interface for Fortran subroutine runvmec in Sources/TimeStep/runvmec.f

        Args:
            mode (str): The running mode of VMEC. It should be one of the
                following options,
                ('all', 'input', 'output', 'main'). (default: 'main').
            ier (int): Flag for error condition. (default: 0).
            numsteps (int): Number time steps to evolve the equilibrium.
                Iterations will stop EITHER if numsteps > 0 and when the
                number of vmec iterations exceeds numsteps; OR if the ftol
                condition is satisfied, whichever comes first. The
                timestep_flag must be set (in ictrl_flag) for this to be
                in effect. If numsteps <= 0, then vmec will choose
                consecutive (and increasing) values from the ns_array
                until ftol is satisfied on each successive multi-grid.
                (default: -1).
            ns_index (int): if > 0 on entry, specifies index (in ns_array)
                of the radial grid to be used for the present iteration
                phase. If ns_index <= 0, vmec will use the previous value
                of this index (if the ftol condition was not satisfied
                during the last call to runvmec) or the next value of this
                index, and it will iterate through each successive
                non-zero member of the ns_array until ftol-convergence
                occurs on each multigrid.
                (default: -1).
            iseq (int): specifies a unique sequence label for identifying
                output files in a sequential vmec run.
                (default: 0).
            input_file (str): Filename for VMEC input namelist.
                (default: None -> self.input_file).
            verbose (bool): If wants scree outputs.
                (default: None -> self.verbose).
            comm (int): MPI_Communicater, should be converted to Fortran
                format using MPI.py2f().
                (default: None -> self.comm).
            reset_file (str): Filename for reset runs. (default: '').

        Returns:
            None
        """
        # check arguments
        mode = mode.lower()
        assert mode in run_modes, ("Unsupported running mode. Should "
                "be one of [{:}].").format(','.join(run_modes.keys()))
        assert ier==0, "Error flag should be zero at input."
        if input_file is None:
            input_file = self.input_file+'_{:03d}_{:06d}'.format(self.group, self.iter)
            #input_file = self.input_file+'_{:06d}'.format(self.iter)
        else:
            if 'input.' not in input_file:
                input_file = 'input.'+input_file
        #self.output_file = input_file.replace('input.', 'wout_')+'.nc'
        # Need to include os.getcwd() if the input file is not in the current working directory.
        self.output_file = os.path.join(os.getcwd(), \
               os.path.basename(input_file).replace('input.', 'wout_')+'.nc')
        if verbose is None:
            verbose = self.verbose
        if comm is None:
            comm = self.comm
        # prepare Fortran arguments
        self.ictrl[0] = run_modes[mode]
        self.ictrl[1] = ier
        self.ictrl[2] = numsteps
        self.ictrl[3] = ns_index
        self.ictrl[4] = iseq

        #vmec.parallel_vmec_module.parvmec = True
        #vmec.parallel_vmec_module.ns_resltn = 0 # Sam says "Need to do this otherwise situations arrise which cause problems."
        
        # run VMEC
        vmec.runvmec(self.ictrl, input_file, verbose, comm, reset_file)
        self.iter += 1
        self.success = self.ictrl[1] in self.success_code

        #vmec.parallel_vmec_module.finalizesurfacecomm(vmec.parallel_vmec_module.ns_comm)
        #vmec.parallel_vmec_module.finalizerunvmec(vmec.parallel_vmec_module.runvmec_comm_world)
        
        return self.success

    def load(self, **kwargs):
        ierr = 0
        if self.success:
            logger.info("Attempting to read file " + self.output_file)
            vmec.read_wout_mod.read_wout_file(self.output_file, ierr)
            if self.verbose:
                if ierr == 0:
                    logger.info('Successufully load VMEC results from ' + \
                          self.output_file)
                else:
                    print('Load VMEC results from {:} failed!'.format(
                            self.output_file))
            """
            wout = vmec.read_wout_mod
            print('xm:', wout.xm)
            print('xn:', wout.xn)
            print('xm_nyq:', wout.xm_nyq)
            print('xn_nyq:', wout.xn_nyq)
            print('type(xm):', type(wout.xm), ' type(xn):', type(wout.xn), ' type(xm_nyq):', type(wout.xm_nyq), ' type(xn_nyq):', type(wout.xn_nyq))
            print('ierr:', ierr)
            print('mnmax:', wout.mnmax, ' len(xm):', len(wout.xm), ' len(xn):', len(wout.xn))
            print('mnmax_nyq:', wout.mnmax_nyq, ' len(xm_nyq):', len(wout.xm_nyq), ' len(xn_nyq):', len(wout.xn_nyq))
            assert len(wout.xm) == wout.mnmax
            assert len(wout.xn) == wout.mnmax
            assert len(wout.xm_nyq) == wout.mnmax_nyq
            assert len(wout.xn_nyq) == wout.mnmax_nyq
            """

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
