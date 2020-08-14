"""
This module provides a class that handles the VMEC equilibrium code.
"""

import numpy as np
#from FortranNamelist import NamelistFile
#import f90nml
import logging
import os.path
from mpi4py import MPI
from simsopt import *
from simsopt.vmec.core import VMEC

class Vmec(Equilibrium):
    """
    This class represents the VMEC equilibrium code.
    """
    def __init__(self, filename=None):
        """
        Constructor
        """
        self.logger = logging.getLogger(__name__)
        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'input.default')
            self.logger.info("Initializing a VMEC object from defaults in " \
                            + filename)
        else:
            self.logger.info("Initializing a VMEC object from file: " + filename)

        # Get MPI communicator:
        comm = MPI.COMM_WORLD
        self.fcomm = comm.py2f()

        self.VMEC = VMEC(input_file=filename, comm=self.fcomm, \
                             verbose=MPI.COMM_WORLD.rank==0)

        objstr = " for Vmec " + str(hex(id(self)))
        # nfp and stelsym are initialized by the Equilibrium constructor:
        Equilibrium.__init__(self)

        # For each VMEC input parameter in VMEC's fortran modules, create a Parameter:
        vi = self.VMEC.indata # Shorthand
        self.nfp.val = vi.nfp
        self.stelsym.val = not vi.lasym
        # It probably makes sense for a vmec object to have mpol and
        # ntor attributes independent of the boundary, since the
        # boundary may be a kind of surface that does not use the same
        # Fourier representation. But if the surface is a
        # SurfaceRZFourier, how then should the mpol and ntor of this
        # surface be coordinated with the mpol and ntor of the Vmec
        # object?
        self.mpol = Parameter(vi.mpol, min=1, name="mpol" + objstr, observers=self.reset)
        self.ntor = Parameter(vi.ntor, min=0, name="ntor" + objstr, observers=self.reset)
        self.delt = Parameter(vi.delt, min=0, max=1, name="delt" + objstr, observers=self.reset)
        self.tcon0 = Parameter(vi.tcon0, name="tcon0" + objstr, observers=self.reset)
        self.phiedge = Parameter(vi.phiedge, name="phiedge" + objstr, observers=self.reset)
        self.curtor = Parameter(vi.curtor, name="curtor" + objstr, observers=self.reset)
        self.gamma = Parameter(vi.gamma, name="gamma" + objstr, observers=self.reset)
        self.boundary = SurfaceRZFourier(nfp=self.nfp.val, stelsym=self.stelsym.val, \
                                      mpol=self.mpol.val, ntor=self.ntor.val)
        self.boundary.rc.set_observers(self.reset)
        self.boundary.zs.set_observers(self.reset)
        if not self.stelsym.val:
            self.boundary.rs.set_observers(self.reset)
            self.boundary.zc.set_observers(self.reset)
        # Transfer boundary shape data from fortran to the ParameterArray:
        for m in range(vi.mpol + 1):
            for n in range(-vi.ntor, vi.ntor + 1):
                self.boundary.get_rc(m, n).val = vi.rbc[101 + n, m]
                self.boundary.get_zs(m, n).val = vi.zbs[101 + n, m]
        # Handle a few variables that are not Parameters:
        self.ncurr = vi.ncurr
        self.free_boundary = bool(vi.lfreeb)
        self.need_to_run_code = True

        # Define a set of all the Parameters:
        self.params = self.boundary.params.union({self.mpol, self.ntor, \
                 self.delt, self.tcon0, self.phiedge, self.curtor, self.gamma})

        # Create the targets:
        self.aspect = Target(self.params, self.compute_aspect)
        self.volume = Target(self.params, self.compute_volume)
        self.iota_edge = Target(self.params, self.compute_iota_edge)
        self.iota_axis = Target(self.params, self.compute_iota_axis)

    def run(self):
        """
        Run VMEC, if needed.
        """
        if not self.need_to_run_code:
            self.logger.info("run() called but no need to re-run VMEC.")
            return
        self.logger.info("Preparing to run VMEC.")
        # Transfer values from Parameters to VMEC's fortran modules:
        vi = self.VMEC.indata
        vi.nfp = self.nfp.val
        vi.lasym = int(not self.stelsym.val)
        vi.delt = self.delt.val
        vi.phiedge = self.phiedge.val
        vi.curtor = self.curtor.val
        vi.gamma = self.gamma.val
        # VMEC does not allow mpol or ntor above 101:
        mpol_capped = np.min((self.boundary.mpol.val, 101))
        ntor_capped = np.min((self.boundary.ntor.val, 101))
        vi.mpol = mpol_capped
        vi.ntor = ntor_capped
        vi.rbc[:,:] = 0
        vi.zbs[:,:] = 0
        # Transfer boundary shape data from the ParameterArray:
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                vi.rbc[101 + n, m] = self.boundary.get_rc(m, n).val
                vi.zbs[101 + n, m] = self.boundary.get_zs(m, n).val
        
        self.VMEC.reinit()
        self.logger.info("Running VMEC.")
        self.VMEC.run()
        self.logger.info("VMEC run complete. Now loading output.")
        self.VMEC.load()
        self.logger.info("Done loading VMEC output.")
        self.need_to_run_code = False

    def compute_aspect(self):
        """
        Return the aspect ratio.
        """
        self.run()
        return self.VMEC.wout.aspect

    def compute_volume(self):
        """
        Return the volume.
        """
        self.run()
        return self.VMEC.wout.volume

    def compute_iota_axis(self):
        """
        Return the rotational transform on axis
        """
        self.run()
        return self.VMEC.wout.iotaf[0]

    def compute_iota_edge(self):
        """
        Return the rotational transform at the boundary
        """
        self.run()
        return self.VMEC.wout.iotaf[-1]

    def get_max_mn(self):
        """
        Look through the rbc and zbs data in fortran to determine the
        largest m and n for which rbc or zbs is nonzero.
        """
        max_m = 0
        max_n = 0
        for m in range(1, 101):
            for n in range(1, 101):
                if np.abs(self.VMEC.indata.rbc[101+n, m]) > 0 \
                        or np.abs(self.VMEC.indata.zbs[101+n, m]) > 0 \
                        or np.abs(self.VMEC.indata.rbs[101+n, m]) > 0 \
                        or np.abs(self.VMEC.indata.zbc[101+n, m]) > 0 \
                        or np.abs(self.VMEC.indata.rbc[101-n, m]) > 0 \
                        or np.abs(self.VMEC.indata.zbs[101-n, m]) > 0 \
                        or np.abs(self.VMEC.indata.rbs[101-n, m]) > 0 \
                        or np.abs(self.VMEC.indata.zbc[101-n, m]) > 0:
                    max_m = np.max((max_m, m))
                    max_n = np.max((max_n, n))
        # It may happen that mpol or ntor exceed the max_m or max_n
        # according to rbc/zbs. In this case, go with the larger
        # value.
        max_m = np.max((max_m, self.VMEC.indata.mpol))
        max_n = np.max((max_n, self.VMEC.indata.ntor))
        return (max_m, max_n)

    def reset(self):
        """
        This method observes all the parameters so we know to run VMEC
        if any parameters change.
        """
        self.logger.info("Resetting VMEC")
        self.need_to_run_code = True

    def finalize(self):
        """
        This subroutine deallocates arrays in VMEC so VMEC can be
        initialized again.
        """
        self.VMEC.finalize()

    def __repr__(self):
        """
        Print the object in an informative way.
        """
        return "Vmec instance " +str(hex(id(self))) + " (nfp=" + \
            str(self.nfp.val) + " mpol=" + \
            str(self.mpol.val) + " ntor=" + str(self.ntor.val) + ")"

    def _parse_namelist_var(self, varlist, var, default, min=np.NINF, max=np.Inf, \
                               new_name=None, parameter=True):
        """
        This method is used to streamline from_input_file(), and would
        not usually be called by users.
        """
        objstr = " for Vmec " + str(hex(id(self)))
        if var in varlist:
            val_to_use = varlist[var]
        else:
            val_to_use = default

        if new_name is None:
            name = var
        else:
            name = new_name

        if parameter:
            setattr(self, name, Parameter(val_to_use, min=min, max=max, \
                                              name=name + objstr, observers=self.reset))
        else:
            setattr(self, name, val_to_use)

#    @classmethod
#    def from_input_file(cls, filename):
#        """
#        Create an instance of the Vmec class based on settings that
#        are read in from a VMEC input namelist.
#        """
#        logger = logging.getLogger(__name__)
#        logger.info("Initializing a VMEC object from file: " + filename)
#        vmec = cls()
#
#        nml = f90nml.read(filename)
#        varlist = nml['indata']
#
#        vmec._parse_namelist_var(varlist, "nfp", 1, min=1)
#        vmec._parse_namelist_var(varlist, "mpol", 1, min=1)
#        vmec._parse_namelist_var(varlist, "ntor", 0, min=0)
#        vmec._parse_namelist_var(varlist, "delt", 0.7)
#        vmec._parse_namelist_var(varlist, "tcon0", 2.0)
#        vmec._parse_namelist_var(varlist, "phiedge", 1.0)
#        vmec._parse_namelist_var(varlist, "curtor", 0.0)
#        vmec._parse_namelist_var(varlist, "gamma", 0.0)
#        vmec._parse_namelist_var(varlist, "lfreeb", False, \
#                                    new_name="free_boundary", parameter=False)
#        vmec._parse_namelist_var(varlist, "ncurr", 1, parameter=False)
#
#        # Handle a few variables separately:
#        if "lasym" in varlist:
#            lasym = varlist["lasym"]
#        else:
#            lasym = False
#        vmec.stelsym = Parameter(not lasym)
#
#        vmec.boundary = SurfaceRZFourier(nfp=vmec.nfp.val, stelsym=vmec.stelsym.val, \
#                                      mpol=vmec.mpol.val, ntor=vmec.ntor.val)
#
#        return vmec
