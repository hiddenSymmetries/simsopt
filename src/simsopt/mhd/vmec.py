# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
import numpy as np
from mpi4py import MPI
from monty.dev import requires

from simsopt.core import Optimizable, optimizable, SurfaceRZFourier, MpiPartition
try:
    from simsopt.mhd.vmec_f90wrap import VMEC # May need to edit this path.
    vmec_found = True
except ImportError as err:
    vmec_found = False
    print('Unable to load VMEC module, so some functionality will not be available.')
    print('Reason VMEC module was not loaded:')
    print(err)

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

@requires(vmec_found,
          "Running VMEC from simsopt requires VMEC python extension. "
          "Install the VMEC python extension from <link>.")
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
            logger.info("Initializing a VMEC object from defaults in " \
                            + filename)
        else:
            logger.info("Initializing a VMEC object from file: " + filename)

        # Get MPI communicator:
        if mpi is None:
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi
        comm = self.mpi.comm_groups
        self.fcomm = comm.py2f()

        self.VMEC = VMEC(input_file=filename, comm=self.fcomm, \
                             verbose=MPI.COMM_WORLD.rank==0, group=self.mpi.group)
        objstr = " for Vmec " + str(hex(id(self)))
        # nfp and stelsym are initialized by the Equilibrium constructor:
        #Equilibrium.__init__(self)

        # For each VMEC input parameter in VMEC's fortran modules, create an attribute
        vi = self.VMEC.indata # Shorthand
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
                                         stelsym=self.stelsym, mpol=self.mpol, ntor=self.ntor))
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
        return np.array([self.delt, self.tcon0, self.phiedge, self.curtor, self.gamma])

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
        vi = self.VMEC.indata
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
        vi.rbc[:,:] = 0
        vi.zbs[:,:] = 0
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

        self.VMEC.reinit()
        logger.info("Running VMEC.")
        self.VMEC.run()
        logger.info("VMEC run complete. Now loading output.")
        self.VMEC.load()
        logger.info("Done loading VMEC output.")
        self.need_to_run_code = False

    def aspect(self):
        """
        Return the plasma aspect ratio.
        """
        self.run()
        return self.VMEC.wout.aspect
        
    def volume(self):
        """
        Return the volume inside the VMEC last closed flux surface.
        """
        self.run()
        return self.VMEC.wout.volume
        
    def iota_axis(self):
        """
        Return the rotational transform on axis
        """
        self.run()
        return self.VMEC.wout.iotaf[0]

    def iota_edge(self):
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
            str(self.nfp) + " mpol=" + \
            str(self.mpol) + " ntor=" + str(self.ntor) + ")"

