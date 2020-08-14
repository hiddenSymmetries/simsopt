"""
This module provides a class that handles the VMEC equilibrium code.
"""

import numpy as np
from simsopt import *
#from FortranNamelist import NamelistFile
#import f90nml
import logging

class Vmec(Equilibrium):
    """
    This class represents the VMEC equilibrium code.
    """
    def __init__(self):
        """
        Constructor
        """
        objstr = " for Vmec " + str(hex(id(self)))
        # nfp and stelsym are initialized by the Equilibrium constructor:
        Equilibrium.__init__(self)
        self.mpol = Parameter(1, min=1, name="mpol" + objstr, observers=self.reset)
        self.ntor = Parameter(0, min=0, name="ntor" + objstr, observers=self.reset)
        self.delt = Parameter(0.7, min=0, max=1, name="delt" + objstr, observers=self.reset)
        self.tcon0 = Parameter(2.0, name="tcon0" + objstr, observers=self.reset)
        self.phiedge = Parameter(1.0, name="phiedge" + objstr, observers=self.reset)
        self.curtor = Parameter(0.0, name="curtor" + objstr, observers=self.reset)
        self.gamma = Parameter(0.0, name="gamma" + objstr, observers=self.reset)
        self.boundary = SurfaceRZFourier(nfp=self.nfp.val, stelsym=self.stelsym.val, \
                                      mpol=self.mpol.val, ntor=self.ntor.val)
        # Handle a few variables that are not Parameters:
        self.ncurr = 1
        self.free_boundary = False
        self.need_to_run_code = True

    def reset(self):
        """
        This method observes all the parameters so we know to run VMEC
        if any parameters change.
        """
        logger = logging.getLogger(__name__)
        logger.info("Resetting VMEC")
        self.need_to_run_code = True

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
