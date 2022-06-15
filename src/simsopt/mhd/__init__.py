# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

# Import MHD classes can fail if any of the required
# dependencies are not installed
import logging

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None 
    logger.debug(str(e))

if MPI is not None:
    # from .vmec import Vmec
    from .spec import Spec, Residue
    from .boozer import Boozer, Quasisymmetry
else:
    # Vmec = None
    Spec = None
    Residue = None
    Boozer = None
    Quasisymmetry = None
    logger.debug("mpi4py not installed."
                 "Not loading Spec and some other MHD modules.")

from .vmec import *
from .virtual_casing import *
from .vmec_diagnostics import *
