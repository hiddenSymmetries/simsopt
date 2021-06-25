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
    logger.warning(str(e))

if MPI is not None:
    from .vmec import Vmec  # , vmec_found
    from .spec import Spec, Residue  # , spec_found
    from .boozer import Boozer, Quasisymmetry  # , boozer_found
else:
    Vmec = None
    Spec = None
    Residue = None
    Boozer = None
    Quasisymmetry = None
    logger.WARNING("mpi4py not installed. Not loading Vmec, Spec and other MHD modules.")

#try:
#    import vmec
#except BaseException as err:
#    print('Unable to load VMEC module, so some functionality will not be available.')
#    print('Reason VMEC module was not loaded:')
#    print(err)

#try:
#    from .vmec import Vmec
#    vmec_found = True
#except ImportError as err:
#    vmec_found = False
#    print('Unable to load VMEC module, so some functionality will not be available.')
#    print('Reason VMEC module was not loaded:')
#    print(err)

#if vmec_found:
#    from .vmec import Vmec
