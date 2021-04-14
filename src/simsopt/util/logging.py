# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

import logging
import logging.config
from pathlib import Path

from monty.serialization import loadfn
try:
    from mpilogger import MPILogHandler
except:
    MPILogHander = None


def initialize_logging(filename=None, level=None, mpi=False):
    config_dict = loadfn(Path(__file__).parent / 'log_config.yaml')
    if filename:
        config_dict['handlers']['file_handler'].update({'filename': filename})
        config_dict['handlers']['mpi_file_handler'].update({'logfile': filename})
    if level:
        config_dict['handlers']['file_handler'].update({'level': level})
        config_dict['handlers']['mpi_file_handler'].update({'level': level})
    if mpi and MPILogHandler:
        config_dict['root']['handlers'].pop(2) # Remove file hander
    else:
        config_dict['root']['handlers'].pop(3) # Remove mpi hander
        del config_dict['handlers']['mpi_file_handler']

    logging.config.dictConfig(config_dict)

    if mpi and not MPILogHandler:
        logging.warning("mpilogger not installed. Not able to log MPI info")
