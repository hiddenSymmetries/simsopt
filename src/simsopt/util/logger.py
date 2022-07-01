# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

__all__ = ['initialize_logging']

import logging
import logging.config
from pathlib import Path

from ruamel.yaml import YAML
try:
    from mpi4py import MPI
    # from .mpi_logger import MPILogHandler
except:
    MPI = None

__all__ = ['initialize_logging']


def initialize_logging(filename: str = None,
                       level: str = None,
                       mpi: bool = False) -> None:
    """
    Initializes logging in a simple way for both serial and MPI jobs.
    The MPI logging uses MPILogger package.

    Args:
        filename: Name of file to store the logging info
        level: Logging level. Could be 'INFO', 'DEBUG', 'WARNING', etc.
        mpi: If True MPI logging is used provided mpi4py is installed.
    """
    yaml = YAML(typ='safe')
    config_dict = yaml.load(Path(__file__).parent / 'log_config.yaml')
    if filename:
        config_dict['handlers']['file_handler'].update({'filename': filename})
        config_dict['handlers']['mpi_file_handler'].update({'logfile': filename})
    if level:
        config_dict['handlers']['file_handler'].update({'level': level})
        config_dict['handlers']['mpi_file_handler'].update({'level': level})
    if mpi and MPI is not None:
        config_dict['root']['handlers'].pop(2)  # Remove file hander
    else:
        config_dict['root']['handlers'].pop(3)  # Remove mpi hander
        del config_dict['handlers']['mpi_file_handler']

    logging.config.dictConfig(config_dict)

    if mpi and MPI is None:
        logging.warning("mpi4py not installed. Not able to log MPI info")
