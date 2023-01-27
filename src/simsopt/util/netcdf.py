# coding: utf-8

"""
The purpose of this module is to hide the details of the 
netCDF library to allow the user to replace scipy.io.netcdf_file 
with netCDF4's Dataset class if they need netCDF4 support.
"""

_all__ = ['netcdf_file']

import logging
from enum import Enum
from abc import ABC, abstractmethod


class NetCDF(Enum):
    """Supported netCDF libraries."""
    netCDF4 = 0
    scipy = 1


logger = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
except ImportError as e:
    logger.debug(str(e))
    try:
        from scipy.io import netcdf_file as scipy_netcdf_file
    except ImportError as e:
        # this error is fatal
        logger.debug(str(e))
        raise e
    else:
        netcdf = NetCDF.scipy
else:
    netcdf = NetCDF.netCDF4


class AbstractNetCDF:
    """
    This class is not used, it merely defines the interface that
    the objects 'netcdf_file' from scipy and 'Dataset' from NetCDF4 
    both satisfy. 
    """
    @abstractmethod
    def __setattr__(self, attr, value):
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @abstractmethod
    def createDimension(self, name, length):
        pass

    @abstractmethod
    def createVariable(self, name, type, dimensions):
        pass

    @abstractmethod
    def flush(self):
        pass

    def sync(self):
        self.flush()


def netcdf_file(filename, mode='r', mmap=None, format='NETCDF3_CLASSIC'):
    """Wrapper around scipy.io's netcdf_file or Dataset from NetCDF4."""
    if netcdf == NetCDF.scipy:
        if format == 'NETCDF3_CLASSIC':
            version = 1
        elif format == 'NETCDF3_64BIT_OFFSET':
            version = 2
        else:
            raise ValueError('scipy.io netcdf_file only supports "NETCDF3_CLASSIC" or "NETCDF3_64BIT_OFFSET" formats.')
        ret = scipy_netcdf_file(filename, mode, mmap, version)

    elif netcdf == NetCDF.netCDF4:
        ret = Dataset(filename, mode, format)
        ret.set_always_mask(False)
    return ret
