# coding: utf-8

"""
The purpose of this module is to hide the details of the 
netCDF library to allow the user to replace scipy.io.netcdf_file 
with netCDF4's Dataset class if they need netCDF4 support.
"""

_all__ = ['netcdf_file']

import logging
from enum import Enum


class NetCDF(Enum):
    """Supported netCDF libraries."""
    netCDF4 = 0
    scipy = 1


logger = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
except ImportError as e:
    logger.debug(str(e))
    from scipy.io import netcdf_file as scipy_netcdf_file
    netcdf = NetCDF.scipy
else:
    netcdf = NetCDF.netCDF4


def netcdf_file(filename, mode='r', mmap=None, format='NETCDF3_CLASSIC'):
    r"""
    Wrapper that initializes and returns a scipy.io's netcdf_file or Dataset from NetCDF4.
    See  `scipy.io.netcdf_file <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.html#scipy.io.netcdf_file>`_.
    The output object should satisfy the interface of the object returned by scipy.io.netcdf_file, with methods: close(), flush(), sync(), createDimension(name, length), createVariable(name, type, dimensions), __enter__(), __exit__().

    Args:
        filename : string or file-like. File to read or write from or to.
        mode: {'r', 'w', 'a'}. read ('r'), write ('w') or append ('a') mode. Default: 'r'
        mmap: None or bool. Whether to mmap when reading the file.
        format: {1, 2, 'NETCDF3_CLASSIC', 'NETCDF3_64BIT_OFFSET'). Netcdf format. Only used when mode='w'.

    """
    if netcdf == NetCDF.scipy:
        if (format == 'NETCDF3_CLASSIC') or (format == 1):
            version = 1
        elif (format == 'NETCDF3_64BIT_OFFSET') or (format == 2):
            version = 2
        else:
            raise ValueError('scipy.io netcdf_file only supports "NETCDF3_CLASSIC" or "NETCDF3_64BIT_OFFSET" formats.')
        ret = scipy_netcdf_file(filename, mode, mmap, version)

    elif netcdf == NetCDF.netCDF4:
        ret = Dataset(filename, mode, format)
        ret.set_always_mask(False)
    return ret
