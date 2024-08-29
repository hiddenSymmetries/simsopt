# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains small utility functions and classes needed by *_core*
subpackage.
"""

import itertools
from numbers import Integral, Real, Number
from dataclasses import dataclass
from abc import ABCMeta
from weakref import WeakKeyDictionary
import numpy as np

from .types import RealArray
from simsoptpp import simd_alignment
ALIGNMENT = simd_alignment()

def isbool(val):
    """
    Test whether val is any boolean type, either the native python
    ``bool`` or numpy's ``bool_``.
    """
    return isinstance(val, (bool, np.bool_))


def isnumber(val):
    """
    Test whether val is any kind of number, including both native
    python types or numpy types.
    """
    return isinstance(val, Number)


class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """


def unique(inlist):
    """
    Given a list or tuple, return a list in which all duplicate
    entries have been removed. Unlike a python set, the order of
    entries in the original list will be preserved.  There is surely
    a faster algorithm than the one used here, but this function will
    not be used in performance-critical code.
    """

    outlist = []
    seen = set()
    for j in inlist:
        if j not in seen:
            outlist.append(j)
            seen.add(j)
    return outlist


@dataclass(frozen=True)
class ImmutableId:
    """
    Immutable class with a single attribute id to represent instance ids. Used
    in conjuction with InstanceCounterMeta metaclass to generate immutable
    instance ids starting with 1 for each of the different classes sublcassing
    InstanceCounterMeta
    """
    id: Integral


class InstanceCounterMeta(type):
    """
    Metaclass to make instance counter not share count with descendants

    Ref: https://stackoverflow.com/questions/8628123/counting-instances-of-a-class
    Credits: https://stackoverflow.com/users/3246302/ratiotile
    """
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._ids = itertools.count(1)


class RegisterMeta(type):
    """
    RegisterMeta class can be used to register functions with easy to identify
    names.

    Note:
        The class is not used anymore, but kept to explore the idea 3 explained
        below


    The functionality of RegisterMeta is explained with the Spec class
    defined in simsopt.mhd.spec module. Spec class, which is a subclass
    of Optimizable, implements two functions volume and iota, which are
    used by child Optimizables nodes.

    One could register the two functions of Spec class in a couple of ways.

    1. .. code-block:: python

            Spec.return_fn_map = {'volume': Spec.volume, 'iota': Spec.iota}

    2. .. code-block:: python

            Spec.volume = Spec.register_return_fn("volume")(Spec.volume)
            Spec.iota = Spec.register_return_fn("iota")(Spec.iota)

    3. TO BE IMPLEMENTED

       .. code-block:: python

            class Spec
                ...

                @register_return_fn("volume")
                def volume(self, ...):
                    ...

                @register_return_fn("iota")
                def iota(self, ...):
                    ...
    """
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.return_fn_map = {}

        def _register_return_fn(name):
            def inner_register(f):
                cls.return_fn_map[name] = f
                return f
            return inner_register

        cls.register_return_fn = _register_return_fn


def nested_lists_to_array(ll):
    """
    Convert a ragged list of lists to a 2D numpy array.  Any entries
    that are None are replaced by 0. This routine is useful for
    parsing fortran namelists that include 2D arrays using f90nml.

    Args:
        ll: A list of lists to convert.
    """
    mdim = len(ll)
    ndim = np.max([len(x) for x in ll])
    arr = np.zeros((mdim, ndim))
    for jm, l in enumerate(ll):
        for jn, x in enumerate(l):
            if x is not None:
                arr[jm, jn] = x
    return arr


class WeakKeyDefaultDict(WeakKeyDictionary):
    """
    A simple implementation of defaultdict that uses WeakKeyDictionary as its
    parent class instead of standard dictionary.
    """

    def __init__(self, default_factory=None, *args, **kwargs):
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory:
            self[key] = self.default_factory()
            return self[key]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except:
            return self.__missing__(key)


def parallel_loop_bounds(comm, n):
    """
    Split up an array [0, 1, ..., n-1] across an mpi communicator.  Example: n
    = 8, comm with size=2 will return (0, 4) on core 0, (4, 8) on core 1,
    meaning that the array is split up as [0, 1, 2, 3] + [4, 5, 6, 7].
    """

    if comm is None:
        return 0, n
    else:
        size = comm.size
        idxs = [i*n//size for i in range(size+1)]
        assert idxs[0] == 0
        assert idxs[-1] == n
        return idxs[comm.rank], idxs[comm.rank+1]
    
def align_and_pad(array, alignment=ALIGNMENT, dtype=np.dtype(np.float64)): 
    dims = array.ndim
    assert dims <= 2
    if array.shape[0] == 0:
        return array
    length = array.shape[1] if dims == 2 else len(array)
    padded = (length % (alignment//dtype.itemsize)) == 0

    if array.dtype == dtype:
        aligned = (array.ctypes.data % alignment) == 0
        contiguous = array.flags['C_CONTIGUOUS']
        if aligned and padded and contiguous:
            return array

    buf = allocate_aligned_and_padded_array(array.shape, alignment=alignment, dtype=dtype)
    if dims == 1:
        buf[:length] = array.astype(dtype)
    elif dims == 2:
        buf[:, :length] = array.astype(dtype)
    return buf

def allocate_aligned_and_padded_array(shape, alignment=ALIGNMENT, dtype=np.dtype(np.float64)):
    assert len(shape) <= 2
    if shape[0] == 0:
        return np.array([])
    if len(shape) == 1:
        padded_shape = (-shape[0]%(alignment//dtype.itemsize)+shape[0], )
        padded_size = padded_shape[0]
    elif len(shape) == 2:
        padded_shape = (shape[0], -shape[1]%(alignment//dtype.itemsize)+shape[1])
        padded_size = padded_shape[0] * padded_shape[1]
    buf = np.zeros(padded_size + alignment//dtype.itemsize, dtype=dtype)
    offset = (-buf.ctypes.data%alignment) // dtype.itemsize
    buf = buf[offset:offset+padded_size].reshape(padded_shape)
    assert (buf.ctypes.data%alignment) == 0
    return buf
