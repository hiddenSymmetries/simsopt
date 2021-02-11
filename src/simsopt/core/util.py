# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains small utility functions and classes.
"""

import itertools
from typing import Union
from collections.abc import Sequence
from numbers import Integral, Real, Number
from dataclasses import dataclass
from abc import ABCMeta

import numpy as np
from nptyping import NDArray, Float, Int, Bool

Array = Union[Sequence, NDArray]
RealArray = Union[Sequence[Real], NDArray[Float]]
IntArray = Union[Sequence[Integral], NDArray[Int]]
StrArray = Union[Sequence[str], NDArray[str]]
BoolArray = Union[Sequence[bool], NDArray[Bool]]
Key = Union[Integral, str]


def isbool(val):
    """
    Test whether val is any boolean type, either the native python
    bool or numpy's bool_.
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

class InstanceCounterABCMeta(InstanceCounterMeta, ABCMeta):
    pass
