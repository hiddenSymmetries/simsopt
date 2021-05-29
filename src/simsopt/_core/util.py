# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains small utility functions and classes needed by *_core*
subpackage.
"""

import itertools
from numbers import Integral, Number
from dataclasses import dataclass
from abc import ABCMeta

import numpy as np


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


#class OptimizableMeta(InstanceCounterMeta, RegisterMeta, ABCMeta):
class OptimizableMeta(InstanceCounterMeta, ABCMeta):
    """
    Meta class for Optimizable class
    """
    pass


class ObjectiveFailure(Exception):
    """
    Defines a custom exception used to indicate failure when
    evaluating the objective function. For example, if Vmec or Spec
    fail to converge, this exception will be thrown. The simsopt
    solvers will catch this specific exception (not others) and set
    the objective function to a large number.
    """
    pass
