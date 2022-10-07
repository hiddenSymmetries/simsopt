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
from simsoptpp import Curve   # To obtain pybind11 metaclass

__all__ = ['ObjectiveFailure']


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
class OptimizableMeta(InstanceCounterMeta, ABCMeta, type(Curve)):
    """
    Meta class for Optimizable class that works with pybind11. Here
    type(simsoptpp.Curve) is used to obtain the pybind11_type, which can
    be a parent class from py37
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


class DofLengthMismatchError(Exception):
    """
    Exception raised for errors where the length of supplied DOFs does
    not match with the length of free DOFs.
    Especially useful to prevent fully fixed DOFs from not raising Error
    and to prevent broadcasting of a single DOF
    """

    def __init__(self,
                 input_dof_length: Integral,
                 optim_dof_length: Integral,
                 message: str = None):
        if message is None:
            message = f"Input dof proerpty size, {input_dof_length}, does not " + \
                      f"match with Optimizable dof size {optim_dof_length}"
        super().__init__(message)


def finite_difference_steps(x: RealArray,
                            abs_step: Real = 1.0e-7,
                            rel_step: Real = 0.0
                            ) -> RealArray:
    """
    Determine an array of step sizes for calculating finite-difference
    derivatives, using absolute or relative step sizes, or a mixture
    thereof.

    For each element ``x_j`` of the state vector ``x``, the step size
    ``s_j`` is determined from

    ``s_j = max(abs(x_j) * rel_step, abs_step)``

    So, if you want to use the same absolute step size for all
    elements of ``x``, set ``rel_step = 0`` and set ``abs_step``
    positive. Or, if you want to use the same relative step size for
    all elements of ``x``, set ``abs_step = 0`` and ``rel_step``
    positive. If both ``abs_step`` and ``rel_step`` are positive, then
    ``abs_step`` effectively gives the lower bound on the step size.

    It is dangerous to set ``abs_step`` to exactly 0, since then if
    any elements of ``x`` are 0, the step size will be 0. In this
    situation, ``ValueError`` will be raised. It is preferable for
    ``abs_step`` to be a small positive value.

    For one-sided finite differences, the values of ``x_j`` used will
    be ``x_j + s_j``. For centered finite differences, the values of
    ``x_j`` used will be ``x_j + s_j`` and ``x_j - s_j``.

    This routine is used by :func:`simsopt._core.dofs.fd_jac()` and
    :func:`simsopt.solve.mpi.fd_jac_mpi()`.

    Args:
        x: The state vector at which you wish to compute the gradient
          or Jacobian. Must be a 1D array.
        abs_step: The absolute step size.
        rel_step: The relative step size.

    Returns:
        A 1D numpy array of the same size as ``x``, with each element
        being the step size used for each corresponding element of
        ``x``.
    """
    if abs_step < 0:
        raise ValueError('abs_step must be >= 0')
    if rel_step < 0:
        raise ValueError('rel_step must be >= 0')

    steps = np.max((np.abs(x) * rel_step, np.full(len(x), abs_step)),
                   axis=0)

    # If abs_step == 0 and any elements of x are 0, we could end up
    # with a step of size 0:
    if np.any(steps == 0.0):
        raise ValueError('Finite difference step size cannot be 0. ' \
                         'Increase abs_step.')

    return steps


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
