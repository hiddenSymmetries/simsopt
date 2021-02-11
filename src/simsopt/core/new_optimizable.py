# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides classes and functions that are useful for
setting up optimizable objects and objective functions.
"""

from __future__ import annotations

import abc
import copy
import logging
import types
import hashlib
from collections.abc import Callable, Hashable, Sequence, MutableSequence
from numbers import Real, Integral
from typing import Union, Any, Tuple

import numpy as np
import pandas as pd
#from mpi4py import MPI
from deprecated import deprecated
#from monty.json import MSONable

from .util import unique, Array, RealArray, StrArray, BoolArray, Key, IntArray
from .util import ImmutableId, InstanceCounterABCMeta

#logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

# Types
  # To denote arguments for accessing individual dof

class DOF:
    """
    A generalized class to represent an individual degrees of freedom
    associated with optimizable functions.
    """

    def __init__(self,
                 x: Real,
                 name: str,
                 free: bool = True,
                 lower_bound: Real = np.NINF,
                 upper_bound: Real = np.Inf):
        """

        :param name: Name of DOF for easy reference
        :param fixed: Flag to denote if the ODF is constrained?
        :param lower_bound: Minimum allowed value of DOF
        :param upper_bound: Maximum allowed value of DOF
        """
        self._x = x
        self.name = name
        self._free = free
        self._lb = lower_bound
        self._ub = upper_bound

    #def __hash__(self):
    #    return hash(":".join(map(str, [self.owner, self.name])))

    #@property
    #def extended_name(self) -> str:
    #    return ":".join(map(str, [self.owner, self.name]))

    def __repr__(self) -> str:
        return "DOF: {}, value = {}, fixed = {}, bounds = ({}, {})".format(
            self.name, self._x, not self._free, self._lb, self._ub)

    def __eq__(self, other):
        return all([self.name == other.name,
                   np.isclose(self._x, other._x),
                   self._free == other._free,
                   np.isclose(self._lb, other._lb),
                   np.isclose(self._ub, other._ub)])

    def is_fixed(self) -> bool:
        """
        Checks ifs the DOF is fixed

        Returns:
            True if DOF is fixed else False
        """
        return not self._free

    def is_free(self) -> bool:
        """
        Checks ifs the DOF is fixed

        Returns:
            True if DOF is fixed else False
        """
        return self._free

    def fix(self) -> None:
        """
        Denotes that the DOF needs to be fixed during optimization
        """
        self._free = False

    def unfix(self):
        """
        Denotes that the DOF can be varied during optimization
        """
        self._free = True

    @property
    def min(self) -> Real:
        """
        Minimum value of DOF allowed

        :return: Lower bound of DOF if not fixed else None
        """
        return self._lb if self._free else None

    @min.setter
    def min(self, lower_bound):
        self._lb = lower_bound

    @property
    def max(self) -> Real:
        """
        Maximum value of DOF allowed

        :return: Upper bound of DOF if not fixed else None
        """
        return self._ub if self._free else None

    @max.setter
    def max(self, upper_bound: Real) -> None:
        self._ub = upper_bound

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if self.is_fixed():
            raise TypeError("Updating state is forbidded for fixed DOF")
        if x > self.max or x < self.min:
            raise ValueError(
                "Input state is out of bounds for the DOF {}".format(self.name))
        self._x = x


class DOFs(pd.DataFrame):
    """
    Defines the (D)egrees (O)f (F)reedom(s) for optimization

    This class holds data related to the vector of degrees of freedom
    that have been combined from multiple optimizable objects.

    DataFrame Column index table:
    "Internal column, External name, Index
    "_x",             "x",            0
    "free",           "free",         1
    "_lb",            "lower_bounds", 2
    "_ub",            "upper_bounds", 3
    """
    def __init__(self,
                 x: RealArray = None, # To enable empty DOFs object
                 names: StrArray = None,
                 free: BoolArray = None,
                 lower_bounds: RealArray = None,
                 upper_bounds: RealArray = None):
        """
        Args:
            owners: Objects owning the dofs
            names: Names of the dofs
            x: Values of the dofs
            fixed: Array of boolean values denoting if the DOFs is fixed
            lower_bounds: Lower bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.NINF
            upper_bounds: Upper bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.inf
        """
        if x is None:
            x = np.array([])
        else:
            x = np.array(x, dtype=np.float)
        if names is None:
            names = ["x{}".format(i) for i in range(len(x))]

        if free is not None:
            free = np.array(free, dtype=np.bool_)
        else:
            free = np.full(len(x), True)

        if lower_bounds is not None:
            lb = np.array(lower_bounds, np.float)
        else:
            lb = np.full(len(x), np.NINF)

        if upper_bounds is not None:
            ub = np.array(upper_bounds, np.float)
        else:
            ub = np.full(len(x), np.inf)
        super().__init__(data={"_x": x, "free": free, "_lb": lb, "_ub": ub},
                         index=names)

    def fix(self, key: Key) -> None:
        if isinstance(key, str):
            self.loc[key, 'free'] = False
        else:
            self.iloc[key, 1] = False

    def unfix(self, key: Key) -> None:
        if isinstance(key, str):
            self.loc[key, 'free'] = True
        else:
            self.iloc[key, 1] = True

    def fix_all(self) -> None:
        self.free = self.free.apply(lambda x: False)

    def unfix_all(self) -> None:
        """
        Make vall DOFs variable
        Caution: Make sure the bounds are well defined
        """
        self.free = self.free.apply(lambda x: True)

    def any_free(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any free DOF is found
        """
        return self.free.any()

    def any_fixed(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any fixed DOF is found
        """
        return not self.free.all()

    def all_free(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if all DOFs are free to vary
        """
        return self.free.all()

    def all_fixed(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if all DOFs are fixed
        """
        return not self.free.any()

    @property
    def x(self) -> RealArray:
        return self.loc[self.free, "_x"].to_numpy()

    @x.setter
    def x(self, x: RealArray) -> None:
        """

        Args:
            x: Array of values to set x
            Note: This method blindly broadcasts a single value.
            So don't supply a single value unless you really desire
        """
        if len(self.free[self.free]) != len(x):
            raise ValueError  # To prevent fully fixed DOFs from not raising Error
        self.loc[self.free, "_x"] = x

    @property
    def full_x(self) -> RealArray:
        """
        Return all x even the fixed ones

        Returns:
            Pruned DOFs object without any fixed DOFs
        """
        return self._x.to_numpy()

    @property
    def reduced_len(self) -> Integral:
        """
        The standard len function returns the full length of DOFs.
        To get the number of free DOFs use DOFs._reduced_len method
        Returns:

        """
        return len(self.free[self.free])

    @property
    def lower_bounds(self) -> RealArray:
        """

        Returns:

        """
        return self.loc[self.free, "_lb"].to_numpy()

    @lower_bounds.setter
    def lower_bounds(self, lower_bounds: RealArray):
        self.loc[self.free, "_lb"] = lower_bounds

    @property
    def upper_bounds(self) -> RealArray:
        return self.loc[self.free, "_ub"].to_numpy()

    @upper_bounds.setter
    def upper_bounds(self, upper_bounds: RealArray):
        self.loc[self.free, "_ub"] = upper_bounds

    @property
    def bounds(self) -> Tuple[RealArray, RealArray]:
        return (self.lower_bounds, self.upper_bounds)

    def update_lower_bound(self, key: Key, val: Real):
        if isinstance(key, str):
            self.loc[key, "_lb"] = val
        else:
            self.iloc[key, 2] = val

    def update_upper_bound(self, key: Key, val: Real):
        if isinstance(key, str):
            self.loc[key, "_ub"] = val
        else:
            self.iloc[key, 3] = val

    def update_bounds(self, key: Key, val: Tuple[Real, Real]):
        if isinstance(key, str):
            self.loc[key, "_lb"] = val[0]
            self.loc[key, "_ub"] = val[1]
        else:
            self.iloc[key, 2] = val[0]
            self.iloc[key, 3] = val[1]


class Optimizable(Callable, Hashable, metaclass=InstanceCounterABCMeta):
    """
    Callable ABC that provides useful features for optimizable objects.

    The class defines methods that are used by simsopt to know 
    degrees of freedoms (DOFs) associated with the optimizable
    objects. All derived objects have to implement method f that defines
    the objective function. However the users are not expected to call
    *f* to get the objective function. Instead the users should call
    the Optimizable object directly.

    Optimizable and its subclasses define the optimization problem. The
    optimization problem can be thought of a DAG, which each instance of
    Optimizable being a vertex in the DAG. Each Optimizable object can
    take other Optimizable objects as inputs and through this container
    logic, the edges of the DAG are defined.

    Alternatively, the input Optimizable objects can be thought as parents
    to the current Optimizable object. In this approach, the last grand-child
    defines the optimization problem by embodying all the elements of the
    parents and grand-parents. Each DOF defined in a parent gets passed down
    to the children. And each call to child instance gets in turn propagated
    to the parent.

    Currently, this back and forth propagation of DOFs and function calls
    happens at run time.

    Note: __init__ takes instances of subclasses of Optimizable as
          input and modifies them to define the children for input objects
    """
    # Bharat's comment: I think we should deprecate set_dofs and get_dofs
    # in favor of 'dof' or 'state' or 'x' property name? I think if we go
    # this route, having a function to collect dofs for any arbitrary function
    # or class is needed. For functions, it is straight-forward, all the
    # arguments to the function are DOFs unless indicated somehow. If the
    # arguments are Sequences or numpy array, each individual element is a
    # DOF. The bounds # can be supplied similar to scipy.least_squares. Same
    # is the case with objects of a class.
    #
    # For subclasses of Optimizable, instead of making set_dofs and
    # get_dofs as abstract method we will define a new abstract method called
    # collect_dofs to collect dofs into a DOFs object.

    def __init__(self,
                 x0: RealArray = None,
                 names: StrArray = None,
                 fixed: BoolArray = None,
                 lower_bounds: RealArray = None,
                 upper_bounds: RealArray = None,
                 funcs_in: Sequence[Optimizable] = None):
        """
        Args:
            x0: Initial state (or initial values of DOFs)
            names: Human identifiable names for the DOFs
            fixed: Array describing whether the DOFs are free or fixed
            lower_bounds: Lower bounds for the DOFs
            upper_bounds: Upper bounds for the DOFs
            funcs_in: Optimizable objects to define the optimization problem
                      in conjuction with the DOFs
        """
        self._dofs = DOFs(x0,
                          names,
                          np.logical_not(fixed) if fixed is not None else None,
                          lower_bounds,
                          upper_bounds)

        # Generate unique and immutable representation for different
        # instances of same class
        self._id = ImmutableId(next(self.__class__._ids))
        self.name = self.__class__.__name__ + str(self._id.id)

        # Assign self as child to parents
        self.parents = funcs_in if funcs_in is not None else []
        for parent in self.parents:
            parent.add_child(self)

        # Obtain unique list of the ancestors
        self.ancestors = self.get_ancestors()

        # Compute the indices of all the DOFs
        dof_indices = [0]
        free_dof_size = 0
        full_dof_size = 0
        for opt in (self.ancestors + [self]):
            size = opt.local_dof_size
            free_dof_size += size
            full_dof_size += opt.local_full_dof_size
            dof_indices.append(free_dof_size)
        self.dof_indices = dict(zip(self.ancestors + [self],
                                    zip(dof_indices[:-1], dof_indices[1:])))
        self._free_dof_size = free_dof_size
        self._full_dof_size =  full_dof_size

        self._children = [] # This gets populated when the object is passed
                            # as argument to another Optimizable object
        self.new_x = True   # Set this True for dof setter and set it to False
                            # after evaluation of function if True

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        hash_str = hashlib.sha256(self.name.encode('utf-8')).hexdigest()
        return int(hash_str, 16) % 10**32  # 32 digit int as hash

    def __eq__(self, other: Optimizable) -> bool:
        """
        Checks the equality condition

        Args:
            other: Another object of subclass of Optimizable

        Returns: True only if both are the same objects.

        """
        #return (self.__class__ == other.__class__ and
        #        self._id.id == other._id.id)
        return self.name == other.name

    def __call__(self, x: RealArray = None):
        if x is not None:
            self.x = x
        if self.new_x:
            self._val = self.f()
            self.new_x = False
        return self._val

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        """
        Defines the callback function associated with the Optimizable subclass.

        Define the callback method in subclass of Optimizable. The function
        uses the state (x) stored self._dofs. To prevent hard-to-understand
        bugs, don't forget to use self.full_x.

        Args:
            *args:
            **kwargs:

        Returns:

        """

    def add_child(self, other: Optimizable) -> None:
        self._children.append(other)

    @property
    def full_dof_size(self) -> Integral:
        """
        Length of free DOFs associated with the Optimizable including those
        of parents
        """
        return self._full_dof_size

    @property
    def dof_size(self) -> Integral:
        """
        Length of DOFs associated with the Optimizable including those
        of parents
        Returns:

        """
        return self._free_dof_size

    @property
    def local_full_dof_size(self) -> Integral:
        return len(self._dofs)

    @property
    def local_dof_size(self) -> Integral:
        return self._dofs.reduced_len

    def _update_free_dof_size_indices(self) -> None:
        """
        Call the function to update the DOFs lengths for this instance and
        those of the children.

        Call whenever DOFs are fixed or unfixed. Recursively calls the
        function in children

        TODO: This is slow because it walks through the graph repeatedly
        TODO: Develop a faster scheme.
        TODO: Alternatively ask the user to call this manually from the end
        TODO: node after fixing/unfixing any DOF
        """
        dof_indices = [0]
        free_dof_size = 0
        for opt in (self.ancestors + [self]):
            size = opt.local_dof_size
            free_dof_size += size
            dof_indices.append(free_dof_size)
        self._free_dof_size = free_dof_size
        self.dof_indices = dict(zip(self.ancestors + [self],
                                    zip(dof_indices[:-1], dof_indices[1:])))

        # Update the reduced length of children
        for child in self._children:
            child._update_free_dof_size_indices()

    @property
    def dofs(self) -> RealArray:
        return np.concatenate([opt._dofs.x for
                               opt in (self.ancestors + [self])])

    @dofs.setter
    def dofs(self, x: RealArray) -> None:
        if list(self.dof_indices.values())[-1][-1] != len(x):
            raise ValueError
        for opt, indices in self.dof_indices.items():
            opt.local_dofs = x[indices[0]:indices[1]]
        self._set_new_x()

    @property
    def full_dofs(self) -> RealArray:
        return np.concatenate([opt._dofs.full_x for
                               opt in (self.ancestors + [self])])

    @property
    def local_dofs(self) -> RealArray:
        return self._dofs.x

    @local_dofs.setter
    def local_dofs(self, x: RealArray) -> None:
        if self.local_dof_size != len(x):
            raise ValueError
        self._dofs.loc[self._dofs.free, '_x'] = x
        self.new_x = True

    @property
    def local_full_dofs(self) -> RealArray:
        return self._dofs.full_x

    @property
    def state(self) -> RealArray:
        return self.dofs

    @state.setter
    def state(self, x: RealArray) -> None:
        self.dofs = x

    @property
    def full_state(self) -> RealArray:
        return self.full_dofs

    @property
    def local_state(self) -> RealArray:
        return self.local_dofs

    @local_state.setter
    def local_state(self, x: RealArray) -> None:
        self.local_dofs = x

    @property
    def local_full_state(self):
        return self.local_full_dofs

    @property
    def x(self) -> RealArray:
        return self.dofs

    @x.setter
    def x(self, x: RealArray) -> None:
        self.dofs = x

    @property
    def full_x(self) -> RealArray:
        return self.full_dofs

    @property
    def local_x(self) -> RealArray:
        return self.local_dofs

    @local_x.setter
    def local_x(self, x: RealArray) -> None:
        self.local_dofs = x

    @property
    def local_full_x(self):
        return self.local_full_dofs

    def _set_new_x(self):
        self.new_x = True
        for child in self._children:
            child._set_new_x()

    @property
    def bounds(self) -> Tuple[RealArray, RealArray]:
        return (self.lower_bounds, self.upper_bounds)

    @property
    def local_bounds(self) -> Tuple[RealArray, RealArray]:
        return self._dofs.bounds

    @property
    def lower_bounds(self) -> RealArray:
        opts = self.ancestors + [self]
        return np.concatenate([opt._dofs.lower_bounds for opt in opts])

    @property
    def local_lower_bounds(self) -> RealArray:
        return self._dofs.lower_bounds

    @property
    def upper_bounds(self) -> RealArray:
        opts = self.ancestors + [self]
        return np.concatenate([opt._dofs.upper_bounds for opt in opts])

    @property
    def local_upper_bounds(self) -> RealArray:
        return self._dofs.upper_bounds

    @local_upper_bounds.setter
    def local_upper_bounds(self, lub: RealArray) -> None:
        self._dofs.upper_bounds = lub

    def get(self, key: Key) -> Real:
        """
        Return a the value of degree of freedom specified by its name
        or by index.

        Even fixed dofs can be obtained individually.
        """
        if isinstance(key, str):
            return self._dofs.loc[key, '_x']
        else:
            return self._dofs.iloc[key, 0]

    def set(self, key: Key, new_val: Real) -> None:
        """
        Set a degree of freedom specified by its name or by index.

        Even fixed dofs can be set this way
        """
        if isinstance(key, str):
            self._dofs.loc[key, '_x'] = new_val
        else:
            self._dofs.iloc[key, 0] = new_val

    def is_fixed(self, key: Key) -> bool:
        """
        Tells if the dof specified with its name or by index is fixed
        """
        return not self.is_free(key)

    def is_free(self, key: Key) -> bool:
        """
        Tells if the dof specified with its name or by index is fixed
        """
        if isinstance(key, str):
            return self._dofs.loc[key, 'free']
        else:
            return self._dofs.iloc[key, 1]

    def fix(self, key: Key) -> None:
        """
        Set the fixed attribute for a given degree of freedom,
        specified by dof_str.
        """
        # TODO: Question: Should we use ifix similar to pandas' loc and iloc?
        # TODO: If key (str) is not found, it is silently ignored. Instead
        # TODO: raise a warning

        self._dofs.fix(key)
        self._update_free_dof_size_indices()

    def unfix(self, key: Key) -> None:
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.
        """
        # TODO: If key (str) is not found, it is silently ignored. Instead
        # TODO: raise a warning
        self._dofs.unfix(key)
        self._update_free_dof_size_indices()

    def fix_all(self) -> None:
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), True)
        self._dofs.fix_all()
        self._update_free_dof_size_indices()

    def unfix_all(self) -> None:
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), False)
        self._dofs.unfix_all()
        self._update_free_dof_size_indices()

    def get_ancestors(self) -> list[Optimizable]:
        ancestors = []
        for parent in self.parents:
            ancestors += parent.get_ancestors()
        ancestors += self.parents
        return list(dict.fromkeys(ancestors))


# TODO: Target class needs to be reimplemented to account for
# TODO: reimplementation of  Optimizable class
class Target(Optimizable):
    """
    Given an attribute of an object, which typically would be a
    @property, form a callable function that can be used as a target
    for optimization.
    """
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        super().__init__()

        # Attach a dJ function only if obj has one
        def dJ(self0):
            return getattr(self0.obj, 'd' + self0.attr)

        if hasattr(obj, 'd' + attr):
            self.dJ = types.MethodType(dJ, self)

    def f(self):
        # TODO: Implemnt the f to call self.obj.attr
        return getattr(self.obj, self.attr)

    @deprecated(version='0.0.2', reason="Call the object directly. Don't assume"
                                        " J method will be present.")
    def J(self):
        return getattr(self.obj, self.attr)

    #def dJ(self):
    #    return getattr(self.obj, 'd' + self.attr)

    # Bharat's comment: The following two needs to be better defined
    def get_dofs(self):
        return np.array([])

    def set_dofs(self, v):
        pass


def function_from_user(target):
    """
    Given a user-supplied "target" to be optimized, extract the
    associated callable function.
    """
    if callable(target):
        return target
    elif hasattr(target, 'J') and callable(target.J):
        return target.J
    else:
        raise TypeError('Unable to find a callable function associated '
                        'with the user-supplied target ' + str(target))

# TODO: make_optimizable function should be reimplemented to account for
# TODO: reimplementation of Optimizable class
def make_optimizable(obj):
    """
    Given any object, add attributes like fixed, mins, and maxs. fixed
    = False by default. Also, add the other methods of Optimizable to
    the object.
    """

    # If the object does not have a get_dofs() method, attach one,
    # assuming the object does not directly own any dofs.
    def get_dofs(self):
        return np.array([])
    def set_dofs(self, x):
        pass
    if not hasattr(obj, 'get_dofs'):
        obj.get_dofs = types.MethodType(get_dofs, obj)
    if not hasattr(obj, 'set_dofs'):
        obj.set_dofs = types.MethodType(set_dofs, obj)

    n = len(obj.get_dofs())
    if not hasattr(obj, 'dof_fixed'):
        obj.dof_fixed = np.full(n, False)
    if not hasattr(obj, 'mins'):
        obj.mins = np.full(n, np.NINF)
    if not hasattr(obj, 'maxs'):
        obj.maxs = np.full(n, np.Inf)

    # Add the following methods from the Optimizable class:
    #for method in ['index', 'get', 'set', 'get_fixed', 'set_fixed', 'all_fixed']:
    # See https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
    #setattr(obj, method, types.MethodType(getattr(Optimizable, method), obj))

    # New compact implementation
    method_list = [f for f in dir(Optimizable) if \
            callable(getattr(Optimizable, f)) and not f.startswith("__")]
    for f in method_list:
        if not hasattr(obj, f) and f not in ('get_dofs', 'set_dofs'):
            setattr(obj, f, types.MethodType(getattr(Optimizable, f), obj))

    return obj
