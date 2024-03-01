# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
Provides graph based Optimizable class, whose instances can be used to
build an optimization problem in a graph like manner.
"""

from __future__ import annotations

import weakref
import hashlib
from collections.abc import Callable as ABC_Callable, Hashable
from numbers import Real, Integral
from typing import Union, Tuple, Dict, Callable, Sequence, List
from functools import lru_cache
import logging
import json
from pathlib import Path
from fnmatch import fnmatch

import numpy as np
from monty.json import MSONable, MontyDecoder, MontyEncoder
from monty.io import zopen

from .dev import SimsoptRequires
from .types import RealArray, StrArray, BoolArray, Key
from .util import ImmutableId, OptimizableMeta, WeakKeyDefaultDict, \
    DofLengthMismatchError
from .derivative import derivative_dec

try:
    import networkx as nx
except ImportError:
    nx = None
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    pygraphviz = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

log = logging.getLogger(__name__)

__all__ = ['Optimizable', 'make_optimizable', 'load', 'save',
           'OptimizableSum', 'ScaledOptimizable']


class DOFs:
    """
    Defines the (D)egrees (O)f (F)reedom(s) associated with optimization

    This class holds data related to the degrees of freedom
    associated with an Optimizable object. To access the data stored in
    the DOFs class, use the labels shown shown in the table below.

    =============  =============
    External name  Internal name
    =============  =============
    x              _x
    free           _free
    lower_bounds   _lb
    upper_bounds   _ub
    names          _names
    =============  =============

    The class implements the external name column properties in the above
    table as properties. Additional methods to update bounds, fix/unfix DOFs,
    etc. are also defined.
    """
    __slots__ = ["_x", "_free", "_lb", "_ub", "_names"]

    def __init__(self,
                 x: RealArray = None,  # To enable empty DOFs object
                 names: StrArray = None,
                 free: BoolArray = None,
                 lower_bounds: RealArray = None,
                 upper_bounds: RealArray = None) -> None:
        """
        Args:
            x: Numeric values of the DOFs
            names: Names of the dofs
            free: Array of boolean values denoting if the DOFs is are free.
                  False values implies the corresponding DOFs are fixed
            lower_bounds: Lower bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.NINF
            upper_bounds: Upper bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.inf
        """
        if x is None:
            x = np.array([])
        else:
            x = np.asarray(x, dtype=np.double)

        if names is None:
            names = [f"x{i}" for i in range(len(x))]
        assert (len(np.unique(names)) == len(names))  # DOF names should be unique

        if free is None:
            free = np.full(len(x), True)
        else:
            free = np.asarray(free, dtype=np.bool_)

        if lower_bounds is None:
            lower_bounds = np.full(len(x), np.NINF)
        else:
            lower_bounds = np.asarray(lower_bounds, np.double)

        if upper_bounds is None:
            upper_bounds = np.full(len(x), np.inf)
        else:
            upper_bounds = np.asarray(upper_bounds, np.double)

        assert (len(x) == len(free) == len(lower_bounds) == len(upper_bounds) \
                == len(names))
        self._x = x
        self._free = free
        self._lb = lower_bounds
        self._ub = upper_bounds
        self._names = list(names)

    def __len__(self):
        return len(self._free)

    def fix(self, key: Key) -> None:
        """
        Fixes the specified DOF

        Args:
            key: Key to identify the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        self._free[key] = False

    def unfix(self, key: Key) -> None:
        """
        Unfixes the specified DOF

        Args:
            key: Key to identify the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        self._free[key] = True

    def all_free(self) -> bool:
        """
        Checks if all DOFs are allowed to be varied

        Returns:
            True if all DOFs are free to changed
        """
        return self._free.all()

    def all_fixed(self) -> bool:
        """
        Checks if all the DOFs are fixed

        Returns:
            True if all DOFs are fixed
        """
        return not self._free.any()

    @property
    def free_status(self) -> BoolArray:
        return self._free

    def get(self, key: Key) -> Real:
        """
        Get the value of specified DOF. Even fixed DOFs can
        be obtained with this method

        Args:
        key: Key to identify the DOF
        Returns:
            Value of the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        return self._x[key]

    def set(self, key: Key, val: Real):
        """
        Modify the value of specified DOF. Even fixed DOFs can
        modified with this method

        Args:
        key: Key to identify the DOF
        val: Value of the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        self._x[key] = val

    def is_free(self, key: Key) -> bool:
        """
        Get the status of the specified DOF.

        Args:
        key: Key to identify the DOF
        Returns:
            Status of the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        return self._free[key]

    def fix_all(self) -> None:
        """
        Fixes all the DOFs
        """
        self._free.fill(False)

    def unfix_all(self) -> None:
        """
        Makes all DOFs variable
        Caution: Make sure the bounds are well defined
        """
        self._free.fill(True)

    def any_free(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any free DOF is found, else False
        """
        return self._free.any()

    def any_fixed(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any fixed DOF is found, else False
        """
        return not self._free.all()

    @property
    def x(self) -> RealArray:
        """

        Returns:
            The values of the free DOFs.
        """
        return self._x[self._free]

    @x.setter
    def x(self, x: RealArray) -> None:
        """
        Update the values of the free DOFs with the supplied values

        Args:
            x: Array of new DOF values
               (word of caution: This setter blindly broadcasts a single value.
               So don't supply a single value unless you really desire.)
        """
        # To prevent fully fixed DOFs from not raising Error
        # And to prevent broadcasting of a single DOF
        if self.reduced_len != len(x):
            raise DofLengthMismatchError(len(x), self.reduced_len)
        self._x[self._free] = np.asarray(x, dtype=np.double)

    @property
    def full_x(self) -> RealArray:
        """
        Return all x even the fixed ones

        Returns:
            The values of full DOFs without any restrictions
        """
        return self._x

    @full_x.setter
    def full_x(self, x: RealArray) -> None:
        """
        Update the values of the all DOFs with the supplied values

        Args:
            x: Array of new DOF values
        .. warning::
               Even fixed DOFs are assinged
        """
        # To prevent broadcasting of a single DOF
        if len(self._x) != len(x):
            raise DofLengthMismatchError(len(x), len(self._x))
        self._x = np.asarray(x, dtype=np.double)

    @property
    def reduced_len(self) -> Integral:
        """
        The number of free DOFs.

        The standard len function returns the full length of DOFs.

        Returns:
            The number of free DOFs
        """
        return len(self._free[self._free])

    @property
    def lower_bounds(self) -> RealArray:
        """
        Lower bounds of the DOFs

        Returns:
            Lower bounds of the DOFs
        """
        return self._lb[self._free]

    @property
    def full_lower_bounds(self) -> RealArray:
        return self._lb

    @lower_bounds.setter
    def lower_bounds(self, lower_bounds: RealArray) -> None:
        """

        Args:
            lower_bounds: Lower bounds of the DOFs
        """
        # To prevent fully fixed DOFs from not raising Error
        # and to prevent broadcasting of a single DOF
        if self.reduced_len != len(lower_bounds):
            raise DofLengthMismatchError(len(lower_bounds), self.reduced_len)
        self._lb[self._free] = np.asarray(lower_bounds, dtype=np.double)

    @property
    def upper_bounds(self) -> RealArray:
        """

        Returns:
            Upper bounds of the DOFs
        """
        return self._ub[self._free]

    @property
    def full_upper_bounds(self) -> RealArray:
        return self._ub

    @upper_bounds.setter
    def upper_bounds(self, upper_bounds: RealArray) -> None:
        """

        Args:
            upper_bounds: Upper bounds of the DOFs
        """
        # To prevent fully fixed DOFs from not raising Error
        # and to prevent broadcasting of a single DOF
        if self.reduced_len != len(upper_bounds):
            raise DofLengthMismatchError(len(upper_bounds), self.reduced_len)
        self._ub[self._free] = np.asarray(upper_bounds, dtype=np.double)

    @property
    def bounds(self) -> Tuple[RealArray, RealArray]:
        """

        Returns:
            (Lower bounds list, Upper bounds list)
        """
        return (self.lower_bounds, self.upper_bounds)

    @property
    def full_bounds(self) -> Tuple[RealArray, RealArray]:
        return (self.full_lower_bounds, self.full_upper_bounds)

    def update_lower_bound(self, key: Key, val: Real) -> None:
        """
        Updates the lower bound of the specified DOF to the given value

        Args:
            key: DOF identifier
            val: Numeric lower bound of the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        self._lb[key] = val

    def update_upper_bound(self, key: Key, val: Real) -> None:
        """
        Updates the upper bound of the specified DOF to the given value

        Args:
            key: DOF identifier
            val: Numeric upper bound of the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        self._ub[key] = val

    def update_bounds(self, key: Key, val: Tuple[Real, Real]) -> None:
        """
        Updates the bounds of the specified DOF to the given value

        Args:
            key: DOF identifier
            val: (lower, upper) bounds of the DOF
        """
        if isinstance(key, str):
            key = self._names.index(key)
        self._lb[key] = val[0]
        self._ub[key] = val[1]

    @property
    def names(self):
        """

        Returns:
            string identifiers of the DOFs
        """
        @lru_cache()
        def red_names(free):
            rnames = []
            for i, f in enumerate((free)):
                if f:
                    rnames.append(self._names[i])
            return rnames
        return red_names(tuple(self._free))

    @property
    def full_names(self):
        return self._names


class Optimizable(ABC_Callable, Hashable, MSONable, metaclass=OptimizableMeta):
    """
    Experimental callable ABC that provides lego-like optimizable objects
    that can be used to partition the optimization problem into a graph.

    The class provides many features that simplify defining the optimization
    problem.

    1. Optimizable and its subclasses define the optimization problem. The
       optimization problem can be thought of as a directed acycling graph (DAG),
       with each instance of Optimizable being a vertex (node) in the DAG.
       Each Optimizable object can take other Optimizable objects as inputs and
       through this container logic, the edges of the DAG are defined.

       Alternatively, the input Optimizable objects can be thought of as parents
       to the current Optimizable object. In this approach, the last grand-child
       defines the optimization problem by embodying all the elements of the
       parents and grand-parents.

       Each call to child instance gets in turn propagated to the parent. In this
       way, the last child acts as the final optimization problem to be solved.
       For an example of the final optimization node, refer to
       simsopt.objectives.least_squares.LeastSquaresProblem

    2. The class automatically partitions degrees of freedoms (DOFs) of
       the optimization problem to the associated Optimizable nodes. Each DOF
       defined in a parent gets passed down to the children as a needed DOF for
       the child. So a DOF needed by parent node can be given as an input to
       the methods in the child node. Any of the DOFs could be fixed in which
       case, it should be removed as an argument to the call-back
       function from the final Optimizable node.

    3. The class implements a callable hook that provides minimal caching.
       All derived classes have to register methods that return objective function
       type values. This is done by implementing the following class attribute
       in the class definition:
       .. code-block:: python

           return_fn_map = {'name1': method1, 'name2': method, ...}

       The Optimizable class maintains the list of return functions needed by each
       of the calling Optimizable objects either during child initialization or
       later using the provided methods. The calling optimizable object could then
       call the Optimizable object directly using the `__call__` hook or could
       call the individual methods.

       This back and forth propagation of DOFs partitioning and function
       calls happens dynamically.

    4. The class is hashable and the names of the instances are unique. So
       instances of Optimizable class can be used as keys.

    Note:
        1. If the Optimizable object is called using the `__call__` hook, make
           sure to supply the argument `child=self`

        2. __init__ takes instances of subclasses of Optimizable as
           input and modifies them to add the current object as a child for
           input objects. The return fns of the parent object needed by the child
           could be specified by using `opt_return_fns` argument
    """
    return_fn_map: Dict[str, Callable] = NotImplemented

    def __init__(self,
                 x0: RealArray = None,
                 names: StrArray = None,
                 fixed: BoolArray = None,
                 lower_bounds: RealArray = None,
                 upper_bounds: RealArray = None,
                 external_dof_setter: Callable[..., None] = None,
                 depends_on: Sequence[Optimizable] = None,
                 opt_return_fns: Sequence[Sequence[str]] = None,
                 funcs_in: Sequence[Callable[..., Union[RealArray, Real]]] = None,
                 **kwargs):
        """
        Args:
            x0: Initial state (or initial values of DOFs)
            names: Human identifiable names for the DOFs
            fixed: Array describing whether the DOFs are free or fixed
            lower_bounds: Lower bounds for the DOFs
            upper_bounds: Upper bounds for the DOFs
            external_dof_setter: Function used by derivative classes to
                handle DOFs outside of the _dofs object.
                Mainly used when the DOFs are primarily handled by C++ code.
                In that case, for all intents and purposes, the _dofs is a
                duplication of the DOFs stored elsewhere. In such cases, _dofs
                is used to handle the dof partitioning, but external dofs are
                used for computation of the objective function.
            depends_on: Sequence of Optimizable objects on which the current
                Optimizable object depends on to define the optimization
                problem in conjuction with the DOFs. If the optimizable problem
                can be thought of as a direct acyclic graph based on
                dependencies, the optimizable objects
                supplied with depends_on act as parent nodes to the current
                Optimizable object in such an optimization graph
            opt_return_fns: Specifies the return value for each of the
                Optimizable object. Used in the case, where Optimizable object
                can return different return values. Typically return values are
                computed by different functions defined in the Optimizable
                object. The return values are selected by choosing the
                functions. To know the various return values, use the
                Optimizable.get_return_fn_names function. If the list is
                empty, default return value is used. If the Optimizable
                object can return multiple values, the default is the array
                of all possible return values.
            funcs_in: Instead of specifying depends_on and opt_return_fns, specify
                the methods of the Optimizable objects directly. The parent
                objects are identified automatically. Doesn't work with
                funcs_in with a property decorator
        """
        self._dofs = DOFs(x0,
                          names,
                          np.logical_not(fixed) if fixed is not None else None,
                          lower_bounds,
                          upper_bounds)
        self.local_dof_setter = external_dof_setter

        # Generate unique and immutable representation for different
        # instances of same class
        self._id = ImmutableId(next(self.__class__._ids))
        self.name = self.__class__.__name__ + str(self._id.id)
        hash_str = hashlib.sha256(self.name.encode('utf-8')).hexdigest()
        self._hash = int(hash_str, 16) % 10**32  # 32 digit int as hash
        self._children = set()  # This gets populated when the object is passed
        # as argument to another Optimizable object
        self.return_fns = WeakKeyDefaultDict(list)  # Store return fn's required by each child

        # Assign self as child to parents
        funcs_in = list(funcs_in) if funcs_in is not None else []
        depends_on = list(depends_on) if depends_on is not None else []
        assert (not ((len(funcs_in) > 0) and (len(depends_on) > 0)))

        def binder(fn, inst):
            def func(*args, **kwargs):
                return fn(inst, *args, **kwargs)
            return func

        if len(depends_on):
            self.parents = depends_on
            for i, parent in enumerate(self.parents):
                parent._add_child(self)
                return_fns = opt_return_fns[i] if opt_return_fns else []
                try:
                    if not len(return_fns) and len(parent.return_fn_map.values()):
                        return_fns = parent.return_fn_map.values()
                except:
                    pass
                for fn in return_fns:
                    parent.add_return_fn(self, fn)
                    funcs_in.append(binder(fn, parent))
        else:  # Process funcs_in (Assumes depends_on is empty)
            for fn in funcs_in:
                opt_in = fn.__self__
                depends_on.append(opt_in)
                opt_in.add_return_fn(self, fn.__func__)
            self.parents = list(dict.fromkeys(depends_on))
            for i, parent in enumerate(self.parents):
                parent._add_child(self)

        self.funcs_in = funcs_in

        # Obtain unique list of the ancestors
        self.ancestors = self._get_ancestors()

        # Compute the indices of all the DOFs
        self._update_free_dof_size_indices()
        self._update_full_dof_size_indices()
        # Inform the object that it doesn't have valid cache
        self._set_new_x()
        log.debug(f"Unused arguments for {self.__class__} are {kwargs}")
        super().__init__()

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Optimizable) -> bool:
        """
        Checks the equality condition

        Args:
            other: Another object of subclass of Optimizable

        Returns: True only if both are the same objects.

        """
        return self.name == other.name

    def __call__(self, x: RealArray = None, *args, child=None, **kwargs):
        if x is not None:
            self.x = x
        return_fn_map = self.__class__.return_fn_map

        if child:
            return_fns = self.return_fns[child] if self.return_fns[child] else \
                return_fn_map.values()
        else:
            return_fns = return_fn_map.values()

        result = []
        for fn in return_fns:
            result.append(fn(self, *args, **kwargs))

        return result if len(result) > 1 else result[0]

    def get_return_fn_names(self) -> List[str]:
        """
        Return the names of the functions that could be used as objective
        functions.

        Returns:
            List of function names that could be used as objective functions
        """
        return list(self.__class__.return_fn_map.keys())

    def add_return_fn(self, child: Optimizable, fn: Union[str, Callable]) -> None:
        """
        Add return function to the list of the return functions called by
        the child Optimizable object

        Args:
            child: an Optimizable object that is direct dependent of the current
                Optimizable instance
            fn: method of the Optimizable object needed by the child
        """
        self._add_child(child)

        if isinstance(fn, str):
            fn = self.__class__.return_fn_map[fn]
        self.return_fns[child].append(fn)

    def get_return_fns(self, child: Optimizable) -> List[Callable]:
        """
        Gets return functions from this Optimizable object used by the child
        Optimizable object

        Args:
            child: Dependent Optimizable object

        Returns:
            List of methods that return a value when the current Optimizable
            object is called from the child
        """
        return self.return_fns[child]

    def get_return_fn_list(self) -> List[List[Callable]]:
        """
        Gets return functions from this Optimizable object used by all the child
        Optimizable objects

        Returns:
            List of methods that return a value when the current Optimizable
            object is called from the children.
        """
        return list(self.return_fns.values())

    def get_parent_return_fns_list(self) -> List[List[Callable]]:
        """
        Get a list of the funcs returned by the parents as list of lists

        Returns:
            The funcs returned by all the parents of the Optimizable object
        """
        return_fn_list = []
        for parent in self.parents:
            return_fn_list.append(parent.get_return_fns(self))
        return return_fn_list

    @property
    def parent_return_fns_no(self) -> int:
        """
        Compute the total number of the return funcs of all the parents
        of the Optimizable object

        Returns:
            The total number of the return funcs  of the Optimizable object's
            parents.
        """
        return_fn_no = 0
        for parent in self.parents:
            return_fn_no += len(parent.get_return_fns(self))
        return return_fn_no

    def _add_child(self, child: Optimizable) -> None:
        """
        Adds another Optimizable object as child. All the
        required processing of the dependencies is done in the child node.
        This method is used mainly to maintain 2-way link between parent
        and child.

        Args:
            child: Direct dependent (child) of the Optimizable object
        """
        weakref_child = weakref.ref(child)
        if weakref_child not in self._children:
            self._children.add(weakref_child)

    def _remove_child(self, other: Optimizable) -> None:
        """
        Remove the specific Optimizable object from the children list.

        Args:
            child: Direct dependent (child) of the Optimizable object
        """
        weakref_other = weakref.ref(other)
        self._children.remove(weakref_other)
        if other in self.return_fns:
            del self.return_fns[other]

    def add_parent(self, index: int, other: Optimizable) -> None:
        """
        Adds another Optimizable object as parent at specified index.

        Args:
            int: Index of the parent's list
            other: Another Optimizable object to be added as parent
        """
        if other not in self.parents:
            self.parents.insert(index, other)
            other._add_child(self)
            # self.ancestors = self._get_ancestors()
            self._update_full_dof_size_indices()  # Updates ancestors as well
            self._update_free_dof_size_indices()
            self._set_new_x()
        else:
            log.debug("The given Optimizable object is already a parent")

    def append_parent(self, other: Optimizable) -> None:
        """
        Appends another Optimizable object to parents list

        Args:
            other: New parent Optimizable object
        """
        self.add_parent(len(self.parents), other)
        # if other not in self.parents:
        #     self.parents.append(other)
        #     other._add_child(self)
        #     self.ancestors = self._get_ancestors()
        #     self._update_free_dof_size_indices()
        #     self._update_full_dof_size_indices()
        #     self._set_new_x()
        # else:
        #     log.debug("The given Optimizable object is already a parent")

    def pop_parent(self, index: int = -1) -> Optimizable:
        """
        Removes the parent Optimizable object at specified index.

        Args:
            index: Index of the list of the parents

        Returns:
            The removed parent Optimizable object
        """
        discarded_parent = self.parents.pop(index)
        discarded_parent._remove_child(self)
        # self.ancestors = self._get_ancestors()
        self._update_full_dof_size_indices()  # Updates ancestors as well
        self._update_free_dof_size_indices()
        self._set_new_x()

        return discarded_parent

    def remove_parent(self, other: Optimizable):
        """
        Removes the specified Optimizable object from the list of parents.

        Args:
            other: The Optimizable object to be removed from the list of parents
        """
        self.parents.remove(other)
        other._remove_child(self)
        # self.ancestors = self._get_ancestors()
        self._update_full_dof_size_indices()  # updates ancestors as well
        self._update_free_dof_size_indices()
        self._set_new_x()

    def _get_ancestors(self) -> list[Optimizable]:
        """
        Get all the ancestors of the current Optimizable object

        Returns:
            List of Optimizable objects that are parents of current
            Optimizable objects
        """
        ancestors = []
        for parent in self.parents:
            ancestors += parent.ancestors
        ancestors += self.parents
        return sorted(dict.fromkeys(ancestors), key=lambda a: a.name)

    def _update_free_dof_size_indices(self) -> None:
        """
        Updates the DOFs lengths for the Optimizable object as well as
        those of the descendent (dependent) Optimizable objects.

        Call this function whenever DOFs are fixed or unfixed or when parents
        are added/deleted. Recursively calls the same function in children
        """
        # TODO: This is slow because it walks through the graph repeatedly
        # TODO: Develop a faster scheme.
        # TODO: Alternatively ask the user to call this manually from the end
        # TODO: node after fixing/unfixing any DOF
        dof_indices = [0]
        free_dof_size = 0
        for opt in (self.ancestors + [self]):
            size = opt.local_dof_size
            free_dof_size += size
            dof_indices.append(free_dof_size)
        self._free_dof_size = free_dof_size
        self.dof_indices = dict(zip(self.ancestors + [self],
                                    zip(dof_indices[:-1], dof_indices[1:])))

        # Update the reduced dof length of children
        for weakref_child in self._children:
            child = weakref_child()
            if child is not None:
                child._update_free_dof_size_indices()

    def _update_full_dof_size_indices(self) -> None:
        """
        Updates the full DOFs lengths for this instance and
        those of the children. Updates the ancestors attribute as well.

        Call this function whenever parents are added or removed. Recursively
        calls the same function in children.
        """

        # TODO: This is slow because it walks through the graph repeatedly
        # TODO: Develop a faster scheme.
        # TODO: Alternatively ask the user to call this manually from the end
        # TODO: node after fixing/unfixing any DOF
        full_dof_size = 0
        self.ancestors = self._get_ancestors()
        for opt in (self.ancestors + [self]):
            full_dof_size += opt.local_full_dof_size
        self._full_dof_size = full_dof_size

        # Update the full dof length of children
        for weakref_child in self._children:
            child = weakref_child()
            if child is not None:
                child._update_full_dof_size_indices()

    @property
    def full_dof_size(self) -> Integral:
        """
        Total number of all (free and fixed) DOFs associated with the
        Optimizable object as well as parent Optimizable objects.
        """
        return self._full_dof_size

    @property
    def dof_size(self) -> Integral:
        """
        Total number of free DOFs associated with the Optimizable object
        as well as parent Optimizable objects.
        """
        return self._free_dof_size

    @property
    def local_full_dof_size(self) -> Integral:
        """
        Number of all (free and fixed) DOFs associated with the Optimizable
        object.

        Returns:
            Total number of free and fixed DOFs associated with the Optimizable
            object.
        """
        return len(self._dofs)

    @property
    def local_dof_size(self) -> Integral:
        """
        Number of free DOFs associated with the Optimizable object.

        Returns:
            Number of free DOFs associated with the Optimizable object.
        """
        return self._dofs.reduced_len

    @property
    def x(self) -> RealArray:
        """
        Numeric values of the free DOFs associated with the current
        Optimizable object and those of its ancestors
        """
        return np.concatenate([opt._dofs.x for
                               opt in (self.ancestors + [self])])

    @x.setter
    def x(self, x: RealArray) -> None:
        if list(self.dof_indices.values())[-1][-1] != len(x):
            raise ValueError
        for opt, indices in self.dof_indices.items():
            if opt != self:
                opt._set_local_x(x[indices[0]:indices[1]])
                opt.new_x = True
                opt.recompute_bell()
            else:
                opt.local_x = x[indices[0]:indices[1]]

    @property
    def full_x(self) -> RealArray:
        """
        Numeric values of all the DOFs (both free and fixed) associated
        with the current Optimizable object and those of its ancestors
        """
        return np.concatenate([opt._dofs.full_x for
                               opt in (self.ancestors + [self])])

    @property
    def local_x(self) -> RealArray:
        """
        Numeric values of the free DOFs associated with this
        Optimizable object
        """
        return self._dofs.x

    @local_x.setter
    def local_x(self, x: RealArray) -> None:
        """
        Setter for local dofs.
        """
        self._set_local_x(x)
        self._set_new_x()

    def _set_local_x(self, x: RealArray) -> None:
        if self.local_dof_size != len(x):
            raise ValueError
        self._dofs.x = x
        if self.local_dof_setter is not None:
            self.local_dof_setter(self, list(self.local_full_x))

    @property
    def local_full_x(self):
        """
        Numeric values of all DOFs (both free and fixed) associated with
        this Optimizable object
        """
        return self._dofs.full_x

    @local_full_x.setter
    def local_full_x(self, x: RealArray) -> None:
        """
        For those cases, where one wants to assign all DOFs including fixed

        .. warning:: Even fixed DOFs are assigned.
        """
        self._dofs.full_x = x
        if self.local_dof_setter is not None:
            self.local_dof_setter(self, list(self.local_full_x))
        self._set_new_x()

    def _set_new_x(self, parent=None):
        self.new_x = True
        self.recompute_bell(parent=parent)

        # for child in self._children:
        for weakref_child in self._children:
            child = weakref_child()
            if child is not None:
                child._set_new_x(parent=self)

    def get(self, key: Key) -> Real:
        """
        Get the value of specified DOF.
        Even fixed dofs can be obtained individually.

        Args:
            key: DOF identifier
        """
        return self._dofs.get(key)

    def set(self, key: Key, new_val: Real) -> None:
        """
        Update the value held the specified DOF.
        Even fixed dofs can be set this way

        Args:
            key: DOF identifier
            new_val: New value of the DOF
        """
        self._dofs.set(key, new_val)
        if self.local_dof_setter is not None:
            self.local_dof_setter(self, list(self.local_full_x))
        self._set_new_x()

    def recompute_bell(self, parent=None):
        """
        Function to be called whenever new DOFs input is given or if the
        parent Optimizable's data changed, so the output from the current
        Optimizable object is invalid.

        This method gets called by various DOF setters. If only the local
        DOFs of an object are being set, the recompute_bell method is called
        in that object and also in the descendent objects that have a dependency
        on the object, whose local DOFs are being changed. If gloabl DOFs
        of an object are being set, the recompute_bell method is called in
        the object, ancestors of the object, as well as the descendents of
        the object.

        Need to be implemented by classes that provide a dof_setter for
        external handling of DOFs.
        """
        pass

    @property
    def bounds(self) -> Tuple[RealArray, RealArray]:
        """
        Lower and upper bounds of the free DOFs associated with the current
        Optimizable object and those of its ancestors
        """
        return (self.lower_bounds, self.upper_bounds)

    @property
    def local_bounds(self) -> Tuple[RealArray, RealArray]:
        """
        Lower and upper bounds of the free DOFs associated with
        this Optimizable object
        """
        return self._dofs.bounds

    @property
    def lower_bounds(self) -> RealArray:
        """
        Lower bounds of the free DOFs associated with the current
        Optimizable object and those of its ancestors
        """
        opts = self.ancestors + [self]
        return np.concatenate([opt._dofs.lower_bounds for opt in opts])

    @property
    def local_lower_bounds(self) -> RealArray:
        """
        Lower bounds of the free DOFs associated with this Optimizable
        object
        """
        return self._dofs.lower_bounds

    @property
    def local_full_lower_bounds(self) -> RealArray:
        return self._dofs.full_lower_bounds

    @property
    def upper_bounds(self) -> RealArray:
        """
        Upper bounds of the free DOFs associated with the current
        Optimizable object and those of its ancestors
        """
        opts = self.ancestors + [self]
        return np.concatenate([opt._dofs.upper_bounds for opt in opts])

    @property
    def local_upper_bounds(self) -> RealArray:
        """
        Upper bounds of the free DOFs associated with this Optimizable
        object
        """
        return self._dofs.upper_bounds

    @property
    def local_full_upper_bounds(self) -> RealArray:
        return self._dofs.full_upper_bounds

    @local_upper_bounds.setter
    def local_upper_bounds(self, lub: RealArray) -> None:
        self._dofs.upper_bounds = lub

    @property
    def dof_names(self) -> StrArray:
        """
        Names (Identifiers) of the DOFs associated with the current
        Optimizable object and those of its ancestors
        """
        opts = self.ancestors + [self]
        names = []
        for opt in opts:
            names += [opt.name + ":" + dname for dname in opt._dofs.names]
        return names

    @property
    def full_dof_names(self) -> StrArray:
        """
        Names (Identifiers) of the DOFs associated with the current
        Optimizable object and those of its ancestors
        """
        opts = self.ancestors + [self]
        names = []
        for opt in opts:
            names += [opt.name + ":" + dname for dname in opt._dofs.full_names]
        return names

    @property
    def local_dof_names(self) -> StrArray:
        """
        Names (Identifiers) of the DOFs associated with this Optimizable
        object
        """
        return self._dofs.names

    @property
    def local_full_dof_names(self) -> StrArray:
        """
        Names (Identifiers) of the DOFs associated with this Optimizable
        object
        """
        return self._dofs.full_names

    @property
    def dofs_free_status(self) -> BoolArray:
        """
        Boolean array denoting whether the DOFs associated with the
        current and ancestors Optimizable objects are free or not
        """
        return np.concatenate(
            [opt._dofs.free_status for opt in self.ancestors + [self]])

    @property
    def local_dofs_free_status(self) -> BoolArray:
        """
        Boolean array denoting whether the DOFs associated with the
        current Optimizable object are free or not
        """
        return self._dofs.free_status

    def is_fixed(self, key: Key) -> bool:
        """
        Checks if the specified dof is fixed

        Args:
            key: DOF identifier
        """
        return not self.is_free(key)

    def is_free(self, key: Key) -> bool:
        """
        Checks if the specified dof is free

        Args:
            key: DOF identifier
        """
        return self._dofs.is_free(key)

    def fix(self, key: Key) -> None:
        """
        Set the fixed attribute for the given degree of freedom.

        Args:
            key: DOF identifier
        """
        # TODO: Question: Should we use ifix similar to pandas' loc and iloc?

        self._dofs.fix(key)
        self._update_free_dof_size_indices()

    def unfix(self, key: Key) -> None:
        """
        Unset the fixed attribute for the given degree of freedom

        Args:
            key: DOF identifier
        """
        self._dofs.unfix(key)
        self._update_free_dof_size_indices()

    def local_fix_all(self) -> None:
        """
        Set the 'fixed' attribute for all local degrees of freedom associated
        with the current Optimizable object.
        """
        self._dofs.fix_all()
        self._update_free_dof_size_indices()

    def fix_all(self) -> None:
        """
        Set the 'fixed' attribute for all local degrees of freedom associated
        with the current Optimizable object including those of ancestors.
        """
        opts = self.ancestors + [self]
        for opt in opts:
            opt.local_fix_all()

    def local_unfix_all(self) -> None:
        """
        Unset the 'fixed' attribute for all local degrees of freedom associated
        with the current Optimizable object.
        """
        self._dofs.unfix_all()
        self._update_free_dof_size_indices()

    def unfix_all(self) -> None:
        """
        Unset the 'fixed' attribute for all local degrees of freedom associated
        with the current Optimizable object including those of the ancestors.
        """
        opts = self.ancestors + [self]
        for opt in opts:
            opt.local_unfix_all()

    def __add__(self, other):
        """ Add two Optimizable objects """
        return OptimizableSum([self, other])

    def __mul__(self, other):
        """ Multiply an Optimizable object by a scalar """
        return ScaledOptimizable(other, self)

    def __rmul__(self, other):
        """ Multiply an Optimizable object by a scalar """
        return ScaledOptimizable(other, self)

    # https://stackoverflow.com/questions/11624955/avoiding-python-sum-default-start-arg-behavior
    def __radd__(self, other):
        # This allows sum() to work (the default start value is zero)
        if other == 0:
            return self
        return self.__add__(other)

    @SimsoptRequires(nx is not None, "print method for DAG requires networkx")
    @SimsoptRequires(pygraphviz is not None, "print method for DAG requires pygraphviz")
    @SimsoptRequires(plt is not None, "print method for DAG requires matplotlib")
    def plot_graph(self, show=True):
        """
        Plot the directed acyclical graph that represents the dependencies of an 
        ``Optimizable`` on its parents. The workflow is as follows: generate a ``networkx``
        ``DiGraph`` using the ``traversal`` function defined below.  Next, call ``graphviz_layout``
        which determines sensible positions for the nodes of the graph using the ``dot``
        program of ``graphviz``. Finally, ``networkx`` plots the graph using ``matplotlib``.

        Note that the tool ``network2tikz`` at `https://github.com/hackl/network2tikz <https://github.com/hackl/network2tikz>`_
        can be used to convert the networkx ``DiGraph`` and positions to a 
        latex file for publication.

        Args:
            show: Whether to call the ``show()`` function of matplotlib.

        Returns:
            The ``networkx`` graph corresponding to this ``Optimizable``'s directed acyclical graph
            and a dictionary of node names that map to sensible x, y positions determined by ``graphviz``
        """

        G = nx.DiGraph()
        G.add_node(self.name) 

        def traversal(root):
            for p in root.parents:
                n1 = root.name
                n2 = p.name
                G.add_edge(n1, n2)
                traversal(p)

        traversal(self)

        # this command generates sensible positions for nodes of the DAG
        # using the "dot" program
        pos = graphviz_layout(G, prog='dot')
        options = {
            'node_color': 'white',
            'arrowstyle': '-|>',
            'arrowsize': 12,
            'font_size': 12}
        nx.draw_networkx(G, pos=pos, arrows=True, **options)
        if show:
            plt.show()

        return G, pos

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        if len(self.local_full_x):
            d["x0"] = list(self.local_full_x)
            d["names"] = self.local_full_dof_names
            d["fixed"] = list(np.logical_not(self.local_dofs_free_status))
            d["lower_bounds"] = list(self.local_full_lower_bounds)
            d["upper_bounds"] = list(self.local_full_upper_bounds)
        # d["external_dof_setter"] = self.local_dof_setter
        if self.parents:
            d["depends_on"] = []
            for parent in self.parents:
                d["depends_on"].append(parent.as_dict())

        return d

    @staticmethod
    def _decode(d):
        parents_dict = d.pop("depends_on") if "depends_on" in d else None
        if parents_dict:
            parents = []
            decoder = MontyDecoder()
            for pdict in parents_dict:
                parents.append(decoder.process_decoded(pdict))
            return parents

    @classmethod
    def from_dict(cls, d):
        parents = Optimizable._decode(d)
        return cls(depends_on=parents, **d)

    def save(self, filename=None, fmt=None, **kwargs):
        filename = filename or ""
        fmt = "" if fmt is None else fmt.lower()
        fname = Path(filename).name

        if fmt == "json" or fnmatch(fname.lower(), "*.json"):
            if "cls" not in kwargs:
                kwargs["cls"] = MontyEncoder
            if "indent" not in kwargs:
                kwargs["indent"] = 2
            s = json.dumps(self.as_dict(), **kwargs)
            if filename:
                with zopen(filename, "wt") as f:
                    f.write(s)
            return s
        else:
            raise ValueError(f"Invalid format: `{str(fmt)}`")

    @classmethod
    def from_str(cls, input_str: str, fmt="json"):
        fmt_low = fmt.lower()
        if fmt_low == "json":
            return json.loads(input_str, cls=MontyDecoder)
        else:
            raise ValueError(f"Invalid format: `{str(fmt)}`")

    @classmethod
    def from_file(cls, filename: str):
        fname = Path(filename).name
        if fnmatch(filename, "*.json*") or fnmatch(fname, "*.bson*"):
            with zopen(filename, "rt") as f:
                contents = f.read()
            return cls.from_str(contents, fmt="json")


def load(filename, *args, **kwargs):
    """
    Function to load simsopt object from a file.
    Only JSON format is supported at this time. Support for additional
    formats will be added in future
    Args:
        filename:
            Name of file from which simsopt object has to be initialized
    Returns:
        Simsopt object
    """
    fname = Path(filename).suffix.lower()
    if (not fname == '.json'):
        raise ValueError(f"Invalid format: `{str(fname[1:])}`")

    with zopen(filename, "rt") as fp:
        if "cls" not in kwargs:
            kwargs["cls"] = MontyDecoder
        return json.load(fp, *args, **kwargs)


def save(simsopt_objects, filename, *args, **kwargs):
    fname = Path(filename).suffix.lower()
    if (not fname == '.json'):
        raise ValueError(f"Invalid format: `{str(fname[1:])}`")

    with zopen(filename, "wt") as fp:
        if "cls" not in kwargs:
            kwargs["cls"] = MontyEncoder
        if "indent" not in kwargs:
            kwargs["indent"] = 2
        return json.dump(simsopt_objects, fp, *args, **kwargs)


def make_optimizable(func, *args, dof_indicators=None, **kwargs):
    """
    Factory function to generate an Optimizable instance from a function
    to be used with the graph framework.

    Args:
        func: Callable to be used in the optimization
        args: Positional arguments to pass to "func".
        dof_indicators: List of strings that match with the length of the
            args and kwargs. Each string can be either of
            "opt" - to indicate the argument is optimizable object
            "dof" - argument that is a degree of freedom for optimization
            "non-dof" - argument that is not part of optimization.
            Here ordered property of the dict is used to map kwargs to
            dof_indicators. Another important thing to consider is dofs related
            to optimizable objects supplied as arguments should not be given.
        kwargs: Keyword arguments to pass to "func".
    Returns:
        Optimizable object to be used in the graph based optimization.
        This object has a bound function ``J()`` that calls the originally
        supplied ``func()``.
        If ``obj`` is the returned object, pass ``obj.J`` to the
        ``LeastSquaresProblem``
    """
    class TempOptimizable(Optimizable):
        """
        Subclass of Optimizable class to create optimizable objects dynamically.
        dof_indicators argument is used to filter out dofs and
        """

        def __init__(self, func, *args, dof_indicators=None, **kwargs):

            self.func = func
            self.arg_len = len(args)
            self.kwarg_len = len(kwargs)
            self.kwarg_keys = []
            if dof_indicators is not None:
                assert (self.arg_len + self.kwarg_len == len(dof_indicators))
                # Using dof_indicators, map args and kwargs to
                # dofs, non_dofs, and opts
                dofs, non_dofs, opts = [], [], []
                for i, arg in enumerate(args):
                    if dof_indicators[i] == 'opt':
                        opts.append(arg)
                    elif dof_indicators[i] == "non-dof":
                        non_dofs.append(arg)
                    elif dof_indicators[i] == "dof":
                        dofs.append(arg)
                    else:
                        raise ValueError
                for i, k in enumerate(kwargs.keys()):
                    self.kwarg_keys.append(k)
                    if dof_indicators[i + self.arg_len] == 'opt':
                        opts.append(kwargs[k])
                    elif dof_indicators[i + self.arg_len] == "non-dof":
                        non_dofs.append(kwargs[k])
                    elif dof_indicators[i + self.arg_len] == "dof":
                        dofs.append(kwargs[k])
                    else:
                        raise ValueError
            else:
                # nonlocal dof_indicators
                dofs, non_dofs, opts, dof_indicators = [], [], [], []
                for i, arg in enumerate(args):
                    if isinstance(arg, Optimizable):
                        opts.append(arg)
                        dof_indicators.append("opt")
                    else:
                        non_dofs.append(arg)
                        dof_indicators.append("non-dof")
                for k, v in kwargs.items():
                    self.kwarg_keys.append(k)
                    if isinstance(v, Optimizable):
                        opts.append(v)
                        dof_indicators.append("opt")
                    else:
                        non_dofs.append(v)
                        dof_indicators.append("non-dof")

            # Create args map and kwargs map
            super().__init__(x0=dofs, depends_on=opts)
            self.non_dofs = non_dofs
            self.dof_indicators = dof_indicators

        def J(self):
            dofs = self.local_full_x
            # Re-Assemble dofs, non_dofs and opts to args, kwargs
            args = []
            kwargs = {}
            i = 0
            opt_ind = 0
            non_dof_ind = 0
            dof_ind = 0
            for i in range(self.arg_len):
                if self.dof_indicators[i] == 'opt':
                    args.append(self.parents[opt_ind])
                    opt_ind += 1
                elif self.dof_indicators[i] == 'dof':
                    args.append(dofs[dof_ind])
                    dof_ind += 1
                elif self.dof_indicators[i] == 'non-dof':
                    args.append(self.non_dofs[non_dof_ind])
                    non_dof_ind += 1
                else:
                    raise ValueError
                i += 1

            for j in range(self.kwarg_len):
                i = j + self.arg_len
                if self.dof_indicators[i] == 'opt':
                    kwargs[self.kwarg_keys[j]] = self.parents[opt_ind]
                    opt_ind += 1
                elif self.dof_indicators[i] == 'dof':
                    kwargs[self.kwarg_keys[j]] = dofs[dof_ind]
                    dof_ind += 1
                elif self.dof_indicators[i] == 'non-dof':
                    kwargs[self.kwarg_keys[j]] = self.non_dofs[non_dof_ind]
                    non_dof_ind += 1
                else:
                    raise ValueError
                j += 1
            log.info(f'reassembled args len is {len(args)}')

            return self.func(*args, **kwargs)

    return TempOptimizable(func, *args, dof_indicators=dof_indicators, **kwargs)


class ScaledOptimizable(Optimizable):
    """
    Represents an :obj:`~simsopt._core.optimizable.Optimizable`
    object scaled by a constant factor. This class is useful for
    including a weight in front of terms in an objective function. For
    now, this feature works on classes for which ``.J()`` returns an
    objective value and ``.dJ()`` returns the gradient, e.g. coil
    optimization.

    Args:
        factor: (float) The constant scale factor.
        opt: An :obj:`~simsopt._core.optimizable.Optimizable` object to scale.
    """

    def __init__(self, factor, opt):
        self.factor = factor
        self.opt = opt
        super().__init__(depends_on=[opt])

    def J(self):
        return float(self.factor) * self.opt.J()

    @derivative_dec
    def dJ(self):
        # Next line uses __rmul__ function for the Derivative class
        return float(self.factor) * self.opt.dJ(partials=True)

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["factor"] = self.factor
        d["opt"] = self.opt.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        opt = MontyDecoder().process_decoded(d["opt"])
        return cls(d["factor"], opt)


class OptimizableSum(Optimizable):
    """
    Represents a sum of
    :obj:`~simsopt._core.optimizable.Optimizable` objects. This
    class is useful for combining terms in an objective function. For
    now, this feature works on classes for which ``.J()`` returns an
    objective value and ``.dJ()`` returns the gradient, e.g. coil
    optimization.

    Args:
        opts: A python list of :obj:`~simsopt._core.optimizable.Optimizable` object to sum.
    """

    def __init__(self, opts):
        self.opts = opts
        super().__init__(depends_on=opts)

    def J(self):
        return sum([opt.J() for opt in self.opts])

    @derivative_dec
    def dJ(self):
        # Next line uses __add__ function for the Derivative class
        return sum(opt.dJ(partials=True) for opt in self.opts)

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["opts"] = []
        for opt in self.opts:
            d["opts"].append(opt.as_dict())
        return d

    @classmethod
    def from_dict(cls, d):
        opts = []
        decoder = MontyDecoder()
        for odict in d["opts"]:
            opts.append(decoder.process_decoded(odict))
        return cls(opts)

