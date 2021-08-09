# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
Provides graph based Optimizable class, whose instances can be used to
build an optimization problem in a graph like manner.
"""

from __future__ import annotations

import types
import hashlib
from collections.abc import Callable as ABC_Callable, Hashable
from collections import defaultdict
from numbers import Real, Integral
from typing import Union, Tuple, Dict, Callable, Sequence, \
    MutableSequence as MutSeq, List

import numpy as np
from deprecated import deprecated

from ..util.types import RealArray, StrArray, BoolArray, Key
from .util import ImmutableId, OptimizableMeta


class DOFs:
    """
    Defines the (D)egrees (O)f (F)reedom(s) associated with optimization

    This class holds data related to the degrees of freedom
    associated with an Optimizable object.

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
            x = np.array(x, dtype=np.double)

        if names is None:
            names = [f"x{i}" for i in range(len(x))]
        assert(len(np.unique(names)) == len(names))  # DOF names should be unique

        if free is not None:
            free = np.array(free, dtype=np.bool_)
        else:
            free = np.full(len(x), True)

        if lower_bounds is not None:
            lb = np.array(lower_bounds, np.double)
        else:
            lb = np.full(len(x), np.NINF)

        if upper_bounds is not None:
            ub = np.array(upper_bounds, np.double)
        else:
            ub = np.full(len(x), np.inf)

        assert(len(x) == len(free) == len(lb) == len(ub) == len(names))
        self._x = x
        self._free = free
        self._lb = lb
        self._ub = ub
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
        # if not self._free[key]:
        #     raise IndexError("The DOF is fixed")
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
        if len(self._free[self._free]) != len(x):
            # To prevent fully fixed DOFs from not raising Error
            # And to prevent broadcasting of a single DOF
            raise ValueError
        self._x[self._free] = x

    @property
    def full_x(self) -> RealArray:
        """
        Return all x even the fixed ones

        Returns:
            The values of full DOFs without any restrictions
        """
        return self._x

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

    @lower_bounds.setter
    def lower_bounds(self, lower_bounds: RealArray) -> None:
        """

        Args:
            lower_bounds: Lower bounds of the DOFs
        """
        if len(self._free[self._free]) != len(lower_bounds):
            # To prevent fully fixed DOFs from not raising Error
            # And to prevent broadcasting of a single DOF
            raise ValueError
        self._lb[self._free] = lower_bounds

    @property
    def upper_bounds(self) -> RealArray:
        """

        Returns:
            Upper bounds of the DOFs
        """
        return self._ub[self._free]

    @upper_bounds.setter
    def upper_bounds(self, upper_bounds: RealArray) -> None:
        """

        Args:
            upper_bounds: Upper bounds of the DOFs
        """
        if len(self._free[self._free]) != len(upper_bounds):
            # To prevent fully fixed DOFs from not raising Error
            # And to prevent broadcasting of a single DOF
            raise ValueError
        self._ub[self._free] = upper_bounds

    @property
    def bounds(self) -> Tuple[RealArray, RealArray]:
        """

        Returns:
            (Lower bounds list, Upper bounds list)
        """
        return (self.lower_bounds, self.upper_bounds)

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
        return self._names


class Optimizable(ABC_Callable, Hashable, metaclass=OptimizableMeta):
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
        self._children = []  # This gets populated when the object is passed
        # as argument to another Optimizable object
        self.return_fns = defaultdict(list)  # Store return fn's required by each child

        # Assign self as child to parents
        self.parents = depends_on if depends_on is not None else []
        for i, parent in enumerate(self.parents):
            parent._add_child(self)
            return_fns = opt_return_fns[i] if opt_return_fns else []
            for fn in return_fns:
                parent.add_return_fn(self, fn)

        # Process funcs_in (Assumes depends_on is empty)
        if depends_on is None or not len(depends_on):
            depends_on = []
            funcs_in = funcs_in if funcs_in is not None else []
            for fn in funcs_in:
                opt_in = fn.__self__
                depends_on.append(opt_in)
                opt_in.add_return_fn(self, fn.__func__)
            depends_on = list(dict.fromkeys(depends_on))
            self.parents = list(depends_on) if depends_on is not None else []
            for i, parent in enumerate(self.parents):
                parent._add_child(self)

        # Obtain unique list of the ancestors
        self.ancestors = self._get_ancestors()

        # Compute the indices of all the DOFs
        self._update_free_dof_size_indices()
        self._update_full_dof_size_indices()
        # Inform the object that it doesn't have valid cache
        self._set_new_x()
        super().__init__(**kwargs)

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
        return self.name == other.name

    def __call__(self, x: RealArray = None, *args, child=None, **kwargs):
        if x is not None:
            self.x = x
        return_fn_map = self.__class__.return_fn_map
        if self.new_x:
            result = []
            for fn in return_fn_map.values():
                result.append(
                    fn(self, *args, **kwargs))
            self._val = result
            self.new_x = False

        if child:
            return_fns = self.return_fns[child] if self.return_fns[child] else \
                return_fn_map.values()
        else:
            return_fns = return_fn_map.values()

        return_result = []
        for fn in return_fns:
            i = list(return_fn_map.values()).index(fn)
            return_result.append(self._val[i])

        return return_result if len(return_result) > 1 else return_result[0]

    #@abc.abstractmethod
    #def f(self, *args, **kwargs):
    #    """
    #    Defines the callback function associated with the Optimizable subclass.

    #   Define the callback method in subclass of Optimizable. The function
    #    uses the state (x) stored self._dofs. To prevent hard-to-understand
    #    bugs, don't forget to use self.full_x.

    #    Args:
    #        *args:
    #        **kwargs:

    #    Returns:

    #    """

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
        return self.return_fns.get(child, list(self.return_fn_map.values()))

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
        if child not in self._children:
            self._children.append(child)

    def _remove_child(self, other: Optimizable) -> None:
        """
        Remove the specific Optimizable object from the children list.

        Args:
            child: Direct dependent (child) of the Optimizable object
        """
        self._children.remove(other)
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
            self.ancestors = self._get_ancestors()
            self._update_free_dof_size_indices()
            self._update_full_dof_size_indices()
            self._set_new_x()
        else:
            print("The given Optimizable object is already a parent")

    def append_parent(self, other: Optimizable) -> None:
        """
        Appends another Optimizable object to parents list

        Args:
            other: New parent Optimizable object
        """
        if other not in self.parents:
            self.parents.append(other)
            other._add_child(self)
            self.ancestors = self._get_ancestors()
            self._update_free_dof_size_indices()
            self._update_full_dof_size_indices()
            self._set_new_x()
        else:
            print("The given Optimizable object is already a parent")

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
        self.ancestors = self._get_ancestors()
        self._update_free_dof_size_indices()
        self._update_full_dof_size_indices()
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
        self.ancestors = self._get_ancestors()
        self._update_free_dof_size_indices()
        self._update_full_dof_size_indices()
        self._set_new_x()

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

        # Update the reduced length of children
        for child in self._children:
            child._update_free_dof_size_indices()

    def _update_full_dof_size_indices(self) -> None:
        """
        Updates the full DOFs lengths for this instance and
        those of the children.

        Call this function whenever parents are added or removed. Recursively
        calls the same function in children.
        """

        # TODO: This is slow because it walks through the graph repeatedly
        # TODO: Develop a faster scheme.
        # TODO: Alternatively ask the user to call this manually from the end
        # TODO: node after fixing/unfixing any DOF
        full_dof_size = 0
        for opt in (self.ancestors + [self]):
            full_dof_size += opt.local_full_dof_size
        self._full_dof_size = full_dof_size

        # Update the reduced length of children
        for child in self._children:
            child._update_full_dof_size_indices()

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
                # self._set_new_x()

    @property
    def full_x(self) -> RealArray:
        """
        Numeric values of all the DOFs associated with the current
        Optimizable object and those of its ancestors
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
        Numeric values of all DOFs associated with this Optimizable object
        """
        return self._dofs.full_x

    def _set_new_x(self, parent=None):
        self.new_x = True
        #if self.local_dof_setter is not None:
        self.recompute_bell(parent=parent)

        for child in self._children:
            child._set_new_x(parent=self)

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
        return np.concatenate([opt._dofs.names for opt in opts])

    @property
    def local_dof_names(self) -> StrArray:
        """
        Names (Identifiers) of the DOFs associated with this Optimizable
        object
        """
        return self._dofs.names

    def get(self, key: Key) -> Real:
        """
        Get the value of specified DOF
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
        self._set_new_x()

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

    def fix_all(self) -> None:
        """
        Set the 'fixed' attribute for all degrees of freedom associated with
        the current Optimizable object.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), True)
        self._dofs.fix_all()
        self._update_free_dof_size_indices()

    def unfix_all(self) -> None:
        """
        Unset the 'fixed' attribute for all degrees of freedom associated
        with the current Optimizable object.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), False)
        self._dofs.unfix_all()
        self._update_free_dof_size_indices()

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
        return list(dict.fromkeys(ancestors))
