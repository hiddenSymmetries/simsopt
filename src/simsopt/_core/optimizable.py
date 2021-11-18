# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides classes and functions that are useful for
setting up optimizable objects and objective functions.
"""

import logging
import types
import warnings

import numpy as np

warnings.warn("optimizable module is deprecated in favor of"
              " graph_optimizable module and will be removed in future versions"
              " of simsopt", DeprecationWarning, stacklevel=2)

logger = logging.getLogger(__name__)


class Optimizable():
    """
    This base class provides some useful features for optimizable functions.
    """

    def get_dofs(self):
        """
        This base Optimizable object has no degrees of freedom, so return
        an empty array
        """
        return np.array([])

    def set_dofs(self, x):
        """
        This base Optimizable object has no degrees of freedom, so do nothing.
        """
        pass

    def index(self, dof_str):
        """
        Given a string dof_str, returns the index in the dof array whose
        name matches dof_str. If dof_str does not match any of the
        names, ValueError will be raised.
        """
        return self.names.index(dof_str)

    def get(self, dof_str):
        """
        Return a degree of freedom specified by its string name.
        """
        x = self.get_dofs()
        return x[self.index(dof_str)]

    def set(self, dof_str, newval):
        """
        Set a degree of freedom specified by its string name.
        """
        x = self.get_dofs()
        x[self.index(dof_str)] = newval
        self.set_dofs(x)

    def get_fixed(self, dof_str):
        """
        Return the fixed attribute for a given degree of freedon, specified by dof_str.
        """
        return self.fixed[self.index(dof_str)]

    def set_fixed(self, dof_str, fixed_new=True):
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.
        """
        self.fixed[self.index(dof_str)] = fixed_new

    def all_fixed(self, fixed_new=True):
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        self.fixed = np.full(len(self.get_dofs()), fixed_new)


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


class Target(Optimizable):

    """
    Given an attribute of an object, which typically would be a
    @property, form a callable function that can be used as a target
    for optimization.
    """

    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        self.depends_on = ["obj"]

        # Attach a dJ function only if obj has one
        def dJ(self0):
            return getattr(self0.obj, 'd' + self0.attr)
        if hasattr(obj, 'd' + attr):
            self.dJ = types.MethodType(dJ, self)

    def J(self):
        return getattr(self.obj, self.attr)

    #def dJ(self):
    #    return getattr(self.obj, 'd' + self.attr)

    def get_dofs(self):
        return np.array([])

    def set_dofs(self, v):
        pass


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
    if not hasattr(obj, 'fixed'):
        obj.fixed = np.full(n, False)
    if not hasattr(obj, 'mins'):
        obj.mins = np.full(n, np.NINF)
    if not hasattr(obj, 'maxs'):
        obj.maxs = np.full(n, np.Inf)
    # Add the following methods from the Optimizable class:
    for method in ['index', 'get', 'set', 'get_fixed', 'set_fixed', 'all_fixed']:
        # See https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        setattr(obj, method, types.MethodType(getattr(Optimizable, method), obj))

    return obj
