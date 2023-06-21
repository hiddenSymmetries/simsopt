# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a few minimal optimizable objects, each
representing a function. These functions are mostly used for testing.
"""

from numbers import Real, Number
from typing import Sequence

import numpy as np

from .._core.optimizable import Optimizable
from .._core.types import RealArray
from .._core.util import ObjectiveFailure


class Identity(Optimizable):
    """
    Represents a term in an objective function which is just
    the identity. It has one degree of freedom. Conforms to the 
    graph based Optimizable framework.

    The output of the method `f` is equal to this degree of freedom.
    The call hook internally calls method f. It does not have any parent
    Optimizable nodes

    Args:
        x: Value of the DOF
        dof_name: Identifier for the DOF
        dof_fixed: To specify if the dof is fixed
    """

    def __init__(self,
                 x: Real = 0.0,
                 dof_name: str = None,
                 dof_fixed: bool = False):
        super().__init__([x],
                         [dof_name] if dof_name is not None else None,
                         [dof_fixed])

    def f(self):
        """
        Returns the value of the DOF
        """
        return self.full_x[0]

    def dJ(self, x: RealArray = None):
        if x is not None:
            if isinstance(x, Real):
                self.x = [x]
            else:
                self.x = x
        return np.array([1.0])

    return_fn_map = {'f': f}

    def as_dict(self, serial_objs_dict: dict) -> dict:
        d = {}
        d["@class"] = self.__class__.__name__
        d["@module"] = self.__class__.__module__
        d["@name"] = self.name
        d["x"] = self.local_full_x[0]
        d["dof_name"] = self.local_full_dof_names[0]
        d["dof_fixed"] = np.logical_not(self.local_dofs_free_status)[0]
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        return cls(d["x"], d["dof_name"], d["dof_fixed"])


class Adder(Optimizable):
    """
    Defines a minimal graphe based Optimizable object that can be optimized.
    It has n degrees of freedom.

    The method `sum` returns the sum of these dofs. The call hook internally
    calls the `sum` method.

    Args:
        n: Number of degrees of freedom (DOFs)
        x0: Initial values of the DOFs. If not given, equal to zeroes
        dof_names: Identifiers for the DOFs
    """

    def __init__(self, n=3, **kwargs):
        self.n = n
        super().__init__(**kwargs)

    def sum(self):
        """
        Sums the DOFs
        """
        return np.sum(self._dofs.full_x)

    def J(self):
        return self.sum()

    def dJ(self):
        return np.ones(self.n)

    @property
    def df(self):
        """
        Same as the function dJ(), but a property instead of a function.
        """
        return self.dJ()

    return_fn_map = {'sum': sum}


class Rosenbrock(Optimizable):
    """
    Implements Rosenbrock function using the graph based optimization
    framework. The Rosenbrock function is defined as

    .. math::
        f(x,y) = (a-x)^2 + b(y-x^2)^2

    The parameter *a* is fixed to 1. And the *b* parameter can be given as
    input.

    Args:
        b: The *b* parameter of Rosenbrock function
        x0: *x, y* coordinates
    """

    def __init__(self, b=100.0, x0=[0.0, 0.0], **kwargs):
        self._sqrtb = np.sqrt(b)
        if "names" not in kwargs:
            kwargs["names"] = ["x", "y"]
        super().__init__(x0=x0, **kwargs)

    @property
    def term1(self):
        """
        Returns the first of the two quantities that is squared and summed.
        """
        return self.local_full_x[0] - 1

    @property
    def term2(self):
        """
        Returns the second of the two quantities that is squared and summed.
        """
        x = self.local_full_x[0]
        y = self.local_full_x[1]
        return (x * x - y) / self._sqrtb

    @property
    def dterm1(self):
        """
        Returns the gradient of term1
        """
        return np.array([1.0, 0.0])

    @property
    def dterm2(self):
        """
        Returns the gradient of term2
        """
        return np.array([2 * self.local_full_x[0], -1.0]) / self._sqrtb

    def f(self, x=None):
        """
        Returns the total function, squaring and summing the two terms.
        """
        if x is not None:
            self.x = x
        t1 = self.term1
        t2 = self.term2
        return t1 * t1 + t2 * t2

    return_fn_map = {'f': f}

    @property
    def terms(self):
        """
        Returns term1 and term2 together as a 2-element numpy vector.
        """
        return np.array([self.term1, self.term2])

    def dterms(self):
        """
        Returns the 2x2 Jacobian for term1 and term2.
        """
        return np.array([[1.0, 0.0],
                         [2 * self.local_full_x['x'] / self._sqrtb, -1.0 / self._sqrtb]])

    @property
    def b(self):
        return self._sqrtb * self._sqrtb


class TestObject1(Optimizable):
    """
    Implements a graph based optimizable with a single degree of freedom and has
    parent optimizable nodes. Mainly used for testing.

    The output method is named `f`. Call hook internally calls method `f`.

    Args:
        val: Degree of freedom
        opts: Parent optimizable objects. If not given, two Adder objects are
              added as parents
    """

    def __init__(self, x0: Real, depends_on: Sequence[Optimizable] = None,
                 **kwargs):
        if depends_on is None:
            depends_on = [Adder(3), Adder(2)]
        if isinstance(x0, Number):
            x0 = [x0]
        if "names" not in kwargs:
            kwargs["names"] = ["val"]
        super().__init__(x0=x0, depends_on=depends_on,
                         **kwargs)

    def f(self):
        """
        Implements an objective function
        """
        return (self.local_full_x[0] + 2 * self.parents[0]()) / \
               (10.0 + self.parents[1]())

    return_fn_map = {'f': f}

    def dJ(self):
        """
        Same as dJ() but a property instead of a function.
        """
        v = self._dofs.full_x[0]
        a1 = self.parents[0]()
        a2 = self.parents[1]()
        return np.concatenate(
            (np.array([1.0 / (10.0 + a2)]),
             np.full(self.parents[0].n, 2.0 / (10.0 + a2)),
             np.full(self.parents[1].n, -(v + 2 * a1) / ((10.0 + a2) ** 2))))

    @property
    def depends_on(self):
        return self.parents


class TestObject2(Optimizable):
    """
    Implements a graph based optimizable with two single degree of freedom
    and has two parent optimizable nodes. Mainly used for testing.

    The output method is named `f`. Call hook internally calls method `f`.

    Args:
        val1: First degree of freedom
        val2: Second degree of freedom
    """

    def __init__(self, val1, val2):
        x = [val1, val2]
        names = ['val1', 'val2']
        funcs = [TestObject1(0.0), Adder(2)]
        super().__init__(x0=x, names=names, funcs_in=funcs)

    def f(self):
        x = self.local_full_x
        v1 = x[0]
        v2 = x[1]
        t = self.parents[0]()
        a = self.parents[1]()
        return v1 + a * np.cos(v2 + t)

    return_fn_map = {'f': f}

    def dJ(self):
        x = self.local_full_x
        v1 = x[0]
        v2 = x[1]
        t = self.parents[0]()
        a = self.parents[1]()
        cosat = np.cos(v2 + t)
        sinat = np.sin(v2 + t)
        # Order of terms in the gradient: v1, v2, t, a
        return np.concatenate((np.array([1.0, -a * sinat]),
                               -a * sinat * self.parents[0].dJ(),
                               cosat * self.parents[1].dJ()))


class Affine(Optimizable):
    """
    Implements a random affine (i.e. linear plus constant)
    transformation from R^n to R^m. The n inputs to the transformation are
    initially set to zeroes.

    Args:
        nparams: number of independent variables.
        nvals: number of dependent variables.
    """

    def __init__(self, nparams, nvals):
        self.nparams = nparams
        self.nvals = nvals
        self.A = (np.random.rand(nvals, nparams) - 0.5) * 4
        self.B = (np.random.rand(nvals) - 0.5) * 4
        super().__init__(np.zeros(nparams))

    def f(self):
        return np.matmul(self.A, self.full_x) + self.B

    return_fn_map = {'f': f}

    def dJ(self):
        return self.A


class Failer(Optimizable):
    """
    This class is used for testing failures of the objective
    function. This function always returns a vector with entries all
    1.0, except that ObjectiveFailure will be raised on a specified
    evaluation.

    Args:
        nparams: Number of input values.
        nvals: Number of entries in the return vector.
        fail_index: Which function evaluation to fail on.
    """

    def __init__(self,
                 nparams: int = 2,
                 nvals: int = 3,
                 fail_index: int = 2):
        self.nparams = nparams
        self.nvals = nvals
        self.fail_index = fail_index
        self.nevals = 0
        super().__init__(np.zeros(nparams))
        self.x = np.zeros(self.nparams)

    def J(self):
        self.nevals += 1
        if self.nevals == self.fail_index:
            raise ObjectiveFailure("nevals == fail_index")
        else:
            if self.nvals == 0: 
                # return scalar
                return 1.0
            else:
                # return vector
                return np.full(self.nvals, 1.0)

    def get_dofs(self):
        return self.x

    def set_dofs(self, x):
        self.x = x


class Beale(Optimizable):
    """
    This is a test function which does not supply derivatives. It is
    taken from
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def __init__(self, x0=None, **kwargs):
        x = np.zeros(2) if x0 is None else x0
        super().__init__(x0=x, **kwargs)

    def J(self):
        x = self.local_full_x[0]
        y = self.local_full_x[1]
        return np.array([1.5 - x + x * y,
                         2.25 - x + x * y * y,
                         2.625 - x + x * y * y * y])

