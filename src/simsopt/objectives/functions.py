# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a few minimal optimizable objects, each
representing a function. These functions are mostly used for testing.
"""

import numpy as np

from .._core.optimizable import Optimizable
from .._core.util import ObjectiveFailure


class Identity(Optimizable):
    """
    This class represents a term in an objective function which is just
    the identity. It has one degree of freedom, and the output of the function
    is equal to this degree of freedom.
    """

    def __init__(self, x=0.0):
        self.x = x
        self.dx = np.array([1.0])
        self.fixed = np.full(1, False)
        self.names = ['x']

    def J(self):
        return self.x

    def dJ(self):
        return np.array([1.0])

    @property
    def f(self):
        """
        Same as the function J(), but a property instead of a function.
        """
        return self.x

    @property
    def df(self):
        """
        Same as the function dJ(), but a property instead of a function.
        """
        return np.array([1.0])

    def get_dofs(self):
        return np.array([self.x])

    def set_dofs(self, xin):
        self.x = xin[0]


class Adder(Optimizable):
    """This class defines a minimal object that can be optimized. It has
    n degrees of freedom, and has a function that just returns the sum
    of these dofs. This class is used for testing.
    """

    def __init__(self, n=3):
        self.n = n
        self.x = np.zeros(n)
        self.fixed = np.full(n, False)        

    def J(self):
        """
        Returns the sum of the degrees of freedom.
        """
        return np.sum(self.x)

    def dJ(self):
        return np.ones(self.n)

    @property
    def f(self):
        """
        Same as the function J(), but a property instead of a function.
        """
        return self.J()

    @property
    def df(self):
        """
        Same as the function dJ(), but a property instead of a function.
        """
        return np.ones(self.n)

    def get_dofs(self):
        return self.x

    def set_dofs(self, xin):
        self.x = np.array(xin)


class Rosenbrock(Optimizable):
    """
    This class defines a minimal object that can be optimized.
    """

    def __init__(self, b=100.0, x=0.0, y=0.0):
        self._sqrtb = np.sqrt(b)
        self.names = ['x', 'y']
        self._x = x
        self._y = y
        self.fixed = np.full(2, False)        

    def term1(self):
        """
        Returns the first of the two quantities that is squared and summed.
        """
        return self._x - 1

    def term2(self):
        """
        Returns the second of the two quantities that is squared and summed.
        """
        return (self._x * self._x - self._y) / self._sqrtb

    def dterm1(self):
        """
        Returns the gradient of term1
        """
        return np.array([1.0, 0.0])

    def dterm2(self):
        """
        Returns the gradient of term2
        """
        return np.array([2 * self._x, -1.0]) / self._sqrtb

    @property
    def term1prop(self):
        """
        Same as term1, but a property rather than a callable function.
        """
        return self.term1()

    @property
    def term2prop(self):
        """
        Same as term2, but a property rather than a callable function.
        """
        return self.term2()

    @property
    def dterm1prop(self):
        """
        Same as dterm1, but a property rather than a callable function.
        """
        return self.dterm1()

    @property
    def dterm2prop(self):
        """
        Same as dterm2, but a property rather than a callable function.
        """
        return self.dterm2()

    def f(self):
        """
        Returns the total function, squaring and summing the two terms.
        """
        t1 = self.term1()
        t2 = self.term2()
        return t1 * t1 + t2 * t2

    def terms(self):
        """
        Returns term1 and term2 together as a 2-element numpy vector.
        """
        return np.array([self.term1(), self.term2()])

    def dterms(self):
        """
        Returns the 2x2 Jacobian for term1 and term2.
        """
        return np.array([[1.0, 0.0],
                         [2 * self._x / self._sqrtb, -1.0 / self._sqrtb]])

    def get_dofs(self):
        return np.array([self._x, self._y])

    def set_dofs(self, xin):
        self._x = xin[0]
        self._y = xin[1]


class RosenbrockWithFailures(Rosenbrock):
    """
    This class is similar to the Rosenbrock class, except that it
    fails (raising ObjectiveFailure) at regular intervals.  This is
    useful for testing that the simsopt infrastructure handles
    failures in the expected way.
    """

    def __init__(self, *args, fail_interval=8, **kwargs):
        self.nevals = 0
        self.fail_interval = fail_interval
        super().__init__(*args, **kwargs)

    def term1(self):
        self.nevals += 1
        if np.mod(self.nevals, self.fail_interval) == 0:
            raise ObjectiveFailure("Planned failure")

        return super().term1()


class TestObject1(Optimizable):
    """
    This is an optimizable object used for testing. It depends on two
    sub-objects, both of type Adder.
    """

    def __init__(self, val):
        self.val = val
        self.names = ['val']
        self.fixed = np.array([False])
        self.adder1 = Adder(3)
        self.adder2 = Adder(2)
        self.depends_on = ['adder1', 'adder2']

    def set_dofs(self, x):
        self.val = x[0]

    def get_dofs(self):
        return np.array([self.val])

    def J(self):
        return (self.val + 2 * self.adder1.J()) / (10.0 + self.adder2.J())

    def dJ(self):
        v = self.val
        a1 = self.adder1.J()
        a2 = self.adder2.J()
        # J = (v + 2 * a1) / (10 + a2)
        return np.concatenate((np.array([1.0 / (10.0 + a2)]),
                               np.full(self.adder1.n, 2.0 / (10.0 + a2)),
                               np.full(self.adder2.n, -(v + 2 * a1) / ((10.0 + a2) ** 2))))

    @property
    def f(self):
        """
        Same as J() but a property instead of a function.
        """
        return self.J()

    @property
    def df(self):
        """
        Same as dJ() but a property instead of a function.
        """
        return self.dJ()


class TestObject2(Optimizable):
    """
    This is an optimizable object used for testing. It depends on two
    sub-objects, both of type Adder.
    """

    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
        self.names = ['val1', 'val2']
        self.fixed = np.array([False, False])
        self.t = TestObject1(0.0)
        self.adder = Adder(2)
        self.depends_on = ['t', 'adder']

    def set_dofs(self, x):
        self.val1 = x[0]
        self.val2 = x[1]

    def get_dofs(self):
        return np.array([self.val1, self.val2])

    def J(self):
        v1 = self.val1
        v2 = self.val2
        t = self.t.J()
        a = self.adder.J()
        return v1 + a * np.cos(v2 + t)

    def dJ(self):
        v1 = self.val1
        v2 = self.val2
        a = self.adder.J()
        t = self.t.J()
        cosat = np.cos(v2 + t)
        sinat = np.sin(v2 + t)
        # Order of terms in the gradient: v1, v2, t, a
        return np.concatenate((np.array([1.0, -a * sinat]),
                               -a * sinat * self.t.dJ(),
                               cosat * self.adder.dJ()))

    @property
    def f(self):
        """
        Same as J() but a property instead of a function.
        """
        return self.J()

    @property
    def df(self):
        """
        Same as dJ() but a property instead of a function.
        """
        return self.dJ()


class Affine(Optimizable):
    """
    This class represents a random affine (i.e. linear plus constant)
    transformation from R^n to R^m.
    """

    def __init__(self, nparams, nvals):
        """
        nparams = number of independent variables.
        nvals = number of dependent variables.
        """
        self.nparams = nparams
        self.nvals = nvals
        self.A = (np.random.rand(nvals, nparams) - 0.5) * 4
        self.B = (np.random.rand(nvals) - 0.5) * 4
        self.x = np.zeros(nparams)

    def get_dofs(self):
        return self.x

    def set_dofs(self, x):
        self.x = x

    def J(self):
        return np.matmul(self.A, self.x) + self.B

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
        self.x = np.zeros(self.nparams)

    def J(self):
        self.nevals += 1
        if self.nevals == self.fail_index:
            raise ObjectiveFailure("nevals == fail_index")
        else:
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

    def __init__(self):
        self.x = np.zeros(2)

    def get_dofs(self):
        return self.x

    def set_dofs(self, x):
        assert len(x) == 2
        self.x = np.array(x)

    def J(self):
        x = self.x[0]
        y = self.x[1]
        return np.array([1.5 - x + x * y,
                         2.25 - x + x * y * y,
                         2.625 - x + x * y * y * y])
