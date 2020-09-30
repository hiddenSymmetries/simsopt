"""
This module provides a few minimal optimizable objects, each
representing a function. These functions are mostly used for testing.
"""

import numpy as np
import logging
from .optimizable import Optimizable

class Identity(Optimizable):
    """
    This class represents a term in an objective function which is just
    the identity. It has one degree of freedom, and the output of the function
    is equal to this degree of freedom.
    """
    def __init__(self, x=0.0):
        self.x = x
        self.fixed = np.full(1, False)
        self.names = ['x']

    def J(self):
        return self.x

    @property
    def f(self):
        """
        Same as the function J(), but a property instead of a function.
        """
        return self.x
    
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
        self.x = np.zeros(n)
        self.fixed = np.full(n, False)        

    def J(self):
        """
        Returns the sum of the degrees of freedom.
        """
        return np.sum(self.x)
        
    @property
    def f(self):
        """
        Same as the function J(), but a property instead of a function.
        """
        return self.J()
    
    def get_dofs(self):
        return self.x

    def set_dofs(self, xin):
        self.x = np.array(xin)

class Rosenbrock(Optimizable):
    """
    This class defines a minimal object that can be optimized.
    """

    def __init__(self, b=100.0, x=0.0, y=0.0):
        self._logger = logging.getLogger(__name__)
        self._sqrtb = np.sqrt(b)
        self._names = ['x', 'y']
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
    
    def f(self):
        """
        Returns the total function, squaring and summing the two terms.
        """
        t1 = self.term1()
        t2 = self.term2()
        return t1 * t1 + t2 * t2

    def get_dofs(self):
        return np.array([self._x, self._y])

    def set_dofs(self, xin):
        self._x = xin[0]
        self._y = xin[1]

