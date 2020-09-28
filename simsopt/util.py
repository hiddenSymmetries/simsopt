#!/usr/bin/env python3

"""
This module contains small utility functions and classes.
"""

import numpy as np

def isbool(val):
    """
    Test whether val is any boolean type, either the native python
    bool or numpy's bool_.
    """
    return isinstance(val, bool) or isinstance(val, np.bool_)

def isnumber(val):
    """
    Test whether val is any kind of number, including both native
    python types or numpy types.
    """
    return isinstance(val, int) or isinstance(val, float) or \
        isinstance(val, np.int_) or isinstance(val, np.float)

class Identity():
    """
    This class represents a term in an objective function which is just
    the identity. It has one degree of freedom, and the output of the function
    is equal to this degree of freedom.
    """
    def __init__(self, x=0.0):
        self.x = x
        self.fixed = np.full(1, False)

    def f(self):
        return self.x

    def get_dofs(self):
        return np.array([self.x])

    def set_dofs(self, xin):
        self.x = xin[0]
