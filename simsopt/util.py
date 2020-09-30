#!/usr/bin/env python3

"""
This module contains small utility functions and classes.
"""

import numpy as np
from .optimizable import Optimizable

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

class Struct():
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

