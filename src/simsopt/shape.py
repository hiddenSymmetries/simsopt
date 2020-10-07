"""
This module provides a base class for shapes that can be
optimized. Surfaces and Coils will be subclasses of this class.
"""

import numpy as np
from .parameter import Parameter, isbool

class Shape:
    """
    Shape is a base class for shapes that can be optimized, such as
    toroidal surfaces and coils.

    This class has two properites, nfp and stelsym. They are
    implemented using the @property decorator and protected variables
    to ensure users do not set them to any type other than Parameter.

    For both the nfp and stelsym arguments, you can either specify a
    value, or you can specify a Parameter instance. In the former case
    a new Parameter will be created with that value. In the latter
    case, the specified Parameter instance will be used.
    """

    # TODO: For both the nfp and stelsym Parameters, we could add an
    # observer in this class that checks the type is correct, checks
    # nfp >= 1, etc.

    def __init__(self, nfp=1, stelsym=True):
        # Handle nfp:
        if isinstance(nfp, Parameter):
            if nfp.val < 1:
                raise ValueError("nfp val must be at least 1")
            if int(nfp.val) != nfp.val:
                raise ValueError("nfp must be an integer")
            self._nfp = nfp
        elif not isinstance(nfp, int):
            raise RuntimeError("nfp must have type int or Parameter")
        elif nfp < 1:
            raise RuntimeError("nfp must be at least 1")
        else:
            self._nfp = Parameter(nfp, min=1, name="nfp")

        # Handle stelsym:
        if isinstance(stelsym, Parameter):
            if not isbool(stelsym.val):
                raise ValueError('stelsym val must have type bool')
            self._stelsym = stelsym
        if not isbool(stelsym):
            raise RuntimeError("stelsym must have type bool")
        else:
            self._stelsym = Parameter(stelsym, name="stelsym")

    def __repr__(self):
        return "simsopt base Shape (nfp=" + str(self._nfp.val) + \
            ", stelsym=" + str(self._stelsym.val) + ")"

    @property
    def nfp(self):
        return self._nfp

    @nfp.setter
    def nfp(self, newval):
        if not isinstance(newval, Parameter):
            raise ValueError("nfp must have type Parameter")
        if newval.val < 1:
            raise ValueError("nfp val must be at least 1")
        if int(newval.val) != newval.val:
            raise ValueError("nfp must be an integer")
        self._nfp = newval

    @property
    def stelsym(self):
        return self._stelsym

    @stelsym.setter
    def stelsym(self, newval):
        if not isinstance(newval, Parameter):
            raise ValueError("stelsym must have type Parameter")
        if not isbool(newval.val):
            raise RuntimeError("stelsym must have type bool")
        self._stelsym = newval

