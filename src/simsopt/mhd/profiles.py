# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides classes to handle radial profiles of density,
temperature, pressure, and other quantities that are flux functions.
"""

import logging
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from .._core.graph_optimizable import Optimizable

logger = logging.getLogger(__name__)

class Profile(Optimizable):
    """
    Base class for radial profiles. This class should not be used
    directly - use subclasses instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, ax=None, show=True, n=100):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        s = np.linspace(0, 1, n)
        ax.plot(s, self.f(s))
        plt.xlabel('Normalized toroidal flux $s$')
        if show:
            plt.show()


class ProfilePolynomial(Profile):
    """
    A profile described by a polynomial in the normalized toroidal
    flux s.  The polynomial coefficients are dofs that are fixed by
    default.

    Args:
        data: 1D numpy array of the polynomial coefficients.
            The first coefficient is the constant term, the next coefficient
            is the linear term, etc.
    """
    def __init__(self, data):
        Optimizable.__init__(self, x0=np.array(data))
        self.fix_all()

    def f(self, s):
        """ Return the value of the profile at specified points in s. """
        return poly.polyval(s, self.local_full_x)

    def dfds(self, s):
        """ Return the d/ds derivative of the profile at specified points in s. """
        return poly.polyval(s, poly.polyder(self.local_full_x))
        
