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
from scipy.interpolate import InterpolatedUnivariateSpline
import numbers

from .._core.optimizable import Optimizable
from .._core.descriptor import PositiveInteger
from .._core.types import RealArray

__all__ = ['Profile', 'ProfilePolynomial', 'ProfileScaled', 'ProfileSpline',
           'ProfilePressure', 'ProfileSpec']

logger = logging.getLogger(__name__)


class Profile(Optimizable):
    """
    Base class for radial profiles. This class should not be used
    directly - use subclasses instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """ Shortcut for calling f(s) """
        return self.f(*args, **kwargs)

    def plot(self, ax=None, show=True, n=100):
        """
        Plot the profile using matplotlib.

        Args:
            ax: The axis object on which to plot. If ``None``, a new figure will be created.
            show: Whether to call matplotlib's ``show()`` function.
            n: The number of grid points in s to show.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        s = np.linspace(0, 1, n)
        ax.plot(s, self.f(s))
        plt.xlabel('Normalized toroidal flux $s$')
        if show:
            plt.show()


class ProfileSpec(Profile):
    """
    A profile described by an array of size Nvol

    Args:
        data: 1D numpy array containing the profile value in each volume
        cumulative: Set to True if the profile is cumulative, i.e. if the value 
            in volume lvol is the integrated quantity from the axis to volume lvol.
            Only the toroidal flux, poloidal flux and the volume currents are 
            cumulative quantities in SPEC input file. False by default.
    """

    def __init__(self, data, cumulative: bool = False, psi_edge: float = None):
        super().__init__(x0=np.array(data))
        self.local_fix_all()
        self.cumulative = cumulative
        self.psi_edge = psi_edge

    def f(self, lvol: int):
        """
        Return the value of the profile in volume lvol

        Args:
            lvol: int, list or np.array of int, between 0 and Mvol
        """

        # If input is a integer, make an np.array
        if isinstance(lvol, numbers.Number):
            lvol = np.array([lvol])

        # If input are floats, make integer out of them
        lvol = np.array([int(l) for l in lvol])

        # Check that volume index is within bounds
        if (lvol < 0).any():
            raise ValueError('lvol should be larger or equal than zero')
        if (lvol >= self.local_full_x.size).any():
            raise ValueError('lvol should be smaller than Mvol')

        # Return value
        return self.local_full_x[lvol]

    def dfds(self, lvol):
        """
        Returns the derivative of the profile w.r.t s accross interface. 
        The derivative is returned at the interface lvol, with
        the innermost interface being lvol=1. (Volume lvol is bounded
        by interface lvol and lvol+1, with innermost volume being lvol=0)

        Here :math:`s` is defined as :math:`s = \psi_t/\psi_{edge}`. Thus,

        .. math::

            dp/ds = \sum_l [[p]]_l \psi_{edge} \delta(\psi_t-\psi_{t,l})

        with p the profile, and the sum is on the interfaces.

        Args:
            lvol: int, list or np.array of int, between 1 and Mvol-1. 
        """
        # If input is a integer, make an np.array
        if isinstance(lvol, numbers.Number):
            lvol = np.array([lvol])

        # If input are floats, make integer out of them
        lvol = np.array([int(l) for l in lvol])

        # Check that volume index is within bounds
        if (lvol < 0).any():
            raise ValueError('lvol should be larger or equal than zero')
        if (lvol >= self.local_full_x.size-1).any():
            raise ValueError('lvol should be smaller than Mvol-1')
        if self.psi_edge is None:
            raise ValueError('Need to provide psi_edge to perform derivatives')

        lvolin = [l-1 for l in lvol]
        x_out = self.local_full_x[lvol]
        x_in = self.local_full_x[lvolin]

        return (x_out-x_in) * self.psi_edge


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
        super().__init__(x0=np.array(data))
        self.local_fix_all()

    def f(self, s):
        """ Return the value of the profile at specified points in s. """
        return poly.polyval(s, self.local_full_x)

    def dfds(self, s):
        """ Return the d/ds derivative of the profile at specified points in s. """
        return poly.polyval(s, poly.polyder(self.local_full_x))


class ProfileScaled(Profile):
    """
    A Profile which is equivalent to another Profile object but scaled
    by a constant. This constant is an optimizable dof, which is fixed by default.

    Args:
        base: A Profile object to scale
        scalefac: A number by which the base profile will be scaled.
    """

    def __init__(self, base, scalefac):
        self.base = base
        super().__init__(
            x0=np.array(
                [scalefac]),
            names=['scalefac'],
            depends_on=[base])
        self.local_fix_all()

    def f(self, s):
        """ Return the value of the profile at specified points in s. """
        return self.local_full_x[0] * self.base.f(s)

    def dfds(self, s):
        """ Return the d/ds derivative of the profile at specified points in s. """
        return self.local_full_x[0] * self.base.dfds(s)


class ProfileSpline(Profile):
    """
    A Profile that uses spline interpolation via
    `scipy.interpolate.InterpolatedUnivariateSpline
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html>`_

    The ``f`` data are optimizable dofs, which are fixed by default.

    Args:
        s: A 1d array with the x coordinates for the spline.
        f: A 1d array with the y coordinates for the spline.
        degree: The polynomial degree of the spline. Must be in ``[1, 2, 3, 4, 5]``.
    """

    degree = PositiveInteger()

    def __init__(self, s, f, degree=3):
        self.s = s
        self.degree = degree
        super().__init__(x0=f)
        self.local_fix_all()

    def f(self, s):
        """ Return the value of the profile at specified points in s. """
        return InterpolatedUnivariateSpline(
            self.s, self.full_x, k=self.degree)(s)

    def dfds(self, s):
        """ Return the d/ds derivative of the profile at specified points in s. """
        return InterpolatedUnivariateSpline(
            self.s, self.full_x, k=self.degree).derivative()(s)

    def resample(self, new_s, degree=None):
        """
        Return a new ``ProfileSpline`` object that has different grid points (spline nodes).
        The data from the old s grid will be interpolated onto the new s grid.

        Args:
            new_s: A 1d array representing the x coordinates of the new ``ProfileSpline``.
            degree: The polynomial degree used for the new ``ProfileSpline`` object.
                If ``None``, the degree of the original ``ProfileSpline`` will be used.

        Returns:
            A new :obj:`ProfileSpline` object, in which the data have been resampled onto ``new_s``.
        """
        new_degree = self.degree
        if degree is not None:
            new_degree = degree
        return ProfileSpline(new_s, self.f(new_s), degree=new_degree)


class ProfilePressure(Profile):
    r"""
    A Profile :math:`f(s)` which is determined by other profiles :math:`f_j(s)` as follows:

    .. math::

        f(s) = \sum_j f_{2j}(s) f_{2j+1}(s).

    This is useful for creating a pressure profile in terms of density
    and temperature profiles, with any number of species. Typical
    usage is as follows::

        ne = ProfilePolynomial(1.0e20 * np.array([1.0, 0.0, 0.0, 0.0, -1.0]))
        Te = ProfilePolynomial(8.0e3 * np.array([1.0, -1.0]))
        nH = ne
        TH = ProfilePolynomial(7.0e3 * np.array([1.0, -1.0]))
        pressure = ProfilePressure(ne, Te, nH, TH)

    This class does not have any optimizable dofs.

    Args:
        args: An even number of Profile objects.
    """

    def __init__(self, *args):
        if len(args) == 0:
            raise ValueError(
                'At least one density and temperature profile must be provided.')
        if len(args) % 2 == 1:
            raise ValueError(
                'The number of input profiles for a ProfilePressure object must be even')
        super().__init__(depends_on=args)

    def f(self, s):
        """ Return the value of the profile at specified points in s. """
        total = 0
        for j in range(int(len(self.parents) / 2)):
            total += self.parents[2 * j](s) * self.parents[2 * j + 1](s)
        return total

    def dfds(self, s):
        """ Return the d/ds derivative of the profile at specified points in s. """
        total = 0
        for j in range(int(len(self.parents) / 2)):
            total += self.parents[2 * j].f(s) * self.parents[2 * j + 1].dfds(s)\
                + self.parents[2 * j].dfds(s) * self.parents[2 * j + 1](s)
        return total
