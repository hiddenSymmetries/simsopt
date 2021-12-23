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
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.integrate import quad
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
        super().__init__(x0=np.array(data))
        self.fix_all()

    def f(self, s):
        """ Return the value of the profile at specified points in s. """
        return poly.polyval(s, self.local_full_x)

    def dfds(self, s):
        """ Return the d/ds derivative of the profile at specified points in s. """
        return poly.polyval(s, poly.polyder(self.local_full_x))


def compute_trapped_fraction(modB, sqrtg):
    """
    Compute the effective fraction of trapped particles, which enters
    several formulae for neoclassical transport, as well as several
    quantities that go into its calculation.  The input data can be
    provided on a uniform grid of arbitrary toroidal and poloidal
    angles that need not be straight-field-line angles.

    The effective inverse aspect ratio epsilon is defined by

    .. math::
        \frac{Bmax}{Bmin} = \frac{1 + \epsilon}{1 - \epsilon}

    This definition is motivated by the fact that this formula would
    be true in the case of circular cross-section surfaces in
    axisymmetry with :math:`B \propto 1/R` and :math:`R = (1 +
    \epsilon \cos\theta) R_0`.

    Args:
        modB: 3D array of size (ntheta, nphi, ns) representing |B|.
        sqrtg: 3D array of size (ntheta, nphi, ns) representing the Jacobian.
    Returns:
        3-element tuple containing three 1D arrays, corresponding to radial grid points
            Bmin: minimum of |B| on each surface
            Bmax: maximum of |B| on each surface
            epsilon: A measure of the inverse aspect ratio
            fsa_B2: <B^2>, where < > denotes a flux surface average.
            f_t: The effective trapped fraction
    """
    ntheta = modB.shape[0]
    nphi = modB.shape[1]
    ns = modB.shape[2]
    fourpisq = 4 * np.pi * np.pi
    dVds = np.mean(sqrtg, axis=(0, 1)) / fourpisq
    fsa_B2 = np.mean(modB * modB * sqrtg, axis=(0, 1)) / (fourpisq * dVds)

    epsilon = np.zeros(ns)
    f_t = np.zeros(ns)
    Bmin = np.zeros(ns)
    Bmax = np.zeros(ns)

    # Make a slightly enlarged version of the input array with the
    # first row and column appended at the ends, for periodicity.
    modB_big = np.zeros((ntheta + 1, nphi + 1, ns))
    modB_big[:ntheta, :nphi, :] = modB
    modB_big[-1, :nphi, :] = modB[0, :, :]
    modB_big[:, -1, :] = modB_big[:, 0, :]

    theta = np.arange(ntheta + 1)
    phi = np.arange(nphi + 1)
    for js in range(ns):
        index_of_min = np.unravel_index(np.argmin(modB_big[:, :, js]), modB_big.shape[:2])
        index_of_max = np.unravel_index(np.argmax(modB_big[:, :, js]), modB_big.shape[:2])
        modB_spline = RectBivariateSpline(theta, phi, modB_big[:, :, js])
        soln = minimize(lambda x: np.ravel(modB_spline(x[0], x[1])),
                        index_of_min,
                        bounds=((0, ntheta), (0, nphi)))
        modBmin = soln.fun
        soln = minimize(lambda x: -np.ravel(modB_spline(x[0], x[1])),
                        index_of_max,
                        bounds=((0, ntheta), (0, nphi)))
        modBmax = -soln.fun
        Bmin[js] = modBmin
        Bmax[js] = modBmax
        w = modBmax / modBmin
        epsilon[js] = (w - 1) / (w + 1)

        def integrand(lambd):
            # This function gives lambda / <sqrt(1 - lambda B)>:
            return lambd / (np.mean(np.sqrt(1 - lambd * modB[:, :, js]) * sqrtg[:, :, js]) \
                            / (fourpisq * dVds[js]))

        integral = quad(integrand, 0, 1 / modBmax)
        f_t[js] = 1 - 0.75 * fsa_B2[js] * integral[0]

    return Bmin, Bmax, epsilon, fsa_B2, f_t


def quasisymmetry_filtered_trapped_fraction(booz, helicity_m, helicity_n):
    """
    Compute quantities needed for the Redl bootstrap current formula.

    Args:
        booz: An instance of :obj:`simsopt.mhd.boozer.Boozer`
        helicity_m: The poloidal mode number of the desired symmetry, usually 1.
        helicity_n: The toroidal mode number of the desired symmetry.
    Returns:
        3-element tuple containing three 1D arrays, corresponding to radial grid points
            epsilon: A measure of the inverse aspect ratio
            fsa_B2: <B^2>, where < > denotes a flux surface average.
            f_t: The effective trapped fraction
    """
    pass


def j_dot_B_Redl(booz, ne, Te, Ti, Zeff, R, iota):
    """
    Compute the bootstrap current using the formulae in Redl et al,
    Physics of Plasmas (2021).

    The quantity <j dot B> is computed at all surfaces s that are
    available in the booz object.

    The profiles of ne, Te, Ti, and Zeff should all be Profile
    objects, meaning they have ``f()`` and ``dfds()`` functions. If
    Zeff==None, a constant 1 is assumed. If Zeff is a float, a
    constant profile will be assumed.

    Args:
        booz: An instance of :obj:`simsopt.mhd.boozer.Boozer`
        ne: A
    Returns:
        jdotB: 
    """
    if Zeff == None:
        Zeff = ProfilePolynomial(1.0)
    if not isinstance(Zeff, Profile):
        # Zeff is presumably a number. Convert it to a constant profile.
        Zeff = ProfilePolynomial(Zeff)

    booz.run()
