# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides functions to compute the bootstrap current
"""

import logging
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize
from scipy.integrate import quad
from .._core.util import Struct
from ..util.constants import ELEMENTARY_CHARGE
from .profiles import Profile, ProfilePolynomial

logger = logging.getLogger(__name__)


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


def j_dot_B_Redl(s, ne, Te, Ti, Zeff, R, iota, G, epsilon, f_t, psi_edge, helicity_N):
    """
    Compute the bootstrap current (specifically
    :math:`\left<\vec{J}\cdot\vec{B}\right>`) using the formulae in
    Redl et al, Physics of Plasmas (2021).

    The quantity <j dot B> is computed at all surfaces s that are
    available in the booz object.

    The profiles of ne, Te, Ti, and Zeff should all be Profile
    objects, meaning they have ``__call__`` and ``dfds()`` functions. If
    Zeff==None, a constant 1 is assumed. If Zeff is a float, a
    constant profile will be assumed.

    ne should have units of 1/m^3. Ti and Te should have units of eV.

    epsilon, f_t, iota, and R should be 1d arrays evaluated on the s grid.

    Args:
        booz: An instance of :obj:`simsopt.mhd.boozer.Boozer`
        ne: A
        psi_edge: The toroidal flux in Webers divided by (2pi) at the boundary s=1
    Returns:
        jdotB:
        details: An object with intermediate quantities as attributes
    """
    if Zeff is None:
        Zeff = ProfilePolynomial(1.0)
    if not isinstance(Zeff, Profile):
        # Zeff is presumably a number. Convert it to a constant profile.
        Zeff = ProfilePolynomial([Zeff])

    # Evaluate profiles on the grid:
    ne_s = ne(s)
    Te_s = Te(s)
    Ti_s = Ti(s)
    Zeff_s = Zeff(s)
    ni_s = ne_s / Zeff_s
    pe_s = ne_s * Te_s
    pi_s = ni_s * Ti_s
    d_ne_d_s = ne.dfds(s)
    d_Te_d_s = Te.dfds(s)
    d_Ti_d_s = Ti.dfds(s)

    # Profiles may go to 0 at s=1, so exclude the last 2 grid points:
    if np.any(ne_s[:-2] < 1e17):
        logging.warning('ne is surprisingly low. It should have units 1/meters^3')
    if np.any(Te_s[:-2] < 500):
        logging.warning('Te is surprisingly low. It should have units of eV')
    if np.any(Ti_s[:-2] < 50):
        logging.warning('Ti is surprisingly low. It should have units of eV')

    # Eq (18d)-(18e) in Sauter.
    # Check that we do not need to convert units of n or T!
    ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_s) / Te_s)
    ln_Lambda_ii = 30 - np.log(Zeff_s ** 3 * np.sqrt(ni_s) / (Ti_s ** 1.5))
    logging.debug(f'ln Lambda_e: {ln_Lambda_e}')
    logging.debug(f'ln Lambda_ii: {ln_Lambda_ii}')

    # Eq (18b)-(18c) in Sauter:
    nu_e = (6.921e-18) * R * ne_s * Zeff_s * ln_Lambda_e \
        / (iota * Te_s * Te_s * (epsilon ** 1.5))
    nu_i = (4.90e-18) * R * ni_s * (Zeff_s ** 4) * ln_Lambda_ii \
        / (iota * Ti_s * Ti_s * (epsilon ** 1.5))

    # Redl eq (11):
    X31 = f_t / (1 + (0.67 * (1 - 0.7 * f_t) * np.sqrt(nu_e)) / (0.56 + 0.44 * Zeff_s) \
                 + (0.52 + 0.086 * np.sqrt(nu_e)) * (1 + 0.87 * f_t) * nu_e / (1 + 1.13 * np.sqrt(Zeff_s - 1)))

    # Redl eq (10):
    Zfac = Zeff_s ** 1.2 - 0.71
    L31 = (1 + 0.15 / Zfac) * X31 \
        - 0.22 / Zfac * (X31 ** 2) \
        + 0.01 / Zfac * (X31 ** 3) \
        + 0.06 / Zfac * (X31 ** 4)

    # Redl eq (14):
    X32e = f_t / ((1 + 0.23 * (1 - 0.96 * f_t) * np.sqrt(nu_e) / np.sqrt(Zeff_s) \
                   + 0.13 * (1 - 0.38 * f_t) * nu_e / (Zeff_s * Zeff_s) \
                   * (np.sqrt(1 + 2 * np.sqrt(Zeff_s - 1)) \
                      + f_t * f_t * np.sqrt((0.075 + 0.25 * (Zeff_s - 1) ** 2) * nu_e))))

    # Redl eq (13):
    F32ee = (0.1 + 0.6 * Zeff_s) * (X32e - X32e ** 4) \
        / (Zeff_s * (0.77 + 0.63 * (1 + (Zeff_s - 1) ** 1.1))) \
        + 0.7 / (1 + 0.2 * Zeff_s) * (X32e ** 2 - X32e ** 4 - 1.2 * (X32e ** 3 - X32e ** 4)) \
        + 1.3 / (1 + 0.5 * Zeff_s) * (X32e ** 4)

    # Redl eq (16):
    X32ei = f_t / (1 + 0.87 * (1 + 0.39 * f_t) * np.sqrt(nu_e) / (1 + 2.95 * (Zeff_s - 1) ** 2) \
                   + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff_s - 1)))

    # Redl eq (15):
    F32ei = -(0.4 + 1.93 * Zeff_s) / (Zeff_s * (0.8 + 0.6 * Zeff_s)) * (X32ei - X32ei ** 4) \
        + 5.5 / (1.5 + 2 * Zeff_s) * (X32ei ** 2 - X32ei ** 4 - 0.8 * (X32ei ** 3 - X32ei ** 4)) \
        - 1.3 / (1 + 0.5 * Zeff_s) * (X32ei ** 4)

    # Redl eq (12):
    L32 = F32ei + F32ee

    # Redl eq (19):
    L34 = L31

    # Redl eq (20):
    alpha0 = -(0.62 + 0.055 * (Zeff_s - 1)) * (1 - f_t) \
        / ((0.53 + 0.17 * (Zeff_s - 1)) * (1 - (0.31 - 0.065 * (Zeff_s - 1)) * f_t - 0.25 * f_t * f_t))
    # Redl eq (21):    
    alpha = ((alpha0 + 0.7 * Zeff_s * np.sqrt(f_t * nu_i)) / (1 + 0.18 * np.sqrt(nu_i)) \
             - 0.002 * nu_i * nu_i * (f_t ** 6)) \
        / (1 + 0.004 * nu_i * nu_i * (f_t ** 6))

    # Factor of ELEMENTARY_CHARGE is included below to convert temperatures from eV to J
    dnds_term = -G * ELEMENTARY_CHARGE * (ne_s * Te_s + ni_s * Ti_s) * L31 * (d_ne_d_s / ne_s) / (psi_edge * (iota - helicity_N))
    dTeds_term = -G * ELEMENTARY_CHARGE * pe_s * (L31 + L32) * (d_Te_d_s / Te_s) / (psi_edge * (iota - helicity_N))
    dTids_term = -G * ELEMENTARY_CHARGE * pi_s * (L31 + L34 * alpha) * (d_Ti_d_s / Ti_s) / (psi_edge * (iota - helicity_N))
    jdotB = dnds_term + dTeds_term + dTids_term

    details = Struct()
    nu_e_star = nu_e
    nu_i_star = nu_i
    variables = ['ne_s', 'ni_s', 'Zeff_s', 'Te_s', 'Ti_s',
                 'd_ne_d_s', 'd_Te_d_s', 'd_Ti_d_s',
                 'ln_Lambda_e', 'ln_Lambda_ii', 'nu_e_star', 'nu_i_star',
                 'X31', 'X32e', 'X32ei', 'F32ee', 'F32ei',
                 'L31', 'L32', 'L34', 'alpha0', 'alpha',
                 'dnds_term', 'dTeds_term', 'dTids_term']
    for v in variables:
        details.__setattr__(v, eval(v))

    return jdotB, details


def vmec_j_dot_B_Redl(vmec, surfaces, ne, Te, Ti, Zeff, helicity_N, ntheta=64, nphi=65, plot=False):
    """
    Evaluate the Redl bootstrap current formula for a vmec configuration.

    Args:
        plot: Make a plot of many of the quantities computed.
    """
    vmec.run()

    ns = len(surfaces)
    nfp = vmec.wout.nfp
    psi_edge = -vmec.wout.phi[-1] / (2 * np.pi)

    # First, interpolate in s to get the quantities we need on the surfaces we need.
    method = 'linear'

    interp = interp1d(vmec.s_half_grid, vmec.wout.iotas[1:], fill_value="extrapolate")
    iota = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bvco[1:], fill_value="extrapolate")
    G = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.gmnc[:, 1:], fill_value="extrapolate")
    gmnc = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bmnc[:, 1:], fill_value="extrapolate")
    bmnc = interp(surfaces)

    theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi1d, theta1d)
    phi3d = phi2d.reshape((ntheta, nphi, 1))
    theta3d = theta2d.reshape((ntheta, nphi, 1))

    myshape = (ntheta, nphi, ns)
    modB = np.zeros(myshape)
    sqrtg = np.zeros(myshape)
    for jmn in range(len(vmec.wout.xm_nyq)):
        m = vmec.wout.xm_nyq[jmn]
        n = vmec.wout.xn_nyq[jmn]
        angle = m * theta3d - n * phi3d
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        modB += np.kron(bmnc[jmn, :].reshape((1, 1, ns)), cosangle)
        sqrtg += np.kron(gmnc[jmn, :].reshape((1, 1, ns)), cosangle)

    Bmin, Bmax, epsilon, fsa_B2, f_t = compute_trapped_fraction(modB, sqrtg)

    jdotB, details = j_dot_B_Redl(surfaces, ne, Te, Ti, Zeff, vmec.wout.Rmajor_p, iota, G, epsilon, f_t, psi_edge, helicity_N)

    # Add extra info to the return structure
    variables = ['Bmin', 'Bmax', 'epsilon', 'fsa_B2', 'f_t',
                 'modB', 'sqrtg', 'G', 'iota', 'surfaces', 'psi_edge', 'theta1d', 'phi1d']
    for v in variables:
        details.__setattr__(v, eval(v))

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7))
        plt.rcParams.update({'font.size': 8})
        nrows = 4
        ncols = 6
        variables = ['Bmax', 'Bmin', 'epsilon', 'fsa_B2', 'f_t', 'iota', 'G',
                     'details.ne_s', 'details.ni_s', 'details.Zeff_s', 'details.Te_s', 'details.Ti_s',
                     'details.ln_Lambda_e', 'details.ln_Lambda_ii',
                     'details.nu_e_star', 'details.nu_i_star',
                     'details.dnds_term', 'details.dTeds_term', 'details.dTids_term',
                     'details.L31', 'details.L32', 'details.alpha', 'jdotB']
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(surfaces, eval(variable))
            plt.title(variable)
            plt.xlabel('s')
        plt.tight_layout()
        plt.show()

    return jdotB, details
