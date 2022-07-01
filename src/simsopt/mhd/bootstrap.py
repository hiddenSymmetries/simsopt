# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides functions to compute the bootstrap current
"""

import logging

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize, Bounds
from scipy.integrate import quad

from .._core.optimizable import Optimizable
from .._core.util import Struct
from ..util.constants import ELEMENTARY_CHARGE
from .profiles import Profile, ProfilePolynomial

__all__ = ['compute_trapped_fraction', 'j_dot_B_Redl', 'RedlGeomVmec',
           'RedlGeomBoozer', 'VmecRedlBootstrapMismatch']

logger = logging.getLogger(__name__)


def compute_trapped_fraction(modB, sqrtg):
    r"""
    Compute the effective fraction of trapped particles, which enters
    several formulae for neoclassical transport, as well as several
    quantities that go into its calculation.  The input data can be
    provided on a uniform grid of arbitrary toroidal and poloidal
    angles that need not be straight-field-line angles.

    The trapped fraction ``f_t`` has a standard definition in neoclassical theory:

    .. math::
        f_t = 1 - \frac{3}{4} \left< B^2 \right> \int_0^{1/Bmax}
            \frac{\lambda\; d\lambda}{\left< \sqrt{1 - \lambda B} \right>}

    where :math:`\left< \ldots \right>` is a flux surface average.

    The effective inverse aspect ratio epsilon is defined by

    .. math::
        \frac{Bmax}{Bmin} = \frac{1 + \epsilon}{1 - \epsilon}

    This definition is motivated by the fact that this formula would
    be true in the case of circular cross-section surfaces in
    axisymmetry with :math:`B \propto 1/R` and :math:`R = (1 +
    \epsilon \cos\theta) R_0`.

    Args:
        modB: 2D array of size (ntheta, ns) or 3D array of size
            (ntheta, nphi, ns) with :math:`|B|` on the grid points.
        sqrtg: 2D array of size (ntheta, ns) or 3D array of size
            (ntheta, nphi, ns) with the Jacobian
            :math:`1/(\nabla s \times\nabla\theta\cdot\nabla\phi)`
            on the grid points.

    Returns:
        Tuple containing

        - **Bmin**: A 1D array, with the minimum of :math:`|B|` on each surface.
        - **Bmax**: A 1D array, with the maximum of :math:`|B|` on each surface.
        - **epsilon**: A 1D array, with the effective inverse aspect ratio on each surface.
        - **fsa_B2**: A 1D array with :math:`\left<B^2\right>` on each surface,
          where :math:`\left< \ldots \right>` denotes a flux surface average.
        - **fsa_1overB**: A 1D array with :math:`\left<1/B\right>` on each surface,
          where :math:`\left< \ldots \right>` denotes a flux surface average.
        - **f_t**: A 1D array, with the effective trapped fraction on each surface.
    """
    assert modB.shape == sqrtg.shape
    ntheta = modB.shape[0]
    ns = modB.shape[-1]
    epsilon = np.zeros(ns)
    f_t = np.zeros(ns)
    Bmin = np.zeros(ns)
    Bmax = np.zeros(ns)

    if modB.ndim == 3:
        # Input arrays are 3D, with phi dependence.

        nphi = modB.shape[1]
        fourpisq = 4 * np.pi * np.pi
        dVds = np.mean(sqrtg, axis=(0, 1)) / fourpisq
        fsa_B2 = np.mean(modB * modB * sqrtg, axis=(0, 1)) / (fourpisq * dVds)
        fsa_1overB = np.mean(sqrtg / modB, axis=(0, 1)) / (fourpisq * dVds)

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

    elif modB.ndim == 2:
        # Input arrays are 2D, with no phi dependence.

        twopi = 2 * np.pi
        dVds = np.mean(sqrtg, axis=0) / twopi
        fsa_B2 = np.mean(modB * modB * sqrtg, axis=0) / (twopi * dVds)
        fsa_1overB = np.mean(sqrtg / modB, axis=0) / (twopi * dVds)

        # Make a slightly enlarged version of the input array with the
        # first row and column appended at the ends, for periodicity.
        modB_big = np.zeros((ntheta + 1, ns))
        modB_big[:ntheta, :] = modB
        modB_big[-1, :] = modB[0, :]

        theta = np.arange(ntheta + 1)
        for js in range(ns):
            index_of_min = np.argmin(modB_big[:, js])
            index_of_max = np.argmax(modB_big[:, js])
            modB_spline = interp1d(theta, modB_big[:, js], kind='cubic')
            bounds = Bounds(0, ntheta)
            soln = minimize(modB_spline,
                            [index_of_min],
                            bounds=bounds)
            modBmin = soln.fun
            soln = minimize(lambda x: -modB_spline(x[0]),
                            [index_of_max],
                            bounds=bounds)
            modBmax = -soln.fun
            Bmin[js] = modBmin
            Bmax[js] = modBmax
            w = modBmax / modBmin
            epsilon[js] = (w - 1) / (w + 1)

            def integrand(lambd):
                # This function gives lambda / <sqrt(1 - lambda B)>:
                return lambd / (np.mean(np.sqrt(1 - lambd * modB[:, js]) * sqrtg[:, js]) \
                                / (twopi * dVds[js]))

            integral = quad(integrand, 0, 1 / modBmax)
            f_t[js] = 1 - 0.75 * fsa_B2[js] * integral[0]

    else:
        raise ValueError('Input arrays must be 2D or 3D')

    logging.debug(f'Bmin: {Bmin}  Bmax: {Bmax}  epsilon: {epsilon}  '
                  f'fsa_B2: {fsa_B2}  fsa_1overB: {fsa_1overB}  f_t: {f_t}')
    return Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t


def j_dot_B_Redl(ne, Te, Ti, Zeff, helicity_n=None, s=None, G=None, R=None, iota=None,
                 epsilon=None, f_t=None, psi_edge=None, nfp=None,
                 geom=None, plot=False):
    r"""
    Compute the bootstrap current (specifically
    :math:`\left<\vec{J}\cdot\vec{B}\right>`) using the formulae in
    Redl et al, Physics of Plasmas 28, 022502 (2021).

    The profiles of ne, Te, Ti, and Zeff should all be instances of
    subclasses of :obj:`simsopt.mhd.profiles.Profile`, i.e. they should
    have ``__call__()`` and ``dfds()`` functions. If ``Zeff == None``, a
    constant 1 is assumed. If ``Zeff`` is a float, a constant profile will
    be assumed.

    ``ne`` should have units of 1/m^3. ``Ti`` and ``Te`` should have
    units of eV.

    Geometric data can be specified in one of two ways. In the first
    approach, the arguments ``s``, ``G``, ``R``, ``iota``,
    ``epsilon``, ``f_t``, ``psi_edge``, and ``nfp`` are specified,
    while the argument ``geom`` is not. In the second approach, the
    argument ``geom`` is set to an instance of either
    :obj:`RedlGeomVmec` or :obj:`RedlGeomBoozer`, and this object will
    be used to set all the other geometric quantities. In this case,
    the arguments ``s``, ``G``, ``R``, ``iota``, ``epsilon``, ``f_t``,
    ``psi_edge``, and ``nfp`` should not be specified.

    The input variable ``s`` is a 1D array of values of normalized
    toroidal flux.  The input arrays ``G``, ``R``, ``iota``,
    ``epsilon``, and ``f_t``, should be 1d arrays evaluated on this
    same ``s`` grid. The bootstrap current
    :math:`\left<\vec{J}\cdot\vec{B}\right>` will be computed on this
    same set of flux surfaces.

    If you provide a :obj:`RedlGeomBoozer` object for ``geom``, then
    it is not necessary to specify the argument ``helicity_n`` here,
    in which case ``helicity_n`` will be taken from ``geom``.

    Args:
        ne: A :obj:`~simsopt.mhd.profiles.Profile` object with the electron density profile.
        Te: A :obj:`~simsopt.mhd.profiles.Profile` object with the electron temperature profile.
        Ti: A :obj:`~simsopt.mhd.profiles.Profile` object with the ion temperature profile.
        Zeff: A :obj:`~simsopt.mhd.profiles.Profile` object with the profile of the average
            impurity charge :math:`Z_{eff}`. Or, a single number can be provided if this profile is constant.
            Or, if ``None``, Zeff = 1 will be used.
        helicity_n: 0 for quasi-axisymmetry, or +/- 1 for quasi-helical symmetry.
            This quantity is used to apply the quasisymmetry isomorphism to map the collisionality
            and bootstrap current from the tokamak expressions to quasi-helical symmetry.
        s: A 1D array of values of normalized toroidal flux.
        G: A 1D array with the flux function multiplying :math:`\nabla\varphi` in the Boozer covariant representation,
            equivalent to :math:`R B_{toroidal}` in axisymmetry.
        R: A 1D array with the effective major radius to use when evaluating
            the collisionality in the Sauter/Redl formulae.
        iota: A 1D array with the rotational transform.
        epsilon: A 1D array with the effective inverse aspect ratio to use for
            evaluating the collisionality in the Sauter/Redl formulae.
        f_t: A 1D array with the effective trapped fraction.
        psi_edge: The toroidal flux (in Webers) divided by (2pi) at the boundary s=1
        nfp: The number of field periods. Irrelevant for axisymmetry or quasi-axisymmetry;
            matters only if ``helicity_n`` is not 0.
        geom: Optional. An instance of either :obj:`RedlGeomVmec` or :obj:`RedlGeomBoozer`.
        plot: Whether to make a plot of many of the quantities computed.

    Returns:
        Tuple containing

        - **jdotB**: A 1D array containing the bootstrap current :math:`\left<\vec{J}\cdot\vec{B}\right>`
          on the specified flux surfaces.
        - **details**: An object holding intermediate quantities from the computation
          (e.g. L31, L32, alpha) as attributes
    """
    if geom is not None:
        if (s is not None) or (G is not None) or (R is not None) \
           or (iota is not None) or (epsilon is not None) or (psi_edge is not None) \
           or (f_t is not None) or (nfp is not None):
            raise ValueError('Geometry is being specified two ways. Pick one or the other.')
        geom_data = geom()
        s = geom_data.surfaces
        G = geom_data.G
        R = geom_data.R
        iota = geom_data.iota
        epsilon = geom_data.epsilon
        psi_edge = geom_data.psi_edge
        f_t = geom_data.f_t
        nfp = geom_data.nfp

    if helicity_n is None:
        helicity_n = geom.helicity_n

    helicity_N = nfp * helicity_n
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
    if np.any(Te_s[:-2] < 50):
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
    geometry_factor = abs(R / (iota - helicity_N))
    nu_e = geometry_factor * (6.921e-18) * ne_s * Zeff_s * ln_Lambda_e \
        / (Te_s * Te_s * (epsilon ** 1.5))
    nu_i = geometry_factor * (4.90e-18) * ni_s * (Zeff_s ** 4) * ln_Lambda_ii \
        / (Ti_s * Ti_s * (epsilon ** 1.5))
    if np.any(nu_e[:-2] < 1e-6):
        logging.warning('nu_*e is surprisingly low. Check that the density and temperature are correct.')
    if np.any(nu_i[:-2] < 1e-6):
        logging.warning('nu_*i is surprisingly low. Check that the density and temperature are correct.')
    if np.any(nu_e[:-2] > 1e5):
        logging.warning('nu_*e is surprisingly large. Check that the density and temperature are correct.')
    if np.any(nu_i[:-2] > 1e5):
        logging.warning('nu_*i is surprisingly large. Check that the density and temperature are correct.')

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
    variables = ['s', 'ne_s', 'ni_s', 'Zeff_s', 'Te_s', 'Ti_s',
                 'd_ne_d_s', 'd_Te_d_s', 'd_Ti_d_s',
                 'ln_Lambda_e', 'ln_Lambda_ii', 'nu_e_star', 'nu_i_star',
                 'X31', 'X32e', 'X32ei', 'F32ee', 'F32ei',
                 'L31', 'L32', 'L34', 'alpha0', 'alpha',
                 'dnds_term', 'dTeds_term', 'dTids_term', 'jdotB']
    for v in variables:
        details.__setattr__(v, eval(v))

    if geom is not None:
        # Copy geom_data into details:
        for v in dir(geom_data):
            if v[0] != '_':
                details.__setattr__(v, eval("geom_data." + v))

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7))
        plt.rcParams.update({'font.size': 8})
        nrows = 5
        ncols = 5
        variables = ['Bmax', 'Bmin', 'epsilon', 'fsa_B2', 'fsa_1overB',
                     'f_t', 'iota', 'G', 'R',
                     'ne_s', 'ni_s', 'Zeff_s', 'Te_s', 'Ti_s',
                     'ln_Lambda_e', 'ln_Lambda_ii',
                     'nu_e_star', 'nu_i_star',
                     'dnds_term', 'dTeds_term', 'dTids_term',
                     'L31', 'L32', 'alpha', 'jdotB']
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(details.s, eval("details." + variable))
            plt.title(variable)
            plt.xlabel('s')
        plt.tight_layout()
        plt.show()

    return jdotB, details


class RedlGeomVmec(Optimizable):
    """
    This class evaluates geometry data needed to evaluate the Redl
    bootstrap current formula from a vmec configuration, such as the
    effective fraction of trapped particles.  The advantage of this
    class over :obj:`RedlGeomBoozer` is that no transformation to
    Boozer coordinates is involved in this method. However, the
    approach here may over-estimate ``epsilon``.

    Args:
        vmec: An instance of :obj:`simsopt.mhd.vmec.Vmec`.
        surfaces: A 1D array of values of s (normalized toroidal flux) on which
            to compute the geometric quantities. If ``None``, the half grid points from the
            VMEC solution will be used.
        ntheta: Number of grid points in the poloidal angle for evaluating geometric quantities in the Redl formulae.
        nphi: Number of grid points in the toroidal angle for evaluating geometric quantities in the Redl formulae.
        plot: Whether to make a plot of many of the quantities computed.
    """

    def __init__(self, vmec, surfaces=None, ntheta=64, nphi=65, plot=False):
        self.vmec = vmec
        self.surfaces = surfaces
        self.ntheta = ntheta
        self.nphi = nphi
        self.plot = plot
        super().__init__(depends_on=[vmec])

    def __call__(self):
        """
        Evaluate the geometric quantities needed for the Redl bootstrap
        current formula.
        """
        self.vmec.run()

        if self.surfaces is None:
            self.surfaces = self.vmec.s_half_grid
        surfaces = self.surfaces
        ntheta = self.ntheta
        nphi = self.nphi

        ns = len(surfaces)
        nfp = self.vmec.wout.nfp
        psi_edge = -self.vmec.wout.phi[-1] / (2 * np.pi)

        # First, interpolate in s to get the quantities we need on the surfaces we need.
        method = 'linear'

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.iotas[1:], fill_value="extrapolate")
        iota = interp(surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bvco[1:], fill_value="extrapolate")
        G = interp(surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.buco[1:], fill_value="extrapolate")
        I = interp(surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.gmnc[:, 1:], fill_value="extrapolate")
        gmnc = interp(surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bmnc[:, 1:], fill_value="extrapolate")
        bmnc = interp(surfaces)

        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        phi2d, theta2d = np.meshgrid(phi1d, theta1d)
        phi3d = phi2d.reshape((ntheta, nphi, 1))
        theta3d = theta2d.reshape((ntheta, nphi, 1))

        myshape = (ntheta, nphi, ns)
        modB = np.zeros(myshape)
        sqrtg = np.zeros(myshape)
        for jmn in range(len(self.vmec.wout.xm_nyq)):
            m = self.vmec.wout.xm_nyq[jmn]
            n = self.vmec.wout.xn_nyq[jmn]
            angle = m * theta3d - n * phi3d
            cosangle = np.cos(angle)
            sinangle = np.sin(angle)
            modB += np.kron(bmnc[jmn, :].reshape((1, 1, ns)), cosangle)
            sqrtg += np.kron(gmnc[jmn, :].reshape((1, 1, ns)), cosangle)

        Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(modB, sqrtg)

        # There are several ways we could define an effective R for shaped geometry:
        R = (G + iota * I) * fsa_1overB
        #R = self.vmec.wout.RMajor_p

        # Pack data into a return structure
        data = Struct()
        data.vmec = self.vmec
        variables = ['nfp', 'surfaces', 'Bmin', 'Bmax', 'epsilon', 'fsa_B2', 'fsa_1overB', 'f_t',
                     'modB', 'sqrtg', 'G', 'R', 'I', 'iota', 'psi_edge', 'theta1d', 'phi1d']
        for v in variables:
            data.__setattr__(v, eval(v))

        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 7))
            plt.rcParams.update({'font.size': 8})
            nrows = 3
            ncols = 4
            variables = ['Bmax', 'Bmin', 'epsilon', 'fsa_B2', 'fsa_1overB', 'f_t', 'iota', 'G', 'I', 'R']
            for j, variable in enumerate(variables):
                plt.subplot(nrows, ncols, j + 1)
                plt.plot(surfaces, eval(variable))
                plt.title(variable)
                plt.xlabel('s')
            plt.tight_layout()
            plt.show()

        return data


class RedlGeomBoozer(Optimizable):
    """
    Evaluate geometry data needed to evaluate the Redl bootstrap
    current formula, such as the effective fraction of trapped
    particles.  In the approach here, Boozer coordinates are computed,
    and all the symmetry-breaking Bmn harmonics are discarded to
    obtain an effectively perfectly quasisymmetric configuration.

    Args:
        booz: An instance of :obj:`simsopt.mhd.boozer.Boozer`
        surfaces: A 1D array with the values of normalized toroidal flux
            to use for the bootstrap current calculation.
        helicity_n: 0 for quasi-axisymmetry, or +/- 1 for quasi-helical symmetry.
            This quantity is used to discard symmetry-breaking :math:`B_{mn}` harmonics.
        ntheta: Number of grid points in the poloidal angle for evaluating geometric quantities in the Redl formulae.
        plot: Make a plot of many of the quantities computed.
    """

    def __init__(self, booz, surfaces, helicity_n, ntheta=64, plot=False):
        booz.register(surfaces)
        self.booz = booz
        self.surfaces = surfaces
        self.helicity_n = helicity_n
        self.ntheta = ntheta
        self.plot = plot
        super().__init__(depends_on=[booz])

    def __call__(self):
        """
        Evaluate the geometric quantities needed for the Redl bootstrap
        current formula.
        """
        booz = self.booz
        booz.run()

        # self.surfaces = self.booz.bx.s_b
        surfaces = self.surfaces
        ns = len(surfaces)
        ntheta = self.ntheta
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        vmec = self.booz.equil
        self.vmec = vmec
        nfp = vmec.wout.nfp
        psi_edge = -vmec.wout.phi[-1] / (2 * np.pi)
        logger.info(f'Surfaces from booz_xform: {self.booz.bx.s_b}  '
                    f'Surfaces for RedlGeomBoozer: {surfaces}')

        # First, interpolate in s to get the quantities we need on the surfaces we need.
        method = 'linear'

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.iotas[1:], fill_value="extrapolate")
        iota = interp(surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bvco[1:], fill_value="extrapolate")
        G = interp(surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.buco[1:], fill_value="extrapolate")
        I = interp(surfaces)

        if self.vmec.mpi.proc0_groups:
            interp = interp1d(self.booz.bx.s_b, self.booz.bx.bmnc_b, fill_value="extrapolate")
            bmnc_b = interp(surfaces)
            logger.info(f'Original bmnc_b.shape: {self.booz.bx.bmnc_b.shape}  Interpolated bmnc_b.shape: {bmnc_b.shape}')

            interp = interp1d(self.booz.bx.s_b, self.booz.bx.gmnc_b, fill_value="extrapolate")
            gmnc_b = interp(surfaces)

            # Evaluate modB and sqrtg on a uniform grid in theta,
            # including only the modes that match the desired symmetry:
            modB = np.zeros((ntheta, ns))
            sqrtg = np.zeros((ntheta, ns))
            s, theta = np.meshgrid(surfaces, theta1d)
            for jmn in range(booz.bx.mnboz):
                if booz.bx.xm_b[jmn] * self.helicity_n * nfp == booz.bx.xn_b[jmn]:
                    # modB += cos(m * theta) * bmnc:
                    modB += np.cos(booz.bx.xm_b[jmn] * theta) \
                        * np.kron(np.ones((ntheta, 1)), bmnc_b[jmn, None, :])
                    sqrtg += np.cos(booz.bx.xm_b[jmn] * theta) \
                        * np.kron(np.ones((ntheta, 1)), gmnc_b[jmn, None, :])
        else:
            modB = 0
            sqrtg = 0

        modB = self.vmec.mpi.comm_groups.bcast(modB)
        sqrtg = self.vmec.mpi.comm_groups.bcast(sqrtg)

        Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(modB, sqrtg)

        # There are several ways we could define an effective R for shaped geometry:
        R = (G + iota * I) * fsa_1overB
        #R = self.vmec.wout.RMajor_p

        # Pack data into a return structure
        data = Struct()
        data.vmec = vmec
        variables = ['nfp', 'surfaces', 'Bmin', 'Bmax', 'epsilon', 'fsa_B2', 'fsa_1overB', 'f_t',
                     'modB', 'sqrtg', 'G', 'R', 'I', 'iota', 'psi_edge', 'theta1d']
        for v in variables:
            data.__setattr__(v, eval(v))

        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 7))
            plt.rcParams.update({'font.size': 8})
            nrows = 3
            ncols = 4
            variables = ['Bmax', 'Bmin', 'epsilon', 'fsa_B2', 'fsa_1overB', 'f_t', 'iota', 'G', 'I', 'R']
            for j, variable in enumerate(variables):
                plt.subplot(nrows, ncols, j + 1)
                plt.plot(surfaces, eval(variable))
                plt.title(variable)
                plt.xlabel('s')
            plt.tight_layout()
            plt.show()

        return data


class VmecRedlBootstrapMismatch(Optimizable):
    r"""
    This class is used to obtain quasi-axisymmetric or quasi-helically
    symmetric VMEC configurations with self-consistent bootstrap
    current. This class represents the objective function

    .. math::

        f = \frac{\int ds \left[\left<\vec{J}\cdot\vec{B}\right>_{vmec}
                                - \left<\vec{J}\cdot\vec{B}\right>_{Redl} \right]^2}
                 {\int ds \left[\left<\vec{J}\cdot\vec{B}\right>_{vmec}
                                + \left<\vec{J}\cdot\vec{B}\right>_{Redl} \right]^2}

    where :math:`\left<\vec{J}\cdot\vec{B}\right>_{vmec}` is the
    bootstrap current profile in a VMEC equilibrium, and
    :math:`\left<\vec{J}\cdot\vec{B}\right>_{Redl}` is the bootstrap
    current profile computed from the fit formulae in Redl et al,
    Physics of Plasmas 28, 022502 (2021).

    Args:
        geom: An instance of either :obj:`RedlGeomVmec` or :obj:`RedlGeomBoozer`.
        ne: A :obj:`~simsopt.mhd.profiles.Profile` object representing the electron density profile.
        Te: A :obj:`~simsopt.mhd.profiles.Profile` object representing the electron temperature profile.
        Ti: A :obj:`~simsopt.mhd.profiles.Profile` object representing the ion temperature profile.
        Zeff: A :obj:`~simsopt.mhd.profiles.Profile` object representing the :math:`Z_{eff}` profile.
            A single number can also be provided, in which case a constant :math:`Z_{eff}` profile will be used.
        helicity_n: 0 for quasi-axisymmetry, or +/- 1 for quasi-helical symmetry.
    """

    def __init__(self, geom, ne, Te, Ti, Zeff, helicity_n, logfile=None):
        if not isinstance(Zeff, Profile):
            # If we get here then Zeff is presumably a number. Convert it to a constant profile.
            Zeff = ProfilePolynomial([Zeff])
        self.geom = geom
        self.ne = ne
        self.Te = Te
        self.Ti = Ti
        self.Zeff = Zeff
        self.helicity_n = helicity_n
        self.iteration = 0
        self.logfile = logfile

        super().__init__(depends_on=[geom, ne, Te, Ti, Zeff])

    def residuals(self):
        r"""
        This function returns a 1d array of residuals, useful for
        representing the objective function as a nonlinear
        least-squares problem.  This is the function handle to use
        with a
        :obj:`~simsopt.objectives.least_squares.LeastSquaresProblem`.

        Specifically, this function returns

        .. math::

            R_j = \frac{\left<\vec{J}\cdot\vec{B}\right>_{vmec}(s_j)
                      - \left<\vec{J}\cdot\vec{B}\right>_{Redl}(s_j)}
                       {\sqrt{\sum_{k=1}^N \left[\left<\vec{J}\cdot\vec{B}\right>_{vmec}(s_k)
                                    + \left<\vec{J}\cdot\vec{B}\right>_{Redl}(s_k) \right]^2}}

        where :math:`j` and :math:`k` range over the surfaces for the
        supplied ``geom`` object (typically the half-grid points for
        the VMEC configuration), :math:`j, k \in \{1, 2, \ldots, N\}`
        and :math:`N` is the number of surfaces for the supplied
        ``geom`` object. This corresponds to approximating the
        :math:`\int ds` integrals in the objective function with
        Riemann integration. The vector of residuals returned has
        length :math:`N`.

        The sum of the squares of these residuals equals the objective
        function. The total scalar objective is approximately
        independent of the number of surfaces.
        """
        jdotB_Redl, _ = j_dot_B_Redl(self.ne,
                                     self.Te,
                                     self.Ti,
                                     self.Zeff,
                                     self.helicity_n,
                                     geom=self.geom)
        # Interpolate vmec's <J dot B> profile from the full grid to the desired surfaces:
        vmec = self.geom.vmec
        interp = interp1d(vmec.s_full_grid, vmec.wout.jdotb)  # VMEC's "jdotb" is on the full grid.
        jdotB_vmec = interp(self.geom.surfaces)

        if self.logfile is not None:
            if self.iteration == 0:
                # Write header
                with open(self.logfile, 'w') as f:
                    f.write('s\n')
                    f.write(str(self.geom.surfaces[0]))
                    for j in range(1, len(self.geom.surfaces)):
                        f.write(', ' + str(self.geom.surfaces[j]))
                    f.write('\n')
                    f.write('iteration, j dot B Redl, j dot B vmec\n')

            with open(self.logfile, 'a') as f:
                f.write(str(self.iteration))
                for j in range(len(self.geom.surfaces)):
                    f.write(', ' + str(jdotB_Redl[j]))
                for j in range(len(self.geom.surfaces)):
                    f.write(', ' + str(jdotB_vmec[j]))
                f.write('\n')

        self.iteration += 1
        denominator = np.sum((jdotB_vmec + jdotB_Redl) ** 2)
        return (jdotB_vmec - jdotB_Redl) / np.sqrt(denominator)

    def J(self):
        """
        Return the scalar objective function, given by the sum of the
        squares of the residuals.
        """
        return np.sum(self.residuals() ** 2)
