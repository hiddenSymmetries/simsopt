"""Implements the force on a coil in its own magnetic field and the field of other coils."""
import math
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt.field import BiotSavart, Coil
from simsopt.geo.jit import jit
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


def k(a, b):
    """Auxiliary function for field in reectangular conductor"""
    return (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a) + \
        (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) - \
        (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)


def delta(a, b):
    """Auxiliary function for field in rectangular conductor"""
    return jnp.exp(-25/6 + k(a, b))


def regularization_circ(a):
    """Regularization for a circular conductor"""
    return a**2 / jnp.sqrt(jnp.e)


def regularization_rect(a, b):
    """Regularization for a rectangular conductor"""
    return a * b * delta(a, b)


def singularity_term_circ(gamma, gammadash, gammadashdash, a):
    """singularity substraction for the regularized field in a circular conductor"""
    A = 1 / jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**3 * \
        jnp.cross(gammadashdash, gammadash) * (jnp.log(a / 8 /
                                                       jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]) + 3 / 4)

    return A


def singularity_term_rect(gammadash, gammadashdash, a, b):
    """singularity substraction for the regularized field in a rectangular conductor"""
    A = 1 / jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**3 / 2 * \
        (jnp.cross(gammadash, gammadashdash)) * (-2 + jnp.log(64 / (delta(a, b)
                                                                    * a * b) * jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**2))

    return A


def integrand_circ(i, j, gamma, gammadash, gammadashdash, phi, phi_0, a):
    """Integrand of the regularized magnetic field formular at position phi[i], phi[j]. i is the index the field is evaluated (gamma[i]=gamma(phi[i])) while j is thee integrration index."""
    dr = gamma[i]-gamma[j]
    regularization = regularization_circ(a)
    first_term = np.cross(
        gammadash[j], dr) / (np.linalg.norm(dr)**2 + regularization) ** 1.5
    cos_fac = 2 - 2 * np.cos(2*np.pi*(phi[j] - phi[i]))
    denominator2 = cos_fac * \
        np.linalg.norm(gammadash[i])**2 + regularization
    factor2 = 0.5 * cos_fac / denominator2**1.5
    second_term = np.cross(gammadashdash[i], gammadash[i]) * factor2
    integrand = first_term + second_term

    return integrand


def integrand_rect(i, j, gamma, gammadash, gammadashdash, phi, phi_0, a, b):
    """Integrand of the regularized magnetic field formular at position phi[i], phi[j]. i is the index the field is evaluated (gamma[i]=gamma(phi[i])) while j is the integrration index."""
    dr = gamma[i]-gamma[j]
    regularization = regularization_rect(a, b)
    first_term = np.cross(
        gammadash[j], dr) / (np.linalg.norm(dr)**2 + regularization) ** 1.5
    cos_fac = 2 - 2 * np.cos(2*np.pi*(phi[j] - phi[i]))
    denominator2 = cos_fac * \
        np.linalg.norm(gammadash[i])**2 + regularization
    factor2 = 0.5 * cos_fac / denominator2**1.5
    second_term = np.cross(gammadashdash[i], gammadash[i]) * factor2
    integrand = first_term + second_term

    return integrand


def integral_circ(gamma, gammadash, gammadashdash, phi, i, a):
    nphi = phi.shape[0]
    dphi = 2*np.pi / nphi
    integral = np.zeros(3)
    for j, phi_0 in enumerate(phi):
        integral += integrand_circ(i, j, gamma, gammadash,
                                   gammadashdash, phi, phi_0, a)

    integral *= dphi
    return integral


def integral_rect(gamma, gammadash, gammadashdash, phi, i, a, b):
    nphi = phi.shape[0]
    dphi = 2*np.pi / nphi
    integral = np.zeros(3)
    for j, phi_0 in enumerate(phi):
        integral += integrand_rect(i, j, gamma, gammadash,
                                   gammadashdash, phi, phi_0, a, b)

    integral *= dphi
    return integral


def field_on_coils(coil, a=0.05):
    """Calculate the regularized field on a coil with circular cross section following the Landreman and Hurwitz method"""
    I = coil._current.current
    phi = coil.curve.quadpoints  # * 2 * np.pi
    phidash = coil.curve.quadpoints  # * 2 * np.pi
    dphidash = phidash[1]
    n_quad = phidash.shape[0]
    gamma = coil.curve.gamma()
    gammadash = coil.curve.gammadash() / 2 / np.pi
    gammadashdash = coil.curve.gammadashdash() / 4 / np.pi**2
    integral_term = np.zeros((n_quad, 3))

    A = singularity_term_circ(gamma, gammadash, gammadashdash, a)
    for i, _ in enumerate(phi):
        integral_term[i] = integral_circ(
            gamma, gammadash, gammadashdash, phi, i, a)
    b_reg = I * Biot_savart_prefactor * (A + integral_term)

    return b_reg


def field_on_coils_rect(coil, a=0.05, b=0.03):
    """Calculate the field on a coil with rectangular cross section follownig the method from Landreman, Hurwitz & Antonsen"""
    I = coil._current.current
    phi = coil.curve.quadpoints  # * 2 * np.pi
    phidash = coil.curve.quadpoints  # * 2 * np.pi
    dphidash = phidash[1]
    n_quad = phidash.shape[0]
    gamma = coil.curve.gamma()
    gammadash = coil.curve.gammadash() / 2 / np.pi
    gammadashdash = coil.curve.gammadashdash() / 4 / np.pi**2
    integral_term = np.zeros((n_quad, 3))

    A = singularity_term_rect(gammadash, gammadashdash, a, b)
    for i, _ in enumerate(phi):
        integral_term[i] = integral_rect(
            gamma, gammadash, gammadashdash, phi, i, a, b)

    b_reg = I * Biot_savart_prefactor * (A+integral_term)

    return b_reg


def G(x, y):
    """Auxiliary function for the calculation of the internal field of a rectangular crosssction"""
    return y * jnp.arctan(x/y) + x/2*jnp.log(1 + y**2 / x**2)


def K(u, v, kappa_1, kappa_2, p, q, a, b):
    K = - 2 * u * v * (kappa_1 * q - kappa_2 * p) * jnp.log(a * u**2 / b + b * v**2 / a) + \
        (kappa_2 * q - kappa_1 * p) * (a * u**2 / b + b * v**2 / a) * jnp.log(a * u**2 / b + b * v**2 / a) + \
        4 * a * u**2 * kappa_2 * p / b * \
        jnp.arctan(b * v / a * u) - 4 * b * v**2 * \
        kappa_1 * q / a * jnp.arctan(a * u / b * v)


def local_field(coil, rho, theta, a=0.05):
    """Calculate the variation of the field on the cross section"""
    I = coil._current.current
    phi = coil.curve.quadpoints
    N_phi = phi.shape[0]
    _, n, b = coil.curve.frenet_frame()
    kappa = coil.curve.kappa()
    b_loc = jnp.zeros((N_phi, 3))

    for i in range(N_phi):
        b_loc[i] = 2*rho/a * (-n[i]*jnp.sin(theta) + b[i]*jnp.cos(theta)) + kappa[i] / 2 * (-rho**2 / 2 * jnp.sin(2*theta) * n[i] +
                                                                                            (1.5-rho**2 + rho**2 / 2 * jnp.cos(2*theta)) * b[i])

    b_loc *= constants.mu_0 * I / 4 / jnp.pi
    return b_loc


def local_field_rect(coil, u, v, a, b):
    """Calculate the variation of the field on a rectangular cross section"""
    I = coil._current.current
    phi = coil.curve.quadpoints
    N_phi = phi.shape[0]
    _, n, b = coil.curve.frenet_frame()
    kappa = coil.curve.kappa()

    kappa_1 = kappa
    kappa_2 = kappa
    p = n
    q = b

    b_kappa = jnp.zeros((N_phi, 3))
    b_b = jnp.zeros((N_phi, 3))
    b_0 = jnp.zeros((N_phi, 3))

    k = (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a) + \
        (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) - \
        (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)

    delta = jnp.exp(-25/6 + k)

    for i in range(N_phi):
        b_b.at[i].set(kappa[i] * b[i] / 2 *
                      (4 + 2*jnp.log(2) + jnp.log(delta)))
        b_kappa.at[i].set(1 / 16 * (K(u - 1, v - 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b) + K(u + 1, v + 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b)
                                    - K(u - 1, v + 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b) - K(u + 1, v - 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b)))
        b_0.at[i].set(1 / (a * b) * ((G(b * (v - 1), a * (u - 1)) + G(b * (v + 1), a * (u + 1)) - G(b * (v + 1), a * (u - 1)) - G(b * (v - 1), a * (u + 1))) * q -
                                     (G(a * (u - 1), b * (v - 1)) + G(a * (u + 1), b * (v + 1)) - G(a * (u - 1), b * (v + 1)) - G(a * (u - 1), b * (v + 1))) * p))

    b_loc = constants.mu_0 * I / 4 / jnp.pi * (b_b + b_kappa + b_0)

    return b_loc


def field_from_other_coils(coil, coils, a=0.05):
    """field on one coil from the other coils"""
    gamma = coil.curve.gamma()
    b_ext = BiotSavart(coils)
    b_ext.set_points(gamma)
    return b_ext.B()


def field_on_coils_pure(gamma, gammadash, gammadashdash, phi, phidash, current,  a=0.05):
    """Regularized field for optimization"""
    I = current
    n_quad = phidash.shape[0]

    A = (1 / jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**3 * jnp.cross(gammadashdash, gammadash)) * (
        1 + jnp.log(a / (8 * math.e**0.25 * jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis])))
    b_int_phi_dash = jnp.zeros((n_quad, n_quad, 3))

    for i in range(len(phidash)):
        b_int_phi_dash.at[i].set((jnp.cross(gammadash[i], (gamma-gamma[i]))) / (jnp.linalg.norm(gamma - gamma[i])**2 + a/jnp.sqrt(math.e)) ** 1.5
                                 + (jnp.cross(gammadashdash, gammadash) * (1 - jnp.cos(phidash[i] - phi)[:, jnp.newaxis]) / (2 * (
                                     1 - jnp.cos(phidash[i] - phi)) * jnp.linalg.norm(gammadash)**2 + a**2/jnp.sqrt(math.e))[:, jnp.newaxis]))

    integral = jnp.trapz(b_int_phi_dash, phidash, axis=0)

    b_reg = (constants.mu_0 * I / 4 / jnp.pi) * (integral + A)

    return b_reg


def field_on_coils_rect_pure(gamma, gammadash, gammadashdash, phi, phidash, current, a=0.05, b=0.03):
    """Regularized field for optimization of a coil with rectangular cross section"""
    I = current
    n_quad = phidash.shape[0]

    k = (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a) + \
        (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) - \
        (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)

    delta = jnp.exp(-25/6 + k)
    A = (jnp.cross(gammadash, gammadashdash)) / (2*jnp.linalg.norm(gammadash, axis=1)
                                                 [:, jnp.newaxis]**3) * (-2 + jnp.log(64 / (delta * a * b) * jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**2))

    b_int_phi_dash = jnp.zeros((n_quad, n_quad, 3))

    for i in range(len(phidash)):
        b_int_phi_dash.at[i].set((jnp.cross(gammadash[i], (gamma-gamma[i]))) / (jnp.linalg.norm(gamma - gamma[i])**2 + delta + a * b) ** 1.5
                                 + (jnp.cross(gammadashdash, gammadash) * (1 - jnp.cos(phidash[i] - phi)[:, jnp.newaxis]) / ((
                                     2 - 2 * jnp.cos(phidash[i] - phi)) * jnp.linalg.norm(gammadash)**2 + delta * a * b)[:, jnp.newaxis]))

    integral = jnp.trapz(b_int_phi_dash, phidash, axis=0)

    b_reg = (constants.mu_0 * I / 4 / jnp.pi) * (integral + A)

    return b_reg


def field_from_other_coils_pure(gamma, curves, currents, a=0.05):
    """field on one coil from the other coils"""
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    b_ext = BiotSavart(coils)
    b_ext.set_points(gamma)
    return b_ext.B()