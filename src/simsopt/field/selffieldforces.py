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
from scipy import integrate

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


def k(a, b):
    """Auxiliary function for field in reectangular conductor"""
    return (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a) + \
        (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) - \
        (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)


def delta(a, b):
    """Auxiliary function for field in rectangular conductor"""
    return jnp.exp(-25/6 + k(a, b))


def compute_regularization_circ(a):
    return a**2 / jnp.sqrt(jnp.e)


def compute_regularization_rect(a, b):
    return a * b * delta(a, b)


def singularity_term_circ(gamma, gammadash, gammadashdash, a):
    A = 1 / jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**3 * \
        jnp.cross(gammadashdash, gammadash) * (jnp.log(a / 8 /
                                                       jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]) + 3 / 4)

    return A


def singularity_term_circ_matt(gamma, gammadash, gammadashdash, a):
    d1gamma_norm = np.linalg.norm(gammadash, axis=1)[:, np.newaxis]
    A = 0.5 / d1gamma_norm**3 * \
        (2 + np.log(a**2/np.sqrt(math.e) / d1gamma_norm / 64)) * \
        np.cross(gammadash, gammadashdash)
    return -A


def singularity_term_rect(gammadash, gammadashdash, a, b):
    A = (jnp.cross(gammadash, gammadashdash)) / (2*jnp.linalg.norm(gammadash, axis=1)
                                                 [:, jnp.newaxis]**3) * (-2 + jnp.log(64 / (delta(a, b) * a * b) * jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**2))

    return A


def integrand_circ(i, gamma, gammadash, gammadashdash, phi, phidash, a):
    first_term = np.cross(gammadash[i], (gamma-gamma[i])) / (
        np.linalg.norm(gamma - gamma[i])**2 + a**2/np.sqrt(math.e)) ** 1.5

    second_term = np.cross(gammadashdash, gammadash[i]) * (1 - np.cos(phidash[i] - phi)[:, np.newaxis])\
        / ((2 - 2*np.cos(phidash[i] - phi)) * np.linalg.norm(gammadashdash) + a**2/np.sqrt(math.e))[:, np.newaxis]**1.5

    #        1 - np.cos(phidash[i] - phi)) * np.linalg.norm(gammadash)**2 + a**2/np.sqrt(math.e))[:, np.newaxis]**1.5)
    # integrand = (np.cross(gammadash[i], (gamma-gamma[i]))) / (np.linalg.norm(gamma - gamma[i])**2 + a/np.sqrt(math.e)) ** 1.5 \
    #    + (np.cross(gammadashdash, gammadash) * (1 - np.cos(phidash[i] - phi)[:, np.newaxis]) / (2 * (
    #        1 - np.cos(phidash[i] - phi)) * np.linalg.norm(gammadash)**2 + a**2/np.sqrt(math.e))[:, np.newaxis]**1.5)

    integrand = first_term + second_term

    # integrand = np.cross(gammadash[i], (gamma-gamma[i]))
    return integrand


def integrand_circ_matt(i, gamma, gammadash, gammadashdash, phi, phidash, a):
    """Function to compare to Matt's julia Repo"""

    regularization = a**2 / np.sqrt(math.e)
    dr = gamma[i] - gamma
    temp = np.linalg.norm(dr, axis=1)**2 + regularization
    denominator = temp * np.sqrt(temp)
    cos_fac = 2 - 2 * np.cos(phidash[i] - phi)
    temp2 = cos_fac * \
        np.linalg.norm(gammadash, axis=1)**2 + regularization
    denominator2 = temp2 * np.sqrt(temp2)
    factor2 = 0.5 * cos_fac / denominator2
    integrand = np.cross(gammadash, dr) / denominator[:, np.newaxis] + \
        factor2[:, np.newaxis] * np.cross(gammadashdash, gammadash)
    return -integrand


def integrand_circ_better(gamma, gammadash, gammadashdash, phi, gammaprime, gammadashprime, phiprime, a):
    """Different implementation of the integration for future use. Nothing new here so far."""

    first_term = np.cross(gammadashprime, (gamma-gammaprime)) / (
        np.linalg.norm(gamma - gammaprime)**2 + a**2/np.sqrt(math.e)) ** 1.5

    second_term = np.cross(gammadashdash, gammadashprime) * (1 - np.cos(phiprime - phi)[:, np.newaxis])\
        / ((2 - 2*np.cos(phiprime - phi)) * np.linalg.norm(gammadashdash) + a**2/np.sqrt(math.e))[:, np.newaxis]**1.5

    integrand = first_term + second_term
    return integrand


def integral_circ(gamma, gammadash, gammadashdash, phi, phidash, a):
    n_quad = phidash.shape[0]
    dphidash = phidash[1]
    integral = np.zeros((n_quad, 3))
    integrand = np.zeros((n_quad, n_quad, 3))

    for i, _ in enumerate(phidash):

        integrand[i] = integrand_circ(i, gamma, gammadash,
                                      gammadashdash, phi, phidash, a) * dphidash
        integral[i, 0] = integrate.simpson(integrand[i, :, 0], phidash)
        integral[i, 1] = integrate.simpson(integrand[i, :, 1], phidash)
        integral[i, 2] = integrate.simpson(integrand[i, :, 2], phidash)
    return integral


def integral_circ_matt(gamma, gammadash, gammadashdash, phi, phidash, a):
    """Function to compare to Matt's julia Repo"""
    n_quad = phidash.shape[0]
    dphidash = phidash[1]
    integral = np.zeros((n_quad, 3))
    integrand = np.zeros((n_quad, n_quad, 3))

    for i, _ in enumerate(phidash):

        integrand[i] = integrand_circ_matt(i, gamma, gammadash,
                                           gammadashdash, phi, phidash, a)
        # integral[i, 0] = integrate.simpson(integrand[i, :, 0], phidash)
        # integral[i, 1] = integrate.simpson(integrand[i, :, 1], phidash)
        # integral[i, 2] = integrate.simpson(integrand[i, :, 2], phidash)
        integral[i, 0] = np.sum(integrand[i, :, 0])
        integral[i, 1] = np.sum(integrand[i, :, 1])
        integral[i, 2] = np.sum(integrand[i, :, 2])

    integral *= dphidash
    return integral


def integral_circ_better(gamma, gammadash, gammadashdash, phi, gammaprime, gammadashprime, phiprime, a):
    """Different implementation of the integration for future use. Nothing new here so far."""
    n_quad = phiprime.shape[0]
    integrand = integrand_circ_better(
        gamma, gammadash, gammadashdash, phi, gammaprime, gammadashprime, phiprime, a)
    weights = np.ones(n_quad)
    integral = (i*w for i, w in zip(integrand, weights))
    return integral


def vector_matrix_product(matrix, vector):
    """Product between matrix and a vector with one free axis (e.g. along a coil)"""
    return jnp.einsum('ijk, ik -> ij', matrix, vector)


def inner(vector_1, vector_2):
    """Dot Product between two vectors with one free axis (e.g. along a coil)"""
    return jnp.sum(vector_1*vector_2, axis=1)


def rotation(angle, axis, vector):
    """Rotates a 3D vector around an axis vector"""

    n_quad = angle.shape[0]

    rotation_matrix = jnp.array([[[jnp.cos(angle[i])+axis[i, 0]**2*(1-jnp.cos(angle[i])),
                                  axis[i, 0]*axis[i, 1] *
                                  (1-jnp.cos(angle[i])) -
                                  axis[i, 2]*jnp.sin(angle[i]),
                                  axis[i, 0]*axis[i, 2]*(1-jnp.cos(angle[i])) + axis[i, 1]*jnp.sin(angle[i])],

                                 [axis[i, 1]*axis[i, 0]*(1-jnp.cos(angle[i])) + axis[i, 2]*jnp.sin(angle[i]),
                                  jnp.cos(angle[i]) + axis[i, 1]**2 *
                                  (1-jnp.cos(angle[i])),
                                  axis[i, 1]*axis[i, 2]*(1-jnp.cos(angle[i]))-axis[i, 0]*jnp.sin(angle[i])],

                                 [axis[i, 0]*axis[i, 2]*(1-jnp.cos(angle[i])) - axis[i, 1]*jnp.sin(angle[i]),
                                  axis[i, 1]*axis[i, 2] *
                                  (1-jnp.cos(angle[i])) +
                                  axis[i, 0]*jnp.sin(angle[i]),
                                  jnp.cos(angle[i]) + axis[i, 2]**2*(1-jnp.cos(angle[i]))]] for i in range(n_quad)])

    return vector_matrix_product(rotation_matrix, vector)


def c_axis_pure(t, b):
    """Implements THEVA-tape-like c-axis"""
    n_quad = t.shape[0]
    rotation_degrees = 60*jnp.ones(n_quad)
    rotation_radians = jnp.radians(rotation_degrees)
    c_axis = rotation(rotation_radians, t, b)

    return c_axis


def c_axis_angle_pure(coil, B):
    """Angle between the magnetic field and the c-axis of the REBCO Tape"""
    t, _, b = coil.curve.frenet_frame()
    b_norm = jnp.einsum("ij, i->ij", B, 1/jnp.linalg.norm(B, axis=1))

    c_axis = c_axis_pure(t, b)
    angle = jnp.arccos(inner(c_axis, b_norm))
    return angle


def field_on_coils(coil, a=0.05):
    """Calculate the regularized field on a coil with circular cross section following the Landreman and Hurwitz method"""
    I = coil._current.current
    phi = coil.curve.quadpoints * 2 * np.pi
    phidash = coil.curve.quadpoints * 2 * np.pi
    dphidash = phidash[1]
    n_quad = phidash.shape[0]
    gamma = coil.curve.gamma()
    gammadash = coil.curve.gammadash() / 2 / np.pi
    gammadashdash = coil.curve.gammadashdash() / 4 / np.pi**2
    A = singularity_term_circ_matt(gamma, gammadash, gammadashdash, a)
    integral = integral_circ_matt(
        gamma, gammadash, gammadashdash, phi, phidash, a)
    b_reg = I * Biot_savart_prefactor * (integral+A)

    return b_reg


def field_on_coils_rect(coil, a=0.05, b=0.03):
    """Calculate the field on a coil with rectangular cross section follownig the method from Landreman, Hurwitz & Antonsen"""
    I = coil._current.current
    phi = coil.curve.quadpoints
    phidash = coil.curve.quadpoints
    n_quad = phidash.shape[0]
    gamma = coil.curve.gamma()
    gammadash = coil.curve.gammadash()
    gammadashdash = coil.curve.gammadashdash()

    A = (jnp.cross(gammadash, gammadashdash)) / (2*jnp.linalg.norm(gammadash, axis=1)
                                                 [:, jnp.newaxis]**3) * (-2 + jnp.log(64 / (delta(a, b) * a * b) * jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**2))

    b_int_phi_dash = jnp.zeros((n_quad, n_quad, 3))
    integral += 0
    for i in range(len(phidash)):
        b_int_phi_dash.at[i].set((jnp.cross(gammadash[i], (gamma-gamma[i]))) / (jnp.linalg.norm(gamma - gamma[i])**2 + delta + a * b) ** 1.5
                                 + (jnp.cross(gammadashdash, gammadash) * (1 - jnp.cos(phidash[i] - phi)[:, jnp.newaxis]) / ((
                                     2 - 2 * jnp.cos(phidash[i] - phi)) * jnp.linalg.norm(gammadash)**2 + delta * a * b)[:, jnp.newaxis]))

    integral = jnp.trapz(b_int_phi_dash, phidash, axis=0)

    b_reg = I * Biot_savart_prefactor * (A + integral)
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


def self_force(coil, a=0.05):
    """force of the coils onto itself"""
    B = field_on_coils(coil, a)
    t, _, _ = coil.curve.frenet_frame()
    I = coil._current.current
    force = np.cross(I*t, B)
    return force


def external_force(coil, coils, a=0.05):
    """force from the other coils"""
    b_ext = field_from_other_coils(coil, coils, a)
    t, _, _ = coil.curve.frenet_frame()
    I = coil._current.current
    f = np.cross(I*t, b_ext)
    return f


def force_on_coil(coil, coils, a=0.05):
    """full force on a coil from self field and external field"""
    b_self = field_on_coils(coil, a)
    b_ext = field_from_other_coils(coil, coils, a)
    b_tot = b_ext + b_self

    t, _, _ = coil.curve.frenet_frame()
    I = coil._current.current

    f = np.cross(I*t, b_tot)
    return f


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


def coil_force_pure(B, I, t, a=0.05):
    """force on coil for optimization"""
    force = jnp.cross(I*t, B)
    return force


@jit
def force_opt_pure(gamma, gammadash, gammadashdash,
                   current, phi, phidash, b_ext):
    """Cost function for force optimization. Optimize for peak self force on the coil (so far)"""
    t = gammadash / jnp.linalg.norm(gammadash)
    b_self = field_on_coils_rect_pure(
        gamma, gammadash, gammadashdash, phi, phidash, current)
    b_tot = b_self + b_ext
    force = coil_force_pure(b_tot, current, t)
    f_norm = jnp.linalg.norm(force, axis=1)
    result = jnp.max(f_norm)
    # result = jnp.sum(f_norm)
    return result


class ForceOpt(Optimizable):
    def __init__(self, coil, coils, a=0.05):
        self.coil = coil
        self.curve = coil.curve
        self.coils = coils
        self.a = a
        self.B_ext = BiotSavart(coils).set_points(self.curve.gamma()).B()
        self.B_self = 0
        self.B = 0
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash,
                         current, phi, phidash, B_ext: force_opt_pure(gamma, gammadash, gammadashdash,
                                                                      current, phi, phidash, B_ext))

        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash, current, phi, phidash, B_ext: grad(
            self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, phi, phidash, B_ext))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash, current, phi, phidash, B_ext: grad(
            self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, phi, phidash, B_ext))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash, current, phi, phidash, B_ext: grad(
            self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, phi, phidash, B_ext))

        super().__init__(depends_on=[coil])

    def J(self):
        gamma = self.coil.curve.gamma()
        d1gamma = self.coil.curve.gammadash()
        d2gamma = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        phi = self.coil.curve.quadpoints
        phidash = self.coil.curve.quadpoints
        B_ext = self.B_ext
        return self.J_jax(gamma, d1gamma, d2gamma, current, phi, phidash, B_ext)

    @derivative_dec
    def dJ(self):
        gamma = self.coil.curve.gamma()
        d1gamma = self.coil.curve.gammadash()
        d2gamma = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        phi = self.coil.curve.quadpoints
        phidash = self.coil.curve.quadpoints
        B_ext = self.B_ext

        grad0 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, phidash, B_ext)
        grad1 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, phidash, B_ext)
        grad2 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, phidash, B_ext)

        return self.coil.curve.dgamma_by_dcoeff_vjp(grad0) + self.coil.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(grad2)

    return_fn_map = {'J': J, 'dJ': dJ}
