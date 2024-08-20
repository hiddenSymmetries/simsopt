"""
This module contains functions for computing the self-field of a coil using the
method from Hurwitz, Landreman, & Antonsen, arXiv (2023).
"""

import math
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad, vjp
from simsopt.geo.jit import jit
from .biotsavart import BiotSavart
from .magneticfield import MagneticField
from .._core.optimizable import Optimizable
from .coil import Coil
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import warnings

Biot_savart_prefactor = constants.mu_0 / (4 * np.pi)

__all__ = ['B_regularized_pure', 'regularization_rect', 'SelfField']

def rectangular_xsection_k(a, b):
    """Auxiliary function for field in rectangular conductor"""
    return (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a) + \
        (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) - \
        (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)


def rectangular_xsection_delta(a, b):
    """Auxiliary function for field in rectangular conductor"""
    return jnp.exp(-25/6 + rectangular_xsection_k(a, b))


def regularization_circ(a):
    """Regularization for a circular conductor"""
    return a**2 / jnp.sqrt(jnp.e)


def regularization_rect(a, b):
    """Regularization for a rectangular conductor"""
    return a * b * rectangular_xsection_delta(a, b)


def B_regularized_singularity_term(rc_prime, rc_prime_prime, regularization):
    """The term in the regularized Biot-Savart law in which the near-singularity
    has been integrated analytically.

    regularization corresponds to delta * a * b for rectangular x-section, or to
    a²/√e for circular x-section.

    A prefactor of μ₀ I / (4π) is not included.

    The derivatives rc_prime, rc_prime_prime refer to an angle that goes up to
    2π, not up to 1.
    """
    norm_rc_prime = jnp.linalg.norm(rc_prime, axis=1)
    return jnp.cross(rc_prime, rc_prime_prime) * (
        0.5 * (-2 + jnp.log(64 * norm_rc_prime * norm_rc_prime / regularization)) / (norm_rc_prime**3)
    )[:, None]


def B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization):
    # The factors of 2π in the next few lines come from the fact that simsopt
    # uses a curve parameter that goes up to 1 rather than 2π.
    phi = quadpoints * 2 * jnp.pi
    rc = gamma
    rc_prime = gammadash / 2 / jnp.pi
    rc_prime_prime = gammadashdash / 4 / jnp.pi**2
    n_quad = phi.shape[0]
    dphi = 2 * jnp.pi / n_quad

    analytic_term = B_regularized_singularity_term(rc_prime, rc_prime_prime, regularization)

    dr = rc[:, None] - rc[None, :]    
    first_term = jnp.cross(rc_prime[None, :], dr) / ((jnp.sum(dr * dr, axis=2) + regularization) ** 1.5)[:, :, None]
    cos_fac = 2 - 2 * jnp.cos(phi[None, :] - phi[:, None])
    denominator2 = cos_fac * jnp.sum(rc_prime * rc_prime, axis=1)[:, None] + regularization
    factor2 = 0.5 * cos_fac / denominator2**1.5
    second_term = jnp.cross(rc_prime_prime, rc_prime)[:, None, :] * factor2[:, :, None]

    integral_term = dphi * jnp.sum(first_term + second_term, 1)
    
    return current * Biot_savart_prefactor * (analytic_term + integral_term)


def B_regularized(coil, regularization):
    """Calculate the regularized field on a coil following the Landreman and Hurwitz method"""
    return B_regularized_pure(
        coil.curve.gamma(),
        coil.curve.gammadash(),
        coil.curve.gammadashdash(),
        coil.curve.quadpoints,
        coil._current.get_value(),
        regularization,
    )


def B_regularized_circ(coil, a):
    return B_regularized(coil, regularization_circ(a))


def B_regularized_rect(coil, a, b):
    return B_regularized(coil, regularization_rect(a, b))


def G(x, y):
    """Auxiliary function for the calculation of the internal field of a rectangular crosssction"""
    return y * jnp.arctan(x/y) + x/2*jnp.log(1 + y**2 / x**2)


def K(u, v, kappa_1, kappa_2, p, q, a, b):
    """Auxiliary function for the calculation of the internal field of a rectangular crosssction"""
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

    for i in range(N_phi):
        b_b.at[i].set(kappa[i] * b[i] / 2 *
                      (4 + 2*jnp.log(2) + jnp.log(delta(a, b))))
        b_kappa.at[i].set(1 / 16 * (K(u - 1, v - 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b) + K(u + 1, v + 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b)
                                    - K(u - 1, v + 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b) - K(u + 1, v - 1, kappa_1[i], kappa_2[i], p[i], q[i], a, b)))
        b_0.at[i].set(1 / (a * b) * ((G(b * (v - 1), a * (u - 1)) + G(b * (v + 1), a * (u + 1)) - G(b * (v + 1), a * (u - 1)) - G(b * (v - 1), a * (u + 1))) * q -
                                     (G(a * (u - 1), b * (v - 1)) + G(a * (u + 1), b * (v + 1)) - G(a * (u - 1), b * (v + 1)) - G(a * (u - 1), b * (v + 1))) * p))

    b_loc = constants.mu_0 * I / 4 / jnp.pi * (b_b + b_kappa + b_0)

    return b_loc


def field_from_other_coils(coil, coils):
    """field on one coil from the other coils"""
    gamma = coil.curve.gamma()
    b_ext = BiotSavart(coils)
    b_ext.set_points(gamma)
    return b_ext.B()


def field_from_other_coils_pure(gamma, curves, currents):
    """field on one coil from the other coils"""
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    b_ext = BiotSavart(coils)
    b_ext.set_points(gamma)
    return b_ext.B()


class SelfField(Optimizable):
    def __init__(self, coil, a, b):
        self._curve = coil.curve
        self._current = coil.current

        self.regularization = regularization_rect(a, b)
        self.quadpoints = self._curve.quadpoints

        self.B_jax = lambda gamma, gammadash, gammadashdash, current: B_regularized_pure(gamma, gammadash, gammadashdash, self.quadpoints, current, self.regularization)

        self.Bgrad0 = jit(
            lambda gamma, gammadash, gammadashdash, current: grad(self.B_jax, argnums=0)(gamma, gammadash, gammadashdash, current)
        )
        self.Bgrad1 = jit(
            lambda gamma, gammadash, gammadashdash, current: grad(self.B_jax, argnums=1)(gamma, gammadash, gammadashdash, current)
        )
        self.Bgrad2 = jit(
            lambda gamma, gammadash, gammadashdash, current: grad(self.B_jax, argnums=2)(gamma, gammadash, gammadashdash, current)
        )
        self.Bgrad3 = jit(
            lambda gamma, gammadash, gammadashdash, current: grad(self.B_jax, argnums=3)(gamma, gammadash, gammadashdash, current)
        )

        self.dB_by_dgamma_vjp = jit(lambda gamma, gammadash, gammadashdash, current, v: vjp(lambda d1g: self.B_jax(d1g, gammadash, gammadashdash, current), gamma)[1](v)[0])
        self.dB_by_dgammadash_vjp = jit(lambda gamma, gammadash, gammadashdash, current, v: vjp(lambda d2g: self.B_jax(gamma, d2g, gammadashdash, current), gammadash)[1](v)[0])
        self.dB_by_dgammadashdash_vjp = jit(lambda gamma, gammadash, gammadashdash, current, v: vjp(lambda d3g: self.B_jax(gamma, gammadash, d3g, current), gammadashdash)[1](v)[0])
        self.dB_by_dcurrent_vjp = jit(lambda gamma, gammadash, gammadashdash, current, v: vjp(lambda dcur: self.B_jax(gamma, gammadash, gammadashdash, dcur), current)[1](v)[0])

        Optimizable.__init__(self, depends_on=[coil])

    def set_points(self, xyz):
        warnings.warn('Warning - SelfField evaluates the field on its curve. Evaluation points cannot be changed.')
        return None

    def set_points_cart(self, xyz):
        warnings.warn('Warning - SelfField evaluates the field on its curve. Evaluation points cannot be changed.')
        return None
    
    def set_points_cyl(self, rphiz):
        warnings.warn('Warning - SelfField evaluates the field on its curve. Evaluation points cannot be changed.')
        return None
    
    def get_points_cart(self):
        return self._curve.gamma()
    
    def get_points_cyl(self):
        g = self.get_points_cart()
        r = np.sqrt(g[:,0]**2 + g[:,1]**2)
        phi = np.arctan2(g[:,1], g[:,0])
        z = g[:,2]
        return np.array([r,phi,z])
    
    def B(self):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        gammadashdash = self._curve.gammadashdash()
        current = self._current.get_value()
        return self.B_jax(gamma, gammadash, gammadashdash, current)

    # def dB_by_dcoeff(self):
    #     gamma = self._curve.gamma()
    #     gammadash = self._curve.gammadash()
    #     gammadashdash = self._curve.gammadashdash()
    #     current = self._current.get_value()

    #     grad0 = self.Bgrad0(gamma, gammadash, gammadashdash, current)
    #     grad1 = self.Bgrad0(gamma, gammadash, gammadashdash, current)
    #     grad2 = self.Bgrad0(gamma, gammadash, gammadashdash, current)
    #     grad3 = self.Bgrad0(gamma, gammadash, gammadashdash, current)

    #     return self._curve.dgamma_by_dcoeff_vjp(grad0) + self._curve.dgammadash_by_dcoeff_vjp(grad1) + self._curve.dgammadashdash_by_dcoeff_vjp(grad2) + self._current.vjp( grad3 )

    def dB_by_dgamma_vjp_impl(self, v):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        gammadashdash = self._curve.gammadashdash()
        current = self._current.get_value()

        return self.dB_by_dgamma_vjp(gamma, gammadash, gammadashdash, current, v)
    
    def dB_by_dgammadash_vjp_impl(self, v):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        gammadashdash = self._curve.gammadashdash()
        current = self._current.get_value()

        return self.dB_by_dgammadash_vjp(gamma, gammadash, gammadashdash, current, v)
    
    def dB_by_dgammadashdash_vjp_impl(self, v):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        gammadashdash = self._curve.gammadashdash()
        current = self._current.get_value()

        return self.dB_by_dgammadashdash_vjp(gamma, gammadash, gammadashdash, current, v)
    
    def dB_by_dcurrent_vjp_impl(self, v):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        gammadashdash = self._curve.gammadashdash()
        current = self._current.get_value()

        return self.dB_by_dcurrent_vjp(gamma, gammadash, gammadashdash, current, v)

    def B_vjp(self, v):
        grad0 = self.dB_by_dgamma_vjp_impl(v)
        grad1 = self.dB_by_dgammadash_vjp_impl(v)
        grad2 = self.dB_by_dgammadashdash_vjp_impl(v)
        grad3 = jnp.array([self.dB_by_dcurrent_vjp_impl(v)])

        return self._curve.dgamma_by_dcoeff_vjp(grad0) \
            + self._curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self._curve.dgammadashdash_by_dcoeff_vjp(grad2) \
            + self._current.vjp(grad3)

