"""
This module contains functions for computing the self-field of a coil using the
methods from Hurwitz, Landreman, & Antonsen, arXiv:2310.09313 (2023) and
Landreman, Hurwitz, & Antonsen, arXiv:2310.12087 (2023).
"""

from scipy import constants
import numpy as np
import jax.numpy as jnp

Biot_savart_prefactor = constants.mu_0 / (4 * np.pi)

__all__ = ['B_regularized_pure', 'regularization_rect']


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
