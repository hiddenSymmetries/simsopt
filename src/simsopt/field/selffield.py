"""
This module contains functions for computing the self-field of a coil using the
methods from:

    Hurwitz, Siena, Matt Landreman, and Thomas M. Antonsen. 
    "Efficient calculation of the self magnetic field, self-force, and self-inductance for 
    electromagnetic coils." IEEE Transactions on Magnetics (2024).

    Landreman, Matt, Siena Hurwitz, and Thomas M. Antonsen. 
    "Efficient calculation of self magnetic field, self-force, and self-inductance for 
    electromagnetic coils with rectangular cross-section." 
    Nuclear Fusion 65.3 (2025): 036008.

"""

from scipy import constants
import numpy as np
import jax.numpy as jnp
from ..geo.jit import jit

Biot_savart_prefactor = constants.mu_0 / (4 * np.pi)

__all__ = ['B_regularized_pure', 'regularization_rect', 'regularization_circ']


def _rectangular_xsection_k(a, b):
    r"""Auxiliary function for regularization in rectangular conductor.

    .. math::
        k = \frac{4 b}{3 a} \arctan \left( \frac a b \right) + \frac{4 a}{3 b} \arctan \left( \frac b a \right) + \frac{b^2}{6 a^2} \log \left( \frac b a \right) + \frac{a^2}{6 b^2} \log \left( \frac a b \right) - \frac{a^4 - 6 a^2 b^2 + b^4}{6 a^2 b^2} \log \left( \frac a b + \frac b a \right)

    where a is the width of the rectangular conductor and b is the height.
    
    Args:
        a (float): The width of the rectangular conductor.
        b (float): The height of the rectangular conductor.

    Returns:
        float: The regularization parameter.
    """
    return (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a) + \
        (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) - \
        (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)


def _rectangular_xsection_delta(a, b):
    r"""Auxiliary function for regularization in rectangular conductor.

    .. math::
        \delta = \exp \left( - \frac{25}{6} + K \right)
    where K is the auxiliary function defined above.
    
    Args:
        a (float): The width of the rectangular conductor.
        b (float): The height of the rectangular conductor.

    Returns:
        float: The regularization parameter.
    """
    return jnp.exp(-25/6 + _rectangular_xsection_k(a, b))


def regularization_circ(a):
    r"""Regularization for a circular conductor.

    .. math::
        \delta = a^2 / \sqrt{e}
    where e = 2.718... is the base of the natural logarithm
    and a is the radius of the circular conductor.

    Args:
        a (float): The radius of the circular conductor.

    Returns:
        float: The regularization parameter.
    """
    return a**2 / jnp.sqrt(jnp.e)


def regularization_rect(a, b):
    r"""Regularization for a rectangular conductor.

    .. math::
        \delta = a b \exp \left( - \frac{25}{6} + K \right)
    where K is the auxiliary function defined above,
    a is the width of the rectangular conductor and b is the height.

    Args:
        a (float): The width of the rectangular conductor.
        b (float): The height of the rectangular conductor.

    Returns:
        float: The regularization parameter.
    """
    return a * b * _rectangular_xsection_delta(a, b)

@jit
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


@jit
def B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization):
    """
    Compute the regularized field on a coil following the Landreman and Hurwitz method
    
    Args:
        gamma (array (shape (n,3))): The curve of the coil.
        gammadash (array (shape (n,3))): The first derivative of the curve.
        gammadashdash (array (shape (n,3))): The second derivative of the curve.
        quadpoints (array (shape (n,))): The quadrature points of the curve.
        current (float): The current in the coil.
        regularization (float): The regularization parameter.

        The factors of 2π in the next few lines come from the fact that simsopt
        uses a curve parameter that goes up to 1 rather than 2π.

    Returns:
        array (shape (n,3)): The regularized field on the coil.
    """
    phi = quadpoints * 2 * jnp.pi
    rc = gamma
    rc_prime = gammadash / 2 / jnp.pi
    rc_prime_prime = gammadashdash / 4 / jnp.pi**2
    dphi = 2 * jnp.pi / phi.shape[0]
    analytic_term = B_regularized_singularity_term(rc_prime, rc_prime_prime, regularization)
    dr = rc[:, None] - rc[None, :]
    first_term = jnp.cross(rc_prime[None, :], dr) / ((jnp.sum(dr * dr, axis=2) + regularization) ** 1.5)[:, :, None]
    cos_fac = 2.0 - 2.0 * jnp.cos(phi[None, :] - phi[:, None])
    second_term = jnp.cross(rc_prime_prime, rc_prime)[:, None, :] * (
        0.5 * cos_fac / (cos_fac * jnp.sum(rc_prime * rc_prime, axis=1)[:, None] + regularization)**1.5)[:, :, None]
    integral_term = dphi * jnp.sum(first_term + second_term, 1)
    return current * Biot_savart_prefactor * (analytic_term + integral_term)


