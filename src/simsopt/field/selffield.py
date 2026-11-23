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

__all__ = ["B_regularized_pure", "regularization_rect", "regularization_circ"]


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
    return (
        (4 * b) / (3 * a) * jnp.arctan(a / b)
        + (4 * a) / (3 * b) * jnp.arctan(b / a)
        + (b**2) / (6 * a**2) * jnp.log(b / a)
        + (a**2) / (6 * b**2) * jnp.log(a / b)
        - (a**4 - 6 * a**2 * b**2 + b**4) / (6 * a**2 * b**2) * jnp.log(a / b + b / a)
    )


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
    return jnp.exp(-25 / 6 + _rectangular_xsection_k(a, b))


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
    return (
        jnp.cross(rc_prime, rc_prime_prime)
        * (
            0.5
            * (-2 + jnp.log(64 * norm_rc_prime * norm_rc_prime / regularization))
            / (norm_rc_prime**3)
        )[:, None]
    )


@jit
def B_regularized_pure(
    gamma, gammadash, gammadashdash, quadpoints, current, regularization
):
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
    analytic_term = B_regularized_singularity_term(
        rc_prime, rc_prime_prime, regularization
    )
    dr = rc[:, None] - rc[None, :]
    first_term = (
        jnp.cross(rc_prime[None, :], dr)
        / ((jnp.sum(dr * dr, axis=2) + regularization) ** 1.5)[:, :, None]
    )
    cos_fac = 2.0 - 2.0 * jnp.cos(phi[None, :] - phi[:, None])
    second_term = (
        jnp.cross(rc_prime_prime, rc_prime)[:, None, :]
        * (
            0.5
            * cos_fac
            / (cos_fac * jnp.sum(rc_prime * rc_prime, axis=1)[:, None] + regularization)
            ** 1.5
        )[:, :, None]
    )
    integral_term = dphi * jnp.sum(first_term + second_term, 1)
    return current * Biot_savart_prefactor * (analytic_term + integral_term)


@jit
def B_bimpl(a, b, current, gammadash, gammadashdash):
    """
    Compute B_b on a coil following the Landreman and Hurwitz method (equation 21)

    Args:
        a, b  (float) : width and height of the conductor
        current (float): The current in the coil.
        gammadash (array (shape (n,3))): The first derivative of the curve.
        gammadashdash (array (shape (n,3))): The second derivative of the curve.

        The factors of 2π in the next few lines come from the fact that simsopt
        uses a curve parameter that goes up to 1 rather than 2π.

    Returns:
        array (shape (n,3)): Bb field on the coil.
    """

    rc_prime = gammadash / (2 * jnp.pi)
    rc_prime_prime = gammadashdash / (4 * jnp.pi**2)

    norm_rc_prime = jnp.linalg.norm(rc_prime, axis=1, keepdims=True)
    delta = _rectangular_xsection_delta(a, b)

    prefac = current * Biot_savart_prefactor * (2 + jnp.log(2) + jnp.log(delta))

    return prefac * jnp.cross(rc_prime, rc_prime_prime) / (norm_rc_prime**3)


def G_func(x, y):
    """
    Compute G function in the Landreman and Hurwitz method (formula 18) used in B0 computation
    """

    return y * jnp.arctan(x / y) + 0.5 * x * jnp.log(1 + (y * y) / (x * x))


@jit
def compute_frame(gamma, gammadash):
    """
    Compute t, p, q vectors (local cross-section coordinates) using equation (40) of Landreman (2025).
    See also figure 3 on the paper for conventions

    Args:
        gamma (array (n,3)): coil curve
        gammadash (array (n,3)): first derivative of the curve

    Returns:
        p (array (n,3))
        q (array (n,3))
        t (array (n,3))
    """
    rc_prime = gammadash / 2 / jnp.pi

    # unit tangent
    t = rc_prime / jnp.linalg.norm(rc_prime, axis=1)[:, None]

    # centroid of the coil
    C = jnp.mean(gamma, axis=0)

    # vector from centroid
    w = gamma - C

    # remove tangent component
    w_perp = w - (jnp.sum(w * t, axis=1)[:, None]) * t

    # normalize
    p = w_perp / jnp.linalg.norm(w_perp, axis=1)[:, None]

    # second perpendicular direction
    q = jnp.cross(t, p)

    return t, p, q


@jit
def B0_impl(u, v, a, b, current, p, q):
    """

    Compute B0 (formaula 17) on a coil following the Landreman and Hurwitz method

    The derivatives rc_prime, rc_prime_prime refer to an angle that goes up to
    2π, not up to 1.

    Args:
        u,v (floats, norm(u)<1, norm(v)<1) : position on the rectangular cross-section (see formula 6)
        a (float): The width of the rectangular conductor.
        b (float): The height of the rectangular conductor.
        current (float) : current in the conductor
        p,q (array (n,3)) : local coordinates on the cross section

    Returns:
        array (shape (n,3)): The magnetic field Bb (formula 21) on the coil."""

    su = jnp.array([-1.0, 1.0])
    sv = jnp.array([-1.0, 1.0])

    U = u[None, None, :] - su[:, None, None]  # (2,1,n)
    V = v[None, None, :] - sv[None, :, None]  # (1,2,n)

    G1 = G_func(b * V, a * U)[..., None]  # (2,2,n,1)
    G2 = G_func(a * U, b * V)[..., None]

    term = G1 * q - G2 * p  # broadcast (2,2,n,3)

    sign = (su[:, None] * sv[None, :])[..., None, None]  # (2,2,1,1)

    B = jnp.sum(sign * term, axis=(0, 1))  # (n,3)

    return current * Biot_savart_prefactor / (a * b) * B


@jit
def K_func(a, b, U, V, rc_prime, rc_prime_prime, t, p, q):

    norm_rc_prime = jnp.linalg.norm(rc_prime, axis=1, keepdims=True)
    kappa_b = jnp.cross(rc_prime, rc_prime_prime) / (norm_rc_prime**3)

    kappa_1 = jnp.sum(kappa_b * q, axis=1, keepdims=True)
    kappa_2 = jnp.sum(-kappa_b * p, axis=1, keepdims=True)

    U = U[:, None]
    V = V[:, None]

    base = a * U**2 / b + b * V**2 / a
    log_term = jnp.log(base)

    return (
        -2 * U * V * kappa_b * log_term
        + kappa_b * base * log_term
        + 4 * a * U**2 * kappa_2 * p * jnp.arctan(b * V / (a * U)) / b
        + 4 * b * V**2 * kappa_1 * q * jnp.arctan(a * U / (b * V)) / a
    )


@jit
def Bkappa_impl(u, v, a, b, current, gammadash, gammadashdash, t, p, q):
    """
    Computes Bkappa (formula 19)

    Args:
        u,v (floats, norm(u)<1, norm(v)<1) : position on the rectangular cross-section (see formula 6)
        a (float): The width of the rectangular conductor.
        b (float): The height of the rectangular conductor.
        current (float) : current in the conductor
        gammadash (array (shape (n,3))): The first derivative of the curve.
        gammadashdash (array (shape (n,3))): The second derivative of the curve.
        t,p,q (array (n,3)) : local coordinates on the cross section

    returns:
        array(n,3), Bkappa field
    """

    rc_prime = gammadash / 2 / jnp.pi
    rc_prime_prime = gammadashdash / 4 / jnp.pi**2

    B = jnp.zeros_like(p)

    for su in [-1, 1]:
        for sv in [-1, 1]:
            U = u - su
            V = v - sv

            term = K_func(a, b, U, V, rc_prime, rc_prime_prime, t, p, q)

            B += su * sv * term

    return current * Biot_savart_prefactor / 16 * B


def B_total_impl(
    gamma,
    gamma_dash,
    gamma_dashdash,
    quadpoints,
    u,
    v,
    a,
    b,
    t,
    p,
    q,
    current,
    regularization,
):
    """
    Compute the total field using B_regularized, B0, Bkappa and Bb
    """
    return (
        B_regularized_pure(
            gamma, gamma_dash, gamma_dashdash, quadpoints, current, regularization
        )
        + B0_impl(u, v, a, b, current, p, q)
        + Bkappa_impl(u, v, a, b, current, gamma_dash, gamma_dashdash, t, p, q)
        + B_bimpl(a, b, current, gamma_dash, gamma_dashdash)
    )
