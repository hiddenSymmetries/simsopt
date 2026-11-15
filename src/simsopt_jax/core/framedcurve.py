"""Pure JAX reference-frame kernels for finite-build coil geometry.

Provide centroid and Frenet frames and Fourier frame rotations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from simsopt_jax.backend.jit import jit

from ._device_scalars import two_pi as _two_pi
from ._math_utils import as_jax_float64 as _as_jax_float64



def _row_norm(matrix: jax.Array) -> jax.Array:
    """Return row norms as a column vector of shape (N, 1)."""
    return jnp.linalg.norm(matrix, axis=1)[:, None]


def _row_dot(left: jax.Array, right: jax.Array) -> jax.Array:
    """Return row-wise dot products as a column vector of shape (N, 1)."""
    return jnp.sum(left * right, axis=1)[:, None]


def _rotate_frame(
    n: jax.Array,
    b: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Rotate normal and binormal using the native frame-angle convention."""
    cos_alpha = jnp.cos(alpha)[:, None]
    sin_alpha = jnp.sin(alpha)[:, None]
    nn = cos_alpha * n - sin_alpha * b
    bb = sin_alpha * n + cos_alpha * b
    return nn, bb


@jit
def centroid_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the unrotated coil-centroid frame ``(t, n, b)``.

    Args:
        gamma (jax.Array): Cartesian curve positions, shape (Q, 3), in meters.
        gammadash (jax.Array): First derivative with respect to the normalized curve
            parameter, shape (Q, 3), in meters.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Dimensionless unit tangent,
            normal and binormal, each shape (Q, 3); nondegenerate tangent/frame
            norms are required.
    """
    gamma_jax = _as_jax_float64(gamma)
    gammadash_jax = _as_jax_float64(gammadash)
    arc_length = _row_norm(gammadash_jax)
    t = gammadash_jax / arc_length
    centroid = jnp.mean(gamma_jax, axis=0)
    delta = gamma_jax - centroid[None, :]
    projection = _row_dot(delta, t) * t
    n_unnormed = delta - projection
    n = n_unnormed / _row_norm(n_unnormed)
    b = jnp.cross(t, n, axis=1)
    return t, n, b


@jit
def rotated_centroid_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the coil-centroid frame ``(t, n, b)`` rotated by ``alpha``.

    Positive angles rotate the normal toward the negative binormal.

    Args:
        gamma (jax.Array): Cartesian curve positions, shape (Q, 3), in meters.
        gammadash (jax.Array): First derivative with respect to the normalized curve
            parameter, shape (Q, 3), in meters.
        alpha (jax.Array): Normal/binormal rotation angles, shape (Q,), in radians.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Dimensionless unit tangent,
            normal and binormal, each shape (Q, 3); nondegenerate tangent/frame
            norms are required.
    """
    t, n, b = centroid_frame(gamma, gammadash)
    alpha_jax = _as_jax_float64(alpha)
    nn, bb = _rotate_frame(n, b, alpha_jax)
    return t, nn, bb


@jit
def frenet_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the unrotated Frenet frame ``(t, n, b)``.

    Args:
        gamma (jax.Array): Cartesian curve positions, shape (Q, 3), in meters. Accepted
            for API symmetry; the Frenet construction does not use positions.
        gammadash (jax.Array): First derivative with respect to the normalized curve
            parameter, shape (Q, 3), in meters.
        gammadashdash (jax.Array): Second parameter derivative, shape (Q, 3), in meters.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Dimensionless unit tangent,
            normal and binormal, each shape (Q, 3); nondegenerate tangent/frame
            norms are required.
    """
    del gamma  # Retained for API symmetry; Frenet frame is curvature-only.
    gammadash_jax = _as_jax_float64(gammadash)
    gammadashdash_jax = _as_jax_float64(gammadashdash)
    t, tdash = jax.jvp(
        lambda tangent: tangent / _row_norm(tangent),
        (gammadash_jax,),
        (gammadashdash_jax,),
    )
    n = tdash / _row_norm(tdash)
    b = jnp.cross(t, n, axis=1)
    return t, n, b


@jit
def rotated_frenet_frame(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the Frenet frame ``(t, n, b)`` rotated by ``alpha``.

    Positive angles rotate the normal toward the negative binormal.
    Results match the native frame to round-off.

    Args:
        gamma (jax.Array): Cartesian curve positions, shape (Q, 3), in meters. Accepted
            for API symmetry; the Frenet construction does not use positions.
        gammadash (jax.Array): First derivative with respect to the normalized curve
            parameter, shape (Q, 3), in meters.
        gammadashdash (jax.Array): Second parameter derivative, shape (Q, 3), in meters.
        alpha (jax.Array): Normal/binormal rotation angles, shape (Q,), in radians.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Dimensionless unit tangent,
            normal and binormal, each shape (Q, 3); nondegenerate tangent/frame
            norms are required.
    """
    t, n, b = frenet_frame(gamma, gammadash, gammadashdash)
    alpha_jax = _as_jax_float64(alpha)
    nn, bb = _rotate_frame(n, b, alpha_jax)
    return t, nn, bb


def _rotation_alpha_impl(dofs: jax.Array, points: jax.Array, order: int) -> jax.Array:
    """Evaluate the unscaled Fourier frame-rotation angle.

    Args:
        dofs (array-like): Coefficients, shape (2*order+1,), in radians:
            constant, sin(1), cos(1), ..., sin(order), cos(order).
        points (array-like): Normalized curve parameters phi in [0, 1), shape (Q,).
        order (int): Highest Fourier mode; static under ``jit``.

    Returns:
        jax.Array: Rotation angles, shape (Q,), in radians, float64.
    """
    dofs_jax = _as_jax_float64(dofs)
    points_jax = _as_jax_float64(points)
    rotation = jnp.broadcast_to(dofs_jax[0], points_jax.shape)
    two_pi = _two_pi(points_jax)
    for mode in range(1, int(order) + 1):
        angle = two_pi * mode * points_jax
        rotation = rotation + dofs_jax[2 * mode - 1] * jnp.sin(angle)
        rotation = rotation + dofs_jax[2 * mode] * jnp.cos(angle)
    return rotation


def _rotation_alphadash_impl(
    dofs: jax.Array, points: jax.Array, order: int
) -> jax.Array:
    """Return the unscaled parameter derivative of :func:`rotation_alpha`.

    Uses the same arguments; returns float64 values of shape (Q,), in radians
    per unit normalized parameter.
    """
    points_jax = _as_jax_float64(points)
    _, alphadash = jax.jvp(
        lambda phi: _rotation_alpha_impl(dofs, phi, order),
        (points_jax,),
        (jnp.ones_like(points_jax),),
    )
    return alphadash


rotation_alpha = jit(_rotation_alpha_impl, static_argnames=("order",))
rotation_alphadash = jit(_rotation_alphadash_impl, static_argnames=("order",))
