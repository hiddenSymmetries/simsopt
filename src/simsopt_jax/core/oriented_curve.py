"""Pure JAX kernels for the OrientedCurveXYZFourier geometry.

These helpers are pure JAX functions over the curve degrees of freedom and
quadrature points. Tests should construct
``OrientedCurveXYZFourierSpec`` payloads directly and compare against an
independent geometry oracle instead of depending on legacy ``simsopt.geo``
adapter hooks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._math_utils import scalar_like
from ._math_utils import as_jax_int32 as _as_jax_int32


def _slice_vector(vector, start: int, stop: int):
    indices = _as_jax_int32(tuple(range(start, stop)))
    return jnp.take(vector, indices, axis=0)


def _vector_entry(vector, index: int):
    return jax.lax.squeeze(_slice_vector(vector, index, index + 1), (0,))


def shift_pure(v, xyz):
    """Translate Cartesian row-vector positions.

    Args:
        v (jax.Array): Cartesian points, shape (Q, 3), in meters.
        xyz (jax.Array): Translation, shape (3,), in meters.

    Returns:
        jax.Array: Translated positions, shape (Q, 3), in meters.
    """
    return v + jnp.expand_dims(xyz, axis=0)


def rotate_pure(v, ypr):
    """Apply yaw, pitch and roll using the native row-vector convention.

    Args:
        v (jax.Array): Cartesian row-vector positions, shape (Q, 3), in meters.
        ypr (jax.Array): Yaw, pitch, roll angles about z, y, x, shape (3,), in radians.

    Returns:
        jax.Array: Rotated positions, shape (Q, 3), in meters; row vectors are
            multiplied as v @ Myaw @ Mpitch @ Mroll.
    """
    yaw = _vector_entry(ypr, 0)
    pitch = _vector_entry(ypr, 1)
    roll = _vector_entry(ypr, 2)
    zero = scalar_like(yaw, 0.0)
    one = scalar_like(yaw, 1.0)

    Myaw = jnp.stack(
        (
            jnp.stack((jnp.cos(yaw), -jnp.sin(yaw), zero)),
            jnp.stack((jnp.sin(yaw), jnp.cos(yaw), zero)),
            jnp.stack((zero, zero, one)),
        )
    )
    Mpitch = jnp.stack(
        (
            jnp.stack((jnp.cos(pitch), zero, jnp.sin(pitch))),
            jnp.stack((zero, one, zero)),
            jnp.stack((-jnp.sin(pitch), zero, jnp.cos(pitch))),
        )
    )
    Mroll = jnp.stack(
        (
            jnp.stack((one, zero, zero)),
            jnp.stack((zero, jnp.cos(roll), -jnp.sin(roll))),
            jnp.stack((zero, jnp.sin(roll), jnp.cos(roll))),
        )
    )

    return v @ Myaw @ Mpitch @ Mroll


def centercurve_pure(dofs, quadpoints, order):
    """Evaluate an oriented Fourier curve from translation, angles and harmonics.

    Args:
        dofs (array-like): Shape (6 + 6 * order,); translation xyz in meters,
            yaw/pitch/roll in radians, then x/y/z sine/cosine blocks in meters with no
            constant modes.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.

    Returns:
        jax.Array: Fourier positions after rotation and translation, shape (Q,
            3), in meters.
    """
    xyz = _slice_vector(dofs, 0, 3)
    ypr = _slice_vector(dofs, 3, 6)
    fmn = _slice_vector(dofs, 6, dofs.shape[0])

    k = fmn.shape[0] // 3
    coeffs = [
        _slice_vector(fmn, 0, k),
        _slice_vector(fmn, k, 2 * k),
        _slice_vector(fmn, 2 * k, fmn.shape[0]),
    ]
    points = quadpoints
    two_pi = scalar_like(points, 2.0 * jnp.pi)
    gamma_components = []
    for i in range(0, 3):
        component = points - points
        for j in range(0, order):
            mode = scalar_like(two_pi, float(j + 1))
            angle = two_pi * mode * points
            component = component + (_vector_entry(coeffs[i], 2 * j) * jnp.sin(angle))
            component = component + (
                _vector_entry(coeffs[i], 2 * j + 1) * jnp.cos(angle)
            )
        gamma_components.append(component)
    gamma = jnp.stack(tuple(gamma_components), axis=1)

    return shift_pure(rotate_pure(gamma, ypr), xyz)
