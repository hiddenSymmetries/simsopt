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
    """Apply translation in cartesian coordinates.

    Args:
     - v: array to translate. Should have size Nx3.
     - xyz: translation vector. Should have size 3.

    Returns:
     - v+xyz: translated array, size Nx3
    """
    return v + jnp.expand_dims(xyz, axis=0)


def rotate_pure(v, ypr):
    """Apply rotation around x, y, and z axis.

    Args:
     - v: set of points to rotate. Should have size Nx3.
     - ypr: rotation angles.
            ypr[0] describes the rotation around the z-axis.
            ypr[1] describes the rotation around the y-axis.
            ypr[2] describes the rotation around the x-axis.

    Returns:
    - v: Rotated set of points
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
    """Construct curve centered at the origin.

    Args:
     - dofs: Set of degrees of freedom
     - quadpoints: Quadrature points. Array of size N, with float values between 0 and 1.
     - order: Maximum Fourier mode number.

    Returns:
     - gamma: Curve that has been translated and rotated to the desired position.
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
