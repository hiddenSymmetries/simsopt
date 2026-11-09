"""Pure JAX XYZ-Fourier curve kernels."""

from __future__ import annotations

import jax.numpy as jnp

from ._device_scalars import device_one, placement_zero, two_pi
from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
)


def _mode_numbers(order, *, reference):
    one = device_one(reference)
    return jnp.cumsum(jnp.broadcast_to(one, (int(order),)))


def _constant_row(length: int, *, is_one: bool, reference):
    # Nested curve derivatives concatenate this row's tangent with harmonics;
    # it must be a placed array rather than stop_gradient's symbolic zero.
    zero = placement_zero(reference)
    scalar = device_one(reference) + zero if is_one else zero
    return jnp.broadcast_to(scalar, (1, int(length)))


def _interleave_harmonics(first, second):
    return jnp.reshape(jnp.stack((first, second), axis=1), (-1, first.shape[1]))


def _fourier_basis(quadpoints, order):
    quadpoints = _as_jax_float64(quadpoints)
    points = two_pi(quadpoints) * quadpoints
    mode_numbers = _mode_numbers(order, reference=points)
    phase = jnp.expand_dims(mode_numbers, axis=1) * jnp.expand_dims(points, axis=0)
    return jnp.concatenate(
        (
            _constant_row(points.shape[0], is_one=True, reference=points),
            _interleave_harmonics(jnp.sin(phase), jnp.cos(phase)),
        ),
        axis=0,
    )


def jaxfouriercurve_pure(dofs, quadpoints, order):
    """Return XYZ-Fourier curve positions.

    Args:
        dofs (array-like): Shape (3 * (2 * order + 1),), in meters; x, y, z blocks each
            use constant, sin(1), cos(1), ..., sin(order), cos(order).
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.

    Returns:
        jax.Array: Cartesian curve positions, shape (Q, 3), in meters.
    """
    dofs = _as_jax_float64(dofs)
    coeffs = jnp.reshape(dofs, (3, dofs.shape[0] // 3))
    basis = _fourier_basis(quadpoints, order)
    gamma = coeffs @ basis
    return jnp.moveaxis(gamma, 0, -1)

