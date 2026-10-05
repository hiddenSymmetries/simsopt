"""Pure JAX planar-Fourier curve kernels."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from ._device_scalars import two_pi as _two_pi
from ._math_utils import as_runtime_float64 as _as_runtime_float64_ref


def _normalized_quaternion(quaternion):
    norm_sq = jnp.sum(quaternion * quaternion)
    zero = _as_runtime_float64_ref(0.0, reference=norm_sq)
    one = _as_runtime_float64_ref(1.0, reference=norm_sq)
    safe_norm_sq = jnp.where(norm_sq > zero, norm_sq, one)
    normalized = quaternion / jnp.sqrt(safe_norm_sq)
    zero_quaternion = quaternion - quaternion
    return jnp.where(norm_sq > zero, normalized, zero_quaternion)


def _quaternion_rotation_matrix(quaternion):
    q0, q1, q2, q3 = quaternion
    one = _as_runtime_float64_ref(1.0, reference=quaternion)
    two = _as_runtime_float64_ref(2.0, reference=quaternion)
    return jnp.stack(
        (
            jnp.stack(
                (
                    one - two * (q2 * q2 + q3 * q3),
                    two * (q1 * q2 - q3 * q0),
                    two * (q1 * q3 + q2 * q0),
                )
            ),
            jnp.stack(
                (
                    two * (q1 * q2 + q3 * q0),
                    one - two * (q1 * q1 + q3 * q3),
                    two * (q2 * q3 - q1 * q0),
                )
            ),
            jnp.stack(
                (
                    two * (q1 * q3 - q2 * q0),
                    two * (q2 * q3 + q1 * q0),
                    one - two * (q1 * q1 + q2 * q2),
                )
            ),
        )
    )


def curveplanarfourier_pure(dofs, quadpoints, order):
    rc_end = order + 1
    rs_end = rc_end + order

    rc = jax.lax.slice_in_dim(dofs, 0, rc_end, axis=0)
    rs = jax.lax.slice_in_dim(dofs, rc_end, rs_end, axis=0)
    quaternion = _normalized_quaternion(
        jax.lax.slice_in_dim(dofs, rs_end, rs_end + 4, axis=0)
    )
    center = jax.lax.slice_in_dim(dofs, rs_end + 4, dofs.shape[0], axis=0)

    quadpoints = _as_runtime_float64_ref(quadpoints, reference=dofs)
    phi = _two_pi(quadpoints) * quadpoints
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)
    zero = _as_runtime_float64_ref(0.0, reference=phi)

    radius = jnp.broadcast_to(
        jnp.sum(jax.lax.slice_in_dim(rc, 0, 1, axis=0)), phi.shape
    )
    if order > 0:
        rc_tail = jax.lax.slice_in_dim(rc, 1, rc.shape[0], axis=0)
        modes = _as_runtime_float64_ref(
            np.arange(1, order + 1, dtype=np.float64),
            reference=phi,
        )
        phase = phi[:, None] * modes[None, :]
        radius = radius + jnp.sum(
            rc_tail[None, :] * jnp.cos(phase) + rs[None, :] * jnp.sin(phase),
            axis=1,
        )

    base_curve = jnp.column_stack(
        (
            radius * cosphi,
            radius * sinphi,
            phi * zero,
        )
    )
    rotation = _quaternion_rotation_matrix(quaternion)
    return base_curve @ rotation.T + center[None, :]
