"""Pure JAX helical-curve kernels."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from ._device_scalars import two_pi as _two_pi
from ._math_utils import as_jax_float64 as _as_jax_float64


def curve_helical_pure(dofs, quadpoints, order, m, ell, R0, r):
    """Pure function for the position vector used by CurveHelical."""
    dofs = _as_jax_float64(dofs)
    quadpoints = _as_jax_float64(quadpoints)
    A = jax.lax.slice_in_dim(dofs, 0, order + 1, axis=0)
    B = jnp.concatenate(
        (
            _as_jax_float64(np.zeros(1, dtype=np.float64)),
            jax.lax.slice_in_dim(dofs, order + 1, dofs.shape[0], axis=0),
        )
    )
    two_pi = _two_pi(quadpoints)
    ell_scale = _as_jax_float64(float(ell))
    m_scale = _as_jax_float64(float(m))
    phi = quadpoints * two_pi * ell_scale
    mode_numbers = _as_jax_float64(np.arange(order + 1, dtype=np.float64))
    k, phi_2d = jnp.meshgrid(mode_numbers, phi)
    phase = k * phi_2d * m_scale / ell_scale
    eta = m_scale * phi / ell_scale + jnp.sum(
        A * jnp.cos(phase) + B * jnp.sin(phase), axis=1
    )
    R0_scale = _as_jax_float64(float(R0))
    r_scale = _as_jax_float64(float(r))
    R = R0_scale + r_scale * jnp.cos(eta)
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    z = -r_scale * jnp.sin(eta)
    gamma = jnp.column_stack((x, y, z))
    return gamma
