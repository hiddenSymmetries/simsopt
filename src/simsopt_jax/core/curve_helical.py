"""Pure JAX helical-curve kernels."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from ._device_scalars import two_pi as _two_pi
from ._math_utils import as_runtime_float64 as _as_runtime_float64_ref


def curve_helical_pure(dofs, quadpoints, order, m, ell, R0, r):
    """Pure function for the position vector used by CurveHelical."""
    dofs = _as_runtime_float64_ref(dofs, reference=quadpoints)
    quadpoints = _as_runtime_float64_ref(quadpoints, reference=dofs)
    A = jax.lax.slice_in_dim(dofs, 0, order + 1, axis=0)
    B = jnp.concatenate(
        (
            _as_runtime_float64_ref(np.zeros(1, dtype=np.float64), reference=dofs),
            jax.lax.slice_in_dim(dofs, order + 1, dofs.shape[0], axis=0),
        )
    )
    two_pi = _two_pi(quadpoints)
    ell_scale = _as_runtime_float64_ref(float(ell), reference=quadpoints)
    m_scale = _as_runtime_float64_ref(float(m), reference=quadpoints)
    phi = quadpoints * two_pi * ell_scale
    mode_numbers = _as_runtime_float64_ref(
        np.arange(order + 1, dtype=np.float64),
        reference=phi,
    )
    k, phi_2d = jnp.meshgrid(mode_numbers, phi)
    phase = k * phi_2d * m_scale / ell_scale
    eta = m_scale * phi / ell_scale + jnp.sum(
        A * jnp.cos(phase) + B * jnp.sin(phase), axis=1
    )
    R0_scale = _as_runtime_float64_ref(float(R0), reference=eta)
    r_scale = _as_runtime_float64_ref(float(r), reference=eta)
    R = R0_scale + r_scale * jnp.cos(eta)
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    z = -r_scale * jnp.sin(eta)
    gamma = jnp.column_stack((x, y, z))
    return gamma
