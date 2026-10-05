"""Pure JAX RZ-Fourier curve kernels."""

from __future__ import annotations

import numpy as np
from typing import cast

import jax
import jax.numpy as jnp

from ._device_scalars import two_pi as _two_pi
from ._math_utils import as_runtime_float64 as _as_runtime_float64_ref


def curverzfourier_pure(dofs, quadpoints, order, nfp, stellsym):
    quadpoints = _as_runtime_float64_ref(quadpoints, reference=dofs)
    phi = _two_pi(quadpoints) * quadpoints
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)

    rc = jax.lax.slice_in_dim(dofs, 0, order + 1, axis=0)
    if stellsym:
        zc = None
        zs = jax.lax.slice_in_dim(dofs, order + 1, dofs.shape[0], axis=0)
    else:
        rs = jax.lax.slice_in_dim(dofs, order + 1, 2 * order + 1, axis=0)
        zc = jax.lax.slice_in_dim(dofs, 2 * order + 1, 3 * order + 2, axis=0)
        zs = jax.lax.slice_in_dim(dofs, 3 * order + 2, dofs.shape[0], axis=0)

    cos_modes = _as_runtime_float64_ref(
        np.arange(order + 1, dtype=np.float64),
        reference=phi,
    )
    nfp_scale = _as_runtime_float64_ref(float(nfp), reference=phi)
    cos_phase = phi[:, None] * (nfp_scale * cos_modes)[None, :]
    radius = jnp.sum(rc[None, :] * jnp.cos(cos_phase), axis=1)

    sin_modes = _as_runtime_float64_ref(
        np.arange(1, order + 1, dtype=np.float64),
        reference=phi,
    )
    if order > 0:
        sin_phase = phi[:, None] * (nfp_scale * sin_modes)[None, :]
        z = jnp.sum(zs[None, :] * jnp.sin(sin_phase), axis=1)
        if not stellsym:
            radius = radius + jnp.sum(rs[None, :] * jnp.sin(sin_phase), axis=1)
    else:
        z = jnp.zeros_like(phi)

    if not stellsym:
        z = z + jnp.sum(cast(jax.Array, zc)[None, :] * jnp.cos(cos_phase), axis=1)

    return jnp.column_stack((radius * cosphi, radius * sinphi, z))
