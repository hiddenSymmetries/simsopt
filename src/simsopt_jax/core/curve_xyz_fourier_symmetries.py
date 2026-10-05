"""Pure JAX XYZ-Fourier-symmetries curve kernels."""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from ._device_scalars import two_pi as _two_pi
from ._math_utils import as_runtime_float64 as _as_runtime_float64


def jaxXYZFourierSymmetriescurve_pure(dofs, quadpoints, order, nfp, stellsym, ntor):
    two_pi = _two_pi(quadpoints)
    nfp_scalar = _as_runtime_float64(float(nfp), reference=quadpoints)
    ntor_scalar = _as_runtime_float64(float(ntor), reference=quadpoints)
    modes = _as_runtime_float64(
        np.arange(order + 1, dtype=np.float64), reference=quadpoints
    )

    theta = jnp.expand_dims(quadpoints, axis=1)
    m_row = jnp.expand_dims(modes, axis=0)
    angle_full = two_pi * nfp_scalar * m_row * theta
    cos_full = jnp.cos(angle_full)
    sin_tail = jnp.sin(angle_full[:, 1:])

    if stellsym:
        xc = dofs[: order + 1]
        ys = dofs[order + 1 : 2 * order + 1]
        zs = dofs[2 * order + 1 :]

        xhat = jnp.sum(xc[None, :] * cos_full, axis=1)
        yhat = jnp.sum(ys[None, :] * sin_tail, axis=1)
        z = jnp.sum(zs[None, :] * sin_tail, axis=1)
    else:
        xc = dofs[0 : order + 1]
        xs = dofs[order + 1 : 2 * order + 1]
        yc = dofs[2 * order + 1 : 3 * order + 2]
        ys = dofs[3 * order + 2 : 4 * order + 2]
        zc = dofs[4 * order + 2 : 5 * order + 3]
        zs = dofs[5 * order + 3 :]

        xhat = jnp.sum(xc[None, :] * cos_full, axis=1) + jnp.sum(
            xs[None, :] * sin_tail, axis=1
        )
        yhat = jnp.sum(yc[None, :] * cos_full, axis=1) + jnp.sum(
            ys[None, :] * sin_tail, axis=1
        )
        z = jnp.sum(zc[None, :] * cos_full, axis=1) + jnp.sum(
            zs[None, :] * sin_tail, axis=1
        )

    angle = two_pi * quadpoints * ntor_scalar
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    x = cos_angle * xhat - sin_angle * yhat
    y = sin_angle * xhat + cos_angle * yhat
    return jnp.stack((x, y, z), axis=1)
