"""Fresh-process child: the raw Biot-Savart kernel computes in float64 with
nothing but ``simsopt_jax`` imported.

The parent sets ``JAX_PLATFORMS=cpu`` and, in one row, ``JAX_ENABLE_X64`` to a
disabling value before this process starts. Neither ``simsopt.geo`` nor
``set_backend`` runs here, so only the ``simsopt_jax`` import can enable x64.
"""

import numpy as np
from simsopt_jax.core.biotsavart import biot_savart_B


def check_on_axis_loop_field() -> None:
    """Compare B on the axis of a circular loop with mu0 I R^2 / (2 (R^2 + z^2)^(3/2))."""
    radius, current, quadpoints = 1.0, 1.0e5, 64
    phi = 2.0 * np.pi * np.arange(quadpoints) / quadpoints
    gamma = radius * np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=-1)
    # d gamma / d t for the normalized parameter t = phi / (2 pi).
    gammadash = 2.0 * np.pi * radius * np.stack(
        (-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1
    )
    heights = np.array([0.0, 0.3, 1.7])
    points = np.stack((np.zeros(3), np.zeros(3), heights), axis=-1)
    B = biot_savart_B(points, gamma[None], gammadash[None], np.array([current]))
    mu0 = 4.0e-7 * np.pi
    B_z = mu0 * current * radius**2 / (2.0 * (radius**2 + heights**2) ** 1.5)
    assert B.dtype == np.float64, B.dtype
    # A float32 field misses this by about 1e-7 relative.
    np.testing.assert_allclose(np.asarray(B)[:, 2], B_z, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(np.asarray(B)[:, :2], 0.0, rtol=0.0, atol=1e-12 * B_z.max())


if __name__ == "__main__":
    check_on_axis_loop_field()
