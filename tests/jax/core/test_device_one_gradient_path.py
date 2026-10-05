"""``device_one`` must add no derivative path.

``device_one(r)`` builds a 1.0 on ``r``'s device as ``exp(sum(r - r))``.  Its
derivative is zero, but when it was left differentiable reverse mode sent the
whole cotangent ``c`` of each product ``device_one(r) * y`` into ``r`` twice, as
``+c`` and ``-c``, beside ``r``'s own cotangent ``g``: the accumulated ``(g + c)
- c`` then carries an error ``u |c|`` instead of ``u |g|``.  In Biot-Savart the
product is ``mu0/4pi * (currents . integral)`` with the currents as reference,
so ``c`` is the field's whole cotangent contraction and ``g = c / I``: the
current gradient of a Stage-II flux lost five digits (relative error 1e-11 at
``I = 1e5``, planar-coils profile 2026-09-29).
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from simsopt_jax.core._device_scalars import device_one
from simsopt_jax.core.biotsavart import biot_savart_B

UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0


def _gamma(k: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)``."""
    return k * UNIT_ROUNDOFF / (1.0 - k * UNIT_ROUNDOFF)


def test_device_one_has_a_zero_cotangent() -> None:
    reference = jnp.asarray([1.0e5, -3.0, 2.5e-7])
    cotangent = jax.grad(lambda r: device_one(r) * jnp.sum(r * r))(reference)
    np.testing.assert_array_equal(np.asarray(cotangent), 2.0 * np.asarray(reference))


def _coils(quadrature: int):
    t = np.arange(quadrature) / quadrature
    phi = 2.0 * np.pi * t
    gammas, gammadashs = [], []
    for radius, height in ((0.5, 0.0), (0.4, 0.3)):
        gammas.append(
            np.stack(
                [1.0 + radius * np.cos(phi), radius * np.sin(phi), height + 0.0 * phi],
                axis=1,
            )
        )
        gammadashs.append(
            2.0
            * np.pi
            * np.stack([-radius * np.sin(phi), radius * np.cos(phi), 0.0 * phi], axis=1)
        )
    return np.stack(gammas), np.stack(gammadashs)


def test_biot_savart_current_cotangent_is_the_unit_field_contraction() -> None:
    """``d(ct . B)/dI_c = ct . B_c(I_c = 1)`` to the rounding of that contraction.

    Both sides evaluate ``1e-7 / Q * sum_{p, j, q} ct[p, j] * k[c, q, p, j]``
    for the Biot-Savart integrand ``k`` over ``Q`` quadrature points, only in
    different orders, so they differ by at most ``2 gamma_K S`` with ``S`` the
    sum of the absolute products and ``K`` the additions (``3 P Q - 1``) plus
    the integrand's own roundings (at most 16: difference, squared radius,
    reciprocal square root and cube, cross product, scalings).  The currents are
    ``1e5``, the planar-coils scale, where the removed ``device_one`` path cost
    a relative 1e-11.
    """
    quadrature = 32
    gammas, gammadashs = _coils(quadrature)
    points = np.array(
        [[1.0, 0.0, 0.1], [1.2, 0.3, -0.2], [0.7, -0.4, 0.25], [1.1, 0.05, 0.6]]
    )
    currents = np.array([1.0e5, -1.0e5])
    cotangent = np.random.default_rng(20260929).standard_normal(points.shape)

    _, pullback = jax.vjp(
        lambda i: biot_savart_B(
            jnp.asarray(points), jnp.asarray(gammas), jnp.asarray(gammadashs), i
        ),
        jnp.asarray(currents),
    )
    (current_cotangent,) = pullback(jnp.asarray(cotangent))

    difference = points[None, None, :, :] - gammas[:, :, None, :]  # [coil, q, point, 3]
    radius = np.linalg.norm(difference, axis=-1)
    integrand = np.cross(gammadashs[:, :, None, :], difference) / radius[..., None] ** 3
    scale = 1.0e-7 / quadrature
    expected = scale * np.einsum("pj,cqpj->c", cotangent, integrand)
    magnitude = scale * np.einsum("pj,cqpj->c", np.abs(cotangent), np.abs(integrand))
    additions = 3 * points.shape[0] * quadrature - 1
    bound = 2.0 * _gamma(additions + 16) * magnitude

    error = np.abs(np.asarray(current_cotangent) - expected)
    assert np.all(error <= bound), (error, bound, error / np.abs(expected))
