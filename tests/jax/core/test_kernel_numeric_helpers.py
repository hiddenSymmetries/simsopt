"""Regression tests for small JAX kernel numeric helpers."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from simsopt_jax.core.biotsavart import (
    _radius_squared,
    biot_savart_A,
    biot_savart_B,
    biot_savart_dB_by_dX,
)
from simsopt_jax.core.curve_xyz_fourier import _constant_row


def test_radius_squared_preserves_zero_singularity():
    diff = jnp.zeros((1, 3), dtype=jnp.float64)

    radius_squared = _radius_squared(diff)

    np.testing.assert_allclose(np.asarray(radius_squared), np.zeros((1,)))


def test_biotsavart_point_singularity_gradient_is_nonfinite():
    def singular_kernel(x):
        diff = jnp.reshape(x, (1, 3))
        r2 = _radius_squared(diff)[0]
        return x[0] / (r2**1.5)

    gradient = jax.grad(singular_kernel)(jnp.zeros((3,), dtype=jnp.float64))

    assert not np.all(np.isfinite(np.asarray(gradient)))


def test_biot_savart_public_kernels_preserve_point_singularity():
    points = jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float64)
    gammas = jnp.asarray([[[0.0, 0.0, 0.0]]], dtype=jnp.float64)
    gammadashs = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float64)
    currents = jnp.asarray([1.0], dtype=jnp.float64)

    for kernel in (biot_savart_A, biot_savart_B, biot_savart_dB_by_dX):
        value = kernel(points, gammas, gammadashs, currents)

        assert not np.all(np.isfinite(np.asarray(value)))


def test_constant_row_builds_its_row_from_a_traced_reference():
    """The row is selected statically but sourced from the reference's device."""

    @jax.jit
    def rows_for(reference):
        return (
            _constant_row(3, is_one=True, reference=reference),
            _constant_row(3, is_one=False, reference=reference),
        )

    ones_row, zeros_row = rows_for(jnp.asarray(1.0, dtype=jnp.float64))

    np.testing.assert_allclose(np.asarray(ones_row), np.ones((1, 3)))
    np.testing.assert_allclose(np.asarray(zeros_row), np.zeros((1, 3)))
