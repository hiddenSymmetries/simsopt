"""Placed exact zeros survive eager AD, batching and disabled JIT."""

from __future__ import annotations

import numpy as np
from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import jax
    import jax.numpy as jnp
    from simsopt_jax.core._device_scalars import placement_zero


class TestPlacementZeroEager(JaxTestCase):
    def test_primal_tangent_and_transpose_stay_placed_without_jit(self) -> None:
        """Eager value, JVP and VJP are +0.0 on the input's device and dtype for non-finite inputs and seeds."""
        references = (
            np.asarray(np.nan),
            np.asarray((1.0e308, 1.0e308)),
            np.asarray((np.nan, np.inf, -np.inf)),
            np.asarray(((np.inf, np.nan), (-np.inf, 1.0))),
            np.empty((0,)),
        )
        for reference in references:
            placed = jnp.array(reference)
            for seed in (np.nan, np.inf, -np.inf, 1.0):
                tangent = jnp.array(np.full(reference.shape, seed))
                cotangent = jnp.array(np.asarray(seed))
                with self.subTest(shape=reference.shape, seed=seed):
                    with jax.disable_jit(), jax.transfer_guard("disallow"):
                        zero, derivative = jax.jvp(
                            placement_zero, (placed,), (tangent,)
                        )
                        _, pullback = jax.vjp(placement_zero, placed)
                        (transpose,) = pullback(cotangent)
                        jax.block_until_ready((zero, derivative, transpose))
                    for actual in (zero, derivative, transpose):
                        self.assertEqual(actual.devices(), placed.devices())
                        self.assertEqual(actual.dtype, placed.dtype)
                        np.testing.assert_array_equal(actual, np.zeros(actual.shape))
                        self.assertFalse(np.any(np.signbit(actual)))

    def test_nested_jvp_and_hvp_are_exact_zeros_without_jit(self) -> None:
        """Second-order JVPs and Hessian-vector products are exact +0.0, with and without JIT."""
        reference = jnp.array(np.asarray((np.nan, np.inf, 1.0e308)))
        direction = jnp.array(np.asarray((np.inf, np.nan, -np.inf)))
        seed = jnp.array(np.asarray(np.inf))
        seed_tangent = jnp.array(np.asarray(np.nan))

        def first_derivative(values: jax.Array) -> jax.Array:
            return jax.jvp(placement_zero, (values,), (values,))[1]

        def gradient(values: jax.Array, weight: jax.Array) -> jax.Array:
            _, pullback = jax.vjp(placement_zero, values)
            return pullback(weight)[0]

        for disable_jit in (False, True):
            with self.subTest(disable_jit=disable_jit):
                with jax.disable_jit(disable_jit), jax.transfer_guard("disallow"):
                    first, second = jax.jvp(
                        first_derivative, (reference,), (direction,)
                    )
                    derivative, hvp = jax.jvp(
                        gradient,
                        (reference, seed),
                        (direction, seed_tangent),
                    )
                    jax.block_until_ready((first, second, derivative, hvp))
                for actual in (first, second, derivative, hvp):
                    np.testing.assert_array_equal(actual, np.zeros(actual.shape))
                    self.assertFalse(np.any(np.signbit(actual)))

    def test_batched_zero_has_exact_nonfinite_jvp_and_transpose(self) -> None:
        """Under vmap, the value, JVP and VJP stay exact +0.0 for non-finite batches."""
        references = jnp.array(np.asarray(((np.nan, np.inf), (1.0e308, 1.0e308))))
        direction = jnp.array(np.full(references.shape, np.nan))
        seed = jnp.array(np.full((references.shape[0],), np.inf))
        with jax.disable_jit(), jax.transfer_guard("disallow"):
            values, tangent = jax.jvp(
                jax.vmap(placement_zero), (references,), (direction,)
            )
            _, pullback = jax.vjp(jax.vmap(placement_zero), references)
            (transpose,) = pullback(seed)
            jax.block_until_ready((values, tangent, transpose))
        for actual in (values, tangent, transpose):
            np.testing.assert_array_equal(actual, np.zeros(actual.shape))
            self.assertFalse(np.any(np.signbit(actual)))

    def test_finite_differences_agree_with_exact_zero_directional_derivative(
        self,
    ) -> None:
        """Central differences of placement_zero are exactly zero at every step size."""
        reference = np.asarray((1.0, -2.0, 3.0))
        direction = np.asarray((0.5, 0.2, -0.3))
        for step in (1.0e-2, 1.0e-4, 1.0e-6):
            plus = jnp.array(reference + step * direction)
            minus = jnp.array(reference - step * direction)
            with self.subTest(step=step):
                with jax.disable_jit(), jax.transfer_guard("disallow"):
                    values = (placement_zero(plus), placement_zero(minus))
                    jax.block_until_ready(values)
                finite_difference = (np.asarray(values[0]) - np.asarray(values[1])) / (
                    2.0 * step
                )
                np.testing.assert_array_equal(finite_difference, 0.0)
