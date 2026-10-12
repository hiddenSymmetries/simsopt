"""An exact placed zero must have exact zero derivatives even for non-finite seeds.

Fixed currents need a tangent on the reference device under strict transfer
guards. Computing zero from reference values could overflow for finite inputs.
"""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax.core._device_scalars import placement_zero


import numpy as np


if JAX_IMPORT_ERROR is None:
    jax.config.update("jax_enable_x64", True)

REFERENCES = (
    (1.0e308, 1.0e308),
    (-1.0e308, -1.0e308),
    (np.nan, 1.0),
    (np.inf, -np.inf),
    (1.0, 2.0),
)
SEEDS = (np.nan, np.inf, -np.inf, 1.0)


def _placed(values) -> jax.Array:
    return jax.device_put(np.asarray(values, dtype=np.float64))


class TestPlacementZero(JaxTestCase):
    def test_placement_zero_is_an_exact_zero_on_the_reference_device(self):
        """placement_zero is bitwise positive zero with the reference dtype and device
        under a strict guard."""
        for reference in REFERENCES:
            with self.subTest(reference=reference), self.case():
                self._case_placement_zero_is_an_exact_zero_on_the_reference_device(
                    reference
                )

    def _case_placement_zero_is_an_exact_zero_on_the_reference_device(self, reference):
        """Run one row with case-local objects released before runtime cleanup."""
        placed = _placed(reference)
        with jax.transfer_guard("disallow"):
            zero = placement_zero(placed)
            jax.block_until_ready(zero)
        self.assertTrue(zero.dtype == jnp.float64, "zero.dtype == jnp.float64")
        self.assertTrue(
            zero.devices() == placed.devices(),
            "zero.devices() == placed.devices()",
        )
        self.assertTrue(
            np.asarray(zero).view(np.uint64) == np.float64(0.0).view(np.uint64),
            "np.asarray(zero).view(np.uint64) == np.float64(0.0).view(np.uint64)",
        )

    def test_placement_zero_tangent_is_exactly_zero(self):
        """placement_zero produces a bitwise zero tangent on the reference device for
        every tested seed."""
        for seed in SEEDS:
            for reference in REFERENCES:
                with self.subTest(seed=seed, reference=reference), self.case():
                    self._case_placement_zero_tangent_is_exactly_zero(seed, reference)

    def _case_placement_zero_tangent_is_exactly_zero(self, seed, reference):
        """Run one row with case-local objects released before runtime cleanup."""
        placed = _placed(reference)
        tangent = _placed((seed, 1.0))
        with jax.transfer_guard("disallow"):
            _, zero_tangent = jax.jvp(placement_zero, (placed,), (tangent,))
            jax.block_until_ready(zero_tangent)
        self.assertTrue(
            zero_tangent.devices() == placed.devices(),
            "zero_tangent.devices() == placed.devices()",
        )
        self.assertTrue(
            np.asarray(zero_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64),
            "np.asarray(zero_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64)",
        )

    def test_placement_zero_cotangent_is_exactly_zero(self):
        """placement_zero sends bitwise positive-zero cotangents to every reference
        entry."""
        for seed in SEEDS:
            for reference in REFERENCES:
                with self.subTest(seed=seed, reference=reference), self.case():
                    self._case_placement_zero_cotangent_is_exactly_zero(seed, reference)

    def _case_placement_zero_cotangent_is_exactly_zero(self, seed, reference):
        """Run one row with case-local objects released before runtime cleanup."""
        placed = _placed(reference)
        cotangent = _placed(seed)
        with jax.transfer_guard("disallow"):
            _, pullback = jax.vjp(placement_zero, placed)
            (reference_cotangent,) = pullback(cotangent)
            jax.block_until_ready(reference_cotangent)
        np.testing.assert_array_equal(
            np.asarray(reference_cotangent).view(np.uint64),
            np.zeros(2, dtype=np.float64).view(np.uint64),
        )

    def test_placement_zero_linearizes_a_fixed_value_on_device(self):
        """A fixed value plus the placed zero linearizes under the strict guard."""
        placed = _placed((1.0e308, 1.0e308))
        fixed = _placed(3.0)
        tangent = _placed((np.inf, np.nan))
        with jax.transfer_guard("disallow"):
            value, linearized = jax.linearize(
                lambda r: fixed + placement_zero(r), placed
            )
            fixed_tangent = linearized(tangent)
            jax.block_until_ready((value, fixed_tangent))
        self.assertTrue(float(value) == 3.0, "float(value) == 3.0")
        self.assertTrue(
            np.asarray(fixed_tangent).view(np.uint64)
            == np.float64(0.0).view(np.uint64),
            "np.asarray(fixed_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64)",
        )
