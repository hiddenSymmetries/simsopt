"""An exact placed zero must have exact zero derivatives even for non-finite seeds.

Fixed currents need a tangent on the reference device under strict transfer
guards. Computing zero from reference values could overflow for finite inputs.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax.core._device_scalars import placement_zero

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


@pytest.mark.parametrize("reference", REFERENCES)
def test_placement_zero_is_an_exact_zero_on_the_reference_device(reference) -> None:
    placed = _placed(reference)
    with jax.transfer_guard("disallow"):
        zero = placement_zero(placed)
        jax.block_until_ready(zero)
    assert zero.dtype == jnp.float64
    assert zero.devices() == placed.devices()
    assert np.asarray(zero).view(np.uint64) == np.float64(0.0).view(np.uint64)


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_placement_zero_tangent_is_exactly_zero(reference, seed) -> None:
    placed = _placed(reference)
    tangent = _placed((seed, 1.0))
    with jax.transfer_guard("disallow"):
        _, zero_tangent = jax.jvp(placement_zero, (placed,), (tangent,))
        jax.block_until_ready(zero_tangent)
    assert zero_tangent.devices() == placed.devices()
    assert np.asarray(zero_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64)


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_placement_zero_cotangent_is_exactly_zero(reference, seed) -> None:
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


def test_placement_zero_linearizes_a_fixed_value_on_device() -> None:
    """A fixed value plus the placed zero linearizes under the strict guard."""
    placed = _placed((1.0e308, 1.0e308))
    fixed = _placed(3.0)
    tangent = _placed((np.inf, np.nan))
    with jax.transfer_guard("disallow"):
        value, linearized = jax.linearize(lambda r: fixed + placement_zero(r), placed)
        fixed_tangent = linearized(tangent)
        jax.block_until_ready((value, fixed_tangent))
    assert float(value) == 3.0
    assert np.asarray(fixed_tangent).view(np.uint64) == np.float64(0.0).view(np.uint64)
