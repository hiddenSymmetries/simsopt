"""Fresh-process children for runtime placement across two JAX devices.

Usage: ``backend_dtypes_placement.py SCENARIO``. Two host devices exist only if
XLA is told so before it initializes, so the parent exports ``JAX_PLATFORMS``,
``JAX_ENABLE_X64`` and, for the CPU scenarios,
``XLA_FLAGS=--xla_force_host_platform_device_count=2`` before this process
starts.
"""

import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.backend import dtypes
from simsopt_jax.backend.runtime import invalidate_backend_cache, set_backend
from simsopt_jax.core._device_scalars import staged_like


def _placed_leaf(value) -> jax.Array:
    """Place ``{"leaf": value}`` with ``runtime_device_put_tree``; return the leaf."""
    leaf = dtypes.runtime_device_put_tree({"leaf": value})["leaf"]
    assert isinstance(leaf, jax.Array), type(leaf)
    return leaf


def staged_device_array() -> None:
    """``staged_like`` moves a device array held elsewhere onto the reference's device."""
    jax.config.update("jax_enable_x64", True)
    first, second = jax.devices("cpu")[:2]
    reference = jax.device_put(np.zeros(2), first)
    tolerance = jax.device_put(np.asarray(1.0e-12), second)
    budget = jax.device_put(np.asarray(40, dtype=np.int32), second)
    with jax.transfer_guard("disallow"):
        staged_tolerance = staged_like(reference, tolerance)
        staged_budget = staged_like(reference, budget, dtype=np.int32)
        total = reference + staged_tolerance
    assert staged_tolerance.devices() == {first}, staged_tolerance.devices()
    assert staged_budget.devices() == {first}, staged_budget.devices()
    assert staged_budget.dtype == np.int32
    assert float(staged_tolerance) == 1.0e-12 and int(staged_budget) == 40
    assert total.devices() == {first}


def unplaced_joins_committed() -> None:
    """Unplaced values join data committed to another device; others are moved."""
    jax.config.update("jax_enable_x64", True)
    first, second = jax.devices("cpu")[:2]
    assert dtypes.get_runtime_jax_device() == first
    elsewhere = jax.device_put(np.arange(3.0), second)
    unplaced = dtypes.runtime_device_put(np.ones(3))
    as_float64 = dtypes.as_jax_float64(np.ones(3))
    assert (unplaced + elsewhere).devices() == {second}
    assert (as_float64 * elsewhere).devices() == {second}
    with jax.default_device(second):
        scoped = dtypes.runtime_device_put(np.ones(3))
    assert scoped.devices() == {second}, scoped.devices()
    moved = dtypes.runtime_device_put(elsewhere)
    assert moved.committed and moved.devices() == {first}, moved.devices()

    # An uncommitted array left on another device by an exited default_device
    # scope is not where the runtime puts values: it is moved there.
    with jax.default_device(second):
        left_behind = jnp.arange(3.0)
    assert not left_behind.committed and left_behind.devices() == {second}
    for placed in (
        dtypes.runtime_device_put(left_behind),
        _placed_leaf(left_behind),
    ):
        assert placed.devices() == {first}, placed.devices()
    # One already there stays uncommitted, like a host value.
    assert not _placed_leaf(jnp.zeros(3)).committed

    # A runtime device JAX would not choose is committed, host values included.
    with mock.patch.object(dtypes, "get_runtime_jax_device", lambda: second):
        for placed in (
            dtypes.runtime_device_put(np.ones(3)),
            _placed_leaf(np.ones(3)),
        ):
            assert placed.committed and placed.devices() == {second}, placed.devices()


def cuda_two_backend() -> None:
    """A GPU policy and a jax-cpu policy place values in one CUDA process."""
    jax.config.update("jax_enable_x64", True)
    gpu = jax.devices("gpu")[0]
    cpu = jax.devices("cpu")[0]
    assert dtypes.get_runtime_jax_device() == gpu
    with jax.default_device(cpu):
        left_behind = jnp.arange(3.0)
        scoped = dtypes.runtime_device_put(np.ones(3))
    assert not left_behind.committed and left_behind.devices() == {cpu}
    assert not scoped.committed and scoped.devices() == {cpu}
    for placed in (
        dtypes.runtime_device_put(left_behind),
        _placed_leaf(left_behind),
    ):
        assert placed.devices() == {gpu}, placed.devices()

    invalidate_backend_cache()
    set_backend("jax_cpu_parity", configure_runtime=False)
    assert dtypes.get_runtime_jax_device() == cpu
    for placed in (
        dtypes.runtime_device_put(np.ones(3)),
        _placed_leaf(np.ones(3)),
    ):
        assert placed.committed and placed.devices() == {cpu}, placed.devices()


_SCENARIOS = {
    "staged_device_array": staged_device_array,
    "unplaced_joins_committed": unplaced_joins_committed,
    "cuda_two_backend": cuda_two_backend,
}


if __name__ == "__main__":
    _SCENARIOS[sys.argv[1]]()
