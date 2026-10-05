"""Runtime dtype and reference placement stay consistent under tracing and device scopes."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import os
import subprocess
import sys
from typing import cast
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from simsopt_jax.backend import dtypes
from simsopt_jax.backend.runtime import invalidate_backend_cache, set_backend
from simsopt_jax.core import _device_scalars
from simsopt_jax.core._device_scalars import staged_like


def test_reference_sharding_short_circuits_on_tracer():
    """On a tracer the sharding-compat path is never reached (the O(jaxpr) walk)."""
    captured: dict[str, object] = {}

    @jax.jit
    def f(x):
        with mock.patch.object(dtypes, "_compatible_reference_sharding") as compat:
            captured["result"] = dtypes._reference_sharding(x, ndim=1)
            captured["compat_calls"] = compat.call_count
        return x

    f(jnp.zeros(3))

    # Old behavior: probed tracer.sharding (-> None after the jaxpr walk) then
    # called _compatible_reference_sharding(None, ...). The guard returns None
    # first, so the compat path is never invoked for a tracer.
    assert captured["result"] is None
    assert captured["compat_calls"] == 0


def test_reference_sharding_still_probes_concrete_array():
    """A concrete (non-traced) committed array is unaffected: it is still probed."""
    arr = jax.device_put(np.zeros(3), jax.local_devices()[0])
    with mock.patch.object(
        dtypes,
        "_compatible_reference_sharding",
        wraps=dtypes._compatible_reference_sharding,
    ) as compat:
        result = dtypes._reference_sharding(arr, ndim=1)

    # The concrete array goes through the probe; a single-device sharding is not
    # a NamedSharding, so the compatible result is None -- but the path runs.
    assert compat.call_count == 1
    assert result is None


def test_runtime_device_put_tree_can_preserve_arrays_and_place_host_leaves():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("device",))
    sharding = NamedSharding(mesh, P("device"))
    array = jax.device_put(np.ones(3, dtype=np.float64), sharding)
    value = {"device": array, "host": (np.asarray(2.0, dtype=np.float32), None)}

    placed = dtypes.runtime_device_put_tree(value, preserve_placement=True)

    assert placed["device"] is array
    assert placed["device"].sharding == sharding
    assert isinstance(placed["host"], tuple)
    assert isinstance(placed["host"][0], jax.Array)
    assert placed["host"][0].dtype == np.float32
    assert placed["host"][1] is None
    np.testing.assert_array_equal(placed["host"][0], 2.0)


def test_staged_scalar_uses_replicated_named_sharding() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("device",))
    vector_sharding = NamedSharding(mesh, P("device"))
    reference = jax.device_put(
        np.ones(3, dtype=np.float64),
        vector_sharding,
    )

    with jax.transfer_guard("disallow"):
        scalar = staged_like(reference, 1.0)

    assert scalar.ndim == 0
    assert isinstance(scalar.sharding, NamedSharding)
    assert scalar.sharding.spec == P()


def test_staged_like_tracer_does_not_embed_a_runtime_device_put(monkeypatch):
    def unexpected_explicit_placement(*args, **kwargs):
        raise AssertionError("traced literals must remain uncommitted")

    monkeypatch.setattr(
        _device_scalars,
        "explicit_device_array",
        unexpected_explicit_placement,
    )

    @jax.jit
    def add_staged_scalar(reference):
        return reference + staged_like(reference, 1.0)

    result = add_staged_scalar(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

    np.testing.assert_array_equal(np.asarray(result), np.asarray((2.0, 3.0)))


def test_staged_like_tracer_preserves_explicit_integer_dtype():
    @jax.jit
    def staged_integer(reference):
        return staged_like(reference, 1, dtype=jnp.int32)

    result = staged_integer(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

    assert result.dtype == jnp.int32
    assert int(np.asarray(result)) == 1


# Two host devices exist only if XLA is told so before it initializes, so the
# check runs in a child process.
_STAGED_DEVICE_ARRAY_CHILD = """
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from simsopt_jax.core._device_scalars import staged_like

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
"""


def test_staged_like_places_a_device_array_held_elsewhere_with_the_reference():
    """A concrete device array is moved onto the reference's device, explicitly.

    Keeping it where it was (``jnp.asarray``) left, e.g., a solver's tolerance
    on another device than the state it is compared with, so the program that
    combines them was rejected.
    """
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        }
    )
    completed = subprocess.run(
        (sys.executable, "-c", _STAGED_DEVICE_ARRAY_CHILD),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_reference_sharding_handles_tracer_leaf_in_sequence():
    """The list/tuple branch returns None for a tracer leaf and does not crash.

    This is a correctness smoke for the leaf-skip edit, not the regression guard
    (the old list branch also fell through to None for a tracer leaf via
    ``getattr(leaf, "sharding", None)`` -> None); the O(jaxpr) cost the fix
    removes is pinned for the scalar case by the first test.
    """
    captured: dict[str, object] = {}

    @jax.jit
    def f(x):
        with mock.patch.object(dtypes, "_compatible_reference_sharding") as compat:
            captured["result"] = dtypes._reference_sharding([x], ndim=1)
            captured["compat_calls"] = compat.call_count
        return x

    f(jnp.zeros(3))

    assert captured["result"] is None
    assert captured["compat_calls"] == 0


def test_runtime_device_put_uses_runtime_device_when_no_target(monkeypatch):
    """Implicit placement follows the runtime policy device, not JAX defaults."""
    runtime_device = object()
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: runtime_device)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.runtime_device_put([1, 2, 3])

    assert isinstance(array, np.ndarray)
    assert placement is runtime_device
    assert placements == [runtime_device]


def test_runtime_device_put_preserves_explicit_target(monkeypatch):
    """Explicit target/sharding placement still takes precedence."""
    explicit_target = object()
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    def _unexpected_runtime_device():
        raise AssertionError("explicit placement must not query runtime device")

    monkeypatch.setattr(dtypes, "get_runtime_jax_device", _unexpected_runtime_device)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.runtime_device_put([1, 2, 3], target=explicit_target)

    assert isinstance(array, np.ndarray)
    assert placement is explicit_target
    assert placements == [explicit_target]


def test_runtime_device_put_keeps_default_placement_without_runtime_device(monkeypatch):
    """Non-JAX policy remains on the unqualified JAX placement path."""
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: None)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.runtime_device_put([1, 2, 3])

    assert isinstance(array, np.ndarray)
    assert placement is None
    assert placements == [None]


def test_explicit_device_array_preserves_requested_float_dtype(monkeypatch):
    """Explicit FP32 placement must not be rewritten by runtime FP64 policy."""
    runtime_device = jax.devices()[0]
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: runtime_device)
    invalidate_backend_cache()
    set_backend("jax_cpu_parity", configure_runtime=False)

    array = dtypes.explicit_device_array([1.0, 2.0], dtype=jnp.float32)

    assert array.dtype == jnp.float32


def test_explicit_device_array_preserves_single_device_reference(monkeypatch):
    """Concrete single-device placement must not fall back to the runtime device.

    The reference's device is used verbatim; the runtime device is never
    consulted. The placement is the bare device rather than the reference's
    concrete ``SingleDeviceSharding``: the two are identical for an eager put,
    but the sharding form pins a put staged inside ``jit`` to one device (see
    ``dtypes._single_device_placement``).
    """
    reference = jax.device_put(np.zeros(3), jax.local_devices()[0])
    (reference_device,) = reference.sharding.device_set
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    def _unexpected_runtime_device():
        raise AssertionError("reference placement must not query runtime device")

    monkeypatch.setattr(dtypes, "get_runtime_jax_device", _unexpected_runtime_device)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.explicit_device_array(
        [1.0, 2.0],
        dtype=jnp.float32,
        reference=reference,
    )

    assert isinstance(array, np.ndarray)
    assert placement is reference_device
    assert placements == [reference_device]


def test_unplaced_values_stay_uncommitted_like_jax_leaves_them(monkeypatch):
    """A value no caller placed is uncommitted on the runtime device, as JAX leaves it.

    Committing it (the old rule) claimed a placement no caller made, so it
    refused every computation with data committed elsewhere. A value placed
    with an uncommitted reference is unplaced too; a committed reference's
    placement is still used.
    """
    default_device = jax.local_devices()[0]
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: default_device)

    unplaced = dtypes.runtime_device_put(np.ones(3))
    unplaced_tree = dtypes.runtime_device_put_tree({"a": np.ones(3)})["a"]
    with_uncommitted_reference = dtypes.as_runtime_float64(
        np.ones(3), reference=jnp.zeros(3)
    )
    with_uncommitted_explicit_reference = dtypes.explicit_device_array(
        np.ones(3), dtype=jnp.float64, reference=jnp.zeros(3)
    )
    with_committed_reference = dtypes.explicit_device_array(
        np.ones(3),
        dtype=jnp.float64,
        reference=jax.device_put(np.zeros(3), default_device),
    )

    for array in (
        unplaced,
        unplaced_tree,
        with_uncommitted_reference,
        with_uncommitted_explicit_reference,
    ):
        assert not cast(jax.Array, array).committed
        assert cast(jax.Array, array).devices() == {default_device}
    assert with_committed_reference.committed
    assert with_committed_reference.devices() == {default_device}


_UNPLACED_JOINS_COMMITTED_CHILD = """
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from simsopt_jax.backend import dtypes

first, second = jax.devices("cpu")[:2]
assert dtypes.get_runtime_jax_device() == first
elsewhere = jax.device_put(np.arange(3.0), second)
unplaced = dtypes.runtime_device_put(np.ones(3))
with_uncommitted_reference = dtypes.as_runtime_float64(
    np.ones(3), reference=jnp.zeros(3)
)
assert (unplaced + elsewhere).devices() == {second}
assert (with_uncommitted_reference * elsewhere).devices() == {second}
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
    dtypes.runtime_device_put_tree({"leaf": left_behind})["leaf"],
):
    assert placed.devices() == {first}, placed.devices()
# One already there stays uncommitted, like a host value.
assert not dtypes.runtime_device_put_tree({"leaf": jnp.zeros(3)})["leaf"].committed

# A runtime device JAX would not choose is committed, host values included.
dtypes.get_runtime_jax_device = lambda: second
for placed in (
    dtypes.runtime_device_put(np.ones(3)),
    dtypes.runtime_device_put_tree({"leaf": np.ones(3)})["leaf"],
):
    assert placed.committed and placed.devices() == {second}, placed.devices()
"""


def test_unplaced_values_join_data_committed_to_another_device():
    """A constant nobody placed joins data committed to another device.

    The committed-to-the-runtime-device rule refused this combination: a
    An array built under ``with_cpu_device_for_construction`` (or on
    an active mesh) met constants committed to the runtime device. A
    ``jax.default_device`` scope is honoured as well; an array committed
    elsewhere, or left uncommitted elsewhere by an exited scope, is moved
    onto the runtime device; and a runtime device JAX would not choose is
    committed.
    """
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        }
    )
    completed = subprocess.run(
        (sys.executable, "-c", _UNPLACED_JOINS_COMMITTED_CHILD),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


_CUDA_TWO_BACKEND_CHILD = """
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from simsopt_jax.backend import dtypes
from simsopt_jax.backend.runtime import (
    invalidate_backend_cache,
    set_backend,
    with_cpu_device_for_construction,
)
from simsopt_jax.backend.dtypes import runtime_device_put_tree as _place_runtime_tree

gpu = jax.devices("gpu")[0]
cpu = jax.devices("cpu")[0]
assert dtypes.get_runtime_jax_device() == gpu
with with_cpu_device_for_construction():
    left_behind = jnp.arange(3.0)
    scoped = dtypes.runtime_device_put(np.ones(3))
assert not left_behind.committed and left_behind.devices() == {cpu}
assert not scoped.committed and scoped.devices() == {cpu}
for placed in (
    dtypes.runtime_device_put(left_behind),
    dtypes.runtime_device_put_tree({"leaf": left_behind})["leaf"],
    _place_runtime_tree({"leaf": left_behind})["leaf"],
):
    assert placed.devices() == {gpu}, placed.devices()

invalidate_backend_cache()
set_backend("jax_cpu_parity", configure_runtime=False)
assert dtypes.get_runtime_jax_device() == cpu
for placed in (
    dtypes.runtime_device_put(np.ones(3)),
    dtypes.runtime_device_put_tree({"leaf": np.ones(3)})["leaf"],
):
    assert placed.committed and placed.devices() == {cpu}, placed.devices()
"""


def test_unplaced_values_follow_the_runtime_policy_in_a_cuda_process():
    """Real two-backend placement: a GPU policy and a jax-cpu policy in one CUDA process.

    An uncommitted CPU array left by ``with_cpu_device_for_construction``
    goes to the GPU runtime device after the scope (also through the runtime
    adapter's ``_place_runtime_tree``); under a jax-cpu policy, which JAX
    would not choose in a CUDA process, host values are committed to the CPU.
    """
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("CUDA device required for the two-backend placement check")
    environment = dict(os.environ)
    environment.update({"JAX_PLATFORMS": "cuda,cpu", "JAX_ENABLE_X64": "1"})
    completed = subprocess.run(
        (sys.executable, "-c", _CUDA_TWO_BACKEND_CHILD),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_commit_in_place_commits_where_the_array_lives():
    """An uncommitted array is committed on its own device; a mesh placement is kept."""
    uncommitted = jnp.arange(3.0)
    mesh_sharding = NamedSharding(
        Mesh(np.asarray(jax.devices()[:1], dtype=object), ("device",)), P("device")
    )
    on_mesh = jax.device_put(np.arange(3.0), mesh_sharding)

    committed = dtypes.commit_in_place(uncommitted)
    kept = dtypes.commit_in_place(on_mesh)

    assert committed.committed
    assert committed.devices() == uncommitted.devices()
    assert committed.dtype == uncommitted.dtype
    assert kept.sharding == mesh_sharding
    np.testing.assert_array_equal(np.asarray(committed), np.arange(3.0))
