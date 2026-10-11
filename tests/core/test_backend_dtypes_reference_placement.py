"""Runtime dtype and reference placement stay consistent under tracing and device scopes."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax.backend import dtypes
    from simsopt_jax.backend.runtime import invalidate_backend_cache, set_backend
    from simsopt_jax.core import _device_scalars
    from simsopt_jax.core._device_scalars import staged_like


import os
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np


# Two host devices exist only if XLA is told so before it initializes, so the
# multi-device checks run in a child process.
_PLACEMENT_CHILD = (
    Path(__file__).resolve().with_name("fresh_process") / "backend_dtypes_placement.py"
)


def _run_placement_child(scenario, environment):
    """Run one ``fresh_process/backend_dtypes_placement.py`` scenario."""
    return subprocess.run(
        (sys.executable, str(_PLACEMENT_CHILD), scenario),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestBackendDtypesReferencePlacement(JaxTestCase):
    def test_reference_placement_short_circuits_on_tracer(self):
        """A tracer reference is never probed for its placement (the O(jaxpr) walk)."""
        captured: dict[str, object] = {}

        @jax.jit
        def f(x):
            with mock.patch.object(dtypes, "_committed_placement") as committed:
                captured["result"] = dtypes._reference_placement(x)
                captured["probe_calls"] = committed.call_count
            return x

        f(jnp.zeros(3))

        self.assertTrue(captured["result"] is None, 'captured["result"] is None')
        self.assertTrue(captured["probe_calls"] == 0, 'captured["probe_calls"] == 0')

    def test_reference_placement_is_the_bare_device_of_a_committed_array(self):
        """A committed concrete reference is probed and reduced to its device."""
        device = jax.local_devices()[0]
        arr = jax.device_put(np.zeros(3), device)
        with mock.patch.object(
            dtypes,
            "_committed_placement",
            wraps=dtypes._committed_placement,
        ) as committed:
            result = dtypes._reference_placement(arr)

        self.assertTrue(committed.call_count == 1, "committed.call_count == 1")
        self.assertTrue(result is device, "result is device")
        self.assertTrue(
            dtypes._reference_placement(jnp.zeros(3)) is None,
            "dtypes._reference_placement(jnp.zeros(3)) is None",
        )

    def test_runtime_device_put_tree_can_preserve_arrays_and_place_host_leaves(self):
        """Preserving placement retains JAX leaf identity and places host leaves without
        changing structure or dtype."""
        array = jax.device_put(np.ones(3, dtype=np.float64), jax.local_devices()[0])
        value = {"device": array, "host": (np.asarray(2.0, dtype=np.float32), None)}

        placed = dtypes.runtime_device_put_tree(value, preserve_placement=True)

        self.assertTrue(placed["device"] is array, 'placed["device"] is array')
        self.assertTrue(
            isinstance(placed["host"], tuple), 'isinstance(placed["host"], tuple)'
        )
        self.assertTrue(
            isinstance(placed["host"][0], jax.Array),
            'isinstance(placed["host"][0], jax.Array)',
        )
        self.assertTrue(
            placed["host"][0].dtype == np.float32,
            'placed["host"][0].dtype == np.float32',
        )
        self.assertTrue(placed["host"][1] is None, 'placed["host"][1] is None')
        np.testing.assert_array_equal(placed["host"][0], 2.0)

    def test_staged_like_tracer_does_not_embed_a_runtime_device_put(self):
        """Scalar staging under JIT computes the expected sum without explicit device
        placement."""
        patches = self.patches

        def unexpected_explicit_placement(*args, **kwargs):
            raise AssertionError("traced literals must remain uncommitted")

        patches.enter_context(
            mock.patch.object(
                _device_scalars, "explicit_device_array", unexpected_explicit_placement
            )
        )

        @jax.jit
        def add_staged_scalar(reference):
            return reference + staged_like(reference, 1.0)

        result = add_staged_scalar(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

        np.testing.assert_array_equal(np.asarray(result), np.asarray((2.0, 3.0)))

    def test_staged_like_tracer_preserves_explicit_integer_dtype(self):
        """Scalar staging under JIT preserves a requested int32 dtype."""

        @jax.jit
        def staged_integer(reference):
            return staged_like(reference, 1, dtype=jnp.int32)

        result = staged_integer(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

        self.assertTrue(result.dtype == jnp.int32, "result.dtype == jnp.int32")
        self.assertTrue(int(np.asarray(result)) == 1, "int(np.asarray(result)) == 1")

    def test_staged_like_places_a_device_array_held_elsewhere_with_the_reference(self):
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
        completed = _run_placement_child("staged_device_array", environment)
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_reference_placement_handles_tracer_leaf_in_sequence(self):
        """The list/tuple branch skips a tracer leaf and returns None."""
        captured: dict[str, object] = {}

        @jax.jit
        def f(x):
            with mock.patch.object(dtypes, "_committed_placement") as committed:
                captured["result"] = dtypes._reference_placement([x])
                captured["probe_calls"] = committed.call_count
            return x

        f(jnp.zeros(3))

        self.assertTrue(captured["result"] is None, 'captured["result"] is None')
        self.assertTrue(captured["probe_calls"] == 0, 'captured["probe_calls"] == 0')

    def test_runtime_device_put_uses_runtime_device_when_no_target(self):
        """Implicit placement follows the runtime policy device, not JAX defaults."""
        patches = self.patches
        runtime_device = object()
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: runtime_device)
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.runtime_device_put([1, 2, 3])

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is runtime_device, "placement is runtime_device")
        self.assertTrue(
            placements == [runtime_device], "placements == [runtime_device]"
        )

    def test_runtime_device_put_preserves_explicit_target(self):
        """Explicit target placement still takes precedence."""
        patches = self.patches
        explicit_target = object()
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        def _unexpected_runtime_device():
            raise AssertionError("explicit placement must not query runtime device")

        patches.enter_context(
            mock.patch.object(
                dtypes, "get_runtime_jax_device", _unexpected_runtime_device
            )
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.runtime_device_put([1, 2, 3], target=explicit_target)

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is explicit_target, "placement is explicit_target")
        self.assertTrue(
            placements == [explicit_target], "placements == [explicit_target]"
        )

    def test_runtime_device_put_keeps_default_placement_without_runtime_device(self):
        """Non-JAX policy remains on the unqualified JAX placement path."""
        patches = self.patches
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: None)
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.runtime_device_put([1, 2, 3])

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is None, "placement is None")
        self.assertTrue(placements == [None], "placements == [None]")

    def test_explicit_device_array_preserves_requested_float_dtype(self):
        """Explicit FP32 placement must not be rewritten by runtime FP64 policy."""
        patches = self.patches
        runtime_device = jax.devices()[0]
        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: runtime_device)
        )
        invalidate_backend_cache()
        set_backend("jax_cpu_parity", configure_runtime=False)

        array = dtypes.explicit_device_array([1.0, 2.0], dtype=jnp.float32)

        self.assertTrue(array.dtype == jnp.float32, "array.dtype == jnp.float32")

    def test_explicit_device_array_preserves_single_device_reference(self):
        """Concrete single-device placement must not fall back to the runtime device.

        The reference's device is used verbatim; the runtime device is never
        consulted. The placement is the bare device rather than the reference's
        placement object: the two are identical for an eager put, but the object
        pins a put staged inside ``jit`` to one device (see
        ``dtypes._array_device``).
        """
        patches = self.patches
        reference = jax.device_put(np.zeros(3), jax.local_devices()[0])
        (reference_device,) = reference.devices()
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        def _unexpected_runtime_device():
            raise AssertionError("reference placement must not query runtime device")

        patches.enter_context(
            mock.patch.object(
                dtypes, "get_runtime_jax_device", _unexpected_runtime_device
            )
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.explicit_device_array(
            [1.0, 2.0],
            dtype=jnp.float32,
            reference=reference,
        )

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is reference_device, "placement is reference_device")
        self.assertTrue(
            placements == [reference_device], "placements == [reference_device]"
        )

    def test_unplaced_values_stay_uncommitted_like_jax_leaves_them(self):
        """A value no caller placed is uncommitted on the runtime device, as JAX leaves it.

        Committing it (the old rule) claimed a placement no caller made, so it
        refused every computation with data committed elsewhere. A value placed
        with an uncommitted reference is unplaced too; a committed reference's
        device is still used.
        """
        patches = self.patches
        default_device = jax.local_devices()[0]
        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: default_device)
        )

        unplaced = dtypes.runtime_device_put(np.ones(3))
        unplaced_tree = dtypes.runtime_device_put_tree({"a": np.ones(3)})["a"]
        as_runtime = dtypes.as_runtime_array(np.ones(3))
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
            as_runtime,
            with_uncommitted_explicit_reference,
        ):
            self.assertTrue(
                not cast(jax.Array, array).committed,
                "not cast(jax.Array, array).committed",
            )
            self.assertTrue(
                cast(jax.Array, array).devices() == {default_device},
                "cast(jax.Array, array).devices() == {default_device}",
            )
        self.assertTrue(
            with_committed_reference.committed, "with_committed_reference.committed"
        )
        self.assertTrue(
            with_committed_reference.devices() == {default_device},
            "with_committed_reference.devices() == {default_device}",
        )

    def test_unplaced_values_join_data_committed_to_another_device(self):
        """A constant nobody placed joins data committed to another device.

        The committed-to-the-runtime-device rule refused this combination: an
        array built under a CPU ``jax.default_device`` scope met constants
        committed to the runtime device. A
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
        completed = _run_placement_child("unplaced_joins_committed", environment)
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_unplaced_values_follow_the_runtime_policy_in_a_cuda_process(self):
        """Real two-backend placement: a GPU policy and a jax-cpu policy in one CUDA process.

        An uncommitted CPU array left by a CPU ``jax.default_device`` scope
        goes to the GPU runtime device after the scope; under a jax-cpu policy,
        which JAX would not choose in a CUDA process, host values are committed
        to the CPU.
        """
        if not any(device.platform == "gpu" for device in jax.devices()):
            self.skipTest("CUDA device required for the two-backend placement check")
        environment = dict(os.environ)
        environment.update({"JAX_PLATFORMS": "cuda,cpu", "JAX_ENABLE_X64": "1"})
        completed = _run_placement_child("cuda_two_backend", environment)
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_commit_in_place_commits_where_the_array_lives(self):
        """An uncommitted array is committed on its own device with its own dtype."""
        uncommitted = jnp.arange(3.0)

        committed = dtypes.commit_in_place(uncommitted)

        self.assertTrue(not uncommitted.committed, "not uncommitted.committed")
        self.assertTrue(committed.committed, "committed.committed")
        self.assertTrue(
            committed.devices() == uncommitted.devices(),
            "committed.devices() == uncommitted.devices()",
        )
        self.assertTrue(
            committed.dtype == uncommitted.dtype, "committed.dtype == uncommitted.dtype"
        )
        np.testing.assert_array_equal(np.asarray(committed), np.arange(3.0))
