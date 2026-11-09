"""Runtime dtype and reference placement stay consistent under tracing and device scopes."""

from __future__ import annotations

from unittest_jax_support import (
    JaxTestCase,
    optional_jax_imports,
    require_cuda,
    run_fresh_process,
)

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax.backend import dtypes
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.backend.runtime import invalidate_backend_cache, set_backend
    from simsopt_jax.core import _device_scalars
    from simsopt_jax.core._device_scalars import staged_like
    from simsopt_jax.runtime.host_boundary import disallow_host_transfers


import os
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np


# Two host devices exist only if XLA is told so before it initializes, so the
# multi-device checks run in a child process.
_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")
_PLACEMENT_CHILD = _FRESH_PROCESS / "backend_dtypes_placement.py"


class TestBackendDtypesReferencePlacement(JaxTestCase):
    def test_reference_placement_short_circuits_on_tracer(self):
        """A tracer reference is never probed for its placement (the O(jaxpr) walk)."""
        captured: dict[str, object] = {}

        @jit
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
        arr = jax.device_put(np.zeros(3), device)  # noqa: TID251 test places on a chosen device
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
        array = jax.device_put(np.ones(3, dtype=np.float64), jax.local_devices()[0])  # noqa: TID251 test places on a chosen device
        value = {"device": array, "host": (np.asarray(2, dtype=np.int32), None)}

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
            placed["host"][0].dtype == np.int32,
            'placed["host"][0].dtype == np.int32',
        )
        self.assertTrue(placed["host"][1] is None, 'placed["host"][1] is None')
        np.testing.assert_array_equal(placed["host"][0], 2)

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

        @jit
        def add_staged_scalar(reference):
            return reference + staged_like(reference, 1.0)

        result = add_staged_scalar(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

        np.testing.assert_array_equal(np.asarray(result), np.asarray((2.0, 3.0)))

    def test_staged_like_tracer_preserves_explicit_integer_dtype(self):
        """Scalar staging under JIT preserves a requested int32 dtype."""

        @jit
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
        completed = run_fresh_process(_PLACEMENT_CHILD, "staged_device_array", env=environment, timeout=300)
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_reference_placement_handles_tracer_leaf_in_sequence(self):
        """The list/tuple branch skips a tracer leaf and returns None."""
        captured: dict[str, object] = {}

        @jit
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

    def test_importing_simsopt_jax_alone_computes_the_raw_kernels_in_float64(self):
        """Without ``simsopt.geo`` or ``set_backend``, the raw B kernel matches the
        analytic on-axis loop field to float64 precision, also with x64 disabled
        in the environment."""
        for x64_setting in (None, "false"):
            with self.subTest(JAX_ENABLE_X64=x64_setting):
                environment = dict(os.environ, JAX_PLATFORMS="cpu")
                environment.pop("JAX_ENABLE_X64", None)
                if x64_setting is not None:
                    environment["JAX_ENABLE_X64"] = x64_setting
                completed = run_fresh_process(
                    _FRESH_PROCESS / "raw_kernel_float64.py", env=environment, timeout=300
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_set_backend_reenables_float64_disabled_after_import(self):
        """With x64 disabled after ``simsopt_jax`` was imported, selecting either
        CPU mode runs the raw B kernel in float64 again (analytic on-axis loop
        field to 1e-13 relative)."""
        completed = run_fresh_process(
            _FRESH_PROCESS / "set_backend_float64.py",
            env=dict(os.environ, JAX_PLATFORMS="cpu"),
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_explicit_device_array_rejects_reduced_precision_floats(self):
        """An exact-dtype placement accepts float64 as its only floating dtype."""
        patches = self.patches
        runtime_device = jax.devices()[0]
        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: runtime_device)
        )
        invalidate_backend_cache()
        set_backend("jax_cpu_parity", configure_runtime=False)

        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(
                    TypeError, r"Floating dtype must be one of \('float64',\)"
                ):
                    dtypes.explicit_device_array([1.0, 2.0], dtype=dtype)
        array = dtypes.explicit_device_array([1.0, 2.0], dtype=jnp.float64)
        self.assertEqual(array.dtype, jnp.float64)

    def test_explicit_device_array_preserves_single_device_reference(self):
        """Concrete single-device placement must not fall back to the runtime device.

        The reference's device is used verbatim; the runtime device is never
        consulted. The placement is the bare device rather than the reference's
        placement object: the two are identical for an eager put, but the object
        pins a put staged inside ``jit`` to one device (see
        ``dtypes._array_device``).
        """
        patches = self.patches
        reference = jax.device_put(np.zeros(3), jax.local_devices()[0])  # noqa: TID251 test places on a chosen device
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
            dtype=jnp.float64,
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
        as_float64 = dtypes.as_jax_float64(np.ones(3))
        with_uncommitted_explicit_reference = dtypes.explicit_device_array(
            np.ones(3), dtype=jnp.float64, reference=jnp.zeros(3)
        )
        with_committed_reference = dtypes.explicit_device_array(
            np.ones(3),
            dtype=jnp.float64,
            reference=jax.device_put(np.zeros(3), default_device),  # noqa: TID251 test places on a chosen device
        )

        for array in (
            unplaced,
            unplaced_tree,
            as_float64,
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
        completed = run_fresh_process(_PLACEMENT_CHILD, "unplaced_joins_committed", env=environment, timeout=300)
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_unplaced_values_follow_the_runtime_policy_in_a_cuda_process(self):
        """Real two-backend placement: a GPU policy and a jax-cpu policy in one CUDA process.

        An uncommitted CPU array left by a CPU ``jax.default_device`` scope
        goes to the GPU runtime device after the scope; under a jax-cpu policy,
        which JAX would not choose in a CUDA process, host values are committed
        to the CPU.
        """
        require_cuda("CUDA device required for the two-backend placement check")
        environment = dict(os.environ)
        environment.update({"JAX_PLATFORMS": "cuda,cpu", "JAX_ENABLE_X64": "1"})
        completed = run_fresh_process(_PLACEMENT_CHILD, "cuda_two_backend", env=environment, timeout=300)
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_explicit_cross_backend_moves_pass_a_disallow_guard(self):
        """Under ``disallow_host_transfers``, explicit moves of a GPU array and a pytree to the CPU and back keep their values."""
        require_cuda("CUDA device required for cross-backend placement")
        gpu = jax.devices("gpu")[0]
        cpu = jax.devices("cpu")[0]
        array = np.arange(5.0)
        pair = np.ones((2, 3))
        on_gpu = jax.device_put(array, gpu)  # noqa: TID251 test places on a chosen device
        tree = {"array": on_gpu, "pair": (jax.device_put(pair, gpu), np.arange(3))}  # noqa: TID251 test places on a chosen device
        with disallow_host_transfers():
            array_on_cpu = dtypes.runtime_device_put_tree(on_gpu, device=cpu)
            tree_on_cpu = dtypes.runtime_device_put_tree(tree, device=cpu)
            array_back = dtypes.runtime_device_put_tree(array_on_cpu, device=gpu)
            tree_back = dtypes.runtime_device_put_tree(tree_on_cpu, device=gpu)
        for placed, device in [
            (array_on_cpu, cpu),
            (tree_on_cpu, cpu),
            (array_back, gpu),
            (tree_back, gpu),
        ]:
            for leaf in jax.tree.leaves(placed):
                self.assertEqual(leaf.devices(), {device})
                self.assertTrue(leaf.committed)
        for placed in (array_on_cpu, array_back):
            np.testing.assert_array_equal(np.asarray(placed), array)
        for placed in (tree_on_cpu, tree_back):
            np.testing.assert_array_equal(np.asarray(placed["array"]), array)
            np.testing.assert_array_equal(np.asarray(placed["pair"][0]), pair)
            np.testing.assert_array_equal(np.asarray(placed["pair"][1]), np.arange(3))

    def test_runtime_placement_moves_a_gpu_array_to_a_cpu_policy_under_a_disallow_guard(
        self,
    ):
        """A jax-cpu policy in a CUDA process moves a committed GPU array to the CPU under a disallow guard."""
        require_cuda("CUDA device required for cross-backend placement")
        gpu = jax.devices("gpu")[0]
        cpu = jax.devices("cpu")[0]
        values = np.arange(4.0)
        on_gpu = jax.device_put(values, gpu)  # noqa: TID251 test places on a chosen device
        set_backend("jax_cpu_parity", configure_runtime=False)
        with disallow_host_transfers():
            placed = dtypes.runtime_device_put_tree({"x": on_gpu})["x"]
        self.assertEqual(placed.devices(), {cpu})
        np.testing.assert_array_equal(np.asarray(placed), values)

    def test_sharded_cross_backend_moves_pass_a_disallow_guard(self):
        """A GPU array moves to a two-CPU sharding and back under a disallow guard, values unchanged."""
        require_cuda("CUDA device required for cross-backend placement")
        environment = dict(os.environ)
        environment.update(
            {
                "JAX_PLATFORMS": "cuda,cpu",
                "JAX_ENABLE_X64": "1",
                "XLA_FLAGS": " ".join(
                    flag
                    for flag in (
                        environment.get("XLA_FLAGS"),
                        "--xla_force_host_platform_device_count=2",
                    )
                    if flag
                ),
            }
        )
        completed = run_fresh_process(_PLACEMENT_CHILD, "cuda_sharded_cross_backend", env=environment, timeout=300)
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
