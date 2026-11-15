"""Who owns the host buffers JAX reads, and where and in which dtype JAX values land."""

from __future__ import annotations

import os
from functools import partial
from typing import cast
from unittest import mock

import numpy as np
from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, aligned_host_copy
from unittest_jax_support import optional_jax_imports, require_cuda, run_fresh_process
from unittest_jax_support import assert_tree_equal, distinct_host_array, host_snapshot, make_execution_gate
from unittest_jax_support import disallow_host_transfers, host_tree_after_ready

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend import dtypes, set_backend
    from simsopt_jax.backend import jit as jit_module
    from simsopt_jax.backend import runtime
    from simsopt_jax.backend.dtypes import as_jax_array, as_jax_float64
    from simsopt_jax.backend.dtypes import explicit_device_array, runtime_device_put
    from simsopt_jax.backend.dtypes import runtime_device_put_tree
    from simsopt_jax.backend.jit import CopyingJit, jit
    from simsopt_jax.backend.runtime import host_value, snapshot_host_tree
    from simsopt_jax.core import _device_scalars, specs
    from simsopt_jax.core import biotsavart as kernels
    from simsopt_jax.core._device_scalars import staged_like
    from simsopt_jax.core.field import coil_set_spec_from_dof_extraction_spec
    from simsopt_jax.core.field import group_biot_savart_B_vjp
    from simsopt_jax.core.specs import make_coil_group_spec
    from simsopt_jax.core.specs import make_grouped_coil_set_spec
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.field import biotsavart_backend as adapter
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

    # Native simsopt.geo imports jax too.
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.geo.curvehelical import CurveHelical
    from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries


class _Table:
    """Pytree holding a NumPy array; hashes by identity, so it can be static."""

    def __init__(self, values):
        self.values = values

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


if JAX_IMPORT_ERROR is None:
    jax.tree_util.register_pytree_node_class(_Table)


def _table_probe(kind: str):
    """``x * sum(table.values)`` with the parameters ``kind`` names; tables it got."""
    received = []

    def scaled(x, table):
        received.append(table)
        return x * jnp.sum(jnp.asarray(table.values))

    probes = {
        "positional-only x": lambda x, /, table: scaled(x, table),
        "keyword-only table": lambda x, *, table: scaled(x, table),
        "positional-or-keyword": lambda x, table: scaled(x, table),
        "defaulted last": lambda x, table, label="": scaled(x, table),
        "*args": lambda x, *tables: scaled(x, tables[-1]),
    }
    return received, probes[kind]


def _compile(jit_function, fun, selector: dict):
    """``jit_function(fun, **selector)``; a tuple selector goes as a fresh iterator."""
    options = {k: iter(v) if isinstance(v, tuple) else v for k, v in selector.items()}
    return jit_function(fun, **options)


def _recording_copies(copies: list):
    """Patch ``snapshot_host_tree`` to append each ``(value, copy)`` to ``copies``."""

    def record(value):
        copies.append((value, snapshot_host_tree(value)))
        return copies[-1][1]

    return mock.patch.object(jit_module, "snapshot_host_tree", record)


class TestCopyingJit(JaxTestCase):
    """``jit`` copies dynamic NumPy leaves and otherwise behaves as ``jax.jit``."""

    def test_static_selectors_match_raw_jit(self):
        """Every selector makes static what ``jax.jit`` does, and copies the rest."""
        table = _Table(np.arange(3.0))
        P, K = (lambda f, x: f(x, table)), (lambda f, x: f(x, table=table))
        label = lambda f, x: f(x, table, label="static")
        nums, names, either = ("static_argnums", "static_argnames", "positional-or-keyword")
        # (name, probe kind, selector, call); -1 names ``label``, which a positional
        # call leaves out, so ``table`` is dynamic. A tuple selector is passed as a
        # fresh iterator on each compile.
        cases = (
            ("argnums, positional", either, {nums: 1}, P),
            ("negative argnums", either, {nums: -1}, P),
            ("negative argnums", "defaulted last", {nums: -1}, P),
            ("negative argnums", "*args", {nums: -1}, P),
            ("negative argnums, keyword", "defaulted last", {nums: -1}, label),
            ("argnums, keyword", either, {nums: 1}, K),
            ("argnames, positional", either, {names: "table"}, P),
            ("argnames, keyword", either, {names: "table"}, K),
            ("argnames", "positional-only x", {names: "table"}, P),
            ("argnums", "positional-only x", {nums: 1}, K),
            ("argnames", "keyword-only table", {names: "table"}, K),
            ("iterator argnums", either, {nums: (1,)}, K),
            ("iterator argnames", either, {names: ("table",)}, P),
        )
        for name, kind, selector, call in cases:
            with self.subTest(selector=name, parameters=kind):
                raw_received, raw_fun = _table_probe(kind)
                call(_compile(jax.jit, raw_fun, selector), np.ones(2))  # noqa: TID251
                received, fun = _table_probe(kind)
                f, x, copies = _compile(jit, fun, selector), np.ones(2), []
                with _recording_copies(copies):
                    for _ in range(3):
                        np.testing.assert_array_equal(call(f, x), np.full(2, 3.0))
                self.assertEqual(len(received), 1)
                if kind == either:  # every selector here makes ``table`` static
                    self.assertIs(received[0], table)
                self.assertEqual(received[0] is table, raw_received[0] is table)
                self.assertEqual(f._cache_size(), 1)
                copied = [value for value, _ in copies]
                self.assertTrue(any(value is x for value in copied), copied)
                copied_table = any(value is table for value in copied)
                self.assertEqual(copied_table, raw_received[0] is not table)

    def test_every_decorator_form_compiles_with_its_options(self):
        """``@jit``, ``partial(jit, ...)``, ``@jit(...)`` and ``jit(f, ...)`` agree."""

        def power(x, n):
            return x**n

        @jit
        def bare(x):
            return x**2

        forms = {
            "bare decorator": (bare, lambda f, x: f(x)),
            "partial": (partial(jit, static_argnames="n")(power), lambda f, x: f(x, 2)),
            "decorator factory": (jit(static_argnames="n")(power), lambda f, x: f(x, n=2)),
            "call": (jit(power, static_argnums=1), lambda f, x: f(x, 2)),
        }
        x = np.linspace(0.0, 1.0, 4)
        for name, (f, call) in forms.items():
            with self.subTest(form=name):
                self.assertIsInstance(f, CopyingJit)
                np.testing.assert_array_equal(call(f, x), x**2)
        self.assertEqual(bare.__name__, "bare")
        self.assertEqual(bare.lower(x).compile()(x).shape, x.shape)
        self.assertEqual(bare.eval_shape(x).shape, x.shape)
        self.assertEqual(bare.trace(x).jaxpr.in_avals[0].shape, x.shape)
        self.assertEqual(bare._cache_size(), 1)
        bare.clear_cache()
        self.assertEqual(bare._cache_size(), 0)
        np.testing.assert_allclose(jax.grad(lambda y: jnp.sum(bare(y)))(x), 2 * x)
        np.testing.assert_array_equal(jit(lambda y: bare(y) + 1.0)(x), x**2 + 1.0)

        class Scaler:
            factor = 3.0

            @partial(jit, static_argnums=0)
            def scale(self, y):
                return self.factor * y

        np.testing.assert_array_equal(Scaler().scale(x), 3.0 * x)

    def test_jax_array_inputs_are_not_copied(self):
        """Arguments without NumPy leaves skip the copy; NumPy leaves are copied."""
        copies = []
        add = jit(lambda a, tree: a + tree["b"][0] * tree["b"][1])
        values, host = jnp.arange(3.0), np.arange(3.0)
        with _recording_copies(copies):
            add(values, {"b": [values, values]})
            add(values, tree={"b": (values, 2.0)})
            self.assertEqual(copies, [])
            add(values, {"b": [host, values]})
        (_, first), (_, tree) = copies
        self.assertIs(first, values)
        self.assertIs(tree["b"][1], values)
        self.assertIsNot(tree["b"][0], host)
        np.testing.assert_array_equal(tree["b"][0], host)

    def test_placement_matches_raw_jit(self):
        """NumPy, uncommitted and committed inputs land where ``jax.jit`` puts them."""
        identity = jit(lambda x: x * 1.0)
        raw = jax.jit(lambda x: x * 1.0)  # noqa: TID251 reference placement of raw jit
        host, cpu = np.arange(4.0), jax.devices("cpu")[0]
        inputs = {
            "NumPy": host,
            "uncommitted": jnp.asarray(host),
            "committed CPU": jax.device_put(host, cpu),  # noqa: TID251 committed input
        }
        for name, value in inputs.items():
            with self.subTest(input=name):
                self.assertEqual(identity(value).devices(), raw(value).devices())

    def test_runtime_gpu_receives_numpy_and_uncommitted_inputs(self):
        """On the GPU runtime, NumPy and uncommitted inputs run there; committed stay."""
        require_cuda("CUDA device required for runtime GPU placement")
        set_backend("jax", device="gpu")
        gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
        identity, host = jit(lambda x: x * 1.0), np.arange(4.0)
        expected = {
            "NumPy": (host, gpu),
            "uncommitted": (jnp.asarray(host), gpu),
            "committed CPU": (jax.device_put(host, cpu), cpu),  # noqa: TID251 committed input
            "committed GPU": (jax.device_put(host, gpu), gpu),  # noqa: TID251 committed input
        }
        for name, (value, device) in expected.items():
            with self.subTest(input=name):
                self.assertEqual(identity(value).devices(), {device})


# Two host devices exist only if XLA is told so before it starts, so the multi-device
# checks run this code in a child interpreter (``python -c``) with the parent's
# ``JAX_PLATFORMS``, ``JAX_ENABLE_X64`` and ``XLA_FLAGS``; its argument names the scenario.
_PLACEMENT_CHILD = '''
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from simsopt_jax.backend import dtypes
from simsopt_jax.backend.runtime import set_backend
from simsopt_jax.core._device_scalars import staged_like

def _placed_leaf(value) -> jax.Array:
    """Place ``{"leaf": value}`` with ``runtime_device_put_tree``; return the leaf."""
    leaf = dtypes.runtime_device_put_tree({"leaf": value})["leaf"]
    assert isinstance(leaf, jax.Array), type(leaf)
    return leaf

def staged_device_array() -> None:
    """``staged_like`` moves a device array held elsewhere onto the reference's device
    (left there, a solver's tolerance broke the program comparing it)."""
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
    # An uncommitted array an exited default_device scope left elsewhere is moved.
    with jax.default_device(second):
        left_behind = jnp.arange(3.0)
    assert not left_behind.committed and left_behind.devices() == {second}
    for placed in (dtypes.runtime_device_put(left_behind), _placed_leaf(left_behind)):
        assert placed.devices() == {first}, placed.devices()
    # One already there stays uncommitted, like a host value.
    assert not _placed_leaf(jnp.zeros(3)).committed
    # A runtime device JAX would not choose is committed, host values included.
    with mock.patch.object(dtypes, "get_runtime_jax_device", lambda: second):
        for placed in (dtypes.runtime_device_put(np.ones(3)), _placed_leaf(np.ones(3))):
            assert placed.committed and placed.devices() == {second}, placed.devices()

def cuda_two_backend() -> None:
    """The GPU and the CPU backend place values in one CUDA process."""
    jax.config.update("jax_enable_x64", True)
    gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
    assert dtypes.get_runtime_jax_device() == gpu
    with jax.default_device(cpu):
        left_behind = jnp.arange(3.0)
        scoped = dtypes.runtime_device_put(np.ones(3))
    assert not left_behind.committed and left_behind.devices() == {cpu}
    assert not scoped.committed and scoped.devices() == {cpu}
    for placed in (dtypes.runtime_device_put(left_behind), _placed_leaf(left_behind)):
        assert placed.devices() == {gpu}, placed.devices()
    set_backend("jax", device="cpu")
    # JAX's default device changed after set_backend: JAX would not choose the CPU.
    jax.config.update("jax_default_device", None)
    assert dtypes.get_runtime_jax_device() == cpu
    for placed in (dtypes.runtime_device_put(np.ones(3)), _placed_leaf(np.ones(3))):
        assert placed.committed and placed.devices() == {cpu}, placed.devices()

def cuda_sharded_cross_backend() -> None:
    """Explicit moves between a GPU device and a two-CPU sharding pass a disallow guard."""
    jax.config.update("jax_enable_x64", True)
    gpu, cpus = jax.devices("gpu")[0], jax.devices("cpu")[:2]
    cpu_sharding = NamedSharding(Mesh(np.array(cpus), ("i",)), PartitionSpec("i"))
    values = np.arange(6.0)
    on_gpu = jax.device_put(values, gpu)
    with jax.transfer_guard("disallow"):
        sharded = dtypes.runtime_device_put_tree({"x": on_gpu}, target=cpu_sharding)
        returned = dtypes.runtime_device_put_tree(sharded, device=gpu)
    assert sharded["x"].sharding == cpu_sharding, sharded["x"].sharding
    assert returned["x"].devices() == {gpu}, returned["x"].devices()
    np.testing.assert_array_equal(np.asarray(sharded["x"]), values)
    np.testing.assert_array_equal(np.asarray(returned["x"]), values)

globals()[sys.argv[1]]()
'''

# Only the ``simsopt_jax`` import can enable x64 here: neither ``simsopt.geo`` nor
# ``set_backend`` runs in this child.
_RAW_KERNEL_FLOAT64_CHILD = """
import numpy as np
from simsopt_jax.core.biotsavart import biot_savart_B

radius, current, quadpoints = 1.0, 1.0e5, 64
phi = 2.0 * np.pi * np.arange(quadpoints) / quadpoints
gamma = radius * np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=-1)
# d gamma / d t for the normalized parameter t = phi / (2 pi).
gammadash = 2.0 * np.pi * radius * np.stack((-np.sin(phi), np.cos(phi), 0 * phi), axis=-1)
heights = np.array([0.0, 0.3, 1.7])
points = np.stack((np.zeros(3), np.zeros(3), heights), axis=-1)
B = biot_savart_B(points, gamma[None], gammadash[None], np.array([current]))
mu0 = 4.0e-7 * np.pi
B_z = mu0 * current * radius**2 / (2.0 * (radius**2 + heights**2) ** 1.5)
assert B.dtype == np.float64, B.dtype
# A float32 field misses this by about 1e-7 relative.
np.testing.assert_allclose(np.asarray(B)[:, 2], B_z, rtol=1e-13, atol=0.0)
np.testing.assert_allclose(np.asarray(B)[:, :2], 0.0, rtol=0.0, atol=1e-12 * B_z.max())
"""
_TWO_HOST_DEVICES = "--xla_force_host_platform_device_count=2"


def _recording_device_put(placements: list):
    """A ``jax.device_put`` stand-in returning ``(array, placement)`` and recording."""

    def device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    return device_put


def _must_not_run(message: str):
    def fail(*args, **kwargs):
        raise AssertionError(message)

    return fail


class TestDtypesPlacement(JaxTestCase):
    """Runtime dtype and reference placement stay consistent under tracing and scopes."""

    def patch_device_put(self, patches, runtime_device) -> list:
        """Patch the runtime device and ``jax.device_put``; return the placements."""
        placements = []
        put = _recording_device_put(placements)
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", put))
        get = runtime_device if callable(runtime_device) else lambda: runtime_device
        patches.enter_context(mock.patch.object(dtypes, "get_runtime_jax_device", get))
        return placements

    def test_reference_placement_short_circuits_on_tracer(self):
        """A tracer, alone or as a list leaf, is never probed (the O(jaxpr) walk)."""
        for name, wrap in (("tracer", lambda x: x), ("tracer in a list", lambda x: [x])):
            with self.subTest(name), self.case():
                captured: dict[str, object] = {}

                @jit
                def f(x):
                    with mock.patch.object(dtypes, "_committed_placement") as probe:
                        captured["result"] = dtypes._reference_placement(wrap(x))
                        captured["probe_calls"] = probe.call_count
                    return x

                f(jnp.zeros(3))
                self.assertEqual(captured, {"result": None, "probe_calls": 0})

    def test_reference_placement_is_the_bare_device_of_a_committed_array(self):
        """A committed concrete reference is probed and reduced to its device."""
        device = jax.local_devices()[0]
        arr = jax.device_put(np.zeros(3), device)  # noqa: TID251 chosen device
        with mock.patch.object(dtypes, "_committed_placement", wraps=dtypes._committed_placement) as committed:
            result = dtypes._reference_placement(arr)
        self.assertEqual(committed.call_count, 1)
        self.assertIs(result, device)
        self.assertIsNone(dtypes._reference_placement(jnp.zeros(3)))

    def test_staged_like_tracer_does_not_embed_a_runtime_device_put(self):
        """Scalar staging under JIT adds without an explicit device placement."""
        fail = _must_not_run("traced literals must remain uncommitted")
        self.patches.enter_context(mock.patch.object(_device_scalars, "explicit_device_array", fail))
        add_staged = jit(lambda reference: reference + staged_like(reference, 1.0))
        result = add_staged(jnp.asarray((1.0, 2.0), dtype=jnp.float64))
        np.testing.assert_array_equal(np.asarray(result), np.asarray((2.0, 3.0)))

    def test_staged_like_tracer_preserves_explicit_integer_dtype(self):
        """Scalar staging under JIT keeps a requested int32 dtype."""
        staged = jit(lambda reference: staged_like(reference, 1, dtype=jnp.int32))
        result = staged(jnp.asarray((1.0, 2.0), dtype=jnp.float64))
        self.assertEqual(result.dtype, jnp.int32)
        self.assertEqual(int(np.asarray(result)), 1)

    def test_runtime_device_put_follows_its_target_or_the_runtime_device(self):
        """An explicit target wins unasked; else the runtime device or JAX's default."""
        runtime_device, target = object(), object()
        unasked = _must_not_run("explicit placement must not query runtime device")
        rows = [
            ("runtime device", lambda: runtime_device, {}, runtime_device),
            ("explicit target", unasked, {"target": target}, target),
            ("no runtime device", lambda: None, {}, None),
        ]
        for name, device, options, expected in rows:
            with self.subTest(name), self.case() as patches:
                placements = self.patch_device_put(patches, device)
                array, placement = dtypes.runtime_device_put([1, 2, 3], **options)
                self.assertIsInstance(array, np.ndarray)
                self.assertIs(placement, expected)
                self.assertEqual(placements, [expected])

    def test_importing_simsopt_jax_alone_computes_the_raw_kernels_in_float64(self):
        """Importing only ``simsopt_jax`` computes the raw B kernel in float64."""
        for x64_setting in (None, "false"):
            with self.subTest(JAX_ENABLE_X64=x64_setting):
                environment = dict(os.environ, JAX_PLATFORMS="cpu")
                environment.pop("JAX_ENABLE_X64", None)
                if x64_setting is not None:
                    environment["JAX_ENABLE_X64"] = x64_setting
                completed = run_fresh_process("-c", _RAW_KERNEL_FLOAT64_CHILD, env=environment, timeout=300)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_explicit_device_array_rejects_reduced_precision_floats(self):
        """An exact-dtype placement accepts float64 as its only floating dtype."""
        runtime_device = jax.devices()[0]
        self.patches.enter_context(mock.patch.object(dtypes, "get_runtime_jax_device", lambda: runtime_device))
        set_backend("jax", device="cpu")
        message = r"Floating dtype must be one of \('float64',\)"
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, message):
                dtypes.explicit_device_array([1.0, 2.0], dtype=dtype)
        array = dtypes.explicit_device_array([1.0, 2.0], dtype=jnp.float64)
        self.assertEqual(array.dtype, jnp.float64)

    def test_explicit_device_array_preserves_single_device_reference(self):
        """A committed reference's bare device is used, never the runtime device."""
        reference = jax.device_put(np.zeros(3), jax.local_devices()[0])  # noqa: TID251
        (reference_device,) = reference.devices()
        unasked = _must_not_run("reference placement must not query runtime device")
        placements = self.patch_device_put(self.patches, unasked)
        array, placement = dtypes.explicit_device_array([1.0, 2.0], dtype=jnp.float64, reference=reference)
        self.assertIsInstance(array, np.ndarray)
        self.assertIs(placement, reference_device)
        self.assertEqual(placements, [reference_device])

    def test_unplaced_values_stay_uncommitted_like_jax_leaves_them(self):
        """Unplaced values stay uncommitted; a committed reference's device is used."""
        device = jax.local_devices()[0]
        self.patches.enter_context(mock.patch.object(dtypes, "get_runtime_jax_device", lambda: device))
        place = partial(dtypes.explicit_device_array, np.ones(3), dtype=jnp.float64)
        unplaced = (
            dtypes.runtime_device_put(np.ones(3)),
            dtypes.runtime_device_put_tree({"a": np.ones(3)})["a"],
            dtypes.as_jax_float64(np.ones(3)),
            place(reference=jnp.zeros(3)),
        )
        committed_reference = jax.device_put(np.zeros(3), device)  # noqa: TID251
        with_committed_reference = place(reference=committed_reference)
        for array in unplaced:
            self.assertFalse(cast(jax.Array, array).committed)
            self.assertEqual(cast(jax.Array, array).devices(), {device})
        self.assertTrue(with_committed_reference.committed)
        self.assertEqual(with_committed_reference.devices(), {device})

    def test_explicit_cross_backend_moves_pass_a_disallow_guard(self):
        """Explicit GPU-CPU moves of an array and a pytree pass a disallow guard."""
        require_cuda("CUDA device required for cross-backend placement")
        gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
        array, pair = np.arange(5.0), np.ones((2, 3))
        on_gpu = jax.device_put(array, gpu)  # noqa: TID251 test places on a chosen device
        pair_on_gpu = jax.device_put(pair, gpu)  # noqa: TID251
        tree = {"array": on_gpu, "pair": (pair_on_gpu, np.arange(3))}
        move = dtypes.runtime_device_put_tree
        with disallow_host_transfers():
            array_on_cpu, tree_on_cpu = move(on_gpu, device=cpu), move(tree, device=cpu)
            array_back = move(array_on_cpu, device=gpu)
            tree_back = move(tree_on_cpu, device=gpu)
        for placed, device in [(array_on_cpu, cpu), (tree_on_cpu, cpu), (array_back, gpu), (tree_back, gpu)]:
            for leaf in jax.tree.leaves(placed):
                self.assertEqual(leaf.devices(), {device})
                self.assertTrue(leaf.committed)
        for placed in (array_on_cpu, array_back):
            np.testing.assert_array_equal(np.asarray(placed), array)
        for placed in (tree_on_cpu, tree_back):
            np.testing.assert_array_equal(np.asarray(placed["array"]), array)
            np.testing.assert_array_equal(np.asarray(placed["pair"][0]), pair)
            np.testing.assert_array_equal(np.asarray(placed["pair"][1]), np.arange(3))

    def test_runtime_placement_moves_a_gpu_array_to_the_cpu_backend_under_a_disallow_guard(self):
        """The CPU backend moves a committed GPU array to the CPU under the guard."""
        require_cuda("CUDA device required for cross-backend placement")
        gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
        values = np.arange(4.0)
        on_gpu = jax.device_put(values, gpu)  # noqa: TID251 test places on a chosen device
        set_backend("jax", device="cpu")
        with disallow_host_transfers():
            placed = dtypes.runtime_device_put_tree({"x": on_gpu})["x"]
        self.assertEqual(placed.devices(), {cpu})
        np.testing.assert_array_equal(np.asarray(placed), values)

    def test_placement_child_scenarios(self):
        """Every ``_PLACEMENT_CHILD`` scenario passes on two CPU devices or with CUDA."""
        two_cpus = dict(JAX_PLATFORMS="cpu", XLA_FLAGS=_TWO_HOST_DEVICES)
        flags = (os.environ.get("XLA_FLAGS"), _TWO_HOST_DEVICES)
        sharded = dict(JAX_PLATFORMS="cuda,cpu", XLA_FLAGS=" ".join(f for f in flags if f))
        rows = [
            ("staged_device_array", False, two_cpus),
            ("unplaced_joins_committed", False, two_cpus),
            ("cuda_two_backend", True, dict(JAX_PLATFORMS="cuda,cpu")),
            ("cuda_sharded_cross_backend", True, sharded),
        ]
        for scenario, needs_cuda, environment in rows:
            with self.subTest(scenario):
                if needs_cuda:
                    require_cuda("CUDA device required for cross-backend placement")
                env = dict(os.environ, JAX_ENABLE_X64="1", **environment)
                completed = run_fresh_process("-c", _PLACEMENT_CHILD, scenario, env=env, timeout=300)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


def _host_values():
    """A float64 vector and an int16 scalar in a tuple, on the host."""
    return {"float": np.asarray([1.0, 2.0], dtype=np.float64), "integer": (np.asarray(3, dtype=np.int16),)}


def _device_values():
    """A float64 vector and an int32 scalar in a tuple, on the device."""
    return {"vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64), "scalar": (jnp.asarray(3, dtype=jnp.int32),)}


class TestHostBoundaryOwners(JaxTestCase):
    """The host boundary owners work under the strict guard and keep pytrees."""

    def test_placement_owners_place_host_values_under_the_strict_guard(self):
        """Placement owners take host inputs and follow the reference under the guard."""
        tree, reference = _host_values(), jnp.array(np.zeros(2, dtype=np.float64))
        with disallow_host_transfers():
            placed_tree = runtime_device_put_tree(tree)
            placed_array = explicit_device_array([5.0, 6.0], dtype=jnp.float64, reference=reference)
        self.assertEqual(placed_tree.keys(), tree.keys())
        self.assertIsInstance(placed_tree["float"], jax.Array)
        self.assertEqual(placed_tree["float"].dtype, jnp.float64)
        self.assertEqual(placed_tree["integer"][0].dtype, jnp.int16)
        np.testing.assert_array_equal(jax.device_get(placed_tree["float"]), tree["float"])
        self.assertEqual(jax.device_get(placed_tree["integer"][0]).item(), 3)
        self.assertEqual(placed_array.dtype, jnp.float64)
        self.assertEqual(placed_array.devices(), reference.devices())
        np.testing.assert_array_equal(jax.device_get(placed_array), [5.0, 6.0])

    def test_host_readers_return_writeable_host_copies_under_the_strict_guard(self):
        """Host readers give writable arrays of the requested dtype under the guard."""
        value = _device_values()
        with disallow_host_transfers():
            array = runtime.host_array(value["vector"], dtype=np.int64)
            tree = host_tree_after_ready(value)
            converted_trees = (
                runtime.host_tree(value, dtype=np.int64),
                host_tree_after_ready(value, dtype=np.int64),
            )
        for converted in converted_trees:
            self.assertEqual(converted.keys(), value.keys())
            self.assertIsInstance(converted["scalar"], tuple)
            for observed, expected in (
                (converted["vector"], value["vector"]),
                (converted["scalar"][0], value["scalar"][0]),
            ):
                self.assertEqual(observed.shape, expected.shape)
                self.assertEqual(observed.dtype, np.dtype(np.int64))
                self.assertTrue(observed.flags.writeable)
                np.testing.assert_array_equal(observed, jax.device_get(expected))
            converted["vector"][0] = -1.0
            np.testing.assert_array_equal(jax.device_get(value["vector"]), [1.0, 2.0])
        self.assertFalse(np.shares_memory(*[converted["vector"] for converted in converted_trees]))
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.dtype, np.int64)
        self.assertTrue(array.flags.writeable)
        np.testing.assert_array_equal(array, [1.0, 2.0])
        self.assertEqual(tree.keys(), value.keys())
        self.assertIsInstance(tree["vector"], np.ndarray)
        np.testing.assert_array_equal(tree["vector"], np.asarray([1.0, 2.0]))
        self.assertTrue(tree["vector"].flags.writeable)
        self.assertEqual(tree["scalar"][0].item(), 3)


_RACE_ATTEMPTS = 5


def _probe_race(test_case, attempt):
    """Retry with longer gates whose matrix stays fixed but queued intermediates can
    stay alive until work finishes (about 640 MB at 16 cylindrical applications).
    """
    for attempt_index in range(_RACE_ATTEMPTS):
        # Bind this attempt's gate and repetition count to avoid late binding to later retries.
        def gate(value, base_gate=make_execution_gate(), repetitions=2**attempt_index):
            for _ in range(repetitions):
                value = base_gate(value)
            return value

        if attempt(gate):
            return
    test_case.fail(f"consumer finished before every caller mutation; race not probed ({_RACE_ATTEMPTS} attempts)")


def _is_pending(result):
    return any(not leaf.is_ready() for leaf in jax.tree.leaves(result))


class _SpecInputs:
    def __init__(self):
        self.dofs = distinct_host_array((10,))
        self.quadpoints = distinct_host_array((65536,))
        self.samples = distinct_host_array((65536, 3))
        self.rotmat = distinct_host_array((3, 3))

    def mutate(self):
        for source in (self.dofs, self.quadpoints, self.samples, self.rotmat):
            source[:] = -7.0


def _curve_spec(factory, source, dof_count=9, **options):
    dofs = source.dofs[:dof_count]
    return factory(dofs=dofs, quadpoints=source.quadpoints, order=1, **options)


def _map_spec(source):
    return specs.make_optimizable_dof_map_spec(
        template_full_dofs=source.dofs[:9], owner_segments=(), input_mode="full", input_start=0, input_end=9
    )


def _on_base_curve(factory, x, **options):
    """``factory`` on nine DOFs over an XYZ Fourier base curve and its DOF map."""
    base = _curve_spec(specs.make_curve_xyzfourier_spec, x)
    return factory(dofs=x.dofs[:9], quadpoints=x.quadpoints, base_curve=base, base_curve_map=_map_spec(x), **options)


def _spec_factories():
    """``(name, factory)`` pairs; each factory builds a spec from ``_SpecInputs``."""
    S, c = specs, _curve_spec
    sampled = lambda x: dict(
        sample_gamma=x.samples,
        sample_gammadash=x.samples,
        sample_gammadashdash=x.samples,
        sample_gammadashdashdash=x.samples,
        sample_derivatives=3,
    )
    framed = lambda x: dict(
        rotation=S.make_zero_rotation_spec(quadpoints=x.quadpoints),
        rotation_map=_map_spec(x),
        frame_kind="frenet",
        dn=0.1,
        db=0.1,
    )
    helical = dict(m=1, ell=1, R0=1.0, r=0.1)
    return [
        ("xyz", lambda x: c(S.make_curve_xyzfourier_spec, x)),
        ("planar", lambda x: c(S.make_curve_planarfourier_spec, x, 10)),
        ("rz", lambda x: c(S.make_curve_rzfourier_spec, x, 6, nfp=1, stellsym=False)),
        ("helical", lambda x: c(S.make_curve_helical_spec, x, 3, **helical)),
        ("symmetries", lambda x: c(S.make_curve_xyzfouriersymmetries_spec, x, nfp=1, stellsym=False, ntor=1)),
        ("dof_map", _map_spec),
        ("frame_rotation", lambda x: c(S.make_frame_rotation_spec, x, 3, scale=1.0)),
        ("zero_rotation", lambda x: S.make_zero_rotation_spec(quadpoints=x.quadpoints)),
        ("coil_symmetry", lambda x: S.make_coil_symmetry_spec(rotmat=x.rotmat)),
        ("field", lambda x: S.make_field_eval_spec(x.samples)),
        ("coil_group", lambda x: make_coil_group_spec(x.samples[None], x.samples[None], x.dofs[:1], (0,))),
        ("perturbed", lambda x: _on_base_curve(S.make_curve_perturbed_spec, x, **sampled(x))),
        ("filament", lambda x: _on_base_curve(S.make_curve_filament_spec, x, **framed(x))),
    ]


def _kernel_inputs():
    points = distinct_host_array((32, 3))
    points *= 0.2
    return points, distinct_host_array((1, 64, 3)), distinct_host_array((1, 64, 3)), distinct_host_array((1,))


_FORWARD_KERNELS = (
    "biot_savart_B biot_savart_A biot_savart_dB_by_dX biot_savart_dA_by_dX"
    " biot_savart_d2B_by_dXdX biot_savart_d2A_by_dXdX biot_savart_B_and_dB"
).split()


def _field():
    curve = create_equally_spaced_curves(1, 1, False, order=1, numquadpoints=64)[0]
    points = distinct_host_array((32, 3)) * 0.2
    return JaxBiotSavart([Coil(curve, Current(1e5))]).set_points(points)


class TestBufferOwnership(JaxTestCase):
    """Caller mutation cannot change captured JAX field state or pending kernels."""

    def test_placement_snapshots_numpy_before_dispatch(self):
        """Placement snapshots aligned and misaligned NumPy inputs before mutation."""

        def check(place, misaligned):
            set_backend("jax", device="cpu")
            source = distinct_host_array((5_000_000,), misaligned=misaligned)
            expected = source.copy()
            result = place(source)
            source[:] = -7.0
            assert_tree_equal(result, expected)

        places = [
            runtime_device_put,
            runtime_device_put_tree,
            lambda x: explicit_device_array(x, dtype=np.float64),
            lambda x: as_jax_array(x, dtype=np.float64),
            as_jax_float64,
        ]
        cases = [dict(place=p, misaligned=m) for p in places for m in (False, True)]
        self.run_cases(check, cases)

    def test_snapshot_preserves_device_arrays_and_tracers(self):
        """Snapshots keep JAX arrays by identity and traced values, in mixed trees."""
        device = jnp.arange(16, dtype=jnp.float64)
        self.assertIs(snapshot_host_tree(device), device)
        tree = snapshot_host_tree({"device": device, "host": np.arange(16)})
        self.assertIs(tree["device"], device)
        np.testing.assert_array_equal(jit(snapshot_host_tree)(device), device)

    def test_mixed_device_host_sequence_snapshots_numpy(self):
        """Mixed host/device sequences snapshot NumPy leaves before deferred reads."""

        def check(convert):
            def attempt(gate):
                source = distinct_host_array((65536,))
                expected = source.copy()
                zeros = jnp.zeros(source.shape)
                gate(zeros).block_until_ready()
                convert([zeros, source]).block_until_ready()
                result = convert([gate(zeros), source])
                pending = _is_pending(result)
                source[:] = -7.0
                np.testing.assert_array_equal(result.block_until_ready()[1], expected)
                return pending

            _probe_race(self, attempt)

        converts = [lambda x: as_jax_array(x, dtype=np.float64), as_jax_float64, runtime_device_put]
        self.run_cases(check, [dict(convert=convert) for convert in converts])

    def test_spec_factories_own_numpy_leaves(self):
        """Spec factories keep owned NumPy snapshots after every input is mutated."""

        def check(case_id_0, factory):
            source = _SpecInputs()
            result = factory(source)
            expected = jax.tree.map(lambda leaf: np.array(leaf, copy=True), result)
            source.mutate()
            assert_tree_equal(result, expected)

        cases = [dict(case_id_0=n, factory=f) for n, f in _spec_factories()]
        self.run_cases(check, cases)

    def test_adapter_curve_capture_owns_native_coefficients(self):
        """Adapter curve capture owns native coefficients and quadrature nodes."""

        def check(case_id_0, native_curve):
            quadpoints = distinct_host_array((65536,))
            quadpoints[:] -= 1.0
            curve = native_curve(quadpoints)
            coefficients = distinct_host_array((curve.num_dofs(),))
            curve.coefficients = coefficients
            result = curve_spec_from_adapter_curve(curve)
            expected = jax.tree.map(lambda x: np.array(x, copy=True), result)
            coefficients[:] = -7.0
            quadpoints[:] = -7.0
            assert_tree_equal(result, expected)

        symmetric = lambda q: CurveXYZFourierSymmetries(q, order=1, nfp=1, stellsym=False)
        curves = [("helical", lambda q: CurveHelical(q, order=1)), ("symmetries", symmetric)]
        cases = [dict(case_id_0=name, native_curve=curve) for name, curve in curves]
        self.run_cases(check, cases)

    def test_cylindrical_points_own_numpy_before_conversion(self):
        """Cylindrical setters snapshot coordinates before deferred conversion."""

        def check(misaligned):
            def attempt(gate):
                source = distinct_host_array((1_666_667, 3), misaligned=misaligned)
                saved = source.copy()
                canonical = adapter._canonical_set_points_cyl
                gate(jnp.asarray(saved)).block_until_ready()
                expected = host_snapshot(adapter._cyl_points_to_cart(canonical(jnp.asarray(saved))))
                gated = lambda points: canonical(gate(points))
                with mock.patch.object(adapter, "_canonical_set_points_cyl", gated):
                    field = JaxBiotSavart([]).set_points_cyl(source)
                    pending = _is_pending(field._points_jax)
                    source[:] = -7.0
                    np.testing.assert_array_equal(jax.block_until_ready(field._points_jax), expected)
                return pending

            _probe_race(self, attempt)

        self.run_cases(check, [dict(misaligned=misaligned) for misaligned in (False, True)])

    def test_raw_forward_kernel_owns_numpy_inputs(self):
        """Raw forward kernels snapshot NumPy inputs before deferred execution."""

        def check(pending_input, case_id_1, kernel):
            def attempt(gate):
                inputs = list(_kernel_inputs())
                expected = host_snapshot(kernel(*inputs))
                index = 0 if pending_input == "points" else 1
                gate(jnp.asarray(inputs[index])).block_until_ready()
                inputs[index] = gate(jnp.asarray(inputs[index]))
                result = kernel(*inputs)
                pending = _is_pending(result)
                for source in inputs:
                    if isinstance(source, np.ndarray):
                        source[:] = -7.0
                assert_tree_equal(result, expected)
                return pending

            _probe_race(self, attempt)

        cases = [
            dict(pending_input=i, case_id_1=k, kernel=getattr(kernels, k))
            for i in ("points", "gamma")
            for k in _FORWARD_KERNELS
        ]
        self.run_cases(check, cases)

    def test_raw_pullback_owns_numpy_inputs(self):
        """Raw pullbacks snapshot geometry, current and cotangent before execution."""

        def check(case_id_0, kernel):
            def attempt(gate):
                points, gamma, dash, current = _kernel_inputs()
                cotangent = distinct_host_array(points.shape)
                expected = host_snapshot(kernel(points, cotangent, gamma, dash, current))
                gate(jnp.asarray(points)).block_until_ready()
                pending_points = gate(jnp.asarray(points))
                result = kernel(pending_points, cotangent, gamma, dash, current)
                pending = _is_pending(result)
                for source in (cotangent, gamma, dash, current):
                    source[:] = -7.0
                assert_tree_equal(result, expected)
                return pending

            _probe_race(self, attempt)

        pullbacks = [("raw", kernels.biot_savart_B_vjp), ("grouped", group_biot_savart_B_vjp)]
        self.run_cases(check, [dict(case_id_0=n, kernel=k) for n, k in pullbacks])

    def test_explicit_dofs_are_owned(self):
        """Explicit-DOF reconstruction keeps owned snapshots through deferred work."""

        def check(entry):
            def attempt(gate):
                field = _field()
                source = distinct_host_array((len(field.x),))
                source[:] = np.array(field.x, dtype=np.float64)
                call = getattr(field, entry)
                expected = host_snapshot(call(source))
                normalize = field._normalize_explicit_coil_dofs
                gate(normalize(source)).block_until_ready()
                gated = lambda dofs: gate(normalize(dofs))
                with mock.patch.object(field, "_normalize_explicit_coil_dofs", gated):
                    result = call(source)
                    # Grouped builders materialize captured curve constants while lowering;
                    # the normalization boundary and per-coil payload remain deferred.
                    pending = _is_pending(result) if entry == "coil_specs_from_dofs" else True
                    source[:] = -7.0
                    assert_tree_equal(result, expected)
                return pending

            _probe_race(self, attempt)

        entries = ("_normalize_explicit_coil_dofs", "coil_specs_from_dofs", "coil_set_spec_from_dofs")
        self.run_cases(check, [dict(entry=entry) for entry in entries])

    def test_adapter_pullback_owns_numpy_cotangent(self):
        """Adapter pullbacks snapshot caller cotangents before deferred reads."""

        def check(entry, shape):
            def attempt(gate):
                field = _field()
                call = getattr(field, entry)
                source = distinct_host_array(shape)
                expected = host_snapshot(call(source))
                gate(field._points_jax).block_until_ready()
                field._points_jax = gate(field._points_jax)
                result = call(source)
                pending = _is_pending(result)
                source[:] = -7.0
                assert_tree_equal(result, expected)
                return pending

            _probe_race(self, attempt)

        cases = [
            dict(entry=f"{name}_pullback_native", shape=shape)
            for name, shape in (("B", (32, 3)), ("A", (32, 3)), ("dB_by_dX", (32, 3, 3)), ("dA_by_dX", (32, 3, 3)))
        ]
        self.run_cases(check, cases)

    def test_group_coil_data_owns_numpy_inputs(self):
        """Grouping per-coil data owns NumPy snapshots."""
        def attempt(gate):
            _, gamma, dash, current = _kernel_inputs()
            inputs = ([gamma[0]], [dash[0]], [current[0:1].reshape(())])
            expected = host_snapshot(kernels.group_coil_data(*inputs))
            convert = kernels._as_jax_float64
            for entry in (gamma[0], current[0:1].reshape(())):
                gate(convert(entry)).block_until_ready()
            with mock.patch.object(kernels, "_as_jax_float64", lambda x: gate(convert(x))):
                result = kernels.group_coil_data(*inputs)
                pending = _is_pending(tuple(group[:3] for group in result))
                for source in (gamma, dash, current):
                    source[:] = -7.0
                assert_tree_equal(result, expected)
            return pending

        _probe_race(self, attempt)


OWNER_DOFS = np.asarray([0.9, 0.1, -0.2, 0.05, 0.8, 0.15, -0.1, 0.02, 0.3, 1.5e5])


def _dof_map_spec(*, size: int, start: int):
    return specs.make_optimizable_dof_map_spec(
        template_full_dofs=np.zeros(size, dtype=np.float64),
        owner_segments=[(start, start + size, 0, size)],
        input_mode="full",
        input_start=start,
        input_end=start + size,
    )


def _extraction_spec():
    return specs.make_coil_dof_extraction_spec(
        curve=specs.make_curve_xyzfourier_spec(
            dofs=np.zeros(9, dtype=np.float64),
            quadpoints=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
            order=1,
        ),
        curve_map=_dof_map_spec(size=9, start=0),
        current_map=_dof_map_spec(size=1, start=9),
        rotmat=np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        scale=-1.0,
    )


def _coil_field_inputs(spec):
    coil_set = specs.make_coil_set_dof_extraction_spec([spec])
    owner_dofs = jnp.array(OWNER_DOFS)
    return coil_set_spec_from_dof_extraction_spec(coil_set, owner_dofs).field_inputs()


class TestCoilDofExtractionSpecPlacement(JaxTestCase):
    """Captured extraction specs are host-resident; argument specs are not."""

    def test_the_maker_leaves_the_extraction_spec_device_resident(self):
        """The argument role is the default, so the maker must not host-place."""
        leaves = jax.tree.leaves(_extraction_spec())
        self.assertTrue(leaves, "leaves")
        off_device = [leaf for leaf in leaves if not isinstance(leaf, jax.Array)]
        self.assertFalse(
            off_device, f"{len(off_device)} leaves off the device: an implicit transfer when passed to jit"
        )

    def test_host_value_moves_every_array_leaf_of_the_spec_to_the_host(self):
        """Nothing may be left behind for lowering to read back."""
        on_host = host_value(_extraction_spec())
        leaves = jax.tree.leaves(on_host)
        on_device = [leaf for leaf in leaves if isinstance(leaf, jax.Array)]
        self.assertFalse(on_device, f"{len(on_device)} device leaves: each is read back to the host at lowering")
        self.assertTrue(all(isinstance(leaf, np.ndarray) for leaf in leaves))
        self.assertEqual(jax.tree.structure(on_host), jax.tree.structure(_extraction_spec()))

    def test_host_placement_reconstructs_the_coil_arrays_bit_for_bit(self):
        """Where the frozen template lives may not change one bit of the coils."""
        device_inputs = _coil_field_inputs(_extraction_spec())
        host_inputs = _coil_field_inputs(host_value(_extraction_spec()))
        self.assertTrue(len(host_inputs) == len(device_inputs) == 1)
        for name, host, device in zip(
            ("gammas", "gammadashs", "currents"), host_inputs[0], device_inputs[0], strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(host), np.asarray(device), err_msg=f"host-resident template changed the reconstructed {name}"
            )

    def test_a_program_capturing_the_host_resident_spec_compiles_under_the_strict_guard(self):
        """Lowering reads no captured template back to the host (GPU-effective)."""
        spec = specs.make_coil_set_dof_extraction_spec([host_value(_extraction_spec())])

        def total_gamma(owner_dofs):
            coil_set = coil_set_spec_from_dof_extraction_spec(spec, owner_dofs)
            gammas, _gammadashs, _currents = coil_set.field_inputs()[0]
            return gammas.sum()

        owner_dofs = as_jax_float64(OWNER_DOFS)
        with jax.transfer_guard("disallow"):
            total = jit(total_gamma)(owner_dofs)
        self.assertTrue(np.isfinite(np.asarray(jax.device_get(total))))


def _group(start: float, indices: tuple[int, ...]):
    """A one-coil group with gammas, gammadashs and current counting up from start."""
    gamma, dash = [start, start + 1, start + 2], [start + 3, start + 4, start + 5]
    return make_coil_group_spec([[gamma]], [[dash]], [start + 6], indices)


class TestGroupedCoilSetSpec(JaxTestCase):
    """Grouped-coil construction keeps supplied specs and owns tuple inputs."""

    def test_existing_specs_retain_identity_and_order(self):
        """Existing specs, including repeated references, keep identity and order."""
        first, second = _group(1.0, (5,)), _group(8.0, (2,))
        result = make_grouped_coil_set_spec([first, second, first])
        self.assertEqual(len(result.groups), 3)
        for group, expected in zip(result.groups, (first, second, first), strict=True):
            self.assertIs(group, expected)

    def test_tuple_inputs_are_converted_and_snapshot_host_arrays(self):
        """Tuple conversion keeps float64 values after callers mutate host inputs."""
        gammas = aligned_host_copy([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        gammadashs = aligned_host_copy([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
        currents = aligned_host_copy([13.0])
        result = make_grouped_coil_set_spec([(gammas, gammadashs, currents, (4,))])
        for value, mutated in ((gammas, -1.0), (gammadashs, -2.0), (currents, -3.0)):
            value[:] = mutated
        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertEqual(group.coil_indices, (4,))
        values = np.arange(1.0, 14.0)
        np.testing.assert_array_equal(group.gammas, values[:6].reshape(1, 2, 3))
        np.testing.assert_array_equal(group.gammadashs, values[6:12].reshape(1, 2, 3))
        np.testing.assert_array_equal(group.currents, [13.0])
        for array in (group.gammas, group.gammadashs, group.currents):
            self.assertEqual(array.dtype, np.dtype("float64"))

    def test_mixed_one_pass_generator_preserves_order(self):
        """A mixed generator is consumed once without dropping or reordering groups."""
        first, last = _group(1.0, (8,)), _group(8.0, (1,))
        middle = (first.gammas, first.gammadashs, first.currents, (3,))
        groups = (group for group in (first, middle, last))
        result = make_grouped_coil_set_spec(groups)
        self.assertEqual(tuple(group.coil_indices for group in result.groups), ((8,), (3,), (1,)))
        self.assertIs(result.groups[0], first)
        self.assertIs(result.groups[2], last)
        np.testing.assert_array_equal(result.groups[1].currents, [7.0])
        self.assertEqual(tuple(groups), ())

    def test_empty_iterable_produces_empty_groups(self):
        """An empty iterable constructs an empty immutable groups tuple."""
        self.assertEqual(make_grouped_coil_set_spec(iter(())).groups, ())
