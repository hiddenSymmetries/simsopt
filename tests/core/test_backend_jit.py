"""Contract of ``simsopt_jax.backend.jit.jit``, the copying ``jax.jit``.

simsopt overwrites cached NumPy arrays in place; ``jax.jit`` on CPU may read
an aligned NumPy argument after the call returned. The wrapper copies NumPy
leaves of dynamic arguments, passes static arguments and JAX arrays by
identity, forwards the compiled function's attributes, and leaves placement
to JAX.
"""

from __future__ import annotations

import warnings
from functools import partial
from unittest import mock

import numpy as np
from unittest_jax_support import (
    JaxTestCase,
    aligned_host_copy,
    optional_jax_imports,
    require_cuda,
)

with optional_jax_imports():
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend import jit as jit_module
    from simsopt_jax.backend.jit import CopyingJit, jit
    from simsopt_jax.backend.runtime import set_backend
    from simsopt_jax.runtime.host_boundary import snapshot_host_tree

# A matrix power slow enough (milliseconds) to still be running when the
# caller overwrites the input; it is the identity, so ``gate[0, 0] == 1``.
_GATE_SIZE = 1000
_GATE_POWER = 32
_ATTEMPTS = 5
_INPUT_LENGTH = 16

# How a NumPy input reaches the wrapper: (name, function, call).
_PASSING_STYLES = (
    ("positional", lambda x, gate: jnp.sum(x * gate[0, 0]), lambda f, x, g: f(x, g)),
    ("list", lambda xs, gate: jnp.sum(xs[0] * gate[0, 0]), lambda f, x, g: f([x], g)),
    ("dict", lambda d, gate: jnp.sum(d["x"] * gate[0, 0]), lambda f, x, g: f({"x": x}, g)),
    ("keyword", lambda gate, x: jnp.sum(x * gate[0, 0]), lambda f, x, g: f(g, x=x)),
)


class _Table:
    """Pytree holding a NumPy array; hashes by identity, so it can be static."""

    def __init__(self, values):
        self.values = values

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


with optional_jax_imports():
    jax.tree_util.register_pytree_node_class(_Table)


def _scaled(received: list, x, table):
    """``x * sum(table.values)``, appending ``table`` to ``received``."""
    received.append(table)
    return x * jnp.sum(jnp.asarray(table.values))


def _table_probe(kind: str):
    """``_scaled`` with the parameter kinds ``kind`` names, and the tables it receives."""
    received = []
    probes = {
        "positional-only x": lambda x, /, table: _scaled(received, x, table),
        "keyword-only table": lambda x, *, table: _scaled(received, x, table),
        "positional-or-keyword": lambda x, table: _scaled(received, x, table),
        "defaulted last": lambda x, table, label="": _scaled(received, x, table),
        "*args": lambda x, *tables: _scaled(received, x, tables[-1]),
    }
    return received, probes[kind]


def _recording_copies(copies: list):
    """Patch ``snapshot_host_tree`` to append each ``(value, copy)`` pair to ``copies``."""

    def record(value):
        copies.append((value, snapshot_host_tree(value)))
        return copies[-1][1]

    return mock.patch.object(jit_module, "snapshot_host_tree", record)


class CopyingJitTests(JaxTestCase):
    def test_numpy_input_overwritten_after_the_call_keeps_the_result(self):
        """A result held across an in-place overwrite of its NumPy input keeps the old value."""
        if jax.default_backend() == "cpu" and not jax.config.values[
            "jax_cpu_enable_async_dispatch"
        ]:
            self.skipTest("synchronous CPU dispatch: no overwrite can race the call")
        slow = jit(lambda m: jnp.linalg.matrix_power(m, _GATE_POWER))
        eye = jnp.eye(_GATE_SIZE)
        for name, fun, call in _PASSING_STYLES:
            with self.subTest(style=name):
                f = jit(fun)
                call(f, np.ones(_INPUT_LENGTH), eye).block_until_ready()  # compiles f
                for _ in range(_ATTEMPTS):
                    buffer = aligned_host_copy(np.ones(_INPUT_LENGTH))
                    gate = slow(eye)
                    held = call(f, buffer, gate)
                    buffer[:] = 2.0
                    if not gate.is_ready():
                        self.assertEqual(float(held), float(_INPUT_LENGTH))
                        break
                else:
                    self.fail("the gate finished before every overwrite; race not probed")

    def test_static_selectors_match_raw_jit(self):
        """Every selector form makes the arguments static that ``jax.jit`` does:
        static ones, even pytrees of NumPy arrays, pass by identity and never
        recompile, and dynamic NumPy arguments are copied."""
        table = _Table(np.arange(3.0))
        positional = lambda f, x: f(x, table)
        keyword = lambda f, x: f(x, table=table)
        # (name, probe kind, compile with the selector, call); each compile
        # builds a fresh iterator.
        cases = (
            ("argnums, positional", "positional-or-keyword", lambda j, fun: j(fun, static_argnums=1), positional),
            ("negative argnums", "positional-or-keyword", lambda j, fun: j(fun, static_argnums=-1), positional),
            # -1 names ``label``, which this call leaves out, so ``table`` is dynamic.
            ("negative argnums", "defaulted last", lambda j, fun: j(fun, static_argnums=-1), positional),
            ("negative argnums", "*args", lambda j, fun: j(fun, static_argnums=-1), positional),
            (
                "negative argnums, keyword",
                "defaulted last",
                lambda j, fun: j(fun, static_argnums=-1),
                lambda f, x: f(x, table, label="static"),
            ),
            ("argnums, keyword", "positional-or-keyword", lambda j, fun: j(fun, static_argnums=1), keyword),
            ("argnames, positional", "positional-or-keyword", lambda j, fun: j(fun, static_argnames="table"), positional),
            ("argnames, keyword", "positional-or-keyword", lambda j, fun: j(fun, static_argnames="table"), keyword),
            ("argnames", "positional-only x", lambda j, fun: j(fun, static_argnames="table"), positional),
            ("argnums", "positional-only x", lambda j, fun: j(fun, static_argnums=1), keyword),
            ("argnames", "keyword-only table", lambda j, fun: j(fun, static_argnames="table"), keyword),
            ("iterator argnums", "positional-or-keyword", lambda j, fun: j(fun, static_argnums=iter((1,))), keyword),
            (
                "iterator argnames",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnames=iter(("table",))),
                positional,
            ),
        )
        for name, kind, compile_with, call in cases:
            with self.subTest(selector=name, parameters=kind):
                raw_received, raw_fun = _table_probe(kind)
                raw = compile_with(jax.jit, raw_fun)  # noqa: TID251 raw jax.jit is the reference
                call(raw, np.ones(2))

                received, fun = _table_probe(kind)
                f = compile_with(jit, fun)
                x = np.ones(2)
                copies = []
                with _recording_copies(copies):
                    for _ in range(3):
                        np.testing.assert_array_equal(call(f, x), np.full(2, 3.0))
                self.assertEqual(len(received), 1)
                if kind == "positional-or-keyword":  # every selector here makes ``table`` static
                    self.assertIs(received[0], table)
                self.assertEqual(received[0] is table, raw_received[0] is table)
                self.assertEqual(f._cache_size(), 1)
                copied = [value for value, _ in copies]
                self.assertTrue(any(value is x for value in copied), copied)
                self.assertEqual(any(value is table for value in copied), raw_received[0] is not table)

    def test_every_decorator_form_compiles_with_its_options(self):
        """``@jit``, ``partial(jit, ...)``, ``@jit(...)`` and ``jit(f, ...)`` give the same wrapper."""

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
        """Arguments without NumPy leaves skip the copy; NumPy leaves are copied privately."""
        copies = []
        add = jit(lambda a, tree: a + tree["b"][0] * tree["b"][1])
        values = jnp.arange(3.0)
        host = np.arange(3.0)
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

    def test_masked_array_is_rejected_like_raw_jit(self):
        """A NumPy masked array keeps its mask through the copy, so JAX rejects it
        as ``jax.jit`` does instead of computing with the masked entries."""
        masked = np.ma.masked_array(np.arange(3.0), mask=[False, True, False])
        with self.assertRaisesRegex(ValueError, "masked arrays are not supported"):
            jit(lambda x: x * 1.0)(masked)

    def test_placement_matches_raw_jit(self):
        """NumPy, uncommitted and committed inputs land where ``jax.jit`` puts them."""
        identity = jit(lambda x: x * 1.0)
        raw = jax.jit(lambda x: x * 1.0)  # noqa: TID251 reference placement of raw jit
        host = np.arange(4.0)
        cpu = jax.devices("cpu")[0]
        inputs = {
            "NumPy": host,
            "uncommitted": jnp.asarray(host),
            "committed CPU": jax.device_put(host, cpu),  # noqa: TID251 committed input under test
        }
        for name, value in inputs.items():
            with self.subTest(input=name):
                self.assertEqual(identity(value).devices(), raw(value).devices())

    def test_runtime_gpu_receives_numpy_and_uncommitted_inputs(self):
        """With the runtime on the GPU, NumPy and uncommitted inputs run there; committed ones stay."""
        require_cuda("CUDA device required for runtime GPU placement")
        with warnings.catch_warnings():
            # Determinism XLA flags are irrelevant to placement.
            warnings.filterwarnings("ignore", "Backend mode", RuntimeWarning)
            set_backend("jax", device="gpu")
        gpu = jax.devices("gpu")[0]
        cpu = jax.devices("cpu")[0]
        identity = jit(lambda x: x * 1.0)
        host = np.arange(4.0)
        expected = {
            "NumPy": (host, gpu),
            "uncommitted": (jnp.asarray(host), gpu),
            "committed CPU": (jax.device_put(host, cpu), cpu),  # noqa: TID251 committed input under test
            "committed GPU": (jax.device_put(host, gpu), gpu),  # noqa: TID251 committed input under test
        }
        for name, (value, device) in expected.items():
            with self.subTest(input=name):
                self.assertEqual(identity(value).devices(), {device})
