"""Native-input ownership, JAX transformation compatibility, and device routing."""

import unittest
from collections import OrderedDict, UserDict, defaultdict, namedtuple
from contextlib import nullcontext
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import simsopt.geo.jit as jit_module
from simsopt.geo.config import parameters
from simsopt.geo.jit import jit
from unittest_jax_support import aligned_host_copy

Arrays = namedtuple("Arrays", ["first", "rest"])


class HashableArray(np.ndarray):
    __eq__ = object.__eq__

    # A deliberately hashable ndarray fixture exercises JAX static arguments.
    def __hash__(self) -> int:
        return object.__hash__(self)


def _static_array():
    constant = np.array([3.0]).view(HashableArray)
    constant.flags.writeable = False
    return constant


def _positional_adder(traced):
    def add_static(value, constant):
        traced.append(constant)
        return value + constant[0]

    return add_static


def _variadic_adder(traced):
    def add_last(value, *constants):
        traced.append(constants[-1])
        return value + constants[0] + constants[-1][0]

    return add_last


def _keyword_only_adder(traced):
    def add_static(value, *, options):
        traced.append(options)
        return value + options[0][0]

    return add_static


def positional(wrapped, static):
    return wrapped(2.0, static)


def keyword(wrapped, static):
    return wrapped(2.0, constant=static)


class IndexSelector:
    def __index__(self):
        return 1


class FrozenDictionary(dict):
    setter_calls = 0

    def __setitem__(self, key, value):
        self.setter_calls += 1
        raise TypeError("FrozenDictionary does not allow item assignment")


class TransformDictionary(dict):
    setter_calls = 0

    def __setitem__(self, key, value):
        self.setter_calls += 1
        super().__setitem__(key, 2 * value)


@jax.tree_util.register_pytree_node_class
class RegisteredArrays:
    def __init__(self, values):
        self.values = values

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0])


class JitOwnershipTests(unittest.TestCase):
    def test_no_jit_snapshots_nested_inputs_and_retains_plain_execution(self):
        """With jit off, bare and nested arrays are copied, the call runs on the native device, and the name is kept."""

        def record(x, *, nested):
            return x, nested

        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit_module.jit(record)
        x, y = np.ones(3), np.ones(4)
        device = object()
        with mock.patch.object(jit_module, "native_jax_device", return_value=device), \
                mock.patch.object(jit_module.jax, "default_device") as context:
            held, nested = wrapped(x, nested={"arrays": [y]})
        x[:] = 4
        y[:] = 7
        np.testing.assert_array_equal(held, np.ones(3))
        np.testing.assert_array_equal(nested["arrays"][0], np.ones(4))
        context.assert_called_once_with(device)
        self.assertEqual(wrapped.__name__, record.__name__)

    def test_compiled_dispatch_receives_owned_inputs_under_native_device_context(self):
        """Both resolved selectors reach jax.jit, so no JAX version infers either;
        a negative index names its parameter and counts from the signature."""
        cases = (
            ((1,), lambda wrapped, x, static: wrapped(x, static)),
            (-1, lambda wrapped, x, static: wrapped(x, marker=static)),
        )
        for argnums, call in cases:
            with self.subTest(static_argnums=argnums):
                device_sentinel = object()
                static_sentinel = object()
                x = np.ones(4)
                fake_compile = mock.Mock(side_effect=lambda fun, **kwargs: fun)
                context = mock.Mock(side_effect=lambda device: nullcontext())
                with mock.patch.dict(parameters, {"jit": True}), \
                        mock.patch.object(jit_module, "jaxjit", fake_compile), \
                        mock.patch.object(jit_module, "native_jax_device", return_value=device_sentinel), \
                        mock.patch.object(jit_module.jax, "default_device", context):
                    wrapped = jit_module.jit(lambda a, marker: (a, marker), static_argnums=argnums)
                    held, marker = call(wrapped, x, static_sentinel)
                x[:] = 5
                np.testing.assert_array_equal(held, np.ones(4))
                self.assertIs(marker, static_sentinel)
                context.assert_called_once_with(device_sentinel)
                self.assertEqual(
                    fake_compile.call_args.kwargs,
                    {"static_argnums": (1,), "static_argnames": ("marker",)},
                )

    def test_real_jax_outputs_keep_call_time_values_after_input_mutation(self):
        """Even ready identity outputs of bare, nested and keyword arrays own their values."""

        def identities(values, bundle, *, other):
            return jnp.asarray(values), jnp.asarray(bundle[0]["arrays"][0]), jnp.asarray(other["value"])

        for use_jit in (True, False):
            with self.subTest(jit=use_jit), mock.patch.dict(parameters, {"jit": use_jit}):
                wrapped = jit_module.jit(identities)
                inputs = [aligned_host_copy(np.arange(1.0, n + 1.0)) for n in (16, 8, 16)]
                expected = [values.copy() for values in inputs]
                held = jax.block_until_ready(wrapped(inputs[0], [{"arrays": (inputs[1],)}], other={"value": inputs[2]}))
                for values in inputs:
                    values[:] = -7.0
                for actual, values in zip(held, expected):
                    np.testing.assert_array_equal(actual, values)
                self.assertEqual(held[0].devices(), {jit_module.native_jax_device()})

    def test_real_jax_static_arguments_and_wraps_metadata_are_preserved(self):
        """Positional and keyword statics compile on the native device, and the wrapper keeps the function metadata."""

        def powers(values, order, *, scale):
            """Scaled powers with positional and keyword static parameters."""
            return scale * jnp.sum(values**order)

        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit_module.jit(powers, static_argnums=(1,), static_argnames=("scale",))
        values = np.array([1.0, 2.0, 3.0])
        result = wrapped(values, 3, scale=0.5)
        self.assertEqual(float(result), 18.0)
        self.assertEqual(result.devices(), {jit_module.native_jax_device()})
        self.assertIs(wrapped.__wrapped__, powers)
        self.assertEqual(wrapped.__name__, powers.__name__)
        self.assertEqual(wrapped.__doc__, powers.__doc__)

    def test_tracers_grad_hessian_nested_jit_and_vmap_preserve_derivatives(self):
        """grad, Hessian, raw jax.jit and vmap through the wrapper match analytic derivatives, with jit on and off."""
        values = np.linspace(0.25, 1.0, 4)
        for use_jit in (True, False):
            with self.subTest(jit=use_jit), mock.patch.dict(parameters, {"jit": use_jit}):
                wrapped = jit_module.jit(lambda x: jnp.sum(x**3))
                np.testing.assert_allclose(jax.grad(wrapped)(values), 3 * values**2, rtol=1e-14, atol=0)
                np.testing.assert_allclose(jax.jacfwd(jax.grad(wrapped))(values), np.diag(6 * values), rtol=1e-14, atol=0)
                nested = jax.jit(jax.grad(wrapped))  # noqa: TID251 checks composition with raw jax.jit
                np.testing.assert_allclose(nested(values), 3 * values**2, rtol=1e-14, atol=0)
                batch = np.stack([values, 2 * values])
                np.testing.assert_allclose(jax.vmap(wrapped)(batch), np.sum(batch**3, axis=1), rtol=1e-14, atol=0)

    def test_plain_builtin_mappings_preserve_key_order_and_factory(self):
        """With jit off, copied dict, OrderedDict and defaultdict inputs keep their type, key order and factory."""
        mappings: list[dict[object, np.ndarray | list[float]]] = [
            {1: np.array([2.0]), "a": np.array([3.0])},
            {"z": np.array([2.0]), "a": np.array([3.0])},
            OrderedDict([("z", np.array([2.0])), ("a", np.array([3.0]))]),
            defaultdict(list, {1: np.array([2.0]), "a": np.array([3.0])}),
            defaultdict(list, {"z": np.array([2.0]), "a": np.array([3.0])}),
        ]
        with mock.patch.dict(parameters, {"jit": False}):
            identity = jit(lambda values: values)
        for mapping in mappings:
            with self.subTest(kind=type(mapping), keys=tuple(mapping)):
                nested = Arrays([mapping], None)
                copied = identity(nested).first[0]
                self.assertIs(type(copied), type(mapping))
                self.assertEqual(tuple(copied), tuple(mapping))
                for key in mapping:
                    self.assertFalse(np.shares_memory(copied[key], mapping[key]))
                    np.testing.assert_array_equal(copied[key], mapping[key])
        self.assertIs(identity(mappings[-1]).default_factory, list)

    def test_explicit_statics_without_signature(self):
        """Explicit static selectors compile builtins such as getattr and min, which have no signature."""
        with mock.patch.dict(parameters, {"jit": True}):
            attribute = jit(getattr, static_argnums=1)
            minimum = jit(min, static_argnums=(0, 1))
        self.assertEqual(tuple(int(item) for item in attribute(jnp.array([3.0, 4.0]), "shape")), (2,))
        self.assertEqual(int(minimum(3, 2)), 2)

    def test_unregistered_dictionaries_are_opaque_leaves_without_hooks(self):
        """With jit off, dictionaries JAX does not register are returned as is and their item setters never run."""
        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit(lambda values: values)
        for kind in (FrozenDictionary, TransformDictionary, UserDict):
            with self.subTest(kind=kind.__name__):
                incoming = kind({"entry": np.array([3.0])})
                self.assertIs(wrapped(incoming), incoming)
                self.assertIs(wrapped(incoming)["entry"], incoming["entry"])
                np.testing.assert_array_equal(wrapped(incoming)["entry"], [3.0])
                self.assertEqual(getattr(incoming, "setter_calls", 0), 0)

    def test_registered_custom_pytree_snapshots_dynamic_numpy_leaves(self):
        """NumPy leaves of a registered custom pytree are copied at call time, with jit on and off."""
        for use_jit in (False, True):
            with self.subTest(jit=use_jit), mock.patch.dict(parameters, {"jit": use_jit}):
                wrapped = jit(lambda container: jnp.asarray(container.values))
                incoming = RegisteredArrays(aligned_host_copy(np.arange(1.0, 17.0)))
                expected = incoming.values.copy()
                held = wrapped(incoming).block_until_ready()
                incoming.values[:] = -7
                np.testing.assert_array_equal(held, expected)

    def test_matrix_semantics_and_storage_with_jit_off(self):
        """With jit off, an np.matrix input is copied and keeps its type and matrix product."""
        with mock.patch.dict(parameters, {"jit": False}):
            square = jit(lambda value: (value, value * value))
        matrix = np.matrix([[1.0, 2.0], [3.0, 4.0]])
        copied, result = square(matrix)
        self.assertIsInstance(copied, np.matrix)
        self.assertIsInstance(result, np.matrix)
        self.assertFalse(np.shares_memory(copied, matrix))
        matrix[:] = -7
        np.testing.assert_array_equal(copied, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, [[7.0, 10.0], [15.0, 22.0]])

    def test_mask_semantics_and_storage_with_jit_off(self):
        """With jit off, a masked-array input is copied with its mask, and the result still ignores masked entries."""
        with mock.patch.dict(parameters, {"jit": False}):
            square = jit(lambda value: (value, value * value))
        masked = np.ma.array([2.0, 9.0], mask=[False, True])
        copied, result = square(masked)
        self.assertIsInstance(copied, np.ma.MaskedArray)
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertFalse(np.shares_memory(copied.data, masked.data))
        self.assertFalse(np.shares_memory(copied.mask, masked.mask))
        masked.data[:] = -7
        np.ma.getmaskarray(masked)[:] = False
        np.testing.assert_array_equal(copied.data, [2.0, 9.0])
        np.testing.assert_array_equal(copied.mask, [False, True])
        np.testing.assert_array_equal(result.mask, [False, True])
        self.assertEqual(result.sum(), 4.0)

    def assert_traced_once(self, make, options, call, static, expected=5.0, is_static=True):
        """Three calls trace once; ``is_static`` says whether tracing saw ``static`` itself."""
        traced = []
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(make(traced), **options)
        for _ in range(3):
            self.assertEqual(float(call(wrapped, static)), expected)
        self.assertEqual(len(traced), 1)
        self.assertEqual(traced[0] is static, is_static)

    def test_static_selectors_keep_identity_cache_and_actual_call_positioning(self):
        """Static NumPy objects keep identity and cache hits; explicit selectors
        do not infer the other selector or bind keyword calls."""
        inferred = ({"static_argnums": 1}, {"static_argnums": (1,)}, {"static_argnums": np.int64(1)},
                    {"static_argnums": np.array(1, dtype=np.int64)}, {"static_argnums": IndexSelector()},
                    {"static_argnums": -1}, {"static_argnames": "constant"})
        cases = [
            *((options, call, True) for options in inferred for call in (positional, keyword)),
            ({"static_argnums": 1, "static_argnames": ()}, positional, True),
            ({"static_argnums": 1, "static_argnames": ()}, keyword, False),
            ({"static_argnums": (), "static_argnames": "constant"}, positional, False),
            ({"static_argnums": (), "static_argnames": "constant"}, keyword, True),
            ({"static_argnums": iter((1,))}, positional, True),
            ({"static_argnames": iter(("constant",))}, positional, True),
        ]
        for options, call, is_static in cases:
            with self.subTest(options=options, call=call.__name__):
                self.assert_traced_once(_positional_adder, options, call, _static_array(), is_static=is_static)
        with self.subTest("negative index in a variadic call"):
            self.assert_traced_once(_variadic_adder, {"static_argnums": -1},
                                    lambda wrapped, static: wrapped(2.0, 4.0, static), _static_array(), expected=9.0)
        with self.subTest("keyword-only static subtree is not copied"):
            self.assert_traced_once(_keyword_only_adder, {"static_argnames": ("options",)},
                                    lambda wrapped, static: wrapped(2.0, options=static), (_static_array(),))

    def test_jax_input_donation_is_preserved(self):
        """donate_argnums still donates a JAX array input, which is deleted after the call."""
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(lambda value: value + 1, donate_argnums=0)
        value = jax.device_put(np.ones(3), jax.devices("cpu")[0])  # noqa: TID251 test places on a chosen device
        np.testing.assert_array_equal(wrapped(value), np.full(3, 2.0))
        self.assertTrue(value.is_deleted())


if __name__ == "__main__":
    unittest.main()
