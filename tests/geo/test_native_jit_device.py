"""Native ``simsopt.geo.jit`` input ownership and the JAX devices native geometry uses."""

import functools
import os
import subprocess
import sys
import unittest
from collections import OrderedDict, UserDict, defaultdict, namedtuple
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from unittest_jax_support import JaxTestCase, aligned_host_copy, optional_jax_imports, require_cuda

with optional_jax_imports():
    from simsopt_jax.backend import set_backend

import jax
import jax.numpy as jnp
import numpy as np
import simsopt.geo.jit as jit_module
from numpy.testing import assert_allclose, assert_array_equal
from simsopt.geo.config import parameters
from simsopt.geo.jit import jit

Arrays = namedtuple("Arrays", ["first", "rest"])


def _jit_flag(use_jit):
    """Set the native jit flag to ``use_jit`` while the returned context is open."""
    return mock.patch.dict(parameters, {"jit": use_jit})


def _wrap(fun, use_jit, **options):
    """``jit(fun, **options)`` built with the native jit flag set to ``use_jit``."""
    with _jit_flag(use_jit):
        return jit(fun, **options)


class HashableArray(np.ndarray):
    """A deliberately hashable ndarray, to pass as a JAX static argument."""

    __eq__ = object.__eq__

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
    """Input ownership, JAX transformations and device routing of the native jit."""

    def test_no_jit_snapshots_nested_inputs_and_retains_plain_execution(self):
        """With jit off, bare and nested arrays are copied, the call runs on the native device, and the name is kept."""

        def record(x, *, nested):
            return x, nested

        wrapped = _wrap(record, False)
        x, y = np.ones(3), np.ones(4)
        device = object()
        native = mock.patch.object(jit_module, "native_jax_device", return_value=device)
        with native, mock.patch.object(jit_module.jax, "default_device") as context:
            held, nested = wrapped(x, nested={"arrays": [y]})
        x[:] = 4
        y[:] = 7
        assert_array_equal(held, np.ones(3))
        assert_array_equal(nested["arrays"][0], np.ones(4))
        context.assert_called_once_with(device)
        self.assertEqual(wrapped.__name__, record.__name__)

    def test_compiled_dispatch_receives_owned_inputs_under_native_device_context(self):
        """Both resolved selectors reach jax.jit, so no JAX version infers either; a negative index names its parameter
        and counts from the signature."""
        cases = (
            ((1,), lambda wrapped, x, static: wrapped(x, static)),
            (-1, lambda wrapped, x, static: wrapped(x, marker=static)),
        )
        for argnums, call in cases:
            with self.subTest(static_argnums=argnums):
                device_sentinel, static_sentinel, x = object(), object(), np.ones(4)
                fake_compile = mock.Mock(side_effect=lambda fun, **kwargs: fun)
                native = mock.Mock(return_value=device_sentinel)
                context = mock.Mock(side_effect=lambda device: nullcontext())
                with mock.patch.multiple(jit_module, jaxjit=fake_compile, native_jax_device=native):
                    with mock.patch.object(jit_module.jax, "default_device", context):
                        wrapped = _wrap(lambda a, marker: (a, marker), True, static_argnums=argnums)
                        held, marker = call(wrapped, x, static_sentinel)
                x[:] = 5
                assert_array_equal(held, np.ones(4))
                self.assertIs(marker, static_sentinel)
                context.assert_called_once_with(device_sentinel)
                self.assertEqual(fake_compile.call_args.kwargs,
                                 {"static_argnums": (1,), "static_argnames": ("marker",)})

    def test_real_jax_outputs_keep_call_time_values_after_input_mutation(self):
        """Even ready identity outputs of bare, nested and keyword arrays own their values."""

        def identities(values, bundle, *, other):
            return jnp.asarray(values), jnp.asarray(bundle[0]["arrays"][0]), jnp.asarray(other["value"])

        for use_jit in (True, False):
            with self.subTest(jit=use_jit), _jit_flag(use_jit):
                wrapped = jit_module.jit(identities)
                inputs = [aligned_host_copy(np.arange(1.0, n + 1.0)) for n in (16, 8, 16)]
                expected = [values.copy() for values in inputs]
                held = jax.block_until_ready(wrapped(inputs[0], [{"arrays": (inputs[1],)}], other={"value": inputs[2]}))
                for values in inputs:
                    values[:] = -7.0
                for actual, values in zip(held, expected):
                    assert_array_equal(actual, values)
                self.assertEqual(held[0].devices(), {jit_module.native_jax_device()})

    def test_real_jax_static_arguments_and_wraps_metadata_are_preserved(self):
        """Positional and keyword statics compile on the native device, and the wrapper keeps the function metadata."""

        def powers(values, order, *, scale):
            """Scaled powers with positional and keyword static parameters."""
            return scale * jnp.sum(values**order)

        wrapped = _wrap(powers, True, static_argnums=(1,), static_argnames=("scale",))
        result = wrapped(np.array([1.0, 2.0, 3.0]), 3, scale=0.5)
        self.assertEqual(float(result), 18.0)
        self.assertEqual(result.devices(), {jit_module.native_jax_device()})
        self.assertIs(wrapped.__wrapped__, powers)
        self.assertEqual(wrapped.__name__, powers.__name__)
        self.assertEqual(wrapped.__doc__, powers.__doc__)

    def test_tracers_grad_hessian_nested_jit_and_vmap_preserve_derivatives(self):
        """grad, Hessian, raw jax.jit and vmap through the wrapper match analytic derivatives, with jit on and off."""
        values = np.linspace(0.25, 1.0, 4)
        batch = np.stack([values, 2 * values])
        close = functools.partial(assert_allclose, rtol=1e-14, atol=0)
        for use_jit in (True, False):
            with self.subTest(jit=use_jit), _jit_flag(use_jit):
                wrapped = jit_module.jit(lambda x: jnp.sum(x**3))
                gradient = jax.grad(wrapped)
                close(gradient(values), 3 * values**2)
                close(jax.jacfwd(gradient)(values), np.diag(6 * values))
                nested = jax.jit(gradient)  # noqa: TID251 checks composition with raw jax.jit
                close(nested(values), 3 * values**2)
                close(jax.vmap(wrapped)(batch), np.sum(batch**3, axis=1))

    def test_plain_builtin_mappings_preserve_key_order_and_factory(self):
        """With jit off, copied dict, OrderedDict and defaultdict inputs keep their type, key order and factory."""
        mappings: list[dict[object, np.ndarray | list[float]]] = [
            {1: np.array([2.0]), "a": np.array([3.0])},
            {"z": np.array([2.0]), "a": np.array([3.0])},
            OrderedDict([("z", np.array([2.0])), ("a", np.array([3.0]))]),
            defaultdict(list, {1: np.array([2.0]), "a": np.array([3.0])}),
            defaultdict(list, {"z": np.array([2.0]), "a": np.array([3.0])}),
        ]
        identity = _wrap(lambda values: values, False)
        for mapping in mappings:
            with self.subTest(kind=type(mapping), keys=tuple(mapping)):
                copied = identity(Arrays([mapping], None)).first[0]
                self.assertIs(type(copied), type(mapping))
                self.assertEqual(tuple(copied), tuple(mapping))
                for key in mapping:
                    self.assertFalse(np.shares_memory(copied[key], mapping[key]))
                    assert_array_equal(copied[key], mapping[key])
        self.assertIs(identity(mappings[-1]).default_factory, list)

    def test_explicit_statics_without_signature(self):
        """Explicit static selectors compile builtins such as getattr and min, which have no signature."""
        attribute = _wrap(getattr, True, static_argnums=1)
        minimum = _wrap(min, True, static_argnums=(0, 1))
        self.assertEqual(tuple(int(item) for item in attribute(jnp.array([3.0, 4.0]), "shape")), (2,))
        self.assertEqual(int(minimum(3, 2)), 2)

    def test_unregistered_dictionaries_are_opaque_leaves_without_hooks(self):
        """With jit off, dictionaries JAX does not register are returned as is and their item setters never run."""
        wrapped = _wrap(lambda values: values, False)
        for kind in (FrozenDictionary, TransformDictionary, UserDict):
            with self.subTest(kind=kind.__name__):
                incoming = kind({"entry": np.array([3.0])})
                self.assertIs(wrapped(incoming), incoming)
                self.assertIs(wrapped(incoming)["entry"], incoming["entry"])
                assert_array_equal(wrapped(incoming)["entry"], [3.0])
                self.assertEqual(getattr(incoming, "setter_calls", 0), 0)

    def test_registered_custom_pytree_snapshots_dynamic_numpy_leaves(self):
        """NumPy leaves of a registered custom pytree are copied at call time, with jit on and off."""
        for use_jit in (False, True):
            with self.subTest(jit=use_jit), _jit_flag(use_jit):
                wrapped = jit(lambda container: jnp.asarray(container.values))
                incoming = RegisteredArrays(aligned_host_copy(np.arange(1.0, 17.0)))
                expected = incoming.values.copy()
                held = wrapped(incoming).block_until_ready()
                incoming.values[:] = -7
                assert_array_equal(held, expected)

    def assert_traced_once(self, make, options, call, static, expected=5.0, is_static=True):
        """Three calls trace once; ``is_static`` says whether tracing saw ``static`` itself."""
        traced = []
        wrapped = _wrap(make(traced), True, **options)
        for _ in range(3):
            self.assertEqual(float(call(wrapped, static)), expected)
        self.assertEqual(len(traced), 1)
        self.assertEqual(traced[0] is static, is_static)

    def test_static_selectors_keep_identity_cache_and_actual_call_positioning(self):
        """Static NumPy objects keep identity and cache hits; explicit selectors do not infer the other selector or bind
        keyword calls."""
        one = (1, (1,), np.int64(1), np.array(1, dtype=np.int64), IndexSelector(), -1)
        inferred: list[dict[str, object]] = [{"static_argnums": n} for n in one]
        inferred.append({"static_argnames": "constant"})
        argnums_only = {"static_argnums": 1, "static_argnames": ()}
        argnames_only = {"static_argnums": (), "static_argnames": "constant"}
        cases = [*((options, call, True) for options in inferred for call in (positional, keyword)),
                 (argnums_only, positional, True), (argnums_only, keyword, False), (argnames_only, positional, False),
                 (argnames_only, keyword, True), ({"static_argnums": iter((1,))}, positional, True),
                 ({"static_argnames": iter(("constant",))}, positional, True)]
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
        wrapped = _wrap(lambda value: value + 1, True, donate_argnums=0)
        value = jax.device_put(np.ones(3), jax.devices("cpu")[0])  # noqa: TID251 test places on a chosen device
        assert_array_equal(wrapped(value), np.full(3, 2.0))
        self.assertTrue(value.is_deleted())


_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")
_STARTUP, _IMPORT = "set_backend_startup.py", "native_import.py"


def _run_fresh_process(script, *arguments, cuda=False, env: Mapping[str, str | None] = {}):
    """Run ``fresh_process/<script>``, which fails on its asserts, in a new interpreter without the parent's ``JAX_*``
    variables but with each ``env`` value not None; ``cuda`` first skips the case without a CUDA device."""
    if cuda:
        require_cuda("CUDA JAX device required")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("JAX_")}
    environment.update((k, v) for k, v in env.items() if v is not None)
    command = [sys.executable, str(_FRESH_PROCESS / script), *arguments]
    subprocess.run(command, env=environment, check=True)


class TestNativeJaxDevice(JaxTestCase):
    """Backend validation and native geometry next to ``set_backend``, in fresh processes because JAX picks its
    platforms once, when it starts."""

    def test_unknown_backend_is_rejected(self):
        """An unknown backend is rejected, naming the accepted calls."""
        message = r"set_backend\('jax_cpu_unknown', device=None\) is not valid; use set_backend\('native'\)"
        with self.assertRaisesRegex(ValueError, message):
            # The name is deliberately outside the accepted backend values.
            set_backend("jax_cpu_unknown")  # pyright: ignore[reportArgumentType]

    def test_native_import_preserves_platform_selection_in_fresh_process(self):
        """Importing native geometry respects explicit CPU platform selection and enables double precision."""
        for platform_variable in [None, "JAX_PLATFORMS", "JAX_PLATFORM_NAME"]:
            with self.subTest(platform_variable=platform_variable), self.case():
                env = {platform_variable: "cpu"} if platform_variable else {}
                _run_fresh_process(_IMPORT, "platform_selection", env=env)

    def test_native_import_enables_x64_despite_explicit_disable_in_fresh_process(self):
        """Importing native geometry restores double precision even when the environment explicitly disables it."""
        for x64_setting in ["false", "0"]:
            with self.subTest(x64_setting=x64_setting), self.case():
                _run_fresh_process(_IMPORT, "x64", env={"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": x64_setting})

    def test_native_jax_curve_constructed_before_set_backend_in_fresh_process(self):
        """A helical curve constructed before backend setup still yields finite float64 length and field values."""
        _run_fresh_process(_STARTUP, "curve_before_set_backend")

    def test_set_backend_default_platform_after_native_import_in_fresh_process(self):
        """``set_backend`` clears the ``jax_platform_name="cpu"`` that ``simsopt.geo`` sets without JAX platform
        variables and makes its platform the default; the CUDA rows are the fixed failures ``Unknown backend cpu``
        and CPU while CUDA was requested."""
        for device, after in [("cpu", None), ("gpu", None), ("gpu", "cuda,cpu")]:
            with self.subTest(device=device, platforms_after_import=after), self.case():
                _run_fresh_process(_STARTUP, "after_native_import", device, after or "", cuda=device == "gpu")

    def test_native_stage_two_with_explicit_platforms_in_fresh_process(self):
        """Native stage II after ``set_backend``; ``gpu-no-platform-env`` exports no JAX platform variable, so
        ``simsopt.geo`` sets ``cpu`` before ``set_backend``."""
        for case_id_0, device, jax_platforms in [("cpu", "cpu", "cpu"), ("gpu-cuda-cpu-platforms", "gpu", "cuda,cpu"),
                                                 ("gpu-no-platform-env", "gpu", None)]:
            with self.subTest(case_id_0=case_id_0, device=device, jax_platforms=jax_platforms), self.case():
                _run_fresh_process(_STARTUP, "stage_two", device, jax_platforms or "", cuda=device == "gpu",
                                   env={"JAX_PLATFORMS": jax_platforms})

    def test_gpu_selected_at_startup_also_starts_the_cpu_in_fresh_process(self):
        """Without ``JAX_PLATFORMS``, a GPU selection starts CUDA as default and the CPU for native kernels."""
        _run_fresh_process(_STARTUP, "gpu_at_startup", cuda=True)

    def test_selections_after_initialization_keep_the_platforms_in_fresh_process(self):
        """Under ``cuda,cpu``, CPU-then-GPU selections after initialization keep native kernels and platforms."""
        _run_fresh_process(_STARTUP, "cpu_then_gpu_after_initialization", cuda=True, env={"JAX_PLATFORMS": "cuda,cpu"})

    def test_explicit_cuda_platforms_start_cuda_alone_in_fresh_process(self):
        """An explicit ``JAX_PLATFORMS=cuda`` starts only CUDA, and native kernels fall back to it."""
        _run_fresh_process(_STARTUP, "explicit_cuda_only", cuda=True, env={"JAX_PLATFORMS": "cuda"})

    def test_cpu_selected_before_initialization_keeps_cuda_off_in_fresh_process(self):
        """A CPU selection before initialization starts only the CPU, also under ``cuda,cpu``."""
        for jax_platforms in (None, "cuda,cpu"):
            with self.subTest(jax_platforms=jax_platforms), self.case():
                _run_fresh_process(_STARTUP, "cpu_before_initialization", env={"JAX_PLATFORMS": jax_platforms})

    def test_native_jit_runs_on_cpu_with_cuda_default_in_fresh_process(self):
        """Under ``JAX_PLATFORMS=cuda,cpu`` native kernels return CPU arrays, also with jit disabled
        (``SIMSGEOJIT=0``)."""
        require_cuda("CUDA JAX device required")
        for use_jit in ("1", "0"):
            with self.subTest(SIMSGEOJIT=use_jit):
                env = {"JAX_PLATFORMS": "cuda,cpu", "SIMSGEOJIT": use_jit}
                _run_fresh_process(_IMPORT, "jit_device_policy", env=env)


if __name__ == "__main__":
    unittest.main()
