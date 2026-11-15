"""The process contract of ``simsopt_jax.backend``: ``set_backend``, its XLA flags, devices and caches, the
host-transfer guards, the copying ``jit`` and the JAX version gate."""

from __future__ import annotations

import dataclasses
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock, skipIf
import warnings

import numpy as np
from unittest_jax_support import (
    JAX_IMPORT_ERROR,
    JaxTestCase,
    aligned_host_copy,
    allow_host_transfers,
    disallow_host_transfers,
    optional_jax_imports,
    run_fresh_process,
)

with optional_jax_imports():
    import jax
    import jax.extend.backend
    import jax.numpy as jnp
    import simsopt_jax.backend.runtime as runtime
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from simsopt_jax._requirements import _require_jax_version
    from simsopt_jax.backend import BackendConfig, get_backend_config, set_backend
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.backend.runtime import host_array, register_backend_cache_clear
    from simsopt_jax.core import biotsavart
    from simsopt_jax_adapters.field import JaxBiotSavart
_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")
_GPU_FLAGS = (
    "--xla_gpu_experimental_enable_fusion_autotuner=false",
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_exclude_nondeterministic_ops=true",
)
_CPU_PRESET = "--xla_cpu_opt_preset=FAST_COMPILE"
# JAX settings a caller owns; ``set_backend`` must leave them alone.
_CALLER_JAX_SETTINGS = {
    "jax_debug_nans": True,
    "jax_disable_jit": True,
    "jax_transfer_guard": "log",
}


def _cuda_running() -> bool:
    return bool({"cuda", "gpu"} & set(jax.extend.backend.backends()))


def _on_axis_loop_error() -> float:
    """Largest relative error of ``biot_savart_B`` on the axis of a circular loop.

    Against mu0 I R^2 / (2 (R^2 + z^2)^(3/2)); a float32 field misses by about 1e-7.
    """
    radius, current, quadpoints = 1.0, 1.0e5, 64
    phi = 2.0 * np.pi * np.arange(quadpoints) / quadpoints
    gamma = radius * np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=-1)
    # d gamma / d t for the normalized parameter t = phi / (2 pi).
    gammadash = 2.0 * np.pi * radius * np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1)
    heights = np.array([0.0, 0.3, 1.7])
    points = np.stack((np.zeros(3), np.zeros(3), heights), axis=-1)
    B = host_array(biotsavart.biot_savart_B(points, gamma[None], gammadash[None], np.array([current])))
    B_z = 4.0e-7 * np.pi * current * radius**2 / (2.0 * (radius**2 + heights**2) ** 1.5)
    assert B.dtype == np.float64, B.dtype
    return float(np.max(np.abs(B[:, 2] / B_z - 1.0)))


def _field_B_devices() -> set:
    """Devices of B of a fresh two-coil ``JaxBiotSavart`` at two off-axis points."""
    curves = create_equally_spaced_curves(2, 1, stellsym=False, R0=1.0, R1=0.5, order=2, numquadpoints=16)
    field = JaxBiotSavart([Coil(curve, Current(1.0e5)) for curve in curves])
    field.set_points(np.array([[1.1, 0.0, 0.1], [0.9, 0.2, -0.1]]))
    return field.B().devices()


def _fake_jax(*, default_backend, selected, devices_by_platform):
    """A started JAX with the given default backend, ``jax_default_device`` and platforms."""
    return SimpleNamespace(
        config=SimpleNamespace(jax_default_device=selected),
        default_backend=lambda: default_backend,
        devices=lambda platform: devices_by_platform[platform],
    )


def _run_startup(scenario: str, **environment: str):
    """Run ``fresh_process/jax_startup.py SCENARIO`` without ``JAX_PLATFORMS`` or ``XLA_FLAGS``."""
    env = {name: value for name, value in os.environ.items() if name not in ("JAX_PLATFORMS", "XLA_FLAGS")}
    return run_fresh_process(_FRESH_PROCESS / "jax_startup.py", scenario, env={**env, **environment}, timeout=300)


class TestBackend(JaxTestCase):
    def test_set_backend_takes_native_or_jax_on_a_device(self):
        """Native, or jax on cpu or gpu, returns the frozen config that ``get_backend_config`` returns; others raise."""
        self.assertEqual([field.name for field in dataclasses.fields(BackendConfig)], ["backend", "device"])
        native = set_backend("native")
        self.assertEqual(native, BackendConfig("native"))
        self.assertIs(get_backend_config(), native)
        config = set_backend("jax", device="cpu")
        self.assertEqual((config.backend, config.device), ("jax", "cpu"))
        self.assertIs(get_backend_config(), config)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.device = "gpu"  # pyright: ignore[reportAttributeAccessIssue]
        for backend, device in (("jax", None), ("native", "cpu"), ("jax", "tpu")):
            with self.subTest(backend=backend, device=device), self.assertRaisesRegex(ValueError, "not valid"):
                set_backend(backend, device=device)  # pyright: ignore[reportArgumentType]
        with self.assertRaises(TypeError):
            set_backend("jax", device="cpu", intent="fast")  # pyright: ignore[reportCallIssue]
        self.assertIs(get_backend_config(), config)

    def test_jax_backend_computes_in_float64_at_the_highest_matmul_precision(self):
        """Selecting the JAX backend restores x64 and the highest matmul precision; the kernel matches a loop field."""
        self.patches.callback(
            jax.config.update, "jax_default_matmul_precision", jax.config.values["jax_default_matmul_precision"]
        )
        jax.config.update("jax_enable_x64", False)
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        set_backend("jax", device="cpu")
        self.assertIs(jax.config.values["jax_enable_x64"], True)
        self.assertEqual(jax.config.values["jax_default_matmul_precision"], "highest")
        self.assertLess(_on_axis_loop_error(), 1e-13)

    def test_public_kernels_use_one_fixed_tiling(self):
        """Every backend tiles the public kernels at 64 coils, 64 quadrature nodes and 1024 points."""
        sizes = (biotsavart.COIL_CHUNK_SIZE, biotsavart.QUADRATURE_BLOCK_SIZE, biotsavart.POINT_CHUNK_SIZE)
        self.assertEqual(sizes, (64, 64, 1024))
        B, value = biotsavart._Integrand.B, biotsavart._DiffMode.VALUE
        for config in (BackendConfig("native"), BackendConfig("jax", "cpu")):
            with self.subTest(config=config):
                set_backend(config.backend, device=config.device)
                self.assertIs(biotsavart._get_kernel(B, value), biotsavart._make_kernel(B, value, *sizes))
                self.assertIs(biotsavart._get_B_vjp_kernel(), biotsavart._make_B_vjp_kernel(*sizes))

    def test_every_call_clears_the_registered_caches(self):
        """Each call runs every registered cache-clear callback once it installed its configuration."""
        self.patches.enter_context(mock.patch.dict(runtime._cache_clear_callbacks))
        seen = []
        register_backend_cache_clear(lambda: seen.append(get_backend_config()))
        B, value = biotsavart._Integrand.B, biotsavart._DiffMode.VALUE
        kernel = biotsavart._get_kernel(B, value)
        set_backend("jax", device="cpu")
        set_backend("native")
        self.assertEqual(seen, [BackendConfig("jax", "cpu"), BackendConfig("native")])
        self.assertIsNot(biotsavart._get_kernel(B, value), kernel)

    def test_selected_platform_becomes_the_default_device(self):
        """A selection on the platform is kept, a stale one cleared, else the platform's first device is selected."""
        cpu, gpu = SimpleNamespace(platform="cpu"), SimpleNamespace(platform="gpu")
        rows = [
            ("default matches", "cpu", "cpu", None, None),
            ("selection matches", "cpu", "gpu", cpu, cpu),
            ("selection by name", "gpu", "cpu", "gpu", "gpu"),
            ("stale selection", "gpu", "gpu", cpu, None),
            ("other default, cpu", "cpu", "gpu", None, cpu),
            ("other default, cuda", "gpu", "cpu", None, gpu),
        ]
        for name, device, default_backend, selected, expected in rows:
            with self.subTest(name), self.case() as patches:
                fake = _fake_jax(
                    default_backend=default_backend, selected=selected, devices_by_platform={"cuda": [gpu], "cpu": [cpu]}
                )
                patches.enter_context(mock.patch.object(runtime, "jax", fake))
                platform = "cpu" if device == "cpu" else "cuda"
                self.assertIs(runtime._default_device(device, platform), expected)

    def test_set_backend_moves_new_arrays_and_fields_between_running_platforms(self):
        """With CUDA and CPU running, each selection makes its platform's first device the default."""
        if "cpu" not in jax.extend.backend.backends() or not _cuda_running():
            self.skipTest("CUDA and CPU JAX backends required for platform selection")
        gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
        set_backend("jax", device="cpu")
        self.assertEqual(jax.config.values["jax_default_device"], cpu)
        self.assertEqual((jnp.zeros(3).devices(), _field_B_devices()), ({cpu}, {cpu}))
        set_backend("jax", device="gpu")
        self.assertIsNone(jax.config.values["jax_default_device"])
        self.assertEqual((jnp.zeros(3).devices(), _field_B_devices()), ({gpu}, {gpu}))

    def test_xla_flag_defaults_keep_the_callers_values(self):
        """Each default joins ``XLA_FLAGS`` unless the caller set its flag (a longer lookalike does not count)."""
        rows = [
            ("gpu", None, " ".join(_GPU_FLAGS)),
            ("gpu", "--a=1 --xla_gpu_autotune_level=4", " ".join(("--a=1 --xla_gpu_autotune_level=4", *_GPU_FLAGS[::2]))),
            ("gpu", "--xla_gpu_exclude_nondeterministic_ops=false", " ".join(("--xla_gpu_exclude_nondeterministic_ops=false", *_GPU_FLAGS[:2]))),
            ("gpu", "--xla_gpu_autotune_level_extra=1", " ".join(("--xla_gpu_autotune_level_extra=1", *_GPU_FLAGS))),
            ("cpu", "  ", _CPU_PRESET),
            ("cpu", "--xla_cpu_opt_preset=DEFAULT", "--xla_cpu_opt_preset=DEFAULT"),
            ("cpu", "--xla_cpu_opt_preset", "--xla_cpu_opt_preset"),
        ]
        for device, caller, expected in rows:
            with self.subTest(device=device, caller=caller), mock.patch.dict(os.environ):
                os.environ.pop("XLA_FLAGS", None)
                if caller is not None:
                    os.environ["XLA_FLAGS"] = caller
                runtime._export_xla_flag_defaults(device)
                runtime._export_xla_flag_defaults(device)
                self.assertEqual(os.environ["XLA_FLAGS"], expected)

    def test_set_backend_exports_platforms_and_flags_before_jax_starts(self):
        """In fresh processes the CPU preset and GPU flags join the caller's ``XLA_FLAGS``; a missing GPU raises."""
        rows = [
            ("cpu_flags", {"CUDA_VISIBLE_DEVICES": "", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}),
            ("gpu_flags", {"XLA_FLAGS": "--xla_gpu_autotune_level=2"}),
            ("gpu_missing", {"CUDA_VISIBLE_DEVICES": ""}),
        ]
        for scenario, environment in rows:
            with self.subTest(scenario):
                if scenario == "gpu_flags" and not _cuda_running():
                    self.skipTest("CUDA JAX device required")
                completed = _run_startup(scenario, **environment)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_missing_platform_raises_and_keeps_the_configuration(self):
        """A device whose platform JAX did not start raises; the previous configuration stays."""
        previous = set_backend("native")
        self.patches.enter_context(mock.patch.object(jax.extend.backend, "backends", lambda: {"cpu": None}))
        with self.assertRaisesRegex(RuntimeError, r"needs JAX platform 'cuda', but JAX started only \['cpu'\]"):
            set_backend("jax", device="gpu")
        self.assertIs(get_backend_config(), previous)

    def test_gpu_flags_after_jax_started_warn_at_the_caller(self):
        """Once JAX started, missing GPU flags warn at the caller's line; a missing CPU preset does not warn."""
        os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            set_backend("jax", device="cpu")
        missing = r"cannot add --xla_gpu_experimental_enable_fusion_autotuner=false --xla_gpu_exclude_nondeterministic_ops=true"
        without_cuda = nullcontext() if _cuda_running() else self.assertRaises(RuntimeError)
        with self.assertWarnsRegex(RuntimeWarning, missing) as caught, without_cuda:
            set_backend("jax", device="gpu")
        self.assertEqual(caught.filename, __file__)
        self.assertEqual(os.environ["XLA_FLAGS"], "--xla_gpu_autotune_level=0")

    def test_set_backend_keeps_the_callers_jax_settings(self):
        """``set_backend`` leaves the JAX settings a caller owns as they are."""
        cache_dir = self.patches.enter_context(TemporaryDirectory())
        caller = {**_CALLER_JAX_SETTINGS, "jax_compilation_cache_dir": cache_dir}
        for name, value in caller.items():
            jax.config.update(name, value)
        self.assertEqual(get_backend_config(), BackendConfig("native"))
        self.assertEqual(set_backend("jax", device="cpu"), BackendConfig("jax", "cpu"))
        self.assertEqual({name: jax.config.values[name] for name in caller}, caller)

    def test_transfer_guard_scopes_refuse_and_permit_implicit_transfers(self):
        """``disallow_host_transfers`` refuses implicit transfers; ``allow_host_transfers`` permits its block."""
        device_array = jnp.asarray([1.0, 2.0])
        host_values = np.asarray([3.0, 4.0])
        with disallow_host_transfers():
            with self.assertRaises(jax.errors.JaxRuntimeError):
                _ = device_array + host_values
            with allow_host_transfers():
                permitted = device_array + host_values
            with self.assertRaises(jax.errors.JaxRuntimeError):
                _ = device_array + host_values
        np.testing.assert_array_equal(host_array(permitted), [4.0, 6.0])

    def test_host_array_reuses_writable_numpy_storage_and_converts_the_rest(self):
        """A writable NumPy array comes back as itself; a read-only array, a NumPy subclass and a device array come
        back as writable NumPy arrays with their values, the read-only one copied."""
        values = np.arange(6.0).reshape(2, 3)
        read_only = values.copy()
        read_only.flags.writeable = False
        device = jnp.asarray(values)
        with disallow_host_transfers():
            self.assertIs(host_array(values), values)
            arrays = [host_array(value) for value in (read_only, np.asmatrix(values), device)]
        for array in arrays:
            self.assertIs(type(array), np.ndarray)
            self.assertTrue(array.flags.writeable)
            np.testing.assert_array_equal(array, values)
        self.assertFalse(np.shares_memory(arrays[0], read_only))
        self.assertEqual(host_array(values, dtype=np.float32).dtype, np.float32)

    def test_numpy_input_overwritten_after_the_call_keeps_the_result(self):
        """A result held across an in-place overwrite of its NumPy input keeps the old value."""
        if jax.default_backend() == "cpu" and not jax.config.values["jax_cpu_enable_async_dispatch"]:
            self.skipTest("synchronous CPU dispatch: no overwrite can race the call")
        slow = jit(lambda m: jnp.linalg.matrix_power(m, 32))
        eye = jnp.eye(1000)
        passing_styles = (
            ("positional", lambda x, gate: jnp.sum(x * gate[0, 0]), lambda f, x, g: f(x, g)),
            ("list", lambda xs, gate: jnp.sum(xs[0] * gate[0, 0]), lambda f, x, g: f([x], g)),
            ("dict", lambda d, gate: jnp.sum(d["x"] * gate[0, 0]), lambda f, x, g: f({"x": x}, g)),
            ("keyword", lambda gate, x: jnp.sum(x * gate[0, 0]), lambda f, x, g: f(g, x=x)),
        )
        for name, fun, call in passing_styles:
            with self.subTest(style=name):
                f = jit(fun)
                call(f, np.ones(16), eye).block_until_ready()  # compiles f
                for _ in range(5):
                    buffer = aligned_host_copy(np.ones(16))
                    gate = slow(eye)
                    held = call(f, buffer, gate)
                    buffer[:] = 2.0
                    if not gate.is_ready():
                        self.assertEqual(float(held), 16.0)
                        break
                else:
                    self.fail("the gate finished before every overwrite; race not probed")


@skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
class TestOptionalJax(TestCase):

    def test_version_gate_accepts_every_0_12_release_and_rejects_0_11(self):
        """The floor is the 0.12 series: four-part and local releases pass, 0.11 fails."""
        for version in ("0.12.0", "0.12.0.1", "0.12.1", "0.12.0.1+cuda12", "0.13.0", "1.0.0"):
            with self.subTest(version=version):
                _require_jax_version("jax", version)
        for version in ("0.11.2", "0.11.9.1", "0.10.0"):
            with self.subTest(version=version), self.assertRaisesRegex(
                ImportError, "requires jax >= 0.12; found " + version.replace(".", r"\.")
            ):
                _require_jax_version("jax", version)
