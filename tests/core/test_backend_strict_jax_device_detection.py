"""Regression tests for the JAX runtime-device and field-tiling contracts.

1. ``test_runtime_jax_device_*`` — the runtime device follows the installed
   policy, else the platforms JAX was configured with, and propagates JAX's own
   errors.
2. ``test_field_kernel_tuning_*`` — GPU modes use their static tiling without
   probing devices or external tools; environment overrides still apply.
3. ``test_set_backend_*`` / ``test_default_device_*`` — in an initialized
   process ``set_backend(device=...)`` makes an initialized platform JAX's
   default device, and still warns (or raises) for one JAX never initialized.
"""

from __future__ import annotations

from unittest_jax_support import (
    JaxTestCase,
    jax_runtime_isolation,
    optional_jax_imports,
)

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.extend.backend
    import jax.numpy as jnp
    import simsopt_jax.backend.runtime as runtime_module
    from simsopt_jax.backend.runtime import (
        BackendPolicy,
        FieldKernelTuning,
        _config_from_mode,
        _default_device,
        _expected_runtime_backend_names,
        _initialized_runtime_platform,
        _policy_from_config,
        get_field_kernel_tuning,
        get_runtime_jax_device,
        set_backend,
    )
    from simsopt_jax_adapters.field import JaxBiotSavart
from unittest import mock
import os


import subprocess
import sys
import types
import warnings

import numpy as np

from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves


def _policy_for_mode(mode: str) -> BackendPolicy:
    """Return the canonical :class:`BackendPolicy` for ``mode``.

    Uses the same pipeline as ``get_backend_policy(mode)`` without touching
    the cached module-level state; this lets tests build orthogonal policies
    without coupling to ``set_backend`` side effects.
    """
    return _policy_from_config(_config_from_mode(mode, strict=False))


def _fake_jax(jax_platforms, local_devices):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(jax_platforms=jax_platforms),
        local_devices=local_devices,
    )


class _FakeDevice:
    def __init__(self, platform: str) -> None:
        self.platform = platform


class _FakeJaxConfig:
    """Records ``jax_default_device`` updates the way ``jax.config`` applies them."""

    def __init__(self, jax_default_device) -> None:
        self.jax_default_device = jax_default_device
        self.updates: list[tuple[str, object]] = []

    def update(self, name: str, value) -> None:
        self.updates.append((name, value))
        setattr(self, name, value)


def _fake_initialized_jax(*, default_backend, selected, devices_by_platform):
    """An initialized JAX with the given default backend, selection and platforms."""
    return types.SimpleNamespace(
        config=_FakeJaxConfig(selected),
        default_backend=lambda: default_backend,
        extend=types.SimpleNamespace(
            backend=types.SimpleNamespace(backends=lambda: dict(devices_by_platform))
        ),
        devices=lambda platform: devices_by_platform[platform],
    )


def _jax_biotsavart_B() -> jax.Array:
    """B of a fresh two-coil ``JaxBiotSavart`` at two off-axis points."""
    curves = create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.5, order=2, numquadpoints=16
    )
    field = JaxBiotSavart([Coil(curve, Current(1.0e5)) for curve in curves])
    field.set_points(np.array([[1.1, 0.0, 0.1], [0.9, 0.2, -0.1]]))
    return field.B()


class TestBackendStrictJaxDeviceDetection(JaxTestCase):
    def test_field_kernel_tuning_uses_static_gpu_tiling_without_probes(self):
        """Tiling is a pure function of the mode: no device or ``nvidia-smi`` probe."""
        for mode, expected in [
            ("jax_gpu_parity", (16, 0, 256)),
            ("jax_gpu_fast", (64, 64, 1024)),
        ]:
            with self.subTest(mode=mode, expected=expected), self.case() as patches:
                self._case_field_kernel_tuning_uses_static_gpu_tiling_without_probes(
                    mode, expected, patches
                )

    def _case_field_kernel_tuning_uses_static_gpu_tiling_without_probes(
        self, mode, expected, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""

        def _no_subprocess(*args, **kwargs):
            raise AssertionError(f"unexpected external command: {args!r}")

        def _no_device_lookup(*args, **kwargs):
            raise AssertionError("field tiling must not look up JAX devices")

        patches.enter_context(mock.patch.object(subprocess, "run", _no_subprocess))
        patches.enter_context(
            mock.patch.object(runtime_module.jax, "local_devices", _no_device_lookup)
        )
        patches.enter_context(
            mock.patch.object(runtime_module.jax, "devices", _no_device_lookup)
        )

        tuning = get_field_kernel_tuning(mode)

        self.assertTrue(
            isinstance(tuning, FieldKernelTuning),
            "isinstance(tuning, FieldKernelTuning)",
        )
        self.assertTrue(
            (
                tuning.coil_chunk_size,
                tuning.quadrature_block_size,
                tuning.point_chunk_size,
            )
            == expected,
            "( tuning.coil_chunk_size, tuning.quadrature_block_size, tuning.point_chunk_size, ) == expected",
        )

    def test_field_kernel_tuning_applies_environment_overrides(self):
        """Environment overrides determine the coil, quadrature and point tile sizes."""
        patches = self.patches
        patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_COIL_CHUNK_SIZE": "8"})
        )
        patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE": "32"})
        )
        patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_POINT_CHUNK_SIZE": "128"})
        )

        tuning = get_field_kernel_tuning("jax_gpu_fast")

        self.assertTrue(
            (
                tuning.coil_chunk_size,
                tuning.quadrature_block_size,
                tuning.point_chunk_size,
            )
            == (8, 32, 128),
            "( tuning.coil_chunk_size, tuning.quadrature_block_size, tuning.point_chunk_size, ) == (8, 32, 128)",
        )

    def test_runtime_jax_device_uses_primary_configured_jax_platform_before_policy(
        self,
    ):
        """Without a JAX policy, placement follows the platforms JAX was configured with."""
        patches = self.patches
        runtime_device = object()
        backend_calls: list[str | None] = []

        def _local_devices(*, backend=None):
            backend_calls.append(backend)
            return [runtime_device]

        patches.enter_context(
            mock.patch.object(
                runtime_module,
                "get_backend_policy",
                lambda mode=None: _policy_for_mode("native_cpu"),
            )
        )
        fake_jax = _fake_jax("cuda", _local_devices)
        patches.enter_context(mock.patch.object(runtime_module, "jax", fake_jax))
        patches.enter_context(mock.patch.dict(sys.modules, {"jax": fake_jax}))

        self.assertTrue(
            get_runtime_jax_device() is runtime_device,
            "get_runtime_jax_device() is runtime_device",
        )
        self.assertTrue(backend_calls == ["gpu"], 'backend_calls == ["gpu"]')

    def test_runtime_jax_device_ignores_jax_platforms_env_rewritten_after_jax_import(
        self,
    ):
        """``set_backend("native_cpu")`` writes ``JAX_PLATFORMS=cpu`` for children.

        The running JAX keeps the platforms it was configured with, so the runtime
        device must too; following the rewritten variable placed arrays on the CPU
        while every default-placed array stayed on the GPU.
        """
        patches = self.patches
        runtime_device = object()
        backend_calls: list[str | None] = []

        def _local_devices(*, backend=None):
            backend_calls.append(backend)
            return [runtime_device]

        patches.enter_context(mock.patch.dict(os.environ, {"JAX_PLATFORMS": "cpu"}))
        patches.enter_context(
            mock.patch.object(
                runtime_module,
                "get_backend_policy",
                lambda mode=None: _policy_for_mode("native_cpu"),
            )
        )
        fake_jax = _fake_jax("cuda,cpu", _local_devices)
        patches.enter_context(mock.patch.object(runtime_module, "jax", fake_jax))
        patches.enter_context(mock.patch.dict(sys.modules, {"jax": fake_jax}))

        self.assertTrue(
            get_runtime_jax_device() is runtime_device,
            "get_runtime_jax_device() is runtime_device",
        )
        self.assertTrue(backend_calls == ["gpu"], 'backend_calls == ["gpu"]')

    def test_runtime_jax_device_reads_the_env_while_jax_is_still_importing(self):
        """A ``jax`` module without ``config`` yet is a first import in progress.

        JAX has not read its platforms at that point either, so ``JAX_PLATFORMS``
        decides, exactly as before any import; touching ``config`` would raise.
        """
        patches = self.patches
        runtime_device = object()
        backend_calls: list[str | None] = []

        def _local_devices(*, backend=None):
            backend_calls.append(backend)
            return [runtime_device]

        patches.enter_context(mock.patch.dict(os.environ, {"JAX_PLATFORMS": "cuda"}))
        patches.enter_context(
            mock.patch.object(
                runtime_module,
                "get_backend_policy",
                lambda mode=None: _policy_for_mode("native_cpu"),
            )
        )
        fake_jax = types.SimpleNamespace(local_devices=_local_devices)
        patches.enter_context(mock.patch.object(runtime_module, "jax", fake_jax))
        patches.enter_context(mock.patch.dict(sys.modules, {"jax": fake_jax}))

        self.assertTrue(
            get_runtime_jax_device() is runtime_device,
            "get_runtime_jax_device() is runtime_device",
        )
        self.assertTrue(backend_calls == ["gpu"], 'backend_calls == ["gpu"]')

    def test_runtime_jax_device_prefers_policy_over_jax_platforms_env(self):
        """Once policy is installed, the policy platform remains authoritative."""
        patches = self.patches
        runtime_device = object()
        backend_calls: list[str | None] = []

        def _local_devices(*, backend=None):
            backend_calls.append(backend)
            return [runtime_device]

        patches.enter_context(
            mock.patch.object(
                runtime_module,
                "get_backend_policy",
                lambda mode=None: _policy_for_mode("jax_gpu_parity"),
            )
        )
        fake_jax = _fake_jax("cpu,cuda", _local_devices)
        patches.enter_context(mock.patch.object(runtime_module, "jax", fake_jax))
        patches.enter_context(mock.patch.dict(sys.modules, {"jax": fake_jax}))

        self.assertTrue(
            get_runtime_jax_device() is runtime_device,
            "get_runtime_jax_device() is runtime_device",
        )
        self.assertTrue(backend_calls == ["gpu"], 'backend_calls == ["gpu"]')

    def test_runtime_jax_device_returns_none_without_policy_or_configured_jax_platforms(
        self,
    ):
        """Native startup with no JAX platform request keeps the default placement path."""
        patches = self.patches

        def _local_devices(*, backend=None):
            raise AssertionError(f"no device lookup expected, got backend={backend!r}")

        patches.enter_context(
            mock.patch.object(
                runtime_module,
                "get_backend_policy",
                lambda mode=None: _policy_for_mode("native_cpu"),
            )
        )
        fake_jax = _fake_jax(None, _local_devices)
        patches.enter_context(mock.patch.object(runtime_module, "jax", fake_jax))
        patches.enter_context(mock.patch.dict(sys.modules, {"jax": fake_jax}))

        self.assertTrue(
            get_runtime_jax_device() is None, "get_runtime_jax_device() is None"
        )

    def test_default_device_rules_for_an_initialized_process(self):
        """Keep a matching default, clear a stale selection, else select the initialized platform."""
        cpu, gpu = _FakeDevice("cpu"), _FakeDevice("gpu")
        both = {"cuda": [gpu], "cpu": [cpu]}
        rows = [
            ("default matches", "jax_cpu_fast", "cpu", None, None),
            ("selection matches", "jax_cpu_fast", "gpu", cpu, cpu),
            ("selection by name", "jax_gpu_fast", "cpu", "gpu", "gpu"),
            ("stale selection", "jax_gpu_fast", "gpu", cpu, None),
            ("initialized cpu", "jax_cpu_fast", "gpu", None, cpu),
            ("initialized cuda", "jax_gpu_parity", "cpu", None, gpu),
        ]
        for name, mode, default_backend, selected, expected in rows:
            with self.subTest(name), self.case():
                fake_jax = _fake_initialized_jax(
                    default_backend=default_backend,
                    selected=selected,
                    devices_by_platform=both,
                )
                config = _config_from_mode(mode, strict=True)
                backends = _expected_runtime_backend_names(config.jax_platform)
                platform = "cpu" if mode.startswith("jax_cpu") else "cuda"
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    self.assertEqual(
                        _initialized_runtime_platform(fake_jax, config, backends), platform
                    )
                    device = _default_device(fake_jax, backends, platform)
                self.assertIs(device, expected)
                # Deciding changes no setting.
                self.assertEqual(fake_jax.config.updates, [])

    def test_default_device_for_a_never_initialized_platform_warns_or_raises(self):
        """A platform JAX never initialized warns, and raises in strict and GPU parity modes."""
        rows = [
            ("jax_gpu_fast", False, False),
            ("jax_gpu_fast", True, True),
            ("jax_gpu_parity", False, True),
        ]
        cuda = _expected_runtime_backend_names("cuda")
        for mode, strict, raises in rows:
            with self.subTest(mode=mode, strict=strict), self.case():
                fake_jax = _fake_initialized_jax(
                    default_backend="cpu",
                    selected=None,
                    devices_by_platform={"cpu": [_FakeDevice("cpu")]},
                )
                config = _config_from_mode(mode, strict=strict)
                expected = r"Requested JAX platform 'cuda'.*initialized only \['cpu'\]"
                if raises:
                    with self.assertRaisesRegex(RuntimeError, expected):
                        _initialized_runtime_platform(fake_jax, config, cuda)
                else:
                    with self.assertWarnsRegex(RuntimeWarning, expected):
                        self.assertIsNone(_initialized_runtime_platform(fake_jax, config, cuda))
                self.assertEqual(fake_jax.config.updates, [])

    def test_set_backend_moves_new_arrays_and_fields_between_initialized_platforms(
        self,
    ):
        """With CUDA and CPU initialized, set_backend(device=...) moves arrays and B between them."""
        initialized = jax.extend.backend.backends()
        if "cpu" not in initialized or not {"cuda", "gpu"} & set(initialized):
            self.skipTest("CUDA and CPU JAX backends required for platform selection")
        gpu = jax.devices("gpu")[0]
        cpu = jax.devices("cpu")[0]
        self.assertEqual(jax.default_backend(), "gpu")

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            set_backend("jax", device="cpu")
        self.assertEqual(jax.config.values["jax_default_device"], cpu)
        self.assertEqual(jnp.zeros(3).devices(), {cpu})
        self.assertEqual(_jax_biotsavart_B().devices(), {cpu})

        set_backend("jax", device="gpu")
        self.assertIsNone(jax.config.values["jax_default_device"])
        self.assertEqual(jnp.zeros(3).devices(), {gpu})
        self.assertEqual(_jax_biotsavart_B().devices(), {gpu})

    def test_runtime_isolation_restores_the_default_device(self):
        """``jax_runtime_isolation`` puts ``jax_default_device`` back after a case selects one."""
        with jax_runtime_isolation():
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        self.assertIsNone(jax.config.values["jax_default_device"])
