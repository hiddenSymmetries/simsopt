"""Regression tests for the JAX runtime-device and field-tiling contracts.

1. ``test_runtime_jax_device_*`` — the runtime device follows the installed
   policy, else the platforms JAX was configured with, and propagates JAX's own
   errors.
2. ``test_field_kernel_tuning_*`` — GPU modes use their static tiling without
   probing devices or external tools; environment overrides still apply.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import simsopt_jax.backend.runtime as runtime_module
    from simsopt_jax.backend.runtime import (
        BackendPolicy,
        FieldKernelTuning,
        _config_from_mode,
        _policy_from_config,
        get_field_kernel_tuning,
        get_runtime_jax_device,
    )
from unittest import mock
import os


import subprocess
import sys
import types


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
