"""Regression tests for the JAX runtime-device and field-tiling contracts.

1. ``test_runtime_jax_device_*`` — the runtime device follows the installed
   policy, else the platforms JAX was configured with, and propagates JAX's own
   errors.
2. ``test_field_kernel_tuning_*`` — GPU modes use their static tiling without
   probing devices or external tools; environment overrides still apply.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import subprocess
import sys
import types

import pytest

import simsopt_jax.backend.runtime as runtime_module
from simsopt_jax.backend.runtime import (
    BackendPolicy,
    FieldKernelTuning,
    _config_from_mode,
    _policy_from_config,
    get_field_kernel_tuning,
    get_runtime_jax_device,
)


def _policy_for_mode(mode: str) -> BackendPolicy:
    """Return the canonical :class:`BackendPolicy` for ``mode``.

    Uses the same pipeline as ``get_backend_policy(mode)`` without touching
    the cached module-level state; this lets tests build orthogonal policies
    without coupling to ``set_backend`` side effects.
    """
    return _policy_from_config(_config_from_mode(mode, strict=False))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("jax_gpu_parity", (16, 0, 256)),
        ("jax_gpu_fast", (64, 64, 1024)),
    ],
)
def test_field_kernel_tuning_uses_static_gpu_tiling_without_probes(
    monkeypatch, mode, expected
):
    """Tiling is a pure function of the mode: no device or ``nvidia-smi`` probe."""

    def _no_subprocess(*args, **kwargs):
        raise AssertionError(f"unexpected external command: {args!r}")

    def _no_device_lookup(*args, **kwargs):
        raise AssertionError("field tiling must not look up JAX devices")

    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    monkeypatch.setattr(runtime_module.jax, "local_devices", _no_device_lookup)
    monkeypatch.setattr(runtime_module.jax, "devices", _no_device_lookup)

    tuning = get_field_kernel_tuning(mode)

    assert isinstance(tuning, FieldKernelTuning)
    assert (
        tuning.coil_chunk_size,
        tuning.quadrature_block_size,
        tuning.point_chunk_size,
    ) == expected


def test_field_kernel_tuning_applies_environment_overrides(monkeypatch):
    monkeypatch.setenv("SIMSOPT_JAX_COIL_CHUNK_SIZE", "8")
    monkeypatch.setenv("SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE", "32")
    monkeypatch.setenv("SIMSOPT_JAX_POINT_CHUNK_SIZE", "128")

    tuning = get_field_kernel_tuning("jax_gpu_fast")

    assert (
        tuning.coil_chunk_size,
        tuning.quadrature_block_size,
        tuning.point_chunk_size,
    ) == (8, 32, 128)


def _fake_jax(jax_platforms, local_devices):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(jax_platforms=jax_platforms),
        local_devices=local_devices,
    )


def test_runtime_jax_device_uses_primary_configured_jax_platform_before_policy(
    monkeypatch,
):
    """Without a JAX policy, placement follows the platforms JAX was configured with."""
    runtime_device = object()
    backend_calls: list[str | None] = []

    def _local_devices(*, backend=None):
        backend_calls.append(backend)
        return [runtime_device]

    monkeypatch.setattr(
        runtime_module,
        "get_backend_policy",
        lambda mode=None: _policy_for_mode("native_cpu"),
    )
    fake_jax = _fake_jax("cuda", _local_devices)
    monkeypatch.setattr(runtime_module, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    assert get_runtime_jax_device() is runtime_device
    assert backend_calls == ["gpu"]


def test_runtime_jax_device_ignores_jax_platforms_env_rewritten_after_jax_import(
    monkeypatch,
):
    """``set_backend("native_cpu")`` writes ``JAX_PLATFORMS=cpu`` for children.

    The running JAX keeps the platforms it was configured with, so the runtime
    device must too; following the rewritten variable placed arrays on the CPU
    while every default-placed array stayed on the GPU.
    """
    runtime_device = object()
    backend_calls: list[str | None] = []

    def _local_devices(*, backend=None):
        backend_calls.append(backend)
        return [runtime_device]

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(
        runtime_module,
        "get_backend_policy",
        lambda mode=None: _policy_for_mode("native_cpu"),
    )
    fake_jax = _fake_jax("cuda,cpu", _local_devices)
    monkeypatch.setattr(runtime_module, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    assert get_runtime_jax_device() is runtime_device
    assert backend_calls == ["gpu"]


def test_runtime_jax_device_reads_the_env_while_jax_is_still_importing(monkeypatch):
    """A ``jax`` module without ``config`` yet is a first import in progress.

    JAX has not read its platforms at that point either, so ``JAX_PLATFORMS``
    decides, exactly as before any import; touching ``config`` would raise.
    """
    runtime_device = object()
    backend_calls: list[str | None] = []

    def _local_devices(*, backend=None):
        backend_calls.append(backend)
        return [runtime_device]

    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setattr(
        runtime_module,
        "get_backend_policy",
        lambda mode=None: _policy_for_mode("native_cpu"),
    )
    fake_jax = types.SimpleNamespace(local_devices=_local_devices)
    monkeypatch.setattr(runtime_module, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    assert get_runtime_jax_device() is runtime_device
    assert backend_calls == ["gpu"]


def test_runtime_jax_device_prefers_policy_over_jax_platforms_env(monkeypatch):
    """Once policy is installed, the policy platform remains authoritative."""
    runtime_device = object()
    backend_calls: list[str | None] = []

    def _local_devices(*, backend=None):
        backend_calls.append(backend)
        return [runtime_device]

    monkeypatch.setattr(
        runtime_module,
        "get_backend_policy",
        lambda mode=None: _policy_for_mode("jax_gpu_parity"),
    )
    fake_jax = _fake_jax("cpu,cuda", _local_devices)
    monkeypatch.setattr(runtime_module, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    assert get_runtime_jax_device() is runtime_device
    assert backend_calls == ["gpu"]


def test_runtime_jax_device_returns_none_without_policy_or_configured_jax_platforms(
    monkeypatch,
):
    """Native startup with no JAX platform request keeps the default placement path."""

    def _local_devices(*, backend=None):
        raise AssertionError(f"no device lookup expected, got backend={backend!r}")

    monkeypatch.setattr(
        runtime_module,
        "get_backend_policy",
        lambda mode=None: _policy_for_mode("native_cpu"),
    )
    fake_jax = _fake_jax(None, _local_devices)
    monkeypatch.setattr(runtime_module, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    assert get_runtime_jax_device() is None
