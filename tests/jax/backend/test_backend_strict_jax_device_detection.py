"""Regression tests for the JAX device-detection strictness contract.

After the silent-fallback cleanup (§2 below) and the narrowed external-boundary
catches (§3), the JAX runtime probes in ``backend/runtime.py`` and ``jax_core/sharding.py`` must
propagate every error from the JAX runtime; the only tolerated boundary
exceptions are:

- ``RuntimeError`` from ``jax.local_devices(backend="gpu")`` (GPU backend
  unavailable on this host).
- ``FileNotFoundError`` / ``subprocess.CalledProcessError`` from ``nvidia-smi``
  (external tool absent or non-zero exit).
- ``ValueError`` when ``_parse_visible_cuda_device_index`` parses an
  ``int`` from ``CUDA_VISIBLE_DEVICES`` (garbage values map to ``None``).

Tests (one per situation):

1. ``test_build_sharding_tuning_skips_jax_device_apis_for_non_jax_backend``
   §2-i — CPU policy must short-circuit before any JAX device API is called.
2. ``test_detect_jax_device_count_propagates_runtime_error`` /
   ``test_detect_jax_device_count_propagates_value_error``
   §2-ii — JAX-mode probes propagate ``RuntimeError`` and ``ValueError``.
3. ``test_inspect_array_sharding_summary_propagates_jax_errors_for_jax_array``
   and ``test_inspect_array_sharding_summary_returns_base_summary_for_non_jax_array``
   §2-v — JAX errors propagate when a real ``jax.Array`` is inspected;
   non-array inputs short-circuit through the pre-check.
4. ``test_detect_imported_jax_cuda_device_index_*`` — §3-vi: narrow
   ``RuntimeError`` catch around ``local_devices(backend="gpu")`` returns
   ``None``; any other exception type propagates.
5. ``test_parse_visible_cuda_device_index_*`` — §3-viii: garbage env values
   map to ``None``; valid non-negative integers parse through.
6. ``test_query_gpu_metric_mb_from_nvidia_smi_*`` — §3-ix: ``nvidia-smi``
   absence / non-zero exit returns ``None``; a valid stdout row parses.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import subprocess
import sys
import types

from simsopt_jax.core import sharding as sharding_module
from simsopt_jax.core.sharding import inspect_array_sharding_summary
import jax
import jax.numpy as jnp

import numpy as np
import pytest

import simsopt_jax.backend.runtime as runtime_module
import simsopt_jax.backend._runtime_tuning as tuning_module
from simsopt_jax.backend.runtime import (
    BackendPolicy,
    _build_sharding_tuning,
    _config_from_mode,
    _detect_imported_jax_cuda_device_index,
    _detect_local_jax_device_count,
    _parse_visible_cuda_device_index,
    _policy_from_config,
    _query_gpu_metric_mb_from_nvidia_smi,
    get_runtime_jax_device,
)

# ---------------------------------------------------------------------------
# Policy builders — small helpers around the canonical mode->policy pipeline.
# ---------------------------------------------------------------------------


def _policy_for_mode(mode: str) -> BackendPolicy:
    """Return the canonical :class:`BackendPolicy` for ``mode``.

    Uses the same pipeline as ``get_backend_policy(mode)`` without touching
    the cached module-level state; this lets tests build orthogonal policies
    without coupling to ``set_backend`` side effects.
    """
    return _policy_from_config(_config_from_mode(mode, strict=False))


# ---------------------------------------------------------------------------
# §2-i — CPU backend must never invoke JAX device APIs.
# ---------------------------------------------------------------------------


def test_build_sharding_tuning_skips_jax_device_apis_for_non_jax_backend(monkeypatch):
    """A non-JAX policy must never reach into ``jax.local_devices``/``devices``.

    Any call to either stub raises ``AssertionError``, so if a future
    refactor accidentally calls the JAX device API on a CPU policy the
    test fails inside ``_build_sharding_tuning`` rather than at a post-hoc
    counter check.
    """
    policy = _policy_for_mode("native_cpu")
    assert policy.backend == "cpu"

    def _explode_local(*, backend=None):
        raise AssertionError(
            f"jax.local_devices should not be called for non-JAX policy "
            f"(backend={backend!r})"
        )

    def _explode_devices(*, backend=None):
        raise AssertionError(
            f"jax.devices should not be called for non-JAX policy (backend={backend!r})"
        )

    fake_jax = types.SimpleNamespace(
        local_devices=_explode_local,
        devices=_explode_devices,
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    tuning = _build_sharding_tuning(policy.mode, policy)

    assert tuning.strategy == "none"
    assert tuning.local_device_count == 0
    assert tuning.device_count == 0


# ---------------------------------------------------------------------------
# §2-ii — JAX-mode device probes propagate runtime errors.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attr_name", "detector"),
    [
        ("local_devices", _detect_local_jax_device_count),
    ],
)
def test_detect_jax_device_count_propagates_runtime_error(
    monkeypatch, attr_name, detector
):
    """``RuntimeError`` from the JAX device API must escape the helper."""
    policy = _policy_for_mode("jax_cpu_parity")
    assert policy.backend == "jax"

    def _raise_runtime(*, backend=None):
        raise RuntimeError(f"backend gone: {backend!r}")

    fake_jax = types.SimpleNamespace(**{attr_name: _raise_runtime})
    monkeypatch.setattr(tuning_module, "jax", fake_jax)

    with pytest.raises(RuntimeError, match="backend gone"):
        detector(policy)


@pytest.mark.parametrize(
    ("attr_name", "detector"),
    [
        ("local_devices", _detect_local_jax_device_count),
    ],
)
def test_detect_jax_device_count_propagates_value_error(
    monkeypatch, attr_name, detector
):
    """``ValueError`` is not absorbed by a broad ``except Exception`` clause."""
    policy = _policy_for_mode("jax_cpu_parity")

    def _raise_value(*, backend=None):
        raise ValueError(f"unexpected backend kwargs: {backend!r}")

    fake_jax = types.SimpleNamespace(**{attr_name: _raise_value})
    monkeypatch.setattr(tuning_module, "jax", fake_jax)

    with pytest.raises(ValueError, match="unexpected backend kwargs"):
        detector(policy)


# ---------------------------------------------------------------------------
# §2-v — ``inspect_array_sharding_summary`` propagation contract.
# ---------------------------------------------------------------------------


def test_inspect_array_sharding_summary_returns_base_summary_for_non_jax_array():
    """Non ``jax.Array`` inputs short-circuit through the pre-check."""

    summary = inspect_array_sharding_summary(np.array([1.0, 2.0]))

    assert summary["kind"] == "non_jax_array"
    assert summary["spec"] is None
    assert summary["device_count"] == 0
    assert summary["fully_replicated"] is None
    # The inspection key must not be added when the pre-check returns early.
    assert "inspected_kind" not in summary
    assert "inspected_spec" not in summary


def test_inspect_array_sharding_summary_propagates_jax_errors_for_jax_array(
    monkeypatch,
):
    """Errors from ``jax.debug.inspect_array_sharding`` must not be swallowed."""


    arr = jnp.array([1.0, 2.0])
    # Precondition: the input must reach the `inspect_fn` branch, not the
    # non-jax short-circuit. If this assert fires, the test below is
    # vacuously checking the non-jax path instead of the propagation gate.
    assert isinstance(arr, jax.Array)

    def _raise_runtime(value, *, callback=None):
        del value, callback
        raise RuntimeError("forced inspect failure")

    monkeypatch.setattr(
        sharding_module.jax.debug,
        "inspect_array_sharding",
        _raise_runtime,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="forced inspect failure"):
        sharding_module.inspect_array_sharding_summary(arr)


# ---------------------------------------------------------------------------
# §3-vi — ``_detect_imported_jax_cuda_device_index`` GPU-availability boundary.
# ---------------------------------------------------------------------------


def test_detect_imported_jax_cuda_device_index_returns_none_on_gpu_runtime_error(
    monkeypatch,
):
    """``RuntimeError`` from ``local_devices(backend='gpu')`` -> ``None``."""

    def _raise_runtime(*, backend=None):
        assert backend == "gpu"
        raise RuntimeError("no gpu backend available")

    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(local_devices=_raise_runtime),
    )

    assert _detect_imported_jax_cuda_device_index() is None


def test_detect_imported_jax_cuda_device_index_propagates_value_error(monkeypatch):
    """Non-``RuntimeError`` exceptions must escape the narrow catch."""

    def _raise_value(*, backend=None):
        assert backend == "gpu"
        raise ValueError("unexpected backend selector")

    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(local_devices=_raise_value),
    )

    with pytest.raises(ValueError, match="unexpected backend selector"):
        _detect_imported_jax_cuda_device_index()


def test_detect_imported_jax_cuda_device_index_returns_value_when_device_present(
    monkeypatch,
):
    """Sanity: a fake CUDA device flows through to its ``local_hardware_id``."""

    def _local_devices(*, backend=None):
        assert backend == "gpu"
        return [types.SimpleNamespace(local_hardware_id=2)]

    monkeypatch.setitem(
        sys.modules,
        "jax",
        types.SimpleNamespace(local_devices=_local_devices),
    )

    assert _detect_imported_jax_cuda_device_index() == 2


# ---------------------------------------------------------------------------
# §3-viii — ``_parse_visible_cuda_device_index`` parses ``CUDA_VISIBLE_DEVICES``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_value",
    [
        "NaN",
        "abc",
        "",
        "-1",
        "none",
        "NoDevFiles",
        "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6",
        "MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2",
    ],
)
def test_parse_visible_cuda_device_index_returns_none_for_non_integer(
    monkeypatch, env_value
):
    """Non-integer or sentinel ``CUDA_VISIBLE_DEVICES`` values must yield ``None``."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env_value)

    assert _parse_visible_cuda_device_index() is None


def test_parse_visible_cuda_device_index_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _parse_visible_cuda_device_index() is None


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("0", 0),
        ("3", 3),
        ("3,1", 3),
    ],
)
def test_parse_visible_cuda_device_index_returns_first_index(
    monkeypatch, env_value, expected
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env_value)

    assert _parse_visible_cuda_device_index() == expected


# ---------------------------------------------------------------------------
# §3-ix — ``_query_gpu_metric_mb_from_nvidia_smi`` external-tool boundary.
# ---------------------------------------------------------------------------


def test_query_gpu_metric_mb_returns_none_when_nvidia_smi_missing(monkeypatch):
    """``FileNotFoundError`` from ``subprocess.run`` -> ``None`` (tool absent)."""
    captured = {"calls": 0}

    def _missing(cmd, *, check, capture_output, text):
        del check, capture_output, text
        captured["calls"] += 1
        assert cmd[0] == "nvidia-smi"
        raise FileNotFoundError("nvidia-smi not on PATH")

    monkeypatch.setattr(subprocess, "run", _missing)

    assert _query_gpu_metric_mb_from_nvidia_smi("memory.total") is None
    assert captured["calls"] == 1


def test_query_gpu_metric_mb_returns_none_on_called_process_error(monkeypatch):
    """``CalledProcessError`` (e.g. no NVIDIA driver) -> ``None``."""

    def _failing(cmd, *, check, capture_output, text):
        del check, capture_output, text
        raise subprocess.CalledProcessError(returncode=9, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", _failing)

    assert _query_gpu_metric_mb_from_nvidia_smi("memory.total") is None


def test_query_gpu_metric_mb_parses_valid_output(monkeypatch):
    """A valid nvidia-smi CSV row parses to a float value."""

    def _fake_run(cmd, *, check, capture_output, text):
        del check, capture_output, text
        assert cmd[0] == "nvidia-smi"
        return types.SimpleNamespace(stdout="3, 24576\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert _query_gpu_metric_mb_from_nvidia_smi(
        "memory.total", device_selector=3
    ) == pytest.approx(24576.0)


def test_query_gpu_metric_mb_filters_by_integer_device_selector(monkeypatch):
    """When the selector is an int, only the matching index row is returned."""

    def _fake_run(cmd, *, check, capture_output, text):
        del check, capture_output, text
        return types.SimpleNamespace(stdout="0, 1024\n1, 2048\n3, 24576\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert _query_gpu_metric_mb_from_nvidia_smi(
        "memory.total", device_selector=1
    ) == pytest.approx(2048.0)


def test_query_gpu_metric_mb_returns_none_on_empty_output(monkeypatch):
    """Empty stdout yields ``None`` rather than a misleading default."""

    def _fake_run(cmd, *, check, capture_output, text):
        del check, capture_output, text
        return types.SimpleNamespace(stdout="\n   \n")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert _query_gpu_metric_mb_from_nvidia_smi("memory.total") is None


def test_query_active_gpu_memory_uses_detected_cuda_selector_without_cuda_policy(
    monkeypatch,
):
    """Memory snapshots follow the active JAX CUDA device, not only policy labels."""
    metric_calls = []

    monkeypatch.setattr(
        runtime_module,
        "get_backend_policy",
        lambda mode=None: _policy_for_mode("native_cpu"),
    )
    monkeypatch.setattr(
        runtime_module,
        "_detect_active_jax_cuda_device_selector",
        lambda: 0,
    )

    def _metric(metric_name, device_selector):
        metric_calls.append((metric_name, device_selector))
        return 4096.0

    monkeypatch.setattr(
        runtime_module,
        "_query_gpu_metric_mb_from_nvidia_smi",
        _metric,
    )

    assert runtime_module.query_active_gpu_memory_mb() == pytest.approx(4096.0)
    assert metric_calls == [("memory.used", 0)]


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
