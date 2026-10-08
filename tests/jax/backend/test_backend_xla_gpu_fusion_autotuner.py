"""CUDA compilation flags compose idempotently and preserve caller overrides."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import os

import pytest
from simsopt_jax.backend.runtime import (
    BackendConfig,
    _CPU_OPT_PRESET_FAST_COMPILE,
    _GPU_AUTOTUNE_LEVEL_PINNED,
    _GPU_FUSION_AUTOTUNER_DISABLED,
    _XLA_FLAGS_ENV,
    _apply_cuda_autotuner_env,
    apply_cuda_xla_flag_pins,
    _xla_flags_with_cpu_compile_preset,
    _xla_flags_with_gpu_autotune_level_pinned,
    _xla_flags_with_gpu_fusion_autotuner_disabled,
)

_CUDA_PINS = f"{_GPU_FUSION_AUTOTUNER_DISABLED} {_GPU_AUTOTUNE_LEVEL_PINNED}"


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_compose_yields_lone_flag_for_empty_input(empty):
    assert (
        _xla_flags_with_gpu_fusion_autotuner_disabled(empty)
        == _GPU_FUSION_AUTOTUNER_DISABLED
    )


def test_compose_preserves_existing_unrelated_flags():
    existing = "--xla_gpu_exclude_nondeterministic_ops=true"
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(existing) == (
        f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}"
    )


def test_compose_is_idempotent():
    once = _xla_flags_with_gpu_fusion_autotuner_disabled(None)
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(once) == once


@pytest.mark.parametrize(
    "existing",
    [
        "--xla_gpu_experimental_enable_fusion_autotuner",
        "--xla_gpu_experimental_enable_fusion_autotuner=true",
        "--a=1 --xla_gpu_experimental_enable_fusion_autotuner=true --b=2",
    ],
)
def test_compose_respects_caller_supplied_value(existing):
    """A caller who deliberately keeps the autotuner on is not overridden."""
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(existing) == existing


def test_compose_does_not_match_prefix_lookalike_flag():
    existing = "--xla_gpu_experimental_enable_fusion_autotuner_extra=1"
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(existing) == (
        f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}"
    )


def test_compose_composes_with_the_cpu_preset_helper():
    """Both helpers share one composition rule, so they stack in either order."""
    both = _xla_flags_with_gpu_fusion_autotuner_disabled(
        _xla_flags_with_cpu_compile_preset(None)
    )
    assert both == f"{_CPU_OPT_PRESET_FAST_COMPILE} {_GPU_FUSION_AUTOTUNER_DISABLED}"
    assert _xla_flags_with_cpu_compile_preset(both) == both


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_autotune_pin_yields_lone_flag_for_empty_input(empty):
    assert (
        _xla_flags_with_gpu_autotune_level_pinned(empty) == _GPU_AUTOTUNE_LEVEL_PINNED
    )


@pytest.mark.parametrize(
    "existing",
    [
        "--xla_gpu_autotune_level=4",
        "--xla_gpu_autotune_level=0",
        "--a=1 --xla_gpu_autotune_level=2 --b=2",
    ],
)
def test_autotune_pin_respects_caller_supplied_level(existing):
    """A caller who deliberately keeps autotuning on is not overridden."""
    assert _xla_flags_with_gpu_autotune_level_pinned(existing) == existing


def test_autotune_pin_is_idempotent():
    once = _xla_flags_with_gpu_autotune_level_pinned(None)
    assert _xla_flags_with_gpu_autotune_level_pinned(once) == once


def test_public_pin_entry_point_sets_both_pins(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    assert apply_cuda_xla_flag_pins() == _CUDA_PINS
    assert os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS


def _config(jax_platform):
    return BackendConfig(mode="jax_gpu_parity", backend="jax", jax_platform=jax_platform)


def test_apply_is_noop_on_cpu(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cuda_autotuner_env(_config("cpu"))
    assert _XLA_FLAGS_ENV not in os.environ


def test_apply_sets_flag_on_cuda(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cuda_autotuner_env(_config("cuda"))
    assert os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS


def test_apply_keeps_caller_flags_on_cuda(monkeypatch):
    monkeypatch.setenv(_XLA_FLAGS_ENV, "--xla_gpu_exclude_nondeterministic_ops=true")
    _apply_cuda_autotuner_env(_config("cuda"))
    assert os.environ[_XLA_FLAGS_ENV] == (
        f"--xla_gpu_exclude_nondeterministic_ops=true {_CUDA_PINS}"
    )


def test_apply_respects_caller_autotune_level_on_cuda(monkeypatch):
    monkeypatch.setenv(_XLA_FLAGS_ENV, "--xla_gpu_autotune_level=4")
    _apply_cuda_autotuner_env(_config("cuda"))
    assert os.environ[_XLA_FLAGS_ENV] == (
        f"--xla_gpu_autotune_level=4 {_GPU_FUSION_AUTOTUNER_DISABLED}"
    )


def test_apply_is_idempotent_across_repeated_calls(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cuda_autotuner_env(_config("cuda"))
    _apply_cuda_autotuner_env(_config("cuda"))
    assert os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS
