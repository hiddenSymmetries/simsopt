"""CUDA compilation flags compose idempotently and preserve caller overrides."""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
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
from unittest import mock


import os


if JAX_IMPORT_ERROR is None:
    _CUDA_PINS = f"{_GPU_FUSION_AUTOTUNER_DISABLED} {_GPU_AUTOTUNE_LEVEL_PINNED}"


def _config(jax_platform):
    return BackendConfig(
        mode="jax_gpu_parity", backend="jax", jax_platform=jax_platform
    )


class TestBackendXlaGpuFusionAutotuner(JaxTestCase):
    def test_compose_yields_lone_flag_for_empty_input(self):
        """Empty XLA flags become the single disabled-fusion-autotuner flag."""
        for empty in [None, "", "   "]:
            with self.subTest(empty=empty), self.case():
                self._case_compose_yields_lone_flag_for_empty_input(empty)

    def _case_compose_yields_lone_flag_for_empty_input(self, empty):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            _xla_flags_with_gpu_fusion_autotuner_disabled(empty)
            == _GPU_FUSION_AUTOTUNER_DISABLED,
            "_xla_flags_with_gpu_fusion_autotuner_disabled(empty) == _GPU_FUSION_AUTOTUNER_DISABLED",
        )

    def test_compose_preserves_existing_unrelated_flags(self):
        """Adding the fusion-autotuner pin preserves unrelated XLA flags."""
        existing = "--xla_gpu_exclude_nondeterministic_ops=true"
        self.assertTrue(
            _xla_flags_with_gpu_fusion_autotuner_disabled(existing)
            == (f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}"),
            '_xla_flags_with_gpu_fusion_autotuner_disabled(existing) == ( f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}" )',
        )

    def test_compose_is_idempotent(self):
        """Reapplying the fusion-autotuner pin leaves XLA flags unchanged."""
        once = _xla_flags_with_gpu_fusion_autotuner_disabled(None)
        self.assertTrue(
            _xla_flags_with_gpu_fusion_autotuner_disabled(once) == once,
            "_xla_flags_with_gpu_fusion_autotuner_disabled(once) == once",
        )

    def test_compose_respects_caller_supplied_value(self):
        """A caller who deliberately keeps the autotuner on is not overridden."""
        for existing in [
            "--xla_gpu_experimental_enable_fusion_autotuner",
            "--xla_gpu_experimental_enable_fusion_autotuner=true",
            "--a=1 --xla_gpu_experimental_enable_fusion_autotuner=true --b=2",
        ]:
            with self.subTest(existing=existing), self.case():
                self._case_compose_respects_caller_supplied_value(existing)

    def _case_compose_respects_caller_supplied_value(self, existing):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            _xla_flags_with_gpu_fusion_autotuner_disabled(existing) == existing,
            "_xla_flags_with_gpu_fusion_autotuner_disabled(existing) == existing",
        )

    def test_compose_does_not_match_prefix_lookalike_flag(self):
        """A flag with a longer lookalike name does not suppress the fusion-autotuner
        pin."""
        existing = "--xla_gpu_experimental_enable_fusion_autotuner_extra=1"
        self.assertTrue(
            _xla_flags_with_gpu_fusion_autotuner_disabled(existing)
            == (f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}"),
            '_xla_flags_with_gpu_fusion_autotuner_disabled(existing) == ( f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}" )',
        )

    def test_compose_composes_with_the_cpu_preset_helper(self):
        """Both helpers share one composition rule, so they stack in either order."""
        both = _xla_flags_with_gpu_fusion_autotuner_disabled(
            _xla_flags_with_cpu_compile_preset(None)
        )
        self.assertTrue(
            both == f"{_CPU_OPT_PRESET_FAST_COMPILE} {_GPU_FUSION_AUTOTUNER_DISABLED}",
            'both == f"{_CPU_OPT_PRESET_FAST_COMPILE} {_GPU_FUSION_AUTOTUNER_DISABLED}"',
        )
        self.assertTrue(
            _xla_flags_with_cpu_compile_preset(both) == both,
            "_xla_flags_with_cpu_compile_preset(both) == both",
        )

    def test_autotune_pin_yields_lone_flag_for_empty_input(self):
        """Empty XLA flags become the single level-zero GPU autotune flag."""
        for empty in [None, "", "   "]:
            with self.subTest(empty=empty), self.case():
                self._case_autotune_pin_yields_lone_flag_for_empty_input(empty)

    def _case_autotune_pin_yields_lone_flag_for_empty_input(self, empty):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            _xla_flags_with_gpu_autotune_level_pinned(empty)
            == _GPU_AUTOTUNE_LEVEL_PINNED,
            "_xla_flags_with_gpu_autotune_level_pinned(empty) == _GPU_AUTOTUNE_LEVEL_PINNED",
        )

    def test_autotune_pin_respects_caller_supplied_level(self):
        """A caller who deliberately keeps autotuning on is not overridden."""
        for existing in [
            "--xla_gpu_autotune_level=4",
            "--xla_gpu_autotune_level=0",
            "--a=1 --xla_gpu_autotune_level=2 --b=2",
        ]:
            with self.subTest(existing=existing), self.case():
                self._case_autotune_pin_respects_caller_supplied_level(existing)

    def _case_autotune_pin_respects_caller_supplied_level(self, existing):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            _xla_flags_with_gpu_autotune_level_pinned(existing) == existing,
            "_xla_flags_with_gpu_autotune_level_pinned(existing) == existing",
        )

    def test_autotune_pin_is_idempotent(self):
        """Reapplying the GPU autotune-level pin leaves XLA flags unchanged."""
        once = _xla_flags_with_gpu_autotune_level_pinned(None)
        self.assertTrue(
            _xla_flags_with_gpu_autotune_level_pinned(once) == once,
            "_xla_flags_with_gpu_autotune_level_pinned(once) == once",
        )

    def test_public_pin_entry_point_sets_both_pins(self):
        """The public entry point returns and installs both CUDA autotuner pins."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        self.assertTrue(
            apply_cuda_xla_flag_pins() == _CUDA_PINS,
            "apply_cuda_xla_flag_pins() == _CUDA_PINS",
        )
        self.assertTrue(
            os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS,
            "os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS",
        )

    def test_apply_is_noop_on_cpu(self):
        """Applying CUDA autotuner configuration to CPU leaves XLA_FLAGS unset."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        _apply_cuda_autotuner_env(_config("cpu"))
        self.assertTrue(
            _XLA_FLAGS_ENV not in os.environ, "_XLA_FLAGS_ENV not in os.environ"
        )

    def test_apply_sets_flag_on_cuda(self):
        """CUDA configuration installs both autotuner pins."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        _apply_cuda_autotuner_env(_config("cuda"))
        self.assertTrue(
            os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS,
            "os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS",
        )

    def test_apply_keeps_caller_flags_on_cuda(self):
        """CUDA autotuner configuration preserves unrelated caller-supplied XLA flags."""
        patches = self.patches
        patches.enter_context(
            mock.patch.dict(
                os.environ,
                {_XLA_FLAGS_ENV: "--xla_gpu_exclude_nondeterministic_ops=true"},
            )
        )
        _apply_cuda_autotuner_env(_config("cuda"))
        self.assertTrue(
            os.environ[_XLA_FLAGS_ENV]
            == (f"--xla_gpu_exclude_nondeterministic_ops=true {_CUDA_PINS}"),
            'os.environ[_XLA_FLAGS_ENV] == ( f"--xla_gpu_exclude_nondeterministic_ops=true {_CUDA_PINS}" )',
        )

    def test_apply_respects_caller_autotune_level_on_cuda(self):
        """An explicit GPU autotune level takes precedence over the default pin."""
        patches = self.patches
        patches.enter_context(
            mock.patch.dict(os.environ, {_XLA_FLAGS_ENV: "--xla_gpu_autotune_level=4"})
        )
        _apply_cuda_autotuner_env(_config("cuda"))
        self.assertTrue(
            os.environ[_XLA_FLAGS_ENV]
            == (f"--xla_gpu_autotune_level=4 {_GPU_FUSION_AUTOTUNER_DISABLED}"),
            'os.environ[_XLA_FLAGS_ENV] == ( f"--xla_gpu_autotune_level=4 {_GPU_FUSION_AUTOTUNER_DISABLED}" )',
        )

    def test_apply_is_idempotent_across_repeated_calls(self):
        """Repeated CUDA configuration installs the autotuner pins only once."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        _apply_cuda_autotuner_env(_config("cuda"))
        _apply_cuda_autotuner_env(_config("cuda"))
        self.assertTrue(
            os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS,
            "os.environ[_XLA_FLAGS_ENV] == _CUDA_PINS",
        )
