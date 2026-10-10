"""CPU compile-preset composition and non-parity runtime configuration tests."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    from simsopt_jax.backend.runtime import (
        BackendConfig,
        _CPU_OPT_PRESET_FAST_COMPILE,
        _XLA_FLAGS_ENV,
        _apply_cpu_compile_preset_env,
        _config_from_mode,
        _policy_from_config,
        _xla_flags_with_cpu_compile_preset,
    )
from unittest import mock


import os


# ---------------------------------------------------------------------------
# Pure composition: _xla_flags_with_cpu_compile_preset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CPU-scoped application: _apply_cpu_compile_preset_env
# ---------------------------------------------------------------------------


def _config(jax_platform):
    return BackendConfig(
        mode="jax_cpu_parity", backend="jax", jax_platform=jax_platform
    )


def _policy(parity_mode=False):
    return _policy_from_config(
        _config_from_mode(
            "jax_cpu_parity" if parity_mode else "jax_cpu_fast", strict=False
        )
    )


class TestBackendXlaCompilePreset(JaxTestCase):
    def test_compose_yields_lone_preset_for_empty_input(self):
        """No prior flags must produce exactly the preset token (no stray space)."""
        for empty in [None, "", "   "]:
            with self.subTest(empty=empty), self.case():
                self._case_compose_yields_lone_preset_for_empty_input(empty)

    def _case_compose_yields_lone_preset_for_empty_input(self, empty):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            _xla_flags_with_cpu_compile_preset(empty) == _CPU_OPT_PRESET_FAST_COMPILE,
            "_xla_flags_with_cpu_compile_preset(empty) == _CPU_OPT_PRESET_FAST_COMPILE",
        )

    def test_compose_preserves_existing_unrelated_flags(self):
        """An unrelated flag is kept verbatim with the preset appended after it."""
        existing = "--xla_force_host_platform_device_count=8"
        self.assertTrue(
            _xla_flags_with_cpu_compile_preset(existing)
            == (f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}"),
            '_xla_flags_with_cpu_compile_preset(existing) == ( f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}" )',
        )

    def test_compose_is_idempotent_for_same_preset(self):
        """Re-composing an already-present FAST_COMPILE preset must not duplicate it."""
        once = _xla_flags_with_cpu_compile_preset(None)
        self.assertTrue(
            _xla_flags_with_cpu_compile_preset(once) == once,
            "_xla_flags_with_cpu_compile_preset(once) == once",
        )

    def test_compose_respects_caller_supplied_preset(self):
        """A caller's explicit preset (any value) wins; we neither override nor dup."""
        for existing in [
            "--xla_cpu_opt_preset",  # bare flag, no value (exercises the `==` guard arm)
            "--xla_cpu_opt_preset=DEFAULT",
            "--a=1 --xla_cpu_opt_preset=DEFAULT --b=2",
        ]:
            with self.subTest(existing=existing), self.case():
                self._case_compose_respects_caller_supplied_preset(existing)

    def _case_compose_respects_caller_supplied_preset(self, existing):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            _xla_flags_with_cpu_compile_preset(existing) == existing,
            "_xla_flags_with_cpu_compile_preset(existing) == existing",
        )

    def test_compose_does_not_match_prefix_lookalike_flag(self):
        """A flag that merely starts with the preset name must not block appending."""
        existing = "--xla_cpu_opt_preset_extra=1"
        self.assertTrue(
            _xla_flags_with_cpu_compile_preset(existing)
            == (f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}"),
            '_xla_flags_with_cpu_compile_preset(existing) == ( f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}" )',
        )

    def test_compose_preserves_malformed_user_flags(self):
        """Malformed user flags are still preserved; the helper must not overwrite them."""
        for existing in ['--a="unterminated', "--a='unterminated"]:
            with self.subTest(existing=existing), self.case():
                self._case_compose_preserves_malformed_user_flags(existing)

    def _case_compose_preserves_malformed_user_flags(self, existing):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            _xla_flags_with_cpu_compile_preset(existing)
            == (f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}"),
            '_xla_flags_with_cpu_compile_preset(existing) == ( f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}" )',
        )

    def test_apply_is_noop_on_cuda(self):
        """CUDA must not gain the CPU preset; its XLA_FLAGS stay untouched."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        _apply_cpu_compile_preset_env(_config("cuda"), _policy())
        self.assertTrue(
            _XLA_FLAGS_ENV not in os.environ, "_XLA_FLAGS_ENV not in os.environ"
        )

    def test_apply_is_noop_on_cpu_parity(self):
        """The bit-exact CPU parity lane must not gain the optimization-skipping preset."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        _apply_cpu_compile_preset_env(_config("cpu"), _policy(parity_mode=True))
        self.assertTrue(
            _XLA_FLAGS_ENV not in os.environ, "_XLA_FLAGS_ENV not in os.environ"
        )

    def test_apply_sets_preset_on_cpu(self):
        """A non-parity CPU lane with no prior flags gets exactly the preset."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        _apply_cpu_compile_preset_env(_config("cpu"), _policy(parity_mode=False))
        self.assertTrue(
            os.environ[_XLA_FLAGS_ENV] == _CPU_OPT_PRESET_FAST_COMPILE,
            "os.environ[_XLA_FLAGS_ENV] == _CPU_OPT_PRESET_FAST_COMPILE",
        )

    def test_apply_is_idempotent_across_repeated_calls(self):
        """Re-applying the config (e.g. repeated init) must not duplicate the preset."""
        patches = self.patches
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop(_XLA_FLAGS_ENV, None)
        _apply_cpu_compile_preset_env(_config("cpu"), _policy())
        _apply_cpu_compile_preset_env(_config("cpu"), _policy())
        self.assertTrue(
            os.environ[_XLA_FLAGS_ENV] == _CPU_OPT_PRESET_FAST_COMPILE,
            "os.environ[_XLA_FLAGS_ENV] == _CPU_OPT_PRESET_FAST_COMPILE",
        )
