"""Regression tests for ``set_backend`` process-state changes.

1. A rejected call leaves the previous mode, environment and every JAX
   setting in place, and the field still evaluates as before; inside a JAX
   configuration context it changes no setting the context hides.
2. In a process whose JAX backends are initialized, GPU allocator settings
   and XLA flags cannot apply: selecting the GPU warns instead of raising and
   leaves the caller's allocator variables and ``XLA_FLAGS`` alone, and the CPU
   compile preset stays out of ``XLA_FLAGS``.
3. ``SIMSOPT_JAX_DEBUG_NANS`` is parsed like every other boolean setting.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports, run_fresh_process

with optional_jax_imports():
    import jax
    import jax.extend.backend
    import simsopt_jax  # noqa: F401
    import simsopt_jax.backend.runtime as runtime_module
    from simsopt_jax.backend.runtime import (
        _config_from_mode,
        get_backend_config,
        get_backend_mode,
        set_backend,
    )
    from simsopt_jax_adapters.field import JaxBiotSavart
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves


def _field_B() -> np.ndarray:
    """B (T), shape (2, 3), of two 100 kA circular coils at two off-axis points."""
    curves = create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.5, order=2, numquadpoints=16
    )
    field = JaxBiotSavart([Coil(curve, Current(1.0e5)) for curve in curves])
    field.set_points(np.array([[1.1, 0.0, 0.1], [0.9, 0.2, -0.1]]))
    return np.asarray(field.B())


_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")
_OBSERVED_ENV_PREFIXES = ("SIMSOPT_", "JAX_", "XLA_", "TF_GPU_")
_DIRECTIONAL_TRANSFER_GUARD = "jax_transfer_guard_host_to_device"


def _process_state() -> tuple[dict[str, str], dict[str, object]]:
    """The backend environment variables and every JAX setting of this process."""
    env = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(_OBSERVED_ENV_PREFIXES)
    }
    return env, dict(jax.config.values)


def _run_cpu_child(name: str, **environment: str):
    """Run ``fresh_process/<name>`` in a CPU-only interpreter with ``environment`` added."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", **environment)
    return run_fresh_process(_FRESH_PROCESS / name, env=env, timeout=300)


class TestSetBackendLifecycle(JaxTestCase):
    def test_rejected_set_backend_keeps_the_previous_mode(self):
        """An early or a late rejection restores the CPU parity mode, every JAX
        setting (a directional transfer guard included) and the field."""
        rows = [
            (
                "determinism flags missing",
                "does not enable",
                nullcontext(),
                lambda: set_backend("jax", device="gpu", intent="parity"),
            ),
            (
                "platform rejected",
                "rejected",
                mock.patch.object(
                    runtime_module,
                    "_initialized_runtime_platform",
                    side_effect=RuntimeError("rejected"),
                ),
                lambda: set_backend(
                    "jax",
                    device="gpu",
                    debug_nans=True,
                    disable_jit=True,
                    transfer_guard="disallow",
                ),
            ),
        ]
        for name, message, rejection, request in rows:
            with self.subTest(name), self.case() as patches:
                os.environ.pop("XLA_FLAGS", None)
                config = set_backend("jax", device="cpu", intent="parity")
                reference_B = _field_B()
                # The rejected aggregate transfer guard overwrites this one.
                patches.callback(
                    jax.config.update,
                    _DIRECTIONAL_TRANSFER_GUARD,
                    jax.config.values[_DIRECTIONAL_TRANSFER_GUARD],
                )
                jax.config.update(_DIRECTIONAL_TRANSFER_GUARD, "log")
                state = _process_state()
                with rejection, self.assertRaisesRegex(RuntimeError, message):
                    request()

                self.assertEqual(get_backend_mode(), "jax_cpu_parity")
                self.assertIs(get_backend_config(), config)
                self.assertEqual(_process_state(), state)
                np.testing.assert_array_equal(_field_B(), reference_B)

    def test_rejected_set_backend_inside_config_contexts_changes_no_setting(self):
        """A GPU request JAX never initialized, rejected inside a context that
        hides the transfer guard, NaN debugging or JIT, leaves
        every JAX setting as it was once the context exits."""
        completed = _run_cpu_child(
            "rejected_set_backend_in_config_context.py",
            XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_set_backend_rejected_by_the_default_backend_lookup_changes_no_setting(
        self,
    ):
        """A CPU request that JAX's default-backend lookup rejects (the deprecated
        jax_xla_backend names an uninitialized platform) leaves x64, NaN
        debugging, the transfer guard and every other JAX setting unchanged."""
        completed = _run_cpu_child("rejected_set_backend_default_backend_lookup.py")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_gpu_selection_after_initialization_keeps_the_xla_env(self):
        """Selecting the GPU in an initialized process leaves the caller's allocator
        variables and ``XLA_FLAGS`` alone, so child processes do not inherit
        settings this one never ran with. It warns about the allocator settings
        and, once, about compiling without the CUDA autotuner pins."""
        flags = "--xla_gpu_exclude_nondeterministic_ops=true"
        os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
        os.environ["XLA_FLAGS"] = flags
        self.assertTrue(runtime_module._jax_backends_initialized())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_backend("jax", device="gpu")

        self.assertEqual(get_backend_mode(), "jax_gpu_fast")
        self.assertNotIn("XLA_PYTHON_CLIENT_PREALLOCATE", os.environ)
        self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.3")
        self.assertEqual(os.environ["XLA_FLAGS"], flags)
        messages = [str(warning.message) for warning in caught]
        self.assertTrue(
            any("XLA_PYTHON_CLIENT_PREALLOCATE" in text for text in messages),
            messages,
        )
        self.assertEqual(
            sum("CUDA autotuner pins" in text for text in messages), 1, messages
        )
        # Every warning names this call site, not a line inside simsopt.
        self.assertEqual({warning.filename for warning in caught}, {__file__}, messages)
        if {"cuda", "gpu"} & set(jax.extend.backend.backends()):
            self.assertEqual(jax.numpy.zeros(3).devices(), {jax.devices("gpu")[0]})

    def test_cpu_selection_after_initialization_keeps_xla_flags(self):
        """The fast CPU mode's compile preset, which XLA reads only at backend
        initialization, is not exported to an initialized process."""
        os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"
        self.assertTrue(runtime_module._jax_backends_initialized())

        set_backend("jax", device="cpu", intent="fast")

        self.assertEqual(get_backend_mode(), "jax_cpu_fast")
        self.assertEqual(os.environ["XLA_FLAGS"], "--xla_cpu_enable_fast_math=false")

    def test_debug_nans_env_is_a_strict_boolean(self):
        """SIMSOPT_JAX_DEBUG_NANS accepts the boolean spellings and rejects others."""
        rows = [("on", True), ("0", False), ("maybe", None)]
        for raw, expected in rows:
            with self.subTest(raw=raw), self.case():
                os.environ["SIMSOPT_JAX_DEBUG_NANS"] = raw
                os.environ.pop("SIMSOPT_DEBUG", None)
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "must be a boolean"):
                        _config_from_mode("jax_cpu_fast", strict=False)
                else:
                    config = _config_from_mode("jax_cpu_fast", strict=False)
                    self.assertIs(config.debug_nans, expected)
