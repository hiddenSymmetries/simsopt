"""Public Python objective boundary and explicit runtime initialization."""

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    from simsopt_jax.backend import set_backend
    from simsopt_jax_adapters.field import JaxBiotSavart
from unittest import mock
from tempfile import TemporaryDirectory
from pathlib import Path


import os
import subprocess
import sys


from simsopt.field import BiotSavart
from simsopt.field.magneticfield import MagneticFieldMultiply, MagneticFieldSum


_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")


def _run_fresh_process(script, environment):
    """Run ``fresh_process/<script>`` in a new interpreter; it fails on its asserts."""
    subprocess.run(
        [sys.executable, str(_FRESH_PROCESS / script)], env=environment, check=True
    )


class TestBiotsavartJaxContract(JaxTestCase):
    def test_native_field_arithmetic_is_rejected(self):
        """Adapter field sums and scalar multiplication reject unsupported native
        arithmetic with TypeError."""
        for operation in ["left_scale", "right_scale", "add", "reverse_add"]:
            with self.subTest(operation=operation), self.case():
                self._case_native_field_arithmetic_is_rejected(operation)

    def _case_native_field_arithmetic_is_rejected(self, operation):
        """Run one row with case-local objects released before runtime cleanup."""
        field = JaxBiotSavart([])
        with self.assertRaisesRegex(TypeError, "simsopt.field.BiotSavart"):
            if operation == "left_scale":
                _ = 2 * field
            elif operation == "right_scale":
                _ = field * 2
            elif operation == "add":
                _ = field + field
            else:
                _ = 0 + field

    def test_explicit_runtime_initialization_in_fresh_process(self):
        """Explicit backend setup applies diagnostics, precision, cache and tiling
        settings before CPU field evaluation."""
        for debug in [False, True]:
            with self.subTest(debug=debug), self.case() as patches:
                self._case_explicit_runtime_initialization_in_fresh_process(
                    debug, patches
                )

    def _case_explicit_runtime_initialization_in_fresh_process(self, debug, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        tmp_path = Path(patches.enter_context(TemporaryDirectory()))
        environment = dict(os.environ)
        environment.update(
            JAX_PLATFORMS="cpu",
            SIMSOPT_DEBUG="true" if debug else "false",
            SIMSOPT_JAX_DEBUG_NANS="true",
            SIMSOPT_JAX_DISABLE_JIT="true",
            SIMSOPT_JAX_TRANSFER_GUARD="allow",
            SIMSOPT_JAX_COMPILATION_CACHE_DIR=str(tmp_path / "compilation-cache"),
            SIMSOPT_JAX_COIL_CHUNK_SIZE="2",
            SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE="8",
            SIMSOPT_JAX_POINT_CHUNK_SIZE="1",
        )
        _run_fresh_process("explicit_runtime_initialization.py", environment)

    def test_default_backend_allows_implicit_transfers(self):
        """Native objectives read adapter arrays every call; auditing them is opt-in."""
        for intent in ("fast", "parity"):
            for device in ("cpu", "gpu"):
                with self.subTest(intent=intent, device=device), self.case() as patches:
                    self._case_default_backend_allows_implicit_transfers(
                        intent, device, patches
                    )

    def _case_default_backend_allows_implicit_transfers(self, intent, device, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop("SIMSOPT_JAX_TRANSFER_GUARD", None)
        patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop("SIMSOPT_DEBUG", None)
        config = set_backend(
            "jax",
            device=device,
            intent=intent,
            configure_runtime=device == "cpu",
        )
        self.assertTrue(
            config.transfer_guard == "allow",
            'config.transfer_guard == "allow"',
        )
        if device == "cpu":
            self.assertTrue(
                jax.config.values["jax_transfer_guard"] == "allow",
                'jax.config.values["jax_transfer_guard"] == "allow"',
            )
        patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_TRANSFER_GUARD": "log"})
        )
        self.assertTrue(
            set_backend(
                "jax", device=device, intent=intent, configure_runtime=False
            ).transfer_guard
            == "log",
            'set_backend( "jax", device=device, intent=intent, configure_runtime=False ).transfer_guard == "log"',
        )

    def test_gpu_allocation_settings_can_be_applied_after_import_in_fresh_process(self):
        """GPU allocator settings can be installed after importing JAX without
        initializing its devices."""
        _run_fresh_process("gpu_allocation_after_import.py", dict(os.environ))

    def test_native_field_wrappers_reject_adapter_dependencies(self):
        """Native field sum and multiplication wrappers reject JAX adapter dependencies."""
        for operation in [
            "native_left",
            "native_right",
            "sum_wrapper",
            "scale_wrapper",
        ]:
            with self.subTest(operation=operation), self.case():
                self._case_native_field_wrappers_reject_adapter_dependencies(operation)

    def _case_native_field_wrappers_reject_adapter_dependencies(self, operation):
        """Run one row with case-local objects released before runtime cleanup."""
        field, native = JaxBiotSavart([]), BiotSavart([])
        with self.assertRaisesRegex(TypeError, "simsopt.field.BiotSavart"):
            if operation == "native_left":
                _ = native + field
            elif operation == "native_right":
                _ = field + native
            elif operation == "sum_wrapper":
                MagneticFieldSum([field, native])
            else:
                MagneticFieldMultiply(2, field)
