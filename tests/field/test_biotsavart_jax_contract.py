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
        code = """
import os
import jax
import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves
from simsopt_jax.backend import get_field_kernel_tuning, set_backend
from simsopt_jax.runtime.host_boundary import allow_host_transfers, host_array
from simsopt_jax_adapters.field import JaxBiotSavart

config = set_backend("jax", device="cpu", intent="parity")
debug = os.environ["SIMSOPT_DEBUG"] == "true"
assert config.debug_nans and config.disable_jit
assert config.strict == debug
assert jax.config.jax_debug_nans and jax.config.jax_disable_jit
assert jax.config.jax_transfer_guard == ("disallow" if debug else "allow")
assert jax.config.jax_enable_x64
assert jax.config.values["jax_compilation_cache_dir"] == os.environ["SIMSOPT_JAX_COMPILATION_CACHE_DIR"]
tuning = get_field_kernel_tuning()
assert (tuning.coil_chunk_size, tuning.quadrature_block_size, tuning.point_chunk_size) == ((0, 0, 0) if debug else (2, 8, 1))
curve = create_equally_spaced_curves(1, 1, False, order=1, numquadpoints=16)[0]
field = JaxBiotSavart([Coil(curve, Current(1e5))])
field.set_points(np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]]))
# Eager JAX indexing stages scalar indices; explicitly permit debug evaluation.
if debug:
    with allow_host_transfers():
        value = field.B()
else:
    value = field.B()
assert jax.config.jax_transfer_guard == ("disallow" if debug else "allow")
assert value.dtype == np.float64
assert all(device.platform == "cpu" for device in value.devices())
with allow_host_transfers():
    assert np.all(np.isfinite(host_array(value)))
"""
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
        subprocess.run([sys.executable, "-c", code], env=environment, check=True)

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
        code = """
import os
import jax
from simsopt_jax.backend._runtime_policy import _config_from_mode
from simsopt_jax.backend.runtime import _apply_jax_gpu_memory_env, _jax_backends_initialized

assert not _jax_backends_initialized()
config = _config_from_mode("jax_gpu_fast", strict=False, xla_gpu_preallocate=False, xla_gpu_mem_fraction=0.5)
_apply_jax_gpu_memory_env(config)
assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"
assert not _jax_backends_initialized()
"""
        subprocess.run([sys.executable, "-c", code], env=dict(os.environ), check=True)

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
