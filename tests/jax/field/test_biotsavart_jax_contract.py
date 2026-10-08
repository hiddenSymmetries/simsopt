"""Public Python objective boundary and explicit runtime initialization."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import os
import subprocess
import sys

import jax
import pytest

from simsopt.field import BiotSavart
from simsopt.field.magneticfield import MagneticFieldMultiply, MagneticFieldSum
from simsopt_jax.backend import set_backend
from simsopt_jax_adapters.field import BiotSavartJAX


@pytest.mark.parametrize("operation", ["left_scale", "right_scale", "add", "reverse_add"])
def test_native_field_arithmetic_is_rejected(operation):
    field = BiotSavartJAX([])
    with pytest.raises(TypeError, match="simsopt.field.BiotSavart"):
        if operation == "left_scale":
            _ = 2 * field
        elif operation == "right_scale":
            _ = field * 2
        elif operation == "add":
            _ = field + field
        else:
            _ = 0 + field


@pytest.mark.parametrize("debug", [False, True])
def test_explicit_runtime_initialization_in_fresh_process(tmp_path, debug):
    code = '''
import os
import jax
import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves
from simsopt_jax.backend import get_field_kernel_tuning, set_backend
from simsopt_jax.runtime.host_boundary import allow_host_transfers, host_array
from simsopt_jax_adapters.field import BiotSavartJAX

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
field = BiotSavartJAX([Coil(curve, Current(1e5))])
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
'''
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


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("intent", ["fast", "parity"])
def test_default_backend_allows_implicit_transfers(monkeypatch, device, intent):
    """Native objectives read adapter arrays every call; auditing them is opt-in."""
    monkeypatch.delenv("SIMSOPT_JAX_TRANSFER_GUARD", raising=False)
    monkeypatch.delenv("SIMSOPT_DEBUG", raising=False)
    config = set_backend("jax", device=device, intent=intent, configure_runtime=device == "cpu")
    assert config.transfer_guard == "allow"
    if device == "cpu":
        assert jax.config.values["jax_transfer_guard"] == "allow"
    monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "log")
    assert set_backend(
        "jax", device=device, intent=intent, configure_runtime=False
    ).transfer_guard == "log"


def test_gpu_allocation_settings_can_be_applied_after_import_in_fresh_process():
    code = '''
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
'''
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ), check=True)


@pytest.mark.parametrize("operation", ["native_left", "native_right", "sum_wrapper", "scale_wrapper"])
def test_native_field_wrappers_reject_adapter_dependencies(operation):
    field, native = BiotSavartJAX([]), BiotSavart([])
    with pytest.raises(TypeError, match="simsopt.field.BiotSavart"):
        if operation == "native_left":
            _ = native + field
        elif operation == "native_right":
            _ = field + native
        elif operation == "sum_wrapper":
            MagneticFieldSum([field, native])
        else:
            MagneticFieldMultiply(2, field)
