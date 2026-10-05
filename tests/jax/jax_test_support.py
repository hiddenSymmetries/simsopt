"""JAX test runtime for the JAX-dependent test modules.

Importing this module pins XLA's CUDA autotuners in ``XLA_FLAGS`` and forces
``jax_enable_x64`` for the whole process. A test module opts in by making
``from jax_test_support import fixture_jax_runtime_guard`` its first import,
which also applies the per-test backend-state guard to that module alone.
Fixtures are exported as ``fixture_<name>`` (pytest's ``name=`` convention), so
a test that requests ``parity_lane`` does not shadow the import.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
import gc

import jax
import numpy as np
import pytest

from simsopt_jax.backend.runtime import apply_cuda_xla_flag_pins

# XLA reads ``XLA_FLAGS`` when it initializes a backend, and a JAX test module
# probes devices (lane availability) at collection, before any test installs a
# backend config, so the CUDA autotuner pins must already be in the environment
# here; both are inert on the CPU backend.
apply_cuda_xla_flag_pins()


def _force_x64(jax_module) -> None:
    jax_module.config.update("jax_enable_x64", True)
    if jax_module.config.jax_enable_x64 is not True:
        raise RuntimeError("tests/jax/jax_test_support.py requires jax_enable_x64=True")


_force_x64(jax)

_BACKEND_RUNTIME_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_PRECISION",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_DEBUG",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_DISABLE_JIT",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
    "SIMSOPT_JAX_POINT_CHUNK_SIZE",
    "SIMSOPT_JAX_GPU_PREALLOCATE",
    "SIMSOPT_JAX_GPU_MEM_FRACTION",
    "SIMSOPT_JAX_GPU_ALLOCATOR",
    "SIMSOPT_TF_GPU_ALLOCATOR",
    "SIMSOPT_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_CLIENT_MEM_FRACTION",
    "TF_GPU_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
)
_JAX_RUNTIME_CONFIG_DEFAULTS = {
    "jax_enable_x64": True,
    "jax_debug_nans": False,
    "jax_disable_jit": False,
    "jax_transfer_guard": None,
    "jax_platforms": None,
    "jax_platform_name": "",
    "jax_compilation_cache_dir": None,
}
_PARITY_SEED_BASE = 1729


def _require_jax():
    return jax


def _loaded_backend_module():
    module = sys.modules.get("simsopt_jax.backend")
    if module is not None and hasattr(module, "invalidate_backend_cache"):
        return module
    return None


def _loaded_jax_core_module():
    module = sys.modules.get("simsopt_jax.core")
    if module is not None and hasattr(module, "invalidate_kernel_cache"):
        return module
    return None


def _invalidate_loaded_kernel_cache() -> None:
    jax_core_module = _loaded_jax_core_module()
    if jax_core_module is not None:
        jax_core_module.invalidate_kernel_cache()


def _invalidate_loaded_backend_state() -> None:
    backend_module = _loaded_backend_module()
    if backend_module is not None:
        backend_module.invalidate_backend_cache()
    _invalidate_loaded_kernel_cache()


def _snapshot_loaded_jax_runtime_config() -> dict[str, object]:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return dict(_JAX_RUNTIME_CONFIG_DEFAULTS)
    return {
        name: jax_module.config.values[name] for name in _JAX_RUNTIME_CONFIG_DEFAULTS
    }


def _restore_loaded_jax_runtime_config(snapshot: dict[str, object]) -> None:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return
    for name, value in snapshot.items():
        jax_module.config.update(name, value)


def _restore_backend_runtime_env(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True, name="jax_runtime_guard")
def fixture_jax_runtime_guard():
    """Restore the backend env, the JAX config and the simsopt_jax caches per test.

    pytest orders a module's own autouse fixtures by attribute name, so each of
    them requests ``jax_runtime_guard`` to run inside it.
    """
    env_snapshot = {name: os.environ.get(name) for name in _BACKEND_RUNTIME_ENV_VARS}
    jax_config_snapshot = _snapshot_loaded_jax_runtime_config()
    _invalidate_loaded_backend_state()
    try:
        yield
    finally:
        _restore_backend_runtime_env(env_snapshot)
        _restore_loaded_jax_runtime_config(jax_config_snapshot)
        _invalidate_loaded_backend_state()
        # Bound JAX's XLA executable cache within a long-lived single process:
        # the invalidations above clear simsopt_jax's caches but not JAX's, which
        # otherwise grows unbounded across a module's tests until a native
        # allocation fails (std::bad_alloc -> abort). Clearing per test keeps peak
        # RSS bounded to ~one test's working set.
        _jax_mod = sys.modules.get("jax")
        if _jax_mod is not None:
            _jax_mod.clear_caches()
            gc.collect()


def parity_seed(seed: int = 0) -> int:
    return _PARITY_SEED_BASE + seed


def parity_rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(parity_seed(seed))


def _parity_device_for_lane(jax_module, lane: str):
    if lane not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown parity lane {lane!r}; expected 'cpu' or 'gpu'.")
    for device in jax_module.devices():
        if device.platform == lane:
            return device
    if lane == "gpu":
        pytest.skip("CUDA GPU not available")
    if lane == "cpu":
        pytest.skip("CPU JAX backend not available")


@contextmanager
def parity_default_device(lane: str):
    jax_module = _require_jax()
    with jax_module.default_device(_parity_device_for_lane(jax_module, lane)):
        yield


def _block_until_ready(value, *, jax_module):
    return jax_module.tree.map(
        lambda leaf: (
            leaf.block_until_ready() if isinstance(leaf, jax_module.Array) else leaf
        ),
        value,
    )


def host_materialize(value):
    jax_module = _require_jax()
    return jax_module.device_get(_block_until_ready(value, jax_module=jax_module))


def host_array(value, *, dtype=None):
    return np.asarray(host_materialize(value), dtype=dtype)


@pytest.fixture(
    params=("cpu", "gpu"), ids=("cpu_parity", "gpu_parity"), name="parity_lane"
)
def fixture_parity_lane(request):
    return request.param
