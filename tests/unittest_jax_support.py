"""Shared optional-JAX gating, runtime isolation and logging for unittest."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from collections.abc import Iterator
from unittest import TestCase, SkipTest, skipIf
import logging
import os
import sys
import gc

import numpy as np

JAX_IMPORT_ERROR: str | None = None
try:
    import simsopt_jax  # noqa: F401
    import jax
except ImportError as error:
    JAX_IMPORT_ERROR = str(error)


@contextmanager
def optional_jax_imports() -> Iterator[None]:
    """Suppress import errors only when the optional JAX runtime is unavailable."""
    try:
        yield
    except ImportError:
        if JAX_IMPORT_ERROR is None:
            raise


with optional_jax_imports():
    from simsopt_jax.backend.runtime import apply_cuda_xla_flag_pins


@skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
class JaxTestCase(TestCase):
    """Skip unsupported JAX; isolate methods and each parameterized subtest.

    Use ``with self.subTest(...), self.case() as patches`` for each product row.
    Run each row in a case helper so its locals are released before cleanup.
    The returned ExitStack owns patches and temporary resources inside isolation.
    """

    def setUp(self) -> None:
        self.patches = self.enterContext(self.case())

    @contextmanager
    def case(self) -> Iterator[ExitStack]:
        """Yield a resource stack that unwinds before restoring JAX runtime state."""
        with ExitStack() as patches:
            patches.enter_context(jax_runtime_isolation())
            yield patches


try:
    from simsopt_jax.backend.dtypes import explicit_device_array
except ImportError:
    if JAX_IMPORT_ERROR is None:
        raise


def _force_x64(jax_module) -> None:
    jax_module.config.update("jax_enable_x64", True)
    if jax_module.config.jax_enable_x64 is not True:
        raise RuntimeError("unittest_jax_support.py requires jax_enable_x64=True")


# XLA reads ``XLA_FLAGS`` when it initializes a backend, and a JAX test module
# probes devices (lane availability) at collection, before any test installs a
# backend config, so the CUDA autotuner pins must already be in the environment
# here; both are inert on the CPU backend.
if JAX_IMPORT_ERROR is None:
    apply_cuda_xla_flag_pins()
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


@contextmanager
def jax_runtime_isolation():
    """Restore environment, JAX configuration and kernel caches around one case."""
    with ExitStack() as cleanup:
        env_snapshot = {
            name: os.environ.get(name) for name in _BACKEND_RUNTIME_ENV_VARS
        }
        jax_config_snapshot = _snapshot_loaded_jax_runtime_config()
        # LIFO cleanup restores state before clearing compiled executables.
        cleanup.callback(gc.collect)
        cleanup.callback(jax.clear_caches)
        cleanup.callback(_invalidate_loaded_backend_state)
        cleanup.callback(_restore_loaded_jax_runtime_config, jax_config_snapshot)
        cleanup.callback(_restore_backend_runtime_env, env_snapshot)
        _invalidate_loaded_backend_state()
        yield


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
        raise SkipTest("CUDA GPU not available")
    if lane == "cpu":
        raise SkipTest("CPU JAX backend not available")


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


class CompilationLog(logging.Handler):
    """Collect JAX compilation records, including phases with no records."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()


@contextmanager
def compilation_logs() -> Iterator[CompilationLog]:
    """Capture DEBUG records and restore the logger and handler on every exit."""
    logger = logging.getLogger("jax._src.compiler")
    handler = CompilationLog()
    with ExitStack() as cleanup:
        cleanup.callback(logger.setLevel, logger.level)
        cleanup.callback(logger.removeHandler, handler)
        cleanup.callback(handler.close)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        yield handler


def place_float64(values, reference: jax.Array) -> jax.Array:
    """``values`` as a float64 array placed like ``reference``."""
    return explicit_device_array(values, dtype=np.float64, reference=reference)


@contextmanager
def jax_compilations() -> Iterator[list[str]]:
    """Names of the JAX trace, lowering and compile events inside the block."""
    events: list[str] = []

    def record(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event.startswith("/jax/core/compile/"):
            events.append(event)

    jax.monitoring.register_event_duration_secs_listener(record)
    try:
        yield events
    finally:
        jax.monitoring.unregister_event_duration_listener(record)


def assert_matches_native(
    actual, expected, name: str, rtol: float = 1e-12, scale: float | None = None
) -> None:
    """``actual`` has native's shape and agrees with native's ``expected`` to
    ``rtol`` times ``scale``, by default the largest native entry (a float64
    round-off bound where JAX and native sum in different orders)."""
    actual = host_array(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != native {expected.shape}"
    reference = np.max(np.abs(expected), initial=0.0) if scale is None else scale
    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=rtol * reference, err_msg=f"{name} differs from native"
    )
