"""Shared optional-JAX gating, runtime isolation and logging for unittest."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from collections.abc import Iterator, Mapping
from pathlib import Path
from unittest import TestCase, SkipTest, mock, skipIf
import logging
import os
import subprocess
import sys
import gc

import numpy as np

# simsopt_jax imports its requirement gate (Python, jax and jaxlib versions)
# before anything else. Only a failure of that gate, or a missing jax, means
# the optional runtime is unavailable; once the gate has loaded, an ImportError
# is a broken package import and must fail the run instead of skipping it.
JAX_IMPORT_ERROR: str | None = None
try:
    import simsopt_jax  # noqa: F401
except ImportError as error:
    if "simsopt_jax._requirements" in sys.modules:
        raise
    JAX_IMPORT_ERROR = str(error)
else:
    import jax
    import jax.extend.backend


@contextmanager
def optional_jax_imports() -> Iterator[None]:
    """Suppress import errors only when the optional JAX runtime is unavailable."""
    try:
        yield
    except ImportError:
        if JAX_IMPORT_ERROR is None:
            raise


with optional_jax_imports():
    from simsopt_jax.backend.runtime import (
        apply_cuda_xla_flag_pins,
        invalidate_backend_cache,
    )
    from simsopt_jax.runtime.host_boundary import host_array as _host_array


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


with optional_jax_imports():
    from simsopt_jax.backend.dtypes import explicit_device_array


def _force_x64(jax_module) -> None:
    jax_module.config.update("jax_enable_x64", True)
    if jax_module.config.jax_enable_x64 is not True:
        raise RuntimeError("unittest_jax_support.py requires jax_enable_x64=True")


# XLA reads ``XLA_FLAGS`` when it initializes a backend, and a JAX test module
# probes devices (lane availability) at collection, before any test installs a
# backend config, so the CUDA autotuner pins must already be in the environment
# here; both are inert on the CPU backend.
#
# Every configured platform is initialized here, once, before any case runs.
# ``set_backend(device=...)`` narrows ``jax_platforms``; if a case did that
# before the process first touched a device, JAX would initialize only that
# platform, and the others would stay missing for the rest of the process even
# after ``jax_runtime_isolation`` restores the configuration.
if JAX_IMPORT_ERROR is None:
    apply_cuda_xla_flag_pins()
    _force_x64(jax)
    jax.extend.backend.backends()

_BACKEND_RUNTIME_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
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
    "jax_default_device": None,
}
_PARITY_SEED_BASE = 1729


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
        cleanup.callback(invalidate_backend_cache)
        cleanup.callback(_restore_loaded_jax_runtime_config, jax_config_snapshot)
        cleanup.callback(_restore_backend_runtime_env, env_snapshot)
        invalidate_backend_cache()
        yield


def parity_seed(seed: int = 0) -> int:
    return _PARITY_SEED_BASE + seed


def parity_rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(parity_seed(seed))


def _parity_device_for_lane(jax_module, lane: str):
    """First device of the lane's platform among every initialized backend.

    ``jax.devices()`` lists only the default backend, so the lookup walks each
    initialized platform; a process with both CUDA and CPU runs both lanes.
    """
    if lane not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown parity lane {lane!r}; expected 'cpu' or 'gpu'.")
    for platform in jax_module.extend.backend.backends():
        for device in jax_module.devices(platform):
            if device.platform == lane:
                return device
    if lane == "gpu":
        raise SkipTest("CUDA GPU not available")
    raise SkipTest("CPU JAX backend not available")


@contextmanager
def parity_default_device(lane: str):
    with jax.default_device(_parity_device_for_lane(jax, lane)):
        yield


def before_backend_initialization():
    """Patch the runtime to see a process whose JAX backends are not initialized."""
    return mock.patch(
        "simsopt_jax.backend.runtime._jax_backends_initialized", return_value=False
    )


def require_cuda(reason: str) -> None:
    """Skip the calling test unless the default JAX backend has a CUDA device."""
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise SkipTest(reason)


def aligned_host_copy(values, *, misaligned: bool = False) -> np.ndarray:
    """Float64 C-contiguous copy of ``values`` on a 64-byte boundary.

    CPU JAX placement aliases such a buffer instead of copying it;
    ``misaligned=True`` starts it one element past the boundary, which JAX copies.
    """
    values = np.asarray(values, dtype=np.float64)
    storage = np.empty(values.size + 8, dtype=np.float64)
    offset = (-storage.ctypes.data % 64) // storage.itemsize + int(misaligned)
    array = storage[offset : offset + values.size].reshape(values.shape)
    array[...] = values
    return array


def run_fresh_process(
    script: Path,
    *arguments: str,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``script`` with ``arguments`` in a new interpreter, capturing its text output."""
    return subprocess.run(
        (sys.executable, str(script), *arguments),
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def host_array(value, *, dtype=None) -> np.ndarray:
    """Defined without JAX so test modules import it outside ``optional_jax_imports``."""
    return _host_array(value, dtype=dtype)


class CompilationLog(logging.Handler):
    """Collect JAX compilation records, including phases with no records."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()


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


def assert_errors_shrink(
    testcase: TestCase,
    errors,
    factor: float,
    floor: float = 0.0,
    msg: str | None = None,
    *,
    initial: float = np.inf,
) -> None:
    """Each error in ``errors`` is below ``factor`` times the one before it
    (``initial`` before the first), or below ``floor`` when that is larger (a
    Taylor test at halving steps)."""
    previous = initial
    for error in errors:
        testcase.assertLess(error, max(floor, factor * previous), msg)
        previous = error


def scaled_direction(x0: np.ndarray, seed: int) -> np.ndarray:
    """A seeded normal direction moving each entry of ``x0`` by a fraction of
    its own size (of one, for entries smaller than one)."""
    return parity_rng(seed).standard_normal(x0.size) * np.maximum(np.abs(x0), 1.0)
