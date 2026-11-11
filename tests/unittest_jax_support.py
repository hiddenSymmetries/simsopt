"""Shared optional-JAX gating and runtime isolation for unittest."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TypeVar, cast
from unittest import SkipTest, TestCase, skipIf

import numpy as np

# Only a failure of simsopt_jax's requirement gate (Python, jax and jaxlib versions)
# or a missing jax makes the runtime unavailable; an ImportError raised after the
# gate loaded is a broken package import and fails the run instead of skipping it.
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
    from simsopt_jax.backend.runtime import _export_xla_flag_defaults
    from simsopt_jax.backend.runtime import get_backend_config, set_backend
    from simsopt_jax.backend.runtime import host_array as _host_array
    from simsopt_jax.backend.runtime import host_tree
    import jax.numpy as jnp
    from simsopt_jax.backend.jit import jit
    from simsopt._core.derivative import Derivative
    from simsopt._core.optimizable import Optimizable
    from simsopt.geo import CurveXYZFourier, SurfaceRZFourier


@skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
class JaxTestCase(TestCase):
    """Skip without a supported JAX; isolate each test method and each case."""

    def setUp(self) -> None:
        self.patches = self.enterContext(self.case())

    @contextmanager
    def case(self) -> Iterator[ExitStack]:
        """Yield a resource stack that unwinds before restoring JAX runtime state."""
        with ExitStack() as patches:
            patches.enter_context(jax_runtime_isolation())
            yield patches

    def run_cases(self, check: Callable[..., None], cases: Iterable[dict]) -> None:
        """Run ``check(**case)`` for each case in its own ``subTest`` and ``case()``."""
        for case in cases:
            with self.subTest(**case), self.case():
                check(**case)


# XLA reads XLA_FLAGS when a backend starts, and test modules probe devices at
# collection, so the GPU flags (inert on the CPU) are exported here. Every platform
# starts here, once: a case whose set_backend(device=...) narrowed jax_platforms
# before the first device access would leave the others missing for the process.
if JAX_IMPORT_ERROR is None:
    _export_xla_flag_defaults("gpu")
    jax.config.update("jax_enable_x64", True)
    if jax.config.jax_enable_x64 is not True:
        raise RuntimeError("unittest_jax_support.py requires jax_enable_x64=True")
    jax.extend.backend.backends()

_BACKEND_RUNTIME_ENV_VARS = ("JAX_PLATFORMS", "XLA_FLAGS")  # what set_backend exports
_JAX_RUNTIME_CONFIG = (
    *("jax_enable_x64", "jax_debug_nans", "jax_disable_jit", "jax_transfer_guard"),
    *("jax_platforms", "jax_platform_name", "jax_compilation_cache_dir"),
    "jax_default_device",
)


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _restore_config(snapshot: dict[str, object]) -> None:
    for name, value in snapshot.items():
        jax.config.update(name, value)


@contextmanager
def jax_runtime_isolation() -> Iterator[None]:
    """Restore the backend, environment, JAX configuration and kernel caches."""
    with ExitStack() as cleanup:
        env = {name: os.environ.get(name) for name in _BACKEND_RUNTIME_ENV_VARS}
        config = {name: jax.config.values[name] for name in _JAX_RUNTIME_CONFIG}
        backend = get_backend_config()
        # LIFO: the environment, the backend (its callbacks clear its kernel caches),
        # the JAX settings, then the compiled executables.
        cleanup.callback(gc.collect)
        cleanup.callback(jax.clear_caches)
        cleanup.callback(_restore_config, config)
        cleanup.callback(set_backend, backend.backend, device=backend.device)
        cleanup.callback(_restore_env, env)
        yield


def parity_rng(seed: int = 0) -> np.random.RandomState:
    """A NumPy generator seeded ``1729 + seed``, the parity tests' inputs."""
    return np.random.RandomState(1729 + seed)


@contextmanager
def parity_default_device(lane: str) -> Iterator[None]:
    """Make the first device of the lane's platform, cpu or gpu, the default."""
    if lane not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown parity lane {lane!r}; expected 'cpu' or 'gpu'.")
    for platform in jax.extend.backend.backends():
        for device in jax.devices(platform):
            if device.platform == lane:
                with jax.default_device(device):
                    yield
                return
    raise SkipTest("CUDA GPU not available" if lane == "gpu" else "CPU JAX backend not available")


def require_cuda(reason: str) -> None:
    """Skip the calling test unless the default JAX backend has a CUDA device."""
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise SkipTest(reason)


def aligned_host_copy(values, *, misaligned: bool = False) -> np.ndarray:
    """Float64 copy on a 64-byte boundary (CPU JAX aliases it), or one past."""
    values = np.asarray(values, dtype=np.float64)
    storage = np.empty(values.size + 8, dtype=np.float64)
    offset = (-storage.ctypes.data % 64) // storage.itemsize + int(misaligned)
    array = storage[offset : offset + values.size].reshape(values.shape)
    array[...] = values
    return array


def distinct_host_array(shape, *, misaligned=False):
    """Distinct values in aligned storage; misaligned storage tests delayed copies."""
    size = int(np.prod(shape))
    values = np.arange(size).reshape(shape) / max(size, 1) + 1.0
    return aligned_host_copy(values, misaligned=misaligned)


def assert_tree_equal(actual, expected):
    """Wait for ``actual``, then require each of its leaves to equal the leaf of ``expected`` in the same place."""
    jax.block_until_ready(actual)
    leaves = zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    for observed, saved in leaves:
        np.testing.assert_array_equal(observed, saved)


def host_snapshot(value):
    """A host copy of every leaf of ``value``, after it is ready."""
    return jax.tree.map(lambda x: np.array(x, copy=True), jax.block_until_ready(value))


_TreeT = TypeVar("_TreeT")


@contextmanager
def disallow_host_transfers() -> Iterator[None]:
    """Refuse IMPLICIT host-to-device transfers for the duration of the block.

    That refusal holds on every backend.  Explicit ``device_get`` passes, and so
    does an explicit ``device_put`` of host values, within one backend, or from
    the CPU to an accelerator.  JAX checks a ``device_put`` of an accelerator
    array to the CPU as an implicit device-to-host copy and refuses it; the
    placement helpers in ``simsopt_jax.backend.dtypes`` gather such arrays with
    ``device_get`` first, so their explicit placements pass.  On CPU, where no
    copy happens, implicit device-to-host (``np.sum(x)``, ``x.tolist()``) passes
    too.

    Returns:
        contextlib.AbstractContextManager[None] object: Scoped transfer guard
            that restores the surrounding setting on exit.
    """

    with jax.transfer_guard("disallow"):
        yield


@contextmanager
def allow_host_transfers() -> Iterator[None]:
    """Permit implicit transfers inside one explicit host-driven boundary.

    For host callers that hand device arrays to compiled code while an outer
    strict guard may be active: lowering a program that closes over device
    arrays reads them back, which the strict guard would refuse. The counterpart
    of :func:`disallow_host_transfers`.

    Returns:
        contextlib.AbstractContextManager[None] object: Scoped transfer guard
            that restores the surrounding setting on exit.
    """

    with jax.transfer_guard("allow"):
        yield


def block_until_ready(value: _TreeT) -> _TreeT:
    """Wait for every JAX leaf and return the same pytree structure and values.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.

    Returns:
        pytree object: Same structure and values after all asynchronous JAX
            leaves are ready.
    """
    return jax.block_until_ready(value)


def host_tree_after_ready(value, *, dtype=None):
    """Wait for a pytree, then materialize its array leaves on the host.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.
        dtype (dtype-like or None): Conversion dtype for materialized NumPy leaves
            and host scalars; default None preserves dtypes. Device arrays are
            materialized before this conversion; each leaf keeps its shape.

    Returns:
        pytree object: Ready host leaves with the same shape per leaf and
            privately owned NumPy array buffers.
    """
    return host_tree(block_until_ready(value), dtype=dtype)


_GATE_SIZE = 2896


def make_execution_gate():
    """Queue enough CPU work for the warmed consumer to remain pending."""
    matrix = jnp.ones((_GATE_SIZE, _GATE_SIZE), dtype=jnp.float64)

    @jit
    def gate(value, matrix):
        step = lambda _, x: (x @ matrix) / _GATE_SIZE
        product = jax.lax.fori_loop(0, 4, step, matrix)
        return jnp.where(product[0, 0] > 0, value, -value)

    return lambda value: gate(value, matrix)


def objective_partials(objective: Optimizable) -> Derivative:
    """The partial derivatives of ``objective``, fixed DOFs included."""
    return cast(Callable[..., Derivative], objective.dJ)(partials=True)


def assert_matches_native_including_nans(
    name: str, native: Optimizable, adapter: Optimizable, *, active: bool = False,
) -> None:
    """Value, free gradient and partials equal to native, non-finite entries included."""
    native_value = float(native.J())
    if active:
        assert native_value > 0.0, f"{name} must be active at the test state"
    np.testing.assert_allclose(adapter.J(), native_value, rtol=1e-12, atol=1e-14, err_msg=name)
    native_gradient = native.dJ()
    if active:
        assert np.all(np.isfinite(native_gradient)), name
    np.testing.assert_allclose(adapter.dJ(), native_gradient, rtol=1e-11, atol=1e-13, err_msg=f"{name} free gradient")
    native_partials, adapter_partials = objective_partials(native), objective_partials(adapter)
    assert set(adapter_partials.data) == set(native_partials.data), name
    for owner, expected in native_partials.data.items():
        # Fixed DOFs keep their partials, as in the native Derivative.
        np.testing.assert_allclose(adapter_partials.data[owner], expected, rtol=1e-11, atol=1e-13,
                                   err_msg=f"{name} partials of {owner.name}")


def segment_curve(x_center, x_amplitude, numquadpoints=15):
    """``x = x_center + x_amplitude cos(2 pi t)``, ``y = z = 0``: zero tangent at ``t = 0``."""
    curve = CurveXYZFourier(numquadpoints, 1)
    curve.set("xc(0)", x_center)
    curve.set("xc(1)", x_amplitude)
    return curve


def circle_curve(center, radius, numquadpoints=20):
    """A circle of ``radius`` about ``center``, parallel to the x-y plane, as a Fourier curve."""
    curve = CurveXYZFourier(numquadpoints, 1)
    curve.set("xc(0)", center[0])
    curve.set("yc(0)", center[1])
    curve.set("zc(0)", center[2])
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    return curve


# Minimum distances at which the squared distance of the two points rounds to the
# squared threshold: with and without fused multiply-adds the candidate test
# differs in the last bit (both directions).
DISTANCE_TIE_CASES = (((0.2, 0.5, 0.3), 0.6164414002968976), ((0.1, 0.4, 0.1), 0.42426406871192857))

_QA_INPUT = Path(__file__).resolve().parent / "test_files" / "input.LandremanPaul2021_QA"


def qa_surface() -> SurfaceRZFourier:
    """Half a field period of the Landreman-Paul QA boundary on an 8 by 9 grid."""
    return SurfaceRZFourier.from_vmec_input(str(_QA_INPUT), range="half period", nphi=8, ntheta=9)


def run_fresh_process(
    *arguments: str | Path, env: Mapping[str, str] | None = None, cwd: Path | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a new interpreter on a script or ``"-c", code`` and its arguments."""
    command = (sys.executable, *map(str, arguments))
    return subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def host_array(value, *, dtype=None) -> np.ndarray:
    """Defined without JAX so test modules import it outside ``optional_jax_imports``."""
    return _host_array(value, dtype=dtype)
