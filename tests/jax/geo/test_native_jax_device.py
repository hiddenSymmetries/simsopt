"""Native precision/platform settings and C++ curve length host ownership."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import os
import subprocess
import sys

import jax
import pytest


def _fresh_environment():
    return {
        name: value for name, value in os.environ.items()
        if not name.startswith(("JAX_", "SIMSOPT_"))
    }


@pytest.mark.parametrize("selector", ["set_backend", "environment"])
def test_removed_float32_mode_is_rejected_in_fresh_process(selector):
    environment = _fresh_environment()
    environment["JAX_PLATFORMS"] = "cpu"
    if selector == "environment":
        environment["SIMSOPT_BACKEND_MODE"] = "jax_cpu_float32_smoke"
    code = '''
import sys
import pytest
from simsopt_jax.backend import get_backend_config, set_backend

with pytest.raises(ValueError, match="Backend mode 'jax_cpu_float32_smoke' is not valid. Accepted:") as error:
    if sys.argv[1] == "set_backend":
        set_backend("jax_cpu_float32_smoke")
    else:
        get_backend_config()
assert "jax_cpu_float32_smoke" not in str(error.value).split("Accepted:", 1)[1]
'''
    subprocess.run([sys.executable, "-c", code, selector], env=environment, check=True)


@pytest.mark.parametrize("platform_variable", [None, "JAX_PLATFORMS", "JAX_PLATFORM_NAME"])
def test_native_import_preserves_platform_selection_in_fresh_process(platform_variable):
    environment = _fresh_environment()
    if platform_variable is not None:
        environment[platform_variable] = "cpu"
    code = '''
import os
import jax
from simsopt import geo

explicit_platform = "JAX_PLATFORMS" in os.environ or "JAX_PLATFORM_NAME" in os.environ
assert jax.config.values["jax_platform_name"] == ("cpu" if not explicit_platform or "JAX_PLATFORM_NAME" in os.environ else "")
assert jax.default_backend() == "cpu"
assert jax.config.jax_enable_x64
'''
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


@pytest.mark.parametrize("x64_setting", ["false", "0"])
def test_native_import_respects_explicit_x64_setting_in_fresh_process(x64_setting):
    environment = _fresh_environment()
    environment["JAX_ENABLE_X64"] = x64_setting
    code = '''
import jax
from simsopt import geo

assert jax.default_backend() == "cpu"
assert not jax.config.jax_enable_x64
'''
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


def test_native_jax_curve_constructed_before_parity_backend_in_fresh_process(tmp_path):
    environment = _fresh_environment()
    environment.update(
        SIMSOPT_BACKEND_MODE="jax_cpu_parity",
        SIMSOPT_JAX_COMPILATION_CACHE_DIR=str(tmp_path / "compilation-cache"),
    )
    code = '''
import jax
import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import CurveHelical, CurveLength
from simsopt_jax.backend import set_backend
from simsopt_jax_adapters.field import BiotSavartJAX

curve = CurveHelical(32, 2)
set_backend("jax", device="cpu", intent="parity")
field = BiotSavartJAX([Coil(curve, Current(1e5))])
length_value = CurveLength(curve).J()
assert jax.config.jax_enable_x64
assert np.isfinite(length_value)
np.testing.assert_allclose(length_value, np.mean(np.linalg.norm(curve.gammadash(), axis=1)), rtol=1e-14, atol=1e-14)
field.set_points(np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]]))
value = field.B()
assert value.dtype == np.float64
assert np.all(np.isfinite(jax.device_get(value)))
'''
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


_GPU_DETERMINISM_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"


def _with_gpu_determinism(environment):
    environment["XLA_FLAGS"] = " ".join(
        flag for flag in (environment.get("XLA_FLAGS"), _GPU_DETERMINISM_FLAG) if flag
    )
    return environment


@pytest.mark.parametrize("platforms_after_import", [None, "cuda,cpu"])
def test_set_backend_default_platform_after_native_import_in_fresh_process(
    platforms_after_import,
):
    """``set_backend`` makes its platform the default after ``simsopt.geo`` set ``cpu``.

    Without JAX platform variables, ``simsopt.geo`` sets the legacy
    ``jax_platform_name="cpu"``, which JAX consults before ``jax_platforms``.
    The child stubs JAX's backend table (a CUDA backend, plus CPU when listed)
    so the CPU-only CI host resolves the default backend exactly as a CUDA host
    would: before the fix it was ``Unknown backend cpu`` or, with ``cuda,cpu``,
    CPU while CUDA was requested.
    """
    environment = _with_gpu_determinism(_fresh_environment())
    code = '''
import os
import sys

import jax
from jax._src import xla_bridge
from simsopt import geo  # noqa: F401
from simsopt_jax.backend import set_backend

assert jax.config.values["jax_platform_name"] == "cpu"
platforms_after_import = sys.argv[1]
if platforms_after_import:
    os.environ["JAX_PLATFORMS"] = platforms_after_import


class _Backend:
    def __init__(self, platform):
        self.platform = platform


cuda = _Backend("gpu")
backends = {"cuda": cuda}
if platforms_after_import:
    backends["cpu"] = _Backend("cpu")
xla_bridge.backends = lambda: backends
xla_bridge._default_backend = cuda

set_backend("jax", device="gpu", intent="parity")
assert jax.config.jax_platforms == (platforms_after_import or "cuda")
assert jax.config.values["jax_platform_name"] == ""
assert jax.default_backend() == "gpu"
'''
    subprocess.run(
        [sys.executable, "-c", code, platforms_after_import or ""],
        env=environment,
        check=True,
    )


@pytest.mark.parametrize(
    ("device", "jax_platforms"),
    [("cpu", "cpu"), ("gpu", "cuda,cpu"), ("gpu", None)],
    ids=["cpu", "gpu-cuda-cpu-platforms", "gpu-no-platform-env"],
)
def test_native_stage_two_with_explicit_platforms_in_fresh_process(
    tmp_path, device, jax_platforms
):
    """Native stage II after ``set_backend``; ``gpu-no-platform-env`` exports no
    JAX platform variable, so ``simsopt.geo`` sets ``cpu`` before ``set_backend``
    and native length kernels run on CUDA alone."""
    if device == "gpu" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA JAX device required")
    environment = _with_gpu_determinism(_fresh_environment())
    environment["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = str(
        tmp_path / "compilation-cache"
    )
    if jax_platforms is not None:
        environment["JAX_PLATFORMS"] = jax_platforms
    native_platform = "gpu" if jax_platforms is None else "cpu"
    code = '''
import sys

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.geo.jit import native_jax_device
from simsopt.objectives import SquaredFlux
from simsopt_jax.backend import get_runtime_jax_device, set_backend
from simsopt_jax_adapters.field import BiotSavartJAX

device, native_platform, platform_env = sys.argv[1:]
assert jax.config.values["jax_platform_name"] == ("" if platform_env else "cpu")
set_backend("jax", device=device, intent="parity")
assert jax.config.values["jax_platform_name"] == ""
assert jax.default_backend() == device
assert jnp.ones(2).device.platform == device
runtime_device = get_runtime_jax_device()
assert runtime_device.platform == device
assert native_jax_device().platform == native_platform

curve = create_equally_spaced_curves(1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32)[0]
length = CurveLength(curve)
with jax.transfer_guard("disallow"):
    length_value, length_gradient = length.J(), length.dJ()
assert isinstance(length_value, np.float64)
assert isinstance(length_gradient, np.ndarray)
np.testing.assert_allclose(length_value, np.mean(np.linalg.norm(curve.gammadash(), axis=1)), rtol=1e-14, atol=1e-14)
expected_gradient = np.mean(np.einsum("ij,ijk->ik", curve.gammadash() / np.linalg.norm(curve.gammadash(), axis=1)[:, None], curve.dgammadash_by_dcoeff()), axis=0)
np.testing.assert_allclose(length_gradient, expected_gradient, rtol=1e-12, atol=1e-14)

surface = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
surface.set_rc(0, 0, 1.0)
surface.set_rc(1, 0, 0.3)
surface.set_zs(1, 0, 0.3)
coils = [Coil(curve, Current(1e5))]
native, adapter = BiotSavart(coils), BiotSavartJAX(coils)
native_flux, adapter_flux = SquaredFlux(surface, native), SquaredFlux(surface, adapter)
value = adapter.B()
assert value.devices() == {runtime_device}
np.testing.assert_allclose(jax.device_get(value), native.B(), rtol=1e-12, atol=1e-14)
np.testing.assert_allclose(adapter_flux.J(), native_flux.J(), rtol=1e-12, atol=1e-14)
np.testing.assert_allclose(adapter_flux.dJ(), native_flux.dJ(), rtol=1e-12, atol=1e-14)
assert jax.default_backend() == device
assert jnp.ones(2).device.platform == device
'''
    subprocess.run(
        [sys.executable, "-c", code, device, native_platform, jax_platforms or ""],
        env=environment,
        check=True,
    )
