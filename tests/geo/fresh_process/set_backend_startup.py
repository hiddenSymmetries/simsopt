"""Fresh-process child ``set_backend_startup.py SCENARIO [ARGUMENT...]``: ``set_backend`` and native geometry from a
fresh JAX, under the ``JAX_PLATFORMS`` (or none) the parent exports, since JAX picks its backends once, at first use,
which no import below makes; without a JAX platform variable, importing ``simsopt.geo`` sets
``jax_platform_name="cpu"``."""

import os
import sys

import jax
import jax.extend.backend
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveHelical, CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.geo.jit import native_jax_device
from simsopt.objectives import SquaredFlux
from simsopt_jax.backend import set_backend
from simsopt_jax.backend.runtime import JaxDevice, get_runtime_jax_device
from simsopt_jax_adapters.field import JaxBiotSavart

_DEVICES: dict[str, JaxDevice] = {"cpu": "cpu", "gpu": "gpu"}
# Selecting CUDA also starts the CPU, where native jitted kernels run.
_JAX_PLATFORMS = {"cpu": "cpu", "gpu": "cuda,cpu"}


def _initialized_platforms() -> set[str]:
    return set(jax.extend.backend.backends())


def _mean_speed(curve):
    return np.mean(np.linalg.norm(curve.gammadash(), axis=1))


def curve_before_set_backend() -> None:
    """A native JAX curve built before backend setup stays float64: construct the curve, then set the backend and
    evaluate length and field."""
    curve = CurveHelical(32, 2)
    set_backend("jax", device="cpu")
    field = JaxBiotSavart([Coil(curve, Current(1e5))])
    length_value = CurveLength(curve).J()
    assert jax.config.values["jax_enable_x64"]
    assert np.isfinite(length_value)
    assert_allclose(length_value, _mean_speed(curve), rtol=1e-14, atol=1e-14)
    field.set_points(np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]]))
    value = field.B()
    assert value.dtype == np.float64
    assert np.all(np.isfinite(jax.device_get(value)))


def after_native_import(device: str, platforms_after_import: str) -> None:
    """``set_backend`` overrides the platform ``simsopt.geo`` chose: the parent exports no JAX platform variable, a
    non-empty ``platforms_after_import`` is exported only after the import, as a driver would, and ``device`` must
    then be JAX's default backend."""
    assert jax.config.values["jax_platform_name"] == "cpu"
    if platforms_after_import:
        os.environ["JAX_PLATFORMS"] = platforms_after_import
    set_backend("jax", device=_DEVICES[device])
    assert jax.config.values["jax_platforms"] == (platforms_after_import or _JAX_PLATFORMS[device])
    assert jax.config.values["jax_platform_name"] == ""
    assert jax.default_backend() == device


def stage_two(device: str, platform_env: str) -> None:
    """Native and adapter length, field and flux agree after ``set_backend`` on ``device``; ``platform_env`` is the
    exported ``JAX_PLATFORMS``, or empty when none was."""
    assert jax.config.values["jax_platform_name"] == ("" if platform_env else "cpu")
    set_backend("jax", device=_DEVICES[device])
    assert jax.config.values["jax_platform_name"] == ""
    assert jax.default_backend() == device
    assert {d.platform for d in jnp.ones(2).devices()} == {device}
    runtime_device = get_runtime_jax_device()
    assert runtime_device is not None and runtime_device.platform == device

    curve = create_equally_spaced_curves(1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32)[0]
    length = CurveLength(curve)
    length_value, length_gradient = length.J(), length.dJ()
    assert_allclose(length_value, _mean_speed(curve), rtol=1e-14, atol=1e-14)
    tangent = curve.gammadash() / np.linalg.norm(curve.gammadash(), axis=1)[:, None]
    projected = np.einsum("ij,ijk->ik", tangent, curve.dgammadash_by_dcoeff())
    assert_allclose(length_gradient, np.mean(projected, axis=0), rtol=1e-12, atol=1e-14)

    surface = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    coils = [Coil(curve, Current(1e5))]
    native, adapter = BiotSavart(coils), JaxBiotSavart(coils)
    native_flux, adapter_flux = (SquaredFlux(surface, native), SquaredFlux(surface, adapter))
    value = adapter.B()
    assert value.devices() == {runtime_device}
    assert_allclose(jax.device_get(value), native.B(), rtol=1e-12, atol=1e-14)
    assert_allclose(adapter_flux.J(), native_flux.J(), rtol=1e-12, atol=1e-14)
    assert_allclose(adapter_flux.dJ(), native_flux.dJ(), rtol=1e-12, atol=1e-14)
    assert jax.default_backend() == device
    assert {d.platform for d in jnp.ones(2).devices()} == {device}


def gpu_at_startup() -> None:
    """Selecting the GPU first starts CUDA as the default and the CPU for native kernels."""
    set_backend("jax", device="gpu")
    assert jax.default_backend() == "gpu"
    assert _initialized_platforms() == {"cuda", "cpu"}, _initialized_platforms()
    assert native_jax_device().platform == "cpu", native_jax_device()
    assert {d.platform for d in jnp.zeros(2).devices()} == {"gpu"}
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"


def cpu_then_gpu_after_initialization() -> None:
    """After initialization, selections move the default device but not the platforms."""
    assert _initialized_platforms() == {"cuda", "cpu"}, _initialized_platforms()
    for device in ("cpu", "gpu"):
        set_backend("jax", device=device)
        assert {d.platform for d in jnp.zeros(2).devices()} == {device}
        assert native_jax_device().platform == "cpu", (device, native_jax_device())
        assert jax.config.values["jax_platforms"] == "cuda,cpu"
        assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"


def explicit_cuda_only() -> None:
    """An explicit ``JAX_PLATFORMS=cuda`` starts CUDA alone; native kernels use it."""
    set_backend("jax", device="gpu")
    assert _initialized_platforms() == {"cuda"}, _initialized_platforms()
    assert native_jax_device().platform == "gpu", native_jax_device()
    assert os.environ["JAX_PLATFORMS"] == "cuda"


def cpu_before_initialization() -> None:
    """Selecting the CPU before initialization keeps CUDA from starting."""
    set_backend("jax", device="cpu")
    assert _initialized_platforms() == {"cpu"}, _initialized_platforms()
    assert jax.default_backend() == "cpu"
    assert native_jax_device().platform == "cpu", native_jax_device()


if __name__ == "__main__":
    scenarios = (curve_before_set_backend, after_native_import, stage_two, gpu_at_startup,
                 cpu_then_gpu_after_initialization, explicit_cuda_only, cpu_before_initialization)
    {f.__name__: f for f in scenarios}[sys.argv[1]](*sys.argv[2:])
