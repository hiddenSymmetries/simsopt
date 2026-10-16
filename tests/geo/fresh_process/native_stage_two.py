"""Fresh-process child: native stage II after ``set_backend`` on the selected device.

Usage: ``native_stage_two.py DEVICE NATIVE_PLATFORM PLATFORM_ENV``.
``PLATFORM_ENV`` is the ``JAX_PLATFORMS`` value the parent exported before
this process started, or empty when it exported none.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.geo.jit import native_jax_device
from simsopt.objectives import SquaredFlux
from simsopt_jax.backend import get_runtime_jax_device, set_backend
from simsopt_jax.backend.runtime import JaxDevice
from simsopt_jax_adapters.field import JaxBiotSavart

_DEVICES: dict[str, JaxDevice] = {"cpu": "cpu", "gpu": "gpu"}


def main(device: str, native_platform: str, platform_env: str) -> None:
    """Compare native and adapter length, field and flux on ``device``."""
    assert jax.config.values["jax_platform_name"] == ("" if platform_env else "cpu")
    set_backend("jax", device=_DEVICES[device], intent="parity")
    assert jax.config.values["jax_platform_name"] == ""
    assert jax.default_backend() == device
    assert {d.platform for d in jnp.ones(2).devices()} == {device}
    runtime_device = get_runtime_jax_device()
    assert runtime_device is not None and runtime_device.platform == device
    assert native_jax_device().platform == native_platform

    curve = create_equally_spaced_curves(
        1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32
    )[0]
    length = CurveLength(curve)
    with jax.transfer_guard("disallow"):
        length_value, length_gradient = length.J(), length.dJ()
    assert isinstance(length_value, np.float64)
    assert isinstance(length_gradient, np.ndarray)
    np.testing.assert_allclose(
        length_value,
        np.mean(np.linalg.norm(curve.gammadash(), axis=1)),
        rtol=1e-14,
        atol=1e-14,
    )
    unit_tangent = (
        curve.gammadash() / np.linalg.norm(curve.gammadash(), axis=1)[:, None]
    )
    expected_gradient = np.mean(
        np.einsum("ij,ijk->ik", unit_tangent, curve.dgammadash_by_dcoeff()), axis=0
    )
    np.testing.assert_allclose(
        length_gradient, expected_gradient, rtol=1e-12, atol=1e-14
    )

    surface = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    coils = [Coil(curve, Current(1e5))]
    native, adapter = BiotSavart(coils), JaxBiotSavart(coils)
    native_flux, adapter_flux = (
        SquaredFlux(surface, native),
        SquaredFlux(surface, adapter),
    )
    value = adapter.B()
    assert value.devices() == {runtime_device}
    np.testing.assert_allclose(
        jax.device_get(value), native.B(), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        adapter_flux.J(), native_flux.J(), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        adapter_flux.dJ(), native_flux.dJ(), rtol=1e-12, atol=1e-14
    )
    assert jax.default_backend() == device
    assert {d.platform for d in jnp.ones(2).devices()} == {device}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
