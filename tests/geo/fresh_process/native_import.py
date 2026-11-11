"""Fresh-process child ``native_import.py SCENARIO``: what importing ``simsopt.geo``, the event under test, does to JAX;
it imports nothing from ``simsopt_jax``."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Read before the ``simsopt.geo`` import, which therefore follows it.
_X64_BEFORE_NATIVE_IMPORT = jax.config.values["jax_enable_x64"]

from simsopt.geo import CurveLength, create_equally_spaced_curves


def platform_selection() -> None:
    """``JAX_PLATFORMS=cpu``, ``JAX_PLATFORM_NAME=cpu`` or neither: the import keeps the CPU platform name, default
    backend and x64."""
    explicit_platform = "JAX_PLATFORMS" in os.environ or "JAX_PLATFORM_NAME" in os.environ
    named = not explicit_platform or "JAX_PLATFORM_NAME" in os.environ
    assert jax.config.values["jax_platform_name"] == ("cpu" if named else "")
    assert jax.default_backend() == "cpu"
    assert jax.config.values["jax_enable_x64"]


def x64() -> None:
    """``JAX_PLATFORMS=cpu`` and a disabling ``JAX_ENABLE_X64``: the import enables x64."""
    assert not _X64_BEFORE_NATIVE_IMPORT, "JAX_ENABLE_X64 did not disable x64"
    assert jax.default_backend() == "cpu"
    assert jax.config.values["jax_enable_x64"]


def jit_device_policy() -> None:
    """``JAX_PLATFORMS=cuda,cpu``: CurveLength's jitted value and gradient land on the CPU device."""
    assert jax.default_backend() == "gpu"
    cpu = {jax.devices("cpu")[0]}
    curve = create_equally_spaced_curves(1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32)[0]
    length = CurveLength(curve)
    arclength = curve.incremental_arclength()
    value = length.J()
    arclength_gradient = length.dJ_dl(arclength)
    assert value.devices() == cpu
    assert arclength_gradient.devices() == cpu
    np.testing.assert_allclose(value, np.mean(arclength), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(arclength_gradient, np.full_like(arclength, 1.0 / arclength.size))
    assert {d.platform for d in jnp.ones(2).devices()} == {"gpu"}


if __name__ == "__main__":
    {f.__name__: f for f in (platform_selection, x64, jit_device_policy)}[sys.argv[1]]()
