"""Fresh-process child ``distance_adapters.py SCENARIO KIND`` (``uninitialized`` or ``device_scopes``; ``curve_curve``
or ``curve_surface``) for ``test_curve_objectives_jax``: JAX picks its backends once, at first use."""

import sys
from unittest import mock

import jax
import numpy as np

from simsopt.geo import CurveCurveDistance, CurveSurfaceDistance, SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.backend import set_backend
from simsopt_jax_adapters.geo import JaxCurveCurveDistance, JaxCurveSurfaceDistance


def _objectives(kind):
    """The JAX adapter and native objective of ``kind``; the curves are 0.707 m apart and 0.541 m from the surface
    samples, so both thresholds select candidates and the adapters place operands."""
    curves = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=12)
    surface = SurfaceRZFourier.from_nphi_ntheta(nphi=4, ntheta=5)
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    if kind == "curve_curve":
        return JaxCurveCurveDistance(curves, 0.8), CurveCurveDistance(curves, 0.8)
    return JaxCurveSurfaceDistance(curves, surface, 0.6), CurveSurfaceDistance(curves, surface, 0.6)


def _require_uninitialized_backends(num_cpu_devices):
    """Set ``jax_num_cpu_devices``; JAX raises ``RuntimeError`` if a backend is already initialized."""
    jax.config.update("jax_num_cpu_devices", num_cpu_devices)


def uninitialized(kind):
    """Host shortest-distance queries match native and leave the JAX backends uninitialized."""
    adapter, native = _objectives(kind)
    _require_uninitialized_backends(1)
    for _ in range(2):
        np.testing.assert_allclose(adapter.shortest_distance(), native.shortest_distance(), rtol=1e-14)
    _require_uninitialized_backends(2)
    set_backend("jax", device="cpu")
    assert len(jax.devices("cpu")) == 2, "the CPU backend was created before shortest_distance returned"
    np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(adapter.dJ(), native.dJ(), rtol=1e-11, atol=1e-13)


def device_scopes(kind):
    """Changing default devices replaces operands while reusing host candidates."""
    adapter, native = _objectives(kind)
    set_backend("jax", device="cpu")
    first, second = jax.devices("cpu")[:2]
    with mock.patch.object(adapter, "_candidates", wraps=adapter._candidates) as search:
        for device in (first, second, first):
            with jax.default_device(device):
                snapshot = adapter._snapshot()
                assert snapshot is not None, "the test state must have candidates"
                assert adapter._snapshot() is snapshot, "same-scope operands should be reused"
                assert all(array.devices() == {device} for array in jax.tree.leaves(snapshot.operands)), device
                with jax.transfer_guard("disallow"):
                    value, gradient = adapter.J(), adapter.dJ()
                    shortest = adapter.shortest_distance()
            np.testing.assert_allclose(value, native.J(), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(gradient, native.dJ(), rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(shortest, native.shortest_distance(), rtol=1e-14)
    assert search.call_count == 1, "placement changes must reuse host candidates"


if __name__ == "__main__":
    {"uninitialized": uninitialized, "device_scopes": device_scopes}[sys.argv[1]](sys.argv[2])
