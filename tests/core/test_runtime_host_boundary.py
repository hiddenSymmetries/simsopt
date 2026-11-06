"""Tests for typed JAX runtime and host boundaries."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    from simsopt_jax.runtime.host_boundary import (
        disallow_host_transfers,
    )


import numpy as np


def _cpu_float64(values: list[float]) -> "jax.Array":
    """Place one float64 array on the CPU backend, whatever else is available."""

    cpu_device = jax.devices("cpu")[0]
    assert cpu_device.platform == "cpu"
    placed = jax.device_put(np.asarray(values, dtype=np.float64), cpu_device)
    assert placed.dtype == np.float64
    return placed


class TestRuntimeHostBoundary(JaxTestCase):
    def test_disallow_host_transfers_refuses_an_implicit_host_to_device_transfer(self):
        """The refusal is an owned JAX error type, never a message this test parses.

        Host-to-device is refused on every backend, so this test is not CPU-scoped;
        it still pins the CPU device and x64 so neither can silently change what is
        being asserted.
        """

        with jax.enable_x64(True):
            device_array = _cpu_float64([1.0, 2.0])
            host_values = np.asarray([3.0, 4.0], dtype=np.float64)

            with disallow_host_transfers(), self.assertRaises(
                jax.errors.JaxRuntimeError
            ):
                _ = device_array + host_values

            # Positive control: the same expression is fine outside the guard, so the
            # refusal above is the guard's doing and not a broken expression.
            np.testing.assert_array_equal(
                np.asarray(jax.device_get(device_array + host_values)), [4.0, 6.0]
            )

    def test_disallow_host_transfers_admits_explicit_and_cpu_device_to_host_traffic(
        self,
    ):
        """What the guard does NOT refuse, pinned to the CPU backend.

        Explicit ``device_put``/``device_get`` pass everywhere.  The implicit
        device-to-host case is CPU-ONLY evidence -- on CPU there is no copy for the
        guard to fire on -- so the array is placed on ``jax.devices("cpu")[0]`` and
        the platform asserted, and a CUDA-enabled run cannot flip this test.
        """

        with jax.enable_x64(True):
            device_array = _cpu_float64([1.0, 2.0])

            with disallow_host_transfers():
                placed = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
                fetched = jax.device_get(device_array)
                implicit_sum = float(np.sum(device_array))
                implicit_list = device_array.tolist()

            np.testing.assert_array_equal(
                np.asarray(jax.device_get(placed)), [3.0, 4.0]
            )
            np.testing.assert_array_equal(fetched, [1.0, 2.0])
            self.assertTrue(implicit_sum == 3.0, "implicit_sum == 3.0")
            self.assertTrue(implicit_list == [1.0, 2.0], "implicit_list == [1.0, 2.0]")
