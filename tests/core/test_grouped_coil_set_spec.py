"""Grouped-coil construction preserves supplied specs and owns tuple inputs."""

from __future__ import annotations

# Initialize the shared JAX runtime before importing kernel modules.
# ruff: noqa: I001
from unittest_jax_support import JaxTestCase, aligned_host_copy, optional_jax_imports

import numpy as np

with optional_jax_imports():
    from simsopt_jax.core.specs import (
        make_coil_group_spec,
        make_grouped_coil_set_spec,
    )


class TestGroupedCoilSetSpec(JaxTestCase):
    def test_existing_specs_retain_identity_and_order(self):
        """Existing specs, including repeated references, retain identity and order."""
        first = make_coil_group_spec([[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]], [7.0], (5,))
        second = make_coil_group_spec([[[8.0, 9.0, 10.0]]], [[[11.0, 12.0, 13.0]]], [14.0], (2,))

        result = make_grouped_coil_set_spec([first, second, first])

        self.assertEqual(len(result.groups), 3)
        self.assertIs(result.groups[0], first)
        self.assertIs(result.groups[1], second)
        self.assertIs(result.groups[2], first)

    def test_tuple_inputs_are_converted_and_snapshot_host_arrays(self):
        """Tuple conversion retains float64 values after callers mutate host inputs."""
        gammas = aligned_host_copy([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        gammadashs = aligned_host_copy([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
        currents = aligned_host_copy([13.0])

        result = make_grouped_coil_set_spec([(gammas, gammadashs, currents, (4,))])
        gammas[:] = -1.0
        gammadashs[:] = -2.0
        currents[:] = -3.0

        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertEqual(group.coil_indices, (4,))
        np.testing.assert_array_equal(group.gammas, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        np.testing.assert_array_equal(group.gammadashs, [[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
        np.testing.assert_array_equal(group.currents, [13.0])
        for array in (group.gammas, group.gammadashs, group.currents):
            self.assertEqual(array.dtype, np.dtype("float64"))

    def test_mixed_one_pass_generator_preserves_order(self):
        """A mixed generator is consumed once without dropping or reordering groups."""
        first = make_coil_group_spec([[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]], [7.0], (8,))
        middle = (first.gammas, first.gammadashs, first.currents, (3,))
        last = make_coil_group_spec([[[8.0, 9.0, 10.0]]], [[[11.0, 12.0, 13.0]]], [14.0], (1,))
        groups = (group for group in (first, middle, last))

        result = make_grouped_coil_set_spec(groups)

        self.assertEqual(tuple(group.coil_indices for group in result.groups), ((8,), (3,), (1,)))
        self.assertIs(result.groups[0], first)
        self.assertIs(result.groups[2], last)
        np.testing.assert_array_equal(result.groups[1].currents, [7.0])
        self.assertEqual(tuple(groups), ())

    def test_empty_iterable_produces_empty_groups(self):
        """An empty iterable constructs an empty immutable groups tuple."""
        result = make_grouped_coil_set_spec(iter(()))

        self.assertEqual(result.groups, ())
