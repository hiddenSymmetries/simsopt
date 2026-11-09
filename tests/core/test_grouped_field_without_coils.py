"""A coil set without coils has no field, whatever form the points take."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    from simsopt_jax.core.field import (
        grouped_biot_savart_A_from_spec,
        grouped_biot_savart_B_and_dB_from_spec,
        grouped_biot_savart_B_from_spec,
        grouped_biot_savart_d2B_by_dXdX_from_spec,
        grouped_biot_savart_dB_by_dX_from_spec,
    )
    from simsopt_jax.core.specs import make_grouped_coil_set_spec

import numpy as np


class TestGroupedFieldWithoutCoils(JaxTestCase):
    def test_empty_coil_set_gives_zero_field_for_list_points(self):
        """B, A and their derivatives vanish, shaped (P, 3, ...), for list points."""
        points = [[1.0, 0.0, 0.1], [0.9, 0.2, -0.1]]
        spec = make_grouped_coil_set_spec(())
        rows = [
            ("B", grouped_biot_savart_B_from_spec, (2, 3)),
            ("A", grouped_biot_savart_A_from_spec, (2, 3)),
            ("dB", grouped_biot_savart_dB_by_dX_from_spec, (2, 3, 3)),
            ("d2B", grouped_biot_savart_d2B_by_dXdX_from_spec, (2, 3, 3, 3)),
        ]
        for name, kernel, shape in rows:
            with self.subTest(name):
                np.testing.assert_array_equal(kernel(points, spec), np.zeros(shape))
        B, dB = grouped_biot_savart_B_and_dB_from_spec(points, spec)
        np.testing.assert_array_equal(B, np.zeros((2, 3)))
        np.testing.assert_array_equal(dB, np.zeros((2, 3, 3)))
