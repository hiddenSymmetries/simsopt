import unittest
import warnings

import numpy as np

from simsopt.configs import get_data
from simsopt.configs.LHD_like import get_LHD_like_data
from simsopt.field import BiotSavart, Coil
from simsopt.geo import CurveXYZFourier
import simsoptpp as sopp


class Tests(unittest.TestCase):
    def test_axis(self):
        """
        If we trace a field line starting from the expected magnetic axis, it should
        match the purported axis.
        """
        base_curves, base_currents, axis, nfp, bs = get_data("lhd_like")
        # Flip the sign of current so B points towards +phi. Otherwise
        # fieldline_tracing traces towards -phi.
        coils = [
            Coil(curve, -1*current) for curve, current in zip(base_curves, base_currents)
        ]
        field = BiotSavart(coils)

        axis_gamma = axis.gamma()
        expected_R_axis = 3.629918012474283
        _, res_phi_hit = sopp.fieldline_tracing(
            field,
            [expected_R_axis, 0, 0],
            tmax=10.0,
            tol=1e-10,
            phis=axis.quadpoints * 2 * np.pi,
            stopping_criteria=[],
        )
        # At each phi, compare xyz from fieldline_tracing to the expected axis:
        n_checks = 0
        for item in res_phi_hit:
            np.testing.assert_allclose(
                item[2:],
                axis_gamma[int(item[1]), :],
                atol=1e-9,
            )
            n_checks += 1
        # Make sure tmax was sufficient to check all points:
        np.testing.assert_array_less(len(axis.quadpoints), n_checks)

    def test_get_LHD_like_data_deprecated_wrapper(self):
        """
        The legacy get_LHD_like_data() shim should emit DeprecationWarning,
        return three items, and stay numerically consistent with get_data().
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            curves, currents, axis = get_LHD_like_data(
                numquadpoints_circular=40,
                numquadpoints_helical=80,
                numquadpoints_axis=10,
            )

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        self.assertEqual(len(deprecation_warnings), 1)
        self.assertIn("get_data('lhd_like'",
                      str(deprecation_warnings[0].message))

        # LHD-like always has 6 circular + 2 helical coils.
        self.assertEqual(len(curves), 8)
        self.assertEqual(len(currents), 8)
        # The first six are circular CurveXYZFourier coils.
        for curve in curves[:6]:
            self.assertIsInstance(curve, CurveXYZFourier)
        self.assertEqual(axis.quadpoints.size, 10)

        # The unified loader returns the same first three outputs.
        ref_curves, ref_currents, ref_axis, _, _ = get_data(
            "lhd_like",
            numquadpoints_circular=40,
            numquadpoints_helical=80,
            numquadpoints_axis=10,
        )
        np.testing.assert_allclose(curves[0].gamma(), ref_curves[0].gamma())
        np.testing.assert_allclose(currents[0].get_value(),
                                   ref_currents[0].get_value())
        np.testing.assert_allclose(axis.gamma(), ref_axis.gamma())
