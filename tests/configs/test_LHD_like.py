import unittest
import numpy as np

from simsopt.configs import get_LHD_like_data
from simsopt.field import BiotSavart, Coil, Current
import simsoptpp as sopp


class Tests(unittest.TestCase):
    def test_axis(self):
        """
        If we trace a field line starting from the expected magnetic axis, it should
        match the purported axis.
        """
        coils, currents, axis = get_LHD_like_data()
        # Flip the sign of current so B points towards +phi. Otherwise
        # fieldline_tracing traces towards -phi.
        currents = -np.array(currents)
        coils = [
            Coil(curve, Current(current)) for curve, current in zip(coils, currents)
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
