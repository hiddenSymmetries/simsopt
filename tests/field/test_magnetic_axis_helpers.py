#!/usr/bin/env python3
import unittest
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import coils_via_symmetries
from simsopt.field.magnetic_axis_helpers import compute_on_axis_iota
from simsopt.configs.zoo import get_ncsx_data, get_hsx_data, get_giuliani_data


class MagneticAxisHelpers(unittest.TestCase):

    def test_magnetic_axis_iota(self):
        """
        Verify that the rotational transform can be computed on axis
        """
        for (get_data, target_iota) in zip([get_hsx_data, get_ncsx_data, get_giuliani_data], [1.0418687161633922, 0.39549339846119463, 0.42297724084249616]):
            self.subtest_magnetic_axis_iota(get_data, target_iota)

    def subtest_magnetic_axis_iota(self, get_data, target_iota):
        curves, currents, ma = get_data()
        coils = coils_via_symmetries(curves, currents, ma.nfp, True)
        iota = compute_on_axis_iota(ma, BiotSavart(coils))
        np.testing.assert_allclose(iota, target_iota, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
