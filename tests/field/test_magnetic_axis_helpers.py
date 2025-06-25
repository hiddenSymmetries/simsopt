#!/usr/bin/env python3
import unittest
import numpy as np
from simsopt.field.magnetic_axis_helpers import compute_on_axis_iota
from simsopt.configs.zoo import get_data

class MagneticAxisHelpers(unittest.TestCase):

    def test_magnetic_axis_iota(self):
        """
        Verify that the rotational transform can be computed on axis
        """
        for (config, target_iota) in zip(["hsx", "ncsx", "giuliani"], [1.0418687161633922, 0.39549339846119463, 0.42297724084249616]):
            self.subtest_magnetic_axis_iota(config, target_iota)

    def subtest_magnetic_axis_iota(self, config, target_iota):
        curves, currents, ma, nfp, bs = get_data(config)
        iota = compute_on_axis_iota(ma, bs)
        np.testing.assert_allclose(iota, target_iota, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
