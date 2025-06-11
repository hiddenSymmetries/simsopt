import unittest
import numpy as np

from simsopt.geo import CurveHelical

class Tests(unittest.TestCase):
    def test_dof_names(self):
        """Check the names of the dofs."""
        curve = CurveHelical(10, 2)
        np.testing.assert_equal(
            curve.local_dof_names,
            ["A_0", "A_1", "A_2", "B_1", "B_2"],
            "CurveHelical dof names are not as expected",
        )
        curve.set("A_0", 1.1)
        curve.set("A_1", 2.2)
        curve.set("A_2", 3.3)
        curve.set("B_1", 4.4)
        curve.set("B_2", 5.5)
        np.testing.assert_equal(
            curve.x,
            [1.1, 2.2, 3.3, 4.4, 5.5],
            "CurveHelical dof values are not as expected",
        )
        np.testing.assert_allclose(
            [curve.get("A_0"), curve.get("A_1"), curve.get("A_2"), curve.get("B_1"), curve.get("B_2")],
            [1.1, 2.2, 3.3, 4.4, 5.5],
            err_msg="CurveHelical dof values from get() are not as expected",
        )

if __name__ == "__main__":
    unittest.main()
