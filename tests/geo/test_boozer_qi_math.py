import json
import unittest
from pathlib import Path

import numpy as np
from simsopt.geo.surfaceobjectives import _get_branches, _normalize_modB_global, _qi_objective_single_surface, _repair_shifted_qi_branches, _squash_left_branch, _squash_right_branch, _stretch_left_branch, _stretch_right_branch


FIXTURE = Path(__file__).resolve().parent.parent / "test_files" / "boozer_qi_reference.json"


def _load_fixture():
    with open(FIXTURE, "r", encoding="ascii") as stream:
        return json.load(stream)

class BoozerQIMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _load_fixture()

    def test_get_branches_regular_case(self):
        case = self.data["get_branches_regular"]
        result = _get_branches(case["phi"], case["Ba"], case["Bj"], case["Bmax"], case["Bmin"])
        np.testing.assert_allclose(result, case["expected"])

    def test_get_branches_plateau_case(self):
        case = self.data["get_branches_plateau"]
        result = _get_branches(case["phi"], case["Ba"], case["Bj"], case["Bmax"], case["Bmin"])
        np.testing.assert_allclose(result, case["expected"])

    def test_get_branches_minimum_case(self):
        case = self.data["get_branches_min"]
        result = _get_branches(case["phi"], case["Ba"], case["Bj"], case["Bmax"], case["Bmin"])
        np.testing.assert_allclose(result, case["expected"])

    def test_get_branches_maximum_case(self):
        case = self.data["get_branches_max"]
        result = _get_branches(case["phi"], case["Ba"], case["Bj"], case["Bmax"], case["Bmin"])
        np.testing.assert_allclose(result, case["expected"])

    def test_left_squash_matches_reference(self):
        case = self.data["left_squash"]
        result = _squash_left_branch(case["input"])
        np.testing.assert_allclose(result, case["expected"])

    def test_right_squash_matches_reference(self):
        case = self.data["right_squash"]
        result = _squash_right_branch(case["input"])
        np.testing.assert_allclose(result, case["expected"])

    def test_left_stretch_matches_reference(self):
        case = self.data["left_stretch"]
        result = _stretch_left_branch(case["phi"], case["branch"])
        np.testing.assert_allclose(result, case["expected"])

    def test_right_stretch_matches_reference(self):
        case = self.data["right_stretch"]
        result = _stretch_right_branch(case["phi"], case["branch"])
        np.testing.assert_allclose(result, case["expected"])

    def test_global_affine_normalization_is_invariant(self):
        values = np.array([0.3, 0.9, 1.8, 2.7])
        normalized = _normalize_modB_global(values)[0]
        transformed = _normalize_modB_global(5.0 * values + 7.0)[0]
        np.testing.assert_allclose(normalized, transformed)

    def test_monotonicity_repair_matches_reference(self):
        left_initial = np.array([-0.5, -0.6, -0.2])
        right_initial = np.array([1.8, 2.2, 2.1])
        left_expected = np.array([-0.5, -0.7000000000010003, -0.2])
        right_expected = np.array([1.8, 2.099999999999, 2.1])
        left_repaired, right_repaired = _repair_shifted_qi_branches(left_initial, right_initial)
        np.testing.assert_allclose(left_repaired, left_expected)
        np.testing.assert_allclose(right_repaired, right_expected)

    def test_scalar_objective_matches_reference(self):
        case = self.data["scalar_residual"]
        result = _qi_objective_single_surface(case["target"], case["source"], case["nalpha"], case["nphi_out"])
        self.assertAlmostEqual(result, case["expected"])


if __name__ == "__main__":
    unittest.main()