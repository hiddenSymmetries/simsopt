import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
filtered_meta_path = []
for finder in sys.meta_path:
    known_source_files = getattr(finder, "known_source_files", None)
    if isinstance(known_source_files, dict):
        simsopt_init = known_source_files.get("simsopt")
        if isinstance(simsopt_init, str) and not simsopt_init.startswith(str(REPO_ROOT)):
            continue
    filtered_meta_path.append(finder)
sys.meta_path[:] = filtered_meta_path
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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

    def test_get_branches_single_crossing_returns_degenerate_interval(self):
        phi = np.array([0.0, 0.5, 1.0])
        branch = np.array([0.2, 0.8, 0.9])
        result = _get_branches(phi, branch, 0.5, 1.0, 0.0)
        np.testing.assert_allclose(result, np.array([0.25, 0.25, 0.0, 0.0]))

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

    def test_stretch_helpers_handle_single_point_branch(self):
        phi = np.array([0.3])
        branch = np.array([0.7])
        np.testing.assert_allclose(_stretch_left_branch(phi, branch), np.array([0.0]))
        np.testing.assert_allclose(_stretch_right_branch(phi, branch), np.array([0.0]))

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