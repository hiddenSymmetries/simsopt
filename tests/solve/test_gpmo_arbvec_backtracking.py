"""``GPMO_ArbVec_backtracking``: exact antiparallel test at pi, and history size.

At ``thresh_angle = pi`` a pair is removed only if its moments are exact
negations; a rounded dot product cannot decide that equality. The history
arrays hold every record a run writes and keep nhistory + 2 records if it fits.
"""

import unittest

import numpy as np

import simsoptpp as sopp

# A unit vector whose rounded self dot product is below 1 in every summation
# order, with or without fused multiply-add.
UNIT_VECTOR = np.array([0.8759982916180951, -0.4264361324882868, 0.22534244604736822])
INT_MAX = 2**31 - 1


def _problem(pol_vectors, x_init, ngrid=12, seed=3):
    rng = np.random.default_rng(seed)
    ndipoles = pol_vectors.shape[0]
    return dict(
        A_obj=np.ascontiguousarray(rng.standard_normal((3 * ndipoles, ngrid))),
        b_obj=np.ascontiguousarray(rng.standard_normal(ngrid)),
        mmax=np.ones(ndipoles),
        normal_norms=np.ones(ngrid),
        pol_vectors=np.ascontiguousarray(pol_vectors),
        dipole_grid_xyz=np.ascontiguousarray(
            np.column_stack([np.arange(ndipoles, dtype=float), np.zeros(ndipoles), np.zeros(ndipoles)])),
        x_init=np.ascontiguousarray(x_init),
    )


def _two_placed_magnets_and_a_third(second_moment):
    """Dipoles 0 and 1 start placed; the first iteration places dipole 2."""
    third = np.cross(UNIT_VECTOR, [0.0, 0.0, 1.0])
    third /= np.linalg.norm(third)
    pol_vectors = np.stack([UNIT_VECTOR, -second_moment, third])[:, None, :]
    x_init = np.stack([UNIT_VECTOR, second_moment, np.zeros(3)])
    return _problem(pol_vectors, x_init)


def _run(problem, **kwargs):
    options = dict(K=1, verbose=False, nhistory=1, backtracking=1, Nadjacent=3,
                   thresh_angle=np.pi, max_nMagnets=3)
    options.update(kwargs)
    return sopp.GPMO_ArbVec_backtracking(**problem, **options)[4]


def _run_verbose(K, nhistory, max_nMagnets, ndipoles=40):
    rng = np.random.default_rng(5)
    pol_vectors = rng.standard_normal((ndipoles, 2, 3))
    pol_vectors /= np.linalg.norm(pol_vectors, axis=2)[:, :, None]
    return sopp.GPMO_ArbVec_backtracking(
        **_problem(pol_vectors, np.zeros((ndipoles, 3)), ngrid=20), K=K, verbose=True,
        nhistory=nhistory, backtracking=100, Nadjacent=3, thresh_angle=np.pi,
        max_nMagnets=max_nMagnets)


class GPMOArbVecBacktrackingTests(unittest.TestCase):

    def test_exactly_antiparallel_pair_is_removed_at_pi(self):
        # premise: the rounded dot product does not reach cos(pi) = -1
        self.assertGreater(np.dot(UNIT_VECTOR, -UNIT_VECTOR), -1.0)
        x = _run(_two_placed_magnets_and_a_third(-UNIT_VECTOR))
        np.testing.assert_array_equal(
            x[:2], 0.0, err_msg="an exactly antiparallel neighbouring pair survived backtracking")
        self.assertTrue(np.any(x[2] != 0.0), "the third magnet should stay placed")

    def test_nearly_antiparallel_pair_is_kept_at_pi(self):
        second = -UNIT_VECTOR.copy()
        second[0] = np.nextafter(second[0], 0.0)
        x = _run(_two_placed_magnets_and_a_third(second))
        np.testing.assert_array_equal(x[0], UNIT_VECTOR)
        np.testing.assert_array_equal(x[1], second)

    def test_general_threshold_still_removes_wide_angles(self):
        # below pi the cosine test is unchanged
        x = _run(_two_placed_magnets_and_a_third(-UNIT_VECTOR), thresh_angle=0.75 * np.pi)
        np.testing.assert_array_equal(x[:2], 0.0)

    def test_history_holds_every_record_when_printing_every_iteration(self):
        # K // nhistory = 1 prints every iteration: 1 + 23 records, plus 1 when
        # the magnet limit stops the run; nhistory + 2 = 14 used to be allocated.
        K, nhistory = 23, 12
        for max_nMagnets, records in ((40, 24), (K, 25)):
            with self.subTest(max_nMagnets=max_nMagnets):
                objective_history, Bn_history, m_history, num_nonzeros, x = _run_verbose(
                    K, nhistory, max_nMagnets)
                for history in (objective_history, Bn_history, num_nonzeros):
                    self.assertEqual(history.shape, (records,))
                self.assertEqual(m_history.shape[2], records)
                self.assertTrue(np.all(objective_history > 0), "a history record was not written")
                # every iteration places one magnet and nothing stops the run early
                self.assertEqual(np.count_nonzero(np.any(x != 0, axis=1)), K)
                np.testing.assert_array_equal(m_history[:, :, -1], x)
                np.testing.assert_array_equal(num_nonzeros[:24], np.arange(24))

    def test_history_has_nhistory_plus_2_records_for_a_full_run(self):
        # K = 40, nhistory = 10 prints at k = 0, 4, ..., 36 and 39: 12 records.
        nhistory = 10
        objective_history, Bn_history, m_history, num_nonzeros, x = _run_verbose(
            40, nhistory, 60, ndipoles=60)
        for history in (objective_history, Bn_history, num_nonzeros):
            self.assertEqual(history.shape, (nhistory + 2,))
        self.assertEqual(m_history.shape[2], nhistory + 2)
        np.testing.assert_array_equal(num_nonzeros, [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 40])
        np.testing.assert_array_equal(m_history[:, :, -1], x)

    def test_history_has_nhistory_plus_2_records_for_an_early_stop(self):
        # stopped by the magnet limit after 5 iterations: 3 records
        nhistory = 100
        objective_history, Bn_history, m_history, num_nonzeros, x = _run_verbose(1000, nhistory, 5)
        for history in (objective_history, Bn_history, num_nonzeros):
            self.assertEqual(history.shape, (nhistory + 2,))
        self.assertEqual(m_history.shape[2], nhistory + 2)
        self.assertEqual(np.count_nonzero(objective_history), 3)
        np.testing.assert_array_equal(m_history[:, :, 2], x)

    def test_history_beyond_int_is_rejected_before_allocating(self):
        # each needs at least 2**31 records
        problem = _problem(np.ones((2, 1, 3)) / np.sqrt(3), np.zeros((2, 3)))
        for K, nhistory, verbose in ((INT_MAX - 1, 2**30, True), (INT_MAX, INT_MAX, True),
                                     (INT_MAX, INT_MAX - 1, False)):
            with self.subTest(K=K, nhistory=nhistory, verbose=verbose):
                with self.assertRaises(ValueError):
                    sopp.GPMO_ArbVec_backtracking(
                        **problem, K=K, verbose=verbose, nhistory=nhistory, backtracking=100,
                        Nadjacent=1, thresh_angle=np.pi, max_nMagnets=2)


if __name__ == "__main__":
    unittest.main()
