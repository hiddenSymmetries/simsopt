"""``GPMO_ArbVec_backtracking`` removes exactly antiparallel pairs at ``thresh_angle = pi``."""

import unittest

import numpy as np

import simsoptpp as sopp

# A unit vector whose rounded self dot product is below 1.
UNIT_VECTOR = np.array([0.8759982916180951, -0.4264361324882868, 0.22534244604736822])


class GPMOArbVecBacktrackingTests(unittest.TestCase):

    def test_exactly_antiparallel_pair_is_removed_at_pi(self):
        """Dipoles 0 and 1 start placed with opposite moments; dipole 2 is placed next."""
        self.assertGreater(np.dot(UNIT_VECTOR, -UNIT_VECTOR), -1.0)
        rng = np.random.default_rng(3)
        ndipoles, ngrid = 3, 12
        third = np.cross(UNIT_VECTOR, [0.0, 0.0, 1.0])
        third /= np.linalg.norm(third)
        pol_vectors = np.stack([UNIT_VECTOR, UNIT_VECTOR, third])[:, None, :]
        x_init = np.stack([UNIT_VECTOR, -UNIT_VECTOR, np.zeros(3)])
        x = sopp.GPMO_ArbVec_backtracking(
            A_obj=np.ascontiguousarray(rng.standard_normal((3 * ndipoles, ngrid))),
            b_obj=np.ascontiguousarray(rng.standard_normal(ngrid)),
            mmax=np.ones(ndipoles),
            normal_norms=np.ones(ngrid),
            pol_vectors=np.ascontiguousarray(pol_vectors),
            dipole_grid_xyz=np.ascontiguousarray(
                np.column_stack([np.arange(ndipoles, dtype=float), np.zeros(ndipoles), np.zeros(ndipoles)])),
            x_init=np.ascontiguousarray(x_init),
            K=1, verbose=False, nhistory=1, backtracking=1, Nadjacent=3,
            thresh_angle=np.pi, max_nMagnets=3)[4]
        np.testing.assert_array_equal(x[:2], 0.0)
        self.assertTrue(np.any(x[2] != 0.0))


if __name__ == "__main__":
    unittest.main()
