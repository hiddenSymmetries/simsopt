import simsoptpp as sopp
import numpy as np
import unittest
import time


class Testing(unittest.TestCase):

    def test_MwPGP(self):
        """ 
            Test the MwPGP algorithm for solving the convex
            part of the permanent magnet problem. 
        """
        ndipoles = 100
        nquad = 1024
        max_iter = 100
        m_maxima = np.random.rand(ndipoles) * 10
        m0 = np.zeros((ndipoles, 3))
        b = np.random.rand(nquad)
        A = np.random.rand(nquad, ndipoles, 3)
        ATA = np.tensordot(A, A, axes=([1, 1]))
        alpha = 2.0 / np.linalg.norm(ATA.reshape(nquad * 3, nquad * 3), ord=2) 
        ATb = np.tensordot(A, b, axes=([0, 0]))
        t1 = time.time()
        MwPGP_hist, RS_hist, m_hist, dipoles = sopp.MwPGP_algorithm(
            A_obj=A,
            b_obj=b,
            ATb=ATb,
            m_proxy=m0,
            m0=m0,
            m_maxima=m_maxima,
            alpha=alpha,
            nu=1e100,
            epsilon=1e-4,
            max_iter=max_iter,
            verbose=True,
            reg_l0=0.0,
            reg_l1=0.0,
            reg_l2=0.0,
        )

        t2 = time.time()
        print(t2 - t1) 

    def test_GPMO(self):
        """
            Test the GPMO algorithm for greedily
            solving the permanent magnet optimization
            problem for fully binary, grid-aligned
            solutions. Additionally tests the GPMO
            variants, including GPMOb and GPMOm.
        """
        ndipoles = 1000
        nquad = 1024
        max_iter = 100
        nhistory = 10
        b = np.random.rand(nquad)
        A = np.random.rand(nquad, ndipoles, 3)
        ATb = np.tensordot(A, b, axes=([0, 0]))
        t1 = time.time()
        Nnorms = np.random.rand(nquad)
        m_random = np.random.rand(ndipoles) * 10
        m_zeros = np.zeros(ndipoles)

        # Check that baseline, multi, and backtracking variants
        # of GPMO algorithm all produce same answer in the baseline limit. 
        for m_maxima in [m_zeros, m_random]:
            _, _, m_hist_multi, _ = sopp.GPMO_multi(
                A_obj=A,
                b_obj=b,
                mmax=m_maxima,
                normal_norms=Nnorms,
                K=max_iter,
                nhistory=nhistory,
                verbose=True,
                Nadjacent=1,
                dipole_grid_xyz=np.random.rand(ndipoles, 3)
            )
            _, _, m_hist, _, = sopp.GPMO_baseline(
                A_obj=A,
                b_obj=b,
                mmax=m_maxima,
                normal_norms=Nnorms,
                K=max_iter,
                nhistory=nhistory,
                verbose=True,
            )
            _, _, m_hist_backtracking, _, _ = sopp.GPMO_backtracking(
                A_obj=A,
                b_obj=b,
                mmax=m_maxima,
                normal_norms=Nnorms,
                K=max_iter,
                nhistory=nhistory,
                verbose=True,
                Nadjacent=1,
                backtracking=500,
                dipole_grid_xyz=np.random.rand(ndipoles, 3)
            )
            assert np.allclose(m_hist, m_hist_backtracking)

            print('now testing segfault avoidance.')
            # test that backtracking quits correctly, no segfault
            # when nhistory << K
            _, _, m_hist_backtracking, _, _ = sopp.GPMO_backtracking(
                A_obj=A,
                b_obj=b,
                mmax=m_maxima,
                normal_norms=Nnorms,
                K=max_iter,
                nhistory=1,
                verbose=True,
                Nadjacent=1,
                backtracking=500,
                dipole_grid_xyz=np.random.rand(ndipoles, 3)
            )
            _, _, m_hist_backtracking, _, _ = sopp.GPMO_backtracking(
                A_obj=A,
                b_obj=b,
                mmax=m_maxima,
                normal_norms=Nnorms,
                K=max_iter,
                nhistory=1,
                verbose=True,
                Nadjacent=1,
                backtracking=1,
                dipole_grid_xyz=np.random.rand(ndipoles, 3)
            )

            # Test that mutual coherence optimization works
            _, _, _, m_hist_MC = sopp.GPMO_MC(
                A_obj=A,
                b_obj=b,
                ATb=ATb,
                mmax=m_maxima,
                normal_norms=Nnorms,
                K=max_iter,
                verbose=True,
            )
        t2 = time.time()
        print(t2 - t1) 


if __name__ == "__main__":
    unittest.main()
