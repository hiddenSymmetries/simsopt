import simsoptpp as sopp
import numpy as np
import unittest
import time


class Testing(unittest.TestCase):

    def test_MwPGP(self):
        ndipoles = 100
        nquad = 1024
        max_iter = 1000
        m_maxima = np.random.rand(ndipoles) * 10
        m0 = np.zeros((ndipoles, 3))
        b = np.random.rand(nquad)
        A = np.random.rand(nquad, ndipoles, 3)
        ATA = np.tensordot(A, A, axes=([1, 1]))
        alpha = 2.0 / np.linalg.norm(ATA.reshape(nquad * 3, nquad * 3), ord=2) 
        ATb = np.tensordot(A, b, axes=([0, 0]))
        t1 = time.time()
        MwPGP_hist, _, m_hist, m = sopp.MwPGP_algorithm(
            A_obj=A,
            b_obj=b,
            ATA=ATA,
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
            reg_l2_shifted=0.0
        )

        t2 = time.time()
        print(t2 - t1) 


if __name__ == "__main__":
    unittest.main()
