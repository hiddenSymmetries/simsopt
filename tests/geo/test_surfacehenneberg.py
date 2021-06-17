import unittest
from pathlib import Path
import numpy as np

from simsopt.geo.surfacehenneberg import SurfaceHenneberg

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


class SurfaceHennebergTests(unittest.TestCase):
    def test_names(self):
        """
        Check that the names of the dofs are set correctly.
        """
        surf = SurfaceHenneberg(nfp=1, alpha_fac=1, mmax=2, nmax=1)
        names_correct = ['R0nH(0)', 'R0nH(1)',
                         'Z0nH(1)',
                         'bn(0)', 'bn(1)',
                         'rhomn(0,1)',
                         'rhomn(1,-1)', 'rhomn(1,0)', 'rhomn(1,1)',
                         'rhomn(2,-1)', 'rhomn(2,0)', 'rhomn(2,1)']
        self.assertEqual(surf.names, names_correct)

    def test_set_get_dofs(self):
        """
        If the dofs are set to random numbers, and then you get the dofs,
        you should get what you set.
        """
        for mmax in range(1, 4):
            for nmax in range(4):
                surf = SurfaceHenneberg(nfp=1, alpha_fac=1, mmax=mmax, nmax=nmax)
                self.assertEqual(len(surf.names), len(surf.get_dofs()))

                # R0nH has nmax+1
                # Z0nH has nmax
                # bn has nmax + 1
                # rhomn for m=0 has nmax
                # rhomn for m>0 has 2 * nmax + 1
                nvals = nmax + 1 \
                    + nmax \
                    + nmax + 1 \
                    + nmax \
                    + mmax * (2 * nmax + 1)
                vals = np.random.rand(nvals) - 0.5
                surf.set_dofs(vals)
                np.testing.assert_allclose(surf.get_dofs(), vals)

    def test_set_get_rhomn(self):
        """
        Test get_rhomn() and set_rhomn(). If you set a particular mode and
        then get that mode, you should get what you set.
        """
        for mmax in range(1, 4):
            for nmax in range(4):
                surf = SurfaceHenneberg(nfp=2, alpha_fac=-1, mmax=mmax, nmax=nmax)
                for m in range(mmax + 1):
                    nmin = -nmax
                    if m == 0:
                        nmin = 1
                    for n in range(nmin, nmax + 1):
                        val = np.random.rand() - 0.5
                        surf.set_rhomn(m, n, val)
                        self.assertAlmostEqual(surf.get_rhomn(m, n), val)

    def test_fixed_range(self):
        """
        Test the fixed_range() function.
        """
        surf = SurfaceHenneberg(nfp=1, alpha_fac=1, mmax=2, nmax=1)
        # Order of elements:
        names_correct = ['R0nH(0)', 'R0nH(1)',
                         'Z0nH(1)',
                         'bn(0)', 'bn(1)',
                         'rhomn(0,1)',
                         'rhomn(1,-1)', 'rhomn(1,0)', 'rhomn(1,1)',
                         'rhomn(2,-1)', 'rhomn(2,0)', 'rhomn(2,1)']
        surf.fixed_range(20, 20, True)
        np.testing.assert_equal(surf.fixed, [True]*12)
        surf.fixed_range(20, 20, False)
        np.testing.assert_equal(surf.fixed, [False]*12)
        surf.fixed_range(0, 0, True)
        np.testing.assert_equal(surf.fixed, [True, False, False, False, False, False,
                                             False, False, False, False, False, False])
        surf.fixed_range(1, 0, True)
        np.testing.assert_equal(surf.fixed, [True, False, False, True, False, False,
                                             False, True, False, False, False, False])
        surf.fixed_range(2, 0, True)
        np.testing.assert_equal(surf.fixed, [True, False, False, True, False, False,
                                             False, True, False, False, True, False])
        surf.all_fixed(True)
        surf.fixed_range(0, 1, False)
        np.testing.assert_equal(surf.fixed, [False, False, False, True, True, False,
                                             True, True, True, True, True, True])
        surf.fixed_range(1, 1, False)
        np.testing.assert_equal(surf.fixed, [False, False, False, False, False, False,
                                             False, False, False, True, True, True])

    def test_axisymm(self):
        """
        Verify to_RZFourier() for an axisymmetric surface with circular
        cross-section.
        """
        R0 = 1.5
        a = 0.3
        for nfp in range(1, 3):
            for alpha_fac in [-1, 1]:
                for mmax in range(1, 3):
                    for nmax in range(3):
                        surfH = SurfaceHenneberg(nfp=nfp, alpha_fac=alpha_fac, mmax=mmax, nmax=nmax)
                        surfH.R0nH[0] = R0
                        surfH.bn[0] = a
                        surfH.set_rhomn(1, 0, a)
                        surf = surfH.to_RZFourier()
                        for m in range(surf.mpol + 1):
                            for n in range(-surf.ntor, surf.ntor + 1):
                                if m == 0 and n == 0:
                                    self.assertAlmostEqual(surf.get_rc(m, n), R0)
                                    self.assertAlmostEqual(surf.get_zs(m, n), 0.0)
                                elif m == 1 and n == 0:
                                    self.assertAlmostEqual(surf.get_rc(m, n), a)
                                    self.assertAlmostEqual(surf.get_zs(m, n), a)
                                else:
                                    self.assertAlmostEqual(surf.get_rc(m, n), 0.0)
                                    self.assertAlmostEqual(surf.get_zs(m, n), 0.0)


if __name__ == "__main__":
    unittest.main()
