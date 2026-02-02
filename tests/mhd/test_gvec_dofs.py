import unittest
from pathlib import Path

import numpy as np

from simsopt.mhd.gvec_dofs import GVECSurfaceDoFs

TEST_DIR = Path(__file__).parent / ".." / "test_files"


class GVECSurfaceDoFsTests(unittest.TestCase):
    def test_names(self):
        """
        Check that dof names are correct.
        """
        surf = GVECSurfaceDoFs(mpol=2, ntor=1, nfp=3)
        names = ['X1c(0,0)',  'X1c(0,1)',
                 'X1c(1,-1)', 'X1c(1,0)', 'X1c(1,1)',
                 'X1c(2,-1)', 'X1c(2,0)', 'X1c(2,1)',
                 'X2s(0,1)',
                 'X2s(1,-1)', 'X2s(1,0)', 'X2s(1,1)',
                 'X2s(2,-1)', 'X2s(2,0)', 'X2s(2,1)']
        self.assertEqual(surf.local_dof_names, names)

    def test_names_nonstellsym(self):
        """
        Check that dof names are correct.
        """
        surf = GVECSurfaceDoFs(mpol=1, ntor=1, nfp=3, stellsym=False)
        names = ['X1c(0,0)',  'X1c(0,1)',
                 'X1c(1,-1)', 'X1c(1,0)', 'X1c(1,1)',
                 'X1s(0,1)',
                 'X1s(1,-1)', 'X1s(1,0)', 'X1s(1,1)',
                 'X2c(0,0)',  'X2c(0,1)',
                 'X2c(1,-1)', 'X2c(1,0)', 'X2c(1,1)',
                 'X2s(0,1)',
                 'X2s(1,-1)', 'X2s(1,0)', 'X2s(1,1)']
        self.assertEqual(surf.local_dof_names, names)

    def test_change_resolution(self):
        """
        If we refine the resolution, then coarsen the grid back to the
        original resolution, the initial and final dofs should match.
        """
        s1 = GVECSurfaceDoFs(mpol=2, ntor=3)
        self.assertEqual(len(s1.local_dof_names), len(s1.x))
        self.assertEqual(len(s1.x), 35)  # 1 + 2 * (ntor + mpol * (2 * ntor + 1))

        s1.x[:] = np.random.rand(len(s1.x))

        s2 = s1.change_resolution(mpol=5, ntor=7)
        self.assertEqual(len(s2.local_dof_names), len(s2.x))
        self.assertEqual(len(s2.x), 165)  # 1 + 2 * (ntor + mpol * (2 * ntor + 1))
        self.assertEqual(s1.get("X1c(1,-2)"), s2.get("X1c(1,-2)"))
        self.assertEqual(s2.get("X1c(5,7)"), 0.0)

        s3 = s2.change_resolution(mpol=2, ntor=3)
        self.assertEqual(len(s3.local_dof_names), len(s3.x))
        self.assertEqual(len(s3.x), 35)
        np.testing.assert_allclose(s1.x, s3.x)
    
    def test_split_dof_name(self):
        surf = GVECSurfaceDoFs(mpol=2, ntor=1, nfp=3)
        name = "X2s(2,-1)"
        var, sc, m, n = surf.split_dof_name(name)
        self.assertEqual(var, "X2")
        self.assertEqual(sc, "sin")
        self.assertEqual(m, 2)
        self.assertEqual(n, -1)
    
    def test_fixed_range(self):
        surf = GVECSurfaceDoFs(mpol=2, ntor=1, nfp=3)
        surf.unfix_all()
        surf.fixed_range(mmin=1, mmax=2, nmin=0, nmax=1, fixed=True)
        self.assertTrue(surf.is_fixed("X1c(1,0)"))
        self.assertTrue(surf.is_fixed("X2s(2,1)"))
        self.assertFalse(surf.is_fixed("X1c(0,1)"))
        self.assertFalse(surf.is_fixed("X2s(1,-1)"))


if __name__ == "__main__":
    unittest.main()