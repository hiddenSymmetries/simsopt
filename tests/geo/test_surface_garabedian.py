import unittest
from pathlib import Path
import numpy as np

from simsopt._core.optimizable import make_optimizable
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacegarabedian import SurfaceGarabedian

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False


class SurfaceGarabedianTests(unittest.TestCase):
    def test_init(self):
        """
        Check that the default surface is what we expect, and that the
        'names' array is correctly aligned.
        """
        s = make_optimizable(SurfaceGarabedian(nmin=-1, nmax=2, mmin=-2, mmax=5))
        self.assertAlmostEqual(s.Delta[2, 1], 0.1)
        self.assertAlmostEqual(s.Delta[3, 1], 1.0)
        self.assertAlmostEqual(s.get('Delta(0,0)'), 0.1)
        self.assertAlmostEqual(s.get('Delta(1,0)'), 1.0)
        # Verify all other elements are 0:
        d = np.copy(s.Delta)
        d[2, 1] = 0
        d[3, 1] = 0
        np.testing.assert_allclose(d, np.zeros((8, 4)))

        s.set('Delta(-2,-1)', 42)
        self.assertAlmostEqual(s.Delta[0, 0], 42)
        self.assertAlmostEqual(s.get_Delta(-2, -1), 42)

        s.set('Delta(5,-1)', -7)
        self.assertAlmostEqual(s.Delta[7, 0], -7)
        self.assertAlmostEqual(s.get_Delta(5, -1), -7)

        s.set('Delta(-2,2)', 13)
        self.assertAlmostEqual(s.Delta[0, 3], 13)
        self.assertAlmostEqual(s.get_Delta(-2, 2), 13)

        s.set('Delta(5,2)', -5)
        self.assertAlmostEqual(s.Delta[7, 3], -5)
        self.assertAlmostEqual(s.get_Delta(5, 2), -5)

        s.set_Delta(-2, -1, 421)
        self.assertAlmostEqual(s.Delta[0, 0], 421)

        s.set_Delta(5, -1, -71)
        self.assertAlmostEqual(s.Delta[7, 0], -71)

        s.set_Delta(-2, 2, 133)
        self.assertAlmostEqual(s.Delta[0, 3], 133)

        s.set_Delta(5, 2, -50)
        self.assertAlmostEqual(s.Delta[7, 3], -50)

    def test_convert_back(self):
        """
        If we start with a SurfaceRZFourier, convert to Garabedian, and
        convert back to SurfaceFourier, we should get back what we
        started with.
        """
        for mpol in range(1, 4):
            for ntor in range(5):
                for nfp in range(1, 4):
                    sf1 = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
                    # Set all dofs to random numbers in [-2, 2]:
                    sf1.set_dofs((np.random.rand(len(sf1.get_dofs())) - 0.5) * 4)
                    sg = sf1.to_Garabedian()
                    sf2 = sg.to_RZFourier()
                    np.testing.assert_allclose(sf1.rc, sf2.rc)
                    np.testing.assert_allclose(sf1.zs, sf2.zs)


if __name__ == "__main__":
    unittest.main()
