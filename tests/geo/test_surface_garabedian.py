import logging
import unittest
from pathlib import Path

import numpy as np

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacegarabedian import SurfaceGarabedian

TEST_DIR = Path(__file__).parent / ".." / "test_files"

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SurfaceGarabedianTests(unittest.TestCase):
    def test_init(self):
        """
        Check that the default surface is what we expect, and that the
        'names' array is correctly aligned.
        """
        s = SurfaceGarabedian(nmin=-1, nmax=2, mmin=-2, mmax=5)
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
                    sg = SurfaceGarabedian.from_RZFourier(sf1)
                    sf2 = sg.to_RZFourier()
                    np.testing.assert_allclose(sf1.rc, sf2.rc)
                    np.testing.assert_allclose(sf1.zs, sf2.zs)

    def test_fix_range(self):
        """
        Test the fix_range() function for SurfaceGarabedian.
        """
        s = SurfaceGarabedian(mmin=-3, mmax=2, nmin=-4, nmax=3)
        s.local_fix_all()
        s.fix_range(0, 1, -2, 3, False)
        for m in range(-3, 3):
            for n in range(-4, 4):
                is_fixed = s.is_fixed(f'Delta({m},{n})')
                logger.debug(f'm={m} n={n} fixed={is_fixed}')
                if m >= 0 and m <= 1 and n >= -2 and n <= 3:
                    self.assertTrue(s.is_free(f'Delta({m},{n})'))
                else:
                    self.assertTrue(s.is_fixed(f'Delta({m},{n})'))

        s = SurfaceGarabedian(mmin=0, mmax=3, nmin=-4, nmax=4)
        s.local_unfix_all()
        s.fix_range(1, 2, -3, 2)
        for m in range(0, 4):
            for n in range(-4, 5):
                is_fixed = s.is_fixed(f'Delta({m},{n})')
                logger.debug(f'm={m} n={n} fixed={is_fixed}')
                if m >= 1 and m <= 2 and n >= -3 and n <= 2:
                    self.assertTrue(s.is_fixed(f'Delta({m},{n})'))
                else:
                    self.assertTrue(s.is_free(f'Delta({m},{n})'))


if __name__ == "__main__":
    unittest.main()
