import unittest
from pathlib import Path
import numpy as np

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surface import signed_distance_from_surface
from simsopt.geo.curverzfourier import CurveRZFourier

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False


class SurfaceDistanceTests(unittest.TestCase):
    def test_distance(self):
        c = CurveRZFourier(100, 1, 1, False)
        # dofs = c.get_dofs()
        # dofs[0] = 1.
        # c.set_dofs(dofs)
        # dofs = c.x
        # dofs[0] = 1.0
        c.set(0, 1.0)
        s = SurfaceRZFourier(mpol=1, ntor=1)
        s.fit_to_curve(c, 0.2, flip_theta=True)
        xyz = np.asarray([[0, 0, 0], [1., 0, 0], [2., 0., 0]])
        d = signed_distance_from_surface(xyz, s)
        assert np.allclose(d, [-0.8, 0.2, -0.8])
        s.fit_to_curve(c, 0.2, flip_theta=False)
        d = signed_distance_from_surface(xyz, s)
        assert np.allclose(d, [-0.8, 0.2, -0.8])


if __name__ == "__main__":
    unittest.main()
