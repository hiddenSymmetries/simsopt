import unittest
import json

from simsopt.geo import Surface, SurfaceRZFourier, SurfaceXYZFourier, SurfaceXYZTensorFourier, SurfaceHenneberg
from simsopt.geo.surface_newquadpts import SurfaceNewQuadPoints

surface_types = [
                 'SurfaceRZFourier',
                 'SurfaceXYZFourier',
                 'SurfaceXYZTensorFourier',
                 'SurfaceHenneberg',
                ]

class SurfaceNewQuadPointsTest(unittest.TestCase):
    def test_newquadpts_init(self):
        for surface_type in surface_types:
            s = eval(surface_type + ".from_nphi_ntheta(ntheta=17, nphi=17, range='field period', nfp=3)")
            quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(nphi=62, ntheta=61,
                                                                      range='full torus', nfp=1)
            s1 = SurfaceNewQuadPoints(s, quadpoints_phi=quadpoints_phi,
                                      quadpoints_theta=quadpoints_theta)
            self.assertAlmostEqual(s.area(), s1.area(), places=8)
            self.assertAlmostEqual(s.volume(), s1.volume(), places=8)
