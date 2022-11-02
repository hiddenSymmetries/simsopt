import unittest
import json

from simsopt.geo import Surface, SurfaceRZFourier, SurfaceXYZFourier, SurfaceXYZTensorFourier, SurfaceHenneberg
from simsopt.geo.surface_newquadpts import SurfaceNewQuadPoints
from simsopt._core.json import SIMSON, GSONEncoder, GSONDecoder

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

    def test_serialization(self):
        s = SurfaceRZFourier.from_nphi_ntheta(ntheta=17, nphi=17, range='field period', nfp=3)
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[1, 0] = 0.2
        s.local_full_x = s.get_dofs()
        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(nphi=62, ntheta=61,
                                                                  range='full torus', nfp=1)
        s_newquadpts = SurfaceNewQuadPoints(s, quadpoints_phi=quadpoints_phi,
                                            quadpoints_theta=quadpoints_theta)
        surf_str = json.dumps(SIMSON(s_newquadpts), cls=GSONEncoder)
        s_newquadpts_regen = json.loads(surf_str, cls=GSONDecoder)
        self.assertAlmostEqual(s_newquadpts.area(), s_newquadpts_regen.area(), places=8)
        self.assertAlmostEqual(s_newquadpts.volume(), s_newquadpts_regen.volume(), places=8)
