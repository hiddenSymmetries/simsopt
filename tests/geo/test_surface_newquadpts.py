import unittest
import json

import numpy as np
from numpy.testing import assert_allclose

from simsopt.geo import Surface, SurfaceRZFourier, SurfaceXYZFourier, SurfaceXYZTensorFourier, SurfaceHenneberg
from simsopt.geo.surface_newquadpts import SurfaceNewQuadPoints
from simsopt.geo.surfaceobjectives import PrincipalCurvature
from simsopt._core.json import SIMSON, GSONEncoder, GSONDecoder

surface_types = [
    'SurfaceRZFourier',
    'SurfaceXYZFourier',
    'SurfaceXYZTensorFourier',
]


class SurfaceNewQuadPointsTest(unittest.TestCase):
    def test_newquadpts_init(self):
        for surface_type in surface_types:
            s = eval(surface_type + ".from_nphi_ntheta(ntheta=17, nphi=17, range='field period', nfp=1)")
            quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(nphi=62, ntheta=61,
                                                                      range='full torus', nfp=1)
            s1 = SurfaceNewQuadPoints(s, quadpoints_phi=quadpoints_phi,
                                      quadpoints_theta=quadpoints_theta)
            self.assertAlmostEqual(s.area(), s1.area(), places=8)
            self.assertAlmostEqual(s.volume(), s1.volume(), places=8)

    def test_serialization(self):
        s = SurfaceRZFourier.from_nphi_ntheta(ntheta=17, nphi=17, range='field period', nfp=1)
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

    def test_gamma_functions(self):
        for surface_type in surface_types:
            s = eval(surface_type + ".from_nphi_ntheta(ntheta=31, nphi=30, range='field period', nfp=1)")
            s1 = eval(surface_type + ".from_nphi_ntheta(ntheta=62, nphi=61, range='full torus', nfp=1)")
            quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
                nphi=61, ntheta=62, range='full torus', nfp=1)
            s_newquadpts = SurfaceNewQuadPoints(
                s, quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta)

            assert_allclose(s1.gamma(), s_newquadpts.gamma())
            assert_allclose(s1.gammadash1(), s_newquadpts.gammadash1())
            assert_allclose(s1.gammadash2(), s_newquadpts.gammadash2())
            assert_allclose(s1.gammadash1dash1(), s_newquadpts.gammadash1dash1())
            assert_allclose(s1.gammadash1dash2(), s_newquadpts.gammadash1dash2())
            assert_allclose(s1.gammadash2dash2(), s_newquadpts.gammadash2dash2())
            assert_allclose(s1.dgamma_by_dcoeff(), s_newquadpts.dgamma_by_dcoeff())
            assert_allclose(s1.dgammadash1_by_dcoeff(), s_newquadpts.dgammadash1_by_dcoeff())
            assert_allclose(s1.dgammadash2_by_dcoeff(), s_newquadpts.dgammadash2_by_dcoeff())
            assert_allclose(s1.dgammadash1dash1_by_dcoeff(), s_newquadpts.dgammadash1dash1_by_dcoeff())
            assert_allclose(s1.dgammadash1dash2_by_dcoeff(), s_newquadpts.dgammadash1dash2_by_dcoeff())
            assert_allclose(s1.dgammadash2dash2_by_dcoeff(), s_newquadpts.dgammadash2dash2_by_dcoeff())

    def test_curvature(self):
        for surface_type in surface_types:
            s = eval(surface_type + ".from_nphi_ntheta(ntheta=31, nphi=30, range='field period', nfp=1)")
            s1 = eval(surface_type + ".from_nphi_ntheta(ntheta=62, nphi=61, range='full torus', nfp=1)")
            quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
                nphi=61, ntheta=62, range='full torus', nfp=1)
            s_newquadpts = SurfaceNewQuadPoints(
                s, quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta)

            curvature = PrincipalCurvature(s1)
            curvature_newquadpts = PrincipalCurvature(s_newquadpts)
            self.assertAlmostEqual(curvature.J(), curvature_newquadpts.J())
            assert_allclose(curvature.dJ(), curvature_newquadpts.dJ())
