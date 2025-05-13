import unittest
import json

import numpy as np

from simsopt.geo import parameters
from simsopt.geo.curve import RotatedCurve, create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curveobjectives import CurveLength, LpCurveCurvature, \
    LpCurveTorsion, CurveCurveDistance, ArclengthVariation, \
    MeanSquaredCurvature, CurveSurfaceDistance, LinkingNumber
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries
from simsopt.configs.zoo import get_ncsx_data
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
import simsoptpp as sopp

parameters['jit'] = False


class Testing(unittest.TestCase):

    curvetypes = ["CurveXYZFourier", "JaxCurveXYZFourier", "CurveRZFourier", "CurvePlanarFourier", "JaxCurvePlanarFourier"]

    def create_curve(self, curvetype, rotated):
        np.random.seed(1)
        rand_scale = 0.01
        order = 4
        nquadpoints = 200

        if curvetype == "CurveXYZFourier":
            coil = CurveXYZFourier(nquadpoints, order)
        elif curvetype == "JaxCurveXYZFourier":
            coil = JaxCurveXYZFourier(nquadpoints, order)
        elif curvetype == "CurveRZFourier":
            coil = CurveRZFourier(nquadpoints, order, 2, False)
        elif curvetype == "CurvePlanarFourier":
            coil = CurvePlanarFourier(nquadpoints, order)
        elif curvetype == "JaxCurvePlanarFourier":
            coil = JaxCurvePlanarFourier(nquadpoints, order)
        else:
            # print('Could not find' + curvetype)
            assert False
        dofs = np.zeros((coil.dof_size, ))
        if curvetype in ["CurveXYZFourier", "JaxCurveXYZFourier"]:
            dofs[1] = 1.
            dofs[2*order+3] = 1.
            dofs[4*order+3] = 1.
        elif curvetype in ["CurveRZFourier"]:
            dofs[0] = 1.
            dofs[1] = 0.1
            dofs[order+1] = 0.1
        elif curvetype in ["CurvePlanarFourier", "JaxCurvePlanarFourier"]:
            dofs[0] = 1.
            dofs[:2*order+1] = 0.1  # give the coil a little bit of curvature
            dofs[2*order + 1] = 1. # Set orientation to (1, 0, 0, 0)
            dofs[2*order + 2] = 0.
            dofs[2*order + 3] = 0.
            dofs[2*order + 4] = 0.
        else:
            assert False

        coil.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
        if rotated:
            coil = RotatedCurve(coil, 0.5, flip=False)
        return coil

    def subtest_curve_length_taylor_test(self, curve):
        J = CurveLength(curve)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_length_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_length_taylor_test(curve)

    def subtest_curve_curvature_taylor_test(self, curve):
        J = LpCurveCurvature(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_curvature_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_curvature_taylor_test(curve)

    def subtest_curve_torsion_taylor_test(self, curve):
        J = LpCurveTorsion(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        err = 1e6
        for i in range(10, 20):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_torsion_taylor_test(self):
        for curvetype in self.curvetypes:
            # Planar curves have no torsion
            if "CurvePlanarFourier" not in curvetype:
                for rotated in [True, False]:
                    with self.subTest(curvetype=curvetype, rotated=rotated):
                        curve = self.create_curve(curvetype, rotated)
                        self.subtest_curve_torsion_taylor_test(curve)

    def subtest_curve_minimum_distance_taylor_test(self, curve):
        ncurves = 3
        curve_t = curve.curve.__class__.__name__ if isinstance(curve, RotatedCurve) else curve.__class__.__name__
        curves = [curve] + [RotatedCurve(self.create_curve(curve_t, False), 0.1*i, True) for i in range(1, ncurves)]
        J = CurveCurveDistance(curves, 0.2)
        mindist = 1e10
        for i in range(len(curves)):
            for j in range(i):
                mindist = min(mindist, np.min(np.linalg.norm(curves[i].gamma()[:, None, :] - curves[j].gamma()[None, :, :], axis=2)))
        assert abs(J.shortest_distance() - mindist) < 1e-14
        assert mindist > 1e-10

        for k in range(ncurves):
            curve_dofs = curves[k].x
            h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
            J0 = J.J()
            dJ = J.dJ(partials=True)(curves[k].curve if isinstance(curves[k], RotatedCurve) else curves[k])
            deriv = np.sum(dJ * h)
            assert np.abs(deriv) > 1e-10
            err = 1e6
            for i in range(5, 15):
                eps = 0.5**i
                curves[k].x = curve_dofs + eps * h
                Jh = J.J()
                deriv_est = (Jh-J0)/eps
                err_new = np.linalg.norm(deriv_est-deriv)
                # print("err_new %s" % (err_new))
                assert err_new < 0.55 * err
                err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_minimum_distance_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_minimum_distance_taylor_test(curve)

    def subtest_curve_arclengthvariation_taylor_test(self, curve, nintervals):
        if isinstance(curve, CurveXYZFourier):
            J = ArclengthVariation(curve, nintervals=nintervals)
        else:
            J = ArclengthVariation(curve, nintervals=2)

        curve_dofs = curve.x
        h = 1e-1 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(1, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jp = J.J()
            curve.x = curve_dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp-Jm)/(2*eps)
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.3 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_arclengthvariation_taylor_test(self):
        for curvetype in self.curvetypes:
            for nintervals in ["full", "partial", 2]:
                with self.subTest(curvetype=curvetype, nintervals=nintervals):
                    curve = self.create_curve(curvetype, False)
                    self.subtest_curve_arclengthvariation_taylor_test(curve, nintervals)

    def test_arclength_variation_circle(self):
        """ For a circle, the arclength variation should be 0. """
        c = CurveXYZFourier(16, 1)
        c.set('xc(1)', 4.0)
        c.set('ys(1)', 4.0)
        for nintervals in ["full", "partial", 2]:
            a = ArclengthVariation(c, nintervals=nintervals)
            assert np.abs(a.J()) < 1.0e-12

    def test_arclength_variation_circle_planar(self):
        """ For a circle, the arclength variation should be 0. """
        c = CurvePlanarFourier(16, 1)
        c.set('X', 4.0)
        c.set('Y', 4.0)
        c.set('Z', 0.0)
        for nintervals in ["full", "partial", 2]:
            a = ArclengthVariation(c)
            assert np.abs(a.J()) < 1.0e-12

    def subtest_curve_meansquaredcurvature_taylor_test(self, curve):
        J = MeanSquaredCurvature(curve)
        curve_dofs = curve.x
        h = 1e-1 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jp = J.J()
            curve.x = curve_dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp-Jm)/(2*eps)
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.3 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_meansquaredcurvature_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_meansquaredcurvature_taylor_test(curve)

    def test_minimum_distance_candidates_one_collection(self):
        np.random.seed(0)
        n_clouds = 4
        pointClouds = [np.random.uniform(low=-1.0, high=+1.0, size=(5, 3)) for _ in range(n_clouds)]
        true_min_dists = {}
        from scipy.spatial.distance import cdist

        for i in range(n_clouds):
            for j in range(i):
                true_min_dists[(i, j)] = np.min(cdist(pointClouds[i], pointClouds[j]))

        threshold = max(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(pointClouds, threshold, n_clouds)
        assert len(candidates) == len(true_min_dists)

        threshold = min(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(pointClouds, threshold, n_clouds)
        assert len(candidates) == 1

    def test_minimum_distance_candidates_two_collections(self):
        np.random.seed(0)
        n_clouds = 4
        pointCloudsA = [np.random.uniform(low=-1.0, high=+1.0, size=(5, 3)) for _ in range(n_clouds)]
        pointCloudsB = [np.random.uniform(low=-1.0, high=+1.0, size=(5, 3)) for _ in range(n_clouds)]
        true_min_dists = {}
        from scipy.spatial.distance import cdist

        for i in range(n_clouds):
            for j in range(n_clouds):
                true_min_dists[(i, j)] = np.min(cdist(pointCloudsA[i], pointCloudsB[j]))

        threshold = max(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(pointCloudsA, pointCloudsB, threshold)
        assert len(candidates) == len(true_min_dists)

        threshold = min(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(pointCloudsA, pointCloudsB, threshold)
        assert len(candidates) == 1

    def test_minimum_distance_candidates_symmetry(self):
        from scipy.spatial.distance import cdist
        base_curves, base_currents, _ = get_ncsx_data(Nt_coils=10)
        curves = [c.curve for c in coils_via_symmetries(base_curves, base_currents, 3, True)]
        for t in np.linspace(0.05, 0.5, num=10):
            Jnosym = CurveCurveDistance(curves, t)
            Jsym = CurveCurveDistance(curves, t, num_basecurves=3)
            assert abs(Jnosym.shortest_distance_among_candidates() - Jsym.shortest_distance_among_candidates()) < 1e-15
            print(len(Jnosym.candidates), len(Jsym.candidates), Jnosym.shortest_distance_among_candidates())
            distsnosym = [np.min(cdist(Jnosym.curves[i].gamma(), Jnosym.curves[j].gamma())) for i, j in Jnosym.candidates]
            distssym = [np.min(cdist(Jsym.curves[i].gamma(), Jsym.curves[j].gamma())) for i, j in Jsym.candidates]
            print("distsnosym", distsnosym)
            print("distssym", distssym)
            print((Jnosym.candidates), (Jsym.candidates))

            assert np.allclose(
                np.unique(np.round(distsnosym, 8)),
                np.unique(np.round(distssym, 8))
            )

    def test_curve_surface_distance(self):
        np.random.seed(0)
        base_curves, base_currents, _ = get_ncsx_data(Nt_coils=10)
        curves = [c.curve for c in coils_via_symmetries(base_curves, base_currents, 3, True)]
        ntor = 0
        surface = SurfaceRZFourier.from_nphi_ntheta(nfp=3, nphi=32, ntheta=32, ntor=ntor)
        surface.set(f'rc(0,{ntor})', 1.6)
        surface.set(f'rc(1,{ntor})', 0.2)
        surface.set(f'zs(1,{ntor})', 0.2)

        last_num_candidates = 0
        for t in np.linspace(0.01, 1.0, num=10):
            J = CurveSurfaceDistance(curves, surface, t)
            J.compute_candidates()
            assert len(J.candidates) >= last_num_candidates
            last_num_candidates = len(J.candidates)
            if len(J.candidates) == 0:
                assert J.shortest_distance() > J.shortest_distance_among_candidates()
            else:
                assert J.shortest_distance() == J.shortest_distance_among_candidates()

        assert last_num_candidates == len(curves)
        threshold = 1.0
        J = CurveSurfaceDistance(curves, surface, threshold)

        curve_dofs = J.x
        h = 1e-1 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 12):
            eps = 0.5**i
            J.x = curve_dofs + eps * h
            Jp = J.J()
            J.x = curve_dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp-Jm)/(2*eps)
            err_new = np.linalg.norm(deriv_est-deriv)
            print("err_new %s" % (err_new))
            print(err_new/err)
            assert err_new < 0.3 * err
            err = err_new

    def test_linking_number(self):
        for downsample in [1, 2, 5]:
            curves1 = create_equally_spaced_curves(2, 1, stellsym=True, R0=1, R1=0.5, order=5, numquadpoints=120)
            curve1 = CurveXYZFourier(200, 3)
            coeffs = curve1.dofs_matrix
            coeffs[1][0] = 1.
            coeffs[1][1] = 0.5
            coeffs[2][2] = 0.5
            curve1.set_dofs(np.concatenate(coeffs))

            curve2 = CurveXYZFourier(150, 3)
            coeffs = curve2.dofs_matrix
            coeffs[1][0] = 0.5
            coeffs[1][1] = 0.5
            coeffs[0][0] = 0.1
            coeffs[0][1] = 0.5
            coeffs[0][2] = 0.5
            curve2.set_dofs(np.concatenate(coeffs))
            curves2 = [curve1, curve2]
            curves3 = [curve2, curve1]
            objective1 = LinkingNumber(curves1, downsample)
            objective2 = LinkingNumber(curves2, downsample)
            objective3 = LinkingNumber(curves3, downsample)

            print("Linking number testing (should be 0, 1, 1):", objective1.J(), objective2.J(), objective3.J())
            np.testing.assert_allclose(objective1.J(), 0, atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(objective2.J(), 1, atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(objective3.J(), 1, atol=1e-14, rtol=1e-14)

    def test_linking_number_planar(self):
        for downsample in [1, 2, 5]:
            curve1 = CurvePlanarFourier(200, 3)
            print(curve1.dof_names)
            curve1.set('rc(0)', 1)
            curve1.set('X', 0.25)
            curve1.set('Y', 0.0)
            curve1.set('Z', 0.0)
            curve2 = CurvePlanarFourier(200, 3)
            curve2.set('rc(0)', 1)
            curve2.set('X', 0.0)
            curve2.set('Y', 0.25)
            curve2.set('Z', 0.0)
            curves2 = [curve1, curve2]
            objective = LinkingNumber(curves2, downsample)
            assert np.abs(objective.J()) < 1e-14

            curve1.set('X', 0.0)
            curve1.set('Y', 0.0)
            curve1.set('Z', 0.0)
            curve2.set('X', 0.0)
            curve2.set('Y', 0.0)
            curve2.set('Z', 0.0)
            objective = LinkingNumber(curves2, downsample)
            assert np.abs(objective.J()) < 1e-14    

if __name__ == "__main__":
    unittest.main()
