import unittest
from simsopt.geo import FrameRotation, ZeroRotation, FramedCurveCentroid, FramedCurveFrenet
from simsopt.configs.zoo import get_ncsx_data
from simsopt.geo.strain_optimization import LPBinormalCurvatureStrainPenalty, LPTorsionalStrainPenalty
import numpy as np
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from scipy.optimize import minimize


class CoilStrainTesting(unittest.TestCase):

    def test_strain_opt(self):
        """ 
        Check that for a circular coil, strains 
        can be optimized to vanish using rotation 
        dofs. 
        """
        for centroid in [True, False]:
            quadpoints = np.linspace(0, 1, 10, endpoint=False)
            curve = CurveXYZFourier(quadpoints, order=1)
            curve.set('xc(1)', 1e-4)
            curve.set('ys(1)', 1e-4)
            curve.fix_all()
            order = 2
            np.random.seed(1)
            rotation = FrameRotation(quadpoints, order)
            rotation.x = np.random.standard_normal(size=(2*order+1,))
            if centroid:
                framedcurve = FramedCurveCentroid(curve, rotation)
            else:
                framedcurve = FramedCurveFrenet(curve, rotation)
            Jt = LPTorsionalStrainPenalty(framedcurve, width=1e-3, p=2, threshold=0)
            Jb = LPBinormalCurvatureStrainPenalty(framedcurve, width=1e-3, p=2, threshold=0)
            J = Jt+Jb

            def fun(dofs):
                J.x = dofs
                grad = J.dJ()
                return J.J(), grad
            minimize(fun, J.x, jac=True, method='L-BFGS-B',
                     options={'maxiter': 100, 'maxcor': 10, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)
            assert Jt.J() < 1e-12
            assert Jb.J() < 1e-12

    def test_torsion(self):
        for centroid in [True, False]:
            for order in [None, 1]:
                with self.subTest(order=order):
                    self.subtest_torsion(order, centroid)

    def test_binormal_curvature(self):
        for centroid in [True, False]:
            for order in [None, 1]:
                with self.subTest(order=order):
                    self.subtest_binormal_curvature(order, centroid)

    def subtest_binormal_curvature(self, order, centroid):
        assert order in [1, None]
        curves, currents, ma = get_ncsx_data(Nt_coils=6, ppp=120)
        c = curves[0]

        if order == 1:
            rotation = FrameRotation(c.quadpoints, order)
            rotation.x = np.array([0, 0.1, 0.3])
            rotationShared = FrameRotation(curves[0].quadpoints, order, dofs=rotation.dofs)
            assert np.allclose(rotation.x, rotationShared.x)
            assert np.allclose(rotation.alpha(c.quadpoints), rotationShared.alpha(c.quadpoints))
        else:
            rotation = None

        if centroid:
            framedcurve = FramedCurveCentroid(c, rotation)
        else:
            framedcurve = FramedCurveFrenet(c, rotation)

        J = LPBinormalCurvatureStrainPenalty(framedcurve, width=1e-3, p=2, threshold=1e-4)

        if (not (not centroid and order is None)):
            dofs = J.x

            np.random.seed(1)
            h = np.random.standard_normal(size=dofs.shape)
            df = np.sum(J.dJ()*h)

            errf_old = 1e10
            for i in range(9, 14):
                eps = 0.5**i
                J.x = dofs + eps*h
                f1 = J.J()
                J.x = dofs - eps*h
                f2 = J.J()
                errf = np.abs((f1-f2)/(2*eps) - df)
                assert errf < 0.3 * errf_old
                errf_old = errf
        else:
            # Binormal curvature vanishes in Frenet frame
            assert J.J() < 1e-12

    def subtest_torsion(self, order, centroid):
        assert order in [1, None]
        curves, currents, ma = get_ncsx_data(Nt_coils=6, ppp=120)
        c = curves[0]

        if order == 1:
            rotation = FrameRotation(c.quadpoints, order)
            rotation.x = np.array([0, 0.1, 0.3])
            rotationShared = FrameRotation(curves[0].quadpoints, order, dofs=rotation.dofs)
            assert np.allclose(rotation.x, rotationShared.x)
            assert np.allclose(rotation.alpha(c.quadpoints), rotationShared.alpha(c.quadpoints))
        else:
            rotation = ZeroRotation(c.quadpoints)

        if centroid:
            framedcurve = FramedCurveCentroid(c, rotation)
        else:
            framedcurve = FramedCurveFrenet(c, rotation)

        J = LPTorsionalStrainPenalty(framedcurve, width=1e-3, p=2, threshold=1e-8)

        dofs = J.x

        np.random.seed(1)
        h = np.random.standard_normal(size=dofs.shape)
        df = np.sum(J.dJ()*h)

        errf_old = 1e10
        for i in range(9, 14):
            eps = 0.5**i
            J.x = dofs + eps*h
            f1 = J.J()
            J.x = dofs - eps*h
            f2 = J.J()
            errf = np.abs((f1-f2)/(2*eps) - df)
            assert errf < 0.3 * errf_old
            errf_old = errf
