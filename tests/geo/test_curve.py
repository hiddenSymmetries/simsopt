import logging
import unittest
import json

import numpy as np
from monty.json import MontyEncoder, MontyDecoder

from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curve import RotatedCurve, curves_to_vtk
from simsopt.geo import parameters
from simsopt.configs.zoo import get_ncsx_data

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

parameters['jit'] = False


def taylor_test(f, df, x, epsilons=None, direction=None):
    np.random.seed(1)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(7, 20)))
    # print("################################################################################")
    err_old = 1e9
    counter = 0
    for eps in epsilons:
        if counter > 8:
            break
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps-fminuseps)/(2*eps)
        err = np.linalg.norm(dfest - dfx)
        # print(err)
        assert err < 1e-9 or err < 0.3 * err_old
        if err < 1e-9:
            break
        err_old = err
        counter += 1
    if err > 1e-10:
        assert counter > 3
    # print("################################################################################")


def get_curve(curvetype, rotated, x=np.asarray([0.5])):
    np.random.seed(2)
    rand_scale = 0.01
    order = 4

    if curvetype == "CurveXYZFourier":
        curve = CurveXYZFourier(x, order)
    elif curvetype == "JaxCurveXYZFourier":
        curve = JaxCurveXYZFourier(x, order)
    elif curvetype == "CurveRZFourier":
        curve = CurveRZFourier(x, order, 2, True)
    elif curvetype == "CurveHelical":
        curve = CurveHelical(x, order, 5, 2, 1.0, 0.3)
    else:
        assert False
    dofs = np.zeros((curve.dof_size, ))
    if curvetype in ["CurveXYZFourier", "JaxCurveXYZFourier"]:
        dofs[1] = 1.
        dofs[2*order + 3] = 1.
        dofs[4*order + 3] = 1.
    elif curvetype in ["CurveRZFourier"]:
        dofs[0] = 1.
        dofs[1] = 0.1
        dofs[order+1] = 0.1
    elif curvetype in ["CurveHelical"]:
        dofs[0] = np.pi/2
    else:
        assert False

    curve.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
    if rotated:
        curve = RotatedCurve(curve, 0.5, flip=False)
    return curve


class Testing(unittest.TestCase):

    curvetypes = ["CurveXYZFourier", "JaxCurveXYZFourier", "CurveRZFourier", "CurveHelical"]

    def test_curve_helical_xyzfourier(self):
        x = np.asarray([0.6])
        curve1 = CurveHelical(x, 2, 5, 2, 1.0, 0.3)
        curve1.x = [np.pi/2, 0, 0, 0]
        curve2 = CurveXYZFourier(x, 7)
        curve2.x = \
            [0, 0, 0, 0, 1, -0.15, 0, 0, 0, 0, 0, 0, 0, -0.15, 0,
             0, 0, 0, 1, 0, 0, -0.15, 0, 0, 0, 0, 0, 0, 0, 0.15,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3, 0, 0, 0, 0]
        assert np.allclose(curve1.gamma(), curve2.gamma())
        assert np.allclose(curve1.gammadash(), curve2.gammadash())

    def subtest_curve_first_derivative(self, curvetype, rotated):
        epss = [0.5**i for i in range(10, 15)]
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)
        f0 = curve.gamma()[0]
        deriv = curve.gammadash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = curve.gamma()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_curve_first_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_first_derivative(curvetype, rotated)

    def subtest_curve_second_derivative(self, curvetype, rotated):
        epss = [0.5**i for i in range(10, 15)]
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)
        f0 = curve.gammadash()[0]
        deriv = curve.gammadashdash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = curve.gammadash()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_curve_second_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_second_derivative(curvetype, rotated)

    def subtest_curve_third_derivative(self, curvetype, rotated):
        epss = [0.5**i for i in range(10, 15)] 
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)
        f0 = curve.gammadashdash()[0]
        deriv = curve.gammadashdashdash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = curve.gammadashdash()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def subtest_coil_coefficient_derivative(self, curvetype, rotated):
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.x
        cfc.invalidate_cache()

        def f(dofs):
            cfc.x = dofs
            return cfc.gamma().copy()

        def df(dofs):
            cfc.x = dofs
            return cfc.dgamma_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            cfc.x = dofs
            return cfc.gammadash().copy()

        def df(dofs):
            cfc.x = dofs
            return cfc.dgammadash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            cfc.x = dofs
            return cfc.gammadashdash().copy()

        def df(dofs):
            cfc.x = dofs
            return cfc.dgammadashdash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            cfc.x = dofs
            return cfc.gammadashdashdash().copy()

        def df(dofs):
            cfc.x = dofs
            return cfc.dgammadashdashdash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_coil_coefficient_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_coil_coefficient_derivative(curvetype, rotated)

    def subtest_coil_kappa_derivative(self, curvetype, rotated):
        # This implicitly also tests the higher order derivatives of gamma as these
        # are needed to compute the derivative of the curvature.
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.x

        def f(dofs):
            cfc.x = dofs
            return cfc.kappa().copy()

        def df(dofs):
            cfc.x = dofs
            return cfc.dkappa_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_coil_kappa_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_coil_kappa_derivative(curvetype, rotated)

    def subtest_curve_kappa_first_derivative(self, curvetype, rotated):
        epss = [0.5**i for i in range(12, 17)] 
        x = np.asarray([0.1234] + [0.1234 + eps for eps in epss])
        ma = get_curve(curvetype, rotated, x)
        f0 = ma.kappa()[0]
        deriv = ma.kappadash()[0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = ma.kappa()[i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            # print("err", err)
            assert err < 0.55 * err_old
            err_old = err

    def test_curve_kappa_first_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_kappa_first_derivative(curvetype, rotated)

    def subtest_curve_incremental_arclength_derivative(self, curvetype, rotated):
        # This implicitly also tests the higher order derivatives of gamma as these
        # are needed to compute the derivative of the curvature.
        ma = get_curve(curvetype, rotated)
        coeffs = ma.x

        def f(dofs):
            ma.x = dofs
            return ma.incremental_arclength().copy()

        def df(dofs):
            ma.x = dofs
            return ma.dincremental_arclength_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_incremental_arclength_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_incremental_arclength_derivative(curvetype, rotated)

    def subtest_curve_kappa_derivative(self, curvetype, rotated):
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.x

        def f(dofs):
            cfc.x = dofs
            return cfc.kappa().copy()

        def df(dofs):
            cfc.x = dofs
            return cfc.dkappa_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_kappa_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_kappa_derivative(curvetype, rotated)

    def subtest_curve_torsion_derivative(self, curvetype, rotated):
        cfc = get_curve(curvetype, rotated)
        coeffs = cfc.x

        def f(dofs):
            cfc.x = dofs
            return cfc.torsion().copy()

        def df(dofs):
            cfc.x = dofs
            return cfc.dtorsion_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_torsion_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_torsion_derivative(curvetype, rotated)

    def subtest_curve_frenet_frame(self, curvetype, rotated):
        ma = get_curve(curvetype, rotated)
        (t, n, b) = ma.frenet_frame()
        assert np.allclose(np.sum(n*t, axis=1), 0)
        assert np.allclose(np.sum(n*b, axis=1), 0)
        assert np.allclose(np.sum(t*b, axis=1), 0)
        assert np.allclose(np.sum(t*t, axis=1), 1)
        assert np.allclose(np.sum(n*n, axis=1), 1)
        assert np.allclose(np.sum(b*b, axis=1), 1)

    def test_curve_frenet_frame(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_frenet_frame(curvetype, rotated)

    def subtest_curve_frenet_frame_derivative(self, curvetype, rotated):
        ma = get_curve(curvetype, rotated)
        coeffs = ma.x

        def f(dofs):
            ma.x = dofs
            return ma.frenet_frame()[0].copy()

        def df(dofs):
            ma.x = dofs
            return ma.dfrenet_frame_by_dcoeff()[0].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            ma.x = dofs
            return ma.frenet_frame()[1].copy()

        def df(dofs):
            ma.x = dofs
            return ma.dfrenet_frame_by_dcoeff()[1].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            ma.x = dofs
            return ma.frenet_frame()[2].copy()

        def df(dofs):
            ma.x = dofs
            return ma.dfrenet_frame_by_dcoeff()[2].copy()
        taylor_test(f, df, coeffs)

    def test_curve_frenet_frame_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_frenet_frame_derivative(curvetype, rotated)

    def subtest_curve_dkappa_by_dphi_derivative(self, curvetype, rotated):
        ma = get_curve(curvetype, rotated)
        coeffs = ma.x

        def f(dofs):
            ma.x = dofs
            return ma.kappadash().copy()

        def df(dofs):
            ma.x = dofs
            return ma.dkappadash_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_curve_dkappa_by_dphi_derivative(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_curve_dkappa_by_dphi_derivative(curvetype, rotated)

    @unittest.skipIf(not pyevtk_found, "pyevtk not found")
    def test_curve_to_vtk(self):
        curve0 = get_curve(self.curvetypes[0], False)
        curve1 = get_curve(self.curvetypes[1], True)
        curves_to_vtk([curve0, curve1], '/tmp/curves')

    def test_plot(self):
        """
        Test the plot() function for curves. The ``show`` argument is set
        to ``False`` so the tests do not require human intervention to
        close plot windows.  However, if you do want to actually
        display the figure, you can change ``show`` to ``True`` in the
        first line of this function.
        """
        show = False

        engines = []
        try:
            import matplotlib
        except ImportError:
            pass
        else:
            engines.append("matplotlib")

        try:
            import mayavi
        except ImportError:
            pass
        else:
            engines.append("mayavi")

        try:
            import plotly
        except ImportError:
            pass
        else:
            engines.append("plotly")

        print(f'Testing these plotting engines: {engines}')
        c = CurveXYZFourier(30, 2)
        c.set_dofs(np.random.rand(len(c.get_dofs())) - 0.5)
        coils, currents, ma = get_ncsx_data(Nt_coils=25, Nt_ma=10)
        for engine in engines:
            for close in [True, False]:
                # Plot a single curve:
                c.plot(engine=engine, close=close, plot_derivative=True, show=show, color=(0.9, 0.2, 0.3))

                # Plot multiple curves together:
                ax = None
                for curve in coils:
                    ax = curve.plot(engine=engine, ax=ax, show=False, close=close)
                c.plot(engine=engine, ax=ax, close=close, plot_derivative=True, show=show)

    def test_rotated_curve_gamma_impl(self):
        rc = get_curve("CurveXYZFourier", True, x=100)
        c = rc.curve
        mat = rc.rotmat

        rcg = rc.gamma()
        cg = c.gamma()
        quadpoints = rc.quadpoints

        assert np.allclose(rcg, cg@mat)
        # run gamma_impl so that the `else` in RotatedCurve.gamma_impl gets triggered
        tmp = np.zeros_like(cg[:10, :])
        rc.gamma_impl(tmp, quadpoints[:10])
        assert np.allclose(cg[:10, :]@mat, tmp)

    def subtest_serialization(self, curvetype, rotated):
        epss = [0.5**i for i in range(10, 15)]
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)

        curve_json_str = json.dumps(curve, cls=MontyEncoder)
        curve_regen = json.loads(curve_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(curve.gamma(), curve_regen.gamma()))

    def test_serialization(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_serialization(curvetype, rotated)


if __name__ == "__main__":
    unittest.main()
