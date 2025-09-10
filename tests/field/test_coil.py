import unittest
import json

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curve import RotatedCurve, create_equally_spaced_curves, create_equally_spaced_planar_curves
from simsopt.field.coil import Coil, Current, ScaledCurrent, CurrentSum, coils_via_symmetries
from simsopt.field.coil import coils_to_makegrid, coils_to_focus, load_coils_from_makegrid_file
from simsopt.field.biotsavart import BiotSavart
from simsopt._core.json import GSONEncoder, GSONDecoder, SIMSON
from simsopt.configs import get_ncsx_data

import os
from . import TEST_DIR


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
        curve = CurveHelical(x, order, 5, 1, 1.0, 0.3)
    elif curvetype == "CurvePlanarFourier":
        curve = CurvePlanarFourier(x, order)
    elif curvetype == "JaxCurvePlanarFourier":
        curve = JaxCurvePlanarFourier(x, order)
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
    elif curvetype in ["CurvePlanarFourier", "JaxCurvePlanarFourier"]:
        dofs[0] = 1.
        dofs[1] = 0.1
        dofs[order+1] = 0.1
    else:
        assert False

    curve.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
    if rotated:
        curve = RotatedCurve(curve, 0.5, flip=False)
    return curve


class TestCoil(unittest.TestCase):

    curvetypes = ["CurveXYZFourier", "JaxCurveXYZFourier", "CurveRZFourier", "CurveHelical", "CurvePlanarFourier", "JaxCurvePlanarFourier"]

    def subtest_serialization(self, curvetype, rotated):
        epss = [0.5**i for i in range(10, 15)]
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)

        for current in (Current(1e4), ScaledCurrent(Current(1e4), 4)):
            coil = Coil(curve, current)
            coil_str = json.dumps(SIMSON(coil), cls=GSONEncoder)
            coil_regen = json.loads(coil_str, cls=GSONDecoder)

            points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
            B1 = BiotSavart([coil]).set_points(points).B()
            B2 = BiotSavart([coil_regen]).set_points(points).B()
            self.assertTrue(np.allclose(B1, B2))

    def test_serialization(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_serialization(curvetype, rotated)


class TestCurrentSerialization(unittest.TestCase):
    def test_current_serialization(self):
        current = Current(1e4)
        current_str = json.dumps(SIMSON(current), cls=GSONEncoder)
        current_regen = json.loads(current_str, cls=GSONDecoder)
        self.assertAlmostEqual(current.get_value(), current_regen.get_value())

    def test_scaled_current_serialization(self):
        current = Current(1e4)
        scaled_current = ScaledCurrent(current, 3)
        current_str = json.dumps(SIMSON(scaled_current), cls=GSONEncoder)
        current_regen = json.loads(current_str, cls=GSONDecoder)
        self.assertAlmostEqual(scaled_current.get_value(),
                               current_regen.get_value())

    def test_current_sum_serialization(self):
        current_a = Current(1e4)
        current_b = Current(1.5e4)
        current = CurrentSum(current_a, current_b)
        current_str = json.dumps(SIMSON(current), cls=GSONEncoder)
        current_regen = json.loads(current_str, cls=GSONDecoder)
        self.assertAlmostEqual(current.get_value(),
                               current_regen.get_value())


class ScaledCurrentTesting(unittest.TestCase):

    def test_scaled_current(self):
        one = np.asarray([1.])
        c0 = Current(5.)
        fak = 3.
        c1 = fak * c0
        assert abs(c1.get_value()-fak * c0.get_value()) < 1e-15
        assert np.linalg.norm((c1.vjp(one)-fak * c0.vjp(one))(c0)) < 1e-15

        c2 = c0 * fak
        assert abs(c2.get_value()-fak * c0.get_value()) < 1e-15
        assert np.linalg.norm((c2.vjp(one)-fak * c0.vjp(one))(c0)) < 1e-15

        c3 = ScaledCurrent(c0, fak)
        assert abs(c3.get_value()-fak * c0.get_value()) < 1e-15
        assert np.linalg.norm((c3.vjp(one)-fak * c0.vjp(one))(c0)) < 1e-15

        c4 = -c0
        assert abs(c4.get_value() - (-1.) * c0.get_value()) < 1e-15
        assert np.linalg.norm((c4.vjp(one) - (-1.) * c0.vjp(one))(c0)) < 1e-15

        c00 = Current(6.)

        c5 = c0 + c00
        assert abs(c5.get_value() - (5. + 6.)) < 1e-15
        assert np.linalg.norm((c5.vjp(one)-c0.vjp(one))(c0)) < 1e-15
        assert np.linalg.norm((c5.vjp(one)-c00.vjp(one))(c00)) < 1e-15
        c6 = sum([c0, c00])
        assert abs(c6.get_value() - (5. + 6.)) < 1e-15
        assert np.linalg.norm((c6.vjp(one)-c0.vjp(one))(c0)) < 1e-15
        assert np.linalg.norm((c6.vjp(one)-c00.vjp(one))(c00)) < 1e-15
        c7 = c0 - c00
        assert abs(c7.get_value() - (5. - 6.)) < 1e-15
        assert np.linalg.norm((c7.vjp(one)-c0.vjp(one))(c0)) < 1e-15
        assert np.linalg.norm((c7.vjp(one)+c00.vjp(one))(c00)) < 1e-15


class CoilFormatConvertTesting(unittest.TestCase):
    def test_makegrid(self):
        curves, currents, ma = get_ncsx_data()
        with ScratchDir("."):
            coils_to_focus('test.focus', curves, currents, nfp=3, stellsym=True)

    def test_focus(self):
        curves, currents, ma = get_ncsx_data()
        with ScratchDir("."):
            coils_to_makegrid('coils.test', curves, currents, nfp=3, stellsym=True)

    def test_load_coils_from_makegrid_file(self):
        order = 25
        ppp = 10

        curves, currents, ma = get_ncsx_data(Nt_coils=order, ppp=ppp)
        with ScratchDir("."):
            coils_to_makegrid("coils.file_to_load", curves, currents, nfp=1)
            loaded_coils = load_coils_from_makegrid_file("coils.file_to_load", order, ppp)

        gamma = [curve.gamma() for curve in curves]
        loaded_gamma = [coil.curve.gamma() for coil in loaded_coils]
        loaded_currents = [coil.current for coil in loaded_coils]
        coils = [Coil(curve, current) for curve, current in zip(curves, currents)]

        for j_coil in range(len(coils)):
            np.testing.assert_allclose(
                currents[j_coil].get_value(),
                loaded_currents[j_coil].get_value()
            )
            np.testing.assert_allclose(curves[j_coil].x, loaded_coils[j_coil].curve.x)

        np.random.seed(1)

        bs = BiotSavart(coils)
        loaded_bs = BiotSavart(loaded_coils)

        points = np.asarray(17 * [[0.9, 0.4, -0.85]])
        points += 0.01 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        loaded_bs.set_points(points)

        B = bs.B()
        loaded_B = loaded_bs.B()

        np.testing.assert_allclose(B, loaded_B)
        np.testing.assert_allclose(gamma, loaded_gamma)

    def test_load_coils_from_makegrid_file_group(self):
        order = 25
        ppp = 10

        # Coil group_names is a list of strings
        filecoils = os.path.join(TEST_DIR, "coils.M16N08")
        coils = load_coils_from_makegrid_file(filecoils, order, ppp, group_names=["245th-coil", "100th-coil"])
        all_coils = load_coils_from_makegrid_file(filecoils, order, ppp)
        #     NOTE: coils will be returned in order they appear in the file, not in order of listed groups.
        #     So group_names = ["245th-coil","100th-coil"] gives the array [<coil nr 100>, <coil nr 245>]
        compare_coils = [all_coils[99], all_coils[244]]
        gamma = [coil.curve.gamma() for coil in coils]
        compare_gamma = [coil.curve.gamma() for coil in compare_coils]
        np.testing.assert_allclose(gamma, compare_gamma)

        # Coil group_names is a single string
        coils = load_coils_from_makegrid_file(filecoils, order, ppp, group_names="256th-coil")
        all_coils = load_coils_from_makegrid_file(filecoils, order, ppp)
        compare_coils = [all_coils[255]]
        gamma = [coil.curve.gamma() for coil in coils]
        compare_gamma = [coil.curve.gamma() for coil in compare_coils]
        np.testing.assert_allclose(gamma, compare_gamma)

    def test_equally_spaced_planar_curves(self):
        ncoils = 4
        nfp = 4
        stellsym = False
        R0 = 2.3
        R1 = 0.9

        curves = create_equally_spaced_curves(ncoils, nfp, stellsym, R0=R0, R1=R1)
        currents = [Current(1e5) for i in range(ncoils)]

        curves_planar = create_equally_spaced_planar_curves(ncoils, nfp, stellsym, R0=R0, R1=R1)
        currents_planar = [Current(1e5) for i in range(ncoils)]

        coils = coils_via_symmetries(curves, currents, nfp, stellsym)
        coils_planar = coils_via_symmetries(curves_planar, currents_planar, nfp, stellsym)
        bs = BiotSavart(coils)
        bs_planar = BiotSavart(coils_planar)

        x1d = np.linspace(R0, R0 + 0.3, 4)
        y1d = np.linspace(0, 0.2, 3)
        z1d = np.linspace(-0.2, 0.4, 5)
        x, y, z = np.meshgrid(x1d, y1d, z1d)
        points = np.ascontiguousarray(np.array([x.ravel(), y.ravel(), z.ravel()]).T)

        bs.set_points(points)
        bs_planar.set_points(points)

        np.testing.assert_allclose(bs.B(), bs_planar.B(), atol=1e-16)


if __name__ == "__main__":
    unittest.main()
