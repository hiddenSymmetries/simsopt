import unittest
import json

import numpy as np
from monty.tempfile import ScratchDir
from pathlib import Path

from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curve import RotatedCurve, create_equally_spaced_curves, create_equally_spaced_planar_curves
from simsopt.field.coil import Coil, Current, ScaledCurrent, CurrentSum, coils_via_symmetries, JaxCurrent
from simsopt.field.coil import coils_to_makegrid, coils_to_focus, load_coils_from_makegrid_file
from simsopt.field.biotsavart import BiotSavart
from simsopt._core.json import GSONEncoder, GSONDecoder, SIMSON
from simsopt.configs import get_ncsx_data

import os

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


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


class TestCoil(unittest.TestCase):

    curvetypes = ["CurveXYZFourier", "JaxCurveXYZFourier", "CurveRZFourier", "CurveHelical"]

    def subtest_serialization(self, curvetype, rotated):
        epss = [0.5**i for i in range(10, 15)]
        x = np.asarray([0.6] + [0.6 + eps for eps in epss])
        curve = get_curve(curvetype, rotated, x)

        for current in (Current(1e4), JaxCurrent(1e4), ScaledCurrent(Current(1e4), 4)):
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
        for CurrentCls in [Current, JaxCurrent]:
            current = CurrentCls(1e4)
            current_str = json.dumps(SIMSON(current), cls=GSONEncoder)
            current_regen = json.loads(current_str, cls=GSONDecoder)
            self.assertAlmostEqual(current.get_value(), current_regen.get_value())

    def test_scaled_current_serialization(self):
        for CurrentCls in [Current, JaxCurrent]:
            current = CurrentCls(1e4)
            scaled_current = ScaledCurrent(current, 3)
            current_str = json.dumps(SIMSON(scaled_current), cls=GSONEncoder)
            current_regen = json.loads(current_str, cls=GSONDecoder)
            self.assertAlmostEqual(scaled_current.get_value(),
                                   current_regen.get_value())

    def test_current_sum_serialization(self):
        for CurrentCls in [Current, JaxCurrent]:
            current_a = CurrentCls(1e4)
            current_b = CurrentCls(1.5e4)
            current = CurrentSum(current_a, current_b)
            current_str = json.dumps(SIMSON(current), cls=GSONEncoder)
            current_regen = json.loads(current_str, cls=GSONDecoder)
            self.assertAlmostEqual(current.get_value(),
                                   current_regen.get_value())


class ScaledCurrentTesting(unittest.TestCase):

    def test_scaled_current(self):
        one = np.asarray([1.])
        for CurrentCls in [Current, JaxCurrent]:
            c0 = CurrentCls(5.)
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

            c00 = CurrentCls(6.)

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


class PSCs(unittest.TestCase):
    def test_equally_spaced_planar_curves(self):
        ncoils = 4
        nfp = 4
        stellsym = False
        R0 = 2.3
        R1 = 0.9

        curves = create_equally_spaced_curves(ncoils, nfp, stellsym, R0=R0, R1=R1)
        currents = [Current(1e5) for i in range(ncoils)]
        currents_jax = [JaxCurrent(1e5) for i in range(ncoils)]

        curves_planar = create_equally_spaced_planar_curves(ncoils, nfp, stellsym, R0=R0, R1=R1)
        currents_planar = [Current(1e5) for i in range(ncoils)]
        currents_planar_jax = [JaxCurrent(1e5) for i in range(ncoils)]

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

    def test_psc_array_A(self):
        from simsopt.geo import SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
        from simsopt.field import PSCArray
        ncoils = 4
        R0 = 2.3
        R1 = 0.9
        range_param = "half period"
        nphi = 32
        ntheta = 32
        filename = TEST_DIR / 'input.LandremanPaul2021_QA'
        s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)
        stellsym = s.stellsym
        nfp = s.nfp
        poff = 0.5
        coff = 0.025
        s_inner = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)
        s_outer = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)
        s_inner.extend_via_normal(poff)
        s_outer.extend_via_normal(poff + coff)

        Nx = 4
        Ny = Nx
        Nz = Nx
        # Create the initial coils:
        order = 0
        base_curves, _ = create_planar_curves_between_two_toroidal_surfaces(
            s, s_inner, s_outer, Nx, Ny, Nz, order=order, coil_coil_flag=True, jax_flag=False,
        )
        print(len(base_curves))

        curves = create_equally_spaced_curves(ncoils, nfp, stellsym, R0=R0, R1=R1)
        currents = [Current(1e5) for i in range(ncoils)]
        currents_jax = [JaxCurrent(1e5) for i in range(ncoils)]
        # Fix the TF dofs
        [currents[i].fix_all() for i in range(len(currents))]
        [curves[i].fix_all() for i in range(len(curves))]
        coils_TF = coils_via_symmetries(curves, currents, nfp, stellsym)
        coils_TF_jax = coils_via_symmetries(curves, currents_jax, nfp, stellsym)

        # coils_planar = coils_via_symmetries(curves_planar, currents_planar, nfp, stellsym)
        # bs = BiotSavart(coils_TF)
        eval_points = s.gamma().reshape(-1, 3)
        a_list = np.ones(len(base_curves)) * 0.05
        b_list = a_list
        psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)
        psc_array_jax = PSCArray(base_curves, coils_TF_jax, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)
        psc_array.recompute_currents()
        psc_array_jax.recompute_currents()

        gammas1 = np.array([c.gamma() for c in psc_array.psc_curves])
        currents2 = np.array([c.current.get_value() for c in psc_array.coils_TF])
        gammas2 = np.array([c.curve.gamma() for c in psc_array.coils_TF])
        gammadashs2 = np.array([c.curve.gammadash() for c in psc_array.coils_TF])
        psc_array.biot_savart_TF.set_points(gammas1.reshape(-1, 3))
        A_ext = psc_array.biot_savart_TF.A()
        rij_norm = np.linalg.norm(gammas1[:, :, None, None, :] - gammas2[None, None, :, :, :], axis=-1)
        # sum over the currents, and sum over the biot savart integral
        A_ext2 = 1e-7 * np.sum(currents2[None, None, :, None] * np.sum(gammadashs2[None, None, :, :, :] / rij_norm[:, :, :, :, None],
                                                                       axis=-2), axis=-2) / np.shape(gammadashs2)[1]
        assert np.allclose(A_ext, A_ext2.reshape(-1, 3))

        gammas1_jax = np.array([c.gamma() for c in psc_array_jax.psc_curves])
        currents2_jax = np.array([c.current.get_value() for c in psc_array_jax.coils_TF])
        gammas2_jax = np.array([c.curve.gamma() for c in psc_array_jax.coils_TF])
        gammadashs2_jax = np.array([c.curve.gammadash() for c in psc_array_jax.coils_TF])
        psc_array_jax.biot_savart_TF.set_points(gammas1_jax.reshape(-1, 3))
        A_ext_jax = psc_array_jax.biot_savart_TF.A()
        rij_norm_jax = np.linalg.norm(gammas1_jax[:, :, None, None, :] - gammas2_jax[None, None, :, :, :], axis=-1)
        # sum over the currents, and sum over the biot savart integral
        A_ext2_jax = 1e-7 * np.sum(currents2_jax[None, None, :, None] * np.sum(gammadashs2_jax[None, None, :, :, :] / rij_norm_jax[:, :, :, :, None],
                                                                       axis=-2), axis=-2) / np.shape(gammadashs2_jax)[1]
        assert np.allclose(A_ext_jax, A_ext2_jax.reshape(-1, 3))


if __name__ == "__main__":
    unittest.main()
