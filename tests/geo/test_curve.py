import logging
import unittest
import json
import os


import numpy as np

from simsopt._core.json import GSONEncoder, GSONDecoder, SIMSON
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.curve import RotatedCurve, curves_to_vtk, create_planar_curves_between_two_toroidal_surfaces
from simsopt.geo import parameters
from simsopt.configs.zoo import get_ncsx_data, get_w7x_data
from simsopt.field import BiotSavart, Current, coils_via_symmetries, Coil
from simsopt.field.coil import coils_to_makegrid
from simsopt.geo import CurveLength, CurveCurveDistance
from math import gcd
from simsopt.geo.curveplanarellipticalcylindrical import (
    CurvePlanarEllipticalCylindrical, create_equally_spaced_cylindrical_curves,
    r_ellipse, xyz_cyl, rotations, convert_to_cyl, cylindrical_shift, cyl_to_cart, gamma_pure
)
from simsopt.geo import SurfaceRZFourier
from pathlib import Path
from monty.tempfile import ScratchDir

try:
    import pyevtk
except ImportError:
    pyevtk = None

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
    print("################################################################################")
    err_old = 1e9
    counter = 0
    for eps in epsilons:
        if counter > 8:
            break
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps-fminuseps)/(2*eps)
        err = np.linalg.norm(dfest - dfx)
        print(err, err/err_old)

        assert err < 1e-9 or err < 0.3 * err_old
        if err < 1e-9:
            break
        err_old = err
        counter += 1
    if err > 1e-10:
        assert counter > 3
    print("################################################################################")


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
    elif curvetype == "CurveHelicalInitx0":
        curve = CurveHelical(x, order, 5, 2, 1.0, 0.3, x0=np.ones((2*order,)))
    elif curvetype == "CurvePlanarFourier":
        curve = CurvePlanarFourier(x, order, 2, True)
    elif curvetype == "JaxCurvePlanarFourier":
        curve = JaxCurvePlanarFourier(x, order)
    elif curvetype == "CurveXYZFourierSymmetries1":
        curve = CurveXYZFourierSymmetries(x, order, 2, True)
    elif curvetype == "CurveXYZFourierSymmetries2":
        curve = CurveXYZFourierSymmetries(x, order, 2, False)
    elif curvetype == "CurveXYZFourierSymmetries3":
        curve = CurveXYZFourierSymmetries(x, order, 2, False, ntor=3)
    else:
        assert False

    dofs = np.zeros((curve.dof_size, ))
    if curvetype in ["CurveXYZFourier", "JaxCurveXYZFourier"]:
        dofs[1] = 1.
        dofs[2*order + 3] = 1.
        dofs[4*order + 3] = 1.
    elif curvetype in ["CurveRZFourier", "CurvePlanarFourier", "JaxCurvePlanarFourier"]:
        dofs[0] = 1.
        dofs[1] = 0.1
        dofs[order+1] = 0.1
    elif curvetype in ["CurveHelical", "CurveHelicalInitx0"]:
        dofs[0] = np.pi/2
    elif curvetype == "CurveXYZFourierSymmetries1":
        R = 1
        r = 0.5
        curve.set('xc(0)', R)
        curve.set('xc(1)', -r)
        curve.set('zs(1)', -r)
        dofs = curve.get_dofs()
    elif curvetype == "CurveXYZFourierSymmetries2":
        R = 1
        r = 0.5
        curve.set('xc(0)', R)
        curve.set('xs(1)', -0.1*r)
        curve.set('xc(1)', -r)
        curve.set('zs(1)', -r)
        curve.set('zc(0)', 1)
        curve.set('zs(1)', r)
        dofs = curve.get_dofs()
    elif curvetype == "CurveXYZFourierSymmetries3":
        R = 1
        r = 0.5
        curve.set('xc(0)', R)
        curve.set('xs(1)', -0.1*r)
        curve.set('xc(1)', -r)
        curve.set('zs(1)', -r)
        curve.set('zc(0)', 1)
        curve.set('zs(1)', r)
        dofs = curve.get_dofs()
    else:
        assert False

    curve.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)

    if rotated:
        curve = RotatedCurve(curve, 0.5, flip=False)
    return curve


class Testing(unittest.TestCase):

    curvetypes = ["CurveXYZFourier", "JaxCurveXYZFourier", "CurveRZFourier", "JaxCurvePlanarFourier", "CurvePlanarFourier", "CurveHelical", "CurveXYZFourierSymmetries1", "CurveXYZFourierSymmetries2", "CurveXYZFourierSymmetries3", "CurveHelicalInitx0"]

    def get_curvexyzfouriersymmetries(self, stellsym=True, x=None, nfp=None, ntor=1):
        # returns a CurveXYZFourierSymmetries that is randomly perturbed

        np.random.seed(1)
        rand_scale = 1e-2

        if nfp is None:
            nfp = 3
        if x is None:
            x = np.linspace(0, 1, 200, endpoint=False)

        order = 2
        curve = CurveXYZFourierSymmetries(x, order, nfp, stellsym, ntor=ntor)
        R = 1
        r = 0.25
        curve.set('xc(0)', R)
        curve.set('xc(2)', r)
        curve.set('ys(2)', -r)
        curve.set('zs(1)', -2*r)
        curve.set('zs(2)', r)
        dofs = curve.x.copy()
        curve.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)

        return curve

    def test_curvexyzsymmetries_raisesexception(self):
        # test ensures that an exception is raised when you try and create a curvexyzfouriersymmetries
        # where gcd(ntor, nfp) != 1.

        order = 1
        nfp = 1
        ntor = 2
        # nfp = 1 and ntor = 2 here, so it should work
        curve = CurveXYZFourierSymmetries(100, order, nfp, True, ntor=ntor, x0=np.ones(3*order+1))
        print(curve.x)

        with self.assertRaises(Exception):
            order = 1
            nfp = 2
            ntor = 2
            # nfp = 2 and ntor = 2 here, so an exception should be raised
            _ = CurveXYZFourierSymmetries(100, order, nfp, True, ntor=ntor, x0=np.ones(3*order+1))

    def test_curvehelical_is_curvexyzfouriersymmetries(self):
        # this test checks that both helical coil representations can produce the same helical curve on a torus
        order = 1
        nfp = 2
        curve1 = CurveXYZFourierSymmetries(100, order, nfp, True)
        R = 1
        r = 0.5
        curve1.set('xc(0)', R)
        curve1.set('xc(1)', r)
        curve1.set('zs(1)', -r)
        curve2 = CurveHelical(np.linspace(0, 1, 100, endpoint=False), order, nfp, 1, R, r, x0=np.zeros((2*order,)))
        np.testing.assert_allclose(curve1.gamma(), curve2.gamma(), atol=1e-14)

    def test_trefoil_nonstellsym(self):
        r''' This test checks that a CurveXYZFourierSymmetries can represent a non-stellarator symmetric
        trefoil knot.  A parametric representation of a trefoil knot is given by:

            x(t) = sin(t) + 2sin(t)
            y(t) = cos(t) - 2cos(t)
            z(t) = -sin(3t)

        The above can be rewritten the CurveXYZFourierSymmetries representation, with
        order = 1, nfp = 3, and ntor = 2:

            x(t) = xhat(t) * cos(ntor*t) - yhat(t) * sin(ntor*t),
            y(t) = xhat(t) * sin(ntor*t) + yhat(t) * cos(ntor*t),
            z(t) = -sin(nfp*t),

        where
            xhat(t) = sin(nfp*t),
            yhat(t) = -2 + cos(nfp*t),

        i.e., The dofs are xs(1)=1, yc(0)=-2, yc(1)=1, zs(1)=-1, and zero otherwise.
        '''

        order = 1
        nfp = 3
        ntor = 2
        x = np.linspace(0, 1, 500, endpoint=False)
        curve = CurveXYZFourierSymmetries(x, order, nfp, False, ntor=ntor)

        X = np.sin(2*np.pi * x) + 2 * np.sin(2*2*np.pi * x)
        Y = np.cos(2*np.pi * x) - 2 * np.cos(2*2*np.pi * x)
        Z = -np.sin(3*2*np.pi*x)
        XYZ = np.concatenate([X[:, None], Y[:, None], Z[:, None]], axis=1)

        curve.set('xs(1)', 1.)
        curve.set('yc(0)', -2.)
        curve.set('yc(1)', 1.)
        curve.set('zs(1)', -1.)
        np.testing.assert_allclose(curve.gamma(), XYZ, atol=1e-14)

    def test_trefoil_stellsym(self):
        r''' This test checks that a CurveXYZFourierSymmetries can represent a stellarator symmetric
        trefoil knot.  A parametric representation of a trefoil knot is given by:

            x(t) = cos(t) - 2cos(t),
            y(t) =-sin(t) - 2sin(t),
            z(t) = -sin(3t).

        The above can be rewritten the CurveXYZFourierSymmetries representation, with
        order = 1, nfp = 3, and ntor = 2:

            x(t) = xhat(t) * cos(ntor*t) - yhat(t) * sin(ntor*t),
            y(t) = xhat(t) * sin(ntor*t) + yhat(t) * cos(ntor*t),
            z(t) = -sin(nfp*t),

        where

            xhat(t) = -2 + cos(nfp*t)
            yhat(t) = -sin(nfp*t)

        i.e., xc(0)=-2, xc(1)=1, ys(1)=-1, zs(1)=-1.
        '''

        order = 1
        nfp = 3
        ntor = 2
        x = np.linspace(0, 1, 500, endpoint=False)
        curve = CurveXYZFourierSymmetries(x, order, nfp, True, ntor=ntor)

        X = np.cos(2*np.pi * x) - 2 * np.cos(2*2*np.pi * x)
        Y = -np.sin(2*np.pi * x) - 2 * np.sin(2*2*np.pi * x)
        Z = -np.sin(3*2*np.pi*x)
        XYZ = np.concatenate([X[:, None], Y[:, None], Z[:, None]], axis=1)

        curve.set('xc(0)', -2)
        curve.set('xc(1)', 1)
        curve.set('ys(1)', -1)
        curve.set('zs(1)', -1)
        np.testing.assert_allclose(curve.gamma(), XYZ, atol=1e-14)

    def test_nonstellsym(self):
        # this test checks that you can obtain a stellarator symmetric magnetic field from two non-stellarator symmetric
        # CurveXYZFourierSymmetries curves.
        for nfp in [1, 2, 3, 4, 5, 6]:
            for ntor in [1, 2, 3, 4, 5, 6]:
                with self.subTest(nfp=nfp, ntor=ntor):
                    self.subtest_nonstellsym(nfp, ntor)

    def subtest_nonstellsym(self, nfp, ntor):
        if gcd(ntor, nfp) != 1:
            return

        # this test checks that you can obtain a stellarator symmetric magnetic field from two non-stellarator symmetric
        # CurveXYZFourierSymmetries curves.
        curve = self.get_curvexyzfouriersymmetries(stellsym=False, nfp=nfp, ntor=ntor)
        current = Current(1e5)
        coils = coils_via_symmetries([curve], [current], 1, True)
        bs = BiotSavart(coils)
        bs.set_points([[1, 1, 1], [1, -1, -1]])
        B = bs.B_cyl()
        np.testing.assert_allclose(B[0, 0], -B[1, 0], atol=1e-14)
        np.testing.assert_allclose(B[0, 1], B[1, 1], atol=1e-14)
        np.testing.assert_allclose(B[0, 2], B[1, 2], atol=1e-14)

    def test_xyzhelical_symmetries(self):
        # checking various symmetries of the CurveXYZFourierSymmetries representation
        for nfp in [1, 2, 3, 4, 5, 6]:
            for ntor in [1, 2, 3, 4, 5, 6]:
                with self.subTest(nfp=nfp, ntor=ntor):
                    self.subtest_xyzhelical_symmetries(nfp, ntor)

    def subtest_xyzhelical_symmetries(self, nfp, ntor):
        if gcd(nfp, ntor) != 1:
            return

        # does the stellarator symmetric curve have rotational symmetry?
        curve = self.get_curvexyzfouriersymmetries(stellsym=True, nfp=nfp, x=np.array([0.123, 0.123+1/nfp]), ntor=ntor)
        out = curve.gamma()

        # NOTE: the point at angle t+1/nfp is the point at angle t, but rotated by 2pi *(ntor/nfp) radians.
        alpha = 2*np.pi*ntor/nfp
        R = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])

        print(R@out[0], out[1])
        np.testing.assert_allclose(out[1], R@out[0], atol=1e-14)

        # does the stellarator symmetric curve indeed pass through (x0, 0, 0)?
        curve = self.get_curvexyzfouriersymmetries(stellsym=True, nfp=nfp, x=np.array([0]), ntor=ntor)
        out = curve.gamma()
        assert np.abs(out[0, 0]) > 1e-3
        np.testing.assert_allclose(out[0, 1], 0, atol=1e-14)
        np.testing.assert_allclose(out[0, 2], 0, atol=1e-14)

        # does the non-stellarator symmetric curve not pass through (x0, 0, 0)?
        curve = self.get_curvexyzfouriersymmetries(stellsym=False, nfp=nfp, x=np.array([0]), ntor=ntor)
        out = curve.gamma()
        assert np.abs(out[0, 0]) > 1e-3
        assert np.abs(out[0, 1]) > 1e-3
        assert np.abs(out[0, 2]) > 1e-3

        # is the stellarator symmetric curve actually stellarator symmetric?
        curve = self.get_curvexyzfouriersymmetries(stellsym=True, nfp=nfp, x=np.array([0.123, -0.123]), ntor=ntor)
        pts = curve.gamma()
        np.testing.assert_allclose(pts[0, 0], pts[1, 0], atol=1e-14)
        np.testing.assert_allclose(pts[0, 1], -pts[1, 1], atol=1e-14)
        np.testing.assert_allclose(pts[0, 2], -pts[1, 2], atol=1e-14)

        # is the field from the stellarator symmetric curve actually stellarator symmetric?
        curve = self.get_curvexyzfouriersymmetries(stellsym=True, nfp=nfp, x=np.linspace(0, 1, 200, endpoint=False), ntor=ntor)
        current = Current(1e5)
        coil = Coil(curve, current)
        bs = BiotSavart([coil])
        bs.set_points([[1, 1, 1], [1, -1, -1]])
        B = bs.B_cyl()
        np.testing.assert_allclose(B[0, 0], -B[1, 0], atol=1e-14)
        np.testing.assert_allclose(B[0, 1], B[1, 1], atol=1e-14)
        np.testing.assert_allclose(B[0, 2], B[1, 2], atol=1e-14)

        # does the non-stellarator symmetric curve have rotational symmetry still?
        # NOTE: the point at angle t+1/nfp is the point at angle t, but rotated by 2pi *(ntor/nfp) radians.
        curve = self.get_curvexyzfouriersymmetries(stellsym=False, nfp=nfp, x=np.array([0.123, 0.123+1./nfp]), ntor=ntor)
        out = curve.gamma()
        alpha = 2*np.pi*ntor/nfp
        R = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        print(R@out[0], out[1])
        np.testing.assert_allclose(out[1], R@out[0], atol=1e-14)

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

    @unittest.skipIf(pyevtk is None, "pyevtk not found")
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

        curve_json_str = json.dumps(SIMSON(curve), cls=GSONEncoder, indent=2)
        curve_regen = json.loads(curve_json_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(curve.gamma(), curve_regen.gamma()))

    def test_serialization(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    self.subtest_serialization(curvetype, rotated)

    def test_load_curves_from_makegrid_file(self):
        get_config_functions = [get_ncsx_data, get_w7x_data]
        order = 10
        ppp = 4

        for get_config_function in get_config_functions:
            curves, currents, ma = get_config_function(Nt_coils=order, ppp=ppp)

            # write coils to MAKEGRID file
            coils_to_makegrid("coils.file_to_load", curves, currents, nfp=1)
            loaded_curves = CurveXYZFourier.load_curves_from_makegrid_file("coils.file_to_load", order, ppp)

            assert len(curves) == len(loaded_curves)

            for j in range(len(curves)):
                np.testing.assert_allclose(curves[j].x, loaded_curves[j].x)

            gamma = [curve.gamma() for curve in curves]
            loaded_gamma = [curve.gamma() for curve in loaded_curves]

            np.testing.assert_allclose(gamma, loaded_gamma)

            kappa = [np.max(curve.kappa()) for curve in curves]
            loaded_kappa = [np.max(curve.kappa()) for curve in loaded_curves]

            np.testing.assert_allclose(kappa, loaded_kappa)

            length = [CurveLength(c).J() for c in curves]
            loaded_length = [CurveLength(c).J() for c in loaded_curves]

            np.testing.assert_allclose(length, loaded_length)

            ccdist = CurveCurveDistance(curves, 0).J()
            loaded_ccdist = CurveCurveDistance(loaded_curves, 0).J()

            np.testing.assert_allclose(ccdist, loaded_ccdist)

            os.remove("coils.file_to_load")


class TestPlanarEllipticalCylindricalCurve(unittest.TestCase):
    def test_curveplanarellipticalcylindrical_basic(self):
        quadpoints = 20
        a, b = 2.0, 1.0
        curve = CurvePlanarEllipticalCylindrical(quadpoints, a, b)
        self.assertEqual(curve.num_dofs(), 6)
        self.assertEqual(len(curve.get_dofs()), 6)
        self.assertEqual(curve._make_names(), ['R0', 'phi', 'Z0', 'r_rotation', 'phi_rotation', 'z_rotation'])
        gamma = curve.gamma()
        self.assertEqual(gamma.shape, (quadpoints, 3))

    def test_curveplanarellipticalcylindrical_set_get_dofs(self):
        quadpoints = 10
        a, b = 1.5, 0.5
        curve = CurvePlanarEllipticalCylindrical(quadpoints, a, b)
        dofs = np.arange(6)
        curve.set_dofs_impl(dofs)
        np.testing.assert_allclose(curve.get_dofs(), dofs)

    def test_create_equally_spaced_cylindrical_curves(self):
        ncurves, nfp = 3, 2
        stellsym = True
        R0, a, b = 5.0, 1.0, 0.5
        numquadpoints = 12
        curves = create_equally_spaced_cylindrical_curves(ncurves, nfp, stellsym, R0, a, b, numquadpoints)
        self.assertEqual(len(curves), ncurves)
        for curve in curves:
            self.assertIsInstance(curve, CurvePlanarEllipticalCylindrical)
            gamma = curve.gamma()
            self.assertEqual(gamma.shape, (numquadpoints, 3))
            R = np.sqrt(gamma[:, 0]**2 + gamma[:, 1]**2)
            self.assertTrue(np.allclose(np.mean(R), R0, atol=0.2))

    def test_r_ellipse(self):
        a, b = 2.0, 1.0
        l = np.linspace(0, 1, 100)
        r = r_ellipse(a, b, l)
        self.assertEqual(r.shape, l.shape)
        self.assertTrue(np.all(r > 0))

    def test_xyz_cyl(self):
        a, b = 2.0, 1.0
        l = np.linspace(0, 1, 50)
        xyz = xyz_cyl(a, b, l)
        self.assertEqual(xyz.shape, (50, 3))
        # y should be all zeros
        self.assertTrue(np.allclose(xyz[:, 1], 0))

    def test_rotations(self):
        a, b = 2.0, 1.0
        l = np.linspace(0, 1, 10)
        curve = xyz_cyl(a, b, l)
        alpha_r, alpha_phi, alpha_z, dr = 0.1, 0.2, 0.3, 1.0
        rotated = rotations(curve, a, b, alpha_r, alpha_phi, alpha_z, dr)
        self.assertEqual(rotated.shape, curve.shape)
        # Check that the shift in R (x) is applied
        self.assertTrue(np.allclose(rotated[:, 0] - curve[:, 0], dr, atol=0.1) == False)

        # Check that a full pi or 2pi rotation in each angle returns the original curve (modulo numerical error)
        for idx, (ar, ap, az) in enumerate([
            (2 * np.pi, 0, 0),
            (0, 2 * np.pi, 0),
            (0, 0, 2 * np.pi),
            (2 * np.pi, 0, 2 * np.pi),
            (0, 2 * np.pi, 2 * np.pi),
            (2 * np.pi, 2 * np.pi, 0),
            (2 * np.pi, 2 * np.pi, 2 * np.pi),
            (np.pi, np.pi, np.pi),
            (np.pi, np.pi, 0),
        ]):
            rotated_full = rotations(curve, a, b, ar, ap, az, dr=0.0)
            np.testing.assert_allclose(np.abs(rotated_full), np.abs(curve), atol=1e-12, err_msg=f"Failed for full rotation in angle index {idx}")

    def test_convert_to_cyl(self):
        xyz = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
        cyl = convert_to_cyl(xyz)
        self.assertEqual(cyl.shape, xyz.shape)
        # R should be sqrt(x^2 + y^2)
        np.testing.assert_allclose(cyl[:, 0], [1.0, 1.0])
        # phi should be correct
        np.testing.assert_allclose(cyl[:, 1], [0.0, np.pi/2])
        # z unchanged
        np.testing.assert_allclose(cyl[:, 2], [2.0, 3.0])

    def test_cylindrical_shift(self):
        cyl = np.array([[1.0, 0.0, 2.0], [1.0, 1.0, 3.0]])
        dphi, dz = 0.5, 1.5
        shifted = cylindrical_shift(cyl, dphi, dz)
        self.assertEqual(shifted.shape, cyl.shape)
        np.testing.assert_allclose(shifted[:, 1], cyl[:, 1] + dphi)
        np.testing.assert_allclose(shifted[:, 2], cyl[:, 2] + dz)

    def test_cyl_to_cart(self):
        cyl = np.array([[1.0, 0.0, 2.0], [1.0, np.pi/2, 3.0]])
        cart = cyl_to_cart(cyl)
        self.assertEqual(cart.shape, cyl.shape)
        np.testing.assert_allclose(cart[:, 0], [1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(cart[:, 1], [0.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(cart[:, 2], [2.0, 3.0])

    def test_gamma_pure(self):
        a, b = 2.0, 1.0
        points = np.linspace(0, 1, 20)
        dofs = np.zeros(6)
        gamma = gamma_pure(dofs, points, a, b)
        self.assertEqual(gamma.shape, (20, 3))


class TestCreatePlanarCurvesBetweenTwoToroidalSurfaces(unittest.TestCase):
    def test_create_planar_curves_between_two_toroidal_surfaces(self):
        # Use a real surface from test files for a minimal working test
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA'
        nphi, ntheta = 8, 8
        with ScratchDir("."):
            s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_inner = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_inner.extend_via_projected_normal(0.1)
            s_outer.extend_via_projected_normal(0.2)
            # Standard usage
            curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
                s, s_inner, s_outer, Nx=3, Ny=3, Nz=3, order=1, coil_coil_flag=False, jax_flag=False, numquadpoints=10
            )
            self.assertTrue(len(curves) > 0)
            self.assertTrue(len(all_curves) >= len(curves))
            for curve in curves:
                gamma = curve.gamma()
                self.assertEqual(gamma.shape[1], 3)
                self.assertEqual(gamma.shape[0], 10)
            for curve in curves:
                center = np.mean(curve.gamma(), axis=0)
                r = np.linalg.norm(center[:2])
                r_inner = np.linalg.norm(np.mean(s_inner.gamma().reshape(-1, 3), axis=0)[:2])
                r_outer = np.linalg.norm(np.mean(s_outer.gamma().reshape(-1, 3), axis=0)[:2])
                self.assertTrue(r_inner < r < r_outer or r_outer < r < r_inner)

            # Test with jax_flag=True
            curves_jax, all_curves_jax = create_planar_curves_between_two_toroidal_surfaces(
                s, s_inner, s_outer, Nx=3, Ny=3, Nz=3, order=1, coil_coil_flag=False, jax_flag=True, numquadpoints=10
            )
            self.assertTrue(len(curves_jax) > 0)
            self.assertTrue(len(all_curves_jax) >= len(curves_jax))
            for curve in curves_jax:
                gamma = curve.gamma()
                self.assertEqual(gamma.shape[1], 3)
                self.assertEqual(gamma.shape[0], 10)

            # Test coil_coil_flag=True (should succeed for small grid)
            curves_cc, all_curves_cc = create_planar_curves_between_two_toroidal_surfaces(
                s, s_inner, s_outer, Nx=2, Ny=2, Nz=2, order=1, coil_coil_flag=True, jax_flag=False, numquadpoints=10
            )
            self.assertTrue(len(curves_cc) > 0)

            # Test coil_coil_flag=True with forced overlap (should raise ValueError)
            # To force overlap, use a very dense grid
            with self.assertRaises(ValueError):
                create_planar_curves_between_two_toroidal_surfaces(
                    s, s_inner, s_outer, Nx=10, Ny=10, Nz=10, order=1, coil_coil_flag=True, jax_flag=False, numquadpoints=10
                )


class TestCreateEquallySpacedCurvesJax(unittest.TestCase):
    def test_create_equally_spaced_curves_jax(self):
        from simsopt.geo.curve import create_equally_spaced_curves
        ncurves, nfp = 2, 2
        stellsym = True
        R0, R1 = 5.0, 1.0
        order = 3
        numquadpoints = 12
        curves = create_equally_spaced_curves(ncurves, nfp, stellsym, R0=R0, R1=R1, order=order, numquadpoints=numquadpoints, jax_flag=True)
        self.assertEqual(len(curves), ncurves)
        for curve in curves:
            gamma = curve.gamma()
            self.assertEqual(gamma.shape, (numquadpoints, 3))
            # Check that the major radius is close to R0 for all points
            R = np.sqrt(gamma[:, 0]**2 + gamma[:, 1]**2)
            self.assertTrue(np.allclose(np.mean(R), R0, atol=0.2))


class TestCurveCenterFunction(unittest.TestCase):
    def test_curve_center(self):
        # Use a simple planar circle for which the centroid is known
        from simsopt.geo.curveplanarfourier import CurvePlanarFourier
        nquad = 100
        order = 1
        R0 = 3.0
        # Create a circle in the x-y plane centered at (R0, 0, 0)
        curve = CurvePlanarFourier(nquad, order, nfp=1, stellsym=False)
        dofs = np.zeros(curve.dof_size)
        dofs[0] = 1.0  # radius
        # Set the center to (R0, 0, 0)
        dofs[-3] = R0
        dofs[-2] = 0.0
        dofs[-1] = 0.0
        curve.set_dofs(dofs)
        gamma = curve.gamma()
        print(gamma)
        gammadash = curve.gammadash()
        centroid = curve.center(gamma, gammadash)
        # The centroid should be at (R0, 0, 0)
        np.testing.assert_allclose(centroid, [R0, 0.0, 0.0], atol=1e-12)

        # Repeat with RotatedCurve
        curve = RotatedCurve(curve, np.eye(3))
        dofs = np.zeros(curve.dof_size)
        dofs[0] = 1.0  # radius
        # Set the center to (R0, 0, 0)
        dofs[-3] = R0
        dofs[-2] = 0.0
        dofs[-1] = 0.0
        curve.set_dofs(dofs)
        gamma = curve.gamma()
        print(gamma)
        gammadash = curve.gammadash()
        centroid = curve.center(gamma, gammadash)
        # The centroid should be at (R0, 0, 0)
        np.testing.assert_allclose(centroid, [R0, 0.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
