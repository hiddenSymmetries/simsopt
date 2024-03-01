import unittest
from .surface_test_helpers import get_surface, get_exact_surface
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance
from simsopt.geo.finitebuild import CurveFilament, FilamentRotation, \
    create_multifilament_grid, ZeroRotation
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.configs.zoo import get_ncsx_data

import numpy as np


class MultifilamentTesting(unittest.TestCase):

    def test_multifilament_gammadash(self):
        for order in [None, 1]:
            with self.subTest(order=order):
                self.subtest_multifilament_gammadash(order)

    def subtest_multifilament_gammadash(self, order):
        assert order in [1, None]
        curves, currents, ma = get_ncsx_data(Nt_coils=6, ppp=80)
        c = curves[0]

        if order == 1:
            rotation = FilamentRotation(c.quadpoints, order)
            rotation.x = np.array([0, 0.1, 0.3])
        else:
            rotation = ZeroRotation(c.quadpoints)

        c = CurveFilament(c, 0.01, 0.01, rotation)
        g = c.gamma()
        gd = c.gammadash()
        idx = 16

        dphi = c.quadpoints[1]
        weights = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
        est = 0
        for j in range(-4, 5):
            est += weights[j+4] * g[idx+j, :]
        est *= 1./dphi
        print(est)
        print(gd[idx])
        assert np.all(np.abs(est - gd[idx]) < 1e-10)

    def test_multifilament_coefficient_derivative(self):
        for order in [None, 1]:
            with self.subTest(order=order):
                self.subtest_multifilament_coefficient_derivative(order)

    def subtest_multifilament_coefficient_derivative(self, order):
        assert order in [1, None]

        curves, currents, ma = get_ncsx_data(Nt_coils=4, ppp=10)
        c = curves[0]

        if order == 1:
            rotation = FilamentRotation(c.quadpoints, order)
            rotation.x = np.array([0, 0.1, 0.3])
        else:
            rotation = ZeroRotation(c.quadpoints)

        c = CurveFilament(c, 0.02, 0.02, rotation)

        dofs = c.x

        g = c.gamma()
        v = np.ones_like(g)
        np.random.seed(1)

        v = np.random.standard_normal(size=g.shape)
        h = np.random.standard_normal(size=dofs.shape)
        df = np.sum(c.dgamma_by_dcoeff_vjp(v)(c)*h)
        dg = np.sum(c.dgammadash_by_dcoeff_vjp(v)(c)*h)

        errf_old = 1e10
        errg_old = 1e10

        for i in range(12, 17):
            eps = 0.5**i
            c.x = dofs + eps*h
            f1 = np.sum(c.gamma()*v)
            c.x = dofs - eps*h
            f2 = np.sum(c.gamma()*v)
            errf = (f1-f2)/(2*eps) - df
            print(errf)
            assert errf < 0.3 * errf_old
            errf_old = errf

        print("==============")
        for i in range(10, 17):
            eps = 0.5**i
            c.x = dofs + eps*h
            g1 = np.sum(c.gammadash()*v)
            c.x = dofs - eps*h
            g2 = np.sum(c.gammadash()*v)
            errg = (g1-g2)/(2*eps) - dg
            # errg = (g1-g0)/(eps) - dg
            print(errg)
            assert errg < 0.3 * errg_old
            errg_old = errg

    def test_filamentpack(self):
        curves, currents, ma = get_ncsx_data(Nt_coils=6, ppp=80)
        c = curves[0]

        gapsize_n = 0.01
        gapsize_b = 0.02

        def check(fils, c, numfilaments_n, numfilaments_b):
            assert len(fils) == numfilaments_n * numfilaments_b
            dists = np.linalg.norm(fils[0].gamma()-fils[-1].gamma(), axis=1)
            # check that filaments are equidistant everywhere
            assert np.var(dists) < 1e-16
            # check that first and last filament are on opossing corners of filament pack and have correct distance
            assert abs(dists[0] - (((numfilaments_n-1)*gapsize_n) ** 2+((numfilaments_b-1)*gapsize_b)**2)**0.5) < 1e-13
            # check that the coil pack is centered around the underlying curve
            assert np.linalg.norm(np.mean([f.gamma() for f in fils], axis=0)-c.gamma()) < 1e-13

        numfilaments_n = 2
        numfilaments_b = 3
        fils = create_multifilament_grid(
            c, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b,
            rotation_order=None, rotation_scaling=None)
        check(fils, c, numfilaments_n, numfilaments_b)

        numfilaments_n = 3
        numfilaments_b = 2
        fils = create_multifilament_grid(
            c, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b,
            rotation_order=None, rotation_scaling=None)
        check(fils, c, numfilaments_n, numfilaments_b)

        fils = create_multifilament_grid(
            c, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b,
            rotation_order=3, rotation_scaling=None)
        xr = fils[0].rotation.x
        fils[0].rotation.x = xr + 1e-2*np.random.standard_normal(size=xr.shape)
        check(fils, c, numfilaments_n, numfilaments_b)

    def test_biotsavart_with_symmetries(self):
        """
        More involved test that checks whether the multifilament code interacts
        properly with symmetries, biot savart, and objectives that only depend
        on the underlying curve (not the finite build filaments)
        """
        np.random.seed(1)
        base_curves, base_currents, ma = get_ncsx_data(Nt_coils=5)
        base_curves_finite_build = sum(
            [create_multifilament_grid(c, 2, 2, 0.01, 0.01, rotation_order=1) for c in base_curves], [])
        base_currents_finite_build = sum([[c]*4 for c in base_currents], [])

        nfp = 3

        curves = apply_symmetries_to_curves(base_curves, nfp, True)
        curves_fb = apply_symmetries_to_curves(base_curves_finite_build, nfp, True)
        currents_fb = apply_symmetries_to_currents(base_currents_finite_build, nfp, True)

        coils_fb = [Coil(c, curr) for (c, curr) in zip(curves_fb, currents_fb)]

        bs = BiotSavart(coils_fb)
        s = get_surface("SurfaceXYZFourier", True)
        s.fit_to_curve(ma, 0.1)
        Jf = SquaredFlux(s, bs)
        Jls = [CurveLength(c) for c in base_curves]
        Jdist = CurveCurveDistance(curves, 0.5)
        LENGTH_PEN = 1e-2
        DIST_PEN = 1e-2
        JF = Jf \
            + LENGTH_PEN * sum(QuadraticPenalty(Jls[i], Jls[i].J()) for i in range(len(base_curves))) \
            + DIST_PEN * Jdist

        def fun(dofs, grad=True):
            JF.x = dofs
            return (JF.J(), JF.dJ()) if grad else JF.J()

        dofs = JF.x
        dofs += 1e-2 * np.random.standard_normal(size=dofs.shape)
        np.random.seed(1)
        h = np.random.uniform(size=dofs.shape)
        J0, dJ0 = fun(dofs)
        dJh = sum(dJ0 * h)
        err = 1e6
        for i in range(10, 15):
            eps = 0.5**i
            J1 = fun(dofs + eps*h, grad=False)
            J2 = fun(dofs - eps*h, grad=False)
            err_new = abs((J1-J2)/(2*eps) - dJh)
            assert err_new < 0.55**2 * err
            err = err_new
            print("err", err)
