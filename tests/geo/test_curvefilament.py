import unittest
from simsopt.util.zoo import get_ncsx_data
from simsopt.geo.curvefilament import CurveFilament, FilamentRotation
import numpy as np


class MultifilamentTesting(unittest.TestCase):

    def test_multifilament_gammadash(self):
        curves, currents, ma = get_ncsx_data(Nt_coils=6, ppp=80)
        c = curves[0]

        rotation = FilamentRotation(c.quadpoints, 1)
        rotation.x = np.array([0, 0.1, 0.3])
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

        curves, currents, ma = get_ncsx_data(Nt_coils=4, ppp=10)
        c = curves[0]

        rotation = FilamentRotation(c.quadpoints, 1)
        rotation.x = np.array([0, 0.1, 0.1])

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
