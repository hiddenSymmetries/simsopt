import unittest
from randomgen import PCG64
import json

import numpy as np
from monty.json import MontyDecoder, MontyEncoder

from simsopt.geo.curveperturbed import GaussianSampler, PerturbationSample, CurvePerturbed
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveobjectives import LpCurveTorsion, CurveCurveDistance


class CurvePerturbationTesting(unittest.TestCase):

    def test_perturbed_gammadash(self):
        sigma = 1
        length_scale = 0.5
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=2)
        rg = np.random.Generator(PCG64(1))
        sample = PerturbationSample(sampler, randomgen=rg)

        dphi = points[1]

        for idx in range(2):
            g = sample[idx + 0]
            gd = sample[idx + 1]

            gdest = (-1/12) * g[4:, :] + (2/3) * g[3:-1, :] + 0 * g[2:-2, :] + (-2/3) * g[1:-3, :] + (1/12) * g[0:-4, :]
            gdest *= 1/dphi
            err = np.abs(gdest - gd[2:-2, :])

            print("np.mean(err)", np.mean(err))
            if idx == 0:
                assert np.mean(err) < 3e-4
            else:
                assert np.mean(err) < 2e-3

    def test_perturbed_periodic(self):
        sigma = 1
        length_scale = 0.5
        n = 100
        points = np.linspace(0, 2, 2*n, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=0)
        rg = np.random.Generator(PCG64(1))
        sample = PerturbationSample(sampler, randomgen=rg)
        periodic_err = np.abs(sample[0][:n, :] - sample[0][n:, :])
        print("periodic_err", np.mean(periodic_err))
        assert np.mean(periodic_err) < 1e-6

    def test_perturbed_objective_torsion(self):
        # test the torsion objective as that covers all derivatives (up to
        # third) of a curve
        sigma = 1
        length_scale = 0.5
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=3)
        rg = np.random.Generator(PCG64(1))
        sample = PerturbationSample(sampler, randomgen=rg)

        order = 4
        nquadpoints = 200
        curve = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve.dof_size, ))
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
        curve.x = dofs

        curve = CurvePerturbed(curve, sample)

        J = LpCurveTorsion(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
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

    def test_perturbed_objective_distance(self):
        # test the distance objective as that covers the position and the first
        # derivative of the curve
        sigma = 1
        length_scale = 0.2
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=1)
        rg = np.random.Generator(PCG64(1))
        sample1 = PerturbationSample(sampler, randomgen=rg)
        sample2 = PerturbationSample(sampler, randomgen=rg)

        order = 4
        nquadpoints = 200
        curve1 = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve1.dof_size, ))
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
        curve1.x = dofs

        curve2 = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve2.dof_size, ))
        dofs[1] = 2.
        dofs[2*order+3] = 2.
        dofs[4*order+3] = 2.
        curve2.x = dofs

        curve1 = CurvePerturbed(curve1, sample1)
        curve2 = CurvePerturbed(curve2, sample2)

        J = CurveCurveDistance([curve1, curve2], 2.0)
        J0 = J.J()
        curve1.resample()
        assert J0 != J.J()
        J0 = J.J()
        curve_dofs = J.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(2, 10):
            eps = 0.5**i
            J.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

    def test_serialization(self):
        sigma = 1
        length_scale = 0.5
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=2)
        sample = PerturbationSample(sampler)

        order = 4
        nquadpoints = 200
        curve = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve.dof_size,))
        dofs[1] = 1.
        dofs[2 * order + 3] = 1.
        dofs[4 * order + 3] = 1.
        curve.x = dofs
        curve_per = CurvePerturbed(curve, sample)

        curve_str = json.dumps(curve_per, cls=MontyEncoder)
        curve_per_regen = json.loads(curve_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(curve_per.gamma(), curve_per_regen.gamma()))
