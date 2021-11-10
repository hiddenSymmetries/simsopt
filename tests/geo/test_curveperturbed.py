import unittest
from simsopt.util.zoo import get_ncsx_data
from simsopt.geo.curveperturbed import GaussianSampler, PerturbationSample, CurvePerturbed
from randomgen import PCG64
import numpy as np


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
                assert np.mean(err) < 1e-4
            else:
                assert np.mean(err) < 1e-3
