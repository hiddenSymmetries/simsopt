import numpy as np
import unittest
import simsoptpp as sopp
from numpy.testing import assert_raises


def get_random_polynomial(dim, degree):
    coeffsx = np.random.standard_normal(size=(degree+1, dim))
    coeffsy = np.random.standard_normal(size=(degree+1, dim))
    coeffsz = np.random.standard_normal(size=(degree+1, dim))

    def fun(x, y, z, flatten=True):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        px = sum([coeffsx[i, :] * x[:, None]**i for i in range(degree+1)])
        py = sum([coeffsy[i, :] * y[:, None]**i for i in range(degree+1)])
        pz = sum([coeffsz[i, :] * z[:, None]**i for i in range(degree+1)])
        res = px*py*pz
        if flatten:
            return (np.ascontiguousarray(res)).flatten()
        else:
            return res
    return fun


class Testing(unittest.TestCase):

    def subtest_regular_grid_interpolant_exact(self, dim, degree):
        """
        Build a random, vector valued polynomial of a specific degree and check
        that it is interpolated exactly.
        """
        xran = (1.0, 4.0, 20)
        yran = (1.1, 3.9, 10)
        zran = (1.2, 3.8, 15)

        fun = get_random_polynomial(dim, degree)

        rule = sopp.UniformInterpolationRule(degree)

        interpolant = sopp.RegularGridInterpolant3D(rule, xran, yran, zran, dim, True)
        interpolant.interpolate_batch(fun)

        nsamples = 100
        xpoints = np.random.uniform(low=xran[0], high=xran[1], size=(nsamples, ))
        ypoints = np.random.uniform(low=yran[0], high=yran[1], size=(nsamples, ))
        zpoints = np.random.uniform(low=zran[0], high=zran[1], size=(nsamples, ))
        xyz = np.asarray([xpoints, ypoints, zpoints]).T.copy()

        fhxyz = np.zeros((nsamples, dim))
        fxyz = fun(xyz[:, 0], xyz[:, 1], xyz[:, 2], flatten=False)

        interpolant.evaluate_batch(xyz, fhxyz)

        assert np.allclose(fxyz, fhxyz, atol=1e-12, rtol=1e-12)
        print(np.max(np.abs((fxyz-fhxyz)/fhxyz)))

    def test_regular_grid_interpolant_exact(self):
        for dim in [1, 3, 4, 6]:
            for degree in [1, 2, 3, 4]:
                with self.subTest(dim=dim, degree=degree):
                    self.subtest_regular_grid_interpolant_exact(dim, degree)

    def test_out_of_bounds(self):
        """
        Check that the interpolant behaves correctly when evaluated outside of the defined domain.
        If created with out_of_bounds_ok=True, then nothing should happen, but if out_of_bounds_ok=False,
        then a runtime error should be raised.
        """
        xran = (1.0, 4.0, 20)
        yran = (1.1, 3.9, 10)
        zran = (1.2, 3.8, 15)

        dim = 3
        degree = 2
        fun = get_random_polynomial(dim, degree)

        rule = sopp.UniformInterpolationRule(degree)

        nsamples = 100
        xpoints = np.random.uniform(low=xran[1]+0.1, high=xran[1]+0.3, size=(nsamples, ))
        ypoints = np.random.uniform(low=yran[1]+0.1, high=yran[1]+0.3, size=(nsamples, ))
        zpoints = np.random.uniform(low=zran[1]+0.1, high=zran[1]+0.3, size=(nsamples, ))

        xyz = np.asarray([xpoints, ypoints, zpoints]).T.copy()

        fhxyz = np.ones((nsamples, dim))

        interpolant = sopp.RegularGridInterpolant3D(rule, xran, yran, zran, dim, True)
        interpolant.interpolate_batch(fun)
        interpolant.evaluate_batch(xyz, fhxyz)
        assert np.allclose(fhxyz, 1., atol=1e-14, rtol=1e-14)

        interpolant = sopp.RegularGridInterpolant3D(rule, xran, yran, zran, dim, False)
        interpolant.interpolate_batch(fun)
        with assert_raises(RuntimeError):
            interpolant.evaluate_batch(xyz, fhxyz)

    # def test_skip(self):

    #     xran = (1.0, 4.0, 20)
    #     yran = (1.1, 3.9, 10)
    #     zran = (1.2, 3.8, 15)

    #     dim = 3
    #     degree = 2
    #     fun = get_random_polynomial(dim, degree)

    #     rule = sopp.UniformInterpolationRule(degree)
