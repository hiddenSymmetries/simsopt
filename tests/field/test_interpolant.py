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
        np.random.seed(0)
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
        Check that the interpolant behaves correctly when evaluated outside of
        the defined domain.  If created with out_of_bounds_ok=True, then
        nothing should happen, but if out_of_bounds_ok=False, then a runtime
        error should be raised.
        """
        np.random.seed(0)
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

    def test_skip(self):
        """
        Check that the interpolant correctly identifies which regions in the
        domain to skip
        """
        np.random.seed(0)
        xran = (1.0, 4.0, 30)
        yran = (1.1, 3.9, 30)
        zran = (1.2, 3.8, 30)

        xkeep = (2.0, 3.0)
        ykeep = (2.0, 3.0)
        zkeep = (2.0, 3.0)

        def skip(xs, ys, zs):
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            zs = np.asarray(zs)
            keep = (xkeep[0] < xs) * (xs < xkeep[1]) * (ykeep[0] < ys) * (ys < ykeep[1]) * (zkeep[0] < zs) * (zs < zkeep[1])
            return np.invert(keep)

        dim = 3
        degree = 2
        fun = get_random_polynomial(dim, degree)

        rule = sopp.UniformInterpolationRule(degree)

        interpolant = sopp.RegularGridInterpolant3D(rule, xran, yran, zran, dim, True, skip)
        interpolant.interpolate_batch(fun)

        xyz = np.asarray([
            [2.4, 2.6, 2.8],  # keep
            [2.1, 2.1, 2.9],  # keep
            [2.8, 2.8, 2.1],  # keep
            [1.3, 1.3, 1.3],  # do not keep
            [1.3, 2.9, 3.5],  # do not keep
            [3.5, 1.3, 1.3],  # do not keep
        ])
        fhxyz = 100*np.ones((xyz.shape[0], dim))

        interpolant.evaluate_batch(xyz, fhxyz)
        print("fhxyz %s" % (fhxyz))

        fxyz = fun(xyz[:, 0], xyz[:, 1], xyz[:, 2], flatten=False)
        assert np.allclose(fhxyz[:3, :], fxyz[:3, :], atol=1e-12, rtol=1e-12)
        assert np.allclose(fhxyz[3:, :], 100, atol=1e-12, rtol=1e-12)

    def test_convergence_order(self):
        for dim in [1, 4, 6]:
            for degree in [1, 3]:
                with self.subTest(dim=dim, degree=degree):
                    self.subtest_convergence_order(dim, degree)

    def subtest_convergence_order(self, dim, degree):
        """
        Check that the interpolant converges at the correct order
        """
        np.random.seed(0)

        fun = get_random_polynomial(dim, degree+1)

        rule = sopp.UniformInterpolationRule(degree)
        nsamples = 1000
        xran = [1.0, 4.0, 10]
        yran = [1.1, 3.9, 10]
        zran = [1.2, 3.8, 10]
        xpoints = np.random.uniform(low=xran[0], high=xran[1], size=(nsamples, ))
        ypoints = np.random.uniform(low=yran[0], high=yran[1], size=(nsamples, ))
        zpoints = np.random.uniform(low=zran[0], high=zran[1], size=(nsamples, ))
        xyz = np.asarray([xpoints, ypoints, zpoints]).T.copy()
        fhxyz = np.zeros((nsamples, dim))
        fxyz = fun(xyz[:, 0], xyz[:, 1], xyz[:, 2], flatten=False)

        err = 1e6
        for n in [5, 10, 20, 40]:
            xran[2] = n
            yran[2] = n
            zran[2] = n
            interpolant = sopp.RegularGridInterpolant3D(rule, xran, yran, zran, dim, True)
            interpolant.interpolate_batch(fun)
            interpolant.evaluate_batch(xyz, fhxyz)
            err_new = np.mean(np.linalg.norm(fxyz-fhxyz, axis=1))
            print(err_new/err)
            assert err_new/err < 0.6**(degree+1)
            err = err_new
