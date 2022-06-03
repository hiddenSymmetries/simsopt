import unittest

import numpy as np
from simsopt.geo import parameters
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
parameters['jit'] = False


def taylor_test(f, df, x, epsilons=None, direction=None, order=2):
    np.random.seed(1)
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(8, 20)))
    print("###################################################################")
    err_old = 1e9
    counter = 0
    for eps in epsilons:
        if counter > 8:
            break
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        if order == 2:
            fak = 0.3
            dfest = (fpluseps-fminuseps)/(2*eps)
        elif order == 4:
            fplus2eps = f(x + 2*eps * direction)
            fminus2eps = f(x - 2*eps * direction)
            fak = 0.13
            dfest = ((1/12) * fminus2eps - (2/3) * fminuseps + (2/3)*fpluseps
                     - (1/12)*fplus2eps)/eps
        else:
            raise NotImplementedError
        err = np.linalg.norm(dfest - dfx)
        print(err)
        assert err < 1e-9 or err < 0.3 * err_old
        counter += 1
        if err < 1e-9:
            break
        err_old = err
    if err > 1e-10:
        assert counter > 2
    print("###################################################################")


def get_surface(surfacetype, stellsym, phis=None, thetas=None):
    np.random.seed(2)
    mpol = 4
    ntor = 3
    nfp = 2
    phis = phis if phis is not None else np.linspace(0, 1, 31, endpoint=False)
    thetas = thetas if thetas is not None else np.linspace(0, 1, 31, endpoint=False)
    if surfacetype == "SurfaceRZFourier":
        s = SurfaceRZFourier(nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor,
                             quadpoints_phi=phis, quadpoints_theta=thetas)
        s.x = s.x * 0.
        s.rc[0, ntor + 0] = 1
        s.rc[1, ntor + 0] = 0.3
        s.zs[1, ntor + 0] = 0.3
    elif surfacetype == "SurfaceXYZFourier":
        s = SurfaceXYZFourier(nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor,
                              quadpoints_phi=phis, quadpoints_theta=thetas)
        s.x = s.x * 0.
        s.xc[0, ntor + 1] = 1.
        s.xc[1, ntor + 1] = 0.1
        s.ys[0, ntor + 1] = 1.
        s.ys[1, ntor + 1] = 0.1
        s.zs[1, ntor] = 0.1
    elif surfacetype == "SurfaceXYZTensorFourier":
        s = SurfaceXYZTensorFourier(
            nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor,
            clamped_dims=[False, not stellsym, True],
            quadpoints_phi=phis, quadpoints_theta=thetas)
        s.x = s.x * 0.
        s.xcs[0, 0] = 1.0
        s.xcs[1, 0] = 0.1
        s.zcs[mpol+1, 0] = 0.1
    else:
        assert False

    dofs = s.get_dofs()
    np.random.seed(2)
    rand_scale = 0.01
    s.x = dofs + rand_scale * np.random.rand(len(dofs))  # .reshape(dofs.shape)
    return s


def taylor_test2(f, df, d2f, x, epsilons=None, direction1=None,
                 direction2=None):
    np.random.seed(1)
    if direction1 is None:
        direction1 = np.random.rand(*(x.shape))-0.5
    if direction2 is None:
        direction2 = np.random.rand(*(x.shape))-0.5

    f0 = f(x)
    df0 = df(x) @ direction1
    d2fval = direction2.T @ d2f(x) @ direction1
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(7, 20)))
    print("###################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = df(x + eps * direction2) @ direction1
        d2fest = (fpluseps-df0)/eps
        err = np.abs(d2fest - d2fval)

        print(err/err_old)
        assert err < 0.6 * err_old
        err_old = err
    print("###################################################################")


class SurfaceTaylorTests(unittest.TestCase):
    surfacetypes = ["SurfaceRZFourier", "SurfaceXYZFourier",
                    "SurfaceXYZTensorFourier"]

    def subtest_surface_coefficient_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return s.gamma()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dgamma_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.x = dofs
            return s.gammadash1()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dgammadash1_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.x = dofs
            return s.gammadash2()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dgammadash2_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.x = dofs
            return s.gammadash2dash2()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dgammadash2dash2_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.x = dofs
            return s.gammadash1dash1()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dgammadash1dash1_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.x = dofs
            return s.gammadash1dash2()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dgammadash1dash2_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

    def test_surface_coefficient_derivative(self):
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_coefficient_derivative(s)

    def subtest_surface_normal_coefficient_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return s.normal()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dnormal_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

    def test_surface_normal_coefficient_derivative(self):
        """
        Taylor test for the first derivative of the surface normal w.r.t. the dofs
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_normal_coefficient_derivative(s)

    def subtest_fund_form_coefficient_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return s.first_fund_form()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dfirst_fund_form_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs, epsilons=np.power(2., -np.asarray(range(10, 15))), order=4)

        def f(dofs):
            s.x = dofs
            return s.second_fund_form()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dsecond_fund_form_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs, epsilons=np.power(2., -np.asarray(range(10, 15))), order=4)

        def f(dofs):
            s.x = dofs
            return s.surface_curvatures()[1, 1, 2].copy()

        def df(dofs):
            s.x = dofs
            return s.dsurface_curvatures_by_dcoeff()[1, 1, 2, :].copy()
        taylor_test(f, df, coeffs, epsilons=np.power(2., -np.asarray(range(10, 15))), order=4)

    def test_fund_form_coefficient_derivative(self):
        """
        Taylor test for the first derivative of the surface normal w.r.t. the dofs
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_fund_form_coefficient_derivative(s)

    def subtest_unit_normal_coefficient_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return s.unitnormal()[1, 1, :].copy()

        def df(dofs):
            s.x = dofs
            return s.dunitnormal_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs, epsilons=np.power(2., -np.asarray(range(10, 15))), order=2)

    def test_unit_normal_coefficient_derivative(self):
        """
        Taylor test for the first derivative of the surface normal w.r.t. the dofs
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_unit_normal_coefficient_derivative(s)

    def subtest_surface_area_coefficient_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return np.asarray(s.area())

        def df(dofs):
            s.x = dofs
            return s.darea_by_dcoeff()[None, :].copy()
        taylor_test(f, df, coeffs,
                    epsilons=np.power(2., -np.asarray(range(11, 20))), order=4)

    def test_surface_area_coefficient_derivative(self):
        """
        Taylor test for the first derivative of the surface area w.r.t. the dofs
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_area_coefficient_derivative(s)

    def test_surface_area_coefficient_second_derivative(self):
        """
        Taylor test for second derivative of the surface area w.r.t. the dofs
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_area_coefficient_second_derivative(s)

    def subtest_surface_area_coefficient_second_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return s.area()

        def df(dofs):
            s.x = dofs
            return s.darea_by_dcoeff()

        def d2f(dofs):
            s.x = dofs
            return s.d2area_by_dcoeffdcoeff()
        taylor_test2(f, df, d2f, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 20))))

    def test_volume_coefficient_second_derivative(self):
        """
        Taylor test for the second derivative of the volume w.r.t. the dofs
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_volume_coefficient_second_derivative(s)

    def subtest_volume_coefficient_second_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return s.volume()

        def df(dofs):
            s.x = dofs
            return s.dvolume_by_dcoeff()

        def d2f(dofs):
            s.x = dofs
            return s.d2volume_by_dcoeffdcoeff()
        taylor_test2(f, df, d2f, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 20))))

    def subtest_surface_volume_coefficient_derivative(self, s):
        coeffs = s.x
        s.invalidate_cache()

        def f(dofs):
            s.x = dofs
            return np.asarray(s.volume())

        def df(dofs):
            s.x = dofs
            return s.dvolume_by_dcoeff()[None, :].copy()
        taylor_test(f, df, coeffs)

    def test_surface_volume_coefficient_derivative(self):
        """
        Taylor test to verify the first derivative of the volume with respect to the surface dofs
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_volume_coefficient_derivative(s)

    def subtest_surface_phi_derivative(self, surfacetype, stellsym):
        epss = [0.5**i for i in range(10, 15)]
        phis = np.asarray([0.6] + [0.6 + eps for eps in epss])
        s = get_surface(surfacetype, stellsym, phis=phis)
        f0 = s.gamma()[0, 0, :]
        deriv = s.gammadash1()[0, 0, :]
        err_old = 1e6
        for i in range(len(epss)):
            fh = s.gamma()[i+1, 0, :]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_surface_phi_derivative(self):
        """
        Taylor test to verify that the surface tangent in the phi direction
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_phi_derivative(surfacetype, stellsym)

    def subtest_surface_theta_derivative(self, surfacetype, stellsym):
        epss = [0.5**i for i in range(10, 15)]
        thetas = np.asarray([0.6] + [0.6 + eps for eps in epss])
        s = get_surface(surfacetype, stellsym, thetas=thetas)
        f0 = s.gamma()[0, 0, :]
        deriv = s.gammadash2()[0, 0, :]
        err_old = 1e6
        for i in range(len(epss)):
            fh = s.gamma()[0, i+1, :]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_surface_theta_derivative(self):
        """
        Taylor test to verify that the surface tangent in the theta direction
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_theta_derivative(surfacetype, stellsym)

    def subtest_surface_theta2_derivative(self, surfacetype, stellsym):
        epss = [0.5**i for i in range(5, 10)]
        thetas = np.asarray([0.6] + [0.6 + eps for eps in epss])
        s = get_surface(surfacetype, stellsym, thetas=thetas)
        f0 = s.gammadash2()[0, 0, :]
        deriv = s.gammadash2dash2()[0, 0, :]
        err_old = 1e6
        for i in range(len(epss)):
            fh = s.gammadash2()[0, i+1, :]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_surface_theta2_derivative(self):
        """
        Taylor test to verify that the surface tangent in the theta direction
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_theta2_derivative(surfacetype, stellsym)

    def subtest_surface_phi2_derivative(self, surfacetype, stellsym):
        epss = [0.5**i for i in range(10, 15)]
        phis = np.asarray([0.6] + [0.6 + eps for eps in epss])
        s = get_surface(surfacetype, stellsym, phis=phis)
        f0 = s.gammadash1()[0, 0, :]
        deriv = s.gammadash1dash1()[0, 0, :]
        err_old = 1e6
        for i in range(len(epss)):
            fh = s.gammadash1()[i+1, 0, :]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_surface_phi2_derivative(self):
        """
        Taylor test to verify that the surface tangent in the theta direction
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_phi2_derivative(surfacetype, stellsym)

    def subtest_surface_thetaphi_derivative(self, surfacetype, stellsym):
        epss = [0.5**i for i in range(10, 15)]
        phis = np.asarray([0.6] + [0.6 + eps for eps in epss])
        thetas = np.asarray([0.3] + [0.3 + eps for eps in epss])
        s = get_surface(surfacetype, stellsym, phis=phis, thetas=thetas)
        f0 = s.gammadash1()[0, 0, :]
        deriv = s.gammadash1dash2()[0, 0, :]
        err_old = 1e6
        for i in range(len(epss)):
            fh = s.gammadash1()[0, i+1, :]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_surface_thetaphi_derivative(self):
        """
        Taylor test to verify that the surface tangent in the theta direction
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_thetaphi_derivative(surfacetype, stellsym)

    def subtest_surface_conversion(self, surfacetype, stellsym):
        s = get_surface(surfacetype, stellsym)
        newsurf = s.to_RZFourier()
        assert np.mean((s.gamma() - newsurf.gamma())**2) < 1e-5

    def test_surface_conversion(self):
        """
        Test to verify that the toRZFourier surface conversion
        """
        for surfacetype in ["SurfaceXYZFourier"]:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_theta_derivative(surfacetype, stellsym)


if __name__ == "__main__":
    unittest.main()
