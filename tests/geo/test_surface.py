import numpy as np
import unittest
from simsopt.geo import parameters
parameters['jit'] = False

def taylor_test(f, df, x, epsilons=None, direction=None):
    np.random.seed(1)
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(8, 20)))
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
        print(err)
        assert err < 1e-9 or err < 0.3 * err_old
        if err < 1e-9:
            break
        err_old = err
        counter += 1
    if err > 1e-10:
        assert counter > 3
    print("################################################################################")

def get_surface(surfacetype, stellsym, phis=None, thetas=None):
    mpol = 4
    ntor = 3
    nfp = 2
    phis = phis if phis is not None else np.linspace(0, 1, 31, endpoint=False)
    thetas = thetas if thetas is not None else np.linspace(0, 1, 31, endpoint=False)
    if surfacetype == "SurfaceRZFourier":
        from simsopt.geo.surfacerzfourier import SurfaceRZFourier
        s = SurfaceRZFourier(mpol, ntor, nfp, stellsym, phis, thetas)
        s.rc[0, ntor + 0] = 1
        s.rc[1, ntor + 0] = 0.3
        s.zs[1, ntor + 0] = 0.3
    elif surfacetype == "SurfaceXYZFourier":
        from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
        s = SurfaceXYZFourier(mpol, ntor, nfp, stellsym, phis, thetas)
        s.xc[0, ntor + 1] = 1.
        s.xc[1, ntor + 1] = 0.1
        s.ys[0, ntor + 1] = 1.
        s.ys[1, ntor + 1] = 0.1
        s.zs[1, ntor] = 0.1
    else:
        assert False

    dofs = np.asarray(s.get_dofs())
    np.random.seed(2)
    rand_scale=0.01
    s.set_dofs(dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape))
    return s


class Testing(unittest.TestCase):

    surfacetypes = ["SurfaceRZFourier", "SurfaceXYZFourier"]

    def subtest_surface_coefficient_derivative(self, s):
        coeffs = s.get_dofs()
        s.invalidate_cache()
        def f(dofs):
            s.set_dofs(dofs)
            return s.gamma()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dgamma_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.set_dofs(dofs)
            return s.gammadash1()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dgammadash1_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.set_dofs(dofs)
            return s.gammadash2()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dgammadash2_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

    def test_surface_coefficient_derivative(self):
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_coefficient_derivative(s)


    def subtest_surface_normal_coefficient_derivative(self, s):
        coeffs = s.get_dofs()
        s.invalidate_cache()
        def f(dofs):
            s.set_dofs(dofs)
            return s.normal()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dnormal_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)


    def test_surface_normal_coefficient_derivative(self):
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_normal_coefficient_derivative(s)

    def subtest_surface_area_coefficient_derivative(self, s):
        coeffs = s.get_dofs()
        s.invalidate_cache()
        def f(dofs):
            s.set_dofs(dofs)
            return np.asarray(s.surface_area())
        def df(dofs):
            s.set_dofs(dofs)
            return s.dsurface_area_by_dcoeff()[None, :].copy()
        taylor_test(f, df, coeffs)


    def test_surface_area_coefficient_derivative(self):
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym)
                    self.subtest_surface_area_coefficient_derivative(s)

    def subtest_surface_phi_derivative(self, surfacetype, stellsym):
        h = 0.1
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
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_phi_derivative(surfacetype, stellsym)

    def subtest_surface_theta_derivative(self, surfacetype, stellsym):
        h = 0.1
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
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_surface_theta_derivative(surfacetype, stellsym)

    def test_aspect_ratio(self):
        mpol = 4
        ntor = 3
        nfp = 2
        phis = np.linspace(0, 1, 31, endpoint=False)
        thetas = np.linspace(0, 1, 31, endpoint=False)

        stellsym = False
        from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
        s = SurfaceXYZFourier(mpol, ntor, nfp, stellsym, phis, thetas)
        s.xc[0, ntor] = 1.
        s.xc[1, ntor] = 0.1
        s.zs[1, ntor] = 0.1
        assert np.abs(s.aspect_ratio() - 10) < 1e-13



if __name__ == "__main__":
    print('wtf')
    unittest.main()
