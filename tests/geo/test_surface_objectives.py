import unittest
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import coils_via_symmetries
from simsopt.geo.surfaceobjectives import ToroidalFlux, QfmResidual, parameter_derivatives, Volume, PrincipalCurvature
from simsopt.configs.zoo import get_ncsx_data
from .surface_test_helpers import get_surface, get_exact_surface


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceRZFourier",
                     "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


def taylor_test1(f, df, x, epsilons=None, direction=None):
    np.random.seed(1)
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(10, 20)))
    print("###################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps-fminuseps)/(2*eps)
        err = np.linalg.norm(dfest - dfx)
        print(err, err/err_old)
        assert err < 1e-9 or err < 0.3 * err_old
        err_old = err
    print("###################################################################")


def taylor_test2(f, df, d2f, x, epsilons=None, direction1=None, direction2=None):
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
        print('err: ', err)
        print(err/err_old)
        assert err < 0.6 * err_old
        err_old = err
    print("###################################################################")


class ToroidalFluxTests(unittest.TestCase):
    def test_toroidal_flux_is_constant(self):
        """
        this test ensures that the toroidal flux does not change, regardless
        of the cross section (varphi = constant) across which it is computed
        """
        s = get_exact_surface()
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs_tf = BiotSavart(coils)

        gamma = s.gamma()
        num_phi = gamma.shape[0]

        tf_list = np.zeros((num_phi,))
        for idx in range(num_phi):
            tf = ToroidalFlux(s, bs_tf, idx=idx)
            tf_list[idx] = tf.J()
        mean_tf = np.mean(tf_list)

        max_err = np.max(np.abs(mean_tf - tf_list)) / mean_tf
        assert max_err < 1e-2

    def test_toroidal_flux_first_derivative(self):
        """
        Taylor test for gradient of toroidal flux
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux1(surfacetype, stellsym)

    def test_toroidal_flux_second_derivative(self):
        """
        Taylor test for Hessian of toroidal flux
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux2(surfacetype, stellsym)

    def subtest_toroidal_flux1(self, surfacetype, stellsym):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs_tf = BiotSavart(coils)
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs_tf)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return tf.J()

        def df(dofs):
            s.x = dofs
            return tf.dJ_by_dsurfacecoefficients()
        taylor_test1(f, df, coeffs)

    def subtest_toroidal_flux2(self, surfacetype, stellsym):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return tf.J()

        def df(dofs):
            s.x = dofs
            return tf.dJ_by_dsurfacecoefficients()

        def d2f(dofs):
            s.x = dofs
            return tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        taylor_test2(f, df, d2f, coeffs)


class PrincipalCurvatureTests(unittest.TestCase):
    def test_principal_curvature_first_derivative(self):
        """
        Taylor test for gradient of principal curvature metric.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_principal_curvature(surfacetype, stellsym)

    def subtest_principal_curvature(self, surfacetype, stellsym):
        s = get_surface(surfacetype, stellsym)

        pc = PrincipalCurvature(s, kappamax1=1, kappamax2=2.2, weight1=1, weight2=2.)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return pc.J()

        def df(dofs):
            s.x = dofs
            return pc.dJ()

        taylor_test1(f, df, coeffs, epsilons=np.power(2., -np.asarray(range(13, 22))))


class ParameterDerivativesTest(unittest.TestCase):
    def test_parameter_derivatives_volume(self):
        """
        Test that parameter derivatives of volume (shape_gradient = 1) match
        parameter derivatives computed from Volume class.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_volume(surfacetype, stellsym)

    def subtest_volume(self, surfacetype, stellsym):
        s = get_surface(surfacetype, stellsym, mpol=7, ntor=6,
                        ntheta=32, nphi=31, full=True)
        dofs = s.get_dofs()

        vol = Volume(s)
        dvol_sg = parameter_derivatives(s, np.ones_like(s.gamma()[:, :, 0]))
        dvol = vol.dJ_by_dsurfacecoefficients()
        for i in range(len(dofs)):
            self.assertAlmostEqual(dvol_sg[i], dvol[i], places=10)


class QfmTests(unittest.TestCase):
    def test_qfm_surface_derivative(self):
        """
        Taylor test for derivative of qfm metric wrt surface parameters
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_qfm1(surfacetype, stellsym)

    def subtest_qfm1(self, surfacetype, stellsym):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        s = get_surface(surfacetype, stellsym)
        coeffs = s.x
        qfm = QfmResidual(s, bs)

        def f(dofs):
            s.x = dofs
            return qfm.J()

        def df(dofs):
            s.x = dofs
            return qfm.dJ_by_dsurfacecoefficients()
        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 22))))
