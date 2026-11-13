import unittest

import numpy as np

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, ToroidalFlux, Volume, BoozerSurface, Iotas, \
    MajorRadius
from simsopt.geo.toruxsurface import ToruxSurface, torux_available
from .surface_test_helpers import get_boozer_surface
from .test_surface_objectives import taylor_test1


def get_torux_surface(mpol, ntor, verify=True):
    """
    The flux surface of SIMSOPT's NCSX test coils through which the toroidal flux is that of the
    BoozerExact surface of :obj:`get_boozer_surface`, as a ToruxSurface of resolution ``mpol``, ``ntor``.
    """
    bs, boozer_surface = get_boozer_surface(label="ToroidalFlux")
    s = SurfaceRZFourier(nfp=3, stellsym=True, mpol=mpol, ntor=ntor,
                         quadpoints_phi=np.linspace(0, 1/3, 64, endpoint=False),
                         quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    torux_surface = ToruxSurface(bs, s, ToroidalFlux(s, BiotSavart(bs.coils)), boozer_surface.targetlabel,
                                 options={'verify': verify})
    return bs, boozer_surface, torux_surface


@unittest.skipIf(not torux_available, "torux not found (it needs Python 3.12 or newer)")
class ToruxSurfaceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # torux's own resolution for this surface: the solution is copied, not recomputed
        cls.bs, cls.boozer_surface, cls.torux_surface = get_torux_surface(mpol=32, ntor=48)
        cls.res = cls.torux_surface.run_code()

    def test_flux_surface(self):
        """
        The surface torux finds is a flux surface of the coils, with the requested toroidal flux.
        """
        res, s = self.res, self.torux_surface.surface
        self.assertTrue(res['success'])
        self.assertEqual(res['torux_mpol_ntor'], (s.mpol, s.ntor))
        self.assertEqual(res['iter'], 0)

        self.bs.set_points(s.gamma().reshape((-1, 3)))
        B = self.bs.B().reshape(s.gamma().shape)
        n = s.normal()
        leak = np.abs(np.sum(B * n, axis=2)) / (np.linalg.norm(B, axis=2) * np.linalg.norm(n, axis=2))
        self.assertLess(np.max(leak), 1e-12)

        flux = self.torux_surface.label.J()
        np.testing.assert_allclose(abs(flux), abs(self.torux_surface.targetlabel), rtol=1e-12)

    def test_iota_against_boozer_exact(self):
        """
        BoozerExact's rotational transform on the same coils and toroidal flux approaches torux's as
        its resolution grows. iota changes sign with the direction of theta, as the volume does.
        """
        iota = np.sign(self.torux_surface.surface.volume()) * self.res['iota']
        errors = []
        s_prev, res = self.boozer_surface.surface, self.boozer_surface.res
        for k in [6, 10]:
            phis = np.linspace(0, 1/3, 2*k+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*k+1, endpoint=False)
            s = SurfaceXYZTensorFourier(mpol=k, ntor=k, stellsym=True, nfp=3, quadpoints_phi=phis, quadpoints_theta=thetas)
            s_fit = SurfaceXYZTensorFourier(mpol=s_prev.mpol, ntor=s_prev.ntor, stellsym=True, nfp=3,
                                            quadpoints_phi=phis, quadpoints_theta=thetas, dofs=s_prev.dofs)
            s.least_squares_fit(s_fit.gamma())
            boozer_surface = BoozerSurface(self.bs, s, ToroidalFlux(s, BiotSavart(self.bs.coils)),
                                           self.torux_surface.targetlabel, options={'newton_tol': 1e-12})
            res = boozer_surface.run_code(res['iota'], G=res['G'])
            self.assertTrue(res['success'])
            errors.append(abs(np.sign(s.volume()) * res['iota'] - iota))
            s_prev = s
        self.assertLess(errors[1], 1e-6)
        self.assertLess(errors[1], 1e-2 * errors[0])

    def test_iotas_derivative(self):
        """
        Taylor test for the derivative of the rotational transform of a ToruxSurface wrt coil parameters
        """
        bs, _, torux_surface = get_torux_surface(mpol=12, ntor=12, verify=False)
        torux_surface.run_code()
        io = Iotas(torux_surface)

        def f(dofs):
            bs.x = dofs
            return io.J()

        def df(dofs):
            bs.x = dofs
            return io.dJ()

        taylor_test1(f, df, bs.x, epsilons=np.power(2., -np.asarray(range(13, 16))))

    def test_major_radius_derivative(self):
        """
        Taylor test for the derivative of the major radius of a ToruxSurface wrt coil parameters
        """
        bs, _, torux_surface = get_torux_surface(mpol=12, ntor=12, verify=False)
        torux_surface.run_code()
        mr = MajorRadius(torux_surface)

        def f(dofs):
            bs.x = dofs
            return mr.J()

        def df(dofs):
            bs.x = dofs
            return mr.dJ()

        taylor_test1(f, df, bs.x, epsilons=np.power(2., -np.asarray(range(13, 16))))

    def test_input_checks(self):
        """
        A ToruxSurface takes a stellarator symmetric SurfaceRZFourier and a ToroidalFlux label only.
        """
        bs = self.bs
        s = SurfaceRZFourier(nfp=3, mpol=2, ntor=2)
        with self.assertRaises(ValueError):
            ToruxSurface(bs, s, Volume(s), 1.)
        s = SurfaceRZFourier(nfp=3, stellsym=False, mpol=2, ntor=2)
        with self.assertRaises(ValueError):
            ToruxSurface(bs, s, ToroidalFlux(s, bs), 1.)


if __name__ == "__main__":
    unittest.main()
