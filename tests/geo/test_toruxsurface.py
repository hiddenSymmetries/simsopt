import unittest

import numpy as np

from simsopt.configs import get_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, ToroidalFlux, Volume, BoozerSurface, Iotas, \
    MajorRadius, RotatedCurve
from simsopt.geo.toruxsurface import ToruxSurface, torux_available
from .surface_test_helpers import get_boozer_surface
from .test_surface_objectives import taylor_test1


def get_torux_surface(mpol, ntor, verify=True, stellsym=True):
    """
    The flux surface of SIMSOPT's NCSX test coils through which the toroidal flux is that of the
    BoozerExact surface of :obj:`get_boozer_surface`, as a ToruxSurface of resolution ``mpol``, ``ntor``.
    With ``stellsym=False``, the coils lose their stellarator symmetry, keeping the three field
    periods: each flipped coil carries 3% more current than its base coil, the toroidal flux is a
    quarter of that of :obj:`get_boozer_surface` (half the minor radius, which torux solves in about
    half the time), and the surface is a SurfaceRZFourier without stellarator symmetry.
    """
    bs, boozer_surface = get_boozer_surface(label="ToroidalFlux")
    targetlabel = boozer_surface.targetlabel
    if not stellsym:
        base_curves, base_currents, _, nfp, _ = get_data("ncsx")
        curves_flipped = [RotatedCurve(c, 0, True) for c in base_curves]
        currents_flipped = [-1.03 * c for c in base_currents]
        bs = BiotSavart(coils_via_symmetries(base_curves + curves_flipped, base_currents + currents_flipped, nfp,
                                             False))
        targetlabel = 0.25 * targetlabel
    s = SurfaceRZFourier(nfp=3, stellsym=stellsym, mpol=mpol, ntor=ntor,
                         quadpoints_phi=np.linspace(0, 1/3, 64, endpoint=False),
                         quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    torux_surface = ToruxSurface(bs, s, ToroidalFlux(s, BiotSavart(bs.coils)), targetlabel,
                                 options={'verify': verify})
    return bs, boozer_surface, torux_surface


class SharedTaylorPoints:
    """
    Iotas and MajorRadius of the ToruxSurface of :obj:`get_torux_surface` at resolution 12, 12, at the
    coil dofs a Taylor test visits. Every run_code solves with torux from scratch, and the Taylor tests
    of the two objectives visit the same coil dofs (:obj:`taylor_test1` seeds its direction), so each
    set of coil dofs is solved once for both.
    """

    def __init__(self, stellsym):
        self.bs, _, torux_surface = get_torux_surface(mpol=12, ntor=12, verify=False, stellsym=stellsym)
        torux_surface.run_code()
        self.objectives = [Iotas(torux_surface), MajorRadius(torux_surface)]
        self.x = self.bs.x
        self.J = {self.x.tobytes(): [obj.J() for obj in self.objectives]}
        self.dJ = [obj.dJ() for obj in self.objectives]

    def taylor_test(self, k):
        """
        Taylor test for the derivative of objective ``k`` (0: Iotas, 1: MajorRadius) wrt coil parameters
        """
        def f(dofs):
            if dofs.tobytes() not in self.J:
                self.bs.x = dofs
                self.J[dofs.tobytes()] = [obj.J() for obj in self.objectives]
            return self.J[dofs.tobytes()][k]

        def df(dofs):
            assert np.array_equal(dofs, self.x)
            return self.dJ[k]

        taylor_test1(f, df, self.x, epsilons=np.power(2., -np.asarray(range(13, 16))))


@unittest.skipIf(not torux_available, "torux not found (it needs Python 3.12 or newer)")
class ToruxSurfaceTests(unittest.TestCase):
    shared_taylor_points = None

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

    @classmethod
    def taylor_points(cls):
        if cls.shared_taylor_points is None:
            cls.shared_taylor_points = SharedTaylorPoints(stellsym=True)
        return cls.shared_taylor_points

    def test_iotas_derivative(self):
        """
        Taylor test for the derivative of the rotational transform of a ToruxSurface wrt coil parameters
        """
        self.taylor_points().taylor_test(0)

    def test_major_radius_derivative(self):
        """
        Taylor test for the derivative of the major radius of a ToruxSurface wrt coil parameters
        """
        self.taylor_points().taylor_test(1)

    def test_input_checks(self):
        """
        A ToruxSurface takes a SurfaceRZFourier, with or without stellarator symmetry, and a ToroidalFlux
        label only.
        """
        bs = self.bs
        s = SurfaceRZFourier(nfp=3, mpol=2, ntor=2)
        with self.assertRaises(ValueError):
            ToruxSurface(bs, s, Volume(s), 1.)
        s = SurfaceXYZTensorFourier(nfp=3, mpol=2, ntor=2)
        with self.assertRaises(ValueError):
            ToruxSurface(bs, s, ToroidalFlux(s, bs), 1.)
        s = SurfaceRZFourier(nfp=3, stellsym=False, mpol=2, ntor=2)
        ToruxSurface(bs, s, ToroidalFlux(s, bs), 1.)


@unittest.skipIf(not torux_available, "torux not found (it needs Python 3.12 or newer)")
class ToruxSurfaceWithoutStellsymTests(unittest.TestCase):
    """
    The same on coils without stellarator symmetry (:obj:`get_torux_surface` with ``stellsym=False``),
    with a SurfaceRZFourier without stellarator symmetry.
    """
    shared_taylor_points = None

    @classmethod
    def setUpClass(cls):
        import torux
        if not hasattr(torux.Answer, "system_general"):
            raise unittest.SkipTest("torux solves surfaces without stellarator symmetry from version 0.1.3")
        # torux's own resolution is mpol, ntor = 20, 48: at 20, 36 the surface is already at the leak of the
        # stellarator symmetric test, with about half the memory for its dense LU factors
        cls.bs, cls.boozer_surface, cls.torux_surface = get_torux_surface(mpol=20, ntor=36, stellsym=False)
        cls.res = cls.torux_surface.run_code()

    def test_flux_surface(self):
        """
        The surface torux finds is a flux surface of the coils without stellarator symmetry, with the
        requested toroidal flux.
        """
        res, s = self.res, self.torux_surface.surface
        self.assertTrue(res['success'])
        self.assertTrue(hasattr(res['answer'].surface, 'rs'))      # torux's own surface has no symmetry either
        self.assertGreater(min(np.max(np.abs(s.rs)), np.max(np.abs(s.zc))), 1e-4)

        self.bs.set_points(s.gamma().reshape((-1, 3)))
        B = self.bs.B().reshape(s.gamma().shape)
        n = s.normal()
        leak = np.abs(np.sum(B * n, axis=2)) / (np.linalg.norm(B, axis=2) * np.linalg.norm(n, axis=2))
        self.assertLess(np.max(leak), 1e-12)

        flux = self.torux_surface.label.J()
        np.testing.assert_allclose(abs(flux), abs(self.torux_surface.targetlabel), rtol=1e-12)

    def test_stellarator_symmetric_coils(self):
        """
        A SurfaceRZFourier without stellarator symmetry on the stellarator symmetric coils: the surface
        and the rotational transform of the stellarator symmetric ToruxSurface, with rs and zc zero.
        """
        bs, _, symmetric = get_torux_surface(mpol=12, ntor=12, verify=False)
        s = SurfaceRZFourier(nfp=3, stellsym=False, mpol=12, ntor=12,
                             quadpoints_phi=np.linspace(0, 1/3, 64, endpoint=False),
                             quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
        torux_surface = ToruxSurface(bs, s, ToroidalFlux(s, BiotSavart(bs.coils)), symmetric.targetlabel,
                                     options={'verify': False})
        res, res_symmetric = torux_surface.run_code(), symmetric.run_code()
        self.assertTrue(res['success'] and res_symmetric['success'])
        self.assertAlmostEqual(res['iota'], res_symmetric['iota'], delta=1e-13)
        np.testing.assert_allclose(s.rc, symmetric.surface.rc, atol=1e-13)
        np.testing.assert_allclose(s.zs, symmetric.surface.zs, atol=1e-13)
        self.assertLess(max(np.max(np.abs(s.rs)), np.max(np.abs(s.zc))), 1e-13)

    def test_iota_against_boozer_exact(self):
        """
        BoozerExact's rotational transform on the same coils and toroidal flux, without stellarator
        symmetry, approaches torux's as its resolution grows. It starts from a circle of radius 0.05
        about the magnetic axis, as :obj:`get_boozer_surface` starts from one of radius 0.1.
        """
        iota = np.sign(self.torux_surface.surface.volume()) * self.res['iota']
        errors = []
        ma = get_data("ncsx")[2]
        s_prev, res = None, self.boozer_surface.res
        for k in [6, 10]:
            phis = np.linspace(0, 1/3, 2*k+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*k+1, endpoint=False)
            s = SurfaceXYZTensorFourier(mpol=k, ntor=k, stellsym=False, nfp=3, quadpoints_phi=phis,
                                        quadpoints_theta=thetas)
            if s_prev is None:
                s.fit_to_curve(ma, 0.05, flip_theta=True)
            else:
                s_fit = SurfaceXYZTensorFourier(mpol=s_prev.mpol, ntor=s_prev.ntor, stellsym=False, nfp=3,
                                                quadpoints_phi=phis, quadpoints_theta=thetas, dofs=s_prev.dofs)
                s.least_squares_fit(s_fit.gamma())
            boozer_surface = BoozerSurface(self.bs, s, ToroidalFlux(s, BiotSavart(self.bs.coils)),
                                           self.torux_surface.targetlabel, options={'newton_tol': 1e-12})
            res = boozer_surface.run_code(res['iota'], G=res['G'])
            self.assertTrue(res['success'])
            errors.append(abs(np.sign(s.volume()) * res['iota'] - iota))
            s_prev = s
        self.assertLess(errors[-1], 1e-6)
        self.assertLess(errors[-1], 1e-2 * errors[0])

    @classmethod
    def taylor_points(cls):
        if cls.shared_taylor_points is None:
            cls.shared_taylor_points = SharedTaylorPoints(stellsym=False)
        return cls.shared_taylor_points

    def test_iotas_derivative(self):
        """
        Taylor test for the derivative of the rotational transform of a ToruxSurface without stellarator
        symmetry wrt coil parameters
        """
        self.taylor_points().taylor_test(0)

    def test_major_radius_derivative(self):
        """
        Taylor test for the derivative of the major radius of a ToruxSurface without stellarator
        symmetry wrt coil parameters
        """
        self.taylor_points().taylor_test(1)


if __name__ == "__main__":
    unittest.main()
