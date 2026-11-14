import unittest

import numpy as np

from simsopt.configs import get_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, ToroidalFlux, BoozerSurface, Iotas, MajorRadius, \
    RotatedCurve, PestSurface
from simsopt.geo.pestsurface import pest_surface_residual
from .surface_test_helpers import get_boozer_surface
from .test_surface_objectives import taylor_test1


def get_pest_surface(mpol, ntor, stellsym=True):
    """
    A PestSurface of resolution ``mpol``, ``ntor`` on SIMSOPT's NCSX test coils, at the toroidal flux of
    :obj:`get_boozer_surface`, solved. Newton's method starts from a circle of radius 0.1 m about the magnetic
    axis at mpol = ntor = 6, and that solution is the initial guess at ``mpol``, ``ntor``. With
    ``stellsym=False``, the coils lose their stellarator symmetry, keeping the three field periods: each flipped
    coil carries 3% more current than its base coil, and the surface has no stellarator symmetry.
    """
    bs, boozer_surface = get_boozer_surface(label="ToroidalFlux")
    base_curves, base_currents, ma, nfp, _ = get_data("ncsx")
    if not stellsym:
        curves_flipped = [RotatedCurve(c, 0, True) for c in base_curves]
        currents_flipped = [-1.03 * c for c in base_currents]
        bs = BiotSavart(coils_via_symmetries(base_curves + curves_flipped, base_currents + currents_flipped, nfp,
                                             False))
    s = SurfaceRZFourier(nfp=nfp, stellsym=stellsym, mpol=6, ntor=6)
    s.fit_to_curve(ma, 0.1, flip_theta=True)
    iota = -0.406
    for M, N in [(6, 6), (mpol, ntor)]:
        s = s.copy(mpol=M, ntor=N, quadpoints_phi=np.linspace(0, 1/nfp, 2*N+1, endpoint=False),
                   quadpoints_theta=np.linspace(0, 1, 2*M+1, endpoint=False))
        # the toroidal flux through the cross section phi = 0 alone
        label = ToroidalFlux(s, BiotSavart(bs.coils), range="field period", nphi=1, ntheta=4*M)
        pest_surface = PestSurface(bs, s, label, boozer_surface.targetlabel, options={'verbose': False})
        res = pest_surface.run_code(iota)
        assert res['success']
        iota = res['iota']
    return bs, boozer_surface, pest_surface


def max_BdotN(surface, biotsavart):
    """
    The largest |B.n|/(|B| |n|) on ``surface``, on a grid of one field period with twice the points of the
    collocation grid, shifted from it by half a spacing.
    """
    nphi, ntheta = 4*surface.ntor + 2, 4*surface.mpol + 2
    s = surface.copy(quadpoints_phi=(np.arange(nphi) + 0.5) / (nphi * surface.nfp),
                     quadpoints_theta=(np.arange(ntheta) + 0.5) / ntheta)
    biotsavart.set_points(s.gamma().reshape((-1, 3)))
    B = biotsavart.B().reshape(s.gamma().shape)
    n = s.normal()
    return np.max(np.abs(np.sum(B * n, axis=2)) / (np.linalg.norm(B, axis=2) * np.linalg.norm(n, axis=2)))


def taylor_test_coils(bs, pest_surface, objective):
    """
    Taylor test for the derivative of ``objective`` of ``pest_surface`` with respect to the coil dofs of ``bs``.
    """
    def f(dofs):
        bs.x = dofs
        return objective.J()

    def df(dofs):
        bs.x = dofs
        return objective.dJ()

    taylor_test1(f, df, bs.x, epsilons=np.power(2., -np.asarray(range(13, 16))))


class PestSurfaceTests(unittest.TestCase):

    def test_residual_jacobian(self):
        """
        Taylor test for the Jacobian of pest_surface_residual with respect to the surface dofs and iota, with and
        without stellarator symmetry.
        """
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        for stellsym in [True, False]:
            s = SurfaceRZFourier(nfp=nfp, stellsym=stellsym, mpol=4, ntor=4)
            s.fit_to_curve(ma, 0.1, flip_theta=True)
            x0 = np.concatenate((s.get_dofs(), [-0.4]))

            def f(x):
                s.set_dofs(x[:-1])
                return pest_surface_residual(s, x[-1], bs)[0]

            def df(x):
                s.set_dofs(x[:-1])
                return pest_surface_residual(s, x[-1], bs)[1]

            taylor_test1(f, df, x0, epsilons=np.power(2., -np.asarray(range(8, 14))))

    def test_flux_surface(self):
        """
        Newton's method finds a flux surface of the coils with the requested toroidal flux, with and without
        stellarator symmetry: the field crosses it by less than 1e-8 of its strength at mpol, ntor = 16, 24.
        Without stellarator symmetry, Z at theta = phi = 0 is the mean of Z over theta at phi = 0.
        """
        for stellsym in [True, False]:
            bs, _, pest_surface = get_pest_surface(16, 24, stellsym=stellsym)
            s = pest_surface.surface
            self.assertLess(max_BdotN(s, bs), 1e-8)
            np.testing.assert_allclose(pest_surface.label.J(), pest_surface.targetlabel, rtol=1e-12)
            if not stellsym:
                self.assertGreater(min(np.max(np.abs(s.rs)), np.max(np.abs(s.zc))), 1e-4)
                self.assertAlmostEqual(np.sum(s.zc[1:]), 0., delta=1e-14)

    def test_iota_against_boozer_exact(self):
        """
        The rotational transform agrees with that of a BoozerExact surface on the same coils and toroidal flux at
        mpol, ntor = 14, 21.
        """
        bs, boozer_surface, pest_surface = get_pest_surface(16, 24)
        s_prev, res = boozer_surface.surface, boozer_surface.res
        for mpol, ntor in [(10, 10), (14, 21)]:
            phis = np.linspace(0, 1/3, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=True, nfp=3, quadpoints_phi=phis,
                                        quadpoints_theta=thetas)
            s_fit = SurfaceXYZTensorFourier(mpol=s_prev.mpol, ntor=s_prev.ntor, stellsym=True, nfp=3,
                                            quadpoints_phi=phis, quadpoints_theta=thetas, dofs=s_prev.dofs)
            s.least_squares_fit(s_fit.gamma())
            booz = BoozerSurface(bs, s, ToroidalFlux(s, BiotSavart(bs.coils)), boozer_surface.targetlabel,
                                 options={'newton_tol': 1e-12, 'verbose': False})
            res = booz.run_code(res['iota'], G=res['G'])
            self.assertTrue(res['success'])
            s_prev = s
        self.assertAlmostEqual(res['iota'], pest_surface.res['iota'], delta=1e-11)

    def test_stellarator_symmetric_coils(self):
        """
        A surface without stellarator symmetry on the stellarator symmetric coils, from a guess that is not
        symmetric, is the stellarator symmetric surface, with the same rotational transform, and rs and zc zero.
        """
        bs, _, symmetric = get_pest_surface(12, 12)
        s = symmetric.surface.copy(stellsym=False)
        s.set_rs(1, 0, 1e-3)
        s.set_zc(1, 1, 1e-3)
        label = ToroidalFlux(s, BiotSavart(bs.coils), range="field period", nphi=1, ntheta=48)
        res = PestSurface(bs, s, label, symmetric.targetlabel, options={'verbose': False}).run_code(-0.406)
        self.assertTrue(res['success'])
        self.assertAlmostEqual(res['iota'], symmetric.res['iota'], delta=1e-13)
        np.testing.assert_allclose(s.rc, symmetric.surface.rc, atol=1e-13)
        np.testing.assert_allclose(s.zs, symmetric.surface.zs, atol=1e-13)
        self.assertLess(max(np.max(np.abs(s.rs)), np.max(np.abs(s.zc))), 1e-13)

    def test_warm_start(self):
        """
        After a small change of the coils, Newton's method from the last surface converges in a few iterations, as
        in an optimization loop.
        """
        bs, _, pest_surface = get_pest_surface(12, 12)
        iota = pest_surface.res['iota']
        np.random.seed(1)
        bs.x = bs.x * (1 + 1e-3 * (np.random.rand(bs.x.size) - 0.5))
        self.assertTrue(pest_surface.need_to_run_code)
        res = pest_surface.run_code(iota)
        self.assertTrue(res['success'])
        self.assertLessEqual(res['iter'], 3)
        self.assertIs(pest_surface.run_code(iota), res)

    def test_derivatives(self):
        """
        Taylor tests for the derivatives of Iotas and MajorRadius with respect to the coils, with and without
        stellarator symmetry.
        """
        for stellsym in [True, False]:
            for objective in [Iotas, MajorRadius]:
                bs, _, pest_surface = get_pest_surface(12, 12, stellsym=stellsym)
                taylor_test_coils(bs, pest_surface, objective(pest_surface))

    def test_input_check(self):
        """
        A PestSurface takes a SurfaceRZFourier.
        """
        bs, boozer_surface = get_boozer_surface(label="ToroidalFlux", converge=False)
        s = boozer_surface.surface
        with self.assertRaises(ValueError):
            PestSurface(bs, s, boozer_surface.label, 1.)


if __name__ == "__main__":
    unittest.main()
