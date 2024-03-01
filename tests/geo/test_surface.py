import unittest
import json
from pathlib import Path
import os
import logging
import numpy as np

from monty.json import MontyDecoder, MontyEncoder

from simsopt.geo.surface import Surface
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.surfacehenneberg import SurfaceHenneberg
from simsopt.geo.surfacegarabedian import SurfaceGarabedian
from simsopt.geo.surface import signed_distance_from_surface, SurfaceScaled, \
    best_nphi_over_ntheta
from simsopt.geo.curverzfourier import CurveRZFourier
from .surface_test_helpers import get_surface

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False

surface_types = ["SurfaceRZFourier", "SurfaceXYZFourier", "SurfaceXYZTensorFourier",
                 "SurfaceHenneberg", "SurfaceGarabedian"]

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)


class QuadpointsTests(unittest.TestCase):
    def test_theta(self):
        """
        Check that the different options for initializing the theta
        quadrature points behave as expected.
        """
        for surface_type in surface_types:
            # Try specifying no arguments for theta:
            s = eval(surface_type + "()")
            np.testing.assert_allclose(s.quadpoints_theta,
                                       np.linspace(0.0, 1.0, 62, endpoint=False))

            # Try specifying ntheta:
            s = eval(surface_type + ".from_nphi_ntheta(ntheta=17)")
            np.testing.assert_allclose(s.quadpoints_theta,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))

            # Try specifying quadpoints_theta as a numpy array:
            s = eval(surface_type + "(quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False))")
            np.testing.assert_allclose(s.quadpoints_theta,
                                       np.linspace(0.0, 1.0, 5, endpoint=False))

            # Try specifying quadpoints_theta as a list:
            s = eval(surface_type + "(quadpoints_theta=[0.2, 0.7, 0.3])")
            np.testing.assert_allclose(s.quadpoints_theta, [0.2, 0.7, 0.3])

    def test_phi(self):
        """
        Check that the different options for initializing the phi
        quadrature points behave as expected.
        """
        for surface_type in surface_types:
            # Try specifying no arguments for phi:
            s = eval(surface_type + "()")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 61, endpoint=False))

            # Try specifying nphi but not range:
            s = eval(surface_type + ".from_nphi_ntheta(nphi=17)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))

            # Try specifying nphi plus range as a string, without nfp:
            s = eval(surface_type + ".from_nphi_ntheta(nphi=17, range='full torus')")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + ".from_nphi_ntheta(nphi=17, range='field period')")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + ".from_nphi_ntheta(nphi=17, range='half period')")
            grid = np.linspace(0.0, 0.5, 17, endpoint=False)
            grid += 0.5 * (grid[1] - grid[0])
            np.testing.assert_allclose(s.quadpoints_phi, grid)

            # Try specifying nphi plus range as a string, with nfp:
            s = eval(surface_type + ".from_nphi_ntheta(nphi=17, range='full torus', nfp=3)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + ".from_nphi_ntheta(nphi=17, range='field period', nfp=3)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0 / 3.0, 17, endpoint=False))
            s = eval(surface_type + ".from_nphi_ntheta(nphi=17, range='half period', nfp=3)")
            grid = np.linspace(0.0, 0.5 / 3.0, 17, endpoint=False)
            grid += 0.5 * (grid[1] - grid[0])
            np.testing.assert_allclose(s.quadpoints_phi, grid)

            # Try specifying nphi plus range as a constant, with nfp:
            s = eval(surface_type + ".from_nphi_ntheta(nfp=4, nphi=17, range=" + surface_type + ".RANGE_FULL_TORUS)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + ".from_nphi_ntheta(nfp=4, nphi=17, range=" + surface_type + ".RANGE_FIELD_PERIOD)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0 / 4.0, 17, endpoint=False))
            s = eval(surface_type + ".from_nphi_ntheta(nfp=4, nphi=17, range=" + surface_type + ".RANGE_HALF_PERIOD)")
            grid = np.linspace(0.0, 0.5 / 4.0, 17, endpoint=False)
            grid += 0.5 * (grid[1] - grid[0])
            np.testing.assert_allclose(s.quadpoints_phi, grid)

            # Try specifying quadpoints_phi as a numpy array:
            s = eval(surface_type + "(quadpoints_phi=np.linspace(0.0, 1.0, 5, endpoint=False))")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0, 1.0, 5, endpoint=False))

            # Try specifying quadpoints_phi as a list:
            s = eval(surface_type + "(quadpoints_phi=[0.2, 0.7, 0.3])")
            np.testing.assert_allclose(s.quadpoints_phi, [0.2, 0.7, 0.3])

            # Specifying nphi in init directly should cause an error:
            with self.assertRaises(Exception):
                s = eval(surface_type + "(nphi=5, quadpoints_phi=np.linspace(0.0, 1.0, 5, endpoint=False))")

    def test_spectral(self):
        """
        Verify integration is accurate to around machine precision for the
        predefined phi grid ranges.
        """
        ntheta = 64
        nfp = 4
        area_ref = 74.492696353899
        volume_ref = 11.8435252813064
        for range_str, nphi_fac in [("full torus", 1), ("field period", 1.0 / nfp), ("half period", 0.5 / nfp)]:
            for nphi_base in [200, 400, 800]:
                nphi = int(nphi_fac * nphi_base)
                s = SurfaceRZFourier.from_nphi_ntheta(range=range_str, nfp=nfp,
                                                      mpol=1, ntor=1, ntheta=ntheta, nphi=nphi)
                s.set_rc(0, 0, 2.5)
                s.set_rc(1, 0, 0.4)
                s.set_zs(1, 0, 0.6)
                s.set_rc(0, 1, 1.1)
                s.set_zs(0, 1, 0.8)
                logger.debug(f'range={range_str:13} n={nphi:5} ' \
                             f'area={s.area():22.14} diff={area_ref - s.area():22.14} ' \
                             f'volume={s.volume():22.15} diff={volume_ref - s.volume():22.15}')
                np.testing.assert_allclose(s.area(), area_ref, atol=0, rtol=1e-13)
                np.testing.assert_allclose(s.volume(), volume_ref, atol=0, rtol=1e-13)


class ArclengthTests(unittest.TestCase):
    def test_arclength_poloidal_angle(self):
        """
        Compute arclength poloidal angle from circular cross-section tokamak.
        Check that this matches parameterization angle.
        Check that arclength_poloidal_angle is in [0,1] for both circular
            cross-section tokamak and rotating ellipse boundary.
        """
        s = get_surface('SurfaceRZFourier', True, mpol=1, ntor=0,
                        ntheta=200, nphi=5, full=True)
        s.rc[0, 0] = 5.
        s.rc[1, 0] = 1.5
        s.zs[1, 0] = 1.5

        theta1D = s.quadpoints_theta

        arclength = s.arclength_poloidal_angle()
        nphi = len(arclength[:, 0])
        for iphi in range(nphi):
            np.testing.assert_allclose(arclength[iphi, :], theta1D, atol=1e-3)
            self.assertTrue(np.all(arclength[iphi, :] >= 0))
            self.assertTrue(np.all(arclength[iphi, :] <= 1))

        s = get_surface('SurfaceRZFourier', True, mpol=2, ntor=2,
                        ntheta=20, nphi=20, full=True)
        s.rc[0, 0] = 5.
        s.rc[1, 0] = -1.5
        s.rc[1, 1] = -0.5
        s.rc[0, 1] = -0.5
        s.zs[1, 1] = 0.5
        s.zs[1, 0] = -1.5
        s.zs[0, 1] = 0.5

        theta1D = s.quadpoints_theta

        arclength = s.arclength_poloidal_angle()
        nphi = len(arclength[:, 0])
        for iphi in range(nphi):
            self.assertTrue(np.all(arclength[iphi, :] >= 0))
            self.assertTrue(np.all(arclength[iphi, :] <= 1))

    def test_interpolate_on_arclength_grid(self):
        """
        Check that line integral of (1 + cos(theta - phi)) at constant phi is
        unchanged when evaluated on parameterization or arclength poloidal angle
        grid.
        """
        ntheta = 500
        nphi = 10
        s = get_surface('SurfaceRZFourier', True, mpol=5, ntor=5,
                        ntheta=ntheta, nphi=nphi, full=True)
        s.rc[0, 0] = 5.
        s.rc[1, 0] = -1.5
        s.zs[1, 0] = -1.5
        s.rc[1, 1] = -0.5
        s.zs[1, 1] = 0.5
        s.rc[0, 1] = -0.5
        s.zs[0, 1] = 0.5

        dgamma2 = s.gammadash2()
        theta1D = s.quadpoints_theta
        phi1D = s.quadpoints_phi
        theta, phi = np.meshgrid(theta1D, phi1D)
        integrand = 1 + np.cos(theta - phi)

        norm_drdtheta = np.linalg.norm(dgamma2, axis=2)
        theta_interp = theta
        integrand_arclength = s.interpolate_on_arclength_grid(integrand, theta_interp)
        for iphi in range(nphi):
            integral_1 = np.sum(integrand[iphi, :] * norm_drdtheta[iphi, :]) / np.sum(norm_drdtheta[iphi, :])
            integral_2 = np.sum(integrand_arclength[iphi, :]) / np.sum(np.ones_like(norm_drdtheta[iphi, :]))
            self.assertAlmostEqual(integral_1, integral_2, places=3)


class SurfaceDistanceTests(unittest.TestCase):
    def test_distance(self):
        c = CurveRZFourier(100, 1, 1, False)
        # dofs = c.get_dofs()
        # dofs[0] = 1.
        # c.set_dofs(dofs)
        # dofs = c.x
        # dofs[0] = 1.0
        c.set(0, 1.0)
        s = SurfaceRZFourier(mpol=1, ntor=1)
        s.fit_to_curve(c, 0.2, flip_theta=True)
        xyz = np.asarray([[0, 0, 0], [1., 0, 0], [2., 0., 0]])
        d = signed_distance_from_surface(xyz, s)
        assert np.allclose(d, [-0.8, 0.2, -0.8])
        s.fit_to_curve(c, 0.2, flip_theta=False)
        d = signed_distance_from_surface(xyz, s)
        assert np.allclose(d, [-0.8, 0.2, -0.8])


class SurfaceScaledTests(unittest.TestCase):
    def test_surface_scaled(self):
        mpol = 3
        ntor = 2
        nfp = 4
        surf1 = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=nfp)
        ndofs = surf1.dof_size
        surf1.x = np.random.rand(ndofs)

        scale_factors = 0.1 ** np.sqrt(surf1.m ** 2 + surf1.n ** 2)
        surf_scaled = SurfaceScaled(surf1, scale_factors)

        np.testing.assert_allclose(surf1.x, surf_scaled.x * scale_factors)

        surf_scaled.x = np.random.rand(ndofs)
        np.testing.assert_allclose(surf1.x, surf_scaled.x * scale_factors)

        self.assertEqual(surf_scaled.to_RZFourier(), surf1)

    def test_names(self):
        """
        The dof names should be the same for the SurfaceScaled as for the
        original surface.
        """
        surf1 = SurfaceRZFourier(mpol=2, ntor=3, nfp=2)
        scale_factors = np.random.rand(len(surf1.x))
        surf_scaled = SurfaceScaled(surf1, scale_factors)
        self.assertEqual(surf1.local_full_dof_names, surf_scaled.local_full_dof_names)

    def test_fixed(self):
        """
        Verify that the fixed/free property for a SurfaceScaled can be
        matched to the original surface.
        """
        surf1 = SurfaceRZFourier(mpol=2, ntor=3, nfp=2)
        scale_factors = np.random.rand(len(surf1.x))
        surf_scaled = SurfaceScaled(surf1, scale_factors)
        surf1.local_fix_all()
        surf1.fixed_range(mmin=0, mmax=1,
                          nmin=-2, nmax=3, fixed=False)
        surf1.fix("rc(0,0)")  # Major radius
        surf_scaled.update_fixed()
        np.testing.assert_array_equal(surf1.dofs_free_status, surf_scaled.dofs_free_status)

    def test_serialization(self):
        surfacetypes = ["SurfaceRZFourier", "SurfaceXYZFourier",
                        "SurfaceXYZTensorFourier"]
        for surfacetype in surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym, full=True)
                    dof_size = len(s.x)
                    scale_factors = np.random.random_sample(dof_size)
                    scaled_s = SurfaceScaled(s, scale_factors)
                    scaled_s_str = json.dumps(scaled_s, cls=MontyEncoder)
                    regen_s = json.loads(scaled_s_str, cls=MontyDecoder)


class BestNphiOverNthetaTests(unittest.TestCase):
    def test_axisymm(self):
        """
        For an axisymmetric circular-cross-section torus at high aspect
        ratio, the ideal nphi/ntheta ratio should match the aspect
        ratio.
        """
        surf = SurfaceRZFourier(nfp=2, mpol=1, ntor=0)
        aspect = 150
        surf.set_rc(0, 0, aspect)
        surf.set_rc(1, 0, 1.0)
        surf.set_zs(1, 0, 1.0)
        np.testing.assert_allclose(best_nphi_over_ntheta(surf), aspect, rtol=3e-5)

    def test_independent_of_quadpoints(self):
        """
        Evaluate the ideal ratio of nphi / ntheta for several surfaces,
        and confirm that the results match reference values. This test
        is repeated for all 3 'range' options, and for several ntheta
        and nphi values.
        """
        data = (('wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc', 8.58),
                ('wout_li383_low_res_reference.nc', 5.28))
        for filename_base, correct in data:
            filename = os.path.join(TEST_DIR, filename_base)
            for ntheta in [30, 31, 60, 61]:
                for phi_range in ['full torus', 'field period', 'half period']:
                    if phi_range == 'full torus':
                        nphis = [200, 300]
                    elif phi_range == 'field period':
                        nphis = [40, 61]
                    else:
                        nphis = [25, 44]
                    for nphi in nphis:
                        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
                            range=phi_range, nphi=nphi, ntheta=ntheta)
                        surf = SurfaceRZFourier.from_wout(
                            filename, quadpoints_theta=quadpoints_theta,
                            quadpoints_phi=quadpoints_phi)
                        ratio = best_nphi_over_ntheta(surf)
                        logger.info(f'range: {phi_range}, nphi: {nphi}, ntheta: {ntheta}, best nphi / ntheta: {ratio}')
                        np.testing.assert_allclose(ratio, correct, rtol=0.01)


class CurvatureTests(unittest.TestCase):
    surfacetypes = ["SurfaceRZFourier", "SurfaceXYZFourier",
                    "SurfaceXYZTensorFourier"]

    def test_gauss_bonnet(self):
        """
        Tests the Gauss-Bonnet theorem for a toroidal surface, :math:`S`:

        .. math::
            \int_{S} d^2 x \, K = 0,

        where :math:`K` is the Gaussian curvature.
        """
        for surfacetype in self.surfacetypes:
            for stellsym in [True, False]:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    s = get_surface(surfacetype, stellsym, full=True)
                    K = s.surface_curvatures()[:, :, 1]
                    N = np.sqrt(np.sum(s.normal()**2, axis=2))
                    assert np.abs(np.sum(K*N)) < 1e-12


if __name__ == "__main__":
    unittest.main()
