import unittest
import json
from pathlib import Path
import os
import logging
import numpy as np
from scipy import interpolate

from simsopt.geo.surface import Surface
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.surfacehenneberg import SurfaceHenneberg
from simsopt.geo.surfacegarabedian import SurfaceGarabedian
from simsopt.geo.surface import signed_distance_from_surface, SurfaceScaled, \
    best_nphi_over_ntheta
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
from .surface_test_helpers import get_surface, get_boozer_surface
from simsopt._core import load

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

stellsym_list = [True, False]

surface_types = ["SurfaceRZFourier", "SurfaceXYZFourier", "SurfaceXYZTensorFourier",
                 "SurfaceHenneberg", "SurfaceGarabedian"]

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

try:
    import ground
except:
    ground = None

try:
    import bentley_ottmann
except:
    bentley_ottmann = None


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
                logger.debug(f'range={range_str:13} n={nphi:5} '
                             f'area={s.area():22.14} diff={area_ref - s.area():22.14} '
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
        s.set_rc(0, 0, 5.)
        s.set_rc(1, 0, 1.5)
        s.set_zs(1, 0, 1.5)

        theta1D = s.quadpoints_theta

        arclength = s.arclength_poloidal_angle()
        nphi = len(arclength[:, 0])
        for iphi in range(nphi):
            np.testing.assert_allclose(arclength[iphi, :], theta1D, atol=1e-14, rtol=1e-14)
            self.assertTrue(np.all(arclength[iphi, :] >= 0))
            self.assertTrue(np.all(arclength[iphi, :] < 1))

        s = get_surface('SurfaceRZFourier', True, mpol=2, ntor=2,
                        ntheta=20, nphi=20, full=True)
        s.set_rc(0, 0, 5.)
        s.set_rc(1, 0, 1.5)
        s.set_zs(1, 0, 1.5)
        s.set_rc(1, 1, -0.5)
        s.set_zs(1, 1, 0.5)
        s.set_rc(0, 1, -0.5)
        s.set_zs(0, 1, 0.5)

        theta1D = s.quadpoints_theta

        arclength = s.arclength_poloidal_angle()
        nphi = len(arclength[:, 0])
        for iphi in range(nphi):
            self.assertTrue(np.all(arclength[iphi, :] >= 0))
            self.assertTrue(np.all(arclength[iphi, :] < 1))

    def test_arclength_poloidal_angle_2_ways(self):
        """Compare with an alternative way to code up the same method."""
        filename_base = "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        nphi = 10
        ntheta = 13
        surf = SurfaceRZFourier.from_wout(filename, range="half period", nphi=nphi, ntheta=ntheta)
        gamma = surf.gamma()
        X = gamma[:, :, 0]
        Y = gamma[:, :, 1]
        Z = gamma[:, :, 2]
        R = np.sqrt(X ** 2 + Y ** 2)

        theta_arclength = np.zeros_like(gamma[:, :, 0])
        nphi = len(theta_arclength[:, 0])
        ntheta = len(theta_arclength[0, :])
        for iphi in range(nphi):
            for itheta in range(1, ntheta):
                dr = np.sqrt((R[iphi, itheta] - R[iphi, itheta - 1]) ** 2
                             + (Z[iphi, itheta] - Z[iphi, itheta - 1]) ** 2)
                theta_arclength[iphi, itheta] = \
                    theta_arclength[iphi, itheta - 1] + dr
            dr = np.sqrt((R[iphi, 0] - R[iphi, -1]) ** 2
                         + (Z[iphi, 0] - Z[iphi, -1]) ** 2)
            L = theta_arclength[iphi, -1] + dr
            theta_arclength[iphi, :] = theta_arclength[iphi, :] / L

        theta_arclength_alt = surf.arclength_poloidal_angle()
        np.testing.assert_allclose(theta_arclength, theta_arclength_alt, rtol=1e-14)

    def test_interpolate_on_arclength_grid(self):
        """
        Check that line integral of an arbitrary function at constant phi is
        unchanged when evaluated on parameterization or arclength poloidal angle
        grid.
        """
        ntheta = 100
        nphi = 10
        s = get_surface('SurfaceRZFourier', True, mpol=5, ntor=5,
                        ntheta=ntheta, nphi=nphi, full=True)
        s.set_rc(0, 0, 5.)
        s.set_rc(1, 0, 1.5)
        s.set_zs(1, 0, 1.4)
        s.set_rc(1, 1, -0.6)
        s.set_zs(1, 1, 0.6)
        s.set_rc(0, 1, -0.5)
        s.set_zs(0, 1, 0.5)

        dgamma2 = s.gammadash2()
        theta1D = s.quadpoints_theta
        phi1D = s.quadpoints_phi
        theta, phi = np.meshgrid(theta1D, phi1D)
        integrand = np.log(1.2 + np.cos(2 * np.pi * (theta - phi) + 0.3))

        theta_interp = theta
        integrand_arclength = s.interpolate_on_arclength_grid(integrand, theta_interp)

        norm_drdtheta = np.linalg.norm(dgamma2, axis=2)
        integral_1 = np.sum(integrand * norm_drdtheta, axis=1) / np.sum(norm_drdtheta, axis=1)
        integral_2 = np.sum(integrand_arclength, axis=1) / np.sum(np.ones_like(norm_drdtheta), axis=1)
        np.testing.assert_allclose(integral_1, integral_2, rtol=9e-4)

    def test_interpolate_on_arclength_grid_2_ways(self):
        """
        Try doing periodic interpolation a different way (involving ghost
        points) and make sure the result is the same.
        """
        nphi = 4
        ntheta = 40
        ntheta_eval = 13
        filename_base = "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        surf = SurfaceRZFourier.from_wout(filename, range="half period", nphi=nphi, ntheta=ntheta)

        function = 1.2 * (0.3 + surf.quadpoints_phi[:, None] + np.cos(surf.quadpoints_theta[None, :] * 2 * np.pi))
        assert function.shape == (nphi, ntheta)
        theta_evaluate = np.linspace(0, 1, ntheta_eval)[None, :] * np.linspace(0.3, 1, nphi)[:, None]
        assert np.min(theta_evaluate) >= 0
        assert np.max(theta_evaluate) <= 1
        assert theta_evaluate.shape == (nphi, ntheta_eval)

        function_interpolated1 = surf.interpolate_on_arclength_grid(function, theta_evaluate)

        n_ghost = 10  # Number of points to repeat at each end, to ensure periodicity
        theta_arclength = surf.arclength_poloidal_angle()
        theta_arclength_big = np.concatenate(
            (
                theta_arclength[:, -n_ghost:] - 1, 
                theta_arclength, 
                theta_arclength[:, :n_ghost] + 1,
            ),
            axis=1,
        )
        function_big = np.concatenate(
            (
                function[:, -n_ghost:], 
                function, 
                function[:, :n_ghost],  
            ),
            axis=1,
        )
        function_interpolated2 = np.zeros((nphi, ntheta_eval))  
        nphi = len(theta_arclength[:, 0])
        for iphi in range(nphi):
            interpolator = interpolate.InterpolatedUnivariateSpline(
                theta_arclength_big[iphi, :], function_big[iphi, :])
            function_interpolated2[iphi, :] = interpolator(theta_evaluate[iphi, :])

        np.testing.assert_allclose(function_interpolated1, function_interpolated2, rtol=2e-9)

    def test_make_theta_uniform_arclength(self):
        wout_filename = TEST_DIR / 'wout_LandremanSenguptaPlunk_section5p3_reference.nc'
        plasma_surf = SurfaceRZFourier.from_wout(wout_filename, range="half period", ntheta=51, nphi=50)

        def get_specs():
            return plasma_surf.volume(), plasma_surf.area(), plasma_surf.major_radius(), plasma_surf.aspect_ratio()

        specs1 = get_specs()

        # Ensure Fourier resolution is high enough for the new theta coordinate
        # to represent the shape accurately:
        plasma_surf.change_resolution(12, 12)
        plasma_surf.make_theta_uniform_arclength()

        specs2 = get_specs()
        np.testing.assert_allclose(specs1, specs2, rtol=1e-3)

        # Compare to reference calculation using regcoil to represent the same surface
        # using the uniform-arclength theta:
        regcoil_surf = SurfaceRZFourier.from_nescoil_input(
            TEST_DIR / "nescin.LandremanSenguptaPlunk_section5p3_separation0_uniform_arclength",
            "current",
            quadpoints_phi=plasma_surf.quadpoints_phi,
            quadpoints_theta=plasma_surf.quadpoints_theta,
        )
        np.testing.assert_allclose(plasma_surf.gamma(), regcoil_surf.gamma(), atol=1e-3)



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
                    scaled_s_str = json.dumps(SIMSON(scaled_s), cls=GSONEncoder)
                    json.loads(scaled_s_str, cls=GSONDecoder)


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


class isSelfIntersecting(unittest.TestCase):
    """
    Tests the self-intersection algorithm:
    """
    @unittest.skipIf(ground is None or bentley_ottmann is None,
                     "Libraries to check whether self-intersecting or not are missing")

    def test_cross_section(self):
        # this cross section calculation fails on the previous implementation of the cross
        # section algorithm
        filename = os.path.join(TEST_DIR, 'serial2680021.json')
        [surfaces, coils] = load(filename)
        angle = np.pi/10
        xs = surfaces[-1].cross_section(angle/(2*np.pi), thetas=256)
        Z = xs[:, 2]
        assert np.all(Z<-0.08)
        
        # take this surface, and rotate it 30 degrees about the x-axis.  This should cause
        # the surface to 'go back' on itself, and trigger the exception.
        surface_orig = SurfaceXYZTensorFourier(mpol=surfaces[-1].mpol, ntor=surfaces[-1].ntor,\
                stellsym=True, nfp=surfaces[-1].nfp, quadpoints_phi=np.linspace(0, 1, 100),\
                quadpoints_theta=surfaces[-1].quadpoints_theta)
        surface_orig.x = surfaces[-1].x

        angle = 2*np.pi*(30/180)
        R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        gamma = surface_orig.gamma().copy()
        Rgamma = gamma@R.T

        surface_rotated = SurfaceXYZTensorFourier(mpol=surfaces[-1].mpol, ntor=surfaces[-1].ntor,\
                stellsym=True, nfp=1, quadpoints_phi=np.linspace(0, 1, 100),\
                quadpoints_theta=surfaces[-1].quadpoints_theta)
        surface_rotated.least_squares_fit(Rgamma)
        
        # unit test to check that the exceptions are properly raised
        with self.assertRaises(Exception):
            _ = surface_rotated.cross_section(0., thetas=256)

        with self.assertRaises(Exception):
            _ = surface_rotated.cross_section(0., thetas='wrong')
        
    def test_is_self_intersecting(self):
        # dofs results in a surface that is self-intersecting
        dofs = np.array([1., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1,
                         0., 0., 0., 0., 0., 0., 0.1])
        s = get_surface('SurfaceRZFourier', True, full=True, nphi=200, ntheta=200, mpol=2, ntor=2)
        s.x = dofs
        assert s.is_self_intersecting()

        s = get_surface('SurfaceRZFourier', True, full=True, nphi=200, ntheta=200, mpol=2, ntor=2)
        assert not s.is_self_intersecting()

        # make sure it works on an NCSX BoozerSurface
        bs, boozer_surf = get_boozer_surface()
        s = boozer_surf.surface
        assert not s.is_self_intersecting(angle=0.123*np.pi/(2*np.pi))
        assert not s.is_self_intersecting(angle=0.123*np.pi/(2*np.pi), thetas=200)
        assert not s.is_self_intersecting(thetas=231)

        # make sure it works on a perturbed NCSX BoozerSurface
        dofs = s.x.copy()
        dofs[14] += 0.2
        s.x = dofs
        assert s.is_self_intersecting(angle=0.123*np.pi/(2*np.pi))
        assert s.is_self_intersecting(angle=0.123*np.pi/(2*np.pi), thetas=200)
        assert s.is_self_intersecting(thetas=202)


class UtilTests(unittest.TestCase):
    def test_extend_via_normal(self):
        """
        If you apply extend_via_normal() or extend_via_projected_normal() to a
        circular-cross-section axisymmetric torus, you should get back a similar
        torus but with the expected larger minor radius.
        """
        mpol = 4
        ntor = 5
        nfp = 3
        surf1 = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, nfp=nfp)
        surf2 = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, nfp=nfp)
        R0 = 1.7
        aminor1 = 0.3
        aminor2 = 0.5
        surf1.set_rc(0, 0, R0)
        surf2.set_rc(0, 0, R0)
        surf1.set_rc(1, 0, aminor1)
        surf2.set_rc(1, 0, aminor2)
        surf1.set_zs(1, 0, aminor1)
        surf2.set_zs(1, 0, aminor2)
        x0 = surf1.x
        assert max(np.abs(surf1.x - surf2.x)) > 0.1

        surf1.extend_via_normal(aminor2 - aminor1)
        np.testing.assert_allclose(surf1.x, surf2.x, atol=1e-14)

        surf1.x = x0  # Restore the original shape
        assert max(np.abs(surf1.x - surf2.x)) > 0.1

        surf1.extend_via_projected_normal(aminor2 - aminor1)
        np.testing.assert_allclose(surf1.x, surf2.x, atol=1e-14)


class DofNames(unittest.TestCase):
    """
    Check that the dof names correspond to the correct values within each
    Surface class by using the set and get methods and checking against
    the internal arrays.
    """
    def test_dof_names(self):
        surfacetypes = ["SurfaceRZFourier", "SurfaceXYZFourier",
                        "SurfaceXYZTensorFourier"]
        for surfacetype in surfacetypes:
            for stellsym in [False, True]:
                with self.subTest(surfacetype=surfacetype):
                    self.subtest_set_get_surf_dofs(surfacetype, stellsym)

    def subtest_set_get_surf_dofs(self, surfacetype, stellsym):
        s = get_surface(surfacetype, stellsym, full=True)
        # for each surface, set some random dofs and check for consistency
        if surfacetype == "SurfaceRZFourier":
            s.set('rc(3,2)', 1.0)
            s.set('zs(1,-3)', 2.0)
            assert s.get('rc(3,2)') == s.rc[3, s.ntor + 2]
            assert s.get('zs(1,-3)') == s.zs[1, s.ntor - 3]
            if not stellsym:
                s.set('rs(3,2)', 1.0)
                s.set('zc(1,-3)', 2.0)
                assert s.get('rs(3,2)') == s.rs[3, s.ntor + 2]
                assert s.get('zc(1,-3)') == s.zc[1, s.ntor - 3]
            else:
                with self.assertRaises(Exception):
                    # make sure that stellarator symmetric surfaces don't
                    # have stellarator asymmetric modes
                    s.set('rs(3,2)', 1.0)
                    s.set('zc(1,-3)', 2.0)
        elif surfacetype == "SurfaceXYZFourier":
            s.set('xc(3,2)', 1.0)
            s.set('ys(1,-3)', 2.0)
            s.set('zs(2,1)', 3.0)
            assert s.get('xc(3,2)') == s.xc[3, s.ntor + 2]
            assert s.get('ys(1,-3)') == s.ys[1, s.ntor - 3]
            assert s.get('zs(2,1)') == s.zs[2, s.ntor + 1]
            if not stellsym:
                s.set('xs(3,2)', 1.0)
                s.set('yc(1,-3)', 2.0)
                s.set('zc(0,0)', 3.0)
                assert s.get('xs(3,2)') == s.xs[3, s.ntor + 2]
                assert s.get('yc(1,-3)') == s.yc[1, s.ntor - 3]
                assert s.get('zc(0,0)') == s.zc[0, s.ntor]
            else:
                with self.assertRaises(Exception):
                    s.set('xs(3,2)', 1.0)
                    s.set('yc(1,-3)', 2.0)
                    s.set('zc(0,0)', 3.0)
        elif surfacetype == "SurfaceXYZTensorFourier":
            s.set('x(3,2)', 1.0)
            s.set('y(7,1)', 2.0)
            s.set('z(10,5)', 3.0)
            # note for this class, indices start from 0 not -ntor
            assert s.get('x(3,2)') == s.xcs[3, 2]
            assert s.get('y(7,1)') == s.ycs[7, 1]
            assert s.get('z(10,5)') == s.zcs[10, 5]
            if not stellsym:
                s.set('x(3,7)', 1.0)
                s.set('y(7,6)', 2.0)
                s.set('z(0,0)', 3.0)
                assert s.get('x(3,7)') == s.xcs[3, 7]
                assert s.get('y(7,6)') == s.ycs[7, 6]
                assert s.get('z(0,0)') == s.zcs[0, 0]
            else:
                with self.assertRaises(Exception):
                    s.set('x(3,7)', 1.0)
                    s.set('y(7,6)', 2.0)
                    s.set('z(0,0)', 3.0)
        else:
            raise NotImplementedError("Surface type not implemented")


if __name__ == "__main__":
    unittest.main()
