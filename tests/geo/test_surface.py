import unittest
from pathlib import Path
import numpy as np
import os

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.surfacehenneberg import SurfaceHenneberg
from simsopt.geo.surfacegarabedian import SurfaceGarabedian
from simsopt.geo.surface import signed_distance_from_surface
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
            s = eval(surface_type + "(ntheta=17)")
            np.testing.assert_allclose(s.quadpoints_theta,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))

            # Try specifying quadpoints_theta as a numpy array:
            s = eval(surface_type + "(quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False))")
            np.testing.assert_allclose(s.quadpoints_theta,
                                       np.linspace(0.0, 1.0, 5, endpoint=False))

            # Try specifying quadpoints_theta as a list:
            s = eval(surface_type + "(quadpoints_theta=[0.2, 0.7, 0.3])")
            np.testing.assert_allclose(s.quadpoints_theta, [0.2, 0.7, 0.3])

            # Specifying both ntheta and quadpoints_theta should cause an error:
            with self.assertRaises(ValueError):
                s = eval(surface_type + "(ntheta=5, quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False))")

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
            s = eval(surface_type + "(nphi=17)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))

            # Try specifying nphi plus range as a string, without nfp:
            s = eval(surface_type + "(nphi=17, range='full torus')")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + "(nphi=17, range='field period')")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + "(nphi=17, range='half period')")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 0.5, 17, endpoint=False))

            # Try specifying nphi plus range as a string, with nfp:
            s = eval(surface_type + "(nphi=17, range='full torus', nfp=3)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + "(nphi=17, range='field period', nfp=3)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0 / 3.0, 17, endpoint=False))
            s = eval(surface_type + "(nphi=17, range='half period', nfp=3)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 0.5 / 3.0, 17, endpoint=False))

            # Try specifying nphi plus range as a constant, with nfp:
            s = eval(surface_type + "(nfp=4, nphi=17, range=" + surface_type + ".RANGE_FULL_TORUS)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0, 17, endpoint=False))
            s = eval(surface_type + "(nfp=4, nphi=17, range=" + surface_type + ".RANGE_FIELD_PERIOD)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 1.0 / 4.0, 17, endpoint=False))
            s = eval(surface_type + "(nfp=4, nphi=17, range=" + surface_type + ".RANGE_HALF_PERIOD)")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0.0, 0.5 / 4.0, 17, endpoint=False))

            # Try specifying quadpoints_phi as a numpy array:
            s = eval(surface_type + "(quadpoints_phi=np.linspace(0.0, 1.0, 5, endpoint=False))")
            np.testing.assert_allclose(s.quadpoints_phi,
                                       np.linspace(0, 1.0, 5, endpoint=False))

            # Try specifying quadpoints_phi as a list:
            s = eval(surface_type + "(quadpoints_phi=[0.2, 0.7, 0.3])")
            np.testing.assert_allclose(s.quadpoints_phi, [0.2, 0.7, 0.3])

            # Specifying both nphi and quadpoints_phi should cause an error:
            with self.assertRaises(ValueError):
                s = eval(surface_type + "(nphi=5, quadpoints_phi=np.linspace(0.0, 1.0, 5, endpoint=False))")


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


if __name__ == "__main__":
    unittest.main()
