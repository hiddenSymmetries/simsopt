import unittest

import numpy as np

from simsopt.field import VmecField
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import B_cartesian, vmec_compute_geometry

from . import TEST_DIR

LBD_QH = TEST_DIR / "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"


class VmecFieldTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vmec = Vmec(str(LBD_QH))
        cls.field = VmecField(cls.vmec)
        cls.field._build()
        cls.rng = np.random.default_rng(0)

    def points(self, s):
        theta, phi = self.rng.uniform(0, 2 * np.pi, len(s)), self.rng.uniform(0, 2 * np.pi, len(s))
        return self.field._geometry(np.sqrt(s), theta, phi)[0], theta, phi

    def test_vmec_coordinates(self):
        s0 = self.rng.uniform(1e-3, 1, 500)
        x, theta0, phi0 = self.points(s0)
        s, theta, phi, inside = self.field.vmec_coordinates(x)
        self.assertTrue(inside.all())
        np.testing.assert_allclose(s, s0, atol=1e-12)
        np.testing.assert_allclose(self.field._geometry(np.sqrt(s), theta, phi)[0], x, atol=1e-10)
        np.testing.assert_allclose(np.cos(phi - phi0), 1, atol=1e-12)

    def test_boundary_field(self):
        """On the boundary, B is that of B_cartesian (same extrapolation to s = 1)."""
        x, theta, phi = self.points(np.ones(40))
        self.field.set_points(x)
        Bx, By, Bz = B_cartesian(self.vmec, quadpoints_phi=phi / (2 * np.pi), quadpoints_theta=theta / (2 * np.pi))
        B = np.stack([np.diag(Bx), np.diag(By), np.diag(Bz)], axis=-1)
        np.testing.assert_allclose(self.field.B(), B, rtol=0, atol=1e-12 * np.abs(B).max())

    def test_modB(self):
        s = np.array([0.1, 0.4, 0.7, 0.95])
        theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        phi = np.linspace(0, np.pi / 2, 4, endpoint=False)
        modB = vmec_compute_geometry(self.vmec, s, theta, phi).modB
        S, T, P = np.meshgrid(s, theta, phi, indexing="ij")
        self.field.set_points(self.field._geometry(np.sqrt(S.ravel()), T.ravel(), P.ravel())[0])
        np.testing.assert_allclose(np.linalg.norm(self.field.B(), axis=1).reshape(S.shape), modB, rtol=1e-4)

    def test_outside(self):
        x, theta, phi = self.points(np.ones(20))
        _, n = self.field._outward_normal(theta, phi)
        s, _, _, inside = self.field.vmec_coordinates(x + 0.01 * n)
        self.assertFalse(inside.any())
        self.field.set_points(x + 0.01 * n)
        self.assertTrue(np.isnan(self.field.B()).all())


if __name__ == "__main__":
    unittest.main()
