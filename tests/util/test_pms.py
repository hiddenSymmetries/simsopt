from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import InterpolatedField, DipoleField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.field.coil import Current, coils_via_symmetries
import simsoptpp as sopp
import numpy as np
import unittest
# File for the desired boundary magnetic surface:
filename = '../test_files/input.LandremanPaul2021_QA'


class Testing(unittest.TestCase):

    def test_bad_params(self):
        """
            Test that the permanent magnet optimizer initialization
            correctly catches bad instances of the function arguments.
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, coil_offset=0.1, dr=-0.05, plasma_offset=0.1,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, coil_offset=-0.1, dr=0.05, plasma_offset=0.1,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, coil_offset=0.1, dr=0.05, plasma_offset=0.0,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                10,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=10
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_outer_surface=10
            )
        inner = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta-2)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta+2)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi-2, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi+2, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )

    def test_projected_normal(self):
        """
            Make two RZFourier surfaces, extend one of them with
            the projected normal vector in the RZ plane, and check
            that both surfaces still have the same values of the
            toroidal angle. 
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        p = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s.extend_via_projected_normal(s.quadpoints_phi, 0.1)
        pgamma = p.gamma().reshape(-1, 3)
        pphi = np.arctan2(pgamma[:, 1], pgamma[:, 0])
        sgamma = s.gamma().reshape(-1, 3)
        sphi = np.arctan2(sgamma[:, 1], sgamma[:, 0])
        assert np.allclose(pphi, sphi)
        assert not np.allclose(pgamma[:, 0], sgamma[:, 0])
        assert not np.allclose(pgamma[:, 1], sgamma[:, 1])
        assert not np.allclose(pgamma[:, 2], sgamma[:, 2])

    def test_fp_symmetry(self):
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        # Create some initial coils:
        base_curves = create_equally_spaced_curves(2, s.nfp, stellsym=False, R0=1.0, R1=0.5, order=5)
        base_currents = [Current(1e5) for i in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        # Create PM class 
        pm_opt = PermanentMagnetOptimizer(s, dr=0.15, B_plasma_surface=bs.B().reshape((nphi, ntheta, 3)))
        _, _, _, dipoles = pm_opt._optimize()
        #b_dipole = DipoleField(pm_opt.dipole_grid, dipoles, pm_opt)
        n = 16
        rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
        zs = s.gamma()[:, :, 2]
        rrange = np.linspace(np.min(rs), np.max(rs), n)
        phirange = np.linspace(0, 2 * np.pi / s.nfp, n * 2)
        zrange = np.linspace(0, np.max(zs), n // 2)
        points = np.zeros((n, 2 * n, n // 2, 3))
        for i in range(n):
            for j in range(n * 2):
                for k in range(n // 2):
                    points[i, j, k, 0] = rrange[i]
                    points[i, j, k, 1] = phirange[j]
                    points[i, j, k, 2] = zrange[k]
        #b_dipole.set_points(points)
        #print(b_dipole.B())


if __name__ == "__main__":
    unittest.main()
