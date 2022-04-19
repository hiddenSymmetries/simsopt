from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
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

    #def test(self):


if __name__ == "__main__":
    unittest.main()
