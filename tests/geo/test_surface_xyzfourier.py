import unittest
import json

import numpy as np
from monty.json import MontyDecoder, MontyEncoder

from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from .surface_test_helpers import get_surface, get_exact_surface

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False


class SurfaceXYZFourierTests(unittest.TestCase):
    def test_toRZFourier_perfect_torus(self):
        """
        This test checks that a perfect torus can be converted from SurfaceXYZFourier to SurfaceRZFourier
        completely losslessly.
        """
        for stellsym in stellsym_list:
            with self.subTest(stellsym=stellsym):
                self.subtest_toRZFourier_perfect_torus("SurfaceXYZFourier", stellsym)

    def subtest_toRZFourier_perfect_torus(self, surfacetype, stellsym):
        """
        The test obtains a perfect torus as a SurfaceXYZFourier, then converts it to a SurfaceRZFourier.  Next,
        it computes the cross section of both surfaces at a random angle and compares the pointwise values.
        """
        s = get_surface(surfacetype, stellsym)
        sRZ = s.to_RZFourier()

        np.random.seed(0)
        angle = np.random.random()*1000
        scs = s.cross_section(angle, thetas=100)
        sRZcs = sRZ.cross_section(angle, thetas=100)

        max_pointwise_err = np.max(np.abs(scs - sRZcs))
        print(max_pointwise_err)

        # compute the cylindrical angle error of the cross section
        an = np.arctan2(scs[:, 1], scs[:, 0])
        phi = angle
        phi = phi - np.sign(phi) * np.floor(np.abs(phi) / (2*np.pi)) * (2. * np.pi)
        if phi > np.pi:
            phi = phi - 2. * np.pi
        if phi < -np.pi:
            phi = phi + 2. * np.pi
        max_angle_err = np.max(np.abs(an - phi))

        # check that the angle error is what we expect
        assert max_angle_err < 1e-12
        # check that the pointwise error is what we expect
        assert max_pointwise_err < 1e-12

    def test_toRZFourier_lossless_at_quadraturepoints(self):
        """
        This test obtains a more complex surface (not a perfect torus) as a SurfaceXYZFourier, then
        converts that surface to the SurfaceRZFourier representation.  Then, the test checks that both
        surface representations coincide at the points where the least squares fit was completed,
        i.e., the conversion is lossless at the quadrature points.

        Additionally, this test checks that the cross sectional angle is correct.
        """
        s = get_exact_surface()
        sRZ = s.to_RZFourier()

        max_angle_error = -1
        max_pointwise_error = -1
        for angle in sRZ.quadpoints_phi:
            scs = s.cross_section(angle * 2 * np.pi)
            sRZcs = sRZ.cross_section(angle * 2 * np.pi)

            # compute the cylindrical angle error of the cross section
            phi = angle * 2. * np.pi
            phi = phi - np.sign(phi) * np.floor(np.abs(phi) / (2*np.pi)) * (2. * np.pi)
            if phi > np.pi:
                phi = phi - 2. * np.pi
            if phi < -np.pi:
                phi = phi + 2. * np.pi

            an = np.arctan2(scs[:, 1], scs[:, 0])
            curr_angle_err = np.max(np.abs(an - phi))
            if max_angle_error < curr_angle_err:
                max_angle_error = curr_angle_err
            curr_pointwise_err = np.max(np.abs(scs - sRZcs))
            if max_pointwise_error < curr_pointwise_err:
                max_pointwise_error = curr_pointwise_err

        # check that the angle of the cross section is what we expect
        assert max_pointwise_error < 1e-12
        # check that the pointwise error is what we expect
        assert max_angle_error < 1e-12

    def test_toRZFourier_small_loss_elsewhere(self):
        """
        Away from the quadrature points, the conversion is not lossless and this test verifies that the
        error is small.

        Additionally, this test checks that the cross sectional angle is correct.        
        """
        s = get_exact_surface()
        sRZ = s.to_RZFourier()

        np.random.seed(0)
        angle = np.random.random()*1000
        scs = s.cross_section(angle)
        sRZcs = sRZ.cross_section(angle)

        # compute the cylindrical angle error of the cross section
        phi = angle
        phi = phi - np.sign(phi) * np.floor(np.abs(phi) / (2*np.pi)) * (2. * np.pi)
        if phi > np.pi:
            phi = phi - 2. * np.pi
        if phi < -np.pi:
            phi = phi + 2. * np.pi

        max_pointwise_err = np.max(np.abs(scs - sRZcs))
        print(max_pointwise_err)
        assert max_pointwise_err < 1e-3

        an = np.arctan2(scs[:, 1], scs[:, 0])
        max_angle_err1 = np.max(np.abs(an - phi))
        assert max_angle_err1 < 1e-12

        an = np.arctan2(sRZcs[:, 1], sRZcs[:, 0])
        max_angle_err2 = np.max(np.abs(an - phi))
        assert max_angle_err2 < 1e-12

    def test_cross_section_torus(self):
        """
        Test that the cross sectional area at a certain number of cross sections
        of a torus is what it should be. The cross sectional angles are chosen
        to test any degenerate cases of the bisection algorithm.

        Additionally, this test checks that the cross sectional angle is correct.
        """
        mpol = 4
        ntor = 3
        nfp = 2
        phis = np.linspace(0, 1, 31, endpoint=False)
        thetas = np.linspace(0, 1, 31, endpoint=False)

        np.random.seed(0)

        stellsym = False
        s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                              quadpoints_phi=phis, quadpoints_theta=thetas)
        s.xc = s.xc * 0
        s.xs = s.xs * 0
        s.ys = s.ys * 0
        s.yc = s.yc * 0
        s.zs = s.zs * 0
        s.zc = s.zc * 0
        r1 = np.random.random_sample() + 0.1
        r2 = np.random.random_sample() + 0.1
        major_R = np.max([r1, r2])
        minor_R = np.min([r1, r2])
        s.xc[0, ntor] = major_R
        s.xc[1, ntor] = minor_R
        s.zs[1, ntor] = minor_R

        num_cs = 9
        angle = np.zeros((num_cs,))
        angle[0] = 0.
        angle[1] = np.pi/2.
        angle[2] = np.pi
        angle[3] = 3 * np.pi / 2.
        angle[4] = 2. * np.pi
        angle[5] = -np.pi/2.
        angle[6] = -np.pi
        angle[7] = -3. * np.pi / 2.
        angle[8] = -2. * np.pi
        cs = np.zeros((num_cs, 100, 3))
        for idx in range(angle.size):
            cs[idx, :, :] = s.cross_section(angle[idx], thetas=100)

        cs_area = np.zeros((num_cs,))
        max_angle_error = -1

        from scipy import fftpack
        for i in range(num_cs):

            phi = angle[i]
            phi = phi - np.sign(phi) * np.floor(np.abs(phi) / (2*np.pi)) * (2. * np.pi)
            if phi > np.pi:
                phi = phi - 2. * np.pi
            if phi < -np.pi:
                phi = phi + 2. * np.pi

            # check that the angle of the cross section is what we expect
            an = np.arctan2(cs[i, :, 1], cs[i, :, 0])
            curr_angle_err = np.max(np.abs(an - phi))

            if max_angle_error < curr_angle_err:
                max_angle_error = curr_angle_err

            R = np.sqrt(cs[i, :, 0]**2 + cs[i, :, 1]**2)
            Z = cs[i, :, 2]
            Rp = fftpack.diff(R, period=1.)
            Zp = fftpack.diff(Z, period=1.)
            cs_area[i] = np.abs(np.mean(Z*Rp))
        exact_area = np.pi * minor_R**2.

        # check that the cross sectional area is what we expect
        assert np.max(np.abs(cs_area - exact_area)) < 1e-14
        # check that the angle error is what we expect
        assert max_angle_error < 1e-12

    def test_aspect_ratio_random_torus(self):
        """
        This is a simple aspect ratio validation on a torus with minor radius = r1
        and major radius = r2, where 0.1 <= r1 <= r2 are random numbers
        """
        mpol = 4
        ntor = 3
        nfp = 2
        phis = np.linspace(0, 1, 31, endpoint=False)
        thetas = np.linspace(0, 1, 31, endpoint=False)

        stellsym = False
        s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                              quadpoints_phi=phis, quadpoints_theta=thetas)
        s.xc = s.xc * 0
        s.xs = s.xs * 0
        s.ys = s.ys * 0
        s.yc = s.yc * 0
        s.zs = s.zs * 0
        s.zc = s.zc * 0
        np.random.seed(0)
        r1 = np.random.random_sample() + 0.1
        r2 = np.random.random_sample() + 0.1
        major_R = np.max([r1, r2])
        minor_R = np.min([r1, r2])
        s.xc[0, ntor] = major_R
        s.xc[1, ntor] = minor_R
        s.zs[1, ntor] = minor_R

        print("AR approx: ", s.aspect_ratio(), "Exact: ", major_R/minor_R)
        self.assertAlmostEqual(s.aspect_ratio(), major_R/minor_R)

    def test_aspect_ratio_compare_with_cross_sectional_computation(self):
        """
        This test validates the VMEC aspect ratio computation in the Surface
        class by comparing with an approximation based on cross section
        computations.
        """
        s = get_exact_surface()
        vpr = s.quadpoints_phi.size + 20
        tr = s.quadpoints_theta.size + 20
        cs_area = np.zeros((vpr,))

        from scipy import fftpack
        angle = np.linspace(-np.pi, np.pi, vpr, endpoint=False)
        for idx in range(angle.size):
            cs = s.cross_section(angle[idx], thetas=tr)
            R = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
            Z = cs[:, 2]
            Rp = fftpack.diff(R, period=1.)
            Zp = fftpack.diff(Z, period=1.)
            ar = np.mean(Z*Rp)
            cs_area[idx] = ar

        mean_cross_sectional_area = np.mean(cs_area)
        R_minor = np.sqrt(mean_cross_sectional_area / np.pi)
        R_major = np.abs(s.volume()) / (2. * np.pi**2 * R_minor**2)
        AR_cs = R_major / R_minor
        AR = s.aspect_ratio()

        rel_err = np.abs(AR-AR_cs) / AR
        print(AR, AR_cs)
        print("AR rel error is:", rel_err)
        assert rel_err < 1e-5

    @unittest.skipIf(not pyevtk_found, "pyevtk not found")
    def test_to_vtk(self):
        mpol = 4
        ntor = 3
        nfp = 2
        phis = np.linspace(0, 1, 31, endpoint=False)
        thetas = np.linspace(0, 1, 31, endpoint=False)
        stellsym = False
        s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, quadpoints_phi=phis, quadpoints_theta=thetas)

        s.to_vtk('/tmp/surface')

    def test_serialization(self):
        mpol = 4
        ntor = 3
        nfp = 2
        phis = np.linspace(0, 1, 31, endpoint=False)
        thetas = np.linspace(0, 1, 31, endpoint=False)

        np.random.seed(0)

        stellsym = False
        s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                              quadpoints_phi=phis, quadpoints_theta=thetas)
        s.xc = s.xc * 0
        s.xs = s.xs * 0
        s.ys = s.ys * 0
        s.yc = s.yc * 0
        s.zs = s.zs * 0
        s.zc = s.zc * 0
        r1 = np.random.random_sample() + 0.1
        r2 = np.random.random_sample() + 0.1
        major_R = np.max([r1, r2])
        minor_R = np.min([r1, r2])
        s.xc[0, ntor] = major_R
        s.xc[1, ntor] = minor_R
        s.zs[1, ntor] = minor_R
        # TODO: x should be updated whenever [x,y,z][c,s] are modified without
        # TODO: explict setting of local_full_x
        s.local_full_x = s.get_dofs()

        surf_str = json.dumps(s, cls=MontyEncoder)
        s_regen = json.loads(surf_str, cls=MontyDecoder)

        self.assertAlmostEqual(s.area(), s_regen.area(), places=4)
        self.assertAlmostEqual(s.volume(), s_regen.volume(), places=3)


if __name__ == "__main__":
    unittest.main()
