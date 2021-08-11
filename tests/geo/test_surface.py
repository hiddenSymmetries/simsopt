import unittest
import logging
from pathlib import Path
import numpy as np

from simsopt._core.dofs import Dofs
from simsopt._core.optimizable import make_optimizable
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacegarabedian import SurfaceGarabedian
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surface import signed_distance_from_surface
from simsopt.geo.curverzfourier import CurveRZFourier
from .surface_test_helpers import get_surface, get_exact_surface

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False

# logging.basicConfig(level=logging.DEBUG)


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
        Test that the cross sectional area at a certain number of cross sections of a torus
        is what it should be.  The cross sectional angles are chosen to test any degenerate 
        cases of the bisection algorithm.

        Additionally, this test checks that the cross sectional angle is correct.
        """
        mpol = 4
        ntor = 3
        nfp = 2
        phis = np.linspace(0, 1, 31, endpoint=False)
        thetas = np.linspace(0, 1, 31, endpoint=False)

        np.random.seed(0)

        stellsym = False
        s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, quadpoints_phi=phis, quadpoints_theta=thetas)
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
        s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, quadpoints_phi=phis, quadpoints_theta=thetas)
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
        This test validates the VMEC aspect ratio computation in the Surface class by 
        comparing with an approximation based on cross section computations.
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


class SurfaceRZFourierTests(unittest.TestCase):
    def test_aspect_ratio(self):
        """
        Test that the aspect ratio of a torus with random minor and major radius 0.1 <= minor_R <= major_R
        is properly computed to be major_R/minor_R.
        """

        s = SurfaceRZFourier(nfp=2, mpol=3, ntor=2)
        s.rc = s.rc * 0
        s.rs = s.rs * 0
        s.zc = s.zc * 0
        s.zs = s.zs * 0
        r1 = np.random.random_sample() + 0.1
        r2 = np.random.random_sample() + 0.1
        major_R = np.max([r1, r2])
        minor_R = np.min([r1, r2])
        s.rc[0, 2] = major_R
        s.rc[1, 2] = minor_R
        s.zs[1, 2] = minor_R
        print("AR approx: ", s.aspect_ratio(), "Exact: ", major_R/minor_R)
        self.assertAlmostEqual(s.aspect_ratio(), major_R/minor_R)

    def test_init(self):
        s = SurfaceRZFourier(nfp=2, mpol=3, ntor=2)
        self.assertEqual(s.rc.shape, (4, 5))
        self.assertEqual(s.zs.shape, (4, 5))

        s = SurfaceRZFourier(nfp=10, mpol=1, ntor=3, stellsym=False)
        self.assertEqual(s.rc.shape, (2, 7))
        self.assertEqual(s.zs.shape, (2, 7))
        self.assertEqual(s.rs.shape, (2, 7))
        self.assertEqual(s.zc.shape, (2, 7))

    def test_area_volume(self):
        """
        Test the calculation of area and volume for an axisymmetric surface
        """
        s = SurfaceRZFourier()
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[1, 0] = 0.2

        true_area = 15.827322032265993
        true_volume = 2.0528777154265874
        self.assertAlmostEqual(s.area(), true_area, places=4)
        self.assertAlmostEqual(s.volume(), true_volume, places=3)

    def test_get_dofs(self):
        """
        Test that we can convert the degrees of freedom into a 1D vector
        """

        # First try an axisymmetric surface for simplicity:
        s = SurfaceRZFourier()
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[0, 0] = 0.3
        s.zs[1, 0] = 0.2
        dofs = s.get_dofs()
        self.assertEqual(dofs.shape, (3,))
        self.assertAlmostEqual(dofs[0], 1.3)
        self.assertAlmostEqual(dofs[1], 0.4)
        self.assertAlmostEqual(dofs[2], 0.2)

        # Now try a nonaxisymmetric shape:
        s = SurfaceRZFourier(mpol=3, ntor=1)
        s.rc[:, :] = [[100, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] 
        s.zs[:, :] = [[101, 102, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]] 
        dofs = s.get_dofs()
        self.assertEqual(dofs.shape, (21,))
        for j in range(21):
            self.assertAlmostEqual(dofs[j], j + 2)

    def test_set_dofs(self):
        """
        Test that we can set the shape from a 1D vector
        """

        # First try an axisymmetric surface for simplicity:
        s = SurfaceRZFourier()
        s.set_dofs([2.9, -1.1, 0.7])
        self.assertAlmostEqual(s.rc[0, 0], 2.9)
        self.assertAlmostEqual(s.rc[1, 0], -1.1)
        self.assertAlmostEqual(s.zs[1, 0], 0.7)

        # Now try a nonaxisymmetric shape:
        s = SurfaceRZFourier(mpol=3, ntor=1)
        s.set_dofs(np.array(list(range(21))) + 1)
        self.assertAlmostEqual(s.rc[0, 0], 0)
        self.assertAlmostEqual(s.rc[0, 1], 1)
        self.assertAlmostEqual(s.rc[0, 2], 2)
        self.assertAlmostEqual(s.rc[1, 0], 3)
        self.assertAlmostEqual(s.rc[1, 1], 4)
        self.assertAlmostEqual(s.rc[1, 2], 5)
        self.assertAlmostEqual(s.rc[2, 0], 6)
        self.assertAlmostEqual(s.rc[2, 1], 7)
        self.assertAlmostEqual(s.rc[2, 2], 8)
        self.assertAlmostEqual(s.rc[3, 0], 9)
        self.assertAlmostEqual(s.rc[3, 1], 10)
        self.assertAlmostEqual(s.rc[3, 2], 11)

        self.assertAlmostEqual(s.zs[0, 0], 0)
        self.assertAlmostEqual(s.zs[0, 1], 0)
        self.assertAlmostEqual(s.zs[0, 2], 12)
        self.assertAlmostEqual(s.zs[1, 0], 13)
        self.assertAlmostEqual(s.zs[1, 1], 14)
        self.assertAlmostEqual(s.zs[1, 2], 15)
        self.assertAlmostEqual(s.zs[2, 0], 16)
        self.assertAlmostEqual(s.zs[2, 1], 17)
        self.assertAlmostEqual(s.zs[2, 2], 18)
        self.assertAlmostEqual(s.zs[3, 0], 19)
        self.assertAlmostEqual(s.zs[3, 1], 20)
        self.assertAlmostEqual(s.zs[3, 2], 21)

    def test_from_wout(self):
        """
        Test reading in surfaces from a VMEC wout file.
        """

        # First try a stellarator-symmetric example:
        filename = TEST_DIR / 'wout_li383_low_res_reference.nc'
        s = SurfaceRZFourier.from_wout(filename)
        # The value in the next line includes m values up to 4 even
        # though the RBC/ZBS arrays for the input file go up to m=6,
        # since mpol is only 4.
        true_volume = 2.98138727016329
        self.assertAlmostEqual(s.volume(), true_volume, places=8)
        # Try specifying the number of quadrature points:
        s = SurfaceRZFourier.from_wout(filename, quadpoints_phi=71, quadpoints_theta=78)
        self.assertAlmostEqual(s.volume(), true_volume, places=8)
        # If you ask for the s=0 surface, which is just the magnetic
        # axis, the volume and area should be 0.
        s = SurfaceRZFourier.from_wout(filename, 0)
        self.assertTrue(np.abs(s.volume()) < 1.0e-13)
        self.assertTrue(np.abs(s.area()) < 1.0e-13)

        # Now try a non-stellarator-symmetric example:
        filename = TEST_DIR / 'wout_LandremanSenguptaPlunk_section5p3_reference.nc'
        s = SurfaceRZFourier.from_wout(filename)
        self.assertAlmostEqual(s.volume(), 0.199228326859097, places=8)
        # If you ask for the s=0 surface, which is just the magnetic
        # axis, the volume and area should be 0.
        s = SurfaceRZFourier.from_wout(filename, 0)
        self.assertTrue(np.abs(s.volume()) < 1.0e-13)
        self.assertTrue(np.abs(s.area()) < 1.0e-13)

    def test_from_vmec_input(self):
        """
        Test reading in surfaces from a VMEC input file.
        """

        # First try some stellarator-symmetric examples:
        filename = TEST_DIR / 'input.li383_low_res'
        s = SurfaceRZFourier.from_vmec_input(filename)
        # The value in the next line includes m values up through 6,
        # even though mpol in the file is 4.
        true_volume = 2.97871721453671
        self.assertAlmostEqual(s.volume(), true_volume, places=8)
        # Try specifying the number of quadrature points:
        s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=78, quadpoints_theta=71)
        self.assertAlmostEqual(s.volume(), true_volume, places=8)

        filename = TEST_DIR / 'input.NuhrenbergZille_1988_QHS'
        s = SurfaceRZFourier.from_vmec_input(filename)
        true_volume = 188.552389137478
        self.assertAlmostEqual(s.volume(), true_volume, places=8)

        filename = TEST_DIR / 'input.cfqs_2b40'
        s = SurfaceRZFourier.from_vmec_input(filename)
        true_volume = 1.03641220569946
        self.assertAlmostEqual(s.volume(), true_volume, places=8)

        filename = TEST_DIR / 'input.circular_tokamak'
        s = SurfaceRZFourier.from_vmec_input(filename)
        true_volume = 473.741011252289
        self.assertAlmostEqual(s.volume(), true_volume, places=8)

        # Now try a non-stellarator-symmetric example:
        filename = TEST_DIR / 'input.LandremanSenguptaPlunk_section5p3'
        s = SurfaceRZFourier.from_vmec_input(filename)
        true_volume = 0.199228326303124
        self.assertAlmostEqual(s.volume(), true_volume, places=8)
        # Try specifying the number of quadrature points:
        s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=67, quadpoints_theta=69)
        self.assertAlmostEqual(s.volume(), true_volume, places=8)

    def test_from_vmec_2_ways(self):
        """
        Verify that from_wout() and from_vmec_input() give consistent
        surfaces for a given VMEC run.
        """
        # First try a stellarator-symmetric example:
        filename1 = TEST_DIR / 'input.li383_low_res'
        filename2 = TEST_DIR / 'wout_li383_low_res_reference.nc'
        s1 = SurfaceRZFourier.from_vmec_input(filename1)
        s2 = SurfaceRZFourier.from_wout(filename2)
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        places = 13
        self.assertEqual(s1.nfp, s2.nfp)
        self.assertEqual(s1.stellsym, s2.stellsym)
        for m in range(mpol + 1):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n), places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n), places=places)

        # Now try a non-stellarator-symmetric example:
        filename1 = TEST_DIR / 'input.LandremanSenguptaPlunk_section5p3'
        filename2 = TEST_DIR / 'wout_LandremanSenguptaPlunk_section5p3_reference.nc'
        s1 = SurfaceRZFourier.from_vmec_input(filename1)
        s2 = SurfaceRZFourier.from_wout(filename2)
        self.assertEqual(s1.nfp, s2.nfp)
        self.assertEqual(s1.stellsym, s2.stellsym)
        # For non-stellarator-symmetric cases, we must be careful when
        # directly comparing the rc/zs/rs/zc coefficients, because
        # VMEC shifts the poloidal angle in readin.f upon loading the
        # file. Moreover, in versions of VMEC other than the
        # hiddenSymmetries python module, the shift to theta may have
        # a bug so the angle shift is not by the claimed value. The
        # specific input file used here has a boundary that should not
        # be shifted by the hiddenSymmetries VMEC2000 module.  For any
        # input file and version of VMEC, we can compare
        # coordinate-independent properties like the volume and area.
        self.assertAlmostEqual(np.abs(s1.volume()), np.abs(s2.volume()), places=13)
        self.assertAlmostEqual(s1.area(), s2.area(), places=7)
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        places = 13
        for m in range(mpol + 1):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n), places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n), places=places)
                self.assertAlmostEqual(s1.get_rs(m, n), s2.get_rs(m, n), places=places)
                self.assertAlmostEqual(s1.get_zc(m, n), s2.get_zc(m, n), places=places)

    def test_write_nml(self):
        """
        Test the write_nml() function. To do this, we read in a VMEC input
        namelist, call write_nml(), read in the resulting namelist as
        a new surface, and compare the data to the original surface.
        """
        # Try a stellarator-symmetric case
        filename = TEST_DIR / 'input.li383_low_res'
        s1 = SurfaceRZFourier.from_vmec_input(filename)
        new_filename = 'boundary.li383_low_res'
        s1.write_nml(new_filename)
        s2 = SurfaceRZFourier.from_vmec_input(new_filename)
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        places = 13
        self.assertEqual(s1.nfp, s2.nfp)
        self.assertEqual(s1.stellsym, s2.stellsym)
        for m in range(mpol + 1):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n), places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n), places=places)

        # Try a non-stellarator-symmetric case
        filename = TEST_DIR / 'input.LandremanSenguptaPlunk_section5p3'
        s1 = SurfaceRZFourier.from_vmec_input(filename)
        s1.write_nml()
        new_filename = 'boundary'
        s2 = SurfaceRZFourier.from_vmec_input(new_filename)
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        places = 13
        self.assertEqual(s1.nfp, s2.nfp)
        self.assertEqual(s1.stellsym, s2.stellsym)
        for m in range(mpol + 1):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n), places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n), places=places)
                self.assertAlmostEqual(s1.get_rs(m, n), s2.get_rs(m, n), places=places)
                self.assertAlmostEqual(s1.get_zc(m, n), s2.get_zc(m, n), places=places)

    def test_from_focus(self):
        """
        Try reading in a focus-format file.
        """
        filename = TEST_DIR / 'tf_only_half_tesla.plasma'

        s = SurfaceRZFourier.from_focus(filename)

        self.assertEqual(s.nfp, 3)
        self.assertTrue(s.stellsym)
        self.assertEqual(s.rc.shape, (11, 13))
        self.assertEqual(s.zs.shape, (11, 13))
        self.assertAlmostEqual(s.rc[0, 6], 1.408922E+00)
        self.assertAlmostEqual(s.rc[0, 7], 2.794370E-02)
        self.assertAlmostEqual(s.zs[0, 7], -1.909220E-02)
        self.assertAlmostEqual(s.rc[10, 12], -6.047097E-05)
        self.assertAlmostEqual(s.zs[10, 12], 3.663233E-05)

        self.assertAlmostEqual(s.get_rc(0, 0), 1.408922E+00)
        self.assertAlmostEqual(s.get_rc(0, 1), 2.794370E-02)
        self.assertAlmostEqual(s.get_zs(0, 1), -1.909220E-02)
        self.assertAlmostEqual(s.get_rc(10, 6), -6.047097E-05)
        self.assertAlmostEqual(s.get_zs(10, 6), 3.663233E-05)

        true_area = 24.5871075268402
        true_volume = 2.96201898538042
        #print("computed area: ", area, ", correct value: ", true_area, \
        #    " , difference: ", area - true_area)
        #print("computed volume: ", volume, ", correct value: ", \
        #    true_volume, ", difference:", volume - true_volume)
        self.assertAlmostEqual(s.area(), true_area, places=4)
        self.assertAlmostEqual(s.volume(), true_volume, places=3)

    def test_derivatives(self):
        """
        Check the automatic differentiation for area and volume.
        """
        for mpol in range(1, 3):
            for ntor in range(2):
                for nfp in range(1, 4):
                    s = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
                    x0 = s.get_dofs()
                    x = np.random.rand(len(x0)) - 0.5
                    x[0] = np.random.rand() + 2
                    # This surface will probably self-intersect, but I
                    # don't think this actually matters here.
                    s.set_dofs(x)

                    dofs = Dofs([s.area, s.volume])
                    jac = dofs.jac()
                    fd_jac = dofs.fd_jac()
                    print('difference for surface test_derivatives:', jac - fd_jac)
                    np.testing.assert_allclose(jac, fd_jac, rtol=1e-4, atol=1e-4)

    def test_change_resolution(self):
        """
        Check that we can change mpol and ntor.
        """
        for mpol in [1, 2]:
            for ntor in [0, 1]:
                s = SurfaceRZFourier(mpol=mpol, ntor=ntor)
                n = len(s.get_dofs())
                s.set_dofs((np.random.rand(n) - 0.5) * 0.01)
                s.set_rc(0, 0, 1.0)
                s.set_rc(1, 0, 0.1)
                s.set_zs(1, 0, 0.13)
                v1 = s.volume()
                a1 = s.area()

                s.change_resolution(mpol+1, ntor)
                s.recalculate = True
                v2 = s.volume()
                a2 = s.area()
                self.assertAlmostEqual(v1, v2)
                self.assertAlmostEqual(a1, a2)

                s.change_resolution(mpol, ntor+1)
                s.recalculate = True
                v2 = s.volume()
                a2 = s.area()
                self.assertAlmostEqual(v1, v2)
                self.assertAlmostEqual(a1, a2)

                s.change_resolution(mpol+1, ntor+1)
                s.recalculate = True
                v2 = s.volume()
                a2 = s.area()
                self.assertAlmostEqual(v1, v2)
                self.assertAlmostEqual(a1, a2)


class SurfaceGarabedianTests(unittest.TestCase):
    def test_init(self):
        """
        Check that the default surface is what we expect, and that the
        'names' array is correctly aligned.
        """
        s = make_optimizable(SurfaceGarabedian(nmin=-1, nmax=2, mmin=-2, mmax=5))
        self.assertAlmostEqual(s.Delta[2, 1], 0.1)
        self.assertAlmostEqual(s.Delta[3, 1], 1.0)
        self.assertAlmostEqual(s.get('Delta(0,0)'), 0.1)
        self.assertAlmostEqual(s.get('Delta(1,0)'), 1.0)
        # Verify all other elements are 0:
        d = np.copy(s.Delta)
        d[2, 1] = 0
        d[3, 1] = 0
        np.testing.assert_allclose(d, np.zeros((8, 4)))

        s.set('Delta(-2,-1)', 42)
        self.assertAlmostEqual(s.Delta[0, 0], 42)
        self.assertAlmostEqual(s.get_Delta(-2, -1), 42)

        s.set('Delta(5,-1)', -7)
        self.assertAlmostEqual(s.Delta[7, 0], -7)
        self.assertAlmostEqual(s.get_Delta(5, -1), -7)

        s.set('Delta(-2,2)', 13)
        self.assertAlmostEqual(s.Delta[0, 3], 13)
        self.assertAlmostEqual(s.get_Delta(-2, 2), 13)

        s.set('Delta(5,2)', -5)
        self.assertAlmostEqual(s.Delta[7, 3], -5)
        self.assertAlmostEqual(s.get_Delta(5, 2), -5)

        s.set_Delta(-2, -1, 421)
        self.assertAlmostEqual(s.Delta[0, 0], 421)

        s.set_Delta(5, -1, -71)
        self.assertAlmostEqual(s.Delta[7, 0], -71)

        s.set_Delta(-2, 2, 133)
        self.assertAlmostEqual(s.Delta[0, 3], 133)

        s.set_Delta(5, 2, -50)
        self.assertAlmostEqual(s.Delta[7, 3], -50)

    def test_convert_back(self):
        """
        If we start with a SurfaceRZFourier, convert to Garabedian, and
        convert back to SurfaceFourier, we should get back what we
        started with.
        """
        for mpol in range(1, 4):
            for ntor in range(5):
                for nfp in range(1, 4):
                    sf1 = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
                    # Set all dofs to random numbers in [-2, 2]:
                    sf1.set_dofs((np.random.rand(len(sf1.get_dofs())) - 0.5) * 4)
                    sg = sf1.to_Garabedian()
                    sf2 = sg.to_RZFourier()
                    np.testing.assert_allclose(sf1.rc, sf2.rc)
                    np.testing.assert_allclose(sf1.zs, sf2.zs)


class SurfaceDistanceTests(unittest.TestCase):
    def test_distance(self):
        c = CurveRZFourier(100, 1, 1, False)
        dofs = c.get_dofs()
        dofs[0] = 1.
        c.set_dofs(dofs)
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
