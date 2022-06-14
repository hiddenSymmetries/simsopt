import unittest
from pathlib import Path
import json

import numpy as np
from monty.json import MontyDecoder, MontyEncoder

from simsopt.geo.surfacerzfourier import SurfaceRZFourier, SurfaceRZPseudospectral

TEST_DIR = Path(__file__).parent / ".." / "test_files"

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False


class SurfaceRZFourierTests(unittest.TestCase):

    def test_aspect_ratio(self):
        """
        Test that the aspect ratio of a torus with random minor and major
        radius 0.1 <= minor_R <= major_R is properly computed to be
        major_R/minor_R.
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
        # Make sure that the graph framework dofs are sync-ed with
        # the rc/zs arrays:
        np.testing.assert_allclose(s.x, s.get_dofs())
        # The value in the next line includes m values up to 4 even
        # though the RBC/ZBS arrays for the input file go up to m=6,
        # since mpol is only 4.
        true_volume = 2.98138727016329
        self.assertAlmostEqual(s.volume(), true_volume, places=8)
        # Try specifying the number of quadrature points:
        s = SurfaceRZFourier.from_wout(filename, nphi=71,
                                       ntheta=78)
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
        # Make sure that the graph framework dofs are sync-ed with
        # the rc/zs arrays:
        np.testing.assert_allclose(s.x, s.get_dofs())
        # The value in the next line includes m values up through 6,
        # even though mpol in the file is 4.
        true_volume = 2.97871721453671
        self.assertAlmostEqual(s.volume(), true_volume, places=8)
        # Try specifying the number of quadrature points:
        s = SurfaceRZFourier.from_vmec_input(filename, nphi=78,
                                             ntheta=71)
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
        s = SurfaceRZFourier.from_vmec_input(filename, nphi=67,
                                             ntheta=69)
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
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n),
                                       places=places)

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
        self.assertAlmostEqual(np.abs(s1.volume()), np.abs(s2.volume()),
                               places=13)
        self.assertAlmostEqual(s1.area(), s2.area(), places=7)
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        places = 13
        for m in range(mpol + 1):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_rs(m, n), s2.get_rs(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_zc(m, n), s2.get_zc(m, n),
                                       places=places)

    def test_get_and_write_nml(self):
        """
        Test the get_nml() and write_nml() functions. To do this, we read in a VMEC input
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
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n),
                                       places=places)

        # Try a non-stellarator-symmetric case
        filename = TEST_DIR / 'input.LandremanSenguptaPlunk_section5p3'
        s1 = SurfaceRZFourier.from_vmec_input(filename)
        nml_str = s1.get_nml()  # This time, cover the case in which a string is returned
        new_filename = 'boundary'
        with open(new_filename, 'w') as f:
            f.write(nml_str)
        s2 = SurfaceRZFourier.from_vmec_input(new_filename)
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        places = 13
        self.assertEqual(s1.nfp, s2.nfp)
        self.assertEqual(s1.stellsym, s2.stellsym)
        for m in range(mpol + 1):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_rs(m, n), s2.get_rs(m, n),
                                       places=places)
                self.assertAlmostEqual(s1.get_zc(m, n), s2.get_zc(m, n),
                                       places=places)

    def test_from_focus(self):
        """
        Try reading in a focus-format file.
        """
        filename = TEST_DIR / 'tf_only_half_tesla.plasma'

        s = SurfaceRZFourier.from_focus(filename)

        # Make sure that the graph framework dofs are sync-ed with
        # the rc/zs arrays:
        np.testing.assert_allclose(s.x, s.get_dofs())

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

    def test_repr(self):
        s = SurfaceRZFourier(nfp=2, mpol=3, ntor=5)
        s_str = repr(s)
        self.assertIn("SurfaceRZFourier", s_str)
        self.assertIn("nfp=2", s_str)
        self.assertIn("stellsym=True", s_str)
        self.assertIn("mpol=3", s_str)
        self.assertIn("ntor=5", s_str)

    def test_get_rc(self):
        s = SurfaceRZFourier()
        s.x = [2.9, -1.1, 0.7]
        self.assertAlmostEqual(s.get_rc(0, 0), 2.9)
        self.assertAlmostEqual(s.get_rc(1, 0), -1.1)

    def test_get_zs(self):
        s = SurfaceRZFourier(mpol=3, ntor=1)
        s.x = np.array(list(range(21))) + 1

        self.assertAlmostEqual(s.get_zs(0, -1), 0)
        self.assertAlmostEqual(s.get_zs(0, 0), 0)
        self.assertAlmostEqual(s.get_zs(0, 1), 12)
        self.assertAlmostEqual(s.get_zs(1, -1), 13)
        self.assertAlmostEqual(s.get_zs(1, 0), 14)
        self.assertAlmostEqual(s.get_zs(1, 1), 15)
        self.assertAlmostEqual(s.get_zs(2, -1), 16)
        self.assertAlmostEqual(s.get_zs(2, 0), 17)
        self.assertAlmostEqual(s.get_zs(2, 1), 18)
        self.assertAlmostEqual(s.get_zs(3, -1), 19)
        self.assertAlmostEqual(s.get_zs(3, 0), 20)
        self.assertAlmostEqual(s.get_zs(3, 1), 21)

    def test_set_rc(self):
        s = SurfaceRZFourier()
        s.x = [2.9, -1.1, 0.7]
        s.set_rc(0, 0, 3.1)
        self.assertAlmostEqual(s.x[0], 3.1)

    def test_set_zs(self):
        s = SurfaceRZFourier()
        s.x = [2.9, -1.1, 0.7]
        s.set_zs(1, 0, 1.4)
        self.assertAlmostEqual(s.x[2], 1.4)

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

    @unittest.skip
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

    def test_vjps(self):
        mpol = 10
        ntor = 10
        nfp = 1
        s = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
        h = np.random.standard_normal(size=s.gamma().shape)

        via_vjp = s.dgamma_by_dcoeff_vjp(h)
        via_matvec = np.sum(s.dgamma_by_dcoeff()*h[..., None], axis=(0, 1, 2))
        assert np.linalg.norm(via_vjp-via_matvec)/np.linalg.norm(via_vjp) < 1e-13

        via_vjp = s.dgammadash1_by_dcoeff_vjp(h)
        via_matvec = np.sum(s.dgammadash1_by_dcoeff()*h[..., None], axis=(0, 1, 2))
        assert np.linalg.norm(via_vjp-via_matvec)/np.linalg.norm(via_vjp) < 1e-13

        via_vjp = s.dgammadash2_by_dcoeff_vjp(h)
        via_matvec = np.sum(s.dgammadash2_by_dcoeff()*h[..., None], axis=(0, 1, 2))
        assert np.linalg.norm(via_vjp-via_matvec)/np.linalg.norm(via_vjp) < 1e-13

    def test_names_order(self):
        """
        Verify that the order of rc, rs, zc, zs in the dof names is
        correct. This requires that the order of these four arrays in
        ``_make_names()`` matches the order in the C++ functions
        ``set_dofs_impl()`` and ``get_dofs()`` in
        ``src/simsoptpp/surfacerzfourier.h``.
        """
        mpol = 1
        ntor = 1
        nfp = 4
        s = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
        s.set_rc(0, 0, 100.0)
        s.set_zs(0, 1, 200.0)
        self.assertAlmostEqual(s.get('rc(0,0)'), 100.0)
        self.assertAlmostEqual(s.get('zs(0,1)'), 200.0)

        # Now try non-stellarator-symmetry
        s = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        s.set_rc(0, 0, 10.0)
        s.set_zs(0, 1, 20.0)
        s.set_zc(0, 0, 30.0)
        s.set_rs(0, 1, 40.0)
        self.assertAlmostEqual(s.get('rc(0,0)'), 10.0)
        self.assertAlmostEqual(s.get('zs(0,1)'), 20.0)
        self.assertAlmostEqual(s.get('zc(0,0)'), 30.0)
        self.assertAlmostEqual(s.get('rs(0,1)'), 40.0)

    def test_mn(self):
        """
        Test the arrays of mode numbers m and n.
        """
        mpol = 3
        ntor = 2
        nfp = 4
        s = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
        m_correct = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                     0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        n_correct = [0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2,
                     1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
        np.testing.assert_array_equal(s.m, m_correct)
        np.testing.assert_array_equal(s.n, n_correct)

        # Now try a non-stellarator-symmetric case
        s = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        np.testing.assert_array_equal(s.m, m_correct + m_correct)
        np.testing.assert_array_equal(s.n, n_correct + n_correct)

    def test_mn_matches_names(self):
        """
        Verify that the m and n attributes match the dof names.
        """
        mpol = 2
        ntor = 3
        nfp = 5
        surf = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
        # Drop the rc or zs from the start of the names:
        names = [s[2:] for s in surf.local_dof_names]
        names2 = [f'({m},{n})' for m, n in zip(surf.m, surf.n)]
        self.assertEqual(names, names2)

        # Now try a non-stellarator-symmetric case:
        surf = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        assert 'zc(0,0)' in surf.local_dof_names
        # Drop the rc or zs from the start of the names:
        names = [s[2:] for s in surf.local_dof_names]
        names2 = [f'({m},{n})' for m, n in zip(surf.m, surf.n)]
        self.assertEqual(names, names2)

    def test_serialization(self):
        """
        Test the serialization of an axisymmetric surface using area and volume
        """
        s = SurfaceRZFourier()
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[1, 0] = 0.2
        # TODO: x should be updated whenever rc and zs are modified without
        # TODO: explict setting of x
        s.local_full_x = s.get_dofs()

        surf_str = json.dumps(s, cls=MontyEncoder)
        s_regen = json.loads(surf_str, cls=MontyDecoder)

        self.assertAlmostEqual(s.area(), s_regen.area(), places=4)
        self.assertAlmostEqual(s.volume(), s_regen.volume(), places=3)


class SurfaceRZPseudospectralTests(unittest.TestCase):
    def test_names(self):
        """
        Check that dof names are correct.
        """
        surf = SurfaceRZPseudospectral(mpol=2, ntor=1, nfp=3)
        names = ['r(0,0)', 'r(0,1)', 'r(0,2)',
                 'r(1,0)', 'r(1,1)', 'r(1,2)', 'r(1,3)', 'r(1,4)',
                 'z(0,1)', 'z(0,2)',
                 'z(1,0)', 'z(1,1)', 'z(1,2)', 'z(1,3)', 'z(1,4)']
        self.assertEqual(surf.local_dof_names, names)

    def test_from_RZFourier(self):
        """
        Create a SurfaceRZPseudospectral object by converting a
        SurfaceRZFourier object, and make sure the real-space dofs
        have the correct values.
        """
        surf1 = SurfaceRZFourier(mpol=1, ntor=2, nfp=5)
        surf1.set('rc(0,0)', 2000.0)
        surf1.set('rc(1,0)', 30.0)
        surf1.set('zs(1,0)', 20.0)
        surf1.set('rc(0,1)', 50.0)
        surf1.set('zs(0,1)', 40.0)
        surf1.fix('zs(1,0)')  # The from_RZFourier function should work even if some dofs are fixed.
        surf2 = SurfaceRZPseudospectral.from_RZFourier(surf1, r_shift=2000.0, a_scale=100.0)
        #for name, x in zip(surf2.local_dof_names, surf2.x):
        #    print(name, x)
        theta = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        self.assertAlmostEqual(surf2.get('r(0,0)'), 0.3 + 0.5)
        self.assertAlmostEqual(surf2.get('r(0,1)'), 0.3 * np.cos(theta[1]) + 0.5)
        self.assertAlmostEqual(surf2.get('z(0,1)'), 0.2 * np.sin(theta[1]))
        nasserts = 3
        for jphi in range(1, 3):
            for jtheta in range(3):
                nasserts += 2
                self.assertAlmostEqual(surf2.get(f'r({jphi},{jtheta})'),
                                       0.3 * np.cos(theta[jtheta]) + 0.5 * np.cos(-phi[jphi]))
                self.assertAlmostEqual(surf2.get(f'z({jphi},{jtheta})'),
                                       0.2 * np.sin(theta[jtheta]) + 0.4 * np.sin(-phi[jphi]))
        assert nasserts == len(surf2.x)

    def test_complete_grid(self):
        """
        Make sure the _complete_grid() function of SurfaceRZPseudospectral
        returns values that agree with the gamma() function for a
        SurfaceRZFourier object describing the same shape.
        """
        filename = TEST_DIR / 'input.li383_low_res'
        s0 = SurfaceRZFourier.from_vmec_input(filename)
        s1 = SurfaceRZFourier.from_vmec_input(filename, range='field period',
                                              ntheta=2 * s0.mpol + 1,
                                              nphi=2 * s0.ntor + 1)
        gamma = s1.gamma()
        r1 = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
        z1 = gamma[:, :, 2]

        s2 = SurfaceRZPseudospectral.from_RZFourier(s1, r_shift=2.2, a_scale=0.4)
        r2, z2 = s2._complete_grid()

        np.testing.assert_allclose(r1, r2.T)
        np.testing.assert_allclose(z1, z2.T)

    def test_convert_back(self):
        """
        Start with a SurfaceRZFourier object, convert to a
        SurfaceRZPseudospectral object, and convert back to a new
        SurfaceRZFourier object. The dofs of the initial and final
        object should match.
        """
        filename = TEST_DIR / 'input.li383_low_res'
        s1 = SurfaceRZFourier.from_vmec_input(filename)
        #print('Original SurfaceRZFourier dofs:', s1.x)
        x1 = s1.x
        s2 = SurfaceRZPseudospectral.from_RZFourier(s1, r_shift=2.2, a_scale=0.4)
        s3 = s2.to_RZFourier()
        x3 = s3.x
        #for j, name in enumerate(s1.local_dof_names):
        #    print(name, x1[j], x3[j], x1[j] - x3[j])
        np.testing.assert_allclose(s1.full_x, s3.full_x)

    def test_change_resolution(self):
        """
        If we refine the resolution, then coarsen the grid back to the
        original resolution, the initial and final dofs should match.
        """
        filename = TEST_DIR / 'input.li383_low_res'
        s1 = SurfaceRZFourier.from_vmec_input(filename)
        s2 = SurfaceRZPseudospectral.from_RZFourier(s1, r_shift=2.2, a_scale=0.4)
        # Increase the resolution:
        s3 = s2.change_resolution(mpol=s1.mpol + 3, ntor=s1.ntor + 4)
        # Decrease the resolution back to where it was originally:
        s4 = s3.change_resolution(mpol=s1.mpol, ntor=s1.ntor)
        np.testing.assert_allclose(s2.x, s4.x)


if __name__ == "__main__":
    unittest.main()
