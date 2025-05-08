import unittest
from pathlib import Path

from qsc import Qsc
import numpy as np
from monty.tempfile import ScratchDir

from simsopt import save, load
from simsopt.geo.surfacerzfourier import SurfaceRZFourier, SurfaceRZPseudospectral
from simsopt.geo.surface import Surface
from simsopt._core.optimizable import Optimizable

try:
    import vmec
except ImportError:
    vmec = None

from simsopt.mhd import Vmec

TEST_DIR = Path(__file__).parent / ".." / "test_files"

stellsym_list = [True, False]


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

    def test_shared_dof_init(self):
        s = SurfaceRZFourier()
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[1, 0] = 0.2
        s.local_full_x = s.get_dofs()

        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
            ntheta=31, nphi=30, range='field period')
        s2 = SurfaceRZFourier(quadpoints_phi=quadpoints_phi,
                              quadpoints_theta=quadpoints_theta,
                              dofs=s.dofs)
        self.assertIs(s.dofs, s2.dofs)
        true_area = 15.827322032265993
        true_volume = 2.0528777154265874
        self.assertAlmostEqual(s2.area(), true_area, places=4)
        self.assertAlmostEqual(s2.volume(), true_volume, places=3)

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

    def test_from_nescoil_input(self):
        """
        Test reading in surfaces from a NESCOIL input file.
        """

        filename = TEST_DIR / 'nescin.LandremanPaul2021_QA'
        s_plas = SurfaceRZFourier.from_nescoil_input(filename, 'plasma')
        SurfaceRZFourier.from_nescoil_input(filename, 'current')
        with self.assertRaises(ValueError):
            SurfaceRZFourier.from_nescoil_input(filename, 'other')

        # The plasma surface in the nescoil file should be approximately the
        # same as the LandremanPaul2021_QA surface, although Fourier resolution
        # is different
        filename_ref = TEST_DIR / 'input.LandremanPaul2021_QA'
        s_ref = SurfaceRZFourier.from_vmec_input(filename_ref)
        self.assertAlmostEqual(s_plas.volume(), s_ref.volume(), places=1)

        with self.assertRaises(AssertionError):
            SurfaceRZFourier.from_nescoil_input(filename_ref, 'plasma')

    def test_from_nescoil_input_distance(self):
        """
        Load a nescin file generated by regcoil that should have a uniform
        3.0 m separation from the plasma and confirm that the distance is as
        expected.
        """
        nescin_filename = TEST_DIR / 'nescin.LandremanPaul2021_QH_reactorScale_separation3'
        coil_surf = SurfaceRZFourier.from_nescoil_input(nescin_filename, "current", range="half period", ntheta=21, nphi=20)

        wout_filename = TEST_DIR / 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc'
        plasma_surf = SurfaceRZFourier.from_wout(wout_filename, range="full torus", ntheta=100, nphi=400)

        separation_vectors = coil_surf.gamma()[:, :, None, None, :] - plasma_surf.gamma()[None, None, :, :, :]
        distances = np.linalg.norm(separation_vectors, axis=-1)
        min_distances = np.min(distances, axis=(2, 3))
        np.testing.assert_allclose(min_distances, 3, rtol=1e-2)

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
        with ScratchDir("."):
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
        nml_str = s1.get_nml()  # This time, cover the case in which a string is returned
        with ScratchDir("."):
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
        self.assertAlmostEqual(s.area(), true_area, places=4)
        self.assertAlmostEqual(s.volume(), true_volume, places=3)

    def test_extend_via_normal(self):
        """
        Extend a surface using extend_via_normal(), and confirm that the
        distance between the old and new surfaces is indeed uniform. Also
        compare the new surface to a reference surface generated by the
        uniform-offset-surface feature in regcoil.
        """
        wout_filename = TEST_DIR / 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc'
        plasma_surf = SurfaceRZFourier.from_wout(wout_filename, range="full torus", ntheta=100, nphi=400)

        coil_surf = SurfaceRZFourier.from_wout(wout_filename, range="half period", ntheta=71, nphi=70)
        # Increase Fourier resolution so we can represent the surface accurately
        coil_surf.change_resolution(24, 24)
        separation = 3.0
        coil_surf.extend_via_normal(separation)

        # Make a copy of the surface with fewer quadrature points for testing
        coil_surf_test = SurfaceRZFourier.from_nphi_ntheta(
            mpol=coil_surf.mpol,
            ntor=coil_surf.ntor,
            range="half period",
            nphi=20,
            ntheta=21,
            nfp=coil_surf.nfp,
            stellsym=coil_surf.stellsym,
        )
        coil_surf_test.x = coil_surf.x

        separation_vectors = coil_surf_test.gamma()[:, :, None, None, :] - plasma_surf.gamma()[None, None, :, :, :]
        distances = np.linalg.norm(separation_vectors, axis=-1)
        min_distances = np.min(distances, axis=(2, 3))

        np.testing.assert_allclose(min_distances, separation, rtol=1e-2)

        # Compare to a uniform-separation surface calculated by regcoil:
        nescin_filename = TEST_DIR / 'nescin.LandremanPaul2021_QH_reactorScale_separation3'
        regcoil_surf = SurfaceRZFourier.from_nescoil_input(nescin_filename, "current", range="half period", ntheta=21, nphi=20)
        np.testing.assert_allclose(coil_surf_test.gamma(), regcoil_surf.gamma(), rtol=0.03)

    def test_extend_via_normal_non_stellsym(self):
        """Same as test_extend_via_normal but for non-stellarator-symmetric surfaces."""
        wout_filename = TEST_DIR / 'wout_LandremanSenguptaPlunk_section5p3_reference.nc'
        plasma_surf = SurfaceRZFourier.from_wout(wout_filename, range="full torus", ntheta=100, nphi=400)

        coil_surf = SurfaceRZFourier.from_wout(wout_filename, range="field period", ntheta=101, nphi=100)
        # Increase Fourier resolution so we can represent the surface accurately
        coil_surf.change_resolution(24, 24)
        separation = 0.2
        coil_surf.extend_via_normal(separation)

        # Make a copy of the surface with fewer quadrature points for testing
        coil_surf_test = SurfaceRZFourier.from_nphi_ntheta(
            mpol=coil_surf.mpol,
            ntor=coil_surf.ntor,
            range="field period",
            nphi=20,
            ntheta=21,
            nfp=coil_surf.nfp,
            stellsym=coil_surf.stellsym,
        )
        coil_surf_test.x = coil_surf.x

        separation_vectors = coil_surf_test.gamma()[:, :, None, None, :] - plasma_surf.gamma()[None, None, :, :, :]
        distances = np.linalg.norm(separation_vectors, axis=-1)
        min_distances = np.min(distances, axis=(2, 3))

        np.testing.assert_allclose(min_distances, separation, rtol=2e-3)

        # Compare to a uniform-separation surface calculated by regcoil:
        nescin_filename = TEST_DIR / 'nescin.LandremanSenguptaPlunk_section5p3_separation0p2'
        regcoil_surf = SurfaceRZFourier.from_nescoil_input(nescin_filename, "current", range="field period", ntheta=21, nphi=20)
        np.testing.assert_allclose(coil_surf_test.gamma(), regcoil_surf.gamma(), rtol=2e-4)

    def test_from_pyQSC(self):
        """
        Try reading in a near-axis pyQSC equilibrium.
        """
        stel = Qsc.from_paper("r1 section 5.1")
        filename = TEST_DIR / 'input.near_axis_test'

        ntheta = 20
        mpol = 10
        ntor = 10
        r = 0.1

        stel.to_vmec(filename, r=r, ntheta=ntheta, ntorMax=ntor, params={'mpol': mpol, 'ntor': ntor})

        s1 = SurfaceRZFourier.from_pyQSC(stel, r=r, ntheta=ntheta, ntor=ntor, mpol=mpol)
        s2 = SurfaceRZFourier.from_vmec_input(filename)

        np.testing.assert_allclose(s1.rc, s2.rc)
        np.testing.assert_allclose(s1.zs, s2.zs)
        np.testing.assert_allclose(s1.nfp, s2.nfp)
        np.testing.assert_allclose(s1.stellsym, s2.stellsym)
        np.testing.assert_allclose(s1.area(), s2.area())
        np.testing.assert_allclose(s1.volume(), s2.volume())

        # test possible bug due to memory leak
        # stell sym
        from simsopt.configs import get_ncsx_data
        _, _, ma = get_ncsx_data()
        qsc = Qsc(ma.rc, np.insert(ma.zs, 0, 0), nfp=3, etabar=-0.408)
        phis = np.linspace(0, 1/qsc.nfp, 2*ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        full_torus = SurfaceRZFourier.from_pyQSC(qsc, r=0.1, ntheta=100, mpol=6, ntor=6)
        full_period = SurfaceRZFourier(mpol=full_torus.mpol, ntor=full_torus.ntor, stellsym=full_torus.stellsym, nfp=full_torus.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        full_period.x = full_torus.x

        np.testing.assert_allclose(full_torus.rc, full_period.rc)
        np.testing.assert_allclose(full_torus.zs, full_period.zs)

        np.random.seed(1)
        # non stell sym for code coverage
        qsc = Qsc(ma.rc, np.insert(ma.zs, 0, 0), rs=np.random.rand(5)*1e-7, zc=np.random.rand(5)*1e-7, nfp=3, etabar=-0.408)
        phis = np.linspace(0, 1/qsc.nfp, 2*ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        full_torus = SurfaceRZFourier.from_pyQSC(qsc, r=0.1, ntheta=100, mpol=6, ntor=6)
        full_period = SurfaceRZFourier(mpol=full_torus.mpol, ntor=full_torus.ntor, stellsym=full_torus.stellsym, nfp=full_torus.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        full_period.x = full_torus.x

        np.testing.assert_allclose(full_torus.rc, full_period.rc)
        np.testing.assert_allclose(full_torus.zs, full_period.zs)

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

        surf_str = s.save(fmt="json")
        s_regen = Optimizable.from_str(surf_str)

        self.assertAlmostEqual(s.area(), s_regen.area(), places=4)
        self.assertAlmostEqual(s.volume(), s_regen.volume(), places=3)

    def test_shared_dof_serialization(self):
        import tempfile
        from pathlib import Path

        s = SurfaceRZFourier()
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[1, 0] = 0.2
        s.local_full_x = s.get_dofs()
        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
            ntheta=31, nphi=30, range='field period')
        s2 = SurfaceRZFourier(quadpoints_phi=quadpoints_phi,
                              quadpoints_theta=quadpoints_theta,
                              dofs=s.dofs)

        self.assertAlmostEqual(s.volume(), s2.volume())

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "surf.json"
            save([s, s2], fpath)

            surf_objs = load(fpath)
            self.assertAlmostEqual(s.volume(), surf_objs[0].volume())
            self.assertAlmostEqual(s.area(), surf_objs[0].area())
            self.assertAlmostEqual(s2.volume(), surf_objs[1].volume())
            self.assertAlmostEqual(s2.area(), surf_objs[1].area())
            self.assertIs(surf_objs[0].dofs, surf_objs[1].dofs)

    def test_make_rotating_ellipse(self):
        major_radius = 8.4
        minor_radius = 2.3
        elongation = 2.7
        torsion = 0.6
        nfp = 3
        sqrt_elong = np.sqrt(elongation)
        surf = SurfaceRZFourier.from_nphi_ntheta(ntor=2, mpol=3, nphi=2, ntheta=4, range="field period", nfp=nfp)
        surf.make_rotating_ellipse(major_radius, minor_radius, elongation, torsion)

        xyz = surf.gamma()
        R = np.sqrt(xyz[:, :, 0]**2 + xyz[:, :, 1]**2)
        Z = xyz[:, :, 2]

        # Check phi=0 plane:
        np.testing.assert_allclose(
            R[0, :],
            [major_radius + torsion + minor_radius / sqrt_elong,
             major_radius + torsion,
             major_radius + torsion - minor_radius / sqrt_elong,
             major_radius + torsion]
        )
        np.testing.assert_allclose(
            Z[0, :],
            [0,
             minor_radius * sqrt_elong,
             0,
             -minor_radius * sqrt_elong],
            atol=1e-14,
        )

        # Check phi=pi/nfp plane:
        np.testing.assert_allclose(
            R[1, :],
            [major_radius - torsion + minor_radius * sqrt_elong,
             major_radius - torsion,
             major_radius - torsion - minor_radius * sqrt_elong,
             major_radius - torsion]
        )
        np.testing.assert_allclose(
            Z[1, :],
            [0,
             minor_radius / sqrt_elong,
             0,
             -minor_radius / sqrt_elong],
            atol=1e-14,
        )

        # Now make the same surface shape with more quadpoints:
        surf = SurfaceRZFourier.from_nphi_ntheta(ntor=1, mpol=1, nphi=64, ntheta=65, range="field period", nfp=nfp)
        surf.make_rotating_ellipse(major_radius, minor_radius, elongation, torsion)
        np.testing.assert_allclose(surf.major_radius(), major_radius)
        np.testing.assert_allclose(surf.minor_radius(), minor_radius)
        np.testing.assert_allclose(surf.aspect_ratio(), major_radius / minor_radius)

        # Check that the cross-sectional area is correct at every phi:
        gamma = surf.gamma()
        R = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
        gammadash2 = surf.gammadash2()
        dZdtheta = gammadash2[:, :, 2]
        dtheta = surf.quadpoints_theta[1] - surf.quadpoints_theta[0]
        area_vs_phi = np.abs(np.sum(R * dZdtheta, axis=1) * dtheta)
        np.testing.assert_allclose(area_vs_phi, np.pi * minor_radius**2)

    @unittest.skipIf(vmec is None, "vmec python extension is not installed")
    def test_make_rotating_ellipse_iota(self):
        """make_rotating_ellipse() should give positive iota."""
        filename = str(TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres')
        with ScratchDir("."):
            eq = Vmec(filename)
            eq.indata.mpol = 4  # Lower resolution to expedite test
            eq.indata.ntor = 4
            eq.indata.ftol_array[:2] = [1e-8, 1e-10]

            # Try the case of elongation=1 with positive axis torsion:
            major_radius = 8.4
            minor_radius = 1.3
            elongation = 1.0
            torsion = 0.9
            eq.boundary.make_rotating_ellipse(major_radius, minor_radius, elongation, torsion)
            eq.run()
            np.testing.assert_array_less(0, eq.wout.iotaf)
            np.testing.assert_allclose(eq.mean_iota(), 0.26990720954583547, rtol=1e-6)

            # Try the case of zero axis torsion with rotating elongation:
            major_radius = 8.4
            minor_radius = 1.3
            elongation = 2.1
            torsion = 0.0
            eq.boundary.make_rotating_ellipse(major_radius, minor_radius, elongation, torsion)
            eq.run()
            np.testing.assert_array_less(0, eq.wout.iotaf)
            np.testing.assert_allclose(eq.mean_iota(), 0.4291137962772453, rtol=1e-6)

    def test_fourier_transform_scalar(self):
        """
        Test the Fourier transform of a field on a surface.
        """
        s = SurfaceRZFourier(mpol=4, ntor=5)
        s.rc[0, 0] = 1.3
        s.rc[1, 0] = 0.4
        s.zs[1, 0] = 0.2

        # Create the grid of quadpoints:
        phi2d, theta2d = np.meshgrid(2 * np.pi * s.quadpoints_phi,
                                     2 * np.pi * s.quadpoints_theta,
                                     indexing='ij')

        # create a test field where only Fourier elements [m=2, n=3]
        # and [m=4,n=5] are nonzero:
        field = 0.8 * np.sin(2*theta2d - 3*s.nfp*phi2d) + 0.2*np.sin(4*theta2d - 5*s.nfp*phi2d) + 0.7*np.cos(3*theta2d - 3*s.nfp*phi2d)

        # Transform the field to Fourier space:
        ft_sines, ft_cosines = s.fourier_transform_scalar(field, stellsym=False)
        self.assertAlmostEqual(ft_sines[2, 3+s.ntor], 0.8)
        self.assertAlmostEqual(ft_sines[4, 5+s.ntor], 0.2)
        self.assertAlmostEqual(ft_cosines[3, 3+s.ntor], 0.7)

        # Test that all other elements are close to zero
        sines_mask = np.ones_like(ft_sines, dtype=bool)
        cosines_mask = np.copy(sines_mask)
        sines_mask[2, 3 + s.ntor] = False
        sines_mask[4, 5 + s.ntor] = False
        cosines_mask[3, 3 + s.ntor] = False
        self.assertEqual(np.all(np.abs(ft_sines[sines_mask]) < 1e-10), True)
        self.assertEqual(np.all(np.abs(ft_cosines[cosines_mask]) < 1e-10), True)

        # Transform back to real space:
        field2 = s.inverse_fourier_transform_scalar(ft_sines, ft_cosines, stellsym=False, normalization=1/2*np.pi**2)

        # Check that the result is the same as the original field:
        np.testing.assert_allclose(field/2*np.pi**2, field2)

        # Do a similar transform+inverse tests for a case with
        # modes aliasing to 0 and for a diffrent normalization
        # and for a stellarator non-symmetric surface.
        normalization = 1/(np.sqrt(2) * np.pi)
        nphi = 10
        ntheta = 12
        quadpoints_phi = np.linspace(0, 1, nphi, endpoint=False)
        quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=False)
        mpol = int(ntheta//2)
        ntor = int(nphi//2)
        s = SurfaceRZFourier(mpol=mpol, ntor=ntor,
                             quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta, stellsym=False)
        s.rc[0, 0] = 1.0
        s.rs[1, 0] = 0.3
        s.zc[0, 1] = 0.1
        phi2d, theta2d = np.meshgrid(2 * np.pi * s.quadpoints_phi,
                                     2 * np.pi * s.quadpoints_theta,
                                     indexing='ij')
        field = 0.6 * np.sin(4*theta2d - 1*s.nfp*phi2d) + 0.4*np.sin(5 *
                                                                     theta2d - 2*s.nfp*phi2d) + 0.5*np.cos(5*theta2d + 2*s.nfp*phi2d)
        ft_sines, ft_cosines = s.fourier_transform_scalar(
            field, stellsym=False, normalization=normalization)
        field2 = s.inverse_fourier_transform_scalar(
            ft_sines, ft_cosines, stellsym=False, normalization=normalization)
        np.testing.assert_allclose(field, field2, err_msg = 'Fourier transform + inverse transform does not give the original field.', atol=1e-13)
        
    def test_copy_method(self):
        """
        Tests the copy method under various conditions.
        """
        s = SurfaceRZFourier(mpol=4, ntor=5, nfp=3)
        s2 = s.copy(quadpoints_phi=Surface.get_phi_quadpoints(nphi=100, range='field period'))
        self.assertEqual(len(s2.quadpoints_phi), 100)
        s3 = s.copy(quadpoints_theta=Surface.get_theta_quadpoints(ntheta=50))
        self.assertEqual(len(s3.quadpoints_theta), 50)
        s4 = s.copy(ntheta=42)
        self.assertEqual(len(s4.quadpoints_theta), 42)
        s5 = s.copy(nphi=17)
        self.assertEqual(len(s5.quadpoints_phi), 17)
        s6 = s.copy(range='field period')
        self.assertEqual(s6.deduced_range, Surface.RANGE_FIELD_PERIOD)
        s7 = s.copy(nfp=10)
        self.assertEqual(s7.nfp, 10)
        s8 = s.copy(mpol=5, ntor=6)
        self.assertEqual(s8.mpol, 5)
        self.assertEqual(s8.ntor, 6)
        s.copy()
        s.copy(quadpoints_phi=Surface.get_phi_quadpoints(nphi=100, range='field period'), ntheta=82)

        # Making a stellarator non-symmetric copy
        # and setting a non-symmetric mode to something non-zero
        s9 = s.copy(stellsym=False)
        np.testing.assert_allclose(s9.gamma(), s.gamma(),
                                   err_msg='Copied surface is not close to original surface when the copy has stellsym=False.')
        s9.set('rs(1,0)', s9.get('rc(0,0)') * 0.01)
        # Copy the newly-created stellarator non-symemtric surface
        s10 = s9.copy()
        np.testing.assert_allclose(
            s10.x, s9.x, err_msg='Copying surface is broken for stellarator symmetric surfaces!')
        
    def test_fixed_range(self):
        """
        Test that DOFs are fixed correctly by invoking fixed_range().
        For stellarator non-symmetric surfaces. Fixed/unfixed invoked one after the other.
        """

        # These are the results we expect
        expected_local_dof_names1 = ['rc(1,-5)', 'rc(2,-5)', 'rc(3,-5)', 'rc(4,-5)', 'rc(4,-4)', 'rc(4,-3)', 'rc(4,-2)', 'rc(4,-1)', 'rc(4,0)', 'rc(4,1)', 'rc(4,2)', 'rc(4,3)', 'rc(4,4)', 'rc(4,5)', 'rs(1,-5)', 'rs(2,-5)', 'rs(3,-5)', 'rs(4,-5)', 'rs(4,-4)', 'rs(4,-3)', 'rs(4,-2)', 'rs(4,-1)', 'rs(4,0)', 'rs(4,1)', 'rs(4,2)', 'rs(4,3)', 'rs(4,4)', 'rs(4,5)',
                               'zc(1,-5)', 'zc(2,-5)', 'zc(3,-5)', 'zc(4,-5)', 'zc(4,-4)', 'zc(4,-3)', 'zc(4,-2)', 'zc(4,-1)', 'zc(4,0)', 'zc(4,1)', 'zc(4,2)', 'zc(4,3)', 'zc(4,4)', 'zc(4,5)', 'zs(1,-5)', 'zs(2,-5)', 'zs(3,-5)', 'zs(4,-5)', 'zs(4,-4)', 'zs(4,-3)', 'zs(4,-2)', 'zs(4,-1)', 'zs(4,0)', 'zs(4,1)', 'zs(4,2)', 'zs(4,3)', 'zs(4,4)', 'zs(4,5)']
        expected_local_dof_names2 = ['rc(1,-5)', 'rc(2,-5)', 'rc(2,5)', 'rc(3,-5)', 'rc(3,5)', 'rc(4,-5)', 'rc(4,-4)', 'rc(4,-3)', 'rc(4,-2)', 'rc(4,-1)', 'rc(4,0)', 'rc(4,1)', 'rc(4,2)', 'rc(4,3)', 'rc(4,4)', 'rc(4,5)', 'rs(1,-5)', 'rs(2,-5)', 'rs(2,5)', 'rs(3,-5)', 'rs(3,5)', 'rs(4,-5)', 'rs(4,-4)', 'rs(4,-3)', 'rs(4,-2)', 'rs(4,-1)', 'rs(4,0)', 'rs(4,1)', 'rs(4,2)', 'rs(4,3)', 'rs(4,4)', 'rs(4,5)',
                               'zc(1,-5)', 'zc(2,-5)', 'zc(2,5)', 'zc(3,-5)', 'zc(3,5)', 'zc(4,-5)', 'zc(4,-4)', 'zc(4,-3)', 'zc(4,-2)', 'zc(4,-1)', 'zc(4,0)', 'zc(4,1)', 'zc(4,2)', 'zc(4,3)', 'zc(4,4)', 'zc(4,5)', 'zs(1,-5)', 'zs(2,-5)', 'zs(2,5)', 'zs(3,-5)', 'zs(3,5)', 'zs(4,-5)', 'zs(4,-4)', 'zs(4,-3)', 'zs(4,-2)', 'zs(4,-1)', 'zs(4,0)', 'zs(4,1)', 'zs(4,2)', 'zs(4,3)', 'zs(4,4)', 'zs(4,5)']

        s = SurfaceRZFourier(mpol=4, ntor=5, stellsym=False)
        s.fixed_range(mmin=0, mmax=3, nmin=-4, nmax=5, fixed=True)
        self.assertEqual(s.local_dof_names, expected_local_dof_names1,
                         msg='dof_names not the expected ones after fixed_range(..., fixed=True)')
        s.fixed_range(mmin=2, mmax=3, nmin=5, nmax=5, fixed=False)
        self.assertEqual(s.local_dof_names, expected_local_dof_names2,
                         msg='dof_names not the expected ones after fixed_range(..., fixed=False)')

    def test_area_derivative(self):
        """
        This is purely a regression test to check that the area calculation
        against a precalculated result of d(area)/d(coef).
        """
        da_truth = np.array([9.02414352e+01, 2.50276774e+01, 4.98898579e+00, 1.10959697e+00,
                             -8.63569330e-01, 1.26121052e-01, 2.77952703e+00, -1.05465885e+00,
                             -1.61497613e+01, 1.11297192e+02, -4.71965838e+01, 3.09681218e+00,
                             -1.71101865e+00, -1.00224411e+00, 1.34844600e+00, -1.52610705e+00,
                             -5.52718153e+00, 2.58202898e+01, 1.35398524e+02, 1.06702325e+02,
                             3.18041581e+01, -8.90876955e+00, -5.34029533e-01, -4.83479972e-01,
                             -8.73020447e-01, 6.53690772e+00, 1.28624879e+01, -5.44077196e+01,
                             -6.83727070e+01, -1.39857033e+01, -2.17980284e+01, 7.42718812e+00,
                             2.61172931e-02, 5.04037512e-02, -2.56310851e+00, -1.18730770e+01,
                             4.83621541e+00, -2.14978837e+01, 2.73169154e+01, -1.24407988e+01,
                             1.46774570e+01, 4.73746270e-01, -5.11348115e-01, -3.22309157e+00,
                             9.56832718e+00, 2.82703399e+00, 8.29792054e+01, 3.21948501e+01,
                             -2.64810625e+01, 1.25719947e+01,  3.45331822e-01, -1.02198905e+00,
                             4.33865224e+00, 1.95050439e+00, 9.77670854e+00, -4.17357485e+01,
                             2.61524211e+01, -1.72745430e+01, 2.56731758e+01, 7.94305341e+00,
                             -3.56069518e+00, 5.21691648e-02, 2.89563751e-01, -1.28621348e-01,
                             3.66936658e+00, -5.98842663e-01, -5.02913209e+00, 1.54267154e+02,
                             3.95922804e+01, -8.60225489e-02, -1.11027491e+00, 1.24241501e+00,
                             9.92604671e-01, -5.96641870e-01, -5.02788624e+00, 2.43837837e+01,
                             4.95036805e+01, -1.18847347e+01, -3.16279611e+01, 1.04794546e+01,
                             2.65277404e+00, -1.40633022e-01, -1.79828675e+00, 4.99622353e+00,
                             -2.59174379e-01, 1.18153604e+01, -5.48481430e+01, 3.60391998e+01,
                             2.27041674e+01, -9.00280999e+00, -5.16219267e-01, 1.47111391e+00,
                             -1.09374188e+00, -3.07314487e+00, -6.58441453e+00, 2.80439130e+01,
                             -1.13012765e+00, -1.18300363e+01, -1.80101972e+01, 6.36464701e-01,
                             -4.60409125e-02, -3.19459713e+00,  1.09220102e+01, -8.75154081e+00,
                             8.69977766e+01, 2.43203543e+01, -2.04318690e+01, -2.73848167e+00,
                             3.12269398e-01, -1.34844406e+00,  3.72001576e+00, -9.54737287e-01,
                             7.81856143e+00, -5.64189729e+01, -9.48423585e+00, -2.85529217e+01,
                             2.08250069e+01])

        filename = TEST_DIR / 'input.n3are_R7.75B5.7_lowres'
        s = SurfaceRZFourier.from_vmec_input(filename)
        da = s.darea()
        np.testing.assert_allclose(da, da_truth,
                                   err_msg = 'Area derivative does not match precalculated results.', atol = 1e-14)
        

    def test_volume_derivative(self):
        """
        This is purely a regression test to check the volume calculation
        against a precalculated result of d(volume)/d(coef).
        """

        dv_truth = [ 1.72604774e+01,  9.19649738e-02, -1.11881835e-01,  1.28519817e+01,
                     1.22407186e+02,  2.33393247e+02,  1.04371757e+01, -2.74799395e-01,
                     -1.62786886e+01,  1.07452160e+02,  4.20531210e+01,  1.32844875e+00,
                     -1.77554183e-02,  3.26981158e-15,  8.72919621e-16, -1.20583866e+01,
                     -1.01535638e+02,  2.01312334e+02,  9.53758104e+00, -3.05710996e-01,
                     2.08573955e+01,  9.95361447e+01,  4.07908776e+01,  1.33172546e+00,
                     -1.43207960e-02]
        filename = TEST_DIR / 'input.NuhrenbergZille_1988_QHS'
        s = SurfaceRZFourier.from_vmec_input(filename)
        dv = s.dvolume()
        np.testing.assert_allclose(dv, dv_truth,
                                   err_msg = 'Volume derivative does not match precalculated results.', atol = 1e-14)
        
        
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
        s2 = SurfaceRZPseudospectral.from_RZFourier(s1, r_shift=2.2, a_scale=0.4)
        s3 = s2.to_RZFourier()
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
