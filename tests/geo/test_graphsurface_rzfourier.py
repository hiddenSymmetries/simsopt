import unittest
from pathlib import Path

import numpy as np

from simsopt.geo.graph_surfacerzfourier import SurfaceRZFourier

TEST_DIR = Path(__file__).parent / ".." / "test_files"

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False


class SurfaceRZFourierTests(unittest.TestCase):

    # TODO: Unit test for cross section and make it consistent for all
    # TODO: surface classes
    def test_cross_section(self):
        self.fail("Test Not Implemented")

    # TODO: Make test_aspect_ratio consistent for all surface classes
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

        # Expand the test by checking DOF names

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

    # TODO: Implement the tests
    def test_to_RZFOurier(self):
        self.fail("Test Not Implemented")

    def test_repr(self):
        self.fail("Test Not Implemented")

    def test_get_rc(self):
        self.fail("Test Not Implemented")

    def test_get_rs(self):
        self.fail("Test Not Implemented")

    def test_get_zc(self):
        self.fail("Test Not Implemented")

    def test_get_zs(self):
        self.fail("Test Not Implemented")

    def test_set_rc(self):
        self.fail("Test Not Implemented")

    def test_set_rs(self):
        self.fail("Test Not Implemented")

    def test_set_zc(self):
        self.fail("Test Not Implemented")

    def test_set_zs(self):
        self.fail("Test Not Implemented")

    def test_fixed_range(self):
        self.fail("Test Not Implemented")

    def test_to_Garabedian(self):
        self.fail("Test Not Implemented")

    def recompute_bell(self):
        self.fail("Test Not Implemented")

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

    # TODO: Test the SurfaceRZFourier class w.r.t. graph optimizable framework

if __name__ == "__main__":
    unittest.main()
