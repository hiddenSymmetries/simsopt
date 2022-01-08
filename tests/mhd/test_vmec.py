import unittest
import logging
import os
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    import vmec
    vmec_found = True
except ImportError:
    vmec_found = False

from simsopt.objectives.graph_least_squares import LeastSquaresProblem
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

from simsopt.mhd.vmec import Vmec

from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)


class InitializedFromWout(unittest.TestCase):
    def test_vacuum_well(self):
        """
        Test the calculation of magnetic well. This is done by comparison
        to a high-aspect-ratio configuration, in which case the
        magnetic well can be computed analytically by the method in
        Landreman & Jorge, J Plasma Phys 86, 905860510 (2020). The
        specific configuration considered is the one of section 5.4 in
        Landreman & Sengupta, J Plasma Phys 85, 815850601 (2019), also
        considered in the 2020 paper. We increase the mean field B0 to
        2T in that configuration to make sure all the factors of B0
        are correct.

        """
        filename = os.path.join(TEST_DIR, 'wout_LandremanSengupta2019_section5.4_B2_A80_reference.nc')
        vmec = Vmec(filename)

        well_vmec = vmec.vacuum_well()
        # Let psi be the toroidal flux divided by (2 pi)
        abs_psi_a = np.abs(vmec.wout.phi[-1]) / (2 * np.pi)

        # Data for this configuration from the near-axis construction code qsc:
        # https://github.com/landreman/pyQSC
        # or
        # https://github.com/landreman/qsc
        B0 = 2.0
        G0 = 2.401752071286676
        d2_volume_d_psi2 = 25.3041656424299

        # See also "20210504-01 Computing magnetic well from VMEC.docx" by MJL
        well_analytic = -abs_psi_a * B0 * B0 * d2_volume_d_psi2 / (4 * np.pi * np.pi * np.abs(G0))

        logger.info(f'well_vmec: {well_vmec}  well_analytic: {well_analytic}')
        np.testing.assert_allclose(well_vmec, well_analytic, rtol=2e-2, atol=0)

    def test_iota(self):
        """
        Test the functions related to iota.
        """
        filename = os.path.join(TEST_DIR, 'wout_LandremanSengupta2019_section5.4_B2_A80_reference.nc')
        vmec = Vmec(filename)

        iota_axis = vmec.iota_axis()
        iota_edge = vmec.iota_edge()
        mean_iota = vmec.mean_iota()
        mean_shear = vmec.mean_shear()
        # These next 2 lines are different ways the mean iota and
        # shear could be defined. They are not mathematically
        # identical to mean_iota() and mean_shear(), but they should
        # be close.
        mean_iota_alt = (iota_axis + iota_edge) * 0.5
        mean_shear_alt = iota_edge - iota_axis
        logger.info(f"iota_axis: {iota_axis}, iota_edge: {iota_edge}")
        logger.info(f"    mean_iota: {mean_iota},     mean_shear: {mean_shear}")
        logger.info(f"mean_iota_alt: {mean_iota_alt}, mean_shear_alt: {mean_shear_alt}")

        self.assertAlmostEqual(mean_iota, mean_iota_alt, places=3)
        self.assertAlmostEqual(mean_shear, mean_shear_alt, places=3)

    def test_error_on_rerun(self):
        """
        If a vmec object is initialized from a wout file, and if the dofs
        are then changed, vmec output functions should raise an
        exception.
        """
        filename = os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc')
        vmec = Vmec(filename)
        iota = vmec.mean_iota()
        vmec.boundary.set_rc(1, 0, 2.0)
        with self.assertRaises(RuntimeError):
            iota2 = vmec.mean_iota()


@unittest.skipIf((MPI is not None) and (vmec_found), "Interface to MPI and VMEC found")
class VmecTestsWithoutMPIorvmec(unittest.TestCase):
    def test_runnable_raises(self):
        """
        Test that RuntimeError is raised if Vmec initialized from
        input file without vmec or MPI
        """
        from simsopt.mhd.vmec import Vmec
        with self.assertRaises(RuntimeError):
            v = Vmec()


@unittest.skipIf((MPI is None) or (not vmec_found), "Valid Python interface to VMEC not found")
class VmecTests(unittest.TestCase):
    def test_init_defaults(self):
        """
        Just create a Vmec instance using the standard constructor,
        and make sure we can read some of the attributes.
        """
        v = Vmec()
        self.assertEqual(v.indata.nfp, 5)
        self.assertFalse(v.indata.lasym)
        self.assertEqual(v.indata.mpol, 5)
        self.assertEqual(v.indata.ntor, 4)
        self.assertEqual(v.indata.delt, 0.5)
        self.assertEqual(v.indata.tcon0, 2.0)
        self.assertEqual(v.indata.phiedge, 1.0)
        self.assertEqual(v.indata.curtor, 0.0)
        self.assertEqual(v.indata.gamma, 0.0)
        self.assertEqual(v.indata.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)

    def test_init_from_file(self):
        """
        Try creating a Vmec instance from a specified input file.
        """

        filename = os.path.join(TEST_DIR, 'input.li383_low_res')

        v = Vmec(filename)
        self.assertEqual(v.indata.nfp, 3)
        self.assertEqual(v.indata.mpol, 4)
        self.assertEqual(v.indata.ntor, 3)
        self.assertEqual(v.boundary.mpol, 4)
        self.assertEqual(v.boundary.ntor, 3)

        # n = 0, m = 0:
        self.assertAlmostEqual(v.boundary.get_rc(0, 0), 1.3782)

        # n = 0, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 0), 4.6465E-01)

        # n = 1, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 1), 1.6516E-01)

        self.assertEqual(v.indata.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)

    def test_surface_4_ways(self):
        """
        If we initialize a Vmec object, the boundary surface object should
        be (almost) the same as if we initialize a SurfaceRZFourier
        using SurfaceRZFourier.from_wout() or
        SurfaceRZFourier.from_vmec_input(). The possible difference is
        that mpol, ntor, and the quadrature points may be different.
        """

        def compare_surfaces_sym(s1, s2):
            logger.debug('compare_surfaces_sym called')
            mpol = min(s1.mpol, s2.mpol)
            ntor = min(s1.ntor, s2.ntor)
            places = 13
            for m in range(mpol + 1):
                nmin = 0 if m == 0 else -ntor
                for n in range(nmin, ntor + 1):
                    self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n), places=places)
                    self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n), places=places)

        def compare_surfaces_asym(s1, s2, places):
            logger.debug('compare_surfaces_asym called')
            self.assertAlmostEqual(np.abs(s1.volume()), np.abs(s2.volume()), places=13)
            self.assertAlmostEqual(s1.area(), s2.area(), places=7)
            mpol = min(s1.mpol, s2.mpol)
            ntor = min(s1.ntor, s2.ntor)
            for m in range(mpol + 1):
                nmin = 0 if m == 0 else -ntor
                for n in range(nmin, ntor + 1):
                    self.assertAlmostEqual(s1.get_rc(m, n), s2.get_rc(m, n), places=places)
                    self.assertAlmostEqual(s1.get_zs(m, n), s2.get_zs(m, n), places=places)
                    self.assertAlmostEqual(s1.get_rs(m, n), s2.get_rs(m, n), places=places)
                    self.assertAlmostEqual(s1.get_zc(m, n), s2.get_zc(m, n), places=places)

        # First try a stellarator-symmetric example:
        filename1 = os.path.join(TEST_DIR, 'input.li383_low_res')
        filename2 = os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc')
        v = Vmec(filename1)
        s1 = v.boundary
        # Compare initializing a Vmec object from an input file vs from a wout file:
        v2 = Vmec(filename2)
        s2 = v2.boundary
        compare_surfaces_sym(s1, s2)
        # Compare to initializing a surface using from_wout()
        s2 = SurfaceRZFourier.from_wout(filename2)
        compare_surfaces_sym(s1, s2)
        # Now try from_vmec_input() instead of from_wout():
        s2 = SurfaceRZFourier.from_vmec_input(filename1)
        compare_surfaces_sym(s1, s2)

        # Now try a non-stellarator-symmetric example.
        # For non-stellarator-symmetric cases, we must be careful when
        # directly comparing the rc/zs/rs/zc coefficients, because
        # VMEC shifts the poloidal angle in readin.f upon loading the
        # file. Moreover, in versions of VMEC other than the
        # hiddenSymmetries python module, the shift to theta may have
        # a bug so the angle shift is not by the claimed value. The
        # specific input file used here has a boundary that should not
        # be shifted by the hiddenSymmetries VMEC2000 module.  For any
        # input file and version of VMEC, we can compare
        # coordinate-independent properties like the volume.
        filename1 = os.path.join(TEST_DIR, 'input.LandremanSenguptaPlunk_section5p3')
        filename2 = os.path.join(TEST_DIR, 'wout_LandremanSenguptaPlunk_section5p3_reference.nc')
        v = Vmec(filename1)
        s1 = v.boundary

        # Compare initializing a Vmec object from an input file vs from a wout file:
        v2 = Vmec(filename2)
        s2 = v2.boundary
        compare_surfaces_asym(s1, s2, 13)

        s2 = SurfaceRZFourier.from_wout(filename2)
        compare_surfaces_asym(s1, s2, 13)

        s2 = SurfaceRZFourier.from_vmec_input(filename1)
        compare_surfaces_asym(s1, s2, 13)

    def test_2_init_methods(self):
        """
        If we initialize a Vmec object from an input file, or initialize a
        Vmec object from the corresponding wout file, physics quantities
        should be the same.
        """
        for jsym in range(2):
            if jsym == 0:
                # Try a stellarator-symmetric scenario:
                filename1 = os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc')
                filename2 = os.path.join(TEST_DIR, 'input.li383_low_res')
            else:
                # Try a non-stellarator-symmetric scenario:
                filename1 = os.path.join(TEST_DIR, 'wout_LandremanSenguptaPlunk_section5p3_reference.nc')
                filename2 = os.path.join(TEST_DIR, 'input.LandremanSenguptaPlunk_section5p3')

            vmec1 = Vmec(filename1)
            iota1 = vmec1.wout.iotaf
            bmnc1 = vmec1.wout.bmnc

            vmec2 = Vmec(filename2)
            vmec2.run()
            iota2 = vmec2.wout.iotaf
            bmnc2 = vmec2.wout.bmnc

            np.testing.assert_allclose(iota1, iota2, atol=1e-10)
            np.testing.assert_allclose(bmnc1, bmnc2, atol=1e-10)

    def test_vmec_failure(self):
        """
        Verify that failures of VMEC are correctly caught and represented
        by large values of the objective function.
        """
        for j in range(2):
            filename = os.path.join(TEST_DIR, 'input.li383_low_res')
            vmec = Vmec(filename)
            # Use the objective function from
            # stellopt_scenarios_2DOF_targetIotaAndVolume:
            if j == 0:
                prob = LeastSquaresProblem.from_tuples(
                    [(vmec.iota_axis, 0.41, 1),
                     (vmec.volume, 0.15, 1)])
                fail_val = 1e12
            else:
                # Try a custom failure value
                fail_val = 2.0e30
                prob = LeastSquaresProblem.from_tuples(
                    [(vmec.iota_axis, 0.41, 1),
                     (vmec.volume, 0.15, 1)],
                    fail=fail_val)

            r00 = vmec.boundary.get_rc(0, 0)
            # The first evaluation should succeed.
            # f = prob.f()
            f = prob.residuals()
            print(f[0], f[1])
            correct_f = [-0.004577338528148067, 2.8313872701632925]
            # Don't worry too much about accuracy here.
            np.testing.assert_allclose(f, correct_f, rtol=0.1)

            # Now set a crazy boundary shape to make VMEC fail. This
            # boundary causes VMEC to hit the max number of iterations
            # without meeting ftol.
            vmec.boundary.set_rc(0, 0, 0.2)
            vmec.need_to_run_code = True
            f = prob.residuals()
            print(f)
            np.testing.assert_allclose(f, np.full(2, fail_val))

            # Restore a reasonable boundary shape. VMEC should work again.
            vmec.boundary.set_rc(0, 0, r00)
            vmec.need_to_run_code = True
            f = prob.residuals()
            print(f)
            np.testing.assert_allclose(f, correct_f, rtol=0.1)

            # Now set a self-intersecting boundary shape. This causes VMEC
            # to fail with "ARNORM OR AZNORM EQUAL ZERO IN BCOVAR" before
            # it even starts iterating.
            orig_mode = vmec.boundary.get_rc(1, 3)
            vmec.boundary.set_rc(1, 3, 0.5)
            vmec.need_to_run_code = True
            f = prob.residuals()
            print(f)
            np.testing.assert_allclose(f, np.full(2, fail_val))

            # Restore a reasonable boundary shape. VMEC should work again.
            vmec.boundary.set_rc(1, 3, orig_mode)
            vmec.need_to_run_code = True
            f = prob.residuals()
            print(f)
            np.testing.assert_allclose(f, correct_f, rtol=0.1)

    #def test_stellopt_scenarios_1DOF_circularCrossSection_varyR0_targetVolume(self):
        """
        This script implements the "1DOF_circularCrossSection_varyR0_targetVolume"
        example from
        https://github.com/landreman/stellopt_scenarios

        This optimization problem has one independent variable, representing
        the mean major radius. The problem also has one objective: the plasma
        volume. There is not actually any need to run an equilibrium code like
        VMEC since the objective function can be computed directly from the
        boundary shape. But this problem is a fast way to test the
        optimization infrastructure with VMEC.

        Details of the optimum and a plot of the objective function landscape
        can be found here:
        https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyR0_targetVolume
        """


"""
        # Start with a default surface, which is axisymmetric with major
        # radius 1 and minor radius 0.1.
        equil = Vmec()
        surf = equil.boundary

        # Set the initial boundary shape. Here is one way to do it:
        surf.set('rc(0,0)', 1.0)
        # Here is another syntax that works:
        surf.set_rc(0, 1, 0.1)
        surf.set_zs(0, 1, 0.1)

        surf.set_rc(1, 0, 0.1)
        surf.set_zs(1, 0, 0.1)

        # VMEC parameters are all fixed by default, while surface
        # parameters are all non-fixed by default. You can choose
        # which parameters are optimized by setting their 'fixed'
        # attributes.
        surf.all_fixed()
        surf.set_fixed('rc(0,0)', False)

        # Each Target is then equipped with a shift and weight, to become a
        # term in a least-squares objective function
        desired_volume = 0.15
        term1 = LeastSquaresTerm(equil.volume, desired_volume, 1)

        # A list of terms are combined to form a nonlinear-least-squares
        # problem.
        prob = LeastSquaresProblem([term1])

        # Check that the problem was set up correctly:
        self.assertEqual(prob.names, ['rc(0,0)'])
        np.testing.assert_allclose(prob.x, [1.0])
        self.assertEqual(prob.all_owners, [equil, surf])
        self.assertEqual(prob.dof_owners, [surf])

        # Solve the minimization problem:
        prob.solve()

        print("At the optimum,")
        print(" rc(m=0,n=0) = ", surf.get_rc(0, 0))
        print(" volume, according to VMEC    = ", equil.volume())
        print(" volume, according to Surface = ", surf.volume())
        print(" objective function = ", prob.objective)

        self.assertAlmostEqual(surf.get_rc(0, 0), 0.7599088773175, places=5)
        self.assertAlmostEqual(equil.volume(), 0.15, places=6)
        self.assertAlmostEqual(surf.volume(), 0.15, places=6)
        self.assertLess(np.abs(prob.objective), 1.0e-15)

        equil.finalize()
"""

if __name__ == "__main__":
    unittest.main()
