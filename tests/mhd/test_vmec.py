import unittest
import logging
import os
import numpy as np
from simsopt.mhd.vmec import IotaTargetMetric

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    import vmec
    vmec_found = True
except ImportError:
    vmec_found = False

from simsopt.objectives.least_squares import LeastSquaresProblem

# from simsopt.mhd.vmec import vmec_found
if (MPI is not None) and vmec_found:
    from simsopt.mhd.vmec import Vmec
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


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
                prob = LeastSquaresProblem([(vmec.iota_axis, 0.41, 1),
                                            (vmec.volume, 0.15, 1)])
                fail_val = 1e12
            else:
                # Try a custom failure value
                fail_val = 2.0e30
                prob = LeastSquaresProblem([(vmec.iota_axis, 0.41, 1),
                                            (vmec.volume, 0.15, 1)],
                                           fail=fail_val)

            r00 = vmec.boundary.get_rc(0, 0)
            # The first evaluation should succeed.
            f = prob.f()
            print(f[0], f[1])
            correct_f = [-0.004577338528148067, 2.8313872701632925]
            # Don't worry too much about accuracy here.
            np.testing.assert_allclose(f, correct_f, rtol=0.1)

            # Now set a crazy boundary shape to make VMEC fail. This
            # boundary causes VMEC to hit the max number of iterations
            # without meeting ftol.
            vmec.boundary.set_rc(0, 0, 0.2)
            vmec.need_to_run_code = True
            f = prob.f()
            print(f)
            np.testing.assert_allclose(f, np.full(2, fail_val))

            # Restore a reasonable boundary shape. VMEC should work again.
            vmec.boundary.set_rc(0, 0, r00)
            vmec.need_to_run_code = True
            f = prob.f()
            print(f)
            np.testing.assert_allclose(f, correct_f, rtol=0.1)

            # Now set a self-intersecting boundary shape. This causes VMEC
            # to fail with "ARNORM OR AZNORM EQUAL ZERO IN BCOVAR" before
            # it even starts iterating.
            orig_mode = vmec.boundary.get_rc(1, 3)
            vmec.boundary.set_rc(1, 3, 0.5)
            vmec.need_to_run_code = True
            f = prob.f()
            print(f)
            np.testing.assert_allclose(f, np.full(2, fail_val))

            # Restore a reasonable boundary shape. VMEC should work again.
            vmec.boundary.set_rc(1, 3, orig_mode)
            vmec.need_to_run_code = True
            f = prob.f()
            print(f)
            np.testing.assert_allclose(f, correct_f, rtol=0.1)

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
        filename = os.path.join(TEST_DIR, 'input.LandremanSengupta2019_section5.4_B2_A80')
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

        logger.info('well_vmec:', well_vmec, '  well_analytic:', well_analytic)
        np.testing.assert_allclose(well_vmec, well_analytic, rtol=2e-2, atol=0)

    def test_iota(self):
        """
        Test the functions related to iota.
        """
        filename = os.path.join(TEST_DIR, 'input.LandremanSengupta2019_section5.4_B2_A80')
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

    def test_iota_target_metric(self):
        """
        Check that iota_target_metric computed on full instead of half grid
        yields approximately same result.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename,ntheta=50,nphi=50)

        target_function = lambda s : np.cos(s)
        metric1 = vmec.iota_target_metric(target_function)
        stencil = np.ones_like(vmec.s_full_grid)
        stencil[0] = 0.5
        stencil[-1] = 0.5
        metric2 = 0.5 * np.sum((vmec.wout.iotaf - target_function(vmec.s_full_grid))**2 * stencil) * vmec.ds

        self.assertAlmostEqual(metric1,metric2,places=3)

    def test_iota_weighted(self):
        """
        Check that objective value with peaked gaussian weight function yields
        approximately the value of iota at center of weight function.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename,ntheta=50,nphi=50)

        vmec.run()
        iota_center = vmec.wout.iotas[50]
        s_center = vmec.s_half_grid[50]
        target_function = lambda s : np.exp(-(s-s_center)**2/0.01**2)
        self.assertAlmostEqual(vmec.iota_weighted(target_function),iota_center,places=2)

    def test_well_weighted(self):
        """
        Check that objective value with peaked gaussian weight function at axis
        and edge matches well metric computed with V'(0) and V'(1).
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename,ntheta=50,nphi=50)

        vmec.run()
        vp_l = 1.5*vmec.wout.vp[1]-0.5*vmec.wout.vp[2]
        vp_r = 1.5*vmec.wout.vp[-1]-0.5*vmec.wout.vp[-2]
        well1 = (vp_l-vp_r)/(vp_l+vp_r)
        weight1 = lambda s: np.exp(-s**2/0.01**2)
        weight2 = lambda s: np.exp(-(1-s)**2/0.01**2)
        well2 = vmec.well_weighted(weight1,weight2)
        self.assertAlmostEqual(well1,well2,places=2)

    def test_B_on_arclength_grid(self):
        """
        Check that B^2 matches bmnc from wout file.
        Check that B is unchanged when evaluated on arclength grid from first run.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename,ntheta=50,nphi=50)

        Bx, By, Bz, arclength0 = vmec.B_on_arclength_grid()
        B2 = Bx*Bx + By*By + Bz*Bz

        theta1D = vmec.boundary.quadpoints_theta * 2 * np.pi
        phi1D = vmec.boundary.quadpoints_phi * 2 * np.pi
        nphi = len(phi1D)
        ntheta = len(theta1D)
        theta, phi = np.meshgrid(theta1D, phi1D)

        bmnc = 1.5 * vmec.wout.bmnc[:,-1] - 0.5 * vmec.wout.bmnc[:,-2]
        xm = vmec.wout.xm_nyq
        xn = vmec.wout.xn_nyq
        angle = vmec.wout.xm_nyq[:,None,None] * theta[None,:,:] \
            - vmec.wout.xn_nyq[:,None,None] * phi[None,:,:]
        B = np.sum(bmnc[:,None,None] * np.cos(angle), axis=0)
        self.assertTrue(np.max(np.abs(B**2-B2))<1.e-4)

        Bx, By, Bz, arclength = vmec.B_on_arclength_grid()
        self.assertTrue(np.max(np.abs(Bx-Bx))<1.0e-4)
        self.assertTrue(np.max(np.abs(By-By))<1.0e-4)
        self.assertTrue(np.max(np.abs(Bz-Bz))<1.0e-4)

    def test_d_iota_target_metric(self):
        """
        Compare d_iota_target_metric with finite differences for a surface
        perturbation in a random direction.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename,ntheta=50,nphi=50)

        target_function = lambda s : 0.68
        epsilon = 1.e-4 # FD step size
        adjoint_epsilon = 1.e-1 # perturbation amplitude for adjoint solve

        # Compute random direction for surface perturbation
        surf = vmec.boundary
        dofs = np.copy(vmec.boundary.get_dofs())
        vec = np.random.standard_normal(dofs.shape)
        unitvec = vec / np.sqrt(np.vdot(vec, vec))

        def iota_fun(epsilon):
            vmec.boundary.set_dofs(dofs + epsilon*unitvec)
            vmec.need_to_run_code = True
            return vmec.iota_target_metric(target_function)

        d_iota_fd = (iota_fun(epsilon)-iota_fun(-epsilon))/(2*epsilon)

        vmec.boundary.set_dofs(dofs)
        vmec.need_to_run_code = True
        d_iota_adjoint = np.dot(vmec.d_iota_target_metric(target_function,adjoint_epsilon),unitvec)

        relative_error = np.abs(d_iota_fd-d_iota_adjoint)/np.abs(d_iota_fd)
        logger.info('adjoint jac:', d_iota_adjoint, '  fd jac:', d_iota_fd)
        logger.info('relative error: ',relative_error)
        self.assertTrue(relative_error < 5.e-2)

    def test_IotaTargetMetric(self):
        """
        Compare dJ() with finite differences for a surface perturbation in a
        random direction.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename,ntheta=50,nphi=50)

        target_function = lambda s : 0.68
        epsilon = 1.e-4 # FD step size
        adjoint_epsilon = 1.e-1 # perturbation amplitude for adjoint solve

        obj = IotaTargetMetric(vmec,target_function,adjoint_epsilon)

        # Compute random direction for surface perturbation
        dofs = np.copy(vmec.boundary.get_dofs())
        vec = np.random.standard_normal(dofs.shape)
        unitvec = vec / np.sqrt(np.vdot(vec, vec))

        def iota_fun(epsilon):
            vmec.boundary.set_dofs(dofs + epsilon*unitvec)
            return obj.J()

        d_iota_fd = (iota_fun(epsilon)-iota_fun(-epsilon))/(2*epsilon)

        vmec.boundary.set_dofs(dofs)
        vmec.need_to_run_code = True
        d_iota_adjoint = np.dot(obj.dJ(),unitvec)

        relative_error = np.abs(d_iota_fd-d_iota_adjoint)/np.abs(d_iota_fd)
        logger.info('adjoint jac:', d_iota_adjoint, '  fd jac:', d_iota_fd)
        logger.info('relative error: ',relative_error)
        self.assertTrue(relative_error < 5.e-2)

    def test_IotaTargetMetric_LeastSquaresProblem(self):
        """
        Compare jacobian for least-squares problem with supplied gradient wrt
        one surface coefficient matches finite-differences.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename,ntheta=50,nphi=50)

        target_function = lambda s : 0.68
        epsilon = 1.e-4 # FD step size
        adjoint_epsilon = 1.e-1 # perturbation amplitude for adjoint solve

        # Compute random direction for surface perturbation
        surf = vmec.boundary
        surf.all_fixed()
        surf.set_fixed("rc(0,0)",False)  # Major radius

        obj = IotaTargetMetric(vmec,target_function,adjoint_epsilon)

        prob = LeastSquaresProblem([(obj, 0, 1)])

        jac = prob.dofs.jac()
        fd_jac = prob.dofs.fd_jac()

        relative_error = np.abs(fd_jac-jac)/np.abs(fd_jac)
        logger.info('adjoint jac:', jac, '  fd jac:', fd_jac)
        logger.info('relative error: ',relative_error)
        self.assertTrue(relative_error < 1.e-2)

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
