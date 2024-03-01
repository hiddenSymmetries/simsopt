import unittest
import os
import logging
import numpy as np

from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual, \
    B_cartesian, IotaTargetMetric, IotaWeighted, WellWeighted, \
    vmec_fieldlines
from simsopt.objectives.least_squares import LeastSquaresProblem

try:
    import matplotlib
    matplotlib_found = True
except:
    matplotlib_found = False

try:
    import vmec
except ImportError as e:
    vmec = None

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None

from simsopt.mhd.vmec import Vmec

from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


class QuasisymmetryRatioResidualWoutTests(unittest.TestCase):
    # These are the tests related to QuasisymmetryRatioResidual that
    # do not require actually running vmec.

    def test_independent_of_resolution(self):
        """
        The total quasisymmetry error should be nearly independent of the
        resolution parameters used.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc'))

        ss = [[0.5], [0.7], [0.3, 0.6], [0.2, 0.4, 0.7]]
        weightss = [None, [1], None, [2.2, 5.5, 0.9]]
        ms = [1, 1, 1, 0]
        ns = [0, 1, -1, 1]
        for surfaces, weights in zip(ss, weightss):
            logger.info(f'surfaces={surfaces} weights={weights}')
            for m, n in zip(ms, ns):
                qs_ref = QuasisymmetryRatioResidual(vmec, surfaces, helicity_m=m, helicity_n=n,
                                                    weights=weights, ntheta=130, nphi=114)

                qs = QuasisymmetryRatioResidual(vmec, surfaces, helicity_m=m, helicity_n=n,
                                                weights=weights, ntheta=65, nphi=57)

                total_ref = qs_ref.total()
                total = qs.total()
                logger.info(f'm={m} n={n} '
                            f'qs_ref={qs_ref.total()}, qs={qs.total()} diff={total_ref-total}')
                np.testing.assert_allclose(qs_ref.total(), qs.total(), atol=0, rtol=1e-2)

    def test_profile(self):
        """
        Verify that the profile() function returns the same results as
        objects that return the quasisymmetry error for just a single
        surface.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc'))
        surfs = [0, 0.4, 1]
        weights = [0.2, 0.7, 1.3]
        m = 1
        n = 1
        qs_profile = QuasisymmetryRatioResidual(vmec, surfs, helicity_m=m, helicity_n=n, weights=weights)
        profile = qs_profile.profile()
        np.testing.assert_allclose(np.sum(profile), qs_profile.total())
        for j in range(len(surfs)):
            qs_single = QuasisymmetryRatioResidual(vmec, [surfs[j]], helicity_m=m, helicity_n=n, weights=[weights[j]])
            np.testing.assert_allclose(profile[j], qs_single.total())

    def test_compute(self):
        """
        Check that several fields returned by ``compute()`` can be
        accessed.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc'))
        qs = QuasisymmetryRatioResidual(vmec, 0.5, 1, 1)
        r = qs.compute()
        np.testing.assert_allclose(r.bsupu * r.d_B_d_theta + r.bsupv * r.d_B_d_phi, r.B_dot_grad_B)
        np.testing.assert_allclose(r.B_cross_grad_B_dot_grad_psi,
                                   r.d_psi_d_s * (r.bsubu * r.d_B_d_phi - r.bsubv * r.d_B_d_theta) / r.sqrtg)


@unittest.skipIf(vmec is None, "vmec python package is not found")
class QuasisymmetryRatioResidualTests(unittest.TestCase):
    def test_axisymmetry(self):
        """
        For an axisymmetric field, the QA error should be 0, while the QH
        and QP error should be significant.
        """

        vmec = Vmec(os.path.join(TEST_DIR, 'input.circular_tokamak'))

        for surfaces in [[0.5], [0.3, 0.6]]:
            qa = QuasisymmetryRatioResidual(vmec, surfaces, helicity_m=1, helicity_n=0)
            residuals = qa.residuals()
            logger.info(f'max QA error: {np.max(np.abs(residuals))}')
            np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=2e-6)

            qh = QuasisymmetryRatioResidual(vmec, surfaces, helicity_m=1, helicity_n=1)
            logger.info(f'QH error: {qh.total()}')
            self.assertTrue(qh.total() > 0.01)

            qp = QuasisymmetryRatioResidual(vmec, surfaces, helicity_m=0, helicity_n=1)
            logger.info(f'QP error: {qp.total()}')
            self.assertTrue(qp.total() > 0.01)

    def test_good_qa(self):
        """
        For a configuration that is known to have good quasi-axisymmetry,
        the QA error should have a low value. The QH and QP errors should be larger.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.simsopt_nfp2_QA_20210328-01-020_000_000251'))
        qa = QuasisymmetryRatioResidual(vmec, 0.5, helicity_m=1, helicity_n=0)
        total = qa.total()
        logger.info(f"QA error for simsopt_nfp2_QA_20210328-01-020_000_000251: {total}")
        self.assertTrue(total < 2e-5)

        qh = QuasisymmetryRatioResidual(vmec, 0.5, helicity_m=1, helicity_n=1)
        logger.info(f'QH error: {qh.total()}')
        self.assertTrue(qh.total() > 0.002)

        qp = QuasisymmetryRatioResidual(vmec, 0.5, helicity_m=0, helicity_n=1)
        logger.info(f'QP error: {qp.total()}')
        self.assertTrue(qp.total() > 0.002)

    def test_good_qh(self):
        """
        For a configuration that is known to have good quasi-helical symmetry,
        the QH error should have a low value. The QA and QP errors should be larger.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.20210406-01-002-nfp4_QH_000_000240'))

        qh = QuasisymmetryRatioResidual(vmec, 0.5, helicity_m=1, helicity_n=-1)
        logger.info(f'QH error (n=-1) for 20210406-01-002-nfp4_QH_000_000240: {qh.total()}')
        self.assertTrue(qh.total() < 5e-5)

        qh2 = QuasisymmetryRatioResidual(vmec, 0.5, helicity_m=1, helicity_n=1)
        logger.info(f'QH error (n=+1) for 20210406-01-002-nfp4_QH_000_000240: {qh2.total()}')
        self.assertTrue(qh2.total() > 0.01)

        qa = QuasisymmetryRatioResidual(vmec, 0.5, helicity_m=1, helicity_n=0)
        total = qa.total()
        logger.info(f'QA error for 20210406-01-002-nfp4_QH_000_000240: {total}')
        self.assertTrue(total > 0.05)

        qp = QuasisymmetryRatioResidual(vmec, 0.5, helicity_m=0, helicity_n=1)
        logger.info(f'QP error: {qp.total()}')
        self.assertTrue(qp.total() > 0.005)

    def test_independent_of_scaling(self):
        """
        The quasisymmetry error should be unchanged under a scaling of the
        configuration in length and/or magnetic field strength.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_low_res'))
        qs1 = QuasisymmetryRatioResidual(vmec, [0, 0.7, 1], helicity_m=1, helicity_n=-1, weights=[0.8, 1.1, 0.9])
        results1 = qs1.compute()
        residuals1 = qs1.residuals()

        R_scale = 2.7
        B_scale = 1.6

        # Now scale the vmec configuration:
        vmec.boundary.rc *= R_scale
        vmec.boundary.zs *= R_scale
        vmec.indata.phiedge *= B_scale * R_scale * R_scale
        vmec.indata.pres_scale *= B_scale * B_scale
        vmec.indata.curtor *= B_scale * R_scale

        vmec.need_to_run_code = True
        qs2 = QuasisymmetryRatioResidual(vmec, [0, 0.7, 1], helicity_m=1, helicity_n=-1, weights=[0.8, 1.1, 0.9])
        results2 = qs2.compute()
        residuals2 = qs2.residuals()

        logger.info('Max difference in residuals after scaling vmec:' \
                    + str(np.max(np.abs(residuals1 - residuals2))))
        np.testing.assert_allclose(residuals1, residuals2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(results1.total, results2.total, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(results1.profile, results2.profile, rtol=1e-10, atol=1e-10)

    def test_iota_0(self):
        """
        Verify the metric works even when iota = 0, for a vacuum
        axisymmetric configuration with circular cross-section.
        """
        vmec = Vmec()
        qs = QuasisymmetryRatioResidual(vmec, 0.5)
        logger.info(f'QuasisymmetryRatioResidual for a vacuum axisymmetric config with iota = 0: {qs.total()}')
        self.assertTrue(qs.total() < 1e-12)


@unittest.skipIf((MPI is None) or (vmec is None), "Valid Python interface to VMEC not found")
class BCartesianTests(unittest.TestCase):
    def test_B_cartesian(self):
        """
        Check that B^2 matches bmnc from wout file.
        """
        # Cover both the cases of initializing from a wout file and from an input file:
        filenames = ['wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc',
                     'input.rotating_ellipse']
        for filename_base in filenames:
            for grid_option in range(3):
                filename = os.path.join(TEST_DIR, filename_base)
                vmec = Vmec(filename, ntheta=50, nphi=53)

                if grid_option == 0:
                    # Use the (phi, theta) grid from vmec.boundary:
                    Bx, By, Bz = B_cartesian(vmec)
                    theta1D = vmec.boundary.quadpoints_theta * 2 * np.pi
                    phi1D = vmec.boundary.quadpoints_phi * 2 * np.pi
                elif grid_option == 1:
                    # Specify nphi and ntheta:
                    ntheta = 55
                    nphi = 52
                    Bx, By, Bz = B_cartesian(vmec, ntheta=ntheta, nphi=nphi, range="field period")
                    # The previous line runs vmec so now vmec.wout.nfp is available.
                    theta1D = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
                    phi1D = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
                elif grid_option == 2:
                    # Specify a custom (phi, theta) grid:
                    quadpoints_theta = np.array([-0.1, 0.4, 0.9])
                    quadpoints_phi = np.array([0.02, 0.3, 0.35, 2.3])
                    Bx, By, Bz = B_cartesian(vmec,
                                             quadpoints_phi=quadpoints_phi,
                                             quadpoints_theta=quadpoints_theta)
                    theta1D = quadpoints_theta * 2 * np.pi
                    phi1D = quadpoints_phi * 2 * np.pi

                B2 = Bx*Bx + By*By + Bz*Bz

                nphi = len(phi1D)
                ntheta = len(theta1D)
                theta, phi = np.meshgrid(theta1D, phi1D)

                bmnc = 1.5 * vmec.wout.bmnc[:, -1] - 0.5 * vmec.wout.bmnc[:, -2]
                xm = vmec.wout.xm_nyq
                xn = vmec.wout.xn_nyq
                angle = vmec.wout.xm_nyq[:, None, None] * theta[None, :, :] \
                    - vmec.wout.xn_nyq[:, None, None] * phi[None, :, :]
                B = np.sum(bmnc[:, None, None] * np.cos(angle), axis=0)
                logger.info(f'Max difference in B2: {np.max(np.abs(B**2 - B2))}')
                np.testing.assert_allclose(B**2, B2, atol=1e-4, rtol=1e-4)


@unittest.skipIf((MPI is None) or (vmec is None), "Valid Python interface to VMEC not found")
class IotaTargetMetricTests(unittest.TestCase):
    def test_IotaTargetMetric_J(self):
        """
        Check that iota_target_metric computed on full instead of half grid
        yields approximately same result.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=50, nphi=50)

        target_function = lambda s: np.cos(s)
        iota_target_metric = IotaTargetMetric(vmec, target_function)

        metric1 = iota_target_metric.J()
        stencil = np.ones_like(vmec.s_full_grid)
        stencil[0] = 0.5
        stencil[-1] = 0.5
        metric2 = 0.5 * np.sum((vmec.wout.iotaf - target_function(vmec.s_full_grid))**2 * stencil) * vmec.ds

        self.assertAlmostEqual(metric1, metric2, places=3)

    def test_IotaTargetMetric_dJ(self):
        """
        Compare dJ() with finite differences for a surface perturbation in a
        random direction.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=100, nphi=100)

        target_function = lambda s: 0.68
        epsilon = 1.e-4  # FD step size
        adjoint_epsilon = 1.e-2  # perturbation amplitude for adjoint solve

        obj = IotaTargetMetric(vmec, target_function, adjoint_epsilon)

        # Compute random direction for surface perturbation
        dofs = np.copy(vmec.boundary.get_dofs())
        np.random.seed(0)
        vec = np.random.standard_normal(dofs.shape)
        unitvec = vec / np.sqrt(np.vdot(vec, vec))

        def iota_fun(epsilon):
            vmec.boundary.set_dofs(dofs + epsilon*unitvec)
            return obj.J()

        d_iota_fd = (iota_fun(epsilon)-iota_fun(-epsilon))/(2*epsilon)

        vmec.boundary.set_dofs(dofs)
        vmec.need_to_run_code = True
        d_iota_adjoint = np.dot(obj.dJ(), unitvec)

        relative_error = np.abs(d_iota_fd-d_iota_adjoint)/np.abs(d_iota_fd)
        logger.info(f"adjoint jac: {d_iota_adjoint},   fd jac: {d_iota_fd}")
        logger.info(f"relative error: {relative_error}")
        self.assertLessEqual(relative_error, 5.e-2)


@unittest.skipIf((MPI is None) or (vmec is None), "Valid Python interface to VMEC not found")
class IotaWeightedTests(unittest.TestCase):
    def test_IotaWeighted_J(self):
        """
        Check that objective value with peaked Gaussian weight function yields
        approximately the value of iota at center of weight function.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=50, nphi=50)

        vmec.run()
        iota_center = vmec.wout.iotas[50]
        s_center = vmec.s_half_grid[50]
        weight_function = lambda s: np.exp(-(s-s_center)**2/0.01**2)
        iota_weighted = IotaWeighted(vmec, weight_function)

        self.assertAlmostEqual(iota_weighted.J(), iota_center, places=2)

    def test_IotaWeighted_dJ(self):
        """
        Compare dJ() with finite differences for a surface perturbation in a
        random direction.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=100, nphi=100)

        weight_function = lambda s: s**2
        epsilon = 1.e-4  # FD step size
        adjoint_epsilon = 5.e-2  # perturbation amplitude for adjoint solve

        obj = IotaWeighted(vmec, weight_function, adjoint_epsilon)

        # Compute random direction for surface perturbation
        dofs = np.copy(vmec.boundary.x)
        np.random.seed(0)
        vec = np.random.standard_normal(dofs.shape)
        unitvec = vec / np.sqrt(np.vdot(vec, vec))

        def iota_fun(epsilon):
            vmec.boundary.x = dofs + epsilon*unitvec
            return obj.J()

        d_iota_fd = (iota_fun(epsilon)-iota_fun(-epsilon))/(2*epsilon)

        vmec.boundary.x = dofs
        # vmec.need_to_run_code = True
        d_iota_adjoint = np.dot(obj.dJ(), unitvec)

        relative_error = np.abs(d_iota_fd-d_iota_adjoint)/np.abs(d_iota_fd)
        logger.info(f"adjoint jac: {d_iota_adjoint},   fd jac: {d_iota_fd}")
        logger.info(f"relative error: {relative_error}")
        self.assertLessEqual(relative_error, 5.e-2)


@unittest.skipIf((MPI is None) or (vmec is None), "Valid Python interface to VMEC not found")
class WellWeightedTests(unittest.TestCase):
    def test_WellWeighted_J(self):
        """
        Check that objective value with peaked Gaussian weight function at axis
        and edge matches well metric computed with V'(0) and V'(1).
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=50, nphi=50)

        vmec.run()
        vp_l = 1.5*vmec.wout.vp[1]-0.5*vmec.wout.vp[2]
        vp_r = 1.5*vmec.wout.vp[-1]-0.5*vmec.wout.vp[-2]
        well1 = (vp_l-vp_r)/(vp_l+vp_r)
        weight1 = lambda s: np.exp(-s**2/0.01**2)
        weight2 = lambda s: np.exp(-(1-s)**2/0.01**2)

        well_weighted = WellWeighted(vmec, weight1, weight2)
        well2 = well_weighted.J()
        self.assertAlmostEqual(well1, well2, places=2)

    def test_WellWeighted_dJ(self):
        """
        Compare dJ() with finite differences for a surface perturbation in a
        random direction.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=100, nphi=100)

        epsilon = 1.e-2  # FD step size
        adjoint_epsilon = 1.e0  # perturbation amplitude for adjoint solve

        weight1 = lambda s: np.exp(-s**2/0.5**2)
        weight2 = lambda s: np.exp(-(1-s)**2/0.5**2)

        obj = WellWeighted(vmec, weight1, weight2, adjoint_epsilon)

        # Compute random direction for surface perturbation
        dofs = np.copy(vmec.boundary.get_dofs())
        np.random.seed(0)
        vec = np.random.standard_normal(dofs.shape)
        unitvec = vec / np.sqrt(np.vdot(vec, vec))

        def well_fun(epsilon):
            vmec.boundary.set_dofs(dofs + epsilon*unitvec)
            return obj.J()

        d_well_fd = (well_fun(epsilon)-well_fun(-epsilon))/(2*epsilon)

        vmec.boundary.set_dofs(dofs)
        vmec.need_to_run_code = True
        d_well_adjoint = np.dot(obj.dJ(), unitvec)

        relative_error = np.abs(d_well_fd-d_well_adjoint)/np.abs(d_well_fd)
        logger.info(f"adjoint jac: {d_well_adjoint},   fd jac: {d_well_fd}")
        logger.info(f"relative error: {relative_error}")
        self.assertLessEqual(relative_error, 5.e-2)


class VmecFieldlinesTests(unittest.TestCase):
    def test_fieldline_grids(self):
        """
        Check the grids in theta and phi created by vmec_fieldlines().
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc'))

        # Try a case in which theta is specified:
        s = [1.0e-15, 0.5, 1]
        alpha = [0, np.pi]
        theta = np.linspace(-np.pi, np.pi, 5)
        fl = vmec_fieldlines(vmec, s, alpha, theta1d=theta)
        for js in range(fl.ns):
            for jalpha in range(fl.nalpha):
                np.testing.assert_allclose(fl.theta_pest[js, jalpha, :], theta, atol=1e-15)
                np.testing.assert_allclose(fl.theta_pest[js, jalpha, :] - fl.iota[js] * fl.phi[js, jalpha, :], alpha[jalpha],
                                           atol=1e-15)

        # Try a case in which phi is specified:
        s = 1
        alpha = -np.pi
        phi = np.linspace(-np.pi, np.pi, 6)
        fl = vmec_fieldlines(vmec, s, alpha, phi1d=phi)
        for js in range(fl.ns):
            for jalpha in range(fl.nalpha):
                np.testing.assert_allclose(fl.phi[js, jalpha, :], phi)
                np.testing.assert_allclose(fl.theta_pest[js, jalpha, :] - fl.iota[js] * fl.phi[js, jalpha, :], alpha)

        # Try specifying both theta and phi:
        with self.assertRaises(ValueError):
            fl = vmec_fieldlines(vmec, s, alpha, phi1d=phi, theta1d=theta)
        # Try specifying neither theta nor phi:
        with self.assertRaises(ValueError):
            fl = vmec_fieldlines(vmec, s, alpha)

    def test_consistency(self):
        """
        Check internal consistency of the results of vmec_fieldlines().
        """
        filenames = ['wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc',
                     'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc',
                     'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc']
        for filename in filenames:
            for j_theta_phi_in in range(2):
                logger.info(f'Testing vmec_fieldline for file {filename}, j_theta_phi_in={j_theta_phi_in}')
                vmec = Vmec(os.path.join(TEST_DIR, filename))
                s = [0.25, 0.75]
                ns = len(s)
                alpha = np.linspace(0, 2 * np.pi, 3, endpoint=False)
                z_grid = np.linspace(-np.pi / 2, np.pi / 2, 7)
                if j_theta_phi_in == 0:
                    fl = vmec_fieldlines(vmec, s=s, alpha=alpha, phi1d=z_grid)
                else:
                    fl = vmec_fieldlines(vmec, s=s, alpha=alpha, theta1d=z_grid)

                np.testing.assert_allclose(fl.sqrt_g_vmec, fl.sqrt_g_vmec_alt, rtol=1e-4, atol=1e-4)

                # Verify that (B dot grad theta_pest) / (B dot grad phi) = iota
                should_be_iota = (fl.B_sup_theta_vmec * (1 + fl.d_lambda_d_theta_vmec) + fl.B_sup_phi * fl.d_lambda_d_phi) / fl.B_sup_phi
                for js in range(ns):
                    np.testing.assert_allclose(fl.iota[js], should_be_iota[js, :, :], rtol=1e-4, atol=1e-4)

                # Compare 2 methods of computing B_sup_theta_pest:
                np.testing.assert_allclose(fl.B_sup_theta_vmec * (1 + fl.d_lambda_d_theta_vmec) + fl.B_sup_phi * fl.d_lambda_d_phi,
                                           fl.B_sup_theta_pest, rtol=1e-4)

                # grad_phi_X should be -sin(phi) / R:
                np.testing.assert_allclose(fl.grad_phi_X, -fl.sinphi / fl.R, rtol=1e-4)
                # grad_phi_Y should be cos(phi) / R:
                np.testing.assert_allclose(fl.grad_phi_Y, fl.cosphi / fl.R, rtol=1e-4)
                # grad_phi_Z should be 0:
                np.testing.assert_allclose(fl.grad_phi_Z, 0, atol=1e-16)

                # Verify that the Jacobian equals the appropriate cross
                # product of the basis vectors.
                test_arr = 0 \
                    + fl.d_X_d_s * fl.d_Y_d_theta_vmec * fl.d_Z_d_phi \
                    + fl.d_Y_d_s * fl.d_Z_d_theta_vmec * fl.d_X_d_phi \
                    + fl.d_Z_d_s * fl.d_X_d_theta_vmec * fl.d_Y_d_phi \
                    - fl.d_Z_d_s * fl.d_Y_d_theta_vmec * fl.d_X_d_phi \
                    - fl.d_X_d_s * fl.d_Z_d_theta_vmec * fl.d_Y_d_phi \
                    - fl.d_Y_d_s * fl.d_X_d_theta_vmec * fl.d_Z_d_phi
                np.testing.assert_allclose(test_arr, fl.sqrt_g_vmec, rtol=1e-4)

                test_arr = 0 \
                    + fl.grad_s_X * fl.grad_theta_vmec_Y * fl.grad_phi_Z \
                    + fl.grad_s_Y * fl.grad_theta_vmec_Z * fl.grad_phi_X \
                    + fl.grad_s_Z * fl.grad_theta_vmec_X * fl.grad_phi_Y \
                    - fl.grad_s_Z * fl.grad_theta_vmec_Y * fl.grad_phi_X \
                    - fl.grad_s_X * fl.grad_theta_vmec_Z * fl.grad_phi_Y \
                    - fl.grad_s_Y * fl.grad_theta_vmec_X * fl.grad_phi_Z
                np.testing.assert_allclose(test_arr, 1 / fl.sqrt_g_vmec, rtol=2e-4)

                # Verify that \vec{B} dot (each of the covariant and
                # contravariant basis vectors) matches the corresponding term
                # from VMEC.
                test_arr = fl.B_X * fl.d_X_d_theta_vmec + fl.B_Y * fl.d_Y_d_theta_vmec + fl.B_Z * fl.d_Z_d_theta_vmec
                np.testing.assert_allclose(test_arr, fl.B_sub_theta_vmec, rtol=0.01, atol=0.01)

                test_arr = fl.B_X * fl.d_X_d_s + fl.B_Y * fl.d_Y_d_s + fl.B_Z * fl.d_Z_d_s
                np.testing.assert_allclose(test_arr, fl.B_sub_s, rtol=2e-3, atol=0.005)

                test_arr = fl.B_X * fl.d_X_d_phi + fl.B_Y * fl.d_Y_d_phi + fl.B_Z * fl.d_Z_d_phi
                np.testing.assert_allclose(test_arr, fl.B_sub_phi, rtol=1e-3)

                test_arr = fl.B_X * fl.grad_s_X + fl.B_Y * fl.grad_s_Y + fl.B_Z * fl.grad_s_Z
                np.testing.assert_allclose(test_arr, 0, atol=1e-14)

                test_arr = fl.B_X * fl.grad_phi_X + fl.B_Y * fl.grad_phi_Y + fl.B_Z * fl.grad_phi_Z
                np.testing.assert_allclose(test_arr, fl.B_sup_phi, rtol=1e-4)

                test_arr = fl.B_X * fl.grad_theta_vmec_X + fl.B_Y * fl.grad_theta_vmec_Y + fl.B_Z * fl.grad_theta_vmec_Z
                np.testing.assert_allclose(test_arr, fl.B_sup_theta_vmec, rtol=2e-4)

                # Check 2 ways of computing B_cross_grad_s_dot_grad_alpha:
                np.testing.assert_allclose(fl.B_cross_grad_s_dot_grad_alpha, fl.B_cross_grad_s_dot_grad_alpha_alternate, rtol=1e-3)

                # Check 2 ways of computing B_cross_grad_B_dot_grad_alpha:
                np.testing.assert_allclose(fl.B_cross_grad_B_dot_grad_alpha, fl.B_cross_grad_B_dot_grad_alpha_alternate, atol=0.02)

                # Check 2 ways of computing cvdrift:
                cvdrift_alt = -1 * 2 * fl.B_reference * fl.L_reference * fl.L_reference \
                    * np.sqrt(fl.s)[:, None, None] * fl.B_cross_kappa_dot_grad_alpha \
                    / (fl.modB * fl.modB) * fl.toroidal_flux_sign
                np.testing.assert_allclose(fl.cvdrift, cvdrift_alt)

    def test_stella_regression(self):
        """
        Test vmec_fieldlines() by comparing to calculations with the
        geometry interface in the gyrokinetic code stella.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'))
        s = [0.5, 1.0]
        alpha = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        phi = np.linspace(-np.pi / 2, np.pi / 2, 7)
        fl = vmec_fieldlines(vmec, s=s, alpha=alpha, phi1d=phi)
        theta_vmec_reference = np.array([[-0.54696449126914626, -0.48175613245664178, -0.27486119402681097, 0.0000000000000000, 0.27486119402681097, 0.48175613245664178, 0.54696449126914626],
                                         [1.2860600505790374, 1.4762629621552081, 1.7205057726038357, 1.8975933573125818, 2.0499492968214290, 2.3142882369220339, 2.7218102365172787],
                                         [3.5613750706623071, 3.9688970702575519, 4.2332360103581568, 4.3855919498670044, 4.5626795345757500, 4.8069223450243772, 4.9971252566005484]])
        # Tolerances for s=1 are a bit looser since we extrapolate lambda off the half grid:
        np.testing.assert_allclose(fl.theta_vmec[1, :, :], theta_vmec_reference, rtol=3e-5, atol=3e-5)

        theta_vmec_reference = np.array([[-0.58175490233450466, -0.47234459364571935, -0.27187109234173445, 0.0000000000000000, 0.27187109234173445, 0.47234459364571935, 0.58175490233450466],
                                         [1.3270163720562491, 1.5362773754015560, 1.7610074217338225, 1.9573739260757410, 2.1337336171762495, 2.3746560860701522, 2.7346254905898566],
                                         [3.5485598165897296, 3.9085292211094336, 4.1494516900033354, 4.3258113811038443, 4.5221778855704500, 4.7469079317780292, 4.9561689351233360]])
        np.testing.assert_allclose(fl.theta_vmec[0, :, :], theta_vmec_reference, rtol=3e-10, atol=3e-10)

        modB_reference = np.array([[5.5314557915824105, 5.4914726973937169, 5.4623705859501106, 5.4515507758815724, 5.4623705859501106, 5.4914726973937169, 5.5314557915824105],
                                   [5.8059964497774104, 5.8955040231659970, 6.0055216679618466, 6.1258762025742488, 6.2243609712000030, 6.2905766503767877, 6.3322981435019834],
                                   [6.3322981435019843, 6.2905766503767877, 6.2243609712000039, 6.1258762025742506, 6.0055216679119008, 5.8955040231659970, 5.8059964497774113]])
        np.testing.assert_allclose(fl.modB[0, :, :], modB_reference, rtol=1e-10, atol=1e-10)

        gradpar_reference = np.array([[0.16966605523903308, 0.16128460473382750, 0.13832030925488656, 0.12650509948655936, 0.13832030925488656, 0.16128460473382750, 0.16966605523903308],
                                      [0.18395371643088332, 0.16267463440217919, 0.13722808138129616, 0.13475541851920048, 0.16332776641155652, 0.20528460628581335, 0.22219989371670806],
                                      [0.22219989371670806, 0.20528460628581333, 0.16332776641155652, 0.13475541851920050, 0.13722808138029491, 0.16267463440217922, 0.18395371643088337]])
        L_reference = vmec.wout.Aminor_p
        np.testing.assert_allclose(L_reference * fl.B_sup_phi[0, :, :] / fl.modB[0, :, :],
                                   gradpar_reference, rtol=1e-11, atol=1e-11)

        # Compare to an output file from the stella geometry interface
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'))
        s = [0.5]
        nalpha = 3
        alpha = np.linspace(0, 2 * np.pi, nalpha, endpoint=False)
        phi = np.linspace(-np.pi / 5, np.pi / 5, 7)
        fl = vmec_fieldlines(vmec, s=s, alpha=alpha, phi1d=phi)
        with open(os.path.join(TEST_DIR, 'geometry_W7-X_without_coil_ripple_beta0p05_d23p4_tm.dat')) as f:
            lines = f.readlines()
        np.testing.assert_allclose(fl.alpha, np.fromstring(lines[4], sep=' '))
        phi_stella = np.fromstring(lines[6], sep=' ')
        for j in range(nalpha):
            np.testing.assert_allclose(fl.phi[0, j, :], phi_stella)
            np.testing.assert_allclose(fl.bmag[0, j, :], np.fromstring(lines[8 + j], sep=' '), rtol=1e-6)
            np.testing.assert_allclose(fl.gradpar_phi[0, j, :], np.fromstring(lines[12 + j], sep=' '), rtol=1e-6)
            np.testing.assert_allclose(fl.gds2[0, j, :], np.fromstring(lines[16 + j], sep=' '), rtol=2e-4)
            np.testing.assert_allclose(fl.gds21[0, j, :], np.fromstring(lines[20 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(fl.gds22[0, j, :], np.fromstring(lines[24 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(-1 * fl.gbdrift[0, j, :], np.fromstring(lines[28 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(fl.gbdrift0[0, j, :], np.fromstring(lines[32 + j], sep=' '), atol=1e-4)
            np.testing.assert_allclose(-1 * fl.cvdrift[0, j, :], np.fromstring(lines[36 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(fl.cvdrift0[0, j, :], np.fromstring(lines[40 + j], sep=' '), atol=1e-4)

    def test_axisymm(self):
        """
        Test that vmec_fieldlines() gives sensible results for axisymmetry.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_circular_tokamak_aspect_100_reference.nc'))
        theta = np.linspace(-3 * np.pi, 3 * np.pi, 200)
        fl = vmec_fieldlines(vmec, s=1, alpha=0, theta1d=theta, plot=0)
        B0 = vmec.wout.volavgB
        eps = 1 / vmec.wout.aspect
        safety_factor_q = 1 / vmec.wout.iotaf[-1]
        phi = theta / vmec.wout.iotaf[-1]
        R = vmec.wout.Rmajor_p
        Aminor = vmec.wout.Aminor_p
        d_iota_d_s = (vmec.wout.iotas[-1] - vmec.wout.iotas[-2]) / vmec.ds
        d_iota_d_r = d_iota_d_s * Aminor / 2
        #print('sign of psi in grad psi cross grad theta + iota grad phi cross grad psi:', fl.toroidal_flux_sign)

        # See Matt Landreman's note "20220315-02 Geometry arrays for gyrokinetics in a circular tokamak.docx"
        # for the analytic results below
        np.testing.assert_allclose(fl.modB.flatten(), B0 * (1 - eps * np.cos(theta)), rtol=0.0002)
        np.testing.assert_allclose(fl.B_sup_theta_vmec.flatten(), B0 / (safety_factor_q * R), rtol=0.0006)
        np.testing.assert_allclose(fl.B_cross_grad_B_dot_grad_psi.flatten(), -(B0 ** 3) * eps * np.sin(theta), rtol=0.03)
        np.testing.assert_allclose(fl.B_cross_kappa_dot_grad_psi.flatten(), -(B0 ** 2) * eps * np.sin(theta), rtol=0.02)
        np.testing.assert_allclose(fl.grad_psi_dot_grad_psi.flatten(), B0 * B0 * Aminor * Aminor, rtol=0.03)
        np.testing.assert_allclose(fl.grad_alpha_dot_grad_psi.flatten(), -fl.toroidal_flux_sign * phi * d_iota_d_r * Aminor * B0, rtol=0.02)
        np.testing.assert_allclose(fl.grad_alpha_dot_grad_alpha.flatten(), 1 / (Aminor * Aminor) + (phi * phi * d_iota_d_r * d_iota_d_r), rtol=0.04)
        np.testing.assert_allclose(fl.B_cross_grad_B_dot_grad_alpha.flatten(),
                                   fl.toroidal_flux_sign * (B0 * B0 / Aminor) * (-np.cos(theta) / R + phi * d_iota_d_r * eps * np.sin(theta)),
                                   atol=0.006)
        np.testing.assert_allclose(fl.B_cross_kappa_dot_grad_alpha.flatten(),
                                   fl.toroidal_flux_sign * (B0 / Aminor) * (-np.cos(theta) / R + phi * d_iota_d_r * eps * np.sin(theta)),
                                   atol=0.006)

    @unittest.skipIf(not matplotlib_found, "Matplotlib python module not found")
    def test_plot(self):
        """
        Test the plotting function of vmec_fieldlines()
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'))

        phi = np.linspace(-np.pi / 5, np.pi / 5, 7)
        fl = vmec_fieldlines(vmec, s=1, alpha=0, phi1d=phi, plot=True, show=False)

        theta = np.linspace(-np.pi, np.pi, 100)
        fl = vmec_fieldlines(vmec, s=0.5, alpha=np.pi, theta1d=theta, plot=True, show=False)

        alpha = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        fl = vmec_fieldlines(vmec, s=[0.25, 0.5], alpha=alpha, phi1d=phi, plot=True, show=False)


if __name__ == "__main__":
    unittest.main()
