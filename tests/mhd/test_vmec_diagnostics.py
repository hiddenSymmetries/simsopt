import unittest
import os
import logging
import numpy as np

from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual, \
    B_cartesian, IotaTargetMetric, IotaWeighted, WellWeighted
from simsopt.objectives.graph_least_squares import LeastSquaresProblem

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


class InitializedFromWout(unittest.TestCase):
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
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=50, nphi=50)

        Bx, By, Bz = B_cartesian(vmec)
        B2 = Bx*Bx + By*By + Bz*Bz

        theta1D = vmec.boundary.quadpoints_theta * 2 * np.pi
        phi1D = vmec.boundary.quadpoints_phi * 2 * np.pi
        nphi = len(phi1D)
        ntheta = len(theta1D)
        theta, phi = np.meshgrid(theta1D, phi1D)

        bmnc = 1.5 * vmec.wout.bmnc[:, -1] - 0.5 * vmec.wout.bmnc[:, -2]
        xm = vmec.wout.xm_nyq
        xn = vmec.wout.xn_nyq
        angle = vmec.wout.xm_nyq[:, None, None] * theta[None, :, :] \
            - vmec.wout.xn_nyq[:, None, None] * phi[None, :, :]
        B = np.sum(bmnc[:, None, None] * np.cos(angle), axis=0)
        np.testing.assert_allclose(B**2, B2, atol=1e-4)


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


if __name__ == "__main__":
    unittest.main()
