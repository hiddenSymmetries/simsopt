import unittest
import logging
import os
import numpy as np
from simsopt.mhd.vmec_diagnostics import B_cartesian, IotaTargetMetric, IotaWeighted, WellWeighted

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    import vmec
    vmec_found = True
except ImportError:
    vmec_found = False

if (MPI is not None) and vmec_found:
    from simsopt.mhd.vmec import Vmec

from simsopt.objectives.least_squares import LeastSquaresProblem

from . import TEST_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@unittest.skipIf((MPI is None) or (not vmec_found), "Valid Python interface to VMEC not found")
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


@unittest.skipIf((MPI is None) or (not vmec_found), "Valid Python interface to VMEC not found")
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

    def test_IotaTargetMetric_LeastSquaresProblem(self):
        """
        Compare Jacobian for least-squares problem with supplied gradient wrt
        one surface coefficient with finite-differences.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=100, nphi=100)

        target_function = lambda s: 0.68
        epsilon = 1.e-3  # FD step size
        adjoint_epsilon = 5.e-1  # perturbation amplitude for adjoint solve

        # Compute random direction for surface perturbation
        surf = vmec.boundary
        surf.all_fixed()
        surf.set_fixed("rc(0,0)", False)  # Major radius

        obj = IotaTargetMetric(vmec, target_function, adjoint_epsilon)

        prob = LeastSquaresProblem([(obj, 0, 1)])

        prob.dofs.abs_step = epsilon
        jac = prob.dofs.jac()
        fd_jac = prob.dofs.fd_jac()

        relative_error = np.abs(fd_jac-jac)/np.abs(fd_jac)
        logger.info(f"adjoint jac: {jac},   fd jac: {fd_jac}")
        logger.info(f"relative error: {relative_error}")
        self.assertLessEqual(relative_error, 2.e-2)


@unittest.skipIf((MPI is None) or (not vmec_found), "Valid Python interface to VMEC not found")
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
        adjoint_epsilon = 1.e-2  # perturbation amplitude for adjoint solve

        obj = IotaWeighted(vmec, weight_function, adjoint_epsilon)

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

    def test_IotaWeighted_LeastSquaresProblem(self):
        """
        Compare Jacobian for least-squares problem with supplied gradient wrt
        one surface coefficient with finite-differences.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=100, nphi=100)

        target_function = lambda s: 0.68
        epsilon = 1.e-3  # FD step size
        adjoint_epsilon = 5.e-1  # perturbation amplitude for adjoint solve

        # Compute random direction for surface perturbation
        surf = vmec.boundary
        surf.all_fixed()
        surf.set_fixed("rc(0,0)", False)  # Major radius

        obj = IotaWeighted(vmec, target_function, adjoint_epsilon)

        prob = LeastSquaresProblem([(obj, 0, 1)])

        prob.dofs.abs_step = epsilon
        jac = prob.dofs.jac()
        fd_jac = prob.dofs.fd_jac()

        relative_error = np.abs(fd_jac-jac)/np.abs(fd_jac)
        logger.info(f"adjoint jac: {jac},   fd jac: {fd_jac}")
        logger.info(f"relative error: {relative_error}")
        self.assertLessEqual(relative_error, 2.e-2)


@unittest.skipIf((MPI is None) or (not vmec_found), "Valid Python interface to VMEC not found")
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

    def test_WellWeighted_LeastSquaresProblem(self):
        """
        Compare Jacobian for least-squares problem with supplied gradient wrt
        one surface coefficient with finite-differences.
        """
        filename = os.path.join(TEST_DIR, 'input.rotating_ellipse')
        vmec = Vmec(filename, ntheta=100, nphi=100)

        epsilon = 1.e-2  # FD step size
        adjoint_epsilon = 1.e0  # perturbation amplitude for adjoint solve

        weight1 = lambda s: np.exp(-s**2/0.5**2)
        weight2 = lambda s: np.exp(-(1-s)**2/0.5**2)

        # Compute random direction for surface perturbation
        surf = vmec.boundary
        surf.all_fixed()
        surf.set_fixed("rc(0,0)", False)  # Major radius

        obj = WellWeighted(vmec, weight1, weight2, adjoint_epsilon)

        prob = LeastSquaresProblem([(obj, 0, 1)])
        prob.dofs.abs_step = epsilon

        jac = prob.dofs.jac()
        fd_jac = prob.dofs.fd_jac()

        relative_error = np.abs(fd_jac-jac)/np.abs(fd_jac)
        logger.info(f"adjoint jac: {jac},   fd jac: {fd_jac}")
        logger.info(f"relative error: {relative_error}")
        self.assertLessEqual(relative_error, 2.e-2)


if __name__ == "__main__":
    unittest.main()
