import unittest
import numpy as np
import os
import logging

from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioError

try:
    import vmec
except ImportError as e:
    vmec = None 

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None

if MPI is not None:
    from simsopt.mhd.vmec import Vmec  # , vmec_found
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


@unittest.skipIf(vmec is None, "vmec python package is not found")
class QuasisymmetryRatioErrorTests(unittest.TestCase):
    def test_axisymmetry(self):
        """
        For an axisymmetric field, the QA error should be 0, while the QH
        and QP error should be significant.
        """

        vmec = Vmec(os.path.join(TEST_DIR, 'input.circular_tokamak'))

        for s in [[0.5], [0.3, 0.6]]:
            qa = QuasisymmetryRatioError(vmec, s, m=1, n=0)
            residuals = qa.residuals()
            logger.info(f'max QA error: {np.max(np.abs(residuals))}')
            np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=2e-6)

            qh = QuasisymmetryRatioError(vmec, s, m=1, n=1)
            logger.info(f'QH error: {qh.total()}')
            self.assertTrue(qh.total() > 0.01)

            qp = QuasisymmetryRatioError(vmec, s, m=0, n=1)
            logger.info(f'QP error: {qp.total()}')
            self.assertTrue(qp.total() > 0.01)

    def test_independent_of_resolution(self):
        """
        The total quasisymmetry error should be nearly independent of the
        resolution parameters used.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_low_res'))

        ss = [[0.5], [0.7], [0.3, 0.6], [0.2, 0.4, 0.7]]
        weightss = [None, [1], None, [2.2, 5.5, 0.9]]
        ms = [1, 1, 1, 0]
        ns = [0, 1, -1, 1]
        for s, weights in zip(ss, weightss):
            logger.info(f's={s} weights={weights}')
            for m, n in zip(ms, ns):
                qs_ref = QuasisymmetryRatioError(vmec, s, m=m, n=n, weights=weights, ntheta=130, nphi=114)
                qs = QuasisymmetryRatioError(vmec, s, m=m, n=n, weights=weights, ntheta=65, nphi=57)
                total_ref = qs_ref.total()
                total = qs.total()
                logger.info(f'm={m} n={n} '
                            f'qs_ref={qs_ref.total()}, qs={qs.total()} diff={total_ref-total}')
                np.testing.assert_allclose(qs_ref.total(), qs.total(), atol=0, rtol=1e-2)

    def test_good_qa(self):
        """
        For a configuration that is known to have good quasi-axisymmetry,
        the QA error should have a low value. The QH and QP errors should be larger.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.simsopt_nfp2_QA_20210328-01-020_000_000251'))
        qa = QuasisymmetryRatioError(vmec, 0.5, m=1, n=0)
        total = qa.total()
        logger.info(f"QA error for simsopt_nfp2_QA_20210328-01-020_000_000251: {total}")
        self.assertTrue(total < 2e-5)

        qh = QuasisymmetryRatioError(vmec, 0.5, m=1, n=1)
        logger.info(f'QH error: {qh.total()}')
        self.assertTrue(qh.total() > 0.002)

        qp = QuasisymmetryRatioError(vmec, 0.5, m=0, n=1)
        logger.info(f'QP error: {qp.total()}')
        self.assertTrue(qp.total() > 0.002)

    def test_good_qh(self):
        """
        For a configuration that is known to have good quasi-helical symmetry,
        the QH error should have a low value. The QA and QP errors should be larger.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.20210406-01-002-nfp4_QH_000_000240'))

        qh = QuasisymmetryRatioError(vmec, 0.5, m=1, n=-1)
        logger.info(f'QH error (n=-1) for 20210406-01-002-nfp4_QH_000_000240: {qh.total()}')
        self.assertTrue(qh.total() < 5e-5)

        qh2 = QuasisymmetryRatioError(vmec, 0.5, m=1, n=1)
        logger.info(f'QH error (n=+1) for 20210406-01-002-nfp4_QH_000_000240: {qh2.total()}')
        self.assertTrue(qh2.total() > 0.01)

        qa = QuasisymmetryRatioError(vmec, 0.5, m=1, n=0)
        total = qa.total()
        logger.info(f'QA error for 20210406-01-002-nfp4_QH_000_000240: {total}')
        self.assertTrue(total > 0.05)

        qp = QuasisymmetryRatioError(vmec, 0.5, m=0, n=1)
        logger.info(f'QP error: {qp.total()}')
        self.assertTrue(qp.total() > 0.005)

    def test_independent_of_scaling(self):
        """
        The quasisymmetry error should be unchanged under a scaling of the
        configuration in length and/or magnetic field strength.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_low_res'))
        qs1 = QuasisymmetryRatioError(vmec, [0, 0.7, 1], m=1, n=-1, weights=[0.8, 1.1, 0.9])
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
        qs2 = QuasisymmetryRatioError(vmec, [0, 0.7, 1], m=1, n=-1, weights=[0.8, 1.1, 0.9])
        results2 = qs2.compute()
        residuals2 = qs2.residuals()

        logger.info('Max difference in residuals after scaling vmec:' \
                    + str(np.max(np.abs(residuals1 - residuals2))))
        np.testing.assert_allclose(residuals1, residuals2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(results1.total, results2.total, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(results1.profile, results2.profile, rtol=1e-10, atol=1e-10)

    def test_profile(self):
        """
        Verify that the profile() function returns the same results as
        objects that return the quasisymmetry error for just a single
        surface.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_low_res'))
        surfs = [0, 0.4, 1]
        weights = [0.2, 0.7, 1.3]
        m = 1
        n = 1
        qs_profile = QuasisymmetryRatioError(vmec, surfs, m=m, n=n, weights=weights)
        profile = qs_profile.profile()
        np.testing.assert_allclose(np.sum(profile), qs_profile.total())
        for j in range(len(surfs)):
            qs_single = QuasisymmetryRatioError(vmec, [surfs[j]], m=m, n=n, weights=[weights[j]])
            np.testing.assert_allclose(profile[j], qs_single.total())

    def test_iota_0(self):
        """
        Verify the metric works even when iota = 0, for a vacuum
        axisymmetric configuration with circular cross-section.
        """
        vmec = Vmec()
        qs = QuasisymmetryRatioError(vmec, 0.5)
        logger.info(f'QuasisymmetryRatioError for a vacuum axisymmetric config with iota = 0: {qs.total()}')
        self.assertTrue(qs.total() < 1e-12)

    def test_compute(self):
        """
        Check that several fields returned by ``compute()`` can be
        accessed.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_low_res'))
        qs = QuasisymmetryRatioError(vmec, 0.5, 1, 1)
        r = qs.compute()
        np.testing.assert_allclose(r.bsupu * r.d_B_d_theta + r.bsupv * r.d_B_d_phi, r.B_dot_grad_B)
        np.testing.assert_allclose(r.B_cross_grad_B_dot_grad_psi,
                                   r.d_psi_d_s * (r.bsubu * r.d_B_d_phi - r.bsubv * r.d_B_d_theta) / r.sqrtg)


if __name__ == "__main__":
    unittest.main()
