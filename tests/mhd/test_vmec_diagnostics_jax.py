import os
import unittest

import numpy as np

try:
    import vmec_jax
except ImportError:
    vmec_jax = None

from simsopt.mhd import QuasisymmetryRatioResidual, QuasisymmetryRatioResidualJax, Vmec, VmecJax

from . import TEST_DIR


@unittest.skipIf(vmec_jax is None, "vmec_jax not found")
class QuasisymmetryRatioResidualJaxTests(unittest.TestCase):
    def test_residuals_match_existing_implementation(self):
        vmec = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        cases = [
            ([0.5], None, 1, 0),
            ([0.2, 0.4, 0.7], [0.8, 1.1, 0.9], 1, -1),
            ([0.2, 0.4, 0.7], None, 0, 1),
        ]

        for surfaces, weights, helicity_m, helicity_n in cases:
            qs = QuasisymmetryRatioResidual(
                vmec,
                surfaces,
                helicity_m=helicity_m,
                helicity_n=helicity_n,
                weights=weights,
                ntheta=63,
                nphi=64,
            )
            qs_j = QuasisymmetryRatioResidualJax(
                vmec,
                surfaces,
                helicity_m=helicity_m,
                helicity_n=helicity_n,
                weights=weights,
                ntheta=63,
                nphi=64,
            )
            np.testing.assert_allclose(qs_j.residuals(), qs.residuals(), atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(qs_j.profile(), qs.profile(), atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(qs_j.total(), qs.total(), atol=1e-13, rtol=1e-13)

    def test_vmec_jax_and_legacy_wout_match(self):
        filename = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")
        qs = QuasisymmetryRatioResidualJax(Vmec(filename), [0.3, 0.6], 1, 1)
        qs_j = QuasisymmetryRatioResidualJax(VmecJax(filename), [0.3, 0.6], 1, 1)
        np.testing.assert_allclose(qs_j.residuals(), qs.residuals())
        np.testing.assert_allclose(qs_j.profile(), qs.profile())
        np.testing.assert_allclose(qs_j.total(), qs.total())

    def test_compute_exposes_diagnostics(self):
        vmec = VmecJax(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        qs_j = QuasisymmetryRatioResidualJax(vmec, 0.5, 1, 1)
        results = qs_j.compute()
        np.testing.assert_allclose(
            results.bsupu * results.d_B_d_theta + results.bsupv * results.d_B_d_phi,
            results.B_dot_grad_B,
        )
        np.testing.assert_allclose(
            results.B_cross_grad_B_dot_grad_psi,
            -vmec.wout.phi[-1] / (2 * np.pi)
            * (results.bsubu * results.d_B_d_phi - results.bsubv * results.d_B_d_theta)
            / results.sqrtg,
        )

    def test_weights_must_match_surfaces(self):
        vmec = VmecJax(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        with self.assertRaisesRegex(ValueError, "weights must have the same length"):
            QuasisymmetryRatioResidualJax(vmec, [0.3, 0.6], weights=[1.0])

    def test_residuals_from_state_matches_wrapper_result(self):
        from vmec_jax.static import build_static
        from vmec_jax.wout import state_from_wout
        import simsopt.mhd.vmec_jax as vmec_jax_module

        vmec = VmecJax(
            os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_lowres.nc"),
            nphi=8,
            ntheta=8,
            verbose=False,
        )
        _cfg, indata = vmec_jax.load_config(
            os.path.join(TEST_DIR, "input.LandremanPaul2021_QA_lowres")
        )
        static = build_static(vmec_jax_module._wout_config(vmec._wout_jax, 8, 8))
        state = state_from_wout(vmec._wout_jax)
        qs = QuasisymmetryRatioResidualJax(
            vmec,
            [0.5],
            helicity_m=1,
            helicity_n=0,
            ntheta=17,
            nphi=18,
        )
        residuals_from_state = qs.residuals_from_state(
            static,
            indata,
            signgs=getattr(vmec._wout_jax, "signgs", 1),
        )
        residuals = np.asarray(residuals_from_state(state))
        self.assertEqual(residuals.shape, qs.residuals().shape)
        self.assertTrue(np.all(np.isfinite(residuals)))


if __name__ == "__main__":
    unittest.main()
