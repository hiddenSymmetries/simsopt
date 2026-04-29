import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from simsopt.mhd import VirtualCasing, VirtualCasingJax, Vmec, VmecJax
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.mhd.vmec_diagnostics_jax import QuasisymmetryRatioResidualJax

try:
    import vmec  # noqa: F401
    import mpi4py  # noqa: F401
    import virtual_casing  # noqa: F401
    import vmec_jax
    import virtual_casing_jax  # noqa: F401
except ImportError:
    LEGACY_COMPARISON_DEPS_FOUND = False
else:
    LEGACY_COMPARISON_DEPS_FOUND = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_LEGACY_COMPARISON = (
    os.environ.get("SIMSOPT_RUN_LEGACY_VMEC_JAX_COMPARISON", "").lower()
    in ("1", "true", "yes")
)


@unittest.skipUnless(
    RUN_LEGACY_COMPARISON,
    "Set SIMSOPT_RUN_LEGACY_VMEC_JAX_COMPARISON=1 to run legacy VMEC comparisons.",
)
@unittest.skipUnless(
    LEGACY_COMPARISON_DEPS_FOUND,
    "legacy VMEC, virtual_casing, vmec_jax, or virtual_casing_jax not found",
)
class LegacyVmecJaxComparisonTests(unittest.TestCase):
    def test_reduced_finite_beta_comparison(self):
        try:
            vmec_jax._compat.enable_x64(True)
        except AttributeError:
            pass

        input_file = os.path.abspath(
            os.path.join(
                ROOT,
                "examples",
                "3_Advanced",
                "inputs",
                "input.QH_finitebeta",
            )
        )

        with TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                legacy = Vmec(
                    input_file,
                    verbose=False,
                    keep_all_files=False,
                    nphi=12,
                    ntheta=12,
                    range_surface="half period",
                )
                jax_vmec = VmecJax(
                    input_file,
                    verbose=False,
                    keep_all_files=False,
                    nphi=12,
                    ntheta=12,
                    range_surface="half period",
                )
                legacy.run()
                jax_vmec.run()

                np.testing.assert_allclose(jax_vmec.aspect(), legacy.aspect(), rtol=1e-10)
                np.testing.assert_allclose(
                    jax_vmec.mean_iota(), legacy.mean_iota(), rtol=1e-10
                )
                np.testing.assert_allclose(
                    jax_vmec.external_current(), legacy.external_current(), rtol=1e-10
                )

                surfaces = [0.25, 0.5, 0.75]
                qs_legacy = QuasisymmetryRatioResidual(
                    legacy,
                    surfaces,
                    helicity_m=1,
                    helicity_n=-1,
                    ntheta=12,
                    nphi=12,
                ).total()
                qs_jax = QuasisymmetryRatioResidualJax(
                    jax_vmec,
                    surfaces,
                    helicity_m=1,
                    helicity_n=-1,
                    ntheta=12,
                    nphi=12,
                ).total()
                self.assertLess(abs(qs_legacy - qs_jax) / abs(qs_legacy), 1e-4)

                vc_legacy = VirtualCasing.from_vmec(
                    legacy,
                    src_nphi=12,
                    src_ntheta=12,
                    trgt_nphi=12,
                    trgt_ntheta=12,
                    digits=6,
                    filename=None,
                )
                vc_jax_same_field = VirtualCasingJax.from_vmec(
                    legacy,
                    src_nphi=12,
                    src_ntheta=12,
                    trgt_nphi=12,
                    trgt_ntheta=12,
                    digits=6,
                    filename=None,
                )
                same_field_error = (
                    np.linalg.norm(
                        vc_jax_same_field.B_external_normal
                        - vc_legacy.B_external_normal
                    )
                    / np.linalg.norm(vc_legacy.B_external_normal)
                )
                self.assertLess(same_field_error, 1e-3)

                vc_jax_full_path = VirtualCasingJax.from_vmec(
                    jax_vmec,
                    src_nphi=12,
                    src_ntheta=12,
                    trgt_nphi=12,
                    trgt_ntheta=12,
                    digits=6,
                    filename=None,
                )
                full_path_error = (
                    np.linalg.norm(
                        vc_jax_full_path.B_external_normal
                        - vc_legacy.B_external_normal
                    )
                    / np.linalg.norm(vc_legacy.B_external_normal)
                )
                self.assertLess(full_path_error, 5e-3)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
