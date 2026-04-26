import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

try:
    import virtual_casing_jax
except ImportError:
    virtual_casing_jax = None

try:
    from virtual_casing_jax.functional import compute_external_B_normal_functional
except ImportError:
    compute_external_B_normal_functional = None

from simsopt.mhd import (
    B_external_normal_from_data,
    B_external_normal_jvp_from_data,
    VirtualCasingJax,
    Vmec,
    VmecJax,
)

from . import TEST_DIR


VARIABLES = [
    "src_nphi",
    "src_ntheta",
    "src_phi",
    "src_theta",
    "trgt_nphi",
    "trgt_ntheta",
    "trgt_phi",
    "trgt_theta",
    "gamma",
    "unit_normal",
    "B_total",
    "B_external",
    "B_external_normal",
]


@unittest.skipIf(virtual_casing_jax is None, "virtual_casing_jax not found")
class VirtualCasingJaxTests(unittest.TestCase):
    def test_vmec_and_vmec_jax_inputs_agree(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        vc = VirtualCasingJax.from_vmec(
            Vmec(filename),
            src_nphi=6,
            src_ntheta=7,
            trgt_nphi=6,
            trgt_ntheta=7,
            digits=3,
            filename=None,
        )
        vc_j = VirtualCasingJax.from_vmec(
            VmecJax(filename),
            src_nphi=6,
            src_ntheta=7,
            trgt_nphi=6,
            trgt_ntheta=7,
            digits=3,
            filename=None,
        )

        for variable in VARIABLES:
            np.testing.assert_allclose(getattr(vc_j, variable), getattr(vc, variable))

    def test_vacuum_has_small_normal_field(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        vc = VirtualCasingJax.from_vmec(
            VmecJax(filename),
            src_nphi=8,
            src_ntheta=9,
            trgt_nphi=8,
            trgt_ntheta=9,
            digits=4,
            filename=None,
        )
        np.testing.assert_allclose(vc.B_external_normal, 0, atol=0.005)

    def test_save_load(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        with TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, "vcasing.nc")
            vc1 = VirtualCasingJax.from_vmec(
                VmecJax(filename),
                src_nphi=6,
                src_ntheta=7,
                trgt_nphi=5,
                trgt_ntheta=6,
                digits=3,
                filename=outfile,
            )
            vc2 = VirtualCasingJax.load(outfile)

        for variable in VARIABLES:
            np.testing.assert_allclose(getattr(vc1, variable), getattr(vc2, variable))

    @unittest.skipIf(
        compute_external_B_normal_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_from_data_matches_class(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        vc = VirtualCasingJax.from_vmec(
            VmecJax(filename),
            src_nphi=6,
            src_ntheta=7,
            trgt_nphi=6,
            trgt_ntheta=7,
            digits=3,
            filename=None,
        )
        Bnormal = B_external_normal_from_data(
            vc.gamma,
            vc.B_total,
            vc.nfp,
            True,
            digits=3,
            trgt_nphi=vc.trgt_nphi,
            trgt_ntheta=vc.trgt_ntheta,
            unit_normal=vc.unit_normal,
        )

        np.testing.assert_allclose(Bnormal, vc.B_external_normal, rtol=1e-12, atol=1e-12)

    @unittest.skipIf(
        compute_external_B_normal_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_jvp_from_data(self):
        nphi = 5
        ntheta = 4
        phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        theta2d, phi2d = np.meshgrid(theta, phi)
        gamma = np.zeros((nphi, ntheta, 3))
        gamma[:, :, 0] = (2.0 + 0.3 * np.cos(theta2d)) * np.cos(phi2d)
        gamma[:, :, 1] = (2.0 + 0.3 * np.cos(theta2d)) * np.sin(phi2d)
        gamma[:, :, 2] = 0.3 * np.sin(theta2d)
        B_total = 0.02 * gamma + 0.05
        unit_normal = np.zeros_like(gamma)
        unit_normal[:, :, 0] = np.cos(theta2d) * np.cos(phi2d)
        unit_normal[:, :, 1] = np.cos(theta2d) * np.sin(phi2d)
        unit_normal[:, :, 2] = np.sin(theta2d)

        tangent_gamma = np.zeros_like(gamma)
        tangent_gamma[0, 0, 0] = 1.0
        tangent_B_total = np.zeros_like(B_total)
        tangent_B_total[1, 2, 1] = -0.3
        tangent_unit_normal = np.zeros_like(unit_normal)
        tangent_unit_normal[2, 1, 2] = 0.2

        Bnormal, dBnormal = B_external_normal_jvp_from_data(
            gamma,
            B_total,
            tangent_gamma,
            tangent_B_total,
            nfp=1,
            stellsym=False,
            digits=4,
            unit_normal=unit_normal,
            tangent_unit_normal=tangent_unit_normal,
        )
        eps = 1e-5
        Bnormal_plus = B_external_normal_from_data(
            gamma + eps * tangent_gamma,
            B_total + eps * tangent_B_total,
            1,
            False,
            digits=4,
            unit_normal=unit_normal + eps * tangent_unit_normal,
        )
        Bnormal_minus = B_external_normal_from_data(
            gamma - eps * tangent_gamma,
            B_total - eps * tangent_B_total,
            1,
            False,
            digits=4,
            unit_normal=unit_normal - eps * tangent_unit_normal,
        )
        fd = (np.sum(Bnormal_plus**2) - np.sum(Bnormal_minus**2)) / (2 * eps)
        jvp = np.sum(2 * Bnormal * dBnormal)

        np.testing.assert_allclose(jvp, fd, rtol=5e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
