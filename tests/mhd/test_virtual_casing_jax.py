import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

try:
    import virtual_casing_jax
except ImportError:
    virtual_casing_jax = None

from simsopt.mhd import VirtualCasingJax, Vmec, VmecJax

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


if __name__ == "__main__":
    unittest.main()
