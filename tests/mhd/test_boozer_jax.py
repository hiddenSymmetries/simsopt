import os
import unittest

import numpy as np
from scipy.io import netcdf_file

try:
    import booz_xform_jax
except ImportError:
    booz_xform_jax = None

from simsopt.mhd import BoozerJax, QuasisymmetryJax, Vmec, VmecJax

from . import TEST_DIR


@unittest.skipIf(booz_xform_jax is None, "booz_xform_jax not found")
class BoozerJaxTests(unittest.TestCase):
    def test_register(self):
        b = BoozerJax(None)
        self.assertEqual(b.s, set())
        QuasisymmetryJax(b, 0.5, 1, 1)
        self.assertEqual(b.s, {0.5})
        QuasisymmetryJax(b, 0.75, 1, 0)
        self.assertEqual(b.s, {0.5, 0.75})
        QuasisymmetryJax(b, [0.1, 0.2], 1, 0)
        self.assertEqual(b.s, {0.1, 0.2, 0.5, 0.75})

    def test_li383_matches_reference_boozmn(self):
        v = VmecJax(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        b = BoozerJax(v, mpol=32, ntor=16)
        q = QuasisymmetryJax(b, [0.0, 1.0], 1, 0)
        _ = q.J()

        self.assertTrue(hasattr(b.bx, "_last_jax_output"))
        np.testing.assert_allclose(b.bx.compute_surfs, [0, 14])
        self.assertEqual(b.s_to_index, {0.0: 0, 1.0: 1})

        f = netcdf_file(os.path.join(TEST_DIR, "boozmn_li383_low_res.nc"), mmap=False)
        bmnc_ref = f.variables["bmnc_b"][()].transpose()
        f.close()

        np.testing.assert_allclose(b.bx.bmnc_b[:, 0], bmnc_ref[:, 0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(b.bx.bmnc_b[:, 1], bmnc_ref[:, -1], atol=1e-12, rtol=1e-12)

    def test_vmec_and_vmec_jax_inputs_agree(self):
        filename = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")
        b_old_wout = BoozerJax(Vmec(filename), mpol=32, ntor=16)
        b_jax_wout = BoozerJax(VmecJax(filename), mpol=32, ntor=16)

        QuasisymmetryJax(b_old_wout, [0.0, 1.0], 1, 0).J()
        QuasisymmetryJax(b_jax_wout, [0.0, 1.0], 1, 0).J()

        np.testing.assert_allclose(b_jax_wout.bx.bmnc_b, b_old_wout.bx.bmnc_b)
        np.testing.assert_allclose(b_jax_wout.bx.xm_b, b_old_wout.bx.xm_b)
        np.testing.assert_allclose(b_jax_wout.bx.xn_b, b_old_wout.bx.xn_b)


if __name__ == "__main__":
    unittest.main()
