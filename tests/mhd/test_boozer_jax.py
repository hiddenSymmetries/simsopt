import os
import unittest

import numpy as np
from scipy.io import netcdf_file

import simsopt.mhd.boozer_jax as boozer_jax_module

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
        b.register(0.3)
        self.assertEqual(b.s, {0.3})
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            b.register(-0.1)
        QuasisymmetryJax(b, 0.5, 1, 1)
        self.assertEqual(b.s, {0.3, 0.5})
        QuasisymmetryJax(b, 0.75, 1, 0)
        self.assertEqual(b.s, {0.3, 0.5, 0.75})
        QuasisymmetryJax(b, [0.1, 0.2], 1, 0)
        self.assertEqual(b.s, {0.1, 0.2, 0.3, 0.5, 0.75})

    def test_dependency_and_equilibrium_guards(self):
        old_booz_xform_jax = boozer_jax_module.booz_xform_jax
        try:
            boozer_jax_module.booz_xform_jax = None
            with self.assertRaisesRegex(RuntimeError, "booz_xform_jax package"):
                BoozerJax(None)
        finally:
            boozer_jax_module.booz_xform_jax = old_booz_xform_jax

        b = BoozerJax(None)
        b.register(0.5)
        with self.assertRaisesRegex(ValueError, "equilibrium type"):
            b.run()

    def test_li383_matches_reference_boozmn(self):
        v = VmecJax(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        b = BoozerJax(v, mpol=32, ntor=16)
        q = QuasisymmetryJax(b, [0.0, 1.0], 1, 0)
        _ = q.J()
        b.run()
        self.assertEqual(b._calls, 1)

        self.assertTrue(hasattr(b.bx, "_last_jax_output"))
        np.testing.assert_allclose(b.bx.compute_surfs, [0, 14])
        self.assertEqual(b.s_to_index, {0.0: 0, 1.0: 1})

        f = netcdf_file(os.path.join(TEST_DIR, "boozmn_li383_low_res.nc"), mmap=False)
        bmnc_ref = f.variables["bmnc_b"][()].transpose()
        f.close()

        np.testing.assert_allclose(b.bx.bmnc_b[:, 0], bmnc_ref[:, 0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(b.bx.bmnc_b[:, 1], bmnc_ref[:, -1], atol=1e-12, rtol=1e-12)

    def test_init_from_asymmetric_wout_data(self):
        v = VmecJax(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        wout = v.wout
        wout.lasym = True
        wout.rmns = np.zeros_like(wout.rmnc)
        wout.zmnc = np.zeros_like(wout.zmns)
        wout.lmnc = np.zeros_like(wout.lmns)
        wout.bmns = np.zeros_like(wout.bmnc)
        wout.bsubumns = np.zeros_like(wout.bsubumnc)
        wout.bsubvmns = np.zeros_like(wout.bsubvmnc)

        b = BoozerJax(v, mpol=8, ntor=4)
        b._init_booz_xform_from_wout([0])

        self.assertTrue(b.bx.asym)
        self.assertEqual(b.bx.compute_surfs, [0])
        self.assertEqual(b.bx.mboz, 8)
        self.assertEqual(b.bx.nboz, 4)

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
