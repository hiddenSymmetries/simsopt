import os
from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

import numpy as np

try:
    import vmec_jax
except ImportError:
    vmec_jax = None

from simsopt.mhd import B_cartesian_jax, B_cartesian_jax_tangent_columns, Vmec, VmecJax
from simsopt.mhd.vmec_diagnostics import B_cartesian
from simsopt.mhd.profiles import ProfilePolynomial
from simsopt.geo.surface import Surface

from . import TEST_DIR


@unittest.skipIf(vmec_jax is None, "vmec_jax not found")
class VmecJaxInitializedFromWout(unittest.TestCase):
    def test_diagnostics_match_vmec(self):
        filename = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")
        vmec = Vmec(filename)
        vmec_j = VmecJax(filename)

        self.assertEqual(vmec_j.wout.ns, vmec.wout.ns)
        self.assertEqual(vmec_j.wout.nfp, vmec.wout.nfp)
        np.testing.assert_allclose(vmec_j.wout.rmnc, vmec.wout.rmnc)
        np.testing.assert_allclose(vmec_j.wout.zmns, vmec.wout.zmns)
        np.testing.assert_allclose(vmec_j.aspect(), vmec.aspect())
        np.testing.assert_allclose(vmec_j.volume(), vmec.volume())
        np.testing.assert_allclose(vmec_j.iota_axis(), vmec.iota_axis())
        np.testing.assert_allclose(vmec_j.iota_edge(), vmec.iota_edge())
        np.testing.assert_allclose(vmec_j.mean_iota(), vmec.mean_iota())
        np.testing.assert_allclose(vmec_j.mean_shear(), vmec.mean_shear())
        np.testing.assert_allclose(vmec_j.vacuum_well(), vmec.vacuum_well())

    def test_external_current_matches_vmec(self):
        filename = os.path.join(
            TEST_DIR,
            "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc",
        )
        vmec = Vmec(filename)
        vmec_j = VmecJax(filename)
        np.testing.assert_allclose(vmec_j.external_current(), vmec.external_current())

    def test_get_max_mn_from_wout(self):
        filename = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")
        vmec_j = VmecJax(filename)
        self.assertEqual(vmec_j.get_max_mn(), (vmec_j.wout.mpol, vmec_j.wout.ntor))

    def test_B_cartesian_jax_matches_vmec_diagnostic(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        vmec = Vmec(filename)
        vmec_j = VmecJax(filename)

        B_ref = np.asarray(B_cartesian(vmec, nphi=6, ntheta=7, range="half period"))
        B_jax = np.asarray(B_cartesian_jax(vmec_j, nphi=6, ntheta=7, range="half period"))
        B_method = np.asarray(vmec_j.B_cartesian(nphi=6, ntheta=7, range="half period"))

        np.testing.assert_allclose(B_jax, B_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(B_method, B_ref, rtol=1e-12, atol=1e-12)

    def test_error_on_rerun(self):
        filename = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")
        vmec_j = VmecJax(filename)
        _ = vmec_j.mean_iota()
        vmec_j.boundary.set_rc(1, 0, 2.0)
        with self.assertRaises(RuntimeError):
            vmec_j.mean_iota()


@unittest.skipIf(vmec_jax is None, "vmec_jax not found")
class VmecJaxInitializedFromInput(unittest.TestCase):
    def test_init_from_file(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        vmec_j = VmecJax(filename)

        self.assertEqual(vmec_j.indata.nfp, 3)
        self.assertEqual(vmec_j.indata.mpol, 4)
        self.assertEqual(vmec_j.indata.ntor, 3)
        self.assertEqual(vmec_j.boundary.mpol, 4)
        self.assertEqual(vmec_j.boundary.ntor, 3)
        self.assertAlmostEqual(vmec_j.boundary.get_rc(0, 0), 1.3782)
        self.assertAlmostEqual(vmec_j.boundary.get_zs(1, 0), 4.6465e-01)
        self.assertAlmostEqual(vmec_j.boundary.get_zs(1, 1), 1.6516e-01)
        self.assertEqual(vmec_j.indata.ncurr, 1)
        self.assertFalse(vmec_j.free_boundary)
        self.assertTrue(vmec_j.need_to_run_code)

    def test_set_dofs_updates_indata(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        vmec_j = VmecJax(filename)
        vmec_j.set_dofs([0.4, 2.0, 3.0])
        self.assertEqual(vmec_j.indata.phiedge, 0.4)
        self.assertEqual(vmec_j.indata.curtor, 2.0)
        self.assertEqual(vmec_j.indata.pres_scale, 3.0)
        self.assertTrue(vmec_j.need_to_run_code)

    def test_write_input_uses_boundary_dofs(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        vmec_j = VmecJax(filename)
        vmec_j.boundary.set_rc(1, 0, 0.25)
        text = vmec_j.get_input()
        self.assertIn("RBC(0,1)", text)
        self.assertIn("2.5000000000000000E-01", text)

    def test_profile_objects_update_indata(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        vmec_j = VmecJax(filename)
        vmec_j.n_pressure = 3
        vmec_j.pressure_profile = ProfilePolynomial([3.0, 2.0, -1.0])
        vmec_j.n_current = 2
        vmec_j.current_profile = ProfilePolynomial([2.0, 0.0])
        vmec_j.n_iota = 2
        vmec_j.iota_profile = ProfilePolynomial([0.6, 0.1])

        vmec_j.set_indata()

        np.testing.assert_allclose(np.asarray(vmec_j.indata.get("AM")), [3.0, 2.0, -1.0])
        np.testing.assert_allclose(np.asarray(vmec_j.indata.get("AC")), [2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.asarray(vmec_j.indata.get("AI")), [0.6, 0.1], atol=1e-12)
        self.assertEqual(vmec_j.indata.pres_scale, 1.0)
        self.assertAlmostEqual(vmec_j.indata.curtor, 2.0)

    def test_run_low_res_matches_reference(self):
        input_file = os.path.join(TEST_DIR, "input.li383_low_res")
        reference = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        with TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                vmec_j = VmecJax(input_file, verbose=False, keep_all_files=True)
                vmec_j.run()
            finally:
                os.chdir(cwd)

        self.assertLess(vmec_j.wout.fsqr, 1e-10)
        self.assertLess(vmec_j.wout.fsqz, 1e-10)
        self.assertLess(vmec_j.wout.fsql, 1e-10)
        np.testing.assert_allclose(vmec_j.aspect(), reference.aspect(), rtol=1e-12)
        np.testing.assert_allclose(vmec_j.volume(), reference.volume(), rtol=1e-12)
        np.testing.assert_allclose(vmec_j.mean_iota(), reference.mean_iota(), atol=2e-3)

    @unittest.skipIf(
        not hasattr(vmec_jax, "FixedBoundaryExactOptimizer")
        or not hasattr(vmec_jax, "b_cartesian_from_state"),
        "vmec_jax exact tangent helpers not found",
    )
    def test_B_cartesian_tangent_columns_match_exact_jacobian(self):
        from vmec_jax._compat import enable_x64, jnp
        from vmec_jax.grids import AngleGrid
        from vmec_jax.static import build_static

        enable_x64(True)
        input_file = os.path.join(TEST_DIR, "input.li383_low_res")
        cfg, indata = vmec_jax.load_config(input_file)
        static = vmec_jax.build_static(cfg)
        boundary = vmec_jax.boundary_from_indata(indata, static.modes)
        specs = vmec_jax.boundary_param_specs(
            boundary,
            static.modes,
            max_mode=1,
            min_coeff=0.0,
            include=("rc", "zs"),
            fix=("rc00",),
        )[:2]
        params = np.zeros(len(specs))
        nphi = 4
        ntheta = 5
        phi = np.asarray(
            Surface.get_phi_quadpoints(
                range="half period", nphi=nphi, nfp=static.cfg.nfp
            )
        )
        theta = np.asarray(Surface.get_theta_quadpoints(ntheta=ntheta))
        grid = AngleGrid(
            theta=theta * (2 * np.pi),
            zeta=phi * (2 * np.pi * static.cfg.nfp),
            nfp=static.cfg.nfp,
        )
        field_static = build_static(
            replace(static.cfg, ntheta=ntheta, nzeta=nphi),
            grid=grid,
        )

        def residuals_fn(state):
            B = vmec_jax.b_cartesian_from_state(
                state,
                field_static,
                indata=indata,
                signgs=exact_opt._signgs,
            )
            return jnp.ravel(B)

        exact_opt = vmec_jax.FixedBoundaryExactOptimizer(
            static,
            indata,
            boundary,
            specs,
            residuals_fn,
            inner_max_iter=2,
            inner_ftol=1e-5,
        )
        B, B_tangents = B_cartesian_jax_tangent_columns(
            exact_opt,
            params,
            quadpoints_phi=phi,
            quadpoints_theta=theta,
        )
        residuals = exact_opt.residual_fun(params)
        jacobian = exact_opt.jacobian_fun(params)
        B_flat = np.transpose(B, (1, 0, 2)).reshape(-1)
        tangent_jacobian = np.transpose(B_tangents, (1, 0, 2, 3)).reshape(
            (-1, len(specs))
        )

        np.testing.assert_allclose(B_flat, residuals, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            tangent_jacobian, jacobian, rtol=1e-12, atol=1e-12
        )
