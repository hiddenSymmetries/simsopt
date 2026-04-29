import os
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import simsopt.mhd.vmec_jax as vmec_jax_module

try:
    import vmec_jax
except ImportError:
    vmec_jax = None

from simsopt.mhd import (
    B_cartesian_jax,
    B_cartesian_jax_tangent_columns,
    Vmec,
    VmecJax,
    make_vmec_jax_residuals_from_terms,
)
from simsopt.mhd.vmec_diagnostics import B_cartesian
from simsopt.mhd.profiles import ProfilePolynomial, ProfileSpline
from simsopt.geo.surface import Surface
from simsopt.geo import SurfaceRZFourier
from simsopt._core.util import ObjectiveFailure

from . import TEST_DIR


@unittest.skipIf(vmec_jax is None, "vmec_jax not found")
class VmecJaxInitializedFromWout(unittest.TestCase):
    def test_make_vmec_jax_residuals_from_terms(self):
        from vmec_jax._compat import jnp

        def term1(state):
            return jnp.asarray([state.value], dtype=jnp.float64)

        def term2(state):
            return jnp.asarray([[2.0 * state.value]], dtype=jnp.float64)

        self.assertIs(make_vmec_jax_residuals_from_terms([term1]), term1)

        residuals = make_vmec_jax_residuals_from_terms(
            [term1, term2],
            n_non_qs=1,
            qs_total_from_state=lambda state: state.value,
        )
        np.testing.assert_allclose(
            np.asarray(residuals(SimpleNamespace(value=3.0))),
            [3.0, 6.0],
        )
        self.assertEqual(residuals._n_non_qs, 1)
        self.assertEqual(
            residuals._qs_total_from_state(SimpleNamespace(value=4.0)),
            4.0,
        )
        with self.assertRaisesRegex(ValueError, "at least one VMEC-JAX"):
            make_vmec_jax_residuals_from_terms([])

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

    def test_B_cartesian_jax_path_input_grid_defaults_and_lasym_error(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        B_from_path = np.asarray(
            B_cartesian_jax(
                filename,
                quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False),
                quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
            )
        )
        self.assertEqual(B_from_path.shape, (3, 4, 5))

        vmec_j = VmecJax(filename, nphi=4, ntheta=5, range_surface="half period")
        B_from_boundary_grid = np.asarray(B_cartesian_jax(vmec_j))
        self.assertEqual(B_from_boundary_grid.shape, (3, 4, 5))

        vmec_j.wout.lasym = True
        with self.assertRaisesRegex(RuntimeError, "stellarator symmetry"):
            B_cartesian_jax(vmec_j, nphi=4, ntheta=5)

    def test_wout_guards_mpi_repr_and_missing_file_fallback(self):
        filename = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")
        vmec_j = VmecJax(filename)

        with self.assertRaisesRegex(RuntimeError, "initialized from a wout file"):
            vmec_j.set_indata()
        vmec_j.update_mpi("serial")
        self.assertEqual(vmec_j.mpi, "serial")
        self.assertIn("nfp=", repr(vmec_j))

        fallback = object.__new__(VmecJax)
        fallback.output_file = os.path.join(TEST_DIR, "missing_wout_for_jax_test.nc")
        fallback._wout_jax = SimpleNamespace(
            ns=3,
            nfp=2,
            mpol=2,
            ntor=1,
            lasym=np.bool_(False),
            volume_p=7.0,
            iotas=np.asarray([0.0, 0.3, 0.6]),
            matrix=np.arange(6).reshape((2, 3)),
        )
        VmecJax.load_wout(fallback)

        self.assertFalse(fallback.wout.lasym)
        self.assertEqual(fallback.wout.volume, 7.0)
        self.assertEqual(fallback.wout.ier_flag, 0)
        self.assertEqual(fallback.wout.matrix.shape, (3, 2))
        np.testing.assert_allclose(fallback.s_half_grid, [0.25, 0.75])

    def test_error_on_rerun(self):
        filename = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")
        vmec_j = VmecJax(filename)
        _ = vmec_j.mean_iota()
        vmec_j.boundary.set_rc(1, 0, 2.0)
        with self.assertRaises(RuntimeError):
            vmec_j.mean_iota()


@unittest.skipIf(vmec_jax is None, "vmec_jax not found")
class VmecJaxInitializedFromInput(unittest.TestCase):
    def test_defaults_invalid_filename_indata_adapter_and_dependency_guard(self):
        default_vmec = VmecJax(None, verbose=False, nphi=5, ntheta=6)
        self.assertTrue(default_vmec.runnable)

        with self.assertRaisesRegex(ValueError, "Invalid filename"):
            VmecJax("equilibrium.nc")

        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        vmec_j = VmecJax(filename)
        vmec_j.indata.scalars["LIST_VALUE"] = [1.0, 2.0]
        np.testing.assert_allclose(vmec_j.indata.list_value, [1.0, 2.0])
        vmec_j.indata.array_value = np.asarray([3.0, 4.0])
        self.assertEqual(vmec_j.indata.raw.scalars["ARRAY_VALUE"], [3.0, 4.0])
        vmec_j.indata.scalar_value = np.float64(2.5)
        self.assertEqual(vmec_j.indata.raw.scalars["SCALAR_VALUE"], 2.5)
        vmec_j.indata._local_note = "stored on adapter"
        self.assertEqual(vmec_j.indata._local_note, "stored on adapter")
        self.assertIs(vmec_j.indata.raw, vmec_j.indata._indata)
        self.assertIs(vmec_j.indata.indexed, vmec_j.indata.raw.indexed)
        self.assertIs(vmec_j.indata.scalars, vmec_j.indata.raw.scalars)
        self.assertEqual(vmec_j.indata.get_int("NFP"), 3)
        self.assertFalse(vmec_j.indata.get_bool("LASYM"))
        self.assertAlmostEqual(vmec_j.indata.get_float("PHIEDGE"), vmec_j.indata.phiedge)
        with self.assertRaises(AttributeError):
            _ = vmec_j.indata.this_does_not_exist

        old_vmec_jax = vmec_jax_module.vmec_jax_mod
        try:
            vmec_jax_module.vmec_jax_mod = None
            with self.assertRaisesRegex(RuntimeError, "requires the vmec_jax package"):
                VmecJax(filename)
        finally:
            vmec_jax_module.vmec_jax_mod = old_vmec_jax

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

    def test_spline_profile_types_and_invalid_profile_type(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        vmec_j = VmecJax(filename)
        pressure_profile = ProfileSpline(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([1.0, 2.0, 4.0]),
            degree=1,
        )
        current_profile = ProfileSpline(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([1.0, 2.0, 4.0]),
            degree=1,
        )
        iota_profile = ProfileSpline(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([1.0, 2.0, 4.0]),
            degree=1,
        )
        vmec_j.n_pressure = 3
        vmec_j.n_current = 3
        vmec_j.n_iota = 3
        vmec_j.pressure_profile = pressure_profile
        vmec_j.current_profile = current_profile
        vmec_j.iota_profile = iota_profile
        vmec_j.indata.raw.scalars["PMASS_TYPE"] = "cubic_spline"
        vmec_j.indata.raw.scalars["PCURR_TYPE"] = "line_segment"
        vmec_j.indata.raw.scalars["PIOTA_TYPE"] = b"akima_spline"

        vmec_j.set_indata()

        np.testing.assert_allclose(vmec_j.indata.raw.scalars["AM_AUX_S"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(vmec_j.indata.raw.scalars["AC_AUX_F"], [1.0, 2.0, 4.0])
        np.testing.assert_allclose(vmec_j.indata.raw.scalars["AI_AUX_S"], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(vmec_j.indata.curtor, current_profile(1.0))

        vmec_j.pressure_profile = ProfilePolynomial([1.0])
        vmec_j.current_profile = ProfilePolynomial([1.0])
        vmec_j.iota_profile = ProfilePolynomial([0.4])
        self.assertIsInstance(vmec_j.pressure_profile, ProfilePolynomial)
        self.assertIsInstance(vmec_j.current_profile, ProfilePolynomial)
        self.assertIsInstance(vmec_j.iota_profile, ProfilePolynomial)

        vmec_j.indata.raw.scalars["PMASS_TYPE"] = "unsupported"
        with self.assertRaisesRegex(RuntimeError, "power_series"):
            vmec_j.set_profile("pressure", "mass", "m")

    def test_boundary_setter_asymmetric_surface_and_get_max_mn(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        vmec_j = VmecJax(filename)
        boundary = SurfaceRZFourier.from_nphi_ntheta(
            nfp=3,
            stellsym=False,
            mpol=1,
            ntor=1,
            nphi=5,
            ntheta=6,
            range="field period",
        )
        boundary.set_rc(0, 0, 1.4)
        boundary.set_zs(1, 0, 0.2)
        boundary.set_rs(1, 0, 0.03)
        boundary.set_zc(1, 0, -0.04)
        vmec_j.boundary = boundary

        vmec_j.set_indata()

        self.assertTrue(vmec_j.need_to_run_code)
        self.assertTrue(vmec_j.indata.raw.scalars["LASYM"])
        self.assertIn("RBS", vmec_j.indata.raw.indexed)
        self.assertIn("ZBC", vmec_j.indata.raw.indexed)
        self.assertIn((0, 1), vmec_j.indata.raw.indexed["RBS"])
        self.assertIn((0, 1), vmec_j.indata.raw.indexed["ZBC"])

        vmec_j.indata.raw.indexed["RBC"][(5, 4)] = 0.01
        self.assertEqual(vmec_j.get_max_mn(), (4, 5))
        self.assertIn("nfp=3", repr(vmec_j))

    def test_asymmetric_surface_from_indata(self):
        class FakeInData:
            indexed = {
                "RBC": {(0, 0): 1.5},
                "ZBS": {(0, 1): 0.2},
                "RBS": {(0, 1): 0.03},
                "ZBC": {(0, 1): -0.04},
            }

            def get_int(self, name, default=0):
                return {"NFP": 2, "MPOL": 1, "NTOR": 0}.get(name, default)

            def get_bool(self, name, default=False):
                return True if name == "LASYM" else default

        surf = vmec_jax_module._surface_from_indata(
            FakeInData(),
            ntheta=5,
            nphi=6,
            range_surface="field period",
        )

        self.assertFalse(surf.stellsym)
        self.assertAlmostEqual(surf.get_rs(1, 0), 0.03)
        self.assertAlmostEqual(surf.get_zc(1, 0), -0.04)

    def test_run_failure_is_wrapped_as_objective_failure(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        old_run_fixed_boundary = vmec_jax_module.vmec_jax_mod.run_fixed_boundary
        try:
            def fail_run(*args, **kwargs):
                raise RuntimeError("forced failure")

            vmec_jax_module.vmec_jax_mod.run_fixed_boundary = fail_run
            with TemporaryDirectory() as tmp:
                cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    vmec_j = VmecJax(filename, verbose=False)
                    with self.assertRaises(ObjectiveFailure):
                        vmec_j.run()
                finally:
                    os.chdir(cwd)
        finally:
            vmec_jax_module.vmec_jax_mod.run_fixed_boundary = old_run_fixed_boundary

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
        B_from_state = np.asarray(
            B_cartesian_jax(
                vmec_j,
                nphi=4,
                ntheta=5,
                range="half period",
                use_wout_bsup=False,
            )
        )
        self.assertEqual(B_from_state.shape, (3, 4, 5))

    def test_B_cartesian_tangent_columns_guard_errors(self):
        with self.assertRaisesRegex(TypeError, "FixedBoundaryExactOptimizer"):
            B_cartesian_jax_tangent_columns(object(), [])

        if hasattr(vmec_jax, "b_cartesian_from_state"):
            input_file = os.path.join(TEST_DIR, "input.li383_low_res")
            cfg, _ = vmec_jax.load_config(input_file)
            static = vmec_jax.build_static(cfg)
            with self.assertRaisesRegex(RuntimeError, "b_cartesian_tangent_columns_fun"):
                B_cartesian_jax_tangent_columns(
                    SimpleNamespace(_static=static),
                    [],
                )

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
