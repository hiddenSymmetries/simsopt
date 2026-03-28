import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from simsopt.geo import FiniteBetaBoozerSurface, Volume
from simsopt.geo.surfaceobjectives import MU0, finite_beta_sheet_current, surface_field_nonquasisymmetric_ratio

from .surface_test_helpers import get_boozer_surface

try:
    import vmec as vmec_mod
except ImportError:
    vmec_mod = None

try:
    import virtual_casing as virtual_casing_mod
except ImportError:
    virtual_casing_mod = None

from simsopt.mhd import Vmec, VirtualCasing


class FiniteBetaBoozerSurfaceTests(unittest.TestCase):
    def test_init_with_scalar_pressure_jump(self):
        bs, boozer_surface = get_boozer_surface(converge=False)
        finite_beta = FiniteBetaBoozerSurface(
            bs,
            boozer_surface.surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=123.4,
        )

        self.assertEqual(finite_beta.resolve_pressure_jump(), 123.4)
        self.assertTrue(finite_beta.need_to_run_code)
        self.assertEqual(finite_beta.solve_type, 'ls')

    def test_init_with_callable_pressure_jump(self):
        bs, boozer_surface = get_boozer_surface(converge=False)
        finite_beta = FiniteBetaBoozerSurface(
            bs,
            boozer_surface.surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=lambda surface: 10.0,
        )

        self.assertAlmostEqual(finite_beta.resolve_pressure_jump(), 10.0)

    def test_invalid_pressure_jump_raises(self):
        bs, boozer_surface = get_boozer_surface(converge=False)
        finite_beta = FiniteBetaBoozerSurface(
            bs,
            boozer_surface.surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=float('nan'),
        )

        with self.assertRaises(ValueError):
            finite_beta.resolve_pressure_jump()

    def test_run_code_solves_synthetic_fixed_surface_problem(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        surface = boozer_surface.surface
        iota_true = -0.31
        G_fixed = 1.4
        tang = surface.gammadash1() + iota_true * surface.gammadash2()
        tang_norm_sq = np.sum(tang**2, axis=2)
        B_in = (G_fixed / tang_norm_sq)[:, :, None] * tang
        B_out = B_in.copy()

        finite_beta = FiniteBetaBoozerSurface(
            None,
            surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=0.0,
            options={'verbose': False, 'ls_tol': 1e-12, 'ls_max_nfev': 80},
        )

        initial = finite_beta.residual_blocks(iota=-0.1, G=G_fixed, I=0.0, B_in=B_in, B_out=B_out)
        initial_norm = np.linalg.norm(initial['residual'])
        result = finite_beta.run_code(iota=-0.1, G=G_fixed, I=0.0, B_in=B_in, B_out=B_out, optimize_G=False)

        self.assertTrue(result['success'])
        self.assertLess(result['residual_norm'], 1e-8)
        self.assertLess(result['residual_norm'], initial_norm)
        self.assertAlmostEqual(result['iota'], iota_true, places=8)
        self.assertAlmostEqual(result['current_potential'][0, 0], 0.0)
        self.assertIsNotNone(result['least_squares_result'].jac)

    def test_parameter_jacobian_matches_finite_difference(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        surface = boozer_surface.surface
        shape = surface.gamma().shape
        B_in = np.zeros(shape)
        B_out = np.zeros(shape)
        B_in[:, :, 0] = 0.7
        B_in[:, :, 2] = 0.2
        B_out[:, :, 1] = 0.4

        finite_beta = FiniteBetaBoozerSurface(
            None,
            surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=0.0,
            options={'verbose': False},
        )

        iota0 = -0.17
        G0 = 1.3
        I0 = 0.08
        potential0 = np.zeros(shape[:2])
        x0 = finite_beta._pack_state(iota0, G0, I0, potential0, True, True, True, True)

        def residual_from_state(state):
            _, iota, G, I, potential = finite_beta._unpack_state(state, iota0, G0, I0, potential0, True, True, True, True)
            result = finite_beta.residual_blocks(iota=iota, G=G, I=I, current_potential=potential, B_in=B_in, B_out=B_out)
            return finite_beta._weighted_residual_vector(result['blocks'])

        analytic = finite_beta.residual_parameter_jacobian(
            iota=iota0,
            G=G0,
            I=I0,
            B_in=B_in,
            B_out=B_out,
            optimize_iota=True,
            optimize_G=True,
            optimize_I=True,
            optimize_current_potential=True,
        )

        np.random.seed(2)
        direction = np.random.standard_normal(x0.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-7
        fd = (residual_from_state(x0 + eps * direction) - residual_from_state(x0 - eps * direction)) / (2 * eps)
        linearized = analytic @ direction
        assert_allclose(linearized, fd, rtol=5e-5, atol=5e-7)

    def test_surface_jacobian_matches_finite_difference_for_fixed_fields(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        surface = boozer_surface.surface
        shape = surface.gamma().shape
        B_in = np.zeros(shape)
        B_out = np.zeros(shape)
        B_in[:, :, 0] = 0.7
        B_in[:, :, 1] = -0.2
        B_in[:, :, 2] = 0.15
        B_out[:, :, 0] = 0.1
        B_out[:, :, 1] = 0.45
        B_out[:, :, 2] = -0.35
        phi = surface.quadpoints_phi[:, None]
        theta = surface.quadpoints_theta[None, :]
        potential0 = 1e-3 * (np.sin(2 * np.pi * phi) + 0.6 * np.cos(2 * np.pi * theta))

        finite_beta = FiniteBetaBoozerSurface(
            None,
            surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=0.0,
            options={'verbose': False},
        )

        iota0 = -0.17
        G0 = 1.3
        I0 = 0.08
        x0 = surface.x.copy()

        def residual_from_dofs(dofs):
            surface.x = dofs
            result = finite_beta.residual_blocks(
                iota=iota0,
                G=G0,
                I=I0,
                current_potential=potential0,
                B_in=B_in,
                B_out=B_out,
            )
            return finite_beta._weighted_residual_vector(result['blocks'])

        try:
            analytic = finite_beta.residual_surface_jacobian(
                iota=iota0,
                G=G0,
                I=I0,
                current_potential=potential0,
                B_in=B_in,
                B_out=B_out,
            )

            rng = np.random.default_rng(4)
            direction = rng.standard_normal(x0.shape)
            direction /= np.linalg.norm(direction)
            eps = 1e-7
            fd = (residual_from_dofs(x0 + eps * direction) - residual_from_dofs(x0 - eps * direction)) / (2 * eps)
            linearized = analytic @ direction
            assert_allclose(linearized, fd, rtol=1e-4, atol=2e-6)
        finally:
            surface.x = x0

    def test_pressure_jump_case_has_nontrivial_current_potential_residuals(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        surface = boozer_surface.surface
        iota_true = -0.25
        G_fixed = 1.1
        tang = surface.gammadash1() + iota_true * surface.gammadash2()
        tang_norm_sq = np.sum(tang**2, axis=2)
        B_in = (G_fixed / tang_norm_sq)[:, :, None] * tang

        theta = surface.quadpoints_theta[None, :]
        potential_target = 1e-3 * np.sin(2 * np.pi * theta) * np.ones(surface.gamma().shape[:2])
        dphi = np.zeros_like(potential_target)
        dtheta = 1e-3 * 2 * np.pi * np.cos(2 * np.pi * theta) * np.ones(surface.gamma().shape[:2])
        sheet_current_target = finite_beta_sheet_current(
            surface,
            current_potential_derivatives=(dphi, dtheta),
        )
        B_out = B_in - MU0 * np.cross(surface.unitnormal(), sheet_current_target)
        pressure_jump = np.mean(np.sum(B_out**2, axis=2) - np.sum(B_in**2, axis=2)) / (2 * MU0)

        finite_beta = FiniteBetaBoozerSurface(
            None,
            surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=pressure_jump,
            options={'verbose': False},
        )

        zero_potential = finite_beta.residual_blocks(
            iota=iota_true,
            G=G_fixed,
            I=0.0,
            B_in=B_in,
            B_out=B_out,
        )
        with_potential = finite_beta.residual_blocks(
            iota=iota_true,
            G=G_fixed,
            I=0.0,
            current_potential=potential_target,
            B_in=B_in,
            B_out=B_out,
        )

        self.assertGreater(abs(pressure_jump), 0.0)
        self.assertGreater(np.linalg.norm(potential_target), 1e-6)
        self.assertGreater(np.linalg.norm(with_potential['blocks']['sheet_current']), 0.0)
        self.assertLess(
            np.linalg.norm(with_potential['blocks']['jump']),
            np.linalg.norm(zero_potential['blocks']['jump']),
        )
        self.assertGreater(np.linalg.norm(with_potential['blocks']['pressure']), 0.0)

    def test_run_code_can_optimize_surface_with_callable_fields(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        target_label = boozer_surface.label.J()
        iota_target = -0.22
        G_target = 1.05

        def B_in_provider(surface):
            tang = surface.gammadash1() + iota_target * surface.gammadash2()
            tang_norm_sq = np.sum(tang**2, axis=2)
            return (G_target / tang_norm_sq)[:, :, None] * tang

        def B_out_provider(surface):
            return B_in_provider(surface)

        perturbed_surface = boozer_surface.surface.__class__(
            mpol=boozer_surface.surface.mpol,
            ntor=boozer_surface.surface.ntor,
            stellsym=boozer_surface.surface.stellsym,
            nfp=boozer_surface.surface.nfp,
            quadpoints_phi=boozer_surface.surface.quadpoints_phi,
            quadpoints_theta=boozer_surface.surface.quadpoints_theta,
            dofs=boozer_surface.surface.dofs,
        )
        perturbed_surface.x = perturbed_surface.x + 1e-3 * np.random.default_rng(3).standard_normal(perturbed_surface.x.shape)
        finite_beta = FiniteBetaBoozerSurface(
            None,
            perturbed_surface,
            Volume(perturbed_surface),
            target_label,
            pressure_jump=0.0,
            options={'verbose': False, 'ls_tol': 1e-10, 'ls_max_nfev': 60},
        )

        label_before = abs(finite_beta.label.J() - target_label)
        result = finite_beta.run_code(
            iota=iota_target,
            G=G_target,
            I=0.0,
            B_in=B_in_provider,
            B_out=B_out_provider,
            optimize_iota=False,
            optimize_G=False,
            optimize_I=False,
            optimize_current_potential=False,
            optimize_surface=True,
        )

        self.assertTrue(result['success'])
        self.assertLess(abs(finite_beta.label.J() - target_label), label_before)
        self.assertAlmostEqual(finite_beta.surface.gamma()[0, 0, 2], 0.0, places=6)

    def test_run_code_can_optimize_surface_with_fixed_fields(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        target_label = boozer_surface.label.J()
        perturbed_surface = boozer_surface.surface.__class__(
            mpol=boozer_surface.surface.mpol,
            ntor=boozer_surface.surface.ntor,
            stellsym=boozer_surface.surface.stellsym,
            nfp=boozer_surface.surface.nfp,
            quadpoints_phi=boozer_surface.surface.quadpoints_phi,
            quadpoints_theta=boozer_surface.surface.quadpoints_theta,
            dofs=boozer_surface.surface.dofs,
        )
        perturbed_surface.x = perturbed_surface.x + 1e-3 * np.random.default_rng(5).standard_normal(perturbed_surface.x.shape)
        finite_beta = FiniteBetaBoozerSurface(
            None,
            perturbed_surface,
            Volume(perturbed_surface),
            target_label,
            pressure_jump=0.0,
            options={'verbose': False, 'ls_tol': 1e-10, 'ls_max_nfev': 60},
        )

        zeros = np.zeros(perturbed_surface.gamma().shape)
        label_before = abs(finite_beta.label.J() - target_label)
        result = finite_beta.run_code(
            iota=0.0,
            G=0.0,
            I=0.0,
            B_in=zeros,
            B_out=zeros,
            optimize_iota=False,
            optimize_G=False,
            optimize_I=False,
            optimize_current_potential=False,
            optimize_surface=True,
        )

        self.assertTrue(result['success'])
        self.assertLess(abs(finite_beta.label.J() - target_label), label_before)
        self.assertAlmostEqual(finite_beta.surface.gamma()[0, 0, 2], 0.0, places=6)
        self.assertEqual(result['least_squares_result'].jac.shape[1], perturbed_surface.x.size)

    def test_surface_field_nonquasisymmetric_ratio_detects_qs_and_nonqs_content(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        surface = boozer_surface.surface
        theta = surface.quadpoints_theta[None, :]
        phi = surface.quadpoints_phi[:, None]

        qs_field = np.zeros(surface.gamma().shape)
        qs_field[:, :, 0] = 1.0 + 0.2 * np.cos(2 * np.pi * theta)
        nonqs_field = qs_field.copy()
        nonqs_field[:, :, 0] += 0.1 * np.cos(2 * np.pi * phi)

        qs_ratio = surface_field_nonquasisymmetric_ratio(surface, qs_field)
        nonqs_ratio = surface_field_nonquasisymmetric_ratio(surface, nonqs_field)

        self.assertLess(qs_ratio, 1e-12)
        self.assertGreater(nonqs_ratio, 1e-6)

    def test_resolve_field_components_from_virtual_casing(self):
        bs, boozer_surface = get_boozer_surface(converge=False)
        shape = boozer_surface.surface.gamma().shape
        B_total = np.full(shape, 3.0)
        B_external = np.full(shape, 1.25)
        finite_beta = FiniteBetaBoozerSurface(
            bs,
            boozer_surface.surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=0.0,
            virtual_casing=SimpleNamespace(B_total=B_total, B_external=B_external),
        )

        B_in, B_out = finite_beta.resolve_field_components()
        assert_allclose(B_in, B_total - B_external)
        assert_allclose(B_out, B_external)

    def test_explicit_fields_do_not_require_biotsavart(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        shape = boozer_surface.surface.gamma().shape
        B_in = np.zeros(shape)
        B_out = np.zeros(shape)
        finite_beta = FiniteBetaBoozerSurface(
            None,
            boozer_surface.surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=0.0,
        )

        resolved_B_in, resolved_B_out = finite_beta.resolve_field_components(B_in=B_in, B_out=B_out)
        assert_allclose(resolved_B_in, B_in)
        assert_allclose(resolved_B_out, B_out)

    def test_sheet_current_zero_without_potential(self):
        _, boozer_surface = get_boozer_surface(converge=False)
        current = finite_beta_sheet_current(boozer_surface.surface)
        assert_allclose(current, 0.0)

    def test_residual_blocks_match_vacuum_limit_with_explicit_fields(self):
        bs, boozer_surface = get_boozer_surface(converge=False)
        finite_beta = FiniteBetaBoozerSurface(
            bs,
            boozer_surface.surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=0.0,
        )

        surface = boozer_surface.surface
        shape = surface.gamma().shape
        constant_B = np.zeros(shape)
        constant_B[:, :, 2] = 1.0
        result = finite_beta.residual_blocks(
            iota=-0.4,
            G=1.2,
            I=0.0,
            B_in=constant_B,
            B_out=constant_B,
        )

        expected_boozer = 1.2 * constant_B - np.sum(constant_B**2, axis=2)[:, :, None] * (
            surface.gammadash1() - 0.4 * surface.gammadash2()
        )
        assert_allclose(result['blocks']['boozer'], expected_boozer)
        assert_allclose(result['blocks']['normal'], np.sum(constant_B * surface.unitnormal(), axis=2))
        assert_allclose(result['blocks']['pressure'], 0.0)
        assert_allclose(result['blocks']['jump'], 0.0)

    def test_residual_blocks_include_pressure_and_jump_terms(self):
        bs, boozer_surface = get_boozer_surface(converge=False)
        finite_beta = FiniteBetaBoozerSurface(
            bs,
            boozer_surface.surface,
            boozer_surface.label,
            boozer_surface.targetlabel,
            pressure_jump=5.0,
        )

        surface = boozer_surface.surface
        shape = surface.gamma().shape
        B_in = np.zeros(shape)
        B_out = np.zeros(shape)
        B_out[:, :, 0] = 1.0
        dphi = np.ones(shape[:2])
        dtheta = np.zeros(shape[:2])

        result = finite_beta.residual_blocks(
            iota=0.0,
            G=0.0,
            I=0.0,
            B_in=B_in,
            B_out=B_out,
            current_potential_derivatives=(dphi, dtheta),
        )

        expected_sheet_current = finite_beta_sheet_current(
            surface,
            current_potential_derivatives=(dphi, dtheta),
        )
        assert_allclose(result['blocks']['sheet_current'], expected_sheet_current)
        assert_allclose(result['blocks']['pressure'], 1.0 - 2.0 * MU0 * 5.0)
        assert_allclose(
            result['blocks']['jump'],
            MU0 * expected_sheet_current - np.cross(surface.unitnormal(), B_out - B_in),
        )

    @unittest.skipIf(vmec_mod is None or virtual_casing_mod is None, "vmec and virtual_casing python packages are required")
    def test_vmec_fixed_surface_smoke(self):
        test_dir = Path(__file__).resolve().parents[1] / 'test_files'
        vmec_filename = test_dir / 'input.W7-X_without_coil_ripple_beta0p05_d23p4_tm'
        vc_filename = vmec_filename.with_name(vmec_filename.name.replace('input.', 'vcasing_') + '.nc')

        vmec = Vmec(str(vmec_filename))
        vmec.run()
        if vc_filename.exists():
            vc = VirtualCasing.load(str(vc_filename))
        else:
            vc = VirtualCasing.from_vmec(vmec, src_nphi=8, src_ntheta=8, trgt_nphi=8, trgt_ntheta=8, filename=str(vc_filename))

        boundary_rz = vmec.boundary.__class__.from_nphi_ntheta(
            mpol=vmec.wout.mpol,
            ntor=vmec.wout.ntor,
            nfp=vmec.wout.nfp,
            stellsym=not bool(vmec.wout.lasym),
            nphi=8,
            ntheta=8,
            range='half period',
        )
        boundary_rz.x = vmec.boundary.x
        surface = get_boozer_surface(converge=False)[1].surface.__class__(
            mpol=vmec.wout.mpol,
            ntor=vmec.wout.ntor,
            nfp=vmec.wout.nfp,
            stellsym=not bool(vmec.wout.lasym),
            quadpoints_phi=boundary_rz.quadpoints_phi,
            quadpoints_theta=boundary_rz.quadpoints_theta,
        )
        surface.least_squares_fit(boundary_rz.gamma())
        vol = Volume(surface)
        finite_beta = FiniteBetaBoozerSurface(
            None,
            surface,
            vol,
            vol.J(),
            pressure_jump=0.0,
            options={'verbose': False, 'ls_max_nfev': 200},
        )

        initial = finite_beta.residual_blocks(
            iota=vmec.iota_edge(),
            G=0.0,
            I=0.0,
            B_in=vc.B_total - vc.B_external,
            B_out=vc.B_external,
        )
        result = finite_beta.run_code(
            iota=vmec.iota_edge(),
            G=0.0,
            I=0.0,
            B_in=vc.B_total - vc.B_external,
            B_out=vc.B_external,
            optimize_G=False,
        )

        self.assertTrue(result['success'])
        self.assertLess(result['residual_norm'], np.linalg.norm(initial['residual']))
