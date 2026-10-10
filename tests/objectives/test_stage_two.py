"""The fused Stage-II objective against the native composite objective."""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt._core.optimizable import Optimizable
    from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
    from simsopt.geo import (
        CurveCurveDistance,
        CurveLength,
        CurveSurfaceDistance,
        CurveXYZFourier,
        LpCurveCurvature,
        MeanSquaredCurvature,
        SurfaceRZFourier,
        create_equally_spaced_curves,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt_jax.objectives import (
        StageTwoObjectiveConfig,
        fused_stage_two_objective,
        fused_stage_two_values,
        make_stage_two_problem,
        stage_two_geometric_penalty,
    )
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.objectives import JaxSquaredFlux


from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


_QA_INPUT = Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
_NCOILS = 3
# Thresholds that make every penalty active at the perturbed test state.
if JAX_IMPORT_ERROR is None:
    _ACTIVE = StageTwoObjectiveConfig(
        num_basecurves=_NCOILS,
        length_weight=1e-3,
        curve_curve_minimum_distance=0.6,
        curve_curve_weight=10.0,
        curve_surface_minimum_distance=0.4,
        curve_surface_weight=2.0,
        curvature_threshold=1.0,
        curvature_weight=1e-2,
        mean_squared_curvature_threshold=1.0,
        mean_squared_curvature_weight=1e-2,
    )


def _surface() -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(str(_QA_INPUT), range="half period", nphi=8, ntheta=9)


def _coils(surface, *, perturb: bool = True, shared_dofs: bool = False) -> tuple[list, list]:
    """Return base curves and symmetric coils with a fixed current and fixed curve DOFs.

    With ``shared_dofs``, one more coil's curve shares the DOFs object of the
    second base curve (on a rotated, offset grid) and its current shares the
    DOFs of the third base current.
    """
    base = create_equally_spaced_curves(
        _NCOILS, surface.nfp, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24
    )
    if perturb:
        rng = np.random.default_rng(17)
        for curve in base:
            curve.x = curve.x + 0.03 * rng.standard_normal(curve.x.shape)
    base[0].fix("xc(1)")
    base[1].fix("zs(1)")
    currents = [Current(1e5), Current(1.1e5), Current(0.9e5)]
    currents[0].fix_all()
    coils = coils_via_symmetries(base, currents, surface.nfp, True)
    if shared_dofs:
        twin = CurveXYZFourier(
            np.linspace(0, 1, 24, endpoint=False) + 0.011, 3, dofs=base[1].dofs
        )
        coils.append(Coil(twin, -0.5 * Current(0.0, dofs=currents[2].dofs)))
    return base, coils


def _native_composite(
    surface, base, coils, config: StageTwoObjectiveConfig
) -> Optimizable:
    """The native objective that ``config`` describes, as in the Stage-II examples.

    A ``None`` weight leaves its term out; a zero weight keeps ``0 * term``.
    """
    curves = [coil.curve for coil in coils]
    total_length = sum(CurveLength(curve) for curve in base)
    weighted_terms = [
        (config.length_weight, (
            total_length if config.length_target is None
            else QuadraticPenalty(total_length, config.length_target, config.length_target_mode)
        )),
        (config.curve_curve_weight, CurveCurveDistance(
            curves, config.curve_curve_minimum_distance, num_basecurves=config.num_basecurves
        )),
        (config.curve_surface_weight, CurveSurfaceDistance(
            curves, surface, config.curve_surface_minimum_distance
        )),
        (config.curvature_weight, sum(
            LpCurveCurvature(curve, 2, config.curvature_threshold) for curve in base
        )),
        (config.mean_squared_curvature_weight, sum(
            QuadraticPenalty(
                MeanSquaredCurvature(curve),
                config.mean_squared_curvature_threshold,
                config.mean_squared_curvature_target_mode,
            )
            for curve in base
        )),
    ]
    return SquaredFlux(surface, BiotSavart(coils)) + sum(
        weight * term for weight, term in weighted_terms if weight is not None
    )


def _problem(config: StageTwoObjectiveConfig, *, perturb: bool = True, shared_dofs: bool = False):
    surface = _surface()
    base, coils = _coils(surface, perturb=perturb, shared_dofs=shared_dofs)
    field = JaxBiotSavart(coils)
    flux = JaxSquaredFlux(surface, field)
    native = _native_composite(surface, base, coils, config)
    return field, flux, native, make_stage_two_problem(field, flux.fixed_surface_flux_spec(), config)


def _native_gradient(native: Optimizable, field: JaxBiotSavart) -> np.ndarray:
    """The native gradient in the order of the field's free DOFs."""
    return np.asarray(native.dJ(partials=True)(field))


if JAX_IMPORT_ERROR is None:
    _value_and_grad = jax.jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
    _objective = jax.jit(fused_stage_two_objective)
    _values = jax.jit(fused_stage_two_values)
























class TestStageTwo(JaxTestCase):
    def test_fused_objective_matches_native_composite(self):
        """The fused value and gradient match the native composite with shared DOFs."""
        for case_id_0, shared_dofs in zip(["symmetric", "shared_dofs"], [False, True], strict=True):
            for case_id_1, (length_target, length_mode, msc_mode) in zip(["linear_length", "max_length_identity_msc", "identity_length"], [(None, "max", "max"), (5.0, "max", "identity"), (60.0, "identity", "max")], strict=True):
                with self.subTest(case_id_0=case_id_0, shared_dofs=shared_dofs, case_id_1=case_id_1, length_target=length_target, length_mode=length_mode, msc_mode=msc_mode), self.case() as patches:
                    self._case_fused_objective_matches_native_composite(shared_dofs, length_target, length_mode, msc_mode, patches)

    def _case_fused_objective_matches_native_composite(self, shared_dofs, length_target, length_mode, msc_mode, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        config = replace(
            _ACTIVE,
            length_target=length_target,
            length_target_mode=length_mode,
            mean_squared_curvature_target_mode=msc_mode,
        )
        field, _, native, problem = _problem(config, shared_dofs=shared_dofs)
        value, gradient = _value_and_grad(problem, jnp.asarray(field.x))
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(gradient), _native_gradient(native, field), rtol=1e-11, atol=1e-13)

    def test_fused_diagnostics_match_native_terms(self):
        """Flux, penalty, normal-field and length diagnostics match native quantities."""
        field, flux, native, problem = _problem(_ACTIVE)
        surface = flux.surface
        objective, squared_flux, penalty, max_normal_field, length = (
            float(value) for value in _values(problem, jnp.asarray(field.x))
        )
        native_flux = SquaredFlux(surface, BiotSavart(field.coils))
        B = native_flux.field.B().reshape(surface.normal().shape)
        native_max_normal_field = np.max(np.abs(np.sum(B * surface.unitnormal(), axis=2)))
        base = [coil.curve for coil in field.coils[:_NCOILS]]
        np.testing.assert_allclose(objective, native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(squared_flux, native_flux.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(penalty, objective - squared_flux, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(max_normal_field, native_max_normal_field, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(length, sum(CurveLength(c).J() for c in base), rtol=1e-12)

    def test_fused_gradient_matches_central_differences(self):
        """The fused directional derivative agrees with central differences."""
        field, _, _, problem = _problem(_ACTIVE, shared_dofs=True)
        x0 = jnp.asarray(field.x)
        direction = jnp.asarray(np.random.default_rng(4).standard_normal(x0.shape))
        direction = direction * jnp.maximum(jnp.abs(x0), 1.0)
        step = 1e-7
        plus = _objective(problem, x0 + step * direction)
        minus = _objective(problem, x0 - step * direction)
        _, gradient = _value_and_grad(problem, x0)
        np.testing.assert_allclose(
            float(gradient @ direction), float((plus - minus) / (2 * step)), rtol=1e-6
        )

    def test_removed_and_zero_weight_terms_match_native(self):
        """Absent and zero-weight terms preserve native composite values and gradients."""
        for case_id_0, weight in zip(["term_removed", "zero_weight"], [None, 0.0], strict=True):
            with self.subTest(case_id_0=case_id_0, weight=weight), self.case() as patches:
                self._case_removed_and_zero_weight_terms_match_native(weight, patches)

    def _case_removed_and_zero_weight_terms_match_native(self, weight, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        config = replace(_ACTIVE, curve_surface_weight=weight, curvature_weight=weight)
        field, _, native, problem = _problem(config)
        value, gradient = _value_and_grad(problem, jnp.asarray(field.x))
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(gradient), _native_gradient(native, field), rtol=1e-11, atol=1e-13)

    def test_rebuilt_problem_with_new_weights_reuses_the_compiled_program(self):
        """Changing numeric weights reuses compilation and preserves native parity."""
        for case_id_0, length_weight in zip(["smaller", "zero", "integer_zero", "integer", "float32"], [1e-4, 0.0, 0, 2, np.float32(5e-4)], strict=True):
            with self.subTest(case_id_0=case_id_0, length_weight=length_weight), self.case() as patches:
                self._case_rebuilt_problem_with_new_weights_reuses_the_compiled_program(length_weight, patches)

    def _case_rebuilt_problem_with_new_weights_reuses_the_compiled_program(self, length_weight, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        field, flux, _, problem = _problem(_ACTIVE)
        traces = []

        def objective(current_problem, parameters):
            traces.append(parameters)
            return fused_stage_two_objective(current_problem, parameters)

        value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=1))
        x = jnp.asarray(field.x)
        value_and_grad(problem, x)
        lighter = replace(_ACTIVE, length_weight=length_weight)
        rebuilt = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), lighter)
        value, gradient = value_and_grad(rebuilt, x)
        self.assertTrue(len(traces) == 1, 'len(traces) == 1')
        native = _native_composite(flux.surface, [c.curve for c in field.coils[:_NCOILS]], field.coils, lighter)
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(gradient), _native_gradient(native, field), rtol=1e-11, atol=1e-13)

    def test_rebuilt_problem_follows_fixed_dof_values(self):
        """Fixed DOF values are operands of the problem: rebuilding picks up new ones."""
        field, flux, native, problem = _problem(_ACTIVE)
        fixed_curve = field.coils[0].curve
        fixed_current = field.coils[0].current
        fixed_curve.fix_all()
        x = jnp.asarray(field.x)
        problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), _ACTIVE)
        before, _ = _value_and_grad(problem, x)
        fixed_curve.local_full_x = np.asarray(fixed_curve.local_full_x) * 1.02
        fixed_current.local_full_x = np.asarray(fixed_current.local_full_x) * 0.9
        stale, _ = _value_and_grad(problem, x)
        rebuilt = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), _ACTIVE)
        value, gradient = _value_and_grad(rebuilt, x)
        np.testing.assert_allclose(float(stale), float(before), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(gradient), _native_gradient(native, field), rtol=1e-11, atol=1e-13)
        self.assertTrue(abs(float(value) - float(before)) > 1e-6 * abs(float(before)), 'abs(float(value) - float(before)) > 1e-6 * abs(float(before))')

    def test_distance_terms_of_a_far_degenerate_coil_have_finite_zero_gradients(self):
        """A coil collapsed to a far point (zero tangents) adds nothing, not NaN gradients."""
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ring = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)
        ring_dash = 2.0 * np.pi * np.stack((-np.sin(angles), np.cos(angles), np.zeros_like(angles)), axis=1)
        gamma = jnp.asarray(np.stack((ring, 1.02 * ring, np.full_like(ring, 50.0))))
        gammadash = jnp.asarray(np.stack((ring_dash, 1.02 * ring_dash, np.zeros_like(ring))))
        config = StageTwoObjectiveConfig(
            num_basecurves=3,
            curve_curve_minimum_distance=0.1,
            curve_curve_weight=1.0,
            curve_surface_minimum_distance=0.2,
            curve_surface_weight=1.0,
        )

        def penalty(current_gamma, current_gammadash):
            return stage_two_geometric_penalty(
                current_gamma, current_gammadash, jnp.zeros_like(current_gammadash),
                jnp.asarray(1.1 * ring), jnp.asarray(ring), config,
            )

        value, (dgamma, dgammadash) = jax.value_and_grad(penalty, argnums=(0, 1))(gamma, gammadash)
        self.assertTrue(float(value) > 0.0, 'float(value) > 0.0')
        self.assertTrue(np.all(np.isfinite(dgamma)) and np.all(np.isfinite(dgammadash)), 'np.all(np.isfinite(dgamma)) and np.all(np.isfinite(dgammadash))')
        self.assertTrue(not np.any(dgamma[2]) and not np.any(dgammadash[2]), 'not np.any(dgamma[2]) and not np.any(dgammadash[2])')
        self.assertTrue(np.any(dgammadash[0]), 'np.any(dgammadash[0])')

    def test_fused_program_compiles_and_runs_under_the_strict_transfer_guard(self):
        """The fused value and gradient compile and run with explicit transfers only."""
        field, _, _, problem = _problem(_ACTIVE)
        value_and_grad = jax.jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
        x = field.x
        with jax.transfer_guard("disallow"):
            value, gradient = jax.device_get(value_and_grad(problem, jax.device_put(x)))
        # A separately compiled program may reassociate GPU reductions.
        expected_value, expected_gradient = _value_and_grad(problem, jnp.asarray(x))
        np.testing.assert_allclose(float(value), float(expected_value), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(gradient, np.asarray(expected_gradient), rtol=1e-11, atol=1e-13)

    def test_problem_rejects_invalid_configurations(self):
        """Invalid counts, nonfinite weights, modes and partial surface overrides are rejected."""
        field, flux, _, _ = _problem(_ACTIVE)
        flux_spec = flux.fixed_surface_flux_spec()
        invalid = (
            (replace(_ACTIVE, num_basecurves=0), "num_basecurves must be a positive integer"),
            (replace(_ACTIVE, num_basecurves=len(field.coils) + 1), "num_basecurves exceeds"),
            (replace(_ACTIVE, curvature_weight=float("nan")), "curvature_weight must be finite"),
            (replace(_ACTIVE, length_target_mode="min"), "length_target_mode must be"),
        )
        for config, message in invalid:
            with self.assertRaisesRegex(ValueError, message):
                make_stage_two_problem(field, flux_spec, config)
        with self.assertRaisesRegex(ValueError, "Pass both surface_gamma and surface_normal"):
            make_stage_two_problem(field, flux_spec, _ACTIVE, surface_gamma=flux_spec.points)

    def test_fused_problem_requires_one_quadrature_grid(self):
        """Fused problem construction rejects unequal coil quadrature counts."""
        surface = _surface()
        _, coils = _coils(surface)
        odd = CurveXYZFourier(30, 3)
        odd.local_full_x = coils[2].curve.local_full_x
        field = JaxBiotSavart([*coils, Coil(odd, Current(1e4))])
        flux = JaxSquaredFlux(surface, field)
        with self.assertRaisesRegex(ValueError, "matching nonempty quadrature grids"):
            make_stage_two_problem(field, flux.fixed_surface_flux_spec(), _ACTIVE)

    def test_small_stage_two_run_follows_the_native_trajectory(self):
        """L-BFGS-B on the fused objective retraces the native run of the same problem.

            Both runs start from the unperturbed coils with the upstream example weights.
            Their gradients agree to round-off, which the first iterations amplify only
            to about 1e-13 (20 iterations); later, line-search decisions let round-off
            grow chaotically, so the comparison stops at 20 iterations.
            """
        config = StageTwoObjectiveConfig(
            num_basecurves=_NCOILS,
            length_weight=1e-6,
            curve_curve_minimum_distance=0.1,
            curve_curve_weight=1000.0,
            curve_surface_minimum_distance=0.3,
            curve_surface_weight=10.0,
            curvature_threshold=5.0,
            curvature_weight=1e-6,
            mean_squared_curvature_threshold=5.0,
            mean_squared_curvature_weight=1e-6,
        )
        field, _, native, problem = _problem(config, perturb=False)
        x0 = field.x.copy()
        options = {"maxiter": 20, "maxcor": 300}

        def native_fun(x):
            native.x = x
            return native.J(), native.dJ()

        def fused_fun(x):
            value, gradient = jax.device_get(_value_and_grad(problem, jax.device_put(x)))
            return float(value), gradient

        self.assertTrue(native.dof_names == field.dof_names, 'native.dof_names == field.dof_names')
        native_result = minimize(native_fun, x0, jac=True, method="L-BFGS-B", options=options, tol=1e-15)
        fused_result = minimize(fused_fun, x0, jac=True, method="L-BFGS-B", options=options, tol=1e-15)
        self.assertTrue(native_result.fun < 0.5 * native_fun(x0)[0], 'native_result.fun < 0.5 * native_fun(x0)[0]')
        self.assertTrue((fused_result.nit, fused_result.nfev) == (native_result.nit, native_result.nfev), '(fused_result.nit, fused_result.nfev) == (native_result.nit, native_result.nfev)')
        np.testing.assert_allclose(fused_result.fun, native_result.fun, rtol=1e-9)
        np.testing.assert_allclose(fused_result.x, native_result.x, rtol=1e-9, atol=1e-12)

    def test_fused_gradient_taylor(self):
        """The active fused composite attains native centered Taylor convergence."""
        field, _, _, problem = _problem(_ACTIVE, shared_dofs=True)
        x0 = jnp.asarray(field.x)
        direction = jnp.asarray(np.random.RandomState(1).uniform(size=x0.shape))
        _, gradient = _value_and_grad(problem, x0)
        derivative = float(gradient @ direction)
        previous_error = 1e10
        for exponent in range(11, 17):
            step = 0.5 ** exponent
            plus = float(_objective(problem, x0 + step * direction))
            minus = float(_objective(problem, x0 - step * direction))
            error = abs((plus - minus) / (2 * step) - derivative)
            self.assertLess(error, 0.6 ** 2 * previous_error, f"step={step}")
            previous_error = error
