"""The fused Stage-II objective against the native composite objective."""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt._core.optimizable import Optimizable
    from simsopt_jax.backend.dtypes import as_jax_float64
    from simsopt_jax.backend.jit import jit
    from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
    from simsopt.geo import (
        CurveCurveDistance,
        CurveLength,
        CurvePerturbed,
        CurveSurfaceDistance,
        CurveXYZFourier,
        GaussianSampler,
        LpCurveCurvature,
        MeanSquaredCurvature,
        PerturbationSample,
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
    from geo.test_curve_objectives_jax import _TIE_CASES, _circle
    from objectives.test_squared_flux_jax import _surface


from dataclasses import replace
from typing import cast

import numpy as np
from scipy.optimize import minimize


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


def _native_penalty(
    surface, base, coils, config: StageTwoObjectiveConfig
) -> Optimizable:
    """The native weighted geometric penalty that ``config`` describes, as in the Stage-II examples.

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
    return cast(Optimizable, sum(weight * term for weight, term in weighted_terms if weight is not None))


def _native_composite(
    surface, base, coils, config: StageTwoObjectiveConfig
) -> Optimizable:
    """The native objective that ``config`` describes: squared flux plus :func:`_native_penalty`."""
    return SquaredFlux(surface, BiotSavart(coils)) + _native_penalty(surface, base, coils, config)


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
    _value_and_grad = jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
    _objective = jit(fused_stage_two_objective)
    _values = jit(fused_stage_two_values)


class TestStageTwo(JaxTestCase):
    def test_fused_objective_matches_native_composite(self):
        """The fused value and gradient match the native composite with shared DOFs, every
        length and curvature target mode, and absent or zero-weight terms."""
        rows = [
            (
                f"{case_id_0}-{case_id_1}",
                replace(
                    _ACTIVE,
                    length_target=length_target,
                    length_target_mode=length_mode,
                    mean_squared_curvature_target_mode=msc_mode,
                ),
                shared_dofs,
            )
            for case_id_0, shared_dofs in (("symmetric", False), ("shared_dofs", True))
            for case_id_1, (length_target, length_mode, msc_mode) in (
                ("linear_length", (None, "max", "max")),
                ("max_length_identity_msc", (5.0, "max", "identity")),
                ("identity_length", (60.0, "identity", "max")),
            )
        ] + [
            (case_id, replace(_ACTIVE, curve_surface_weight=weight, curvature_weight=weight), False)
            for case_id, weight in (("term_removed", None), ("zero_weight", 0.0))
        ]
        for case_id, config, shared_dofs in rows:
            with self.subTest(case=case_id), self.case():
                self._case_fused_objective_matches_native_composite(config, shared_dofs)

    def _case_fused_objective_matches_native_composite(self, config, shared_dofs):
        """Run one row with case-local objects released before runtime cleanup."""
        field, _, native, problem = _problem(config, shared_dofs=shared_dofs)
        value, gradient = _value_and_grad(problem, jnp.asarray(field.x))
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(gradient), _native_gradient(native, field), rtol=1e-11, atol=1e-13)

    def test_fused_objective_reads_sample_second_derivatives_only_for_curvature(self):
        """As on native, perturbation samples with first derivatives only (GaussianSampler's
        default) suffice unless a curvature term reads a perturbed curve: the flux and length
        objective of perturbed coils and curvature of unperturbed base curves beside perturbed
        coils match native; curvature of a perturbed base curve raises, as native does."""
        surface = _surface()
        base, coils = _coils(surface)
        sampler = GaussianSampler(base[0].quadpoints, 1e-3, 0.2, n_derivs=1)
        rng = np.random.default_rng(11)
        perturbed = coils_via_symmetries(
            [CurvePerturbed(curve, PerturbationSample(sampler, randomgen=rng)) for curve in base],
            [coil.current for coil in coils[:_NCOILS]],
            surface.nfp,
            True,
        )
        flux_and_length = StageTwoObjectiveConfig(num_basecurves=_NCOILS, length_weight=1e-3)
        curvature = replace(
            flux_and_length,
            curvature_threshold=1.0,
            curvature_weight=1e-2,
            mean_squared_curvature_threshold=1.0,
            mean_squared_curvature_weight=1e-2,
        )
        for case_id, case_coils, config in (
            ("perturbed_flux_and_length", perturbed, flux_and_length),
            ("unperturbed_base_curvature", coils[:_NCOILS] + perturbed, curvature),
        ):
            with self.subTest(case=case_id), self.case():
                field = JaxBiotSavart(case_coils)
                flux = JaxSquaredFlux(surface, field)
                case_base = [coil.curve for coil in case_coils[:_NCOILS]]
                native = _native_composite(surface, case_base, case_coils, config)
                problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), config)
                value, gradient = _value_and_grad(problem, jnp.asarray(field.x))
                np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(
                    np.asarray(gradient), _native_gradient(native, field), rtol=1e-11, atol=1e-13
                )
        with self.case():
            field = JaxBiotSavart(perturbed)
            flux = JaxSquaredFlux(surface, field)
            problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), curvature)
            with self.assertRaisesRegex(ValueError, "has 1 derivatives"):
                _value_and_grad(problem, jnp.asarray(field.x))
            with self.assertRaises(ValueError):
                LpCurveCurvature(perturbed[0].curve, 2, 1.0).J()

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
        native_penalty = _native_penalty(surface, base, field.coils, _ACTIVE)
        np.testing.assert_allclose(objective, native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(squared_flux, native_flux.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(penalty, native_penalty.J(), rtol=1e-12, atol=1e-14)
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

    def test_rebuilt_problem_with_new_weights_reuses_the_compiled_program(self):
        """Changing numeric weights reuses compilation and preserves native parity."""
        for case_id_0, length_weight in zip(["smaller", "zero", "integer_zero", "integer", "float32"], [1e-4, 0.0, 0, 2, np.float32(5e-4)], strict=True):
            with self.subTest(case_id_0=case_id_0, length_weight=length_weight), self.case():
                self._case_rebuilt_problem_with_new_weights_reuses_the_compiled_program(length_weight)

    def _case_rebuilt_problem_with_new_weights_reuses_the_compiled_program(self, length_weight):
        """Run one row with case-local objects released before runtime cleanup."""
        field, flux, _, problem = _problem(_ACTIVE)
        traces = []

        def objective(current_problem, parameters):
            traces.append(parameters)
            return fused_stage_two_objective(current_problem, parameters)

        value_and_grad = jit(jax.value_and_grad(objective, argnums=1))
        x = jnp.asarray(field.x)
        value_and_grad(problem, x)
        lighter = replace(_ACTIVE, length_weight=length_weight)
        rebuilt = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), lighter)
        value, gradient = value_and_grad(rebuilt, x)
        self.assertEqual(len(traces), 1)
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
        self.assertGreater(abs(float(value) - float(before)), 1e-6 * abs(float(before)))

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
        self.assertGreater(float(value), 0.0)
        self.assertTrue(np.all(np.isfinite(dgamma)) and np.all(np.isfinite(dgammadash)))
        self.assertFalse(np.any(dgamma[2]) or np.any(dgammadash[2]))
        self.assertTrue(np.any(dgammadash[0]))

    def test_fused_distance_candidates_at_the_threshold(self):
        """Off a tie the fused candidate choice is native's; at a tie it is one of the two."""
        for case_id_0, (position, minimum_distance) in zip(["tie_a", "tie_b"], _TIE_CASES, strict=True):
            for term in ("curve_curve", "curve_surface"):
                with self.subTest(case_id_0=case_id_0, term=term), self.case():
                    self._case_fused_distance_candidates_at_the_threshold(position, minimum_distance, term)

    def _case_fused_distance_candidates_at_the_threshold(self, position, minimum_distance, term):
        """Compare fused and native outcomes for constant curves at and near one tie.

        The curves have zero tangents, so a candidate pair has a non-finite gradient (native
        and fused alike) and a non-candidate pair a finite zero gradient; the value is zero
        either way. A relative margin of 1e-9 moves the squared distance about 1e7 ulp off
        the squared threshold, where rounding cannot change the decision."""
        origin_curve = _circle((0.0, 0.0, 0.0), 0.0, numquadpoints=8)
        point_curve = _circle(position, 0.0, numquadpoints=8)
        surface = SurfaceRZFourier.from_nphi_ntheta(nphi=4, ntheta=4)
        surface.local_full_x = np.zeros_like(surface.local_full_x)
        curves = [origin_curve, point_curve] if term == "curve_curve" else [point_curve]
        gamma = jnp.asarray(np.stack([curve.gamma() for curve in curves]))
        gammadash = jnp.asarray(np.stack([curve.gammadash() for curve in curves]))
        surface_gamma = jnp.asarray(surface.gamma().reshape((-1, 3)))
        surface_normal = jnp.asarray(surface.normal().reshape((-1, 3)))
        outcomes = {}
        for side, scale in (("below", 1.0 - 1e-9), ("tie", 1.0), ("above", 1.0 + 1e-9)):
            threshold = minimum_distance * scale
            if term == "curve_curve":
                native = CurveCurveDistance(curves, threshold)
                config = StageTwoObjectiveConfig(
                    num_basecurves=2, curve_curve_minimum_distance=threshold, curve_curve_weight=1.0
                )
            else:
                native = CurveSurfaceDistance(curves, surface, threshold)
                config = StageTwoObjectiveConfig(
                    num_basecurves=1, curve_surface_minimum_distance=threshold, curve_surface_weight=1.0
                )

            def penalty(current_gamma, current_gammadash, config=config):
                return stage_two_geometric_penalty(
                    current_gamma, current_gammadash, jnp.zeros_like(current_gammadash),
                    surface_gamma, surface_normal, config,
                )

            value, gradient = jit(jax.value_and_grad(penalty, argnums=(0, 1)))(gamma, gammadash)
            self.assertEqual(float(value), 0.0, side)
            gradient = np.concatenate([np.asarray(part).ravel() for part in gradient])
            outcomes[side] = (bool(np.all(np.isfinite(native.dJ()))), gradient)
        for side, native_finite in (("below", True), ("above", False)):
            finite, gradient = outcomes[side]
            self.assertEqual(finite, native_finite, f"native {side} the threshold")
            self.assertEqual(bool(np.all(np.isfinite(gradient))), native_finite, f"fused {side} the threshold")
        self.assertFalse(np.any(outcomes["below"][1]), "fused non-candidate gradient")
        tie = outcomes["tie"][1]
        self.assertTrue(
            any(np.array_equal(tie, outcomes[side][1], equal_nan=True) for side in ("below", "above")),
            f"fused gradient at the tie is neither off-tie outcome: {tie}",
        )

    def test_fused_program_compiles_and_runs_under_the_strict_transfer_guard(self):
        """The fused value and gradient compile and run with explicit transfers only."""
        field, _, _, problem = _problem(_ACTIVE)
        value_and_grad = jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
        x = field.x
        with jax.transfer_guard("disallow"):
            value, gradient = jax.device_get(value_and_grad(problem, as_jax_float64(x)))
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
        x0 = np.array(field.x, dtype=np.float64)
        options = {"maxiter": 20, "maxcor": 300}

        def native_fun(x):
            native.x = x
            return native.J(), native.dJ()

        def fused_fun(x):
            value, gradient = jax.device_get(_value_and_grad(problem, x))
            return float(value), gradient

        self.assertEqual(native.dof_names, field.dof_names)
        native_result = minimize(native_fun, x0, jac=True, method="L-BFGS-B", options=options, tol=1e-15)
        fused_result = minimize(fused_fun, x0, jac=True, method="L-BFGS-B", options=options, tol=1e-15)
        self.assertLess(native_result.fun, 0.5 * native_fun(x0)[0])
        self.assertEqual((fused_result.nit, fused_result.nfev), (native_result.nit, native_result.nfev))
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
