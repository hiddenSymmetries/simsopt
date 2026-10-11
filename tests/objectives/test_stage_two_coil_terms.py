"""Coil force and finite-build terms of the fused Stage-II objective against native composites."""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt._core.derivative import Derivative
    from simsopt._core.optimizable import Optimizable
    from simsopt.field import (
        BiotSavart,
        Coil,
        Current,
        RegularizedCoil,
        apply_symmetries_to_currents,
        apply_symmetries_to_curves,
        coils_via_symmetries,
    )
    from simsopt.field.force import (
        B2Energy,
        LpCurveForce,
        LpCurveTorque,
        SquaredMeanForce,
        SquaredMeanTorque,
    )
    from simsopt.field.selffield import regularization_circ, regularization_rect
    from simsopt.geo import (
        CurveCurveDistance,
        CurveLength,
        CurveSurfaceDistance,
        CurveXYZFourier,
        LpCurveCurvature,
        MeanSquaredCurvature,
        SurfaceRZFourier,
        create_equally_spaced_curves,
        create_multifilament_grid,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt_jax.objectives import (
        StageTwoObjectiveConfig,
        fused_stage_two_objective,
        fused_stage_two_values,
        make_stage_two_problem,
        stage_two_coil_geometry,
        stage_two_geometry,
    )
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo import (
        JaxCurveCurveDistance,
        JaxCurveLength,
        JaxCurveSurfaceDistance,
        JaxLpCurveCurvature,
        JaxMeanSquaredCurvature,
    )
    from simsopt_jax_adapters.objectives import JaxSquaredFlux


from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import minimize


_QA_INPUT = Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
_NCOILS = 3
_FILAMENTS = 4


def _surface() -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(str(_QA_INPUT), range="half period", nphi=8, ntheta=9)


def _force_coils(surface, *, perturb: bool = True, shared_dofs: bool = False) -> list:
    """Regularized symmetric coils, base coils first, with fixed DOFs.

    With ``shared_dofs``, one more coil on an offset grid shares the DOFs of
    the second base curve and of the third base current.
    """
    base = create_equally_spaced_curves(
        _NCOILS, surface.nfp, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24
    )
    if perturb:
        rng = np.random.default_rng(23)
        for curve in base:
            curve.x = curve.x + 0.03 * rng.standard_normal(curve.x.shape)
    base[0].fix("xc(1)")
    currents = [Current(1e5), Current(1.1e5), Current(0.9e5)]
    currents[0].fix_all()
    regularizations = [regularization_circ(0.05), regularization_rect(0.04, 0.06), regularization_circ(0.07)]
    coils = coils_via_symmetries(base, currents, surface.nfp, True, regularizations)
    if shared_dofs:
        twin = CurveXYZFourier(np.linspace(0, 1, 24, endpoint=False) + 0.011, 3, dofs=base[1].dofs)
        coils.append(RegularizedCoil(twin, -0.5 * Current(0.0, dofs=currents[2].dofs), regularizations[0]))
    return coils


def _coil_terms(config: StageTwoObjectiveConfig, coils) -> list[tuple[float | None, Optimizable]]:
    """The native coil objectives ``config`` describes, with their weights."""
    targets = coils[: config.num_basecurves]
    downsample = config.force_downsample
    return [
        (config.force_weight, LpCurveForce(
            targets, coils, p=config.force_p, threshold=config.force_threshold, downsample=downsample)),
        (config.torque_weight, LpCurveTorque(
            targets, coils, p=config.torque_p, threshold=config.torque_threshold, downsample=downsample)),
        (config.squared_mean_force_weight, SquaredMeanForce(targets, coils, downsample=downsample)),
        (config.squared_mean_torque_weight, SquaredMeanTorque(targets, coils, downsample=downsample)),
        (config.vacuum_energy_weight, B2Energy(coils, downsample=downsample)),
    ]


if JAX_IMPORT_ERROR is None:
    _NATIVE_GEOMETRY = (CurveLength, CurveCurveDistance, CurveSurfaceDistance, LpCurveCurvature, MeanSquaredCurvature)
    _JAX_GEOMETRY = (
        JaxCurveLength, JaxCurveCurveDistance, JaxCurveSurfaceDistance, JaxLpCurveCurvature, JaxMeanSquaredCurvature,
    )


    def _geometric_terms(config: StageTwoObjectiveConfig, base, centerlines, surface, classes=_NATIVE_GEOMETRY):
        """The coil-geometry objectives ``config`` describes, with their weights.

        ``classes`` are the native objectives or their drop-in JAX mirrors.
        """
        length_class, curve_curve_class, curve_surface_class, curvature_class, msc_class = classes
        lengths = [length_class(curve) for curve in base]
        num_basecurves = config.num_basecurves if config.curve_curve_pairs == "base" else None
        return [
            (config.length_weight, (
                sum(lengths) if config.length_target is None
                else QuadraticPenalty(sum(lengths), config.length_target, config.length_target_mode)
            )),
            (config.individual_length_weight, sum(
                QuadraticPenalty(length, target, config.individual_length_target_mode)
                for length, target in zip(lengths, config.individual_length_targets, strict=True)
            ) if config.individual_length_weight is not None else None),
            (config.curve_curve_weight, curve_curve_class(
                centerlines, config.curve_curve_minimum_distance, num_basecurves=num_basecurves)),
            (config.curve_surface_weight, curve_surface_class(
                centerlines, surface, config.curve_surface_minimum_distance)),
            (config.curvature_weight, sum(
                curvature_class(curve, 2, config.curvature_threshold) for curve in base)),
            (config.mean_squared_curvature_weight, sum(
                QuadraticPenalty(
                    msc_class(curve),
                    config.mean_squared_curvature_threshold,
                    config.mean_squared_curvature_target_mode,
                )
                for curve in base
            )),
        ]


def _native_force_composite(surface, coils, config) -> Optimizable:
    base = [coil.curve for coil in coils[: config.num_basecurves]]
    centerlines = [coil.curve for coil in coils]
    terms = _geometric_terms(config, base, centerlines, surface) + _coil_terms(config, coils)
    return SquaredFlux(surface, BiotSavart(coils)) + sum(
        weight * term for weight, term in terms if weight is not None
    )


def _native_gradient(native: Optimizable, field: JaxBiotSavart) -> np.ndarray:
    """The native gradient in the order of the field's free DOFs."""
    return np.asarray(native.dJ(partials=True)(field))


def _assert_close(actual, expected, name: str = "") -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-11, atol=1e-12 * np.max(np.abs(expected)), err_msg=name
    )


if JAX_IMPORT_ERROR is None:
    _value_and_grad = jax.jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
    _objective = jax.jit(fused_stage_two_objective)
    _penalties_value_and_grad = jax.jit(
        jax.value_and_grad(lambda problem, x: fused_stage_two_values(problem, x)[2], argnums=1)
    )

# Weights that give every term a visible share of the objective at the test state.
if JAX_IMPORT_ERROR is None:
    _ALL_FORCE_TERMS = StageTwoObjectiveConfig(
        num_basecurves=_NCOILS,
        length_weight=1e-3,
        length_target=8.0,
        curve_curve_minimum_distance=0.6,
        curve_curve_weight=10.0,
        curve_surface_minimum_distance=0.4,
        curve_surface_weight=2.0,
        curvature_threshold=1.0,
        curvature_weight=1e-2,
        mean_squared_curvature_threshold=1.0,
        mean_squared_curvature_weight=1e-2,
        force_weight=1e5,
        force_p=4,
        torque_weight=1e5,
        torque_p=3,
        torque_threshold=1e-4,
        squared_mean_force_weight=1e2,
        squared_mean_torque_weight=1e4,
        vacuum_energy_weight=0.1,
    )


def _force_problem(config, *, perturb=True, shared_dofs=False):
    surface = _surface()
    coils = _force_coils(surface, perturb=perturb, shared_dofs=shared_dofs)
    field = JaxBiotSavart(coils)
    flux = JaxSquaredFlux(surface, field)
    problem = make_stage_two_problem(
        field, flux.fixed_surface_flux_spec(), config,
        regularizations=[coil.regularization for coil in coils],
    )
    return surface, coils, field, flux, problem






def _finite_build(surface, frame: str, rotation_order: int | None, *, perturb: bool = True):
    """Filament packs of the finite-build example on two base curves, and their centerlines."""
    base = create_equally_spaced_curves(
        2, surface.nfp, stellsym=True, R0=1.0, R1=0.6, order=3, numquadpoints=24
    )
    base_currents = [Current(1.0) * (1e5 / _FILAMENTS), Current(1.0) * (1e5 / _FILAMENTS)]
    base_currents[0].current_to_scale.fix_all()
    filaments = sum(
        [create_multifilament_grid(curve, 2, 2, 0.02, 0.04, rotation_order=rotation_order, frame=frame)
         for curve in base],
        [],
    )
    if perturb:
        rng = np.random.default_rng(29)
        for curve in base:
            curve.x = curve.x + 0.02 * rng.standard_normal(curve.x.shape)
        for filament in filaments[::_FILAMENTS]:
            rotation = filament.rotation
            rotation.x = rotation.x + 0.1 * rng.standard_normal(rotation.x.shape)
    base[1].fix("yc(1)")
    filament_curves = apply_symmetries_to_curves(filaments, surface.nfp, True)
    filament_currents = apply_symmetries_to_currents(
        sum([[current] * _FILAMENTS for current in base_currents], []), surface.nfp, True
    )
    coils = [Coil(curve, current) for curve, current in zip(filament_curves, filament_currents)]
    return base, apply_symmetries_to_curves(base, surface.nfp, True), coils


def _finite_build_config(base) -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_basecurves=len(base),
        filaments_per_pack=_FILAMENTS,
        individual_length_weight=1e-1,
        individual_length_targets=tuple(0.95 * CurveLength(curve).J() for curve in base),
        curve_curve_minimum_distance=0.6,
        curve_curve_weight=10.0,
        curve_curve_pairs="all",
        curve_surface_minimum_distance=0.4,
        curve_surface_weight=2.0,
        curvature_threshold=1.0,
        curvature_weight=1e-2,
        mean_squared_curvature_threshold=1.0,
        mean_squared_curvature_weight=1e-2,
    )


def _native_finite_build(surface, base, centerlines, coils, config) -> Optimizable:
    terms = _geometric_terms(config, base, centerlines, surface)
    return SquaredFlux(surface, BiotSavart(coils)) + sum(
        weight * term for weight, term in terms if weight is not None
    )


def _finite_build_problem(frame="centroid", rotation_order=1, *, perturb=True):
    """``(field, flux, native composite, config, problem)`` of a finite-build case, and its
    ``(surface, base curves, centerlines, coils)``."""
    surface = _surface()
    base, centerlines, coils = _finite_build(surface, frame, rotation_order, perturb=perturb)
    config = _finite_build_config(base)
    field = JaxBiotSavart(coils)
    flux = JaxSquaredFlux(surface, field)
    native = _native_finite_build(surface, base, centerlines, coils, config)
    problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), config)
    return (field, flux, native, config, problem), (surface, base, centerlines, coils)




def _native_in_field_order(native: Optimizable, field: JaxBiotSavart):
    """Map the field's free DOF vector onto the native composite's ordering."""
    assert sorted(native.dof_names) == sorted(field.dof_names)
    positions = {name: index for index, name in enumerate(field.dof_names)}
    return np.asarray([positions[name] for name in native.dof_names])












def _run_both(native: Optimizable, field: JaxBiotSavart, problem, scale: float, options: dict):
    """L-BFGS-B on the native composite and on the fused objective from the field's DOFs."""
    order = _native_in_field_order(native, field)
    x0 = field.x.copy()

    def native_fun(x):
        native.x = x[order]
        gradient = np.empty_like(x)
        gradient[order] = native.dJ()
        return scale * native.J(), scale * gradient

    def fused_fun(x):
        value, gradient = jax.device_get(_value_and_grad(problem, jax.device_put(x)))
        return scale * float(value), scale * gradient

    native_result = minimize(native_fun, x0, jac=True, method="L-BFGS-B", options=options, tol=1e-15)
    fused_result = minimize(fused_fun, x0, jac=True, method="L-BFGS-B", options=options, tol=1e-15)
    return native_fun(x0)[0], native_result, fused_result




class TestStageTwoCoilTerms(JaxTestCase):
    def test_each_coil_term_matches_its_native_objective(self):
        """Each enabled fused coil term matches its native value and DOF gradient."""
        for case_id_0, (term, settings) in zip(["force", "force_threshold_downsample", "torque", "squared_mean_force_downsample",
         "squared_mean_torque", "vacuum_energy", "vacuum_energy_downsample"], [
        ("force_weight", {"force_p": 4.0}),
        ("force_weight", {"force_p": 2.5, "force_threshold": 0.002, "force_downsample": 2}),
        ("torque_weight", {"torque_p": 3.0, "torque_threshold": 1e-4}),
        ("squared_mean_force_weight", {"force_downsample": 3}),
        ("squared_mean_torque_weight", {}),
        ("vacuum_energy_weight", {}),
        ("vacuum_energy_weight", {"force_downsample": 2}),
    ], strict=True):
            with self.subTest(case_id_0=case_id_0, term=term, settings=settings), self.case() as patches:
                self._case_each_coil_term_matches_its_native_objective(term, settings, patches)

    def _case_each_coil_term_matches_its_native_objective(self, term, settings, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        config = StageTwoObjectiveConfig(num_basecurves=_NCOILS, **{term: 1.0}, **settings)
        _, coils, field, _, problem = _force_problem(config)
        (native,) = [objective for weight, objective in _coil_terms(config, coils) if weight is not None]
        value, gradient = _penalties_value_and_grad(problem, jnp.asarray(field.x))
        native_value = float(native.J())
        self.assertTrue(native_value > 0.0, 'native_value > 0.0')
        np.testing.assert_allclose(float(value), native_value, rtol=1e-12, atol=0.0)
        _assert_close(gradient, _native_gradient(native, field), term)

    def test_fused_force_objective_matches_native_composite(self):
        """Combined fused force terms match the native composite value and gradient."""
        for case_id_0, shared_dofs in zip(["symmetric", "shared_dofs"], [False, True], strict=True):
            with self.subTest(case_id_0=case_id_0, shared_dofs=shared_dofs), self.case() as patches:
                self._case_fused_force_objective_matches_native_composite(shared_dofs, patches)

    def _case_fused_force_objective_matches_native_composite(self, shared_dofs, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        surface, coils, field, _, problem = _force_problem(_ALL_FORCE_TERMS, shared_dofs=shared_dofs)
        native = _native_force_composite(surface, coils, _ALL_FORCE_TERMS)
        value, gradient = _value_and_grad(problem, jnp.asarray(field.x))
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=0.0)
        _assert_close(gradient, _native_gradient(native, field))

    def test_coil_geometry_keeps_the_pr4_contract_and_geometry_adds_centerlines(self):
        """``stage_two_coil_geometry`` returns ``(gamma, gammadash, gammadashdash, currents)``
        of every coil; ``stage_two_geometry`` adds the pack centerlines of finite build."""
        surface = _surface()
        coils = _force_coils(surface)
        field = JaxBiotSavart(coils)
        gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
            field.coil_dof_extraction_spec(), jnp.asarray(field.x)
        )
        for actual, expected in (
            (gamma, [coil.curve.gamma() for coil in coils]),
            (gammadash, [coil.curve.gammadash() for coil in coils]),
            (gammadashdash, [coil.curve.gammadashdash() for coil in coils]),
            (currents, [coil.current.get_value() for coil in coils]),
        ):
            _assert_close(actual, np.asarray(expected))

        (field, _, _, _, _), (_, _, centerlines, coils) = _finite_build_problem()
        geometry = stage_two_geometry(field.coil_dof_extraction_spec(), jnp.asarray(field.x), _FILAMENTS)
        for actual, expected in (
            (geometry.gamma, [coil.curve.gamma() for coil in coils]),
            (geometry.gammadash, [coil.curve.gammadash() for coil in coils]),
            (geometry.currents, [coil.current.get_value() for coil in coils]),
            (geometry.centerline_gamma, [curve.gamma() for curve in centerlines]),
            (geometry.centerline_gammadash, [curve.gammadash() for curve in centerlines]),
            (geometry.centerline_gammadashdash, [curve.gammadashdash() for curve in centerlines]),
        ):
            _assert_close(actual, np.asarray(expected))

    def test_fused_finite_build_matches_native_composite(self):
        """The fused objective and the drop-in JAX Optimizables on filament coils equal native."""
        for case_id_0, (frame, rotation_order) in zip(["centroid_frame_rotation", "frenet_frame_fixed"], [("centroid", 1), ("frenet", None)], strict=True):
            with self.subTest(case_id_0=case_id_0, frame=frame, rotation_order=rotation_order), self.case() as patches:
                self._case_fused_finite_build_matches_native_composite(frame, rotation_order, patches)

    def _case_fused_finite_build_matches_native_composite(self, frame, rotation_order, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        (field, flux, native, config, problem), (surface, base, centerlines, _) = (
            _finite_build_problem(frame, rotation_order)
        )
        value, gradient = _value_and_grad(problem, jnp.asarray(field.x))
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=0.0)
        native_gradient = _native_gradient(native, field)
        _assert_close(gradient, native_gradient)
        terms = _geometric_terms(config, base, centerlines, surface, _JAX_GEOMETRY)
        drop_in = flux + sum(weight * term for weight, term in terms if weight is not None)
        np.testing.assert_allclose(drop_in.J(), native.J(), rtol=1e-12, atol=0.0)
        _assert_close(cast(Callable[..., Derivative], drop_in.dJ)(partials=True)(field), native_gradient)

    def test_fused_coil_terms_match_central_differences(self):
        """Fused coil-term directional gradients match independent central differences."""
        rng = np.random.default_rng(31)
        _, _, force_field, _, force_problem = _force_problem(_ALL_FORCE_TERMS, shared_dofs=True)
        (finite_build_field, _, _, _, finite_build_problem), _ = _finite_build_problem()
        for field, problem in ((force_field, force_problem), (finite_build_field, finite_build_problem)):
            x0 = jnp.asarray(field.x)
            direction = jnp.asarray(rng.standard_normal(x0.shape)) * jnp.maximum(jnp.abs(x0), 1.0)
            step = 1e-7
            plus = _objective(problem, x0 + step * direction)
            minus = _objective(problem, x0 - step * direction)
            _, gradient = _value_and_grad(problem, x0)
            np.testing.assert_allclose(
                float(gradient @ direction), float((plus - minus) / (2 * step)), rtol=1e-6
            )

    def test_rebuilt_problems_with_new_coil_term_settings_reuse_the_compiled_program(self):
        """Weights, exponents, thresholds and length targets are operands, not program constants."""
        surface, coils, field, flux, problem = _force_problem(_ALL_FORCE_TERMS)
        (base_field, base_flux, _, base_config, base_problem), base_case = _finite_build_problem()
        traces = []

        def objective(current_problem, parameters):
            traces.append(parameters)
            return fused_stage_two_objective(current_problem, parameters)

        value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=1))
        x, base_x = jnp.asarray(field.x), jnp.asarray(base_field.x)
        value_and_grad(problem, x)
        value_and_grad(base_problem, base_x)
        changed = replace(
            _ALL_FORCE_TERMS, force_weight=0, force_p=2, force_threshold=1e-3, torque_p=2.5,
            squared_mean_force_weight=np.float32(3), vacuum_energy_weight=0.5,
        )
        for coil in coils:
            coil.regularization = 2.0 * coil.regularization
        rebuilt = make_stage_two_problem(
            field, flux.fixed_surface_flux_spec(), changed,
            regularizations=[coil.regularization for coil in coils],
        )
        base_changed = replace(
            base_config,
            individual_length_targets=tuple(0.9 * target for target in base_config.individual_length_targets),
            individual_length_weight=2,
        )
        base_rebuilt = make_stage_two_problem(base_field, base_flux.fixed_surface_flux_spec(), base_changed)
        value, gradient = value_and_grad(rebuilt, x)
        base_value, base_gradient = value_and_grad(base_rebuilt, base_x)
        self.assertTrue(len(traces) == 2, 'len(traces) == 2')
        for current, native in (
            ((value, gradient, field), _native_force_composite(surface, coils, changed)),
            ((base_value, base_gradient, base_field), _native_finite_build(*base_case, base_changed)),
        ):
            current_value, current_gradient, current_field = current
            np.testing.assert_allclose(float(current_value), native.J(), rtol=1e-12, atol=0.0)
            _assert_close(current_gradient, _native_gradient(native, current_field))

    def test_problem_rejects_invalid_coil_terms_and_packs(self):
        """Invalid coil-term settings and inconsistent filament packs are rejected."""
        _, _, field, flux, _ = _force_problem(_ALL_FORCE_TERMS)
        flux_spec = flux.fixed_surface_flux_spec()
        regularizations = [coil.regularization for coil in field.coils]
        invalid = (
            (replace(_ALL_FORCE_TERMS, num_basecurves=len(field.coils)), "need source coils"),
            (replace(_ALL_FORCE_TERMS, force_downsample=5), "must evenly divide"),
            (replace(_ALL_FORCE_TERMS, force_downsample=0), "force_downsample must be a positive integer"),
            (replace(_ALL_FORCE_TERMS, filaments_per_pack=2), "requires finite-build filament coils"),
            (replace(_ALL_FORCE_TERMS, filaments_per_pack=5), "multiple of filaments_per_pack"),
            (replace(_ALL_FORCE_TERMS, curve_curve_pairs="some"), "curve_curve_pairs must be"),
            (replace(_ALL_FORCE_TERMS, individual_length_weight=1.0), "one length per base curve"),
            (replace(_ALL_FORCE_TERMS, force_p=float("inf")), "force_p must be finite"),
        )
        for config, message in invalid:
            with self.assertRaisesRegex(ValueError, message):
                make_stage_two_problem(field, flux_spec, config, regularizations=regularizations)
        with self.assertRaisesRegex(ValueError, "need the coils' regularizations"):
            make_stage_two_problem(field, flux_spec, _ALL_FORCE_TERMS)
        with self.assertRaisesRegex(ValueError, "one value per coil"):
            make_stage_two_problem(field, flux_spec, _ALL_FORCE_TERMS, regularizations=regularizations[1:])

        (base_field, base_flux, _, base_config, _), _ = _finite_build_problem()
        base_flux_spec = base_flux.fixed_surface_flux_spec()
        for config, message in (
            (replace(base_config, filaments_per_pack=8), "must share their curve"),
            (replace(base_config, filaments_per_pack=2), "filaments_per_pack is too small"),
            (replace(base_config, num_basecurves=9, individual_length_weight=None), "exceeds the available coil centerlines"),
            (replace(base_config, vacuum_energy_weight=1.0), "without filament coils"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                make_stage_two_problem(
                    base_field, base_flux_spec, config,
                    regularizations=np.full(len(base_field.coils), 1e-3),
                )

    def test_fused_coil_term_programs_run_under_the_strict_transfer_guard(self):
        """Fused coil terms compile and run with explicit transfers only."""
        value_and_grad = jax.jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
        _, _, force_field, _, force_problem = _force_problem(_ALL_FORCE_TERMS)
        (finite_build_field, _, _, _, finite_build_problem), _ = _finite_build_problem()
        for field, problem in ((force_field, force_problem), (finite_build_field, finite_build_problem)):
            x = field.x
            with jax.transfer_guard("disallow"):
                value, gradient = jax.device_get(value_and_grad(problem, jax.device_put(x)))
            # A separately compiled program may reassociate GPU reductions.
            expected_value, expected_gradient = _value_and_grad(problem, jnp.asarray(x))
            np.testing.assert_allclose(float(value), float(expected_value), rtol=1e-12, atol=0.0)
            _assert_close(gradient, expected_gradient)

    def test_short_force_and_finite_build_runs_follow_the_native_trajectories(self):
        """L-BFGS-B on the fused objectives retraces the native runs of the same problems.

        The runs start from the unperturbed coils with weights of the native
        examples; round-off differences stay at about 1e-12 over 15 iterations.
        """
        force_config = StageTwoObjectiveConfig(
            num_basecurves=_NCOILS,
            length_weight=1e-3,
            length_target=17.4,
            curve_curve_minimum_distance=0.1,
            curve_curve_weight=1000.0,
            curve_surface_minimum_distance=0.3,
            curve_surface_weight=10.0,
            curvature_threshold=5.0,
            curvature_weight=1e-6,
            mean_squared_curvature_threshold=5.0,
            mean_squared_curvature_weight=1e-6,
            force_weight=1e-2,
            force_p=4,
            vacuum_energy_weight=1e-4,
        )
        surface, coils, field, _, problem = _force_problem(force_config, perturb=False)
        native = _native_force_composite(surface, coils, force_config)
        (base_field, _, base_native, _, base_problem), _ = _finite_build_problem(perturb=False)
        options = {"maxiter": 15, "maxcor": 300}
        for name, (initial, native_result, fused_result) in (
            ("forces", _run_both(native, field, problem, 1.0, options)),
            ("finite build", _run_both(base_native, base_field, base_problem, 1e-4, options)),
        ):
            self.assertTrue(native_result.fun < 0.5 * initial, name)
            self.assertTrue((fused_result.nit, fused_result.nfev) == (native_result.nit, native_result.nfev), name)
            np.testing.assert_allclose(fused_result.fun, native_result.fun, rtol=1e-9, err_msg=name)
            np.testing.assert_allclose(fused_result.x, native_result.x, rtol=1e-8, atol=1e-10, err_msg=name)

    def test_fused_coil_term_gradients_taylor(self):
        """Force and rotated filament composites attain native centered Taylor rates."""
        for finite_build in (False, True):
            with self.subTest(finite_build=finite_build), self.case():
                self._case_fused_coil_term_gradients_taylor(finite_build)

    def _case_fused_coil_term_gradients_taylor(self, finite_build):
        """Use the native force or finite-build step interval and ratio bound."""
        if finite_build:
            (field, flux, _, config, _), _ = _finite_build_problem()
            # Taylor ratios require a smooth interval: all distance samples and
            # curvature penalties stay active throughout the largest perturbation.
            config = replace(
                config, curve_curve_minimum_distance=5.0,
                curve_surface_minimum_distance=5.0, curvature_threshold=0.0,
                mean_squared_curvature_threshold=0.0,
            )
            problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), config)
        else:
            _, _, field, _, problem = _force_problem(_ALL_FORCE_TERMS, shared_dofs=True)
        x0 = jnp.asarray(field.x)
        direction = jnp.asarray(np.random.RandomState(1).uniform(size=x0.shape))
        _, gradient = _value_and_grad(problem, x0)
        derivative = float(gradient @ direction)
        self.assertGreater(abs(derivative), 0.0)
        previous_error = 1e6 if finite_build else 100
        bound = 0.55 ** 2 if finite_build else 0.31
        for exponent in range(10, 15 if finite_build else 18):
            step = 0.5 ** exponent
            plus = float(_objective(problem, x0 + step * direction))
            minus = float(_objective(problem, x0 - step * direction))
            error = abs((plus - minus) / (2 * step) - derivative)
            if not finite_build:
                error /= abs(derivative)
            self.assertLess(error, bound * previous_error, f"step={step}")
            previous_error = error
