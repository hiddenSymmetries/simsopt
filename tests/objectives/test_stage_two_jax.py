"""JAX Stage-II objectives against native simsopt: squared flux, fused objective and batched coil geometry."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest import mock

import numpy as np
from scipy.optimize import minimize
from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports
from unittest_jax_support import DISTANCE_TIE_CASES, assert_matches_native_including_nans, circle_curve
from unittest_jax_support import distinct_host_array, make_execution_gate, native_field_gradient, qa_surface

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    import simsoptpp as sopp
    from simsopt._core.optimizable import Optimizable
    from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
    from simsopt.geo import CurveCurveDistance, CurveLength, CurvePerturbed, CurveSurfaceDistance, CurveXYZFourier
    from simsopt.geo import GaussianSampler, LpCurveCurvature, MeanSquaredCurvature, PerturbationSample
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.objectives import QuadraticPenalty, SquaredFlux, Weight
    from simsopt_jax.backend.dtypes import as_jax_float64
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core._device_scalars import placement_zero
    from simsopt_jax.core.curve_geometry import curve_geometry_from_dofs
    from simsopt_jax.core.field import coil_specs_from_dof_extraction_spec
    from simsopt_jax.core.integral_bdotn import integral_BdotN
    from simsopt_jax.core.specs import CoilDofExtractionSpec, CoilSetDofExtractionSpec, CurveXYZFourierSpec
    from simsopt_jax.core.specs import OptimizableDofMapSpec, apply_coil_symmetry, make_coil_dof_extraction_spec
    from simsopt_jax.core.specs import make_coil_set_dof_extraction_spec, make_curve_rzfourier_spec
    from simsopt_jax.core.specs import make_curve_xyzfourier_spec, make_optimizable_dof_map_spec
    from simsopt_jax.objectives import StageTwoObjectiveConfig, fused_stage_two_objective, fused_stage_two_values
    from simsopt_jax.objectives import make_stage_two_problem, stage_two_geometric_penalty
    from simsopt_jax.objectives.stage_two import _ordinary_curve_geometry, stage_two_coil_geometry, stage_two_geometry
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo import JaxCurveCurveDistance, JaxCurveLength, JaxCurveSurfaceDistance
    from simsopt_jax_adapters.geo import JaxLpCurveCurvature, JaxMeanSquaredCurvature
    from simsopt_jax_adapters.objectives import JaxFusedStageTwoObjective, JaxSquaredFlux
    from simsopt_jax_adapters.objectives import fused as fused_objective_module
    from unittest_jax_support import compiled_stage_two_objective, compiled_stage_two_value_and_grad


_DEFINITIONS = ("quadratic flux", "normalized", "local")


def _flux_coils(surface, *, shared_dofs: bool = False) -> list:
    """Symmetric coils with a fixed current and fixed curve DOFs at a perturbed state; ``shared_dofs`` adds a coil
    whose curve shares the DOFs of the second base curve and whose current shares the DOFs of the third."""
    base_curves = create_equally_spaced_curves(3, surface.nfp, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24)
    rng = np.random.default_rng(11)
    for curve in base_curves:
        curve.x = curve.x + 0.02 * rng.standard_normal(curve.x.shape)
    base_curves[0].fix("xc(1)")
    base_currents = [Current(1e5), Current(1.2e5), Current(0.9e5)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    if shared_dofs:
        twin_curve = CurveXYZFourier(np.linspace(0, 1, 20, endpoint=False) + 0.01, 3, dofs=base_curves[1].dofs)
        twin_current = Current(0.0, dofs=base_currents[2].dofs)
        coils.append(Coil(twin_curve, 0.5 * twin_current))
    return coils


def _target(surface, kind: str):
    if kind == "none":
        return None
    nphi, ntheta = surface.normal().shape[:2]
    return 0.05 * np.sin(np.linspace(0.0, 3.0, nphi * ntheta)).reshape((nphi, ntheta))


def _objectives(definition, target_kind="none", *, shared_dofs=False):
    surface = qa_surface()
    coils = _flux_coils(surface, shared_dofs=shared_dofs)
    target = _target(surface, target_kind)
    native = SquaredFlux(surface, BiotSavart(coils), target=target, definition=definition)
    adapter = JaxSquaredFlux(surface, JaxBiotSavart(coils), target=target, definition=definition)
    return surface, native, adapter


class TestSquaredFluxJax(JaxTestCase):
    """JaxSquaredFlux and the JAX integral_BdotN against native SquaredFlux and the C++ integral."""

    def test_integral_bdotn_snapshots_numpy_before_pending_evaluation(self):
        """All flux definitions retain caller field, target and normal buffers."""
        for definition, operand_index, misaligned in itertools.product(_DEFINITIONS, range(3), (False, True)):
            with self.subTest(definition=definition, operand=operand_index, misaligned=misaligned), self.case():
                self._case_integral_snapshot(definition, operand_index, misaligned)

    def _case_integral_snapshot(self, definition, operand_index, misaligned):
        """Mutate one host operand while a warmed flux integral waits for device work."""
        host_inputs = [distinct_host_array(shape, misaligned=misaligned)
                       for shape in ((32, 32, 3), (32, 32), (32, 32, 3))]
        expected = np.asarray(integral_BdotN(host_inputs[0], host_inputs[1], host_inputs[2], definition)).copy()
        inputs: list[np.ndarray | jax.Array] = [jnp.asarray(value) for value in host_inputs]
        inputs[operand_index] = host_inputs[operand_index]
        gate_index = 2 if operand_index == 0 else 0
        gate = make_execution_gate()
        gate(inputs[gate_index]).block_until_ready()
        inputs[gate_index] = gate(inputs[gate_index])
        result = integral_BdotN(inputs[0], inputs[1], inputs[2], definition)
        self.assertFalse(result.is_ready(), "flux integral finished before caller mutation")
        host_inputs[operand_index][...] = -7.0
        np.testing.assert_array_equal(result.block_until_ready(), expected)

    def test_integral_bdotn_supports_nested_jit_and_grad(self):
        """Static definition specialization and field derivatives work inside outer jit."""
        B = jnp.asarray(distinct_host_array((2, 3, 3)))
        target = jnp.zeros((2, 3))
        normal = jnp.asarray(distinct_host_array((2, 3, 3)))
        for definition in _DEFINITIONS:
            with self.subTest(definition=definition):
                evaluate = lambda field: integral_BdotN(field, target, normal, definition)
                np.testing.assert_array_equal(jit(evaluate)(B), evaluate(B))
                np.testing.assert_allclose(jit(jax.grad(evaluate))(B), jax.grad(evaluate)(B), rtol=1e-14, atol=1e-14)

    def test_integral_bdotn_matches_cpp(self):
        """All flux definitions and target forms match the native C++ integral."""
        self.run_cases(self._case_integral_bdotn_matches_cpp, [
            dict(target_kind=target_kind, empty=empty, definition=definition)
            for target_kind, empty in (("none", False), ("array", False), ("none", True))
            for definition in _DEFINITIONS])

    def _case_integral_bdotn_matches_cpp(self, target_kind, empty, definition):
        rng = np.random.default_rng(5)
        B = rng.standard_normal((6, 7, 3))
        normal = rng.standard_normal((6, 7, 3))
        target = rng.standard_normal((6, 7)) if target_kind == "array" else np.zeros((6, 7))
        if empty:
            target = np.zeros((0,))
        expected = sopp.integral_BdotN(B, target, normal, definition)
        actual = integral_BdotN(jnp.asarray(B), jnp.asarray(target), jnp.asarray(normal), definition)
        np.testing.assert_allclose(float(actual), expected, rtol=1e-12, atol=1e-14)

    def test_integral_bdotn_singular_inputs_match_cpp(self):
        """Zero normals and zero fields give the C++ nan/inf (or value), not a masked result."""
        self.run_cases(self._case_integral_bdotn_singular_inputs_match_cpp, [
            dict(case=case, definition=definition)
            for case in ("zero_normal", "zero_field") for definition in _DEFINITIONS])

    def _case_integral_bdotn_singular_inputs_match_cpp(self, case, definition):
        rng = np.random.default_rng(6)
        B = rng.standard_normal((6, 7, 3))
        normal = rng.standard_normal((6, 7, 3))
        target = np.zeros((6, 7))
        if case == "zero_normal":
            normal[2, 3] = 0.0
        else:
            B[:] = 0.0
        expected = sopp.integral_BdotN(B, target, normal, definition)
        actual = float(integral_BdotN(jnp.asarray(B), jnp.asarray(target), jnp.asarray(normal), definition))
        if not (case == "zero_field" and definition == "quadratic flux"):
            self.assertFalse(np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_integral_bdotn_evaluates_float32_inputs_in_double_like_cpp(self):
        """float32 inputs are evaluated in float64, as the C++ kernel does: at |B| ~ 1e20 the squared field overflows
        float32 (max 3.4e38) but not double."""
        rng = np.random.default_rng(8)
        B = (1e20 * rng.standard_normal((6, 7, 3))).astype(np.float32)
        normal = rng.standard_normal((6, 7, 3)).astype(np.float32)
        target = (1e19 * rng.standard_normal((6, 7))).astype(np.float32)
        for definition in _DEFINITIONS:
            with self.subTest(definition=definition):
                expected = sopp.integral_BdotN(
                    B.astype(np.float64), target.astype(np.float64), normal.astype(np.float64), definition)
                actual = integral_BdotN(B, target, normal, definition)
                self.assertEqual(actual.dtype, np.float64)
                self.assertTrue(np.isfinite(expected))
                np.testing.assert_allclose(float(actual), expected, rtol=1e-12)

    def test_squared_flux_matches_native_value_gradient_and_partials(self):
        """Squared flux matches native values, free gradients and fixed partials."""
        self.run_cases(self._case_squared_flux_matches_native_value_gradient_and_partials, [
            dict(target_kind=target_kind, shared_dofs=shared_dofs, definition=definition)
            for target_kind, shared_dofs in (("none", False), ("array", True)) for definition in _DEFINITIONS])

    def _case_squared_flux_matches_native_value_gradient_and_partials(self, target_kind, shared_dofs, definition):
        _, native, adapter = _objectives(definition, target_kind, shared_dofs=shared_dofs)
        self.assertEqual(adapter.dof_names, native.dof_names)
        assert_matches_native_including_nans(definition, native, adapter)

    def test_squared_flux_gradient_matches_central_differences(self):
        """Squared-flux gradients agree with independent central differences."""
        self.run_cases(self._case_squared_flux_gradient_matches_central_differences,
                       [dict(definition=definition) for definition in _DEFINITIONS])

    def _case_squared_flux_gradient_matches_central_differences(self, definition):
        _, _, adapter = _objectives(definition, "array", shared_dofs=True)
        x0 = np.array(adapter.x, dtype=float)
        direction = np.random.default_rng(2).standard_normal(x0.shape) * np.maximum(np.abs(x0), 1.0)
        step = 1e-7
        adapter.x = x0 + step * direction
        plus = adapter.J()
        adapter.x = x0 - step * direction
        minus = adapter.J()
        adapter.x = x0
        np.testing.assert_allclose(adapter.dJ() @ direction, (plus - minus) / (2 * step), rtol=1e-6)

    def test_squared_flux_follows_dof_changes_and_sets_field_points(self):
        """Construction sets surface evaluation points and values follow coil DOF changes."""
        surface, native, adapter = _objectives("quadratic flux")
        np.testing.assert_array_equal(adapter.field.get_points_cart(), surface.gamma().reshape((-1, 3)))
        adapter.x = np.asarray(adapter.x) * 1.01
        np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(adapter.dJ(), native.dJ(), rtol=1e-11, atol=1e-13)

    def test_target_and_definition_are_read_at_every_evaluation(self):
        """Target and definition mutations are reflected in values, gradients and specs."""
        surface, native, adapter = _objectives("quadratic flux", "none", shared_dofs=True)
        adapter.J()
        adapter.dJ()
        target = np.ascontiguousarray(_target(surface, "array"))
        for definition in ("local", "normalized", "quadratic flux"):
            native.target, native.definition = target.copy(), definition
            adapter.target, adapter.definition = target.copy(), definition
            np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14, err_msg=definition)
            np.testing.assert_allclose(adapter.dJ(), native.dJ(), rtol=1e-11, atol=1e-13,
                                       err_msg=definition)
            self.assertEqual(adapter.fixed_surface_flux_spec().definition, definition)
        native.target *= 3.0
        adapter.target *= 3.0
        np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(np.asarray(adapter.fixed_surface_flux_spec().target), native.target)

    def test_squared_flux_makes_no_implicit_transfers(self):
        """J and dJ move data only through explicit transfers, also after a DOF change."""
        _, native, adapter = _objectives("quadratic flux", "array", shared_dofs=True)
        adapter.J()
        adapter.dJ()
        with jax.transfer_guard("disallow"):
            adapter.J()
            adapter.dJ()
            adapter.x = np.asarray(adapter.x) * 1.01
            value = adapter.J()
            gradient = adapter.dJ()
        np.testing.assert_allclose(value, native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(gradient, native.dJ(), rtol=1e-11, atol=1e-13)

    def test_squared_flux_rejects_surface_changes_after_construction(self):
        """Changing captured surface DOFs rejects value, gradient and spec evaluation."""
        surface, _, adapter = _objectives("quadratic flux")
        surface.set_rc(1, 0, surface.get_rc(1, 0) * 1.01)
        for evaluate in (adapter.J, adapter.dJ, adapter.fixed_surface_flux_spec):
            with self.assertRaisesRegex(RuntimeError, "surface DOFs have changed"):
                evaluate()

    def test_squared_flux_rejects_unknown_definition(self):
        """An unsupported flux definition raises ValueError at construction."""
        surface = qa_surface()
        with self.assertRaisesRegex(ValueError, "Unrecognized option"):
            JaxSquaredFlux(surface, JaxBiotSavart(_flux_coils(surface)), definition="flux")

    def test_squared_flux_gradient_taylor(self):
        """Every flux definition attains native centered Taylor convergence."""
        self.run_cases(self._case_squared_flux_gradient_taylor,
                       [dict(definition=definition) for definition in _DEFINITIONS])

    def _case_squared_flux_gradient_taylor(self, definition):
        """Use the native flux stencil and six halved steps with shared coil DOFs."""
        _, _, adapter = _objectives(definition, "array", shared_dofs=True)
        x0 = np.array(adapter.x, dtype=float)
        direction = np.random.RandomState(1).uniform(size=x0.shape)
        derivative = adapter.dJ() @ direction
        previous_error = 1e10
        for exponent in range(11, 17):
            step = 0.5 ** exponent
            adapter.x = x0 + step * direction
            plus = adapter.J()
            adapter.x = x0 - step * direction
            error = abs((plus - adapter.J()) / (2 * step) - derivative)
            self.assertLess(error, 0.6 ** 2 * previous_error, f"{definition}, step={step}")
            previous_error = error
        adapter.x = x0


_NCOILS = 3
# Thresholds that make every penalty active at the perturbed test state.
if JAX_IMPORT_ERROR is None:
    _ACTIVE = StageTwoObjectiveConfig(
        num_basecurves=_NCOILS, length_weight=1e-3, curve_curve_minimum_distance=0.6, curve_curve_weight=10.0,
        curve_surface_minimum_distance=0.4, curve_surface_weight=2.0, curvature_threshold=1.0, curvature_weight=1e-2,
        mean_squared_curvature_threshold=1.0, mean_squared_curvature_weight=1e-2)


def _coils(surface, *, perturb: bool = True, shared_dofs: bool = False) -> tuple[list, list]:
    """Base curves and symmetric coils with a fixed current and fixed curve DOFs; ``shared_dofs`` adds a coil whose
    curve shares the DOFs object of the second base curve (on a rotated, offset grid) and whose current shares the
    DOFs of the third base current."""
    base = create_equally_spaced_curves(_NCOILS, surface.nfp, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24)
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
        twin = CurveXYZFourier(np.linspace(0, 1, 24, endpoint=False) + 0.011, 3, dofs=base[1].dofs)
        coils.append(Coil(twin, -0.5 * Current(0.0, dofs=currents[2].dofs)))
    return base, coils


def _penalty_terms(surface, base, coils, config: StageTwoObjectiveConfig, *, jax_terms: bool = False) -> list:
    """The weighted geometric penalty terms that ``config`` describes, as in the Stage-II examples, native or, with
    ``jax_terms``, their JAX adapters: a ``None`` weight leaves its term out, a zero weight keeps ``0 * term``."""
    length, curve_curve, curve_surface, curvature, mean_squared_curvature = (
        (JaxCurveLength, JaxCurveCurveDistance, JaxCurveSurfaceDistance, JaxLpCurveCurvature, JaxMeanSquaredCurvature)
        if jax_terms
        else (CurveLength, CurveCurveDistance, CurveSurfaceDistance, LpCurveCurvature, MeanSquaredCurvature)
    )
    curves = [coil.curve for coil in coils]
    total_length = sum(length(curve) for curve in base)
    weighted_terms = [
        (config.length_weight, (total_length if config.length_target is None
                                else QuadraticPenalty(total_length, config.length_target, config.length_target_mode))),
        (config.curve_curve_weight,
         curve_curve(curves, config.curve_curve_minimum_distance, num_basecurves=config.num_basecurves)),
        (config.curve_surface_weight, curve_surface(curves, surface, config.curve_surface_minimum_distance)),
        (config.curvature_weight, sum(curvature(curve, 2, config.curvature_threshold) for curve in base)),
        (config.mean_squared_curvature_weight, sum(
            QuadraticPenalty(mean_squared_curvature(curve), config.mean_squared_curvature_threshold,
                             config.mean_squared_curvature_target_mode)
            for curve in base
        )),
    ]
    return [weight * term for weight, term in weighted_terms if weight is not None]


def _native_penalty(surface, base, coils, config: StageTwoObjectiveConfig) -> Optimizable:
    """The native weighted geometric penalty that ``config`` describes; see :func:`_penalty_terms`."""
    return cast(Optimizable, sum(_penalty_terms(surface, base, coils, config)))


def _native_composite(surface, base, coils, config: StageTwoObjectiveConfig) -> Optimizable:
    """The native objective that ``config`` describes: squared flux plus :func:`_native_penalty`."""
    return SquaredFlux(surface, BiotSavart(coils)) + _native_penalty(surface, base, coils, config)


def _problem(config: StageTwoObjectiveConfig, *, perturb: bool = True, shared_dofs: bool = False):
    surface = qa_surface()
    base, coils = _coils(surface, perturb=perturb, shared_dofs=shared_dofs)
    field = JaxBiotSavart(coils)
    flux = JaxSquaredFlux(surface, field)
    native = _native_composite(surface, base, coils, config)
    return field, flux, native, make_stage_two_problem(field, flux.fixed_surface_flux_spec(), config)


if JAX_IMPORT_ERROR is None:
    _values = jit(fused_stage_two_values)


class TestStageTwo(JaxTestCase):
    """The fused Stage-II objective against the native composite objective."""

    def test_fused_objective_matches_native_composite(self):
        """The fused value and gradient match the native composite with shared DOFs, every length and curvature target
        mode, and absent or zero-weight terms."""
        modes = (("linear_length", None, "max", "max"), ("max_length_identity_msc", 5.0, "max", "identity"),
                 ("identity_length", 60.0, "identity", "max"))
        rows = [(f"{case_id_0}-{case_id_1}", replace(_ACTIVE, length_target=length_target,
                                                     length_target_mode=length_mode,
                                                     mean_squared_curvature_target_mode=msc_mode), shared_dofs)
                for case_id_0, shared_dofs in (("symmetric", False), ("shared_dofs", True))
                for case_id_1, length_target, length_mode, msc_mode in modes]
        rows += [(case_id, replace(_ACTIVE, curve_surface_weight=weight, curvature_weight=weight), False)
                 for case_id, weight in (("term_removed", None), ("zero_weight", 0.0))]
        for case_id, config, shared_dofs in rows:
            with self.subTest(case=case_id), self.case():
                self._case_fused_objective_matches_native_composite(config, shared_dofs)

    def _case_fused_objective_matches_native_composite(self, config, shared_dofs):
        field, _, native, problem = _problem(config, shared_dofs=shared_dofs)
        value, gradient = compiled_stage_two_value_and_grad(problem, jnp.asarray(field.x))
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(gradient), native_field_gradient(native, field), rtol=1e-11, atol=1e-13)

    def test_fused_objective_reads_sample_second_derivatives_only_for_curvature(self):
        """As on native, perturbation samples with first derivatives only (GaussianSampler's default) suffice unless a
        curvature term reads a perturbed curve: the flux and length objective of perturbed coils and curvature of
        unperturbed base curves beside perturbed coils match native; curvature of a perturbed base curve raises."""
        surface = qa_surface()
        base, coils = _coils(surface)
        sampler = GaussianSampler(base[0].quadpoints, 1e-3, 0.2, n_derivs=1)
        rng = np.random.default_rng(11)
        perturbed = coils_via_symmetries(
            [CurvePerturbed(curve, PerturbationSample(sampler, randomgen=rng)) for curve in base],
            [coil.current for coil in coils[:_NCOILS]], surface.nfp, True)
        flux_and_length = StageTwoObjectiveConfig(num_basecurves=_NCOILS, length_weight=1e-3)
        curvature = replace(flux_and_length, curvature_threshold=1.0, curvature_weight=1e-2,
                            mean_squared_curvature_threshold=1.0, mean_squared_curvature_weight=1e-2)
        for case_id, case_coils, config in (("perturbed_flux_and_length", perturbed, flux_and_length),
                                            ("unperturbed_base_curvature", coils[:_NCOILS] + perturbed, curvature)):
            with self.subTest(case=case_id), self.case():
                field = JaxBiotSavart(case_coils)
                flux = JaxSquaredFlux(surface, field)
                case_base = [coil.curve for coil in case_coils[:_NCOILS]]
                native = _native_composite(surface, case_base, case_coils, config)
                problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), config)
                value, gradient = compiled_stage_two_value_and_grad(problem, jnp.asarray(field.x))
                np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(np.asarray(gradient), native_field_gradient(native, field), rtol=1e-11,
                                           atol=1e-13)
        with self.case():
            field = JaxBiotSavart(perturbed)
            flux = JaxSquaredFlux(surface, field)
            problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), curvature)
            with self.assertRaisesRegex(ValueError, "has 1 derivatives"):
                compiled_stage_two_value_and_grad(problem, jnp.asarray(field.x))
            with self.assertRaises(ValueError):
                LpCurveCurvature(perturbed[0].curve, 2, 1.0).J()

    def test_fused_diagnostics_match_native_terms(self):
        """Flux, penalty, normal-field and length diagnostics match native quantities."""
        field, flux, native, problem = _problem(_ACTIVE)
        surface = flux.surface
        objective, squared_flux, penalty, max_normal_field, length = (float(value) for value in
                                                                      _values(problem, jnp.asarray(field.x)))
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
        plus = compiled_stage_two_objective(problem, x0 + step * direction)
        minus = compiled_stage_two_objective(problem, x0 - step * direction)
        _, gradient = compiled_stage_two_value_and_grad(problem, x0)
        np.testing.assert_allclose(float(gradient @ direction), float((plus - minus) / (2 * step)), rtol=1e-6)

    def test_rebuilt_problem_with_new_weights_reuses_the_compiled_program(self):
        """Changing numeric weights reuses compilation and preserves native parity."""
        self.run_cases(self._case_rebuilt_problem_with_new_weights_reuses_the_compiled_program,
                       [dict(length_weight=weight) for weight in (1e-4, 0.0, 0, 2, np.float32(5e-4))])

    def _case_rebuilt_problem_with_new_weights_reuses_the_compiled_program(self, length_weight):
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
        np.testing.assert_allclose(np.asarray(gradient), native_field_gradient(native, field), rtol=1e-11, atol=1e-13)

    def test_rebuilt_problem_follows_fixed_dof_values(self):
        """Fixed DOF values are operands of the problem: rebuilding picks up new ones."""
        field, flux, native, problem = _problem(_ACTIVE)
        fixed_curve = field.coils[0].curve
        fixed_current = field.coils[0].current
        fixed_curve.fix_all()
        x = jnp.asarray(field.x)
        problem = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), _ACTIVE)
        before, _ = compiled_stage_two_value_and_grad(problem, x)
        fixed_curve.local_full_x = np.asarray(fixed_curve.local_full_x) * 1.02
        fixed_current.local_full_x = np.asarray(fixed_current.local_full_x) * 0.9
        stale, _ = compiled_stage_two_value_and_grad(problem, x)
        rebuilt = make_stage_two_problem(field, flux.fixed_surface_flux_spec(), _ACTIVE)
        value, gradient = compiled_stage_two_value_and_grad(rebuilt, x)
        np.testing.assert_allclose(float(stale), float(before), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(float(value), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(gradient), native_field_gradient(native, field), rtol=1e-11, atol=1e-13)
        self.assertGreater(abs(float(value) - float(before)), 1e-6 * abs(float(before)))

    def test_distance_terms_of_a_far_degenerate_coil_have_finite_zero_gradients(self):
        """A coil collapsed to a far point (zero tangents) adds nothing, not NaN gradients."""
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ring = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)
        ring_dash = 2.0 * np.pi * np.stack((-np.sin(angles), np.cos(angles), np.zeros_like(angles)), axis=1)
        gamma = jnp.asarray(np.stack((ring, 1.02 * ring, np.full_like(ring, 50.0))))
        gammadash = jnp.asarray(np.stack((ring_dash, 1.02 * ring_dash, np.zeros_like(ring))))
        config = StageTwoObjectiveConfig(num_basecurves=3, curve_curve_minimum_distance=0.1, curve_curve_weight=1.0,
                                         curve_surface_minimum_distance=0.2, curve_surface_weight=1.0)

        def penalty(current_gamma, current_gammadash):
            return stage_two_geometric_penalty(current_gamma, current_gammadash, jnp.zeros_like(current_gammadash),
                                               jnp.asarray(1.1 * ring), jnp.asarray(ring), config)

        value, (dgamma, dgammadash) = jax.value_and_grad(penalty, argnums=(0, 1))(gamma, gammadash)
        self.assertGreater(float(value), 0.0)
        self.assertTrue(np.all(np.isfinite(dgamma)) and np.all(np.isfinite(dgammadash)))
        self.assertFalse(np.any(dgamma[2]) or np.any(dgammadash[2]))
        self.assertTrue(np.any(dgammadash[0]))

    def test_fused_distance_candidates_at_the_threshold(self):
        """Off a tie the fused candidate choice is native's; at a tie it is one of the two."""
        self.run_cases(self._case_fused_distance_candidates_at_the_threshold, [
            dict(position=position, minimum_distance=minimum_distance, term=term)
            for position, minimum_distance in DISTANCE_TIE_CASES for term in ("curve_curve", "curve_surface")])

    def _case_fused_distance_candidates_at_the_threshold(self, position, minimum_distance, term):
        """Compare fused and native outcomes for constant curves at and near one tie: with zero tangents a candidate
        pair has a non-finite gradient (native and fused alike) and a non-candidate pair a finite zero gradient, the
        value being zero either way; a relative margin of 1e-9 moves the squared distance about 1e7 ulp off the
        squared threshold, where rounding cannot change the decision."""
        origin_curve = circle_curve((0.0, 0.0, 0.0), 0.0, numquadpoints=8)
        point_curve = circle_curve(position, 0.0, numquadpoints=8)
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
                config = StageTwoObjectiveConfig(num_basecurves=2, curve_curve_minimum_distance=threshold,
                                                 curve_curve_weight=1.0)
            else:
                native = CurveSurfaceDistance(curves, surface, threshold)
                config = StageTwoObjectiveConfig(num_basecurves=1, curve_surface_minimum_distance=threshold,
                                                 curve_surface_weight=1.0)

            def penalty(current_gamma, current_gammadash, config=config):
                return stage_two_geometric_penalty(current_gamma, current_gammadash, jnp.zeros_like(current_gammadash),
                                                   surface_gamma, surface_normal, config)

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
        self.assertTrue(any(np.array_equal(tie, outcomes[side][1], equal_nan=True) for side in ("below", "above")),
                        f"fused gradient at the tie is neither off-tie outcome: {tie}")

    def test_fused_program_compiles_and_runs_under_the_strict_transfer_guard(self):
        """The fused value and gradient compile and run with explicit transfers only."""
        field, _, _, problem = _problem(_ACTIVE)
        value_and_grad = jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
        x = field.x
        with jax.transfer_guard("disallow"):
            value, gradient = jax.device_get(value_and_grad(problem, as_jax_float64(x)))
        # A separately compiled program may reassociate GPU reductions.
        expected_value, expected_gradient = compiled_stage_two_value_and_grad(problem, jnp.asarray(x))
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
        surface = qa_surface()
        _, coils = _coils(surface)
        odd = CurveXYZFourier(30, 3)
        odd.local_full_x = coils[2].curve.local_full_x
        field = JaxBiotSavart([*coils, Coil(odd, Current(1e4))])
        flux = JaxSquaredFlux(surface, field)
        with self.assertRaisesRegex(ValueError, "matching nonempty quadrature grids"):
            make_stage_two_problem(field, flux.fixed_surface_flux_spec(), _ACTIVE)

    def test_small_stage_two_run_follows_the_native_trajectory(self):
        """L-BFGS-B on the fused objective retraces the native run of the same problem from the unperturbed coils with
        the upstream example weights: their gradients agree to round-off, which the first iterations amplify only to
        about 1e-13 (20 iterations); later, line-search decisions let round-off grow chaotically, so the comparison
        stops at 20 iterations."""
        config = StageTwoObjectiveConfig(
            num_basecurves=_NCOILS, length_weight=1e-6, curve_curve_minimum_distance=0.1, curve_curve_weight=1000.0,
            curve_surface_minimum_distance=0.3, curve_surface_weight=10.0, curvature_threshold=5.0,
            curvature_weight=1e-6, mean_squared_curvature_threshold=5.0, mean_squared_curvature_weight=1e-6)
        field, _, native, problem = _problem(config, perturb=False)
        x0 = np.array(field.x, dtype=np.float64)
        options = {"maxiter": 20, "maxcor": 300}

        def native_fun(x):
            native.x = x
            return native.J(), native.dJ()

        def fused_fun(x):
            value, gradient = jax.device_get(compiled_stage_two_value_and_grad(problem, x))
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
        _, gradient = compiled_stage_two_value_and_grad(problem, x0)
        derivative = float(gradient @ direction)
        previous_error = 1e10
        for exponent in range(11, 17):
            step = 0.5 ** exponent
            plus = float(compiled_stage_two_objective(problem, x0 + step * direction))
            minus = float(compiled_stage_two_objective(problem, x0 - step * direction))
            error = abs((plus - minus) / (2 * step) - derivative)
            self.assertLess(error, 0.6 ** 2 * previous_error, f"step={step}")
            previous_error = error


_PENALTY_WEIGHTS = ("length_weight", "curve_curve_weight", "curve_surface_weight", "curvature_weight",
                    "mean_squared_curvature_weight")


def _matching_configs() -> dict[str, StageTwoObjectiveConfig]:
    """Squared flux alone, each penalty alone, and both target modes with targets below and above the test coils'
    total length and mean squared curvatures, where ``"max"`` and ``"identity"`` differ."""
    alone = {name: replace(_ACTIVE, **{other: None for other in _PENALTY_WEIGHTS if other != name})
             for name in _PENALTY_WEIGHTS}
    return {
        "flux": replace(_ACTIVE, **{name: None for name in _PENALTY_WEIGHTS}),
        **alone,
        "length_max_target": replace(alone["length_weight"], length_target=5.0),
        "length_identity_target": replace(alone["length_weight"], length_target=60.0, length_target_mode="identity"),
        "mean_squared_curvature_max_short": replace(alone["mean_squared_curvature_weight"],
                                                    mean_squared_curvature_threshold=10.0),
        "mean_squared_curvature_identity": replace(alone["mean_squared_curvature_weight"],
                                                   mean_squared_curvature_threshold=10.0,
                                                   mean_squared_curvature_target_mode="identity"),
        "all": _ACTIVE,
    }


def _jax_objective(config: StageTwoObjectiveConfig, *, shared_dofs: bool = False) -> tuple[Optimizable, list]:
    """The term-by-term JAX objective that ``config`` describes at the perturbed test coils, and the base curves."""
    surface = qa_surface()
    base, coils = _coils(surface, shared_dofs=shared_dofs)
    flux = JaxSquaredFlux(surface, JaxBiotSavart(coils))
    return cast(Optimizable, sum([flux, *_penalty_terms(surface, base, coils, config, jax_terms=True)])), base


def _example_objective():
    """The objective of examples/2_Intermediate/stage_two_optimization.py, with its weights, thresholds and length
    ``Weight``, at the perturbed test coils."""
    surface = qa_surface()
    base, coils = _coils(surface)
    curves = [coil.curve for coil in coils]
    length_weight = Weight(1e-6)
    objective = (JaxSquaredFlux(surface, JaxBiotSavart(coils))
                 + length_weight * sum(JaxCurveLength(curve) for curve in base)
                 + 1000 * JaxCurveCurveDistance(curves, 0.1, num_basecurves=_NCOILS)
                 + 10 * JaxCurveSurfaceDistance(curves, surface, 0.3)
                 + 1e-6 * sum(JaxLpCurveCurvature(curve, 2, 5.0) for curve in base)
                 + 1e-6 * sum(QuadraticPenalty(JaxMeanSquaredCurvature(curve), 5, "max") for curve in base))
    return objective, length_weight, base


def _random_dofs(objective: Optimizable) -> np.ndarray:
    x0 = np.array(objective.x)
    return x0 + 0.01 * np.maximum(np.abs(x0), 1.0) * np.random.default_rng(5).standard_normal(x0.shape)


@contextmanager
def _counted_fused_program():
    """Count the runs and traces of the program JaxFusedStageTwoObjective runs."""
    runs, traces = [], []

    def objective(problem, parameters):
        traces.append(parameters)
        return fused_stage_two_objective(problem, parameters)

    program = jit(jax.value_and_grad(objective, argnums=1))

    def counted(problem, parameters):
        runs.append(parameters)
        return program(problem, parameters)

    with mock.patch.object(fused_objective_module, "_fused_value_and_grad", counted):
        yield runs, traces


def _term_fixture() -> SimpleNamespace:
    surface = qa_surface()
    base, coils = _coils(surface)
    return SimpleNamespace(surface=surface, base=base, coils=coils, curves=[coil.curve for coil in coils],
                           flux=JaxSquaredFlux(surface, JaxBiotSavart(coils)),
                           lengths=sum(JaxCurveLength(curve) for curve in base))


# Objectives the fused formula cannot represent: builder of the fixture, error type and message pattern.
_UNSUPPORTED = {
    "native_term": (lambda f: f.flux + 1e-3 * sum(CurveLength(curve) for curve in f.base), TypeError,
                    "CurveLength has no fused"),
    "native_flux": (lambda f: SquaredFlux(f.surface, BiotSavart(f.coils)), TypeError, "SquaredFlux has no fused"),
    "no_flux": (lambda f: 1e-3 * f.lengths, ValueError, "needs one JaxSquaredFlux"),
    "second_flux": (lambda f: f.flux + JaxSquaredFlux(f.surface, JaxBiotSavart(_coils(f.surface)[1])), ValueError,
                    "2 JaxSquaredFlux terms"),
    "scaled_flux": (lambda f: 2.0 * f.flux, ValueError, "JaxSquaredFlux unscaled"),
    "curvature_p": (lambda f: f.flux + sum(JaxLpCurveCurvature(curve, 3, 1.0) for curve in f.base), ValueError,
                    "JaxLpCurveCurvature with p=2"),
    "penalty_min": (lambda f: f.flux + QuadraticPenalty(f.lengths, 5.0, "min"), ValueError, "f='min'"),
    "bare_mean_squared_curvature": (lambda f: f.flux + sum(JaxMeanSquaredCurvature(curve) for curve in f.base),
                                    ValueError, "JaxMeanSquaredCurvature only inside QuadraticPenalty"),
    "penalty_of_scaled_lengths": (lambda f: f.flux + QuadraticPenalty(2.0 * f.lengths, 5.0), ValueError,
                                  "QuadraticPenalty of an unscaled sum"),
    "penalty_of_curvature": (lambda f: f.flux + QuadraticPenalty(JaxLpCurveCurvature(f.base[0], 2, 1.0), 1.0),
                             ValueError, "QuadraticPenalty of JaxLpCurveCurvature has no fused form"),
    "penalty_of_flux": (lambda f: f.flux + QuadraticPenalty(f.flux), ValueError,
                        "QuadraticPenalty of JaxSquaredFlux has no fused form"),
    "skipped_leading_coil": (lambda f: f.flux + sum(JaxCurveLength(curve) for curve in f.base[1:]), ValueError,
                             "JaxCurveLength terms must cover the leading coil"),
    "repeated_curve": (lambda f: f.flux + sum(JaxCurveLength(curve) for curve in f.base[:1] + f.base[:2]),
                       ValueError, "JaxCurveLength terms must cover the leading coil"),
    "curve_outside_the_field": (lambda f: f.flux + JaxCurveLength(_coils(f.surface)[0][0]), ValueError,
                                "JaxCurveLength reads a curve that is not a coil"),
    "different_weights": (lambda f: f.flux + 1e-3 * JaxCurveLength(f.base[0])
                          + 2e-3 * sum(JaxCurveLength(curve) for curve in f.base[1:]), ValueError,
                          "JaxCurveLength weights differ"),
    "different_thresholds": (lambda f: f.flux + sum(JaxLpCurveCurvature(curve, 2, 1.0 + index)
                                                    for index, curve in enumerate(f.base)), ValueError,
                             "JaxLpCurveCurvature thresholds differ"),
    "different_base_curves": (lambda f: f.flux + f.lengths + sum(JaxLpCurveCurvature(curve, 2, 1.0)
                                                                 for curve in f.base[:2]), ValueError,
                              "must cover the same base"),
    "lengths_and_length_penalty": (lambda f: f.flux + f.lengths + QuadraticPenalty(f.lengths, 5.0), ValueError,
                                   "one length term"),
    "curve_curve_base_count": (lambda f: f.flux + f.lengths + JaxCurveCurveDistance(f.curves, 0.6, num_basecurves=2),
                               ValueError, "num_basecurves=2"),
    "curve_curve_downsample": (lambda f: f.flux + JaxCurveCurveDistance(f.curves, 0.6, num_basecurves=_NCOILS,
                                                                        downsample=2),
                               ValueError, "downsample=1"),
    "curve_curve_subset": (lambda f: f.flux + JaxCurveCurveDistance(f.curves[:-1], 0.6, num_basecurves=_NCOILS),
                           ValueError, "JaxCurveCurveDistance must take the"),
    "curve_surface_of_base_curves": (lambda f: f.flux + JaxCurveSurfaceDistance(f.base, f.surface, 0.4), ValueError,
                                     "JaxCurveSurfaceDistance must take the"),
    "two_curve_curve_terms": (lambda f: f.flux + JaxCurveCurveDistance(f.curves, 0.6, num_basecurves=_NCOILS)
                              + JaxCurveCurveDistance(f.curves, 0.3, num_basecurves=_NCOILS), ValueError,
                              "2 JaxCurveCurveDistance terms"),
}


class TestFusedStageTwoObjective(JaxTestCase):
    """JaxFusedStageTwoObjective against the term-by-term JAX objective it wraps."""

    def test_fused_objective_matches_the_term_by_term_objective(self):
        """At random DOFs the fused value and gradient equal the term-by-term ones for squared flux alone, for each
        penalty alone, for every target mode with excesses of both signs, for all terms with shared DOFs, and for the
        Stage-II example's objective."""
        self.run_cases(self._case_fused_objective_matches_the_term_by_term_objective,
                       [dict(case=case) for case in (*_matching_configs(), "all_shared_dofs", "stage_two_example")])

    def _case_fused_objective_matches_the_term_by_term_objective(self, case):
        config = _matching_configs().get(case, _ACTIVE)
        objective = (_example_objective()[0] if case == "stage_two_example"
                     else _jax_objective(config, shared_dofs=case == "all_shared_dofs")[0])
        fused = JaxFusedStageTwoObjective(objective)
        self.assertEqual(fused.dof_names, objective.dof_names)
        fused.x = _random_dofs(objective)
        self._assert_matches(fused, objective)

    def test_changed_weight_is_read_at_the_next_evaluation(self):
        """A length ``Weight`` scaled in place, as the Stage-II example does between its stages, changes the fused
        value by the weighted length change and keeps it equal to the term-by-term one, with one compiled program."""
        objective, length_weight, base = _example_objective()
        with _counted_fused_program() as (runs, traces):
            fused = JaxFusedStageTwoObjective(objective)
            fused.x = _random_dofs(objective)
            before = fused.J()
            length_weight *= 0.1
            after = fused.J()
            np.testing.assert_allclose(after, objective.J(), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(fused.dJ(), objective.dJ(), rtol=1e-11, atol=1e-13)
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(traces), 1)
        total_length = sum(JaxCurveLength(curve).J() for curve in base)
        np.testing.assert_allclose(before - after, 0.9 * 1e-6 * total_length, rtol=1e-6)

    def test_value_then_gradient_at_the_same_dofs_runs_the_program_once(self):
        """``J`` and ``dJ`` share one program run per DOF vector, in either order."""
        objective, _ = _jax_objective(_ACTIVE)
        with _counted_fused_program() as (runs, traces):
            fused = JaxFusedStageTwoObjective(objective)
            fused.J()
            fused.dJ()
            fused.J()
            self.assertEqual(len(runs), 1)
            fused.x = _random_dofs(objective)
            fused.dJ()
            fused.J()
            self.assertEqual(len(runs), 2)
        self.assertEqual(len(traces), 1)

    def test_evaluation_reads_the_dofs_the_field_holds_at_the_call(self):
        """After ``x`` is set through the wrapper, the DOFs set through a curve, a current or the field are the ones
        the fused value and gradient are computed at."""
        objective, base = _jax_objective(_ACTIVE)
        fused = JaxFusedStageTwoObjective(objective)
        field = next(opt for opt in objective.ancestors if isinstance(opt, JaxBiotSavart))
        current = next(opt for opt in field.unique_dof_lineage if type(opt) is Current and opt.local_dof_size)
        for label, change in (
            ("curve", lambda: setattr(base[0], "x", np.asarray(base[0].x) + 1e-3)),
            ("current", lambda: setattr(current, "x", 1.1 * np.asarray(current.x))),
            ("field", lambda: setattr(field, "x", np.asarray(field.x) - 2e-3)),
        ):
            fused.x = _random_dofs(objective)
            change()
            self._assert_matches(fused, objective, label)

    def test_unsupported_objectives_raise(self):
        """Each term, form and setting that the fused formula cannot represent raises at construction, naming it."""
        self.run_cases(self._case_unsupported_objective_raises, [dict(case=case) for case in _UNSUPPORTED])

    def _case_unsupported_objective_raises(self, case):
        build, error, message = _UNSUPPORTED[case]
        objective = build(_term_fixture())
        with self.assertRaisesRegex(error, message):
            JaxFusedStageTwoObjective(objective)

    def test_settings_changed_after_construction_raise_at_evaluation(self):
        """An exponent, weight or candidate list changed after construction to one the formula cannot represent
        raises at the next evaluation."""
        fixture = _term_fixture()
        curvatures = [JaxLpCurveCurvature(curve, 2, 1.0) for curve in fixture.base]
        first_weight = Weight(1e-3)
        distance = JaxCurveCurveDistance(fixture.curves, 0.6, num_basecurves=_NCOILS)
        fused = JaxFusedStageTwoObjective(
            fixture.flux + sum(curvatures) + first_weight * JaxCurveLength(fixture.base[0])
            + 1e-3 * sum(JaxCurveLength(curve) for curve in fixture.base[1:]) + distance)
        fused.J()
        for change, undo, message in (
            (lambda: setattr(curvatures[1], "p", 3), lambda: setattr(curvatures[1], "p", 2), "p=2"),
            (lambda: setattr(first_weight, "value", 2e-3), lambda: setattr(first_weight, "value", 1e-3),
             "JaxCurveLength weights differ"),
            (lambda: setattr(distance, "candidates", []), lambda: setattr(distance, "candidates", None),
             "JaxCurveCurveDistance has assigned candidates"),
        ):
            change()
            with self.assertRaisesRegex(ValueError, message):
                fused.J()
            undo()
            fused.J()

    def _assert_matches(self, fused, objective, label=""):
        np.testing.assert_allclose(fused.J(), objective.J(), rtol=1e-12, atol=1e-14, err_msg=label)
        np.testing.assert_allclose(fused.dJ(), objective.dJ(), rtol=1e-11, atol=1e-13, err_msg=label)

    def test_replaced_and_nested_scale_factors_are_read_at_each_evaluation(self):
        """A replaced number, a replaced ``Weight``, the outer factor of nested scales and a ``Weight`` changed in place
        inside them all change the fused objective as they change the term-by-term one; a flux scale found later
        raises."""
        f = _term_fixture()
        inner_weight = Weight(1e-2)
        lengths = 1e-3 * f.lengths
        curvatures = 2.0 * (inner_weight * sum(JaxLpCurveCurvature(curve, 2, 1.0) for curve in f.base))
        mean_squared_curvatures = Weight(1e-2) * sum(QuadraticPenalty(JaxMeanSquaredCurvature(curve), 1.0, "max")
                                                     for curve in f.base)
        flux = 1.0 * f.flux
        objective = flux + lengths + curvatures + mean_squared_curvatures
        fused = JaxFusedStageTwoObjective(objective)
        fused.x = _random_dofs(objective)
        for label, change in (("number", lambda: setattr(lengths, "factor", 2e-3)),
                              ("Weight object", lambda: setattr(mean_squared_curvatures, "factor", Weight(5e-2))),
                              ("outer nested", lambda: setattr(curvatures, "factor", 3.0)),
                              ("inner nested Weight", lambda: setattr(inner_weight, "value", 4e-2))):
            before = fused.J()
            change()
            self._assert_matches(fused, objective, label)
            self.assertNotEqual(fused.J(), before, label)
        flux.factor = 2.0
        with self.assertRaisesRegex(ValueError, "JaxSquaredFlux unscaled"):
            fused.J()

    def test_the_wrapper_cannot_be_part_of_another_objective(self):
        """A sum, a scaling and a penalty holding the wrapper ask it for partial derivatives, which raises; its own
        fused gradient still runs."""
        objective, base = _jax_objective(_ACTIVE)
        fused = JaxFusedStageTwoObjective(objective)
        for label, holder in (("sum", fused + JaxCurveLength(base[0])), ("scaling", Weight(2.0) * fused),
                              ("penalty", QuadraticPenalty(fused, 0.0, "identity"))):
            with self.assertRaisesRegex(TypeError, "provides no partial derivatives", msg=label):
                holder.dJ()
        with self.assertRaisesRegex(TypeError, "provides no partial derivatives"):
            fused.dJ(partials=True)
        self._assert_matches(fused, objective)

    def test_fixing_unfixing_and_fixed_values_after_evaluation(self):
        """Fixing and unfixing DOFs and changing fixed values after the first evaluation keep parity."""
        objective, base = _jax_objective(_ACTIVE)
        fused = JaxFusedStageTwoObjective(objective)
        fused.J()
        for label, change in (("fix", lambda: base[1].fix("xs(2)")),
                              ("fixed value", lambda: base[1].set("xs(2)", base[1].get("xs(2)") + 0.01)),
                              ("unfix", lambda: base[1].unfix("xs(2)"))):
            change()
            self.assertEqual(len(fused.x), len(objective.x), label)
            self._assert_matches(fused, objective, label)

    def test_changed_coil_and_curve_lists_are_validated_again(self):
        """A distance curve list or a length curve changed after construction is read again: an unsupported one raises,
        and restoring it restores parity."""
        f = _term_fixture()
        lengths = [JaxCurveLength(curve) for curve in f.base]
        distance = JaxCurveCurveDistance(list(f.curves), 0.6, num_basecurves=_NCOILS)
        objective = f.flux + 1e-3 * sum(lengths) + 10.0 * distance
        fused = JaxFusedStageTwoObjective(objective)
        fused.J()
        for change, undo, message in (
            (lambda: distance.curves.pop(), lambda: distance.curves.append(f.curves[-1]),
             "JaxCurveCurveDistance must take the"),
            (lambda: setattr(lengths[0], "curve", f.curves[-1]), lambda: setattr(lengths[0], "curve", f.base[0]),
             "JaxCurveLength reads a curve that is not a coil|JaxCurveLength terms must cover the leading coil"),
        ):
            change()
            with self.assertRaisesRegex(ValueError, message):
                fused.J()
            undo()
            self._assert_matches(fused, objective, message)

    def test_moved_distance_surface_rebuilds_the_problem(self):
        """Moving a curve-surface distance's own surface, which is not the flux surface, keeps parity."""
        f = _term_fixture()
        distance_surface = qa_surface()
        objective = f.flux + 2.0 * JaxCurveSurfaceDistance(f.curves, distance_surface, 0.4)
        fused = JaxFusedStageTwoObjective(objective)
        before = fused.J()
        distance_surface.local_full_x = 1.05 * distance_surface.local_full_x
        self._assert_matches(fused, objective)
        self.assertNotEqual(fused.J(), before)

    def test_changed_thresholds_targets_and_modes_keep_parity(self):
        """New penalty targets, thresholds and modes, distance thresholds, and a flux target and definition keep
        parity."""
        f = _term_fixture()
        length_penalty = QuadraticPenalty(f.lengths, 5.0, "max")
        mean_squared_curvatures = [QuadraticPenalty(JaxMeanSquaredCurvature(curve), 1.0, "max") for curve in f.base]
        curvatures = [JaxLpCurveCurvature(curve, 2, 1.0) for curve in f.base]
        curve_curve = JaxCurveCurveDistance(f.curves, 0.6, num_basecurves=_NCOILS)
        curve_surface = JaxCurveSurfaceDistance(f.curves, f.surface, 0.4)
        objective = (f.flux + 1e-3 * length_penalty + 1e-2 * sum(mean_squared_curvatures) + 1e-2 * sum(curvatures)
                     + 10.0 * curve_curve + 2.0 * curve_surface)
        fused = JaxFusedStageTwoObjective(objective)
        fused.x = _random_dofs(objective)
        fused.J()

        def set_all(items, name, value):
            for item in items:
                setattr(item, name, value)

        for label, change in (
            ("length target and mode", lambda: (setattr(length_penalty, "cons", 60.0),
                                                setattr(length_penalty, "f", "identity"))),
            ("mean squared curvature threshold and mode", lambda: (set_all(mean_squared_curvatures, "cons", 10.0),
                                                                   set_all(mean_squared_curvatures, "f", "identity"))),
            ("curvature threshold", lambda: set_all(curvatures, "threshold", 2.0)),
            ("distance thresholds", lambda: (setattr(curve_curve, "minimum_distance", 0.5),
                                             setattr(curve_surface, "minimum_distance", 0.3))),
            ("flux target", lambda: setattr(f.flux, "target", _target(f.surface, "array"))),
            ("flux definition", lambda: setattr(f.flux, "definition", "normalized")),
        ):
            before = fused.J()
            change()
            self._assert_matches(fused, objective, label)
            self.assertNotEqual(fused.J(), before, label)

    def test_expired_candidate_override_is_searched_again(self):
        """Assigned candidates raise until a new threshold expires them, as in the adapter; then parity holds."""
        f = _term_fixture()
        distance = JaxCurveCurveDistance(f.curves, 0.6, num_basecurves=_NCOILS)
        objective = f.flux + 10.0 * distance
        fused = JaxFusedStageTwoObjective(objective)
        distance.candidates = []
        with self.assertRaisesRegex(ValueError, "assigned candidates"):
            fused.J()
        distance.minimum_distance = 0.59
        self._assert_matches(fused, objective)
        self.assertGreater(distance.J(), 0.0)


def _dof_map(template: jax.Array, segments: tuple[tuple[int, int, int, int], ...] = ()) -> OptimizableDofMapSpec:
    return make_optimizable_dof_map_spec(template_full_dofs=template, owner_segments=segments, input_mode="full",
                                         input_start=0, input_end=template.size)


def _with_curve_mapping(coil: CoilDofExtractionSpec, curve_map: OptimizableDofMapSpec,
                        source_index: int | None) -> CoilDofExtractionSpec:
    return make_coil_dof_extraction_spec(
        curve=coil.curve, curve_map=curve_map, current_map=coil.current_map, current_term_maps=coil.current_term_maps,
        current_term_scales=coil.current_term_scales, curve_source_index=source_index,
        rotmat=coil.symmetry.rotmat if coil.symmetry.has_rotation else None, scale=coil.symmetry.scale)


def _mixed_problem() -> tuple[CoilSetDofExtractionSpec, jax.Array]:
    """Include fixed coordinates, shared sources, distinct grids and affine currents."""
    grid = jnp.asarray(np.linspace(0.0, 1.0, 8, endpoint=False))
    first = make_curve_xyzfourier_spec(dofs=jnp.linspace(0.1, 0.9, 9), quadpoints=grid, order=1)
    second = make_curve_xyzfourier_spec(dofs=first.dofs, quadpoints=(grid + 0.037) ** 1.1, order=first.order)
    third = make_curve_xyzfourier_spec(dofs=jnp.linspace(0.1, 0.7, 15), quadpoints=grid, order=2)
    rz = make_curve_rzfourier_spec(dofs=jnp.asarray((1.0, 0.2, 0.3)), quadpoints=grid, order=1, nfp=2, stellsym=True)
    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    fixed_current = _dof_map(jnp.asarray((7.0,)))
    free_current = _dof_map(jnp.asarray((0.0,)), ((32, 33, 0, 1),))
    first_coil = make_coil_dof_extraction_spec(curve=first, curve_map=_dof_map(first.dofs, ((0, 8, 1, 9),)),
                                               current_map=fixed_current, curve_source_index=0)
    coils = (
        first_coil,
        make_coil_dof_extraction_spec(curve=second, curve_map=_dof_map(second.dofs, ((8, 17, 0, 9),)),
                                      current_map=fixed_current,
                                      current_term_maps=(free_current, _dof_map(jnp.asarray((3.0,)))),
                                      current_term_scales=(2.0, -0.5), curve_source_index=1),
        make_coil_dof_extraction_spec(curve=third, curve_map=_dof_map(third.dofs, ((17, 32, 0, 15),)),
                                      current_map=free_current, rotmat=rotation, scale=-0.6, curve_source_index=2),
        make_coil_dof_extraction_spec(curve=rz, curve_map=_dof_map(rz.dofs), current_map=fixed_current),
        # A shared source must use its first mapping, even if a later replica's
        # payload has a different template-only map.
        make_coil_dof_extraction_spec(
            curve=first_coil.curve, curve_map=_dof_map(first.dofs), current_map=first_coil.current_map,
            current_term_maps=first_coil.current_term_maps, current_term_scales=first_coil.current_term_scales,
            curve_source_index=first_coil.curve_source_index, rotmat=rotation, scale=-0.6),
    )
    return make_coil_set_dof_extraction_spec(coils), jnp.linspace(0.1, 0.8, 33)


def _scalar_geometry(extraction: CoilSetDofExtractionSpec,
                     parameters: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Retain the pre-batching per-coil evaluator as a numerical/performance oracle."""
    geometry = []
    geometry_by_curve: dict[int, tuple[jax.Array, ...]] = {}
    parameter_zero = placement_zero(parameters)
    for coil in coil_specs_from_dof_extraction_spec(extraction, parameters):
        curve_id = id(coil.curve)
        if curve_id not in geometry_by_curve:
            geometry_by_curve[curve_id] = curve_geometry_from_dofs(coil.curve, coil.curve.dofs)
        gamma, dash, dashdash = geometry_by_curve[curve_id]
        gamma, dash, current = apply_coil_symmetry(gamma, dash, coil.current.value[0], coil.symmetry)
        if coil.symmetry.has_rotation:
            dashdash = dashdash @ coil.symmetry.rotmat
        geometry.append((gamma, dash, dashdash, current + parameter_zero))
    gammas, dashs, dashdashs, currents = zip(*geometry, strict=True)
    return (jnp.stack(gammas), jnp.stack(dashs), jnp.stack(dashdashs), jnp.stack(currents))


def _ordinary_stage_two_geometry(extraction: CoilSetDofExtractionSpec,
                                 parameters: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project the fused ordinary-coil route onto ``stage_two_coil_geometry``'s outputs."""
    geometry = stage_two_geometry(extraction, parameters)
    return (geometry.gamma, geometry.gammadash, geometry.centerline_gammadashdash, geometry.currents)


if JAX_IMPORT_ERROR is None:
    # Both routes to the batched coil geometry, by the name each subTest reports.
    _GEOMETRY_FUNCTIONS = {function.__name__: function
                           for function in (stage_two_coil_geometry, _ordinary_stage_two_geometry)}
    _GEOMETRIES = [dict(geometry=name) for name in _GEOMETRY_FUNCTIONS]


def _geometry_energy(geometry: tuple[jax.Array, ...]) -> jax.Array:
    # Nonlinear geometry and current terms exercise both first and second derivatives.
    return sum((jnp.mean(jnp.sin(value) ** 2) for value in geometry), start=placement_zero(geometry[0]))


class StageTwoGeometryBatchingTests(JaxTestCase):
    """Batched Stage-II Fourier geometry against per-curve evaluation and native curves."""

    def test_mixed_geometry_matches_scalar_with_distinct_grids_and_symmetries(self) -> None:
        """Batched geometry matches per-curve evaluation for curves with different grids and symmetries."""
        extraction, parameters = _mixed_problem()
        expected = _scalar_geometry(extraction, parameters)
        geometry = stage_two_geometry(extraction, parameters)
        np.testing.assert_allclose(geometry.centerline_gamma, expected[0], rtol=3e-13, atol=3e-12)
        np.testing.assert_allclose(geometry.centerline_gammadash, expected[1], rtol=3e-13, atol=3e-12)
        self.run_cases(self._case_mixed_geometry_matches_scalar_with_distinct_grids_and_symmetries, _GEOMETRIES)

    def _case_mixed_geometry_matches_scalar_with_distinct_grids_and_symmetries(self, geometry: str) -> None:
        extraction, parameters = _mixed_problem()
        actual = jit(_GEOMETRY_FUNCTIONS[geometry])(extraction, parameters)
        expected = _scalar_geometry(extraction, parameters)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference, rtol=3e-13, atol=3e-12)
        np.testing.assert_allclose(actual[3], (7.0, 0.1, -0.48, 7.0, -4.2))
        self.assertFalse(np.allclose(actual[0][0], actual[0][1]))

    def test_mixed_geometry_gradients_and_hessian_vector_match_scalar(self) -> None:
        """Gradients and Hessian-vector products of batched geometry match per-curve evaluation."""
        self.run_cases(self._case_mixed_geometry_gradients_and_hessian_vector_match_scalar, _GEOMETRIES)

    def _case_mixed_geometry_gradients_and_hessian_vector_match_scalar(self, geometry: str) -> None:
        extraction, parameters = _mixed_problem()

        def batched(values: jax.Array) -> jax.Array:
            return _geometry_energy(_GEOMETRY_FUNCTIONS[geometry](extraction, values))

        def scalar(values: jax.Array) -> jax.Array:
            return _geometry_energy(_scalar_geometry(extraction, values))

        direction = jnp.cos(parameters)
        actual_gradient, actual_hvp = jit(lambda values, tangent: jax.jvp(jax.grad(batched), (values,), (tangent,)))(
            parameters, direction)
        expected_gradient, expected_hvp = jax.jvp(jax.grad(scalar), (parameters,), (direction,))
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-11, atol=2e-10)
        np.testing.assert_allclose(actual_hvp, expected_hvp, rtol=2e-10, atol=2e-9)

    def test_absent_source_indices_preserve_separate_curve_mappings(self) -> None:
        """Coils without a shared source curve keep their own DOF mappings."""
        self.run_cases(self._case_absent_source_indices_preserve_separate_curve_mappings, _GEOMETRIES)

    def _case_absent_source_indices_preserve_separate_curve_mappings(self, geometry: str) -> None:
        extraction, parameters = _mixed_problem()
        first = _with_curve_mapping(extraction.coils[0], extraction.coils[0].curve_map, None)
        second = _with_curve_mapping(first, _dof_map(first.curve.dofs), None)
        independent = make_coil_set_dof_extraction_spec((first, second))
        actual = _GEOMETRY_FUNCTIONS[geometry](independent, parameters)
        expected = _scalar_geometry(independent, parameters)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference, rtol=3e-13, atol=3e-12)
        self.assertFalse(np.allclose(actual[0][0], actual[0][1]))

    def test_current_derivatives_do_not_couple_nonfinite_geometry(self) -> None:
        """Infinite geometry tangents do not reach currents; infinite current cotangents do not reach geometry DOFs."""
        self.run_cases(self._case_current_derivatives_do_not_couple_nonfinite_geometry, _GEOMETRIES)

    def _case_current_derivatives_do_not_couple_nonfinite_geometry(self, geometry: str) -> None:
        extraction, parameters = _mixed_problem()

        def currents(values: jax.Array) -> jax.Array:
            return _GEOMETRY_FUNCTIONS[geometry](extraction, values)[3]

        tangent = jnp.zeros_like(parameters).at[0].set(jnp.inf)
        _, current_tangent = jax.jvp(currents, (parameters,), (tangent,))
        np.testing.assert_array_equal(current_tangent, 0.0)
        current_values, pullback = jax.vjp(currents, parameters)
        (cotangent,) = pullback(jnp.full_like(current_values, jnp.inf))
        np.testing.assert_array_equal(cotangent[:32], 0.0)

    def test_geometry_jvp_preserves_explicit_transfer_placement(self) -> None:
        """The compiled geometry JVP runs with host-to-device transfers disallowed."""
        self.run_cases(self._case_geometry_jvp_preserves_explicit_transfer_placement, _GEOMETRIES)

    def _case_geometry_jvp_preserves_explicit_transfer_placement(self, geometry: str) -> None:
        extraction, parameters = _mixed_problem()
        tangent = jnp.ones_like(parameters)
        geometry_function = _GEOMETRY_FUNCTIONS[geometry]

        @jit
        def evaluate(spec: CoilSetDofExtractionSpec, values: jax.Array, direction: jax.Array) -> tuple[
                tuple[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            return jax.jvp(lambda current_values: geometry_function(spec, current_values), (values,), (direction,))

        # The compiled workflow takes every spec leaf explicitly. No warmup or
        # captured host constants may hide transfers on the first guarded call.
        with jax.transfer_guard("disallow"):
            jax.block_until_ready(evaluate(extraction, parameters, tangent))

    def test_homogeneous_geometry_batches_fourier_dot_graph(self) -> None:
        """Identical curve layouts share one batched Fourier contraction instead of one per curve."""
        self.run_cases(self._case_homogeneous_geometry_batches_fourier_dot_graph, _GEOMETRIES)

    def _case_homogeneous_geometry_batches_fourier_dot_graph(self, geometry: str) -> None:
        extraction, parameters = _mixed_problem()
        first = extraction.coils[0]
        homogeneous = make_coil_set_dof_extraction_spec(tuple(_with_curve_mapping(first, first.curve_map, index)
                                                              for index in range(4)))
        scalar = jax.make_jaxpr(_scalar_geometry)(homogeneous, parameters)
        batched = jax.make_jaxpr(_GEOMETRY_FUNCTIONS[geometry])(homogeneous, parameters)
        # Owner-map matvecs remain per curve. Only Fourier contractions have two
        # matrix operands; their batch metadata must identify the new mapped axis.
        scalar_dots, batched_dots = (
            tuple(equation for equation in graph.jaxpr.eqns if equation.primitive.name == "dot_general"
                  and all(len(operand.aval.shape) >= 2  # pyright: ignore[reportAttributeAccessIssue]
                          for operand in equation.invars))
            for graph in (scalar, batched)
        )
        self.assertGreater(len(scalar_dots), 0)
        self.assertEqual(len(batched_dots) * 4, len(scalar_dots))
        self.assertTrue(all(equation.params["dimension_numbers"][1] == ((0,), (0,)) for equation in batched_dots))

    def test_compiled_current_derivatives_match_native_under_transfer_guard(self) -> None:
        """Compiled coil currents and their JVP and VJP match native currents with transfers disallowed."""
        # Match the plain-current native fixture of the source eager AD oracle;
        # affine term assembly is a separate preexisting placement boundary.
        self.run_cases(self._case_compiled_current_derivatives_match_native_under_transfer_guard, _GEOMETRIES)

    def _case_compiled_current_derivatives_match_native_under_transfer_guard(self, geometry: str) -> None:
        curves = create_equally_spaced_curves(2, 2, True, R0=1.0, R1=0.25, order=1, numquadpoints=8)
        native_currents = [Current(1.0e5), Current(1.1e5)]
        native_currents[0].fix_all()
        coils = coils_via_symmetries(curves, native_currents, 2, True)
        field = JaxBiotSavart(coils)
        extraction = field.coil_dof_extraction_spec()
        parameters = jnp.array(field.x)
        tangent = jnp.ones_like(parameters)
        weights = np.linspace(0.3, 1.1, len(coils))
        current_cotangent = jnp.array(weights)
        native_values = np.asarray([coil.current.get_value() for coil in coils])
        native_jacobian = np.stack([coil.current.vjp(np.ones(1))(field) for coil in coils])
        geometry_function = _GEOMETRY_FUNCTIONS[geometry]

        @jit
        def evaluate(spec: CoilSetDofExtractionSpec, values: jax.Array, direction: jax.Array,
                     weights: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            def currents(current_values: jax.Array) -> jax.Array:
                return geometry_function(spec, current_values)[3]

            current_values, pushforward = jax.linearize(currents, values)
            current_tangent = pushforward(direction)
            _, pullback = jax.vjp(currents, values)
            return current_values, current_tangent, pullback(weights)[0]

        with jax.transfer_guard("disallow"):
            current_values, current_tangent, parameter_cotangent = evaluate(extraction, parameters, tangent,
                                                                            current_cotangent)
            jax.block_until_ready((current_values, current_tangent, parameter_cotangent))
        np.testing.assert_allclose(current_values, native_values)
        np.testing.assert_allclose(current_tangent, native_jacobian @ np.ones(parameters.shape))
        np.testing.assert_allclose(parameter_cotangent, weights @ native_jacobian)

    def test_ordinary_geometry_eager_derivatives_match_native(self) -> None:
        """Curve gamma, gammadash, gammadashdash and their JVPs and VJPs match native, with and without JIT."""
        curves = create_equally_spaced_curves(2, 2, True, R0=1.0, R1=0.25, order=1, numquadpoints=8)
        templates = tuple(CurveXYZFourierSpec(dofs=jnp.array(curve.get_dofs()), quadpoints=jnp.array(curve.quadpoints),
                                              order=1) for curve in curves)
        coefficients = tuple(spec.dofs for spec in templates)
        directions = tuple(np.linspace(0.1 + index, 0.9 + index, spec.dofs.size)
                           for index, spec in enumerate(templates))
        tangents = tuple(jnp.array(direction) for direction in directions)
        weights = tuple(tuple(np.cos(np.arange(24).reshape((8, 3)) + curve_index + term_index)
                              for term_index in range(3)) for curve_index in range(len(curves)))
        cotangents = jax.tree.map(jnp.array, weights)

        def geometry(values: tuple[jax.Array, ...]) -> tuple[tuple[jax.Array, ...], ...]:
            specs = tuple(CurveXYZFourierSpec(dofs=dofs, quadpoints=template.quadpoints, order=1)
                          for dofs, template in zip(values, templates, strict=True))
            cache = _ordinary_curve_geometry(specs, specs)
            return tuple(cache[id(spec)] for spec in specs)

        native_values = tuple((curve.gamma(), curve.gammadash(), curve.gammadashdash()) for curve in curves)
        native_tangents = tuple(
            tuple(np.einsum("qcd,d->qc", jacobian, direction) for jacobian in
                  (curve.dgamma_by_dcoeff(), curve.dgammadash_by_dcoeff(), curve.dgammadashdash_by_dcoeff()))
            for curve, direction in zip(curves, directions, strict=True))
        native_cotangents = tuple((curve.dgamma_by_dcoeff_vjp(weight[0]) + curve.dgammadash_by_dcoeff_vjp(weight[1])
                                   + curve.dgammadashdash_by_dcoeff_vjp(weight[2]))(curve)
                                  for curve, weight in zip(curves, weights, strict=True))
        # The ordinary evaluator receives reconstructed full coefficients.
        # Extraction-map slice transposes are a separate inherited boundary.
        for disable_jit in (False, True):
            with self.subTest(disable_jit=disable_jit):
                with jax.disable_jit(disable_jit), jax.transfer_guard("disallow"):
                    values, tangent_values = jax.jvp(geometry, (coefficients,), (tangents,))
                    _, pullback = jax.vjp(geometry, coefficients)
                    (coefficient_cotangents,) = pullback(cotangents)
                    jax.block_until_ready((values, tangent_values, coefficient_cotangents))
                actual = jax.tree.leaves((values, tangent_values, coefficient_cotangents))
                expected = jax.tree.leaves((native_values, native_tangents, native_cotangents))
                for value, reference in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(value, reference, rtol=3e-13, atol=3e-12)
