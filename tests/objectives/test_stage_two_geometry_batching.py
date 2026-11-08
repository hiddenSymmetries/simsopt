"""Parity and trace-size oracles for Stage-II Fourier geometry batching."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import jax
    import jax.numpy as jnp
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core._device_scalars import placement_zero
    from simsopt_jax.core.curve_geometry import curve_geometry_from_dofs
    from simsopt_jax.core.field import coil_specs_from_dof_extraction_spec
    from simsopt_jax.core.specs import (
        CoilDofExtractionSpec,
        CoilSetDofExtractionSpec,
        CurveXYZFourierSpec,
        OptimizableDofMapSpec,
        apply_coil_symmetry,
        make_coil_dof_extraction_spec,
        make_coil_set_dof_extraction_spec,
        make_curve_rzfourier_spec,
        make_curve_xyzfourier_spec,
        make_optimizable_dof_map_spec,
    )
    from simsopt_jax.objectives.stage_two import (
        _ordinary_curve_geometry,
        stage_two_coil_geometry,
        stage_two_geometry,
    )
    from simsopt_jax_adapters.field import JaxBiotSavart


# typing.Callable and typing.Tuple keep this runtime alias importable on Python 3.8,
# where unittest discovery must reach the JAX gate and skip.
GeometryFunction = Callable[
    ["CoilSetDofExtractionSpec", "jax.Array"],
    Tuple["jax.Array", "jax.Array", "jax.Array", "jax.Array"],
]


def _dof_map(
    template: jax.Array,
    segments: tuple[tuple[int, int, int, int], ...] = (),
) -> OptimizableDofMapSpec:
    return make_optimizable_dof_map_spec(
        template_full_dofs=template,
        owner_segments=segments,
        input_mode="full",
        input_start=0,
        input_end=template.size,
    )


def _with_curve_mapping(
    coil: CoilDofExtractionSpec,
    curve_map: OptimizableDofMapSpec,
    source_index: int | None,
) -> CoilDofExtractionSpec:
    return make_coil_dof_extraction_spec(
        curve=coil.curve,
        curve_map=curve_map,
        current_map=coil.current_map,
        current_term_maps=coil.current_term_maps,
        current_term_scales=coil.current_term_scales,
        curve_source_index=source_index,
        rotmat=coil.symmetry.rotmat if coil.symmetry.has_rotation else None,
        scale=coil.symmetry.scale,
    )


def _mixed_problem() -> tuple[CoilSetDofExtractionSpec, jax.Array]:
    """Include fixed coordinates, shared sources, distinct grids and affine currents."""
    grid = jnp.asarray(np.linspace(0.0, 1.0, 8, endpoint=False))
    first = make_curve_xyzfourier_spec(
        dofs=jnp.linspace(0.1, 0.9, 9), quadpoints=grid, order=1
    )
    second = make_curve_xyzfourier_spec(
        dofs=first.dofs, quadpoints=(grid + 0.037) ** 1.1, order=first.order
    )
    third = make_curve_xyzfourier_spec(
        dofs=jnp.linspace(0.1, 0.7, 15), quadpoints=grid, order=2
    )
    rz = make_curve_rzfourier_spec(
        dofs=jnp.asarray((1.0, 0.2, 0.3)),
        quadpoints=grid,
        order=1,
        nfp=2,
        stellsym=True,
    )
    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    fixed_current = _dof_map(jnp.asarray((7.0,)))
    free_current = _dof_map(jnp.asarray((0.0,)), ((32, 33, 0, 1),))
    first_coil = make_coil_dof_extraction_spec(
        curve=first,
        curve_map=_dof_map(first.dofs, ((0, 8, 1, 9),)),
        current_map=fixed_current,
        curve_source_index=0,
    )
    coils = (
        first_coil,
        make_coil_dof_extraction_spec(
            curve=second,
            curve_map=_dof_map(second.dofs, ((8, 17, 0, 9),)),
            current_map=fixed_current,
            current_term_maps=(free_current, _dof_map(jnp.asarray((3.0,)))),
            current_term_scales=(2.0, -0.5),
            curve_source_index=1,
        ),
        make_coil_dof_extraction_spec(
            curve=third,
            curve_map=_dof_map(third.dofs, ((17, 32, 0, 15),)),
            current_map=free_current,
            rotmat=rotation,
            scale=-0.6,
            curve_source_index=2,
        ),
        make_coil_dof_extraction_spec(
            curve=rz, curve_map=_dof_map(rz.dofs), current_map=fixed_current
        ),
        # A shared source must use its first mapping, even if a later replica's
        # payload has a different template-only map.
        make_coil_dof_extraction_spec(
            curve=first_coil.curve,
            curve_map=_dof_map(first.dofs),
            current_map=first_coil.current_map,
            current_term_maps=first_coil.current_term_maps,
            current_term_scales=first_coil.current_term_scales,
            curve_source_index=first_coil.curve_source_index,
            rotmat=rotation,
            scale=-0.6,
        ),
    )
    return make_coil_set_dof_extraction_spec(coils), jnp.linspace(0.1, 0.8, 33)


def _scalar_geometry(
    extraction: CoilSetDofExtractionSpec, parameters: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Retain the pre-batching per-coil evaluator as a numerical/performance oracle."""
    geometry = []
    geometry_by_curve: dict[int, tuple[jax.Array, ...]] = {}
    parameter_zero = placement_zero(parameters)
    for coil in coil_specs_from_dof_extraction_spec(extraction, parameters):
        curve_id = id(coil.curve)
        if curve_id not in geometry_by_curve:
            geometry_by_curve[curve_id] = curve_geometry_from_dofs(
                coil.curve, coil.curve.dofs
            )
        gamma, dash, dashdash = geometry_by_curve[curve_id]
        gamma, dash, current = apply_coil_symmetry(
            gamma, dash, coil.current.value[0], coil.symmetry
        )
        if coil.symmetry.has_rotation:
            dashdash = dashdash @ coil.symmetry.rotmat
        geometry.append((gamma, dash, dashdash, current + parameter_zero))
    gammas, dashs, dashdashs, currents = zip(*geometry, strict=True)
    return (
        jnp.stack(gammas),
        jnp.stack(dashs),
        jnp.stack(dashdashs),
        jnp.stack(currents),
    )


def _ordinary_stage_two_geometry(
    extraction: CoilSetDofExtractionSpec, parameters: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project the fused ordinary-coil route onto ``stage_two_coil_geometry``'s outputs."""
    geometry = stage_two_geometry(extraction, parameters)
    return (
        geometry.gamma,
        geometry.gammadash,
        geometry.centerline_gammadashdash,
        geometry.currents,
    )


if JAX_IMPORT_ERROR is None:
    _GEOMETRY_FUNCTIONS: tuple[GeometryFunction, ...] = (
        stage_two_coil_geometry,
        _ordinary_stage_two_geometry,
    )


def _geometry_energy(geometry: tuple[jax.Array, ...]) -> jax.Array:
    # Nonlinear geometry and current terms exercise both first and second derivatives.
    return sum(
        (jnp.mean(jnp.sin(value) ** 2) for value in geometry),
        start=placement_zero(geometry[0]),
    )


class StageTwoGeometryBatchingTests(JaxTestCase):
    def test_mixed_geometry_matches_scalar_with_distinct_grids_and_symmetries(
        self,
    ) -> None:
        """Batched geometry matches per-curve evaluation for curves with different grids and symmetries."""
        extraction, parameters = _mixed_problem()
        expected = _scalar_geometry(extraction, parameters)
        geometry = stage_two_geometry(extraction, parameters)
        np.testing.assert_allclose(
            geometry.centerline_gamma, expected[0], rtol=3e-13, atol=3e-12
        )
        np.testing.assert_allclose(
            geometry.centerline_gammadash, expected[1], rtol=3e-13, atol=3e-12
        )
        for geometry_function in _GEOMETRY_FUNCTIONS:
            with self.subTest(geometry=geometry_function.__name__), self.case():
                extraction, parameters = _mixed_problem()
                actual = jit(geometry_function)(extraction, parameters)
                expected = _scalar_geometry(extraction, parameters)
                for value, reference in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(value, reference, rtol=3e-13, atol=3e-12)
                np.testing.assert_allclose(actual[3], (7.0, 0.1, -0.48, 7.0, -4.2))
                self.assertFalse(np.allclose(actual[0][0], actual[0][1]))

    def test_mixed_geometry_gradients_and_hessian_vector_match_scalar(self) -> None:
        """Gradients and Hessian-vector products of batched geometry match per-curve evaluation."""
        for geometry_function in _GEOMETRY_FUNCTIONS:
            with self.subTest(geometry=geometry_function.__name__), self.case():
                self._case_mixed_geometry_gradients_and_hessian_vector_match_scalar(
                    geometry_function
                )

    def _case_mixed_geometry_gradients_and_hessian_vector_match_scalar(
        self, geometry_function: GeometryFunction
    ) -> None:
        extraction, parameters = _mixed_problem()

        def batched(values: jax.Array) -> jax.Array:
            return _geometry_energy(geometry_function(extraction, values))

        def scalar(values: jax.Array) -> jax.Array:
            return _geometry_energy(_scalar_geometry(extraction, values))

        direction = jnp.cos(parameters)
        actual_gradient, actual_hvp = jit(
            lambda values, tangent: jax.jvp(jax.grad(batched), (values,), (tangent,))
        )(parameters, direction)
        expected_gradient, expected_hvp = jax.jvp(
            jax.grad(scalar), (parameters,), (direction,)
        )
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=2e-11, atol=2e-10
        )
        np.testing.assert_allclose(actual_hvp, expected_hvp, rtol=2e-10, atol=2e-9)

    def test_absent_source_indices_preserve_separate_curve_mappings(self) -> None:
        """Coils without a shared source curve keep their own DOF mappings."""
        for geometry_function in _GEOMETRY_FUNCTIONS:
            with self.subTest(geometry=geometry_function.__name__), self.case():
                extraction, parameters = _mixed_problem()
                first = _with_curve_mapping(
                    extraction.coils[0], extraction.coils[0].curve_map, None
                )
                second = _with_curve_mapping(first, _dof_map(first.curve.dofs), None)
                independent = make_coil_set_dof_extraction_spec((first, second))
                actual = geometry_function(independent, parameters)
                expected = _scalar_geometry(independent, parameters)
                for value, reference in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(value, reference, rtol=3e-13, atol=3e-12)
                self.assertFalse(np.allclose(actual[0][0], actual[0][1]))

    def test_current_derivatives_do_not_couple_nonfinite_geometry(self) -> None:
        """Infinite geometry tangents do not reach currents, and infinite current cotangents do not reach geometry DOFs."""
        for geometry_function in _GEOMETRY_FUNCTIONS:
            with self.subTest(geometry=geometry_function.__name__), self.case():
                self._case_current_derivatives_do_not_couple_nonfinite_geometry(
                    geometry_function
                )

    def _case_current_derivatives_do_not_couple_nonfinite_geometry(
        self, geometry_function: GeometryFunction
    ) -> None:
        extraction, parameters = _mixed_problem()

        def currents(values: jax.Array) -> jax.Array:
            return geometry_function(extraction, values)[3]

        tangent = jnp.zeros_like(parameters).at[0].set(jnp.inf)
        _, current_tangent = jax.jvp(currents, (parameters,), (tangent,))
        np.testing.assert_array_equal(current_tangent, 0.0)
        current_values, pullback = jax.vjp(currents, parameters)
        (cotangent,) = pullback(jnp.full_like(current_values, jnp.inf))
        np.testing.assert_array_equal(cotangent[:32], 0.0)

    def test_geometry_jvp_preserves_explicit_transfer_placement(self) -> None:
        """The compiled geometry JVP runs with host-to-device transfers disallowed."""
        for geometry_function in _GEOMETRY_FUNCTIONS:
            with self.subTest(geometry=geometry_function.__name__), self.case():
                self._case_geometry_jvp_preserves_explicit_transfer_placement(
                    geometry_function
                )

    def _case_geometry_jvp_preserves_explicit_transfer_placement(
        self, geometry_function: GeometryFunction
    ) -> None:
        extraction, parameters = _mixed_problem()
        tangent = jnp.ones_like(parameters)

        @jit
        def evaluate(
            spec: CoilSetDofExtractionSpec,
            values: jax.Array,
            direction: jax.Array,
        ) -> tuple[
            tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            return jax.jvp(
                lambda current_values: geometry_function(spec, current_values),
                (values,),
                (direction,),
            )

        # The compiled workflow takes every spec leaf explicitly. No warmup or
        # captured host constants may hide transfers on the first guarded call.
        with jax.transfer_guard("disallow"):
            jax.block_until_ready(evaluate(extraction, parameters, tangent))

    def test_homogeneous_geometry_batches_fourier_dot_graph(self) -> None:
        """Identical curve layouts share one batched Fourier contraction instead of one per curve."""
        for geometry_function in _GEOMETRY_FUNCTIONS:
            with self.subTest(geometry=geometry_function.__name__), self.case():
                extraction, parameters = _mixed_problem()
                first = extraction.coils[0]
                homogeneous = make_coil_set_dof_extraction_spec(
                    tuple(
                        _with_curve_mapping(first, first.curve_map, index)
                        for index in range(4)
                    )
                )
                scalar = jax.make_jaxpr(_scalar_geometry)(homogeneous, parameters)
                batched = jax.make_jaxpr(geometry_function)(homogeneous, parameters)
                # Owner-map matvecs remain per curve. Only Fourier contractions have two
                # matrix operands; their batch metadata must identify the new mapped axis.
                scalar_dots, batched_dots = (
                    tuple(
                        equation
                        for equation in graph.jaxpr.eqns
                        if equation.primitive.name == "dot_general"
                        and all(
                            len(operand.aval.shape) >= 2 for operand in equation.invars
                        )
                    )
                    for graph in (scalar, batched)
                )
                self.assertGreater(len(scalar_dots), 0)
                self.assertEqual(len(batched_dots) * 4, len(scalar_dots))
                self.assertTrue(
                    all(
                        equation.params["dimension_numbers"][1] == ((0,), (0,))
                        for equation in batched_dots
                    )
                )

    def test_compiled_current_derivatives_match_native_under_transfer_guard(
        self,
    ) -> None:
        """Compiled coil currents and their JVP and VJP match native currents with transfers disallowed."""
        for geometry_function in _GEOMETRY_FUNCTIONS:
            with self.subTest(geometry=geometry_function.__name__), self.case():
                # Match the plain-current native fixture of the source eager AD oracle;
                # affine term assembly is a separate preexisting placement boundary.
                self._case_compiled_current_derivatives_match_native_under_transfer_guard(
                    geometry_function
                )

    def _case_compiled_current_derivatives_match_native_under_transfer_guard(
        self, geometry_function: GeometryFunction
    ) -> None:
        curves = create_equally_spaced_curves(
            2, 2, True, R0=1.0, R1=0.25, order=1, numquadpoints=8
        )
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
        native_jacobian = np.stack(
            [coil.current.vjp(np.ones(1))(field) for coil in coils]
        )

        @jit
        def evaluate(
            spec: CoilSetDofExtractionSpec,
            values: jax.Array,
            direction: jax.Array,
            weights: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            def currents(current_values: jax.Array) -> jax.Array:
                return geometry_function(spec, current_values)[3]

            current_values, pushforward = jax.linearize(currents, values)
            current_tangent = pushforward(direction)
            _, pullback = jax.vjp(currents, values)
            return current_values, current_tangent, pullback(weights)[0]

        with jax.transfer_guard("disallow"):
            current_values, current_tangent, parameter_cotangent = evaluate(
                extraction, parameters, tangent, current_cotangent
            )
            jax.block_until_ready(
                (current_values, current_tangent, parameter_cotangent)
            )
        np.testing.assert_allclose(current_values, native_values)
        np.testing.assert_allclose(
            current_tangent, native_jacobian @ np.ones(parameters.shape)
        )
        np.testing.assert_allclose(parameter_cotangent, weights @ native_jacobian)

    def test_ordinary_geometry_eager_derivatives_match_native(self) -> None:
        """Curve gamma, gammadash, gammadashdash and their JVPs and VJPs match native, with and without JIT."""
        curves = create_equally_spaced_curves(
            2, 2, True, R0=1.0, R1=0.25, order=1, numquadpoints=8
        )
        templates = tuple(
            CurveXYZFourierSpec(
                dofs=jnp.array(curve.get_dofs()),
                quadpoints=jnp.array(curve.quadpoints),
                order=1,
            )
            for curve in curves
        )
        coefficients = tuple(spec.dofs for spec in templates)
        directions = tuple(
            np.linspace(0.1 + index, 0.9 + index, spec.dofs.size)
            for index, spec in enumerate(templates)
        )
        tangents = tuple(jnp.array(direction) for direction in directions)
        weights = tuple(
            tuple(
                np.cos(np.arange(24).reshape((8, 3)) + curve_index + term_index)
                for term_index in range(3)
            )
            for curve_index in range(len(curves))
        )
        cotangents = jax.tree.map(jnp.array, weights)

        def geometry(
            values: tuple[jax.Array, ...],
        ) -> tuple[tuple[jax.Array, ...], ...]:
            specs = tuple(
                CurveXYZFourierSpec(dofs=dofs, quadpoints=template.quadpoints, order=1)
                for dofs, template in zip(values, templates, strict=True)
            )
            cache = _ordinary_curve_geometry(specs, specs)
            return tuple(cache[id(spec)] for spec in specs)

        native_values = tuple(
            (curve.gamma(), curve.gammadash(), curve.gammadashdash())
            for curve in curves
        )
        native_tangents = tuple(
            tuple(
                np.einsum("qcd,d->qc", jacobian, direction)
                for jacobian in (
                    curve.dgamma_by_dcoeff(),
                    curve.dgammadash_by_dcoeff(),
                    curve.dgammadashdash_by_dcoeff(),
                )
            )
            for curve, direction in zip(curves, directions, strict=True)
        )
        native_cotangents = tuple(
            (
                curve.dgamma_by_dcoeff_vjp(weight[0])
                + curve.dgammadash_by_dcoeff_vjp(weight[1])
                + curve.dgammadashdash_by_dcoeff_vjp(weight[2])
            )(curve)
            for curve, weight in zip(curves, weights, strict=True)
        )
        # The ordinary evaluator receives reconstructed full coefficients.
        # Extraction-map slice transposes are a separate inherited boundary.
        for disable_jit in (False, True):
            with self.subTest(disable_jit=disable_jit):
                with jax.disable_jit(disable_jit), jax.transfer_guard("disallow"):
                    values, tangent_values = jax.jvp(
                        geometry, (coefficients,), (tangents,)
                    )
                    _, pullback = jax.vjp(geometry, coefficients)
                    (coefficient_cotangents,) = pullback(cotangents)
                    jax.block_until_ready(
                        (values, tangent_values, coefficient_cotangents)
                    )
                actual = jax.tree.leaves(
                    (values, tangent_values, coefficient_cotangents)
                )
                expected = jax.tree.leaves(
                    (native_values, native_tangents, native_cotangents)
                )
                for value, reference in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(value, reference, rtol=3e-13, atol=3e-12)
