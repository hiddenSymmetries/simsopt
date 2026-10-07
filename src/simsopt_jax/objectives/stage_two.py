"""Fused pure-JAX objective for filamentary Stage-II coil optimization.

One program maps the free coil DOFs of a :class:`BiotSavartJAX` field to coil
geometry once, then evaluates the squared flux on a fixed surface plus the
coil-geometry penalties of the native Stage-II examples: total base-curve
length, curve-curve and curve-surface distance, Lp curvature and mean squared
curvature. Every term uses the formula of its native objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.dtypes import runtime_device_put_tree
from simsopt_jax.core._device_scalars import placement_zero
from simsopt_jax.core.biotsavart import biot_savart_B
from simsopt_jax.core.curve_geometry import curve_geometry_from_dofs
from simsopt_jax.core.curve_kernels import (
    curvature_p_norm_from_kappa_pure,
    curve_curve_distance_penalty_pure,
    curve_length_from_incremental_arclength_pure,
    curve_surface_distance_penalty_pure,
    distance_candidate_pure,
    kappa_pure,
    mean_squared_curvature_pure,
)
from simsopt_jax.core.field import coil_specs_from_dof_extraction_spec
from simsopt_jax.core.integral_bdotn import fixed_surface_flux_integral_from_B
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    FixedSurfaceFluxSpec,
    apply_coil_symmetry,
)
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.runtime.host_boundary import host_value

__all__ = [
    "CoilDofExtractionProvider",
    "StageTwoObjectiveConfig",
    "StageTwoProblem",
    "fused_stage_two_objective",
    "fused_stage_two_values",
    "make_stage_two_problem",
    "prepare_stage_two_config",
    "stage_two_coil_geometry",
    "stage_two_geometric_penalty",
]

# LpCurveCurvature exponent of the native Stage-II examples.
_CURVATURE_P = 2.0


class CoilDofExtractionProvider(Protocol):
    """Structural contract needed to compose a Stage-II objective."""

    def coil_dof_extraction_spec(self) -> CoilSetDofExtractionSpec: ...


@dataclass(frozen=True, slots=True)
class StageTwoObjectiveConfig:
    """Immutable weights and thresholds of the Stage-II penalties.

    The penalties mirror the native objectives: ``length_weight`` times the
    total length of the first ``num_base_curves`` coils (or
    ``QuadraticPenalty(total length, length_target, length_target_mode)``),
    ``CurveCurveDistance`` over all coils with ``num_basecurves``,
    ``CurveSurfaceDistance`` over all coils, ``LpCurveCurvature(p=2)`` and
    ``QuadraticPenalty(MeanSquaredCurvature, threshold, mode)`` per base curve.
    A term is part of the objective when its weight is not ``None``. Weights,
    zero included, are traced operands: changing them never recompiles, while
    adding or removing a term (or ``length_target``) changes the program.
    """

    num_base_curves: int
    length_weight: float | None = None
    length_target: float | None = None
    length_target_mode: Literal["max", "identity"] = "max"
    curve_curve_minimum_distance: float = 0.1
    curve_curve_weight: float | None = None
    curve_surface_minimum_distance: float = 0.3
    curve_surface_weight: float | None = None
    curvature_threshold: float = 5.0
    curvature_weight: float | None = None
    mean_squared_curvature_threshold: float = 5.0
    mean_squared_curvature_target_mode: Literal["max", "identity"] = "max"
    mean_squared_curvature_weight: float | None = None


# Weights and the length target may be None (term or target absent).
_OPTIONAL_FIELDS = (
    "length_weight",
    "length_target",
    "curve_curve_weight",
    "curve_surface_weight",
    "curvature_weight",
    "mean_squared_curvature_weight",
)
_STAGE_TWO_NUMERIC_FIELDS = (
    "length_weight",
    "length_target",
    "curve_curve_minimum_distance",
    "curve_curve_weight",
    "curve_surface_minimum_distance",
    "curve_surface_weight",
    "curvature_threshold",
    "curvature_weight",
    "mean_squared_curvature_threshold",
    "mean_squared_curvature_weight",
)


@pytree_dataclass(
    data=_STAGE_TWO_NUMERIC_FIELDS,
    meta=(
        "num_base_curves",
        "length_target_mode",
        "mean_squared_curvature_target_mode",
    ),
)
@dataclass(frozen=True, slots=True)
class _PreparedStageTwoConfig(StageTwoObjectiveConfig):
    """Validated config whose numbers are traced device operands.

    ``None`` weights and targets are pytree structure, so the selection of
    terms is part of the compiled program and the weights are not.
    """


def prepare_stage_two_config(
    config: StageTwoObjectiveConfig,
    extraction: CoilSetDofExtractionSpec | None = None,
    surface_gamma: jax.Array | None = None,
    surface_normal: jax.Array | None = None,
) -> _PreparedStageTwoConfig:
    """Validate ``config`` and place its numbers as device operands before tracing."""
    if (
        not isinstance(config.num_base_curves, Integral)
        or isinstance(config.num_base_curves, bool)
        or config.num_base_curves <= 0
    ):
        raise ValueError("num_base_curves must be a positive integer.")
    for name in ("length_target_mode", "mean_squared_curvature_target_mode"):
        if getattr(config, name) not in ("max", "identity"):
            raise ValueError(f"{name} must be 'max' or 'identity'.")
    numeric_values: dict[str, float | None] = dict(
        zip(
            _STAGE_TWO_NUMERIC_FIELDS,
            host_value(tuple(getattr(config, name) for name in _STAGE_TWO_NUMERIC_FIELDS)),
            strict=True,
        )
    )
    for name, value in numeric_values.items():
        if value is None and name in _OPTIONAL_FIELDS:
            continue
        if value is None or not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if extraction is not None:
        if config.num_base_curves > len(extraction.coils):
            raise ValueError("num_base_curves exceeds the available coils.")
        shapes = {coil.curve.quadpoints.shape for coil in extraction.coils}
        if len(shapes) != 1 or not next(iter(shapes))[0]:
            raise ValueError("Stage-II coils require matching nonempty quadrature grids.")
    if surface_gamma is not None and surface_normal is not None:
        if (
            surface_gamma.shape != surface_normal.shape
            or surface_gamma.ndim != 2
            or surface_gamma.shape[-1] != 3
            or surface_gamma.size == 0
        ):
            raise ValueError("Surface positions and normals must have matching (n, 3) shapes.")
    # Every number becomes a float64 operand, whatever type the caller used
    # (int, float, NumPy or JAX scalar): only shapes and None-ness are traced.
    def number(name: str) -> float:
        return np.float64(numeric_values[name])

    def optional(name: str) -> float | None:
        return None if numeric_values[name] is None else number(name)

    return runtime_device_put_tree(
        _PreparedStageTwoConfig(
            num_base_curves=int(config.num_base_curves),
            length_weight=optional("length_weight"),
            length_target=optional("length_target"),
            length_target_mode=config.length_target_mode,
            curve_curve_minimum_distance=number("curve_curve_minimum_distance"),
            curve_curve_weight=optional("curve_curve_weight"),
            curve_surface_minimum_distance=number("curve_surface_minimum_distance"),
            curve_surface_weight=optional("curve_surface_weight"),
            curvature_threshold=number("curvature_threshold"),
            curvature_weight=optional("curvature_weight"),
            mean_squared_curvature_threshold=number("mean_squared_curvature_threshold"),
            mean_squared_curvature_target_mode=config.mean_squared_curvature_target_mode,
            mean_squared_curvature_weight=optional("mean_squared_curvature_weight"),
        ),
    )


def _target_excess(value: jax.Array, target: float | jax.Array, mode: str) -> jax.Array:
    """The signed excess, clipped at zero for the native ``max`` target mode."""
    excess = value - target
    return jnp.maximum(excess, 0.0) if mode == "max" else excess


def _length_penalty(
    total_length: jax.Array, length_weight: float | jax.Array, config: StageTwoObjectiveConfig
) -> jax.Array:
    """``length_weight * L``, or ``length_weight * QuadraticPenalty(L, target, mode)``."""
    if config.length_target is None:
        return length_weight * total_length
    excess = _target_excess(total_length, config.length_target, config.length_target_mode)
    return 0.5 * length_weight * excess * excess


def _curve_curve_penalty(
    gamma: jax.Array,
    gammadash: jax.Array,
    num_base_curves: int,
    minimum_distance: float | jax.Array,
) -> jax.Array:
    """Sum the CurveCurveDistance terms of the pairs ``(i, j)``, ``j < min(i, num_base_curves)``.

    Pairs are formed from contiguous slices rather than gathered: a gather's
    gradient is a scatter-add, which deterministic GPU execution serializes.
    Each pair is evaluated when it is a native candidate (``distance_candidate_pure``).
    """

    def pair(gamma_1, gammadash_1, gamma_2, gammadash_2):
        return curve_curve_distance_penalty_pure(
            gamma_1, gammadash_1, gamma_2, gammadash_2, minimum_distance,
            distance_candidate_pure(gamma_1, gamma_2, minimum_distance),
        )

    against_curves = jax.vmap(pair, in_axes=(None, None, 0, 0))
    total = placement_zero(gamma)
    num_curves = int(gamma.shape[0])
    for index in range(1, min(num_base_curves, num_curves)):
        total = total + jnp.sum(
            against_curves(gamma[index], gammadash[index], gamma[:index], gammadash[:index])
        )
    if num_curves > num_base_curves:
        total = total + jnp.sum(
            jax.vmap(against_curves, in_axes=(0, 0, None, None))(
                gamma[num_base_curves:],
                gammadash[num_base_curves:],
                gamma[:num_base_curves],
                gammadash[:num_base_curves],
            )
        )
    return total


def stage_two_geometric_penalty(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> jax.Array:
    """Evaluate the weighted coil-geometry penalties for stacked coil geometry.

    ``gamma``, ``gammadash`` and ``gammadashdash`` have shape
    ``(ncoils, nquadpoints, 3)`` with the base curves first;
    ``surface_gamma`` and ``surface_normal`` have shape ``(npoints, 3)``.
    """
    if not isinstance(config, _PreparedStageTwoConfig):
        config = prepare_stage_two_config(config)
    base_gammadash = gammadash[: config.num_base_curves]
    base_gammadashdash = gammadashdash[: config.num_base_curves]
    base_speed = jnp.linalg.norm(base_gammadash, axis=2)
    result = placement_zero(gamma)

    if config.length_weight is not None:
        lengths = jax.vmap(curve_length_from_incremental_arclength_pure)(base_speed)
        result = result + _length_penalty(jnp.sum(lengths), config.length_weight, config)

    if config.curvature_weight is not None or config.mean_squared_curvature_weight is not None:
        base_kappa = jax.vmap(kappa_pure)(base_gammadash, base_gammadashdash)
        if config.curvature_weight is not None:
            curvature = jax.vmap(
                lambda current_kappa, current_gammadash: (
                    curvature_p_norm_from_kappa_pure(
                        current_kappa,
                        current_gammadash,
                        _CURVATURE_P,
                        config.curvature_threshold,
                    )
                )
            )(base_kappa, base_gammadash)
            result = result + config.curvature_weight * jnp.sum(curvature)
        if config.mean_squared_curvature_weight is not None:
            mean_squared_curvature = jax.vmap(mean_squared_curvature_pure)(
                base_kappa, base_gammadash
            )
            excess = _target_excess(
                mean_squared_curvature,
                config.mean_squared_curvature_threshold,
                config.mean_squared_curvature_target_mode,
            )
            result = result + (
                0.5 * config.mean_squared_curvature_weight * jnp.sum(excess * excess)
            )

    if config.curve_curve_weight is not None:
        result = result + config.curve_curve_weight * _curve_curve_penalty(
            gamma,
            gammadash,
            config.num_base_curves,
            config.curve_curve_minimum_distance,
        )

    if config.curve_surface_weight is not None:
        curve_surface = jax.vmap(
            lambda current_gamma, current_gammadash: (
                curve_surface_distance_penalty_pure(
                    current_gamma,
                    current_gammadash,
                    surface_gamma,
                    surface_normal,
                    config.curve_surface_minimum_distance,
                    distance_candidate_pure(
                        current_gamma, surface_gamma, config.curve_surface_minimum_distance
                    ),
                )
            )
        )(gamma, gammadash)
        result = result + config.curve_surface_weight * jnp.sum(curve_surface)

    return result


def stage_two_coil_geometry(
    extraction: CoilSetDofExtractionSpec,
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return stacked ``(gamma, gammadash, gammadashdash, currents)`` of every coil.

    ``parameters`` is the field's free DOF vector; coils keep extraction order.
    """
    coil_specs = coil_specs_from_dof_extraction_spec(extraction, parameters)
    # Places every current's tangent on the parameters' device (a fixed
    # current's would otherwise be a symbolic zero) without coupling any
    # parameter's tangent or cotangent into the currents.
    parameter_zero = placement_zero(parameters)
    geometry: list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = []
    geometry_by_curve: dict[int, tuple[jax.Array, ...]] = {}
    for coil_spec in coil_specs:
        curve_id = id(coil_spec.curve)
        curve_geometry = geometry_by_curve.get(curve_id)
        if curve_geometry is None:
            curve_geometry = curve_geometry_from_dofs(coil_spec.curve, coil_spec.curve.dofs)
            geometry_by_curve[curve_id] = curve_geometry
        gamma, gammadash, gammadashdash = curve_geometry
        gamma, gammadash, current = apply_coil_symmetry(
            gamma,
            gammadash,
            coil_spec.current.value[0],
            coil_spec.symmetry,
        )
        if coil_spec.symmetry.has_rotation:
            gammadashdash = gammadashdash @ coil_spec.symmetry.rotmat
        geometry.append((gamma, gammadash, gammadashdash, current + parameter_zero))
    gammas, gammadashs, gammadashdashs, currents = zip(
        *geometry,
        strict=True,
    )
    return (
        jnp.stack(gammas),
        jnp.stack(gammadashs),
        jnp.stack(gammadashdashs),
        jnp.stack(currents),
    )


@pytree_dataclass(
    data=("extraction", "flux_spec", "surface_gamma", "surface_normal", "config"),
    meta=(),
)
class StageTwoProblem:
    """Device operands of the fused Stage-II objective.

    Pass it to jitted programs as an argument, never through a closure: a
    rebuilt problem with new weights (any values, zero included) then reuses
    the compiled program.
    """

    extraction: CoilSetDofExtractionSpec
    flux_spec: FixedSurfaceFluxSpec
    surface_gamma: jax.Array
    surface_normal: jax.Array
    config: _PreparedStageTwoConfig


def make_stage_two_problem(
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    config: StageTwoObjectiveConfig,
    *,
    surface_gamma: jax.Array | None = None,
    surface_normal: jax.Array | None = None,
) -> StageTwoProblem:
    """Capture ``field``'s coil DOF layout and fixed values for the fused objective.

    The curve-surface distance uses the flux surface unless ``surface_gamma`` and
    ``surface_normal`` (both ``(n, 3)``) name another one. Rebuild the problem
    after fixing or unfixing coil DOFs or changing fixed values.
    """
    if (surface_gamma is None) != (surface_normal is None):
        raise ValueError("Pass both surface_gamma and surface_normal, or neither.")
    if surface_gamma is None or surface_normal is None:
        surface_gamma = flux_spec.points
        surface_normal = flux_spec.normal.reshape((-1, 3))
    extraction = field.coil_dof_extraction_spec()
    return StageTwoProblem(
        extraction=extraction,
        flux_spec=flux_spec,
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        config=prepare_stage_two_config(config, extraction, surface_gamma, surface_normal),
    )


def fused_stage_two_values(
    problem: StageTwoProblem,
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate the objective and diagnostics from one coil geometry pass.

    Returns ``(objective, squared_flux, geometric_penalty, max |B·n̂|,
    total base-curve length)`` for the free DOF vector ``parameters``.
    """
    gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
        problem.extraction,
        parameters,
    )
    flux_spec = problem.flux_spec
    magnetic_field = biot_savart_B(
        flux_spec.points,
        gamma,
        gammadash,
        currents,
    )
    squared_flux = fixed_surface_flux_integral_from_B(magnetic_field, flux_spec)
    geometric_penalty = stage_two_geometric_penalty(
        gamma,
        gammadash,
        gammadashdash,
        problem.surface_gamma,
        problem.surface_normal,
        problem.config,
    )
    base_speed = jnp.linalg.norm(
        gammadash[: problem.config.num_base_curves],
        axis=2,
    )
    total_curve_length = jnp.sum(jnp.mean(base_speed, axis=1))
    surface_normal_flat = flux_spec.normal.reshape((-1, 3))
    unit_normal = surface_normal_flat / jnp.linalg.norm(
        surface_normal_flat,
        axis=1,
        keepdims=True,
    )
    maximum_normal_field = jnp.max(
        jnp.abs(jnp.sum(magnetic_field * unit_normal, axis=1))
    )
    return (
        squared_flux + geometric_penalty,
        squared_flux,
        geometric_penalty,
        maximum_normal_field,
        total_curve_length,
    )


def fused_stage_two_objective(problem: StageTwoProblem, parameters: jax.Array) -> jax.Array:
    """Return the fused Stage-II objective for the free DOF vector ``parameters``.

    The distance terms decide which curve pairs (and curves near the surface)
    to evaluate from the device geometry, which is not bit-identical to the
    native geometry. A pair whose closest distance rounds to the other side of
    the threshold can therefore be evaluated here and skipped by native, or the
    reverse. Its penalty is then negligibly small either way, but degenerate
    geometry inside such a pair (a zero tangent or coincident points) can give
    a NaN gradient on one side and a finite zero on the other. The drop-in
    ``CurveCurveDistanceJAX`` and ``CurveSurfaceDistanceJAX`` use native's own
    candidate search and match it exactly.
    """
    return fused_stage_two_values(problem, parameters)[0]
