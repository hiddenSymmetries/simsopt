"""Fused pure-JAX objective for filamentary Stage-II coil optimization.

One program maps the free coil DOFs of a :class:`JaxBiotSavart` field to coil
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
    """Structural contract for capturing coil layout and fixed DOFs.
    """

    def coil_dof_extraction_spec(self) -> CoilSetDofExtractionSpec:
        """Capture the field free-DOF layout and fixed values.

        Returns:
            CoilSetDofExtractionSpec object: immutable extraction payload.
        """
        ...


@dataclass(frozen=True, slots=True)
class StageTwoObjectiveConfig:
    """Immutable settings of the native Stage-II geometric penalties.

    None omits a term; zero retains it. Numerical weights are traced operands,
    so changing a weight reuses compilation. Term presence and target modes
    are static. Weight units set the units of the resulting weighted sum.

    Args:
        num_basecurves: int, positive count of leading base curves; other coils participate in distances.
        length_weight: float or None, multiplier of total base length; None omits the term.
        length_target: float or None, target total length in m; None selects a linear length term.
        length_target_mode: str, "max" clips excess below zero; "identity" keeps signed excess.
        curve_curve_minimum_distance: float, curve separation threshold in m.
        curve_curve_weight: float or None, multiplier of the curve-pair penalty in m^4.
        curve_surface_minimum_distance: float, curve-surface separation threshold in m.
        curve_surface_weight: float or None, multiplier of the curve-surface penalty in m^5.
        curvature_threshold: float, Lp curvature threshold in 1/m, with exponent fixed at 2.
        curvature_weight: float or None, multiplier of the curvature integral in 1/m.
        mean_squared_curvature_threshold: float, target arclength-averaged squared curvature in 1/m^2.
        mean_squared_curvature_target_mode: str, "max" clips excess below zero; "identity" keeps it signed.
        mean_squared_curvature_weight: float or None, multiplier of half the squared excess in 1/m^4.
    """

    num_basecurves: int
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
        "num_basecurves",
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
    """Validate settings and place numerical values before tracing.

    Args:
        config: StageTwoObjectiveConfig object, finite weights and thresholds.
        extraction: CoilSetDofExtractionSpec or None, optional coil-count/grid validation.
        surface_gamma: Array of shape (npoints, 3) or None, optional surface positions in m.
        surface_normal: Array of shape (npoints, 3) or None, optional normals in m^2.

    Returns:
        _PreparedStageTwoConfig object: validated config with float64 scalar device operands.
    """
    if (
        not isinstance(config.num_basecurves, Integral)
        or isinstance(config.num_basecurves, bool)
        or config.num_basecurves <= 0
    ):
        raise ValueError("num_basecurves must be a positive integer.")
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
        if config.num_basecurves > len(extraction.coils):
            raise ValueError("num_basecurves exceeds the available coils.")
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
            num_basecurves=int(config.num_basecurves),
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
    num_basecurves: int,
    minimum_distance: float | jax.Array,
) -> jax.Array:
    """Sum the CurveCurveDistance terms of the pairs ``(i, j)``, ``j < min(i, num_basecurves)``.

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
    for index in range(1, min(num_basecurves, num_curves)):
        total = total + jnp.sum(
            against_curves(gamma[index], gammadash[index], gamma[:index], gammadash[:index])
        )
    if num_curves > num_basecurves:
        total = total + jnp.sum(
            jax.vmap(against_curves, in_axes=(0, 0, None, None))(
                gamma[num_basecurves:],
                gammadash[num_basecurves:],
                gamma[:num_basecurves],
                gammadash[:num_basecurves],
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
    """Sum weighted geometric penalties over ordered coils, base curves first.

    Args:
        gamma: Array of shape (ncoils, nquadpoints, 3), positions in m.
        gammadash: Array of shape (ncoils, nquadpoints, 3), first unit-period parameter derivatives in m.
        gammadashdash: Array of shape (ncoils, nquadpoints, 3), second parameter derivatives in m.
        surface_gamma: Array of shape (npoints, 3), distance surface positions in m.
        surface_normal: Array of shape (npoints, 3), unnormalized distance surface normals in m^2.
        config: StageTwoObjectiveConfig object, weights, thresholds and base-curve count.

    Returns:
        Array: scalar weighted sum; units depend on the chosen weights.
    """
    if not isinstance(config, _PreparedStageTwoConfig):
        config = prepare_stage_two_config(config)
    base_gammadash = gammadash[: config.num_basecurves]
    base_gammadashdash = gammadashdash[: config.num_basecurves]
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
            config.num_basecurves,
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
    """Reconstruct geometry in extraction order, including symmetry copies.

    Args:
        extraction: CoilSetDofExtractionSpec object, ordered coil layout and fixed DOF snapshots.
        parameters: Array of shape (ndofs,), field free DOFs in extraction order, with native geometry/current units.

    Returns:
        tuple: (gamma, gammadash, gammadashdash, currents), three arrays of shape (ncoils, nquadpoints, 3) in m and one array of shape (ncoils,) in A.
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

    Pass as a jitted argument so rebuilt weights reuse compilation.

    Args:
        extraction: CoilSetDofExtractionSpec object, ordered coil layout and fixed DOF snapshots.
        flux_spec: FixedSurfaceFluxSpec object, fixed flux surface operands.
        surface_gamma: Array of shape (npoints, 3), distance surface positions in m.
        surface_normal: Array of shape (npoints, 3), unnormalized distance surface normals in m^2.
        config: _PreparedStageTwoConfig object, validated weights as device operands.
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
    """Capture coil layout and fixed values for a fused objective.

    Rebuild after fixing/unfixing DOFs or changing fixed values. The flux
    surface supplies the distance geometry unless both overrides are provided.

    Args:
        field: CoilDofExtractionProvider object, field whose free DOFs parameterize the objective.
        flux_spec: FixedSurfaceFluxSpec object, fixed flux surface operands.
        config: StageTwoObjectiveConfig object, geometric penalty settings.
        surface_gamma: Array of shape (npoints, 3) or None, distance positions in m.
        surface_normal: Array of shape (npoints, 3) or None, distance normals in m^2.

    Returns:
        StageTwoProblem object: captured device operands.
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
    """Evaluate flux and geometry once, returning objective diagnostics.

    Args:
        problem: StageTwoProblem object, captured device operands.
        parameters: Array of shape (ndofs,), field free DOFs in extraction order, with native geometry/current units.

    Returns:
        tuple of scalar arrays: (objective, squared_flux, geometric_penalty, maximum absolute normal field in T, total base length in m). Flux units follow integral_BdotN; weighted sum units follow config.
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
        gammadash[: problem.config.num_basecurves],
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
    """Evaluate squared flux plus weighted geometric penalties.

    Device candidate-search rounding may differ from native exactly at a
    distance threshold. Degenerate geometry in such a pair can therefore
    have a NaN gradient on only one side. Drop-in distance adapters instead
    use the native candidate search.

    Args:
        problem: StageTwoProblem object, captured device operands.
        parameters: Array of shape (ndofs,), field free DOFs in extraction order, with native geometry/current units.

    Returns:
        Array: scalar objective; flux definition and penalty weights determine units.
    """
    return fused_stage_two_values(problem, parameters)[0]
