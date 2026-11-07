"""Fused pure-JAX objective for filamentary and finite-build Stage-II coil optimization.

One program maps the free coil DOFs of a :class:`JaxBiotSavart` field to coil
geometry once, then evaluates the squared flux on a fixed surface plus the
penalties of the native Stage-II examples: coil-geometry penalties on the coil
centerlines (total and per-curve length, curve-curve and curve-surface
distance, Lp curvature and mean squared curvature) and the coil force, torque
and vacuum-energy objectives of :mod:`simsopt.field.force`. Every term uses the
formula of its native objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.dtypes import as_jax_float64, runtime_device_put_tree
from simsopt_jax.core import coil_forces
from simsopt_jax.core._device_scalars import placement_zero
from simsopt_jax.core.biotsavart import biot_savart_B
from simsopt_jax.core.curve_geometry import (
    _filament_offset,
    curve_filament_frame_from_dofs,
    curve_geometry_from_dofs,
)
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
    CoilDofExtractionSpec,
    CoilSetDofExtractionSpec,
    CoilSymmetrySpec,
    CurveFilamentSpec,
    FixedSurfaceFluxSpec,
    apply_coil_symmetry,
)
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.runtime.host_boundary import host_array, host_value

__all__ = [
    "CoilDofExtractionProvider",
    "StageTwoGeometry",
    "StageTwoObjectiveConfig",
    "StageTwoProblem",
    "fused_stage_two_objective",
    "fused_stage_two_values",
    "make_stage_two_problem",
    "prepare_stage_two_config",
    "stage_two_coil_geometry",
    "stage_two_geometric_penalty",
    "stage_two_geometry",
]

# LpCurveCurvature exponent of the native Stage-II examples.
_CURVATURE_P = 2.0


class CoilDofExtractionProvider(Protocol):
    """Structural contract for capturing coil layout and fixed DOFs."""

    def coil_dof_extraction_spec(self) -> CoilSetDofExtractionSpec:
        """Capture the field free-DOF layout and fixed values.

        Returns:
            CoilSetDofExtractionSpec object: immutable extraction payload.
        """
        ...


@dataclass(frozen=True, slots=True)
class StageTwoObjectiveConfig:
    """Immutable weights and thresholds of the Stage-II penalties.

    The penalties act on the coil centerlines, the first ``num_basecurves``
    of which are the base curves. Each coil is its own centerline, except in
    finite-build fields: their filaments (``CurveFilament``) come in packs of
    ``filaments_per_pack`` consecutive coils that share an underlying curve,
    frame rotation and symmetry, the order of ``create_multifilament_grid``
    and the symmetry helpers, and each pack has that curve as its centerline.

    The terms mirror the native objectives: ``length_weight`` times the total
    base-curve length (or ``QuadraticPenalty(total length, length_target,
    length_target_mode)``); ``individual_length_weight`` times the sum of
    ``QuadraticPenalty(CurveLength, target, individual_length_target_mode)``
    over the base curves; ``CurveCurveDistance`` over all centerlines with
    ``num_basecurves=num_basecurves`` (``curve_curve_pairs="base"``) or every
    pair (``"all"``); ``CurveSurfaceDistance`` over all centerlines;
    ``LpCurveCurvature(p=2)`` and ``QuadraticPenalty(MeanSquaredCurvature,
    threshold, mode)`` per base curve. The coil terms take the first
    ``num_basecurves`` coils as targets and the other coils as sources,
    ``force_downsample`` as ``downsample``, and need a field without
    filaments: ``LpCurveForce(targets, sources, p=force_p, threshold=force_threshold)``,
    ``LpCurveTorque`` likewise, ``SquaredMeanForce``, ``SquaredMeanTorque``
    and ``B2Energy`` over all coils.

    A term is part of the objective when its weight is not ``None``. Numbers
    (weights, targets, thresholds and exponents), zero included, are traced
    operands: changing them never recompiles, while adding or removing a term
    (or ``length_target``) or changing an integer or string setting changes
    the program.

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
        individual_length_weight: float or None, multiplier of per-base-curve quadratic length penalties in m^2.
        individual_length_targets: tuple of float, one length target in m per base curve when enabled.
        individual_length_target_mode: str, "max" clips negative excess; "identity" keeps signed excess.
        curve_curve_pairs: str, "base" uses native num_basecurves pairs; "all" uses every centerline pair.
        filaments_per_pack: int, positive number of consecutive filament coils sharing a centerline, rotation and symmetry.
        force_weight: float or None, multiplier of the Lp force integral in (MN/m)^force_p m.
        force_p: float, dimensionless force exponent.
        force_threshold: float, force-density threshold in MN/m.
        torque_weight: float or None, multiplier of the Lp torque integral in MN^torque_p m.
        torque_p: float, dimensionless torque exponent.
        torque_threshold: float, torque-density threshold in MN.
        squared_mean_force_weight: float or None, multiplier of squared integrated force in MN^2.
        squared_mean_torque_weight: float or None, multiplier of squared integrated torque in (MN m)^2.
        vacuum_energy_weight: float or None, multiplier of vacuum energy in MJ.
        force_downsample: int, positive static stride dividing the coil quadrature count for force, torque and energy.
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
    individual_length_weight: float | None = None
    individual_length_targets: tuple[float, ...] = ()
    individual_length_target_mode: Literal["max", "identity"] = "max"
    curve_curve_pairs: Literal["base", "all"] = "base"
    filaments_per_pack: int = 1
    force_weight: float | None = None
    force_p: float = 2.0
    force_threshold: float = 0.0
    torque_weight: float | None = None
    torque_p: float = 2.0
    torque_threshold: float = 0.0
    squared_mean_force_weight: float | None = None
    squared_mean_torque_weight: float | None = None
    vacuum_energy_weight: float | None = None
    force_downsample: int = 1


# Weights and the length target may be None (term or target absent).
_OPTIONAL_FIELDS = (
    "length_weight",
    "length_target",
    "curve_curve_weight",
    "curve_surface_weight",
    "curvature_weight",
    "mean_squared_curvature_weight",
    "individual_length_weight",
    "force_weight",
    "torque_weight",
    "squared_mean_force_weight",
    "squared_mean_torque_weight",
    "vacuum_energy_weight",
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
    "individual_length_weight",
    "force_weight",
    "force_p",
    "force_threshold",
    "torque_weight",
    "torque_p",
    "torque_threshold",
    "squared_mean_force_weight",
    "squared_mean_torque_weight",
    "vacuum_energy_weight",
)
_TARGET_MODE_FIELDS = (
    "length_target_mode",
    "mean_squared_curvature_target_mode",
    "individual_length_target_mode",
)
# Coil terms with targets and sources; those needing regularizations; all coil terms.
_TARGET_SOURCE_WEIGHTS = (
    "force_weight", "torque_weight", "squared_mean_force_weight", "squared_mean_torque_weight",
)
_REGULARIZED_WEIGHTS = ("force_weight", "torque_weight", "vacuum_energy_weight")
_COIL_TERM_WEIGHTS = (*_TARGET_SOURCE_WEIGHTS, "vacuum_energy_weight")


@pytree_dataclass(
    data=(*_STAGE_TWO_NUMERIC_FIELDS, "individual_length_targets"),
    meta=(
        "num_basecurves",
        *_TARGET_MODE_FIELDS,
        "curve_curve_pairs",
        "filaments_per_pack",
        "force_downsample",
    ),
)
@dataclass(frozen=True, slots=True)
class _PreparedStageTwoConfig(StageTwoObjectiveConfig):
    """Validated config whose numbers are traced device operands.

    ``None`` weights and targets are pytree structure, so the selection of
    terms is part of the compiled program and the weights are not.
    """


def _uses(config: StageTwoObjectiveConfig, weights: tuple[str, ...]) -> bool:
    return any(getattr(config, name) is not None for name in weights)


def _is_positive_integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0


def _same_leaves(first, second) -> bool:
    first_leaves, first_tree = jax.tree.flatten(first)
    second_leaves, second_tree = jax.tree.flatten(second)
    return first_tree == second_tree and all(
        np.array_equal(a, b) for a, b in zip(first_leaves, second_leaves, strict=True)
    )


def _centerline_template(coil: CoilDofExtractionSpec):
    """What a filament coil shares with its pack, or ``None`` for another coil."""
    curve = coil.curve
    if not isinstance(curve, CurveFilamentSpec):
        return None
    return (
        coil.curve_map,
        curve.base_curve,
        curve.base_curve_map,
        curve.rotation,
        curve.rotation_map,
        curve.frame_kind,
        coil.symmetry.rotmat,
        coil.symmetry.has_rotation,
    )


def _check_packs(coils: tuple[CoilDofExtractionSpec, ...], filaments_per_pack: int) -> None:
    """Check that consecutive groups of ``filaments_per_pack`` coils are whole packs."""
    if len(coils) % filaments_per_pack:
        raise ValueError("The number of coils must be a multiple of filaments_per_pack.")
    if filaments_per_pack == 1 and not any(isinstance(coil.curve, CurveFilamentSpec) for coil in coils):
        return
    templates = [_centerline_template(coil) for coil in host_value(coils)]
    for start in range(0, len(coils), filaments_per_pack):
        pack = templates[start:start + filaments_per_pack]
        first = pack[0]
        if first is None:
            if filaments_per_pack > 1:
                raise ValueError("filaments_per_pack > 1 requires finite-build filament coils.")
            continue
        if not all(template is not None and _same_leaves(template, first) for template in pack[1:]):
            raise ValueError(
                "The filaments of a pack must share their curve, frame rotation and symmetry."
            )
        previous = templates[start - 1] if start else None
        if previous is not None and _same_leaves(previous, first):
            raise ValueError("Consecutive packs share one centerline: filaments_per_pack is too small.")


def _check_coil_terms(config: StageTwoObjectiveConfig, coils: tuple[CoilDofExtractionSpec, ...]) -> None:
    if any(isinstance(coil.curve, CurveFilamentSpec) for coil in coils):
        raise ValueError("Force, torque and energy terms require a field without filament coils.")
    if _uses(config, _TARGET_SOURCE_WEIGHTS) and config.num_basecurves >= len(coils):
        raise ValueError(
            "Force and torque terms need source coils besides the num_basecurves targets."
        )
    num_quadpoints = coils[0].curve.quadpoints.shape[0]
    if num_quadpoints % config.force_downsample:
        raise ValueError(
            f"force_downsample ({config.force_downsample}) must evenly divide the "
            f"number of quadrature points ({num_quadpoints})."
        )


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
    for name in ("num_basecurves", "filaments_per_pack", "force_downsample"):
        if not _is_positive_integer(getattr(config, name)):
            raise ValueError(f"{name} must be a positive integer.")
    for name in _TARGET_MODE_FIELDS:
        if getattr(config, name) not in ("max", "identity"):
            raise ValueError(f"{name} must be 'max' or 'identity'.")
    if config.curve_curve_pairs not in ("base", "all"):
        raise ValueError("curve_curve_pairs must be 'base' or 'all'.")
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
    individual_length_targets = tuple(
        np.float64(target) for target in host_value(tuple(config.individual_length_targets))
    )
    if not all(isfinite(target) for target in individual_length_targets):
        raise ValueError("individual_length_targets must be finite.")
    if (
        config.individual_length_weight is not None
        and len(individual_length_targets) != config.num_basecurves
    ):
        raise ValueError("individual_length_targets must hold one length per base curve.")
    if extraction is not None:
        shapes = {coil.curve.quadpoints.shape for coil in extraction.coils}
        if len(shapes) != 1 or not next(iter(shapes))[0]:
            raise ValueError("Stage-II coils require matching nonempty quadrature grids.")
        _check_packs(extraction.coils, config.filaments_per_pack)
        if config.num_basecurves > len(extraction.coils) // config.filaments_per_pack:
            raise ValueError("num_basecurves exceeds the available coil centerlines.")
        if _uses(config, _COIL_TERM_WEIGHTS):
            _check_coil_terms(config, extraction.coils)
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
            individual_length_weight=optional("individual_length_weight"),
            individual_length_targets=individual_length_targets,
            individual_length_target_mode=config.individual_length_target_mode,
            curve_curve_pairs=config.curve_curve_pairs,
            filaments_per_pack=int(config.filaments_per_pack),
            force_weight=optional("force_weight"),
            force_p=number("force_p"),
            force_threshold=number("force_threshold"),
            torque_weight=optional("torque_weight"),
            torque_p=number("torque_p"),
            torque_threshold=number("torque_threshold"),
            squared_mean_force_weight=optional("squared_mean_force_weight"),
            squared_mean_torque_weight=optional("squared_mean_torque_weight"),
            vacuum_energy_weight=optional("vacuum_energy_weight"),
            force_downsample=int(config.force_downsample),
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
    """Sum weighted geometric penalties over ordered centerlines, base curves first.

    Args:
        gamma: Array of shape (ncenterlines, nquadpoints, 3), positions in m.
        gammadash: Array of shape (ncenterlines, nquadpoints, 3), first unit-period parameter derivatives in m.
        gammadashdash: Array of shape (ncenterlines, nquadpoints, 3), second parameter derivatives in m.
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

    if config.length_weight is not None or config.individual_length_weight is not None:
        lengths = jax.vmap(curve_length_from_incremental_arclength_pure)(base_speed)
        if config.length_weight is not None:
            result = result + _length_penalty(jnp.sum(lengths), config.length_weight, config)
        if config.individual_length_weight is not None:
            excess = _target_excess(
                lengths, jnp.stack(config.individual_length_targets),
                config.individual_length_target_mode,
            )
            result = result + 0.5 * config.individual_length_weight * jnp.sum(excess * excess)

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
        num_pair_base_curves = (
            config.num_basecurves if config.curve_curve_pairs == "base" else int(gamma.shape[0])
        )
        result = result + config.curve_curve_weight * _curve_curve_penalty(
            gamma,
            gammadash,
            num_pair_base_curves,
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


@pytree_dataclass(
    data=(
        "gamma",
        "gammadash",
        "currents",
        "centerline_gamma",
        "centerline_gammadash",
        "centerline_gammadashdash",
    ),
    meta=(),
)
class StageTwoGeometry:
    """Stacked geometry of one fused evaluation.

    ``gamma`` and ``gammadash`` ``(ncoils, nquadpoints, 3)`` and ``currents``
    ``(ncoils,)`` of every coil in extraction order; ``centerline_*``
    ``(ncenterlines, nquadpoints, 3)`` of the coil centerlines (see
    :class:`StageTwoObjectiveConfig`).

    Args:
        gamma: Array of shape (ncoils, nquadpoints, 3), coil positions in m.
        gammadash: Array of shape (ncoils, nquadpoints, 3), coil first parameter derivatives in m.
        currents: Array of shape (ncoils,), coil currents in A.
        centerline_gamma: Array of shape (ncenterlines, nquadpoints, 3), centerline positions in m.
        centerline_gammadash: Array of shape (ncenterlines, nquadpoints, 3), centerline first parameter derivatives in m.
        centerline_gammadashdash: Array of shape (ncenterlines, nquadpoints, 3), centerline second parameter derivatives in m.
    """

    gamma: jax.Array
    gammadash: jax.Array
    currents: jax.Array
    centerline_gamma: jax.Array
    centerline_gammadash: jax.Array
    centerline_gammadashdash: jax.Array


def _with_symmetry(*vectors: jax.Array, symmetry: CoilSymmetrySpec) -> tuple[jax.Array, ...]:
    """Rotate stacks of row vectors by a coil symmetry, as ``RotatedCurve`` does."""
    if not symmetry.has_rotation:
        return vectors
    return tuple(vector @ symmetry.rotmat for vector in vectors)


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


def stage_two_geometry(
    extraction: CoilSetDofExtractionSpec,
    parameters: jax.Array,
    filaments_per_pack: int = 1,
) -> StageTwoGeometry:
    """Return the stacked coil and centerline geometry for the free DOF vector ``parameters``.

    ``parameters`` is the field's free DOF vector; coils keep extraction order
    and each group of ``filaments_per_pack`` coils has one centerline. The
    filaments of a pack are offsets of their pack's frame, which is evaluated
    once, from the pack's first coil (see :class:`StageTwoObjectiveConfig`).

    Args:
        extraction: CoilSetDofExtractionSpec object, ordered coil layout and fixed DOF snapshots.
        parameters: Array of shape (ndofs,), field free DOFs in native units and extraction order.
        filaments_per_pack: int, validated consecutive pack size, one for ordinary coils.

    Returns:
        StageTwoGeometry object: coil and centerline arrays; field geometry includes filament offsets while penalties use centerlines.
    """
    coil_specs = coil_specs_from_dof_extraction_spec(extraction, parameters)
    # Places every current's tangent on the parameters' device (a fixed
    # current's would otherwise be a symbolic zero) without coupling any
    # parameter's tangent or cotangent into the currents.
    parameter_zero = placement_zero(parameters)
    # (gamma, gammadash, gammadashdash) of a curve, followed for a filament
    # pack by its rotated frame (normal, binormal and their derivatives); keyed
    # by the curve of the coil, or of the pack's first coil for filaments.
    curve_geometry: dict[int, tuple[jax.Array, ...]] = {}
    filament_geometry: dict[int, tuple[jax.Array, jax.Array]] = {}
    coils: list[tuple[jax.Array, jax.Array, jax.Array]] = []
    centerlines: list[tuple[jax.Array, ...]] = []
    for index, coil_spec in enumerate(coil_specs):
        curve = coil_spec.curve
        is_filament = isinstance(curve, CurveFilamentSpec)
        source = coil_specs[index - index % filaments_per_pack].curve if is_filament else curve
        if id(source) not in curve_geometry:
            curve_geometry[id(source)] = (
                curve_filament_frame_from_dofs(source, source.dofs)
                if isinstance(source, CurveFilamentSpec)
                else curve_geometry_from_dofs(source, source.dofs)
            )
        geometry = curve_geometry[id(source)]
        if isinstance(curve, CurveFilamentSpec):
            if id(curve) not in filament_geometry:
                gamma, gammadash, _gammadashdash, normal, binormal, normal_dash, binormal_dash = geometry
                filament_geometry[id(curve)] = (
                    _filament_offset(gamma, normal, binormal, curve.dn, curve.db),
                    _filament_offset(gammadash, normal_dash, binormal_dash, curve.dn, curve.db),
                )
            gamma, gammadash = filament_geometry[id(curve)]
        else:
            gamma, gammadash = geometry[0], geometry[1]
        gamma, gammadash, current = apply_coil_symmetry(
            gamma, gammadash, coil_spec.current.value[0], coil_spec.symmetry
        )
        coils.append((gamma, gammadash, current + parameter_zero))
        if index % filaments_per_pack:
            continue
        if is_filament:
            centerlines.append(_with_symmetry(*geometry[:3], symmetry=coil_spec.symmetry))
        else:
            centerlines.append((gamma, gammadash, *_with_symmetry(geometry[2], symmetry=coil_spec.symmetry)))
    gammas, gammadashs, currents = zip(*coils, strict=True)
    centerline_gammas, centerline_gammadashs, centerline_gammadashdashs = zip(*centerlines, strict=True)
    return StageTwoGeometry(
        gamma=jnp.stack(gammas),
        gammadash=jnp.stack(gammadashs),
        currents=jnp.stack(currents),
        centerline_gamma=jnp.stack(centerline_gammas),
        centerline_gammadash=jnp.stack(centerline_gammadashs),
        centerline_gammadashdash=jnp.stack(centerline_gammadashdashs),
    )


@pytree_dataclass(
    data=("extraction", "flux_spec", "surface_gamma", "surface_normal", "regularizations", "config"),
    meta=(),
)
class StageTwoProblem:
    """Device operands of the fused Stage-II objective.

    Pass it to jitted programs as an argument, never through a closure: a
    rebuilt problem with new weights (any values, zero included) then reuses
    the compiled program. ``regularizations`` holds one cross-section
    regularization per coil, or none when no term uses them.

    Args:
        extraction: CoilSetDofExtractionSpec object, ordered coil layout and fixed DOF snapshots.
        flux_spec: FixedSurfaceFluxSpec object, fixed flux surface operands.
        surface_gamma: Array of shape (npoints, 3), distance surface positions in m.
        surface_normal: Array of shape (npoints, 3), unnormalized distance normals in m^2.
        regularizations: Array of shape (ncoils,), fixed coil cross-section regularizations in m^2.
        config: _PreparedStageTwoConfig object, validated settings and numerical device operands.
    """

    extraction: CoilSetDofExtractionSpec
    flux_spec: FixedSurfaceFluxSpec
    surface_gamma: jax.Array
    surface_normal: jax.Array
    regularizations: jax.Array
    config: _PreparedStageTwoConfig


def make_stage_two_problem(
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    config: StageTwoObjectiveConfig,
    *,
    regularizations: jax.typing.ArrayLike | Sequence[jax.typing.ArrayLike] | None = None,
    surface_gamma: jax.Array | None = None,
    surface_normal: jax.Array | None = None,
) -> StageTwoProblem:
    """Capture ``field``'s coil DOF layout and fixed values for the fused objective.

    The force, torque and energy terms need ``regularizations``, the
    ``regularization`` of every coil in the field's order. The curve-surface
    distance uses the flux surface unless ``surface_gamma`` and
    ``surface_normal`` (both ``(n, 3)``) name another one. Rebuild the problem
    after fixing or unfixing coil DOFs or changing fixed values.

    Args:
        field: CoilDofExtractionProvider object, field defining the free DOF layout.
        flux_spec: FixedSurfaceFluxSpec object, fixed flux surface operands.
        config: StageTwoObjectiveConfig object, weights, thresholds, packs and coil-term settings.
        surface_gamma: Array of shape (npoints, 3) or None, distance surface positions in m; None selects flux geometry.
        surface_normal: Array of shape (npoints, 3) or None, unnormalized normals in m^2; supply both surface overrides.
        regularizations: Array of shape (ncoils,) or None, fixed cross-section values in m^2; required for Lp or energy terms.

    Returns:
        StageTwoProblem object: validated captured device operands.
    """
    if (surface_gamma is None) != (surface_normal is None):
        raise ValueError("Pass both surface_gamma and surface_normal, or neither.")
    if surface_gamma is None or surface_normal is None:
        surface_gamma = flux_spec.points
        surface_normal = flux_spec.normal.reshape((-1, 3))
    extraction = field.coil_dof_extraction_spec()
    prepared = prepare_stage_two_config(config, extraction, surface_gamma, surface_normal)
    if regularizations is None:
        if _uses(prepared, _REGULARIZED_WEIGHTS):
            raise ValueError(
                "The force, torque and energy terms need the coils' regularizations."
            )
        regularization_values = np.zeros((0,), dtype=np.float64)
    else:
        regularization_values = host_array(regularizations, dtype=np.float64)
        if regularization_values.shape != (len(extraction.coils),):
            raise ValueError("regularizations must hold one value per coil.")
    return StageTwoProblem(
        extraction=extraction,
        flux_spec=flux_spec,
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        regularizations=as_jax_float64(regularization_values),
        config=prepared,
    )


def _coil_terms(geometry: StageTwoGeometry, problem: StageTwoProblem) -> jax.Array:
    """Weighted force, torque and energy terms; targets are the first ``num_basecurves`` coils."""
    config = problem.config
    count = config.num_basecurves
    downsample = config.force_downsample
    targets = (geometry.gamma[:count], geometry.gammadash[:count], geometry.currents[:count])
    sources = ((geometry.gamma[count:], geometry.gammadash[count:], geometry.currents[count:]),)
    # Without filaments, the centerlines are the coils.
    target_gammadashdash = geometry.centerline_gammadashdash[:count]
    quadpoints = problem.extraction.coils[0].curve.quadpoints
    target_regularizations = problem.regularizations[:count]
    result = placement_zero(geometry.gamma)
    if config.force_weight is not None:
        result = result + config.force_weight * coil_forces.lp_force(
            targets, target_gammadashdash, quadpoints, target_regularizations, sources,
            config.force_p, config.force_threshold, downsample,
        )
    if config.torque_weight is not None:
        result = result + config.torque_weight * coil_forces.lp_torque(
            targets, target_gammadashdash, quadpoints, target_regularizations, sources,
            config.torque_p, config.torque_threshold, downsample,
        )
    if config.squared_mean_force_weight is not None:
        result = result + config.squared_mean_force_weight * coil_forces.squared_mean_force(
            targets, sources, downsample
        )
    if config.squared_mean_torque_weight is not None:
        result = result + config.squared_mean_torque_weight * coil_forces.squared_mean_torque(
            targets, sources, downsample
        )
    if config.vacuum_energy_weight is not None:
        result = result + config.vacuum_energy_weight * coil_forces.b2energy(
            geometry.gamma, geometry.gammadash, geometry.currents, problem.regularizations, downsample
        )
    return result


def fused_stage_two_values(
    problem: StageTwoProblem,
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate flux and geometry once, returning objective diagnostics.

    Args:
        problem: StageTwoProblem object, captured device operands.
        parameters: Array of shape (ndofs,), field free DOFs in extraction order, with native geometry/current units.

    Returns:
        tuple of scalar arrays: (objective, squared_flux, penalties excluding flux, maximum absolute normal field in T, total base length in m). Flux units follow integral_BdotN; weighted sum units follow config.
    """
    config = problem.config
    geometry = stage_two_geometry(problem.extraction, parameters, config.filaments_per_pack)
    flux_spec = problem.flux_spec
    magnetic_field = biot_savart_B(
        flux_spec.points,
        geometry.gamma,
        geometry.gammadash,
        geometry.currents,
    )
    squared_flux = fixed_surface_flux_integral_from_B(magnetic_field, flux_spec)
    penalties = stage_two_geometric_penalty(
        geometry.centerline_gamma,
        geometry.centerline_gammadash,
        geometry.centerline_gammadashdash,
        problem.surface_gamma,
        problem.surface_normal,
        config,
    )
    if _uses(config, _COIL_TERM_WEIGHTS):
        penalties = penalties + _coil_terms(geometry, problem)
    base_speed = jnp.linalg.norm(
        geometry.centerline_gammadash[: config.num_basecurves],
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
        squared_flux + penalties,
        squared_flux,
        penalties,
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
