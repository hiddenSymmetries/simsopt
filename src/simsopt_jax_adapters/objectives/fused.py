"""Evaluate a native sum of JAX Stage-II terms as one compiled program."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import chain
from math import prod
from operator import is_

import jax
import numpy as np
from simsopt._core.optimizable import Optimizable, OptimizableSum, ScaledOptimizable
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt_jax.backend.dtypes import as_jax_float64
from simsopt_jax.backend.jit import jit
from simsopt_jax.backend.runtime import get_backend_config, get_runtime_jax_device
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    StageTwoProblem,
    fused_stage_two_objective,
    make_stage_two_problem,
)

from simsopt_jax_adapters.field._coil_graph import _unwrap_rotated_curve
from simsopt_jax_adapters.geo.curve_objectives import (
    JaxCurveCurveDistance,
    JaxCurveLength,
    JaxCurveSurfaceDistance,
    JaxLpCurveCurvature,
    JaxMeanSquaredCurvature,
    _array_snapshot_key,
)
from simsopt_jax_adapters.objectives.flux import JaxSquaredFlux, _surface_dofs_fingerprint

# One program per problem layout; new weights and thresholds reuse it.
_fused_value_and_grad = jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))

# The Lp exponent of the fused curvature penalty.
_CURVATURE_P = 2.0
_TARGET_MODES = ("max", "identity")
_SUPPORTED_TERMS = (
    "JaxSquaredFlux, JaxCurveLength, JaxLpCurveCurvature, JaxCurveCurveDistance, JaxCurveSurfaceDistance "
    "and QuadraticPenalty of JaxCurveLength or JaxMeanSquaredCurvature"
)
# Ends each captured list, so that an object moved across a list boundary changes the capture.
_LIST_END = object()


@dataclass(frozen=True)
class _Weighted:
    """A term with the ``ScaledOptimizable`` nodes in front of it, outermost first."""

    term: Optimizable
    scales: tuple[ScaledOptimizable, ...]

    def weight(self) -> float:
        """Multiply the current factors of the scaling nodes; factors may be replaced or changed in place."""
        return float(prod(float(scale.factor) for scale in self.scales))


def _weighted_terms(node: Optimizable, scales: tuple = ()) -> Iterator[_Weighted]:
    """Expand the native sums and scalings of ``node`` into weighted terms."""
    if type(node) is OptimizableSum:
        for opt in node.opts:
            yield from _weighted_terms(opt, scales)
    elif type(node) is ScaledOptimizable:
        yield from _weighted_terms(node.opt, (*scales, node))
    else:
        yield _Weighted(node, scales)


def _penalized(penalty: QuadraticPenalty) -> tuple[str, tuple]:
    """Return what a supported QuadraticPenalty penalizes and the terms it encloses."""
    inner = list(_weighted_terms(penalty.obj))
    kinds = {type(weighted.term) for weighted in inner}
    if not any(weighted.scales for weighted in inner):
        if kinds == {JaxCurveLength}:
            return "length", tuple(weighted.term for weighted in inner)
        if kinds == {JaxMeanSquaredCurvature} and len(inner) == 1:
            return "mean squared curvature", (inner[0].term,)
    raise ValueError(
        f"QuadraticPenalty of {type(penalty.obj).__name__} has no fused form; the fused objective takes "
        "QuadraticPenalty of an unscaled sum of JaxCurveLength terms or of one JaxMeanSquaredCurvature."
    )


def _curve_key(curve) -> tuple:
    """Identify a curve by the curve inside its ``RotatedCurve`` wrappers and their rotation."""
    inner, rotmat = _unwrap_rotated_curve(curve)
    return inner, None if rotmat is None else rotmat.tobytes()


def _coil_positions(curves: Sequence, keys: Sequence[tuple], name: str) -> list[int]:
    """Return each curve's coil position; together the curves must be the leading coils, once each."""
    position: dict[tuple, int] = {}
    for index, key in enumerate(keys):
        position.setdefault(key, index)
    positions = [position.get(_curve_key(curve)) for curve in curves]
    if None in positions:
        raise ValueError(f"{name} reads a curve that is not a coil of the JaxSquaredFlux field.")
    if sorted(positions) != list(range(len(positions))):
        raise ValueError(f"{name} terms must cover the leading coils of the field, one term per coil.")
    return positions


def _in_base_order(
    terms: Sequence[_Weighted], curve_of: Callable[[Optimizable], object], keys: Sequence[tuple], name: str
) -> tuple[_Weighted, ...]:
    """Order single-curve terms by the coil position of the curve each reads."""
    positions = _coil_positions([curve_of(weighted.term) for weighted in terms], keys, name)
    return tuple(weighted for _, weighted in sorted(zip(positions, terms), key=lambda item: item[0]))


def _single(terms: Sequence[_Weighted], name: str) -> _Weighted | None:
    if len(terms) > 1:
        raise ValueError(f"The objective holds {len(terms)} {name} terms; the fused objective takes one.")
    return terms[0] if terms else None


@dataclass(frozen=True)
class _StageTwoTerms:
    """The terms of a native objective that the fused Stage-II formula computes.

    Per-curve tuples hold one term per base curve, in base-curve order. ``enclosed`` holds the terms
    inside the penalties and ``captured`` the objects whose identity the reading depends on (see
    :func:`_captured`).
    """

    flux: _Weighted
    num_basecurves: int
    lengths: tuple[_Weighted, ...]
    length_penalty: _Weighted | None
    curvatures: tuple[_Weighted, ...]
    mean_squared_curvatures: tuple[_Weighted, ...]
    curve_curve: _Weighted | None
    curve_surface: _Weighted | None
    enclosed: tuple[Optimizable, ...] = ()
    captured: tuple = ()


def _captured(terms: _StageTwoTerms) -> tuple:
    """Return the field, coils, curves and enclosed terms that the terms read now, list by list."""
    field = terms.flux.term.field
    penalties = (*([terms.length_penalty] if terms.length_penalty else []), *terms.mean_squared_curvatures)
    return (
        field, *field.coils, _LIST_END,
        *(weighted.term.curve for weighted in (*terms.lengths, *terms.curvatures)), _LIST_END,
        *(penalty.term.obj for penalty in penalties), *terms.enclosed, _LIST_END,
        *(term.curve for term in terms.enclosed), _LIST_END,
        *chain.from_iterable(
            (*distance.term.curves, _LIST_END) for distance in (terms.curve_curve, terms.curve_surface) if distance
        ),
        *([terms.curve_surface.term.surface] if terms.curve_surface else []),
    )


_SINGLE_TERM_KINDS = (
    JaxSquaredFlux, JaxCurveLength, JaxLpCurveCurvature, JaxCurveCurveDistance, JaxCurveSurfaceDistance,
)


def _grouped_terms(objective: Optimizable) -> dict[object, list[_Weighted]]:
    """Group the weighted terms of ``objective`` by type, QuadraticPenalty terms by what they penalize."""
    groups: dict[object, list[_Weighted]] = {}
    for weighted in _weighted_terms(objective):
        kind: object = type(weighted.term)
        if kind is QuadraticPenalty:
            kind = (QuadraticPenalty, _penalized(weighted.term)[0])
        elif kind is JaxMeanSquaredCurvature:
            raise ValueError("The fused objective takes JaxMeanSquaredCurvature only inside QuadraticPenalty.")
        elif kind not in _SINGLE_TERM_KINDS:
            raise TypeError(
                f"{type(weighted.term).__name__} has no fused Stage-II form; the fused objective takes "
                f"{_SUPPORTED_TERMS}."
            )
        groups.setdefault(kind, []).append(weighted)
    return groups


def _read_terms(objective: Optimizable) -> _StageTwoTerms:
    """Sort the weighted terms of ``objective`` into the fused formula's terms.

    Raises:
        TypeError: a term has no fused form.
        ValueError: supported terms appear in a form or combination the formula cannot represent.
    """
    groups = _grouped_terms(objective)
    flux = _single(groups.get(JaxSquaredFlux, []), "JaxSquaredFlux")
    if flux is None:
        raise ValueError("The fused objective needs one JaxSquaredFlux term.")
    keys = [_curve_key(coil.curve) for coil in flux.term.field.coils]

    def penalized_curve(penalty):
        return _penalized(penalty)[1][0].curve

    lengths = _in_base_order(groups.get(JaxCurveLength, []), lambda term: term.curve, keys, "JaxCurveLength")
    length_penalty = _single(groups.get((QuadraticPenalty, "length"), []), "QuadraticPenalty of JaxCurveLength")
    if lengths and length_penalty is not None:
        raise ValueError(
            "The objective holds JaxCurveLength terms and a QuadraticPenalty of their sum; "
            "the fused objective takes one length term."
        )
    penalized_lengths = () if length_penalty is None else _coil_positions(
        [term.curve for term in _penalized(length_penalty.term)[1]], keys, "QuadraticPenalty of JaxCurveLength"
    )
    curvatures = _in_base_order(
        groups.get(JaxLpCurveCurvature, []), lambda term: term.curve, keys, "JaxLpCurveCurvature"
    )
    mean_squared_curvatures = _in_base_order(
        groups.get((QuadraticPenalty, "mean squared curvature"), []), penalized_curve, keys,
        "QuadraticPenalty of JaxMeanSquaredCurvature",
    )
    base_counts = {
        len(terms) for terms in (lengths, penalized_lengths, curvatures, mean_squared_curvatures) if terms
    }
    if len(base_counts) > 1:
        raise ValueError("The length and curvature terms must cover the same base curves.")
    curve_curve = _single(groups.get(JaxCurveCurveDistance, []), "JaxCurveCurveDistance")
    curve_surface = _single(groups.get(JaxCurveSurfaceDistance, []), "JaxCurveSurfaceDistance")
    for distance in (curve_curve, curve_surface):
        if distance is not None and [_curve_key(curve) for curve in distance.term.curves] != keys:
            raise ValueError(
                f"{type(distance.term).__name__} must take the curves of the JaxSquaredFlux field's coils, "
                "in coil order."
            )
    num_basecurves = (
        base_counts.pop() if base_counts
        else curve_curve.term.num_basecurves if curve_curve is not None
        else len(keys)
    )
    terms = _StageTwoTerms(
        flux=flux,
        num_basecurves=num_basecurves,
        lengths=lengths,
        length_penalty=length_penalty,
        curvatures=curvatures,
        mean_squared_curvatures=mean_squared_curvatures,
        curve_curve=curve_curve,
        curve_surface=curve_surface,
        enclosed=tuple(
            term for penalty in (*([length_penalty] if length_penalty else []), *mean_squared_curvatures)
            for term in _penalized(penalty.term)[1]
        ),
    )
    return replace(terms, captured=_captured(terms))


def _shared(values: Sequence, name: str):
    """Return the one value that all the terms of a kind must share."""
    if any(value != values[0] for value in values):
        raise ValueError(f"The {name} differ; the fused objective takes one value.")
    return values[0]


def _target_mode(penalty: QuadraticPenalty) -> str:
    if penalty.f not in _TARGET_MODES:
        raise ValueError(f"QuadraticPenalty f={penalty.f!r} has no fused form; use 'max' or 'identity'.")
    return penalty.f


def _check_searched_candidates(distance: JaxCurveCurveDistance | JaxCurveSurfaceDistance) -> None:
    if distance.candidates_assigned:
        raise ValueError(f"{type(distance).__name__} has assigned candidates; the fused objective finds its own.")


def _stage_two_settings(terms: _StageTwoTerms) -> dict[str, object]:
    """Read the current weights, thresholds and modes as ``StageTwoObjectiveConfig`` arguments.

    Raises:
        ValueError: a current setting has no fused form.
    """
    if terms.flux.weight() != 1.0:
        raise ValueError("The fused objective takes JaxSquaredFlux unscaled.")
    settings: dict[str, object] = {"num_basecurves": terms.num_basecurves}
    if terms.lengths:
        settings["length_weight"] = _shared([term.weight() for term in terms.lengths], "JaxCurveLength weights")
    if terms.length_penalty is not None:
        penalty = terms.length_penalty.term
        settings.update(
            length_weight=terms.length_penalty.weight(),
            length_target=float(penalty.cons),
            length_target_mode=_target_mode(penalty),
        )
    if terms.curvatures:
        if any(float(term.term.p) != _CURVATURE_P for term in terms.curvatures):
            raise ValueError("The fused objective takes JaxLpCurveCurvature with p=2 only.")
        settings.update(
            curvature_weight=_shared([term.weight() for term in terms.curvatures], "JaxLpCurveCurvature weights"),
            curvature_threshold=_shared(
                [float(term.term.threshold) for term in terms.curvatures], "JaxLpCurveCurvature thresholds"
            ),
        )
    if terms.mean_squared_curvatures:
        penalties = [term.term for term in terms.mean_squared_curvatures]
        settings.update(
            mean_squared_curvature_weight=_shared(
                [term.weight() for term in terms.mean_squared_curvatures], "mean squared curvature weights"
            ),
            mean_squared_curvature_threshold=_shared(
                [float(penalty.cons) for penalty in penalties], "mean squared curvature thresholds"
            ),
            mean_squared_curvature_target_mode=_shared(
                [_target_mode(penalty) for penalty in penalties], "mean squared curvature modes"
            ),
        )
    if terms.curve_curve is not None:
        distance = terms.curve_curve.term
        if distance.downsample != 1:
            raise ValueError("The fused objective takes JaxCurveCurveDistance with downsample=1 only.")
        if distance.num_basecurves != terms.num_basecurves:
            raise ValueError(
                f"JaxCurveCurveDistance has num_basecurves={distance.num_basecurves}; the length and curvature "
                f"terms cover {terms.num_basecurves} base curves."
            )
        _check_searched_candidates(distance)
        settings.update(
            curve_curve_weight=terms.curve_curve.weight(),
            curve_curve_minimum_distance=float(distance.minimum_distance),
        )
    if terms.curve_surface is not None:
        distance = terms.curve_surface.term
        _check_searched_candidates(distance)
        settings.update(
            curve_surface_weight=terms.curve_surface.weight(),
            curve_surface_minimum_distance=float(distance.minimum_distance),
        )
    return settings


@dataclass(frozen=True)
class _CachedProblem:
    extraction: object
    key: tuple
    problem: StageTwoProblem
    # Indices of the field's free DOFs in this object's DOF order.
    own_order: np.ndarray


@dataclass(frozen=True)
class _CachedEvaluation:
    problem: StageTwoProblem
    value: float
    # In this object's DOF order.
    gradient: np.ndarray


class JaxFusedStageTwoObjective(Optimizable):
    r"""Evaluate a native sum of JAX Stage-II terms as one compiled program.

    ``JaxFusedStageTwoObjective(JF)`` reads the tree that ``+``, ``*`` and ``sum`` build and computes
    ``J()`` and ``dJ()`` of ``JF`` with :func:`~simsopt_jax.objectives.fused_stage_two_objective`.
    ``JF`` holds one unscaled :class:`~simsopt_jax_adapters.objectives.JaxSquaredFlux` and, each optional
    and weighted by numbers or :class:`~simsopt.objectives.Weight` objects:

    - :class:`~simsopt_jax_adapters.geo.JaxCurveLength` terms, or one ``QuadraticPenalty`` of their
      unscaled sum;
    - :class:`~simsopt_jax_adapters.geo.JaxLpCurveCurvature` terms with ``p=2``;
    - ``QuadraticPenalty`` terms of one :class:`~simsopt_jax_adapters.geo.JaxMeanSquaredCurvature` each;
    - one :class:`~simsopt_jax_adapters.geo.JaxCurveCurveDistance` with ``downsample=1`` and one
      :class:`~simsopt_jax_adapters.geo.JaxCurveSurfaceDistance`, each of all the flux field's coil
      curves in coil order, without assigned candidates.

    Penalties use ``f`` ``"max"`` or ``"identity"``. Each kind of single-curve term has one term per
    leading coil of the field, the same coils for every kind and for the curve-curve distance's
    ``num_basecurves``, and one weight, threshold and mode.

    The set of terms is read at construction. Scale factors, weights, thresholds, exponents, targets
    and modes are read at each evaluation, and so are the coils, curves and enclosed terms that the
    terms hold: changed ones are read again and validated. ``J()`` and ``dJ()`` at the same DOFs run
    the program once. New numeric settings or fixed DOF values rebuild its operands; new penalty
    modes, newly present or absent terms or targets, or new free DOF counts compile it again.
    The wrapper is the whole objective: put every term into ``JF`` before wrapping and give the
    wrapper to the optimizer. It cannot be added to, scaled by or wrapped in other objectives, whose
    gradients would ask it for partial derivatives (``dJ(partials=True)``), which it does not provide.
    At a distance threshold the fused candidate test can round to the other side than the native
    search, which can change whether degenerate geometry gives a NaN gradient. Instances hold mutable
    caches and must not be shared across threads.

    Args:
        objective (Optimizable): the native objective to evaluate.

    Raises:
        TypeError: ``objective`` holds a term with no fused form.
        ValueError: a supported term has a form, setting or curve list the formula cannot represent.
    """

    def __init__(self, objective: Optimizable):
        self.objective = objective
        self._terms = _read_terms(objective)
        self._cached_problem: _CachedProblem | None = None
        self._cached_evaluation: _CachedEvaluation | None = None
        super().__init__(x0=np.asarray([]), depends_on=[objective])
        # Settings and layout errors surface here rather than at the first evaluation.
        self._current_problem()

    def _current_terms(self) -> _StageTwoTerms:
        """Return the terms, read again when a coil, curve or enclosed term they hold changed."""
        captured = self._terms.captured
        current = _captured(self._terms)
        if len(current) != len(captured) or not all(map(is_, current, captured)):
            self._terms = _read_terms(self.objective)
        return self._terms

    def _current_problem(self) -> _CachedProblem:
        """Return the program operands, rebuilt when settings, DOF layout or surface data change."""
        terms = self._current_terms()
        settings = _stage_two_settings(terms)
        flux = terms.flux.term
        field = flux.field
        extraction = field.coil_dof_extraction_spec()
        surface = None if terms.curve_surface is None else terms.curve_surface.term.surface
        flux_surface_dofs = _surface_dofs_fingerprint(flux.surface)
        key = (
            tuple(settings.items()),
            flux_surface_dofs,
            _array_snapshot_key(flux.target),
            flux.definition,
            None if surface is None else (
                flux_surface_dofs if surface is flux.surface else _surface_dofs_fingerprint(surface),
                surface.quadpoints_phi.tobytes(), surface.quadpoints_theta.tobytes(),
            ),
            get_backend_config(),
            get_runtime_jax_device(),
        )
        cached = self._cached_problem
        if cached is not None and cached.extraction is extraction and cached.key == key:
            return cached
        # The curve-surface distance reads its own surface, which need not be the flux surface.
        distance_surface = {} if surface is None else {
            "surface_gamma": as_jax_float64(surface.gamma().reshape((-1, 3))),
            "surface_normal": as_jax_float64(surface.normal().reshape((-1, 3))),
        }
        problem = make_stage_two_problem(
            field, flux.fixed_surface_flux_spec(), StageTwoObjectiveConfig(**settings), **distance_surface
        )
        field_ranges = {id(opt.dofs): range(*field.dof_indices[opt]) for opt in field.unique_dof_lineage}
        own_order = np.fromiter(
            chain.from_iterable(field_ranges[id(opt.dofs)] for opt in self.unique_dof_lineage if opt.local_dof_size),
            dtype=np.intp,
        )
        cached = _CachedProblem(extraction, key, problem, own_order)
        self._cached_problem = cached
        return cached

    def set_recompute_flag(self, parent=None):
        """Drop the cached program result: a DOF of the objective changed.

        A DOF change notifies this object once per path from the DOF owner, so without dependents of
        its own it skips the generic notification walk.

        Args:
            parent (Optimizable or None): Dependency that sent the notification.
        """
        self._cached_evaluation = None
        self.new_x = True
        if self._children:
            super().set_recompute_flag(parent=parent)

    def _value_and_gradient(self) -> tuple[float, np.ndarray]:
        """Return the value and the gradient in this object's DOF order, one program run per DOF vector."""
        cached_problem = self._current_problem()
        problem = cached_problem.problem
        cached = self._cached_evaluation
        if cached is None or cached.problem is not problem:
            x = as_jax_float64(self._terms.flux.term.field.x)
            # An explicit transfer, as under JAX_TRANSFER_GUARD=disallow; the reordering copies the gradient.
            value, gradient = jax.device_get(_fused_value_and_grad(problem, x))
            cached = _CachedEvaluation(problem, float(value), gradient[cached_problem.own_order])
            self._cached_evaluation = cached
        return cached.value, cached.gradient

    def J(self):
        """Evaluate the objective at the current DOFs with the fused program.

        Returns:
            float: the value of ``objective``, in its units.
        """
        return self._value_and_gradient()[0]

    def dJ(self, partials=False):
        """Return the gradient at the current DOFs with the fused program.

        Args:
            partials (bool): must be False; partial derivatives are not provided.

        Returns:
            Array of shape (ndofs,): free-DOF gradient in this object's DOF order, in objective units per
                native DOF unit.

        Raises:
            TypeError: ``partials=True``, as a sum, scaling or penalty holding this object asks.
        """
        if partials:
            raise TypeError(
                "JaxFusedStageTwoObjective provides no partial derivatives, which a sum, scaling or penalty "
                "holding it asks for: put every term into the objective before wrapping it and give the wrapper "
                "to the optimizer."
            )
        return self._value_and_gradient()[1].copy()

    return_fn_map = {"J": J, "dJ": dJ}
