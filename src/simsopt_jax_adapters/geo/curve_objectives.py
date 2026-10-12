"""JAX coil-geometry penalties as drop-in native Optimizables.

Each class mirrors the native objective named by removing the ``Jax`` prefix in
:mod:`simsopt.geo.curveobjectives`: same constructor arguments, value,
dependencies, ``Derivative`` (fixed and free partials of the curve DOFs) and
``shortest_distance``. The curves evaluate their own geometry
(``gamma``/``gammadash``/``gammadashdash``) and coefficient VJPs on the host;
the geometry moves to the active JAX device and the gradients back through
explicit transfers, and the penalty with its gradient runs as one jitted JAX
program. Curvature is computed from ``gammadash`` and ``gammadashdash`` in that
program rather than by the curve. Distance penalties use the native C++
candidate search, then evaluate every point pair within each selected pair.
For C++ curves (and their rotated copies) these paths make no
implicit transfer; JAX-backed native curves (``JaxCurve`` subclasses) still
compute their own geometry and VJPs with implicit transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
import simsoptpp as sopp
from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt_jax.backend.runtime import get_backend_policy, get_runtime_jax_device
from simsopt_jax.core._math_utils import as_jax_array as _as_jax_array
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.core.curve_kernels import (
    curvature_p_norm_from_kappa_pure,
    curve_curve_distance_penalty_pure,
    curve_length_from_incremental_arclength_pure,
    curve_surface_distance_penalty_pure,
    kappa_pure,
    mean_squared_curvature_pure,
)
from simsopt_jax.runtime.host_boundary import host_array, host_tree

__all__ = [
    "JaxCurveCurveDistance",
    "JaxCurveLength",
    "JaxCurveSurfaceDistance",
    "JaxLpCurveCurvature",
    "JaxMeanSquaredCurvature",
]


def _host_float(value) -> float:
    return float(host_array(value, dtype=np.float64))


def _length_from_tangent(gammadash):
    return curve_length_from_incremental_arclength_pure(jnp.linalg.norm(gammadash, axis=1))


def _lp_curvature_from_geometry(gammadash, gammadashdash, p, threshold):
    return curvature_p_norm_from_kappa_pure(
        kappa_pure(gammadash, gammadashdash), gammadash, p, threshold
    )


def _mean_squared_curvature_from_geometry(gammadash, gammadashdash):
    return mean_squared_curvature_pure(kappa_pure(gammadash, gammadashdash), gammadash)


_curve_length = jax.jit(_length_from_tangent)
_curve_length_grad = jax.jit(jax.grad(_length_from_tangent))
_lp_curvature = jax.jit(_lp_curvature_from_geometry)
_lp_curvature_grad = jax.jit(jax.grad(_lp_curvature_from_geometry, argnums=(0, 1)))
_mean_squared_curvature = jax.jit(_mean_squared_curvature_from_geometry)
_mean_squared_curvature_grad = jax.jit(
    jax.grad(_mean_squared_curvature_from_geometry, argnums=(0, 1))
)


def _curve_position_samples(curve, downsample=1):
    gamma = curve.gamma()
    return gamma if downsample == 1 else gamma[::downsample]


def _add_curve_vjp(buffer, values, downsample):
    if downsample == 1:
        buffer += values
    else:
        buffer[::downsample] += values


def _sum_curve_vjp_contributions(curves, dgamma_vjps, dgammadash_vjps):
    return sum(
        curve.dgamma_by_dcoeff_vjp(dgamma_vjp)
        + curve.dgammadash_by_dcoeff_vjp(dgammadash_vjp)
        for curve, dgamma_vjp, dgammadash_vjp in zip(
            curves, dgamma_vjps, dgammadash_vjps
        )
    )


def _tangents(curve):
    """``gammadash`` and ``gammadashdash`` of a native curve, explicitly placed."""
    return _as_jax_float64(curve.gammadash()), _as_jax_float64(curve.gammadashdash())


def _tangent_derivative(curve, grad_gammadash, grad_gammadashdash):
    return curve.dgammadash_by_dcoeff_vjp(
        host_array(grad_gammadash, dtype=np.float64)
    ) + curve.dgammadashdash_by_dcoeff_vjp(host_array(grad_gammadashdash, dtype=np.float64))


@lru_cache(maxsize=64)
def _quadrature_classes(sample_counts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Curve indices grouped by sample count, classes and members in curve order."""
    members_by_count: dict[int, list[int]] = {}
    for index, count in enumerate(sample_counts):
        members_by_count.setdefault(count, []).append(index)
    return tuple(tuple(members) for members in members_by_count.values())


def _class_stacked_geometry(curves, class_members, downsample):
    """Return device stacks of sampled ``gamma``/``gammadash`` per quadrature class."""

    def _stack(samples):
        return tuple(
            _as_jax_float64(
                np.stack(
                    [
                        host_array(samples(curves[index]), dtype=np.float64)[::downsample]
                        for index in members
                    ]
                )
            )
            for members in class_members
        )

    return (
        _stack(lambda curve: curve.gamma()),
        _stack(lambda curve: curve.gammadash()),
    )


def _array_snapshot_key(values) -> tuple[tuple[int, ...], str, bytes]:
    array = host_array(values)
    return array.shape, array.dtype.str, array.tobytes()


def _distance_candidate_key(curves, minimum_distance) -> tuple[object, ...]:
    """Exact native candidate inputs, without resolving a backend or placing arrays."""
    return (
        _array_snapshot_key(minimum_distance),
        tuple(
            (_array_snapshot_key(curve.gamma()), _array_snapshot_key(curve.quadpoints))
            for curve in curves
        ),
    )


def _distance_operand_key(curves, candidate_key: tuple[object, ...]) -> tuple[object, ...]:
    """Invalidate placed operands when geometry, policy or default-device scope changes.

    Backend dtypes remains the placement owner; recording the scope prevents
    cross-scope reuse without duplicating its device-selection rules.
    """
    return candidate_key + (
        get_backend_policy(), get_runtime_jax_device(), jax.config.values["jax_default_device"],
        tuple(_array_snapshot_key(curve.gammadash()) for curve in curves),
    )


def _class_geometry_derivative(curves, class_members, class_dgammas, class_dgammadashes, downsample):
    """Scatter per-class geometry cotangents back to the curves' coefficient derivatives."""
    class_dgammas, class_dgammadashes = host_tree(
        (class_dgammas, class_dgammadashes), dtype=np.float64
    )
    dgamma_buffers = [np.zeros_like(curve.gamma()) for curve in curves]
    dgammadash_buffers = [np.zeros_like(curve.gammadash()) for curve in curves]
    for members, dgammas, dgammadashes in zip(
        class_members, class_dgammas, class_dgammadashes
    ):
        for row, index in enumerate(members):
            _add_curve_vjp(dgamma_buffers[index], dgammas[row], downsample)
            _add_curve_vjp(dgammadash_buffers[index], dgammadashes[row], downsample)
    return _sum_curve_vjp_contributions(curves, dgamma_buffers, dgammadash_buffers)


class JaxCurveLength(Optimizable):
    r"""JAX version of :class:`simsopt.geo.CurveLength`: the length of a curve,

    .. math::
        J = \int_{\text{curve}}~dl = \int_0^1 |\Gamma'(t)|~dt,

    where :math:`\Gamma(t)` is the curve, evaluated as the mean of :math:`|\Gamma'|` over its
    quadrature points.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
    """

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m.
        """
        return _host_float(_curve_length(_as_jax_float64(self.curve.gammadash())))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return self.curve.dgammadash_by_dcoeff_vjp(
            host_array(
                _curve_length_grad(_as_jax_float64(self.curve.gammadash())),
                dtype=np.float64,
            )
        )

    return_fn_map = {"J": J, "dJ": dJ}


class JaxLpCurveCurvature(Optimizable):
    r"""JAX version of :class:`simsopt.geo.LpCurveCurvature`: penalize curvature above a threshold,

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(\kappa - \kappa_0, 0)^p ~dl,

    where :math:`\kappa = |\Gamma' \times \Gamma''| / |\Gamma'|^3` is the curvature and
    :math:`\kappa_0` is ``threshold``. No :math:`p`-th root is taken.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
        p (float): dimensionless exponent.
        threshold (float): curvature threshold :math:`\kappa_0` in 1/m; default 0.0.
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])

    def _parameters(self):
        return _as_jax_float64(self.p), _as_jax_float64(self.threshold)

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m^(1-p).
        """
        return _host_float(_lp_curvature(*_tangents(self.curve), *self._parameters()))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return _tangent_derivative(
            self.curve, *_lp_curvature_grad(*_tangents(self.curve), *self._parameters())
        )

    return_fn_map = {"J": J, "dJ": dJ}


class JaxMeanSquaredCurvature(Optimizable):
    r"""JAX version of :class:`simsopt.geo.MeanSquaredCurvature`: the mean squared curvature,

    .. math::
        J = \frac{1}{L} \int_{\text{curve}} \kappa^2 ~dl,

    where :math:`L` is the curve length and
    :math:`\kappa = |\Gamma' \times \Gamma''| / |\Gamma'|^3` is the curvature.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
    """

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in 1/m^2.
        """
        return _host_float(_mean_squared_curvature(*_tangents(self.curve)))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return _tangent_derivative(
            self.curve, *_mean_squared_curvature_grad(*_tangents(self.curve))
        )

    return_fn_map = {"J": J, "dJ": dJ}


@dataclass(frozen=True)
class _CurvePairBatch:
    """Curve pairs whose first and second members share one quadrature class each.

    ``first_rows[k]`` and ``second_rows[k]`` index pair ``k``'s curves in the
    ``first_class`` and ``second_class`` stacks of :class:`_CurvePairPlan`.
    """

    first_class: int
    second_class: int
    first_rows: tuple[int, ...]
    second_rows: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _CurvePairPlan:
    """Static vectorization of a curve-pair sum over equal-shape curve stacks.

    ``class_members[c]`` lists the curve indices stacked, in curve order, into
    quadrature class ``c``; every curve has a class, including curves that
    appear in no pair. Each batch holds the pairs of one (first class, second
    class) combination, in first-appearance order of the pair sequence; pair
    order is kept only within a batch. Summing batch totals therefore
    reassociates the per-pair loop's running sum (about 1e-16 relative). The
    plan is hashable so jit specializes on it as a static argument.
    """

    class_members: tuple[tuple[int, ...], ...]
    batches: tuple[_CurvePairBatch, ...]


@dataclass(frozen=True)
class _DistanceCandidates:
    """Host-only native candidate data, independent of JAX operand placement."""

    key: tuple[object, ...]
    candidates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _CurveCurveSnapshot:
    """One exact-state set of placed curve-curve penalty operands."""

    key: tuple[object, ...]
    plan: _CurvePairPlan
    operands: tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], jax.Array, tuple[jax.Array, ...]]


@dataclass(frozen=True)
class _CurveSurfaceSnapshot:
    """One exact-state set of placed operands including the surface geometry."""

    key: tuple[object, ...]
    class_members: tuple[tuple[int, ...], ...]
    operands: tuple[
        tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...],
        jax.Array, jax.Array, jax.Array,
    ]


def _curve_pairs(num_curves: int, num_basecurves: int):
    """The native ``CurveCurveDistance`` pairs ``(i, j)``, ``j < min(i, num_basecurves)``."""
    return tuple(
        (i, j) for i in range(num_curves) for j in range(min(i, num_basecurves))
    )


@lru_cache(maxsize=64)
def _curve_pair_plan(sample_counts: tuple[int, ...], num_basecurves: int) -> _CurvePairPlan:
    pairs = _curve_pairs(len(sample_counts), num_basecurves)
    class_members = _quadrature_classes(sample_counts)
    class_positions = {
        index: (cls, row)
        for cls, members in enumerate(class_members)
        for row, index in enumerate(members)
    }
    batch_rows: dict[tuple[int, int], tuple[list[int], list[int], list[tuple[int, int]]]] = {}
    for first, second in pairs:
        first_class, first_row = class_positions[first]
        second_class, second_row = class_positions[second]
        first_rows, second_rows, batch_pairs = batch_rows.setdefault(
            (first_class, second_class), ([], [], [])
        )
        first_rows.append(first_row)
        second_rows.append(second_row)
        batch_pairs.append((first, second))
    return _CurvePairPlan(
        class_members=class_members,
        batches=tuple(
            _CurvePairBatch(
                first_class=first_class,
                second_class=second_class,
                first_rows=tuple(first_rows),
                second_rows=tuple(second_rows),
                pairs=tuple(batch_pairs),
            )
            for (first_class, second_class), (first_rows, second_rows, batch_pairs) in (
                batch_rows.items()
            )
        ),
    )


def _flags(values) -> jax.Array:
    return _as_jax_array(np.asarray(values, dtype=bool), dtype=jnp.bool_)


def _curve_pair_penalty_total(class_gammas, class_gammadashes, minimum_distance, batch_candidates, plan):
    """Sum the curve-curve penalty over every pair of ``plan``, one vmap per batch.

    ``batch_candidates[b][k]`` is the native candidate decision of pair ``k`` of batch ``b``.
    """
    batched_kernel = jax.vmap(
        curve_curve_distance_penalty_pure, in_axes=(0, 0, 0, 0, None, 0)
    )
    total = jnp.zeros((), dtype=jnp.float64)
    for batch, candidates in zip(plan.batches, batch_candidates, strict=True):
        first_rows = np.asarray(batch.first_rows)
        second_rows = np.asarray(batch.second_rows)
        # The gather's gradient is a scatter-add: GPU dJ is bitwise reproducible
        # only under --xla_gpu_exclude_nondeterministic_ops=true.
        pair_values = batched_kernel(
            class_gammas[batch.first_class][first_rows],
            class_gammadashes[batch.first_class][first_rows],
            class_gammas[batch.second_class][second_rows],
            class_gammadashes[batch.second_class][second_rows],
            minimum_distance,
            candidates,
        )
        total = total + jnp.sum(pair_values)
    return total


_curve_pair_penalty = jax.jit(_curve_pair_penalty_total, static_argnames=("plan",))
_curve_pair_penalty_grad = jax.jit(
    jax.grad(_curve_pair_penalty_total, argnums=(0, 1)), static_argnames=("plan",)
)


class JaxCurveCurveDistance(Optimizable):
    r"""JAX version of :class:`simsopt.geo.CurveCurveDistance`: penalize close curve pairs,

    .. math::
        J = \sum_{i} \sum_{j < \min(i, N_b)} d_{i,j},
        \qquad
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j}
            \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i,

    where :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on curves :math:`i` and :math:`j`,
    :math:`d_{\min}` is ``minimum_distance`` and :math:`N_b` is ``num_basecurves``. Each double
    integral is the mean over every ``downsample``-th quadrature point of both curves,

    .. math::
        d_{i,j} \approx \frac{1}{n_i n_j} \sum_{k, l} |\Gamma_i'(t_k)| |\Gamma_j'(t_l)|
            \max(0, d_{\min} - \| \Gamma_i(t_k) - \Gamma_j(t_l) \|_2)^2,

    with :math:`n_i`, :math:`n_j` the numbers of sampled points. As in the native class, only
    pairs the native candidate search finds within :math:`d_{\min}` are evaluated; the others
    contribute zero.

    Args:
        curves (list[Curve]): all curves, including symmetry copies.
        minimum_distance (float): separation threshold :math:`d_{\min}` in m.
        num_basecurves (int or None): leading base-curve count :math:`N_b`; default None. None or
            zero selects all curves, as native does.
        downsample (int): quadrature stride for both candidate search and penalty; default 1.
    """

    def __init__(self, curves, minimum_distance, num_basecurves=None, downsample=1):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self.num_basecurves = num_basecurves or len(curves)
        self.downsample = downsample
        self._candidate_cache: _DistanceCandidates | None = None
        self._distance_snapshot: _CurveCurveSnapshot | None = None
        super().__init__(depends_on=curves)

    def _samples(self):
        return [_curve_position_samples(curve, self.downsample) for curve in self.curves]

    def _candidates(self, samples):
        """The native candidate pairs: ``simsoptpp``'s own search on the native samples."""
        return sopp.get_pointclouds_closer_than_threshold_within_collection(
            samples, self.minimum_distance, self.num_basecurves
        )

    def _candidate_snapshot(self) -> _DistanceCandidates:
        key = _distance_candidate_key(self.curves, self.minimum_distance) + (
            self.num_basecurves, self.downsample,
        )
        cached = self._candidate_cache
        if cached is not None and cached.key == key:
            return cached
        candidates = tuple(tuple(pair) for pair in self._candidates(self._samples()))
        snapshot = _DistanceCandidates(key, candidates)
        self._candidate_cache = snapshot
        return snapshot

    def _snapshot(self) -> _CurveCurveSnapshot:
        candidate_snapshot = self._candidate_snapshot()
        key = _distance_operand_key(self.curves, candidate_snapshot.key)
        cached = self._distance_snapshot
        if cached is not None and cached.key == key:
            return cached
        samples = self._samples()
        plan = _curve_pair_plan(
            tuple(int(sample.shape[0]) for sample in samples), self.num_basecurves
        )
        selected = set(candidate_snapshot.candidates)
        class_gammas, class_gammadashes = _class_stacked_geometry(
            self.curves, plan.class_members, self.downsample
        )
        batch_candidates = tuple(
            _flags([pair in selected for pair in batch.pairs]) for batch in plan.batches
        )
        snapshot = _CurveCurveSnapshot(
            key, plan,
            (class_gammas, class_gammadashes, _as_jax_float64(self.minimum_distance), batch_candidates),
        )
        self._distance_snapshot = snapshot
        return snapshot

    def _operands(self):
        snapshot = self._snapshot()
        return snapshot.plan, snapshot.operands

    def shortest_distance(self):
        """Find the native sampled shortest distance.

        With candidates, return the minimum over candidates and the threshold;
        without candidates, search all eligible geometry.

        Returns:
            float: sampled distance in m.
        """
        samples = self._samples()
        candidates = self._candidate_snapshot().candidates
        pairs = candidates or _curve_pairs(len(samples), len(samples))
        distances = [np.min(cdist(samples[i], samples[j])) for i, j in pairs]
        return min([self.minimum_distance] + distances) if candidates else min(distances)

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m^4.
        """
        plan, operands = self._operands()
        return _host_float(_curve_pair_penalty(*operands, plan=plan))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        plan, operands = self._operands()
        class_dgammas, class_dgammadashes = _curve_pair_penalty_grad(*operands, plan=plan)
        return _class_geometry_derivative(
            self.curves,
            plan.class_members,
            class_dgammas,
            class_dgammadashes,
            self.downsample,
        )

    return_fn_map = {"J": J, "dJ": dJ}


def _curve_surface_penalty_total(
    class_gammas, class_gammadashes, class_candidates, surface_gamma, surface_normal, minimum_distance
):
    """Sum the curve-surface penalty over the curves of every quadrature class.

    ``class_candidates[c][k]`` is the native candidate decision of curve ``k`` of class ``c``.
    """
    batched_kernel = jax.vmap(
        curve_surface_distance_penalty_pure, in_axes=(0, 0, None, None, None, 0)
    )
    total = jnp.zeros((), dtype=jnp.float64)
    for gammas, gammadashes, candidates in zip(
        class_gammas, class_gammadashes, class_candidates, strict=True
    ):
        total = total + jnp.sum(
            batched_kernel(
                gammas, gammadashes, surface_gamma, surface_normal, minimum_distance, candidates
            )
        )
    return total


_curve_surface_penalty = jax.jit(_curve_surface_penalty_total)
_curve_surface_penalty_grad = jax.jit(
    jax.grad(_curve_surface_penalty_total, argnums=(0, 1))
)


class JaxCurveSurfaceDistance(Optimizable):
    r"""JAX version of :class:`simsopt.geo.CurveSurfaceDistance`: penalize curves near a surface,

    .. math::
        J = \sum_{i} d_{i},
        \qquad
        d_{i} = \int_{\text{curve}_i} \int_{S}
            \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds,

    where :math:`\mathbf{r}_i` and :math:`\mathbf{s}` are points on curve :math:`i` and the surface
    and :math:`d_{\min}` is ``minimum_distance``. The surface element uses the unnormalized
    normal :math:`\mathbf{N}`, :math:`ds = |\mathbf{N}|~d\varphi~d\theta`, and each integral is
    the mean over the quadrature points. As in the native class, only curves the native
    candidate search finds within :math:`d_{\min}` are evaluated.

    The surface is not a dependency: only curve DOFs receive derivatives, and the surface
    geometry is read at each evaluation.

    Args:
        curves (list[Curve]): all curves to penalize.
        surface (Surface): surface with positions in m and unnormalized normals in m^2.
        minimum_distance (float): separation threshold :math:`d_{\min}` in m.
    """

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance
        self._candidate_cache: _DistanceCandidates | None = None
        self._distance_snapshot: _CurveSurfaceSnapshot | None = None
        super().__init__(depends_on=curves)

    def _candidates(self, surface_points):
        """The native candidate curves: ``simsoptpp``'s own search on the native samples."""
        return sopp.get_pointclouds_closer_than_threshold_between_two_collections(
            [curve.gamma() for curve in self.curves], [surface_points], self.minimum_distance
        )

    def _candidate_snapshot(self) -> _DistanceCandidates:
        key = _distance_candidate_key(self.curves, self.minimum_distance) + (
            _array_snapshot_key(self.surface.gamma()),
            _array_snapshot_key(self.surface.quadpoints_phi),
            _array_snapshot_key(self.surface.quadpoints_theta),
        )
        cached = self._candidate_cache
        if cached is not None and cached.key == key:
            return cached
        surface_points = self.surface.gamma().reshape((-1, 3))
        candidates = tuple(tuple(pair) for pair in self._candidates(surface_points))
        snapshot = _DistanceCandidates(key, candidates)
        self._candidate_cache = snapshot
        return snapshot

    def _snapshot(self) -> _CurveSurfaceSnapshot:
        candidate_snapshot = self._candidate_snapshot()
        key = _distance_operand_key(self.curves, candidate_snapshot.key) + (
            _array_snapshot_key(self.surface.normal()),
        )
        cached = self._distance_snapshot
        if cached is not None and cached.key == key:
            return cached
        surface_points = self.surface.gamma().reshape((-1, 3))
        class_members = _quadrature_classes(
            tuple(int(curve.gamma().shape[0]) for curve in self.curves)
        )
        selected = {i for i, _ in candidate_snapshot.candidates}
        class_gammas, class_gammadashes = _class_stacked_geometry(
            self.curves, class_members, 1
        )
        operands = (
            class_gammas,
            class_gammadashes,
            tuple(_flags([index in selected for index in members]) for members in class_members),
            _as_jax_float64(surface_points),
            _as_jax_float64(self.surface.normal().reshape((-1, 3))),
            _as_jax_float64(self.minimum_distance),
        )
        snapshot = _CurveSurfaceSnapshot(key, class_members, operands)
        self._distance_snapshot = snapshot
        return snapshot

    def _operands(self):
        snapshot = self._snapshot()
        return snapshot.class_members, snapshot.operands

    def shortest_distance(self):
        """Find the native sampled shortest distance.

        With candidates, return the minimum over candidates and the threshold;
        without candidates, search all eligible geometry.

        Returns:
            float: sampled distance in m.
        """
        surface_points = self.surface.gamma().reshape((-1, 3))
        candidates = self._candidate_snapshot().candidates
        indices = [i for i, _ in candidates] or range(len(self.curves))
        distances = [np.min(cdist(self.curves[i].gamma(), surface_points)) for i in indices]
        return min([self.minimum_distance] + distances) if candidates else min(distances)

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m^5.
        """
        _, operands = self._operands()
        return _host_float(_curve_surface_penalty(*operands))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        class_members, operands = self._operands()
        class_dgammas, class_dgammadashes = _curve_surface_penalty_grad(*operands)
        return _class_geometry_derivative(
            self.curves, class_members, class_dgammas, class_dgammadashes, 1
        )

    return_fn_map = {"J": J, "dJ": dJ}
