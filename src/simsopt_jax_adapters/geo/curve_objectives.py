"""JAX coil-geometry penalties with native Optimizable and distance-query APIs.

Native searches select distance candidates. Geometry moves to the active JAX
device and gradients to the host; native JAX curves may transfer implicitly.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
import simsoptpp as sopp
from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt_jax.backend.jit import jit
from simsopt_jax.backend.runtime import get_backend_config, get_runtime_jax_device, host_array, host_tree
from simsopt_jax.core._math_utils import _next_power_of_two
from simsopt_jax.core._math_utils import as_jax_array as _as_jax_array
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.core._math_utils import as_jax_int32 as _as_jax_int32
from simsopt_jax.core.curve_kernels import (
    curvature_p_norm_from_kappa_pure,
    curve_curve_distance_penalty_pure,
    curve_length_from_incremental_arclength_pure,
    curve_surface_distance_penalty_pure,
    kappa_pure,
    mean_squared_curvature_pure,
)


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


_curve_length = jit(_length_from_tangent)
_curve_length_grad = jit(jax.grad(_length_from_tangent))
_lp_curvature = jit(_lp_curvature_from_geometry)
_lp_curvature_grad = jit(jax.grad(_lp_curvature_from_geometry, argnums=(0, 1)))
_mean_squared_curvature = jit(_mean_squared_curvature_from_geometry)
_mean_squared_curvature_grad = jit(
    jax.grad(_mean_squared_curvature_from_geometry, argnums=(0, 1))
)


def _curve_position_samples(curve, downsample=1):
    gamma = curve.gamma()
    return gamma if downsample == 1 else gamma[::downsample]


def _sum_curve_vjp_contributions(curves, dgamma_vjps, dgammadash_vjps):
    return sum(
        curve.dgamma_by_dcoeff_vjp(dgamma_vjp)
        + curve.dgammadash_by_dcoeff_vjp(dgammadash_vjp)
        for curve, dgamma_vjp, dgammadash_vjp in zip(
            curves, dgamma_vjps, dgammadash_vjps
        )
    )


def _tangents(curve):
    """Place a curve's first and second parameter derivatives on the active device."""
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


def _class_stacked_geometry(curves, class_members):
    """Return device stacks of full-grid ``gamma``/``gammadash`` per quadrature class."""

    def _stack(samples):
        return tuple(
            _as_jax_float64(
                np.stack([host_array(samples(curves[index]), dtype=np.float64) for index in members])
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


def _distance_operand_key(curves, candidates: _DistanceCandidates) -> tuple[object, ...]:
    """Identify cached operands by candidates, geometry, backend and default-device scope."""
    return candidates.key + (
        candidates.candidates,
        get_backend_config(), get_runtime_jax_device(), jax.config.values["jax_default_device"],
        tuple(_array_snapshot_key(curve.gammadash()) for curve in curves),
    )


def _class_geometry_derivative(curves, class_members, class_dgammas, class_dgammadashes):
    """Return curve coefficient derivatives from per-class geometry cotangents."""
    class_dgammas, class_dgammadashes = host_tree(
        (class_dgammas, class_dgammadashes), dtype=np.float64
    )
    positions = _class_positions(class_members)
    rows = [positions[index] for index in range(len(curves))]
    return _sum_curve_vjp_contributions(
        curves,
        [class_dgammas[cls][row] for cls, row in rows],
        [class_dgammadashes[cls][row] for cls, row in rows],
    )


class JaxCurveLength(Optimizable):
    r"""Compute curve length by averaging speed on a unit-period quadrature grid.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
    """

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        """Evaluate the objective on the active JAX device.

        Returns:
            float: objective value in m.
        """
        return _host_float(_curve_length(_as_jax_float64(self.curve.gammadash())))

    @derivative_dec
    def dJ(self):
        """Return the curve DOF gradient, accumulating shared DOFs.

        ``partials=True`` includes fixed DOF partials.

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
    r"""Penalize curvature above a threshold with an arclength-weighted Lp integral.

    Scale the integral by 1/p, without taking a p-th root.

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
        """Evaluate the objective on the active JAX device.

        Returns:
            float: objective value in m^(1-p).
        """
        return _host_float(_lp_curvature(*_tangents(self.curve), *self._parameters()))

    @derivative_dec
    def dJ(self):
        """Return the curve DOF gradient, accumulating shared DOFs.

        ``partials=True`` includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return _tangent_derivative(
            self.curve, *_lp_curvature_grad(*_tangents(self.curve), *self._parameters())
        )

    return_fn_map = {"J": J, "dJ": dJ}


class JaxMeanSquaredCurvature(Optimizable):
    r"""Compute the arclength-averaged squared curvature.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
    """

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        """Evaluate the objective on the active JAX device.

        Returns:
            float: objective value in 1/m^2.
        """
        return _host_float(_mean_squared_curvature(*_tangents(self.curve)))

    @derivative_dec
    def dJ(self):
        """Return the curve DOF gradient, accumulating shared DOFs.

        ``partials=True`` includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return _tangent_derivative(
            self.curve, *_mean_squared_curvature_grad(*_tangents(self.curve))
        )

    return_fn_map = {"J": J, "dJ": dJ}


@dataclass(frozen=True)
class _DistanceCandidates:
    """Candidate pairs for given search inputs; recompute notifications discard assignments."""

    key: tuple[object, ...]
    candidates: tuple[tuple[int, int], ...]
    assigned: bool = False


@dataclass(frozen=True)
class _DistanceOperands:
    """Placed candidate operands with static per-batch quadrature classes."""

    key: tuple[object, ...]
    class_members: tuple[tuple[int, ...], ...]
    batch_classes: tuple[object, ...]
    operands: tuple[object, ...]


def _class_positions(class_members) -> dict[int, tuple[int, int]]:
    """Map each curve index to its (quadrature class, row in the class stack)."""
    return {
        index: (cls, row)
        for cls, members in enumerate(class_members)
        for row, index in enumerate(members)
    }


def _padded_candidate_batches(groups):
    """Return sorted candidate groups padded to power-of-two lengths.

    Padding copies the first row and marks it invalid.

    Args:
        groups (dict): maps a sortable batch key to a non-empty list of row
            tuples of one length ``k``.

    Returns:
        tuple: ``(keys, batches)``: the sorted group keys, and per group ``k``
            int32 index arrays and one bool validity array, each of the padded
            length.
    """
    keys = tuple(sorted(groups))
    batches = []
    for rows in (groups[key] for key in keys):
        padded = np.zeros((_next_power_of_two(len(rows)), len(rows[0])), dtype=np.int32)
        padded[: len(rows)] = rows
        batches.append((
            *(_as_jax_int32(column) for column in padded.T),
            _as_jax_array(np.arange(padded.shape[0]) < len(rows), dtype=jnp.bool_),
        ))
    return keys, tuple(batches)


def _zero_curve_derivative(curves):
    """Return zero coefficient derivatives for every curve when no candidates exist."""
    return _sum_curve_vjp_contributions(
        curves,
        [np.zeros_like(curve.gamma()) for curve in curves],
        [np.zeros_like(curve.gammadash()) for curve in curves],
    )


class _CandidateDistance(Optimizable):
    """Native's candidate API and the placed-operand cache of the distance adapters."""

    def __init__(self, curves, minimum_distance):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self._candidate_cache: _DistanceCandidates | None = None
        self._distance_snapshot: _DistanceOperands | None = None
        super().__init__(depends_on=curves)

    @abstractmethod
    def _candidates(self) -> list:
        """The native search's candidate pairs at the current state."""

    @abstractmethod
    def _candidate_key(self) -> tuple[object, ...]:
        """The exact inputs of the native search, without resolving a backend or placing arrays."""

    @abstractmethod
    def _batch_groups(self, candidates, positions) -> dict:
        """Group the sorted candidates' class-stack rows by batch key."""

    @abstractmethod
    def _shortest_among(self, candidates) -> float | np.floating:
        """The smaller of ``minimum_distance`` and the candidates' shortest sampled distance."""

    @abstractmethod
    def _shortest_without_candidates(self) -> float | np.floating:
        """Native's shortest sampled distance when no pair is a candidate."""

    @abstractmethod
    def _value(self, snapshot: _DistanceOperands) -> jax.Array:
        """Run the penalty program on ``snapshot``."""

    @abstractmethod
    def _gradient(self, snapshot: _DistanceOperands) -> tuple:
        """Run the penalty's gradient program in the class stacks on ``snapshot``."""

    def _operand_key_extra(self) -> tuple[object, ...]:
        return ()

    def _extra_operands(self) -> tuple[object, ...]:
        return ()

    def _cached_candidates(self, key: tuple[object, ...]) -> _DistanceCandidates | None:
        """The cached candidates while the search inputs still equal ``key``."""
        cached = self._candidate_cache
        return cached if cached is not None and cached.key == key else None

    def _candidate_snapshot(self) -> _DistanceCandidates:
        key = self._candidate_key()
        cached = self._cached_candidates(key)
        if cached is not None:
            return cached
        snapshot = _DistanceCandidates(key, tuple(tuple(pair) for pair in self._candidates()))
        self._candidate_cache = snapshot
        return snapshot

    def _snapshot(self) -> _DistanceOperands | None:
        """Return placed candidate operands, or None and release cached operands if empty."""
        candidates = self._candidate_snapshot()
        if not candidates.candidates:
            self._distance_snapshot = None
            return None
        key = _distance_operand_key(self.curves, candidates) + self._operand_key_extra()
        cached = self._distance_snapshot
        if cached is not None and cached.key == key:
            return cached
        class_members = _quadrature_classes(tuple(int(curve.gamma().shape[0]) for curve in self.curves))
        batch_classes, batches = _padded_candidate_batches(
            self._batch_groups(candidates.candidates, _class_positions(class_members))
        )
        class_gammas, class_gammadashes = _class_stacked_geometry(self.curves, class_members)
        operands = (
            class_gammas, class_gammadashes, batches, _as_jax_float64(self.minimum_distance),
            *self._extra_operands(),
        )
        snapshot = _DistanceOperands(key, class_members, batch_classes, operands)
        self._distance_snapshot = snapshot
        return snapshot

    @property
    def candidates(self):
        """Return the current candidate pairs.

        Assign pairs to override the search until a recompute notification or a
        change in geometry, threshold or search settings; assign None to search
        again.

        Returns:
            list[tuple[int, int]]: a copy of the pairs, in no fixed order: curve pairs
                ``(i, j)`` with some sampled points closer than ``minimum_distance``, or, for
                the surface distance, ``(i, 0)`` for a curve ``i`` with a sampled point that
                close to the surface.
        """
        return list(self._candidate_snapshot().candidates)

    @candidates.setter
    def candidates(self, pairs):
        self._candidate_cache = None if pairs is None else _DistanceCandidates(
            self._candidate_key(), tuple(tuple(pair) for pair in pairs), assigned=True
        )

    @property
    def candidates_assigned(self) -> bool:
        """Whether assigned :attr:`candidates` replace the native search at the current state.

        Returns:
            bool: True from an assignment until a recompute notification or a change in geometry,
                threshold or search settings.
        """
        cached = self._candidate_cache
        return cached is not None and cached.assigned and self._cached_candidates(self._candidate_key()) is cached

    def recompute_bell(self, parent=None):
        """Discard assigned candidates, retaining search results cached by search inputs.

        Args:
            parent (Optimizable or None): Dependency that sent the notification.
        """
        if self._candidate_cache is not None and self._candidate_cache.assigned:
            self._candidate_cache = None

    def compute_candidates(self):
        """Run the native candidate search unless its result for the current state is cached."""
        self._candidate_snapshot()

    def shortest_distance_among_candidates(self):
        """Return the sampled shortest distance over candidates, capped by the threshold.

        Returns:
            float: the smaller of ``minimum_distance`` and the candidates' shortest sampled
                distance, in m.
        """
        return self._shortest_among(self._candidate_snapshot().candidates)

    def shortest_distance(self):
        """Return the sampled shortest distance.

        With candidates, cap the distance by the threshold; otherwise search all
        curve pairs or all curves against the surface.

        Returns:
            float: sampled distance in m.
        """
        candidates = self._candidate_snapshot().candidates
        return self._shortest_among(candidates) if candidates else self._shortest_without_candidates()

    def J(self):
        """Evaluate the objective on the active JAX device.

        Returns:
            float: objective value in m^4 (curve-curve) or m^5 (curve-surface).
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return 0.0
        return _host_float(self._value(snapshot))

    @derivative_dec
    def dJ(self):
        """Return the curve DOF gradient, accumulating shared DOFs.

        ``partials=True`` includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return _zero_curve_derivative(self.curves)
        class_dgammas, class_dgammadashes = self._gradient(snapshot)
        return _class_geometry_derivative(
            self.curves, snapshot.class_members, class_dgammas, class_dgammadashes
        )

    return_fn_map = {"J": J, "dJ": dJ}


def _curve_pair_penalty_total(
    class_gammas, class_gammadashes, batches, minimum_distance, batch_classes, downsample
):
    """Sum candidate curve-pair penalties on downsampled quadrature grids."""
    class_gammas = tuple(gammas[:, ::downsample] for gammas in class_gammas)
    class_gammadashes = tuple(gammadashes[:, ::downsample] for gammadashes in class_gammadashes)
    batched_kernel = jax.vmap(
        curve_curve_distance_penalty_pure, in_axes=(0, 0, 0, 0, None, 0)
    )
    total = jnp.zeros((), dtype=jnp.float64)
    for (first_class, second_class), (first_rows, second_rows, valid) in zip(
        batch_classes, batches, strict=True
    ):
        # The gather's gradient is a scatter-add: GPU dJ is bitwise reproducible
        # only under --xla_gpu_exclude_nondeterministic_ops=true.
        total = total + jnp.sum(batched_kernel(
            class_gammas[first_class][first_rows],
            class_gammadashes[first_class][first_rows],
            class_gammas[second_class][second_rows],
            class_gammadashes[second_class][second_rows],
            minimum_distance,
            valid,
        ))
    return total


_curve_pair_penalty = jit(_curve_pair_penalty_total, static_argnames=("batch_classes", "downsample"))
_curve_pair_penalty_grad = jit(
    jax.grad(_curve_pair_penalty_total, argnums=(0, 1)),
    static_argnames=("batch_classes", "downsample"),
)


class JaxCurveCurveDistance(_CandidateDistance):
    r"""Penalize curve pairs closer than a minimum separation.

    Sum the arclength-weighted integrals of squared distance deficits for
    pairs containing a base curve. Quadrature averages over sampled point
    pairs; only native-search candidates contribute. Candidate assignments
    and distance queries follow :attr:`candidates`.

    Args:
        curves (list[Curve]): all curves, including symmetry copies.
        minimum_distance (float): separation threshold :math:`d_{\min}` in m.
        num_basecurves (int or None): leading base-curve count :math:`N_b`; default None. None or
            zero selects all curves, as native does.
        downsample (int): quadrature stride for both candidate search and penalty; default 1.
            As in native, a stride that does not divide a curve's quadrature count is accepted
            and samples that curve nonuniformly.
    """

    def __init__(self, curves, minimum_distance, num_basecurves=None, downsample=1):
        self.num_basecurves = num_basecurves or len(curves)
        self.downsample = downsample
        super().__init__(curves, minimum_distance)

    def _samples(self):
        return [_curve_position_samples(curve, self.downsample) for curve in self.curves]

    def _candidates(self):
        """The native candidate pairs: ``simsoptpp``'s own search on the native samples."""
        return sopp.get_pointclouds_closer_than_threshold_within_collection(
            self._samples(), self.minimum_distance, self.num_basecurves
        )

    def _candidate_key(self) -> tuple[object, ...]:
        return _distance_candidate_key(self.curves, self.minimum_distance) + (
            self.num_basecurves, self.downsample,
        )

    def _batch_groups(self, candidates, positions):
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for first, second in sorted(candidates):
            (first_class, first_row), (second_class, second_row) = positions[first], positions[second]
            groups.setdefault((first_class, second_class), []).append((first_row, second_row))
        return groups

    def _shortest_among(self, candidates):
        samples = self._samples()
        return min([self.minimum_distance] + [np.min(cdist(samples[i], samples[j])) for i, j in candidates])

    def _shortest_without_candidates(self):
        samples = self._samples()
        return min(
            np.min(cdist(samples[i], samples[j])) for i, j in combinations(range(len(samples)), 2)
        )

    def _value(self, snapshot):
        return _curve_pair_penalty(*snapshot.operands, batch_classes=snapshot.batch_classes, downsample=self.downsample)

    def _gradient(self, snapshot):
        return _curve_pair_penalty_grad(
            *snapshot.operands, batch_classes=snapshot.batch_classes, downsample=self.downsample
        )


def _curve_surface_penalty_total(
    class_gammas, class_gammadashes, batches, minimum_distance, surface_gamma, surface_normal,
    batch_classes,
):
    """Sum curve-surface penalties over candidate curves."""
    batched_kernel = jax.vmap(
        curve_surface_distance_penalty_pure, in_axes=(0, 0, None, None, None, 0)
    )
    total = jnp.zeros((), dtype=jnp.float64)
    for cls, (rows, valid) in zip(batch_classes, batches, strict=True):
        total = total + jnp.sum(batched_kernel(
            class_gammas[cls][rows], class_gammadashes[cls][rows],
            surface_gamma, surface_normal, minimum_distance, valid,
        ))
    return total


_curve_surface_penalty = jit(_curve_surface_penalty_total, static_argnames=("batch_classes",))
_curve_surface_penalty_grad = jit(
    jax.grad(_curve_surface_penalty_total, argnums=(0, 1)), static_argnames=("batch_classes",)
)


class JaxCurveSurfaceDistance(_CandidateDistance):
    r"""Penalize curves closer than a minimum distance to a surface.

    Sum squared distance deficits integrated over curve arclength and surface
    area, averaging over quadrature points with unnormalized normal weights.
    Only native-search candidates contribute; see :attr:`candidates`.
    The surface is read at each evaluation and is not a dependency; only
    curve DOFs receive derivatives.

    Args:
        curves (list[Curve]): all curves to penalize.
        surface (Surface): surface with positions in m and unnormalized normals in m^2.
        minimum_distance (float): separation threshold :math:`d_{\min}` in m.
    """

    def __init__(self, curves, surface, minimum_distance):
        self.surface = surface
        super().__init__(curves, minimum_distance)

    def _surface_points(self):
        return self.surface.gamma().reshape((-1, 3))

    def _candidates(self):
        """The native candidate curves: ``simsoptpp``'s own search on the native samples."""
        return sopp.get_pointclouds_closer_than_threshold_between_two_collections(
            [curve.gamma() for curve in self.curves], [self._surface_points()], self.minimum_distance
        )

    def _candidate_key(self) -> tuple[object, ...]:
        return _distance_candidate_key(self.curves, self.minimum_distance) + (
            _array_snapshot_key(self.surface.gamma()),
            _array_snapshot_key(self.surface.quadpoints_phi),
            _array_snapshot_key(self.surface.quadpoints_theta),
        )

    def _operand_key_extra(self):
        return (_array_snapshot_key(self.surface.normal()),)

    def _extra_operands(self):
        return (
            _as_jax_float64(self._surface_points()),
            _as_jax_float64(self.surface.normal().reshape((-1, 3))),
        )

    def _batch_groups(self, candidates, positions):
        groups: dict[int, list[tuple[int]]] = {}
        for index, _ in sorted(candidates):
            cls, row = positions[index]
            groups.setdefault(cls, []).append((row,))
        return groups

    def _shortest_among(self, candidates):
        surface_points = self._surface_points()
        return min([self.minimum_distance] + [
            np.min(cdist(self.curves[i].gamma(), surface_points)) for i, _ in candidates
        ])

    def _shortest_without_candidates(self):
        surface_points = self._surface_points()
        return min(np.min(cdist(curve.gamma(), surface_points)) for curve in self.curves)

    def _value(self, snapshot):
        return _curve_surface_penalty(*snapshot.operands, batch_classes=snapshot.batch_classes)

    def _gradient(self, snapshot):
        return _curve_surface_penalty_grad(*snapshot.operands, batch_classes=snapshot.batch_classes)
