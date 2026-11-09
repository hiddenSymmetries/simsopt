"""Pure JAX coil-geometry penalty kernels.

Each kernel evaluates the formula of the matching native objective in
:mod:`simsopt.geo.curveobjectives` on sampled curve geometry: positions
``gamma`` and tangents ``gammadash`` with shape ``(nquadpoints, 3)``. Distance
penalties keep the native evaluation boundary: native evaluates the dense
formula only for candidate pairs (``simsoptpp``'s candidate search: some point
pair closer than the minimum distance) and skips every other pair. The caller
passes that decision: the drop-in objectives take it from ``simsoptpp`` itself,
the fused objective from :func:`distance_candidate_pure`. Each public boundary
converts floating inputs to float64, as native evaluates in double: caller
NumPy arrays are snapshotted and placed, device arrays and tracers are cast.
float64 needs JAX's 64-bit mode (``jax_enable_x64``), which importing
:mod:`simsopt_jax` (and so this module) or :mod:`simsopt.geo` turns on; if it
is switched off again, JAX computes in float32.
"""

from __future__ import annotations

import jax.numpy as jnp

from simsopt_jax.backend.jit import jit
from simsopt_jax.runtime.host_boundary import snapshot_host_tree

from ._device_scalars import staged_like
from ._math_utils import as_jax_float64

__all__ = [
    "curvature_p_norm_from_kappa_pure",
    "curve_curve_distance_penalty_pure",
    "curve_length_from_incremental_arclength_pure",
    "curve_surface_distance_penalty_pure",
    "distance_candidate_pure",
    "kappa_pure",
    "mean_squared_curvature_pure",
]


def distance_candidate_pure(points1, points2, minimum_distance):
    """Test whether any sampled point pair is strictly closer than the threshold.

    The squared-distance comparison mirrors the native candidate search; fused
    multiply-add rounding can change classification exactly at the threshold.

    Args:
        points1: Array of shape (n, 3), positions in m.
        points2: Array of shape (k, 3), positions in m.
        minimum_distance: float scalar, distance threshold in m.

    Returns:
        Array: bool scalar candidate decision.
    """
    points1, points2 = map(as_jax_float64, (points1, points2))
    minimum_distance = snapshot_host_tree(minimum_distance)
    delta = points1[:, None, :] - points2[None, :, :]
    squared_distances = (
        delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1] + delta[..., 2] * delta[..., 2]
    )
    minimum_distance = staged_like(points1, minimum_distance)
    return jnp.any(squared_distances < minimum_distance * minimum_distance)


def _candidate_pair_penalty(points1, weights1, points2, weights2, minimum_distance, candidate, reduce):
    """``reduce(|weights1_i| |weights2_j| max(d_min - |points1_i - points2_j|, 0)^2)``
    for a native candidate pair, else 0 with a zero gradient.

    Native never evaluates non-candidate pairs, so the double ``where``
    replaces their inputs before every singular operation (``sqrt`` and norms
    at zero); within a candidate pair the formula is native's, singular
    gradients included.
    """
    zero = staged_like(points1, 0.0)
    one = staged_like(points1, 1.0)
    minimum_distance = staged_like(points1, minimum_distance)
    delta = points1[:, None, :] - points2[None, :, :]
    squared_distances = jnp.sum(jnp.square(delta), axis=2)
    distances = jnp.sqrt(jnp.where(candidate, squared_distances, one))
    weight = (
        jnp.linalg.norm(jnp.where(candidate, weights1, one), axis=1)[:, None]
        * jnp.linalg.norm(jnp.where(candidate, weights2, one), axis=1)[None, :]
    )
    penalty = reduce(weight * jnp.square(jnp.maximum(minimum_distance - distances, zero)))
    return jnp.where(candidate, penalty, zero)


def curve_length_from_incremental_arclength_pure(incremental_arclength):
    """Compute length by averaging the speed over a unit-period quadrature grid.

    Args:
        incremental_arclength: Array of shape (n,), speed in m per unit parameter.

    Returns:
        Array: scalar curve length in m.
    """
    return _curve_length_from_incremental_arclength_pure(as_jax_float64(incremental_arclength))


@jit
def _curve_length_from_incremental_arclength_pure(incremental_arclength):
    """Compiled formula on float64 operands from the public boundary."""
    return jnp.mean(incremental_arclength)


def kappa_pure(d1gamma, d2gamma):
    """Compute curvature as |gamma prime cross gamma double prime| / |gamma prime|^3.

    Args:
        d1gamma: Array of shape (n, 3), derivative in m with respect to the unit-period parameter.
        d2gamma: Array of shape (n, 3), second parameter derivative in m.

    Returns:
        Array of shape (n,): curvature in 1/m.
    """
    return _kappa_pure(*map(as_jax_float64, (d1gamma, d2gamma)))


@jit
def _kappa_pure(d1gamma, d2gamma):
    """Compiled formula on float64 operands from the public boundary."""
    return (
        jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)
        / jnp.linalg.norm(d1gamma, axis=1) ** 3
    )


def curvature_p_norm_from_kappa_pure(kappa, gammadash, p, desired_kappa):
    """Compute (1/p) mean(max(kappa - desired_kappa, 0)^p |gammadash|).

    This is the native integral penalty without a p-th root.

    Args:
        kappa: Array of shape (n,), curvature in 1/m.
        gammadash: Array of shape (n, 3), derivative in m with respect to the unit-period parameter.
        p: float scalar, dimensionless exponent.
        desired_kappa: float scalar, curvature threshold in 1/m.

    Returns:
        Array: scalar penalty in m^(1-p).
    """
    return _curvature_p_norm_from_kappa_pure(*map(as_jax_float64, (kappa, gammadash, p, desired_kappa)))


@jit
def _curvature_p_norm_from_kappa_pure(kappa, gammadash, p, desired_kappa):
    """Compiled formula on float64 operands from the public boundary."""
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(kappa - desired_kappa, zero)
    return (one / p) * jnp.mean((excess**p) * arc_length)


def mean_squared_curvature_pure(kappa, gammadash):
    """Compute mean(kappa^2 |gammadash|) / mean(|gammadash|).

    Args:
        kappa: Array of shape (n,), curvature in 1/m.
        gammadash: Array of shape (n, 3), derivative in m with respect to the unit-period parameter.

    Returns:
        Array: scalar arclength-averaged squared curvature in 1/m^2.
    """
    return _mean_squared_curvature_pure(*map(as_jax_float64, (kappa, gammadash)))


@jit
def _mean_squared_curvature_pure(kappa, gammadash):
    """Compiled formula on float64 operands from the public boundary."""
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length) / jnp.mean(arc_length)


def curve_curve_distance_penalty_pure(
    gamma1,
    gammadash1,
    gamma2,
    gammadash2,
    minimum_distance,
    candidate,
):
    """Evaluate one native curve-pair distance penalty.

    The term is mean(|gammadash1_i| |gammadash2_j|
    max(minimum_distance - |gamma1_i - gamma2_j|, 0)^2) over all point pairs
    of a candidate pair; otherwise it is zero.

    Args:
        gamma1: Array of shape (n, 3), positions in m.
        gammadash1: Array of shape (n, 3), derivative in m with respect to the unit-period parameter.
        gamma2: Array of shape (k, 3), second curve positions in m.
        gammadash2: Array of shape (k, 3), second curve parameter derivative in m.
        minimum_distance: float scalar, distance threshold in m.
        candidate: bool scalar, native candidate-search decision; false gives zero value and gradient.

    Returns:
        Array: scalar penalty in m^4.
    """
    gamma1, gammadash1, gamma2, gammadash2 = map(as_jax_float64, (gamma1, gammadash1, gamma2, gammadash2))
    minimum_distance, candidate = snapshot_host_tree((minimum_distance, candidate))
    normalization = staged_like(gamma1, int(gamma1.shape[0]) * int(gamma2.shape[0]))
    return _candidate_pair_penalty(
        gamma1, gammadash1, gamma2, gammadash2, minimum_distance, candidate,
        lambda terms: jnp.sum(terms) / normalization,
    )


def curve_surface_distance_penalty_pure(
    curve_gamma,
    curve_gammadash,
    surface_gamma,
    surface_normal,
    minimum_distance,
    candidate,
):
    """Evaluate one native curve-surface distance penalty.

    The term is mean(|curve_gammadash_i| |surface_normal_j|
    max(minimum_distance - |curve_gamma_i - surface_gamma_j|, 0)^2)
    over all point pairs of a candidate curve; otherwise it is zero.

    Args:
        curve_gamma: Array of shape (n, 3), positions in m.
        curve_gammadash: Array of shape (n, 3), derivative in m with respect to the unit-period parameter.
        surface_gamma: Array of shape (k, 3), flattened surface positions in m.
        surface_normal: Array of shape (k, 3), unnormalized surface normals in m^2.
        minimum_distance: float scalar, distance threshold in m.
        candidate: bool scalar, native candidate-search decision; false gives zero value and gradient.

    Returns:
        Array: scalar penalty in m^5.
    """
    curve_gamma, curve_gammadash, surface_gamma, surface_normal = map(
        as_jax_float64, (curve_gamma, curve_gammadash, surface_gamma, surface_normal)
    )
    minimum_distance, candidate = snapshot_host_tree((minimum_distance, candidate))
    return _candidate_pair_penalty(
        curve_gamma, curve_gammadash, surface_gamma, surface_normal, minimum_distance, candidate,
        jnp.mean,
    )
