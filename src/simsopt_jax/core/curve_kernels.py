"""Pure JAX coil-geometry penalty kernels.

Each kernel evaluates the formula of the matching native objective in
:mod:`simsopt.geo.curveobjectives` on sampled curve geometry: positions
``gamma`` and tangents ``gammadash`` with shape ``(nquadpoints, 3)``. Distance
penalties keep the native evaluation boundary: native evaluates the dense
formula only for candidate pairs (``simsoptpp``'s candidate search: some point
pair closer than the minimum distance) and skips every other pair. The caller
passes that decision: the drop-in objectives take it from ``simsoptpp`` itself,
the fused objective from :func:`distance_candidate_pure`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._device_scalars import staged_like

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
    """Whether two points of ``points1`` and ``points2`` are closer than ``d_min``.

    The final test of ``simsoptpp``'s candidate search, in its expression:
    ``dx*dx + dy*dy + dz*dz < d_min*d_min`` with ``d = points1_i - points2_j``.
    Where its compiler, or XLA, contracts these products into fused
    multiply-adds, the last bit of the squared distance, and so a pair at
    exactly the threshold, can be classified differently from the native build.
    """
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


@jax.jit
def curve_length_from_incremental_arclength_pure(incremental_arclength):
    """Return the curve length, the mean of the incremental arclengths."""
    return jnp.mean(incremental_arclength)


@jax.jit
def kappa_pure(d1gamma, d2gamma):
    """Return pointwise curvature for first and second curve derivatives."""
    return (
        jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)
        / jnp.linalg.norm(d1gamma, axis=1) ** 3
    )


@jax.jit
def curvature_p_norm_from_kappa_pure(kappa, gammadash, p, desired_kappa):
    """Return ``(1/p) mean(max(kappa - desired_kappa, 0)^p |gammadash|)`` (LpCurveCurvature)."""
    p_jax = jnp.asarray(p, dtype=kappa.dtype)
    desired_kappa_jax = jnp.asarray(desired_kappa, dtype=kappa.dtype)
    zero = jnp.asarray(0.0, dtype=kappa.dtype)
    one = jnp.asarray(1.0, dtype=kappa.dtype)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(kappa - desired_kappa_jax, zero)
    return (one / p_jax) * jnp.mean((excess**p_jax) * arc_length)


@jax.jit
def mean_squared_curvature_pure(kappa, gammadash):
    """Return ``mean(kappa^2 |gammadash|) / mean(|gammadash|)`` (MeanSquaredCurvature)."""
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
    """Return one curve pair's CurveCurveDistance term.

    ``sum(|gammadash1_i| |gammadash2_j| max(d_min - |gamma1_i - gamma2_j|, 0)^2)``
    divided by the number of point pairs, for a candidate pair (``candidate``
    true); else 0.
    """
    gamma1 = jnp.asarray(gamma1)
    gammadash1 = jnp.asarray(gammadash1)
    gamma2 = jnp.asarray(gamma2, dtype=gamma1.dtype)
    gammadash2 = jnp.asarray(gammadash2, dtype=gamma1.dtype)
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
    """Return one curve's CurveSurfaceDistance term.

    ``mean(|gammadash_i| |n_j| max(d_min - |gamma_i - x_j|, 0)^2)`` over curve
    points ``i`` and surface points ``x_j`` with (unnormalized) normals ``n_j``,
    for a candidate curve (``candidate`` true); else 0.
    """
    curve_gamma = jnp.asarray(curve_gamma)
    curve_gammadash = jnp.asarray(curve_gammadash)
    surface_gamma = jnp.asarray(surface_gamma, dtype=curve_gamma.dtype)
    surface_normal = jnp.asarray(surface_normal, dtype=curve_gamma.dtype)
    return _candidate_pair_penalty(
        curve_gamma, curve_gammadash, surface_gamma, surface_normal, minimum_distance, candidate,
        jnp.mean,
    )
