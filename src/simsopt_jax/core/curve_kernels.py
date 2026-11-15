"""Pure JAX kernels for sampled coil-geometry penalties.

Inputs use float64 with ``jax_enable_x64`` enabled; NumPy inputs are copied.
Distance penalties require caller-supplied candidate decisions.
"""

from __future__ import annotations

import jax.numpy as jnp

from simsopt_jax.backend.jit import jit
from simsopt_jax.backend.runtime import snapshot_host_tree

from ._device_scalars import staged_like
from ._math_utils import as_jax_float64


def distance_candidate_pure(points1, points2, minimum_distance):
    """Test whether any sampled point pair is strictly closer than the threshold.

    Classification can differ from the native search exactly at the threshold.

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
    """Return the weighted squared distance deficit, or zero value and gradient for a non-candidate."""
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
    """Average float64 speeds to obtain the curve length."""
    return jnp.mean(incremental_arclength)


def kappa_pure(d1gamma, d2gamma):
    """Compute sampled curve curvature from first and second parameter derivatives.

    Args:
        d1gamma: Array of shape (n, 3), derivative in m with respect to the unit-period parameter.
        d2gamma: Array of shape (n, 3), second parameter derivative in m.

    Returns:
        Array of shape (n,): curvature in 1/m.
    """
    return _kappa_pure(*map(as_jax_float64, (d1gamma, d2gamma)))


@jit
def _kappa_pure(d1gamma, d2gamma):
    """Compute curvature from float64 parameter derivatives."""
    return (
        jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)
        / jnp.linalg.norm(d1gamma, axis=1) ** 3
    )


def curvature_p_norm_from_kappa_pure(kappa, gammadash, p, desired_kappa):
    """Integrate the p-th power of curvature excess over arclength, scaled by 1/p.

    No p-th root is taken.

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
    """Integrate float64 curvature excess with arclength weighting and a factor of 1/p."""
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(kappa - desired_kappa, zero)
    return (one / p) * jnp.mean((excess**p) * arc_length)


def mean_squared_curvature_pure(kappa, gammadash):
    """Return the arclength-averaged squared curvature.

    Args:
        kappa: Array of shape (n,), curvature in 1/m.
        gammadash: Array of shape (n, 3), derivative in m with respect to the unit-period parameter.

    Returns:
        Array: scalar arclength-averaged squared curvature in 1/m^2.
    """
    return _mean_squared_curvature_pure(*map(as_jax_float64, (kappa, gammadash)))


@jit
def _mean_squared_curvature_pure(kappa, gammadash):
    """Compute arclength-averaged squared curvature from float64 operands."""
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
    """Return the mean squared distance deficit weighted by both curves' speeds.

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
    """Return the mean squared distance deficit weighted by speed and normal magnitude.

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
