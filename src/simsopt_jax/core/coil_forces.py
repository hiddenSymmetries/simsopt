"""Pure JAX kernels of the native coil force, torque and energy objectives.

Each kernel reproduces the arithmetic of its counterpart in
:mod:`simsopt.field.force` (``regularized_self_field`` is
``simsopt.field.selffield.B_regularized_pure``): the same constants, the
``1e-10`` offset added to every component of the distance vectors of the
mutual field and the inductances, and the same order of operations. Values and
gradients therefore agree with native to round-off, including the NaNs of
degenerate geometry (a zero tangent) and of a zero regularization.

Coil stacks have shape ``(ncoils, nquadpoints, 3)``; currents and
regularizations have shape ``(ncoils,)``. A coil group is a
``(gammas, gammadashs, currents)`` triple. Every kernel takes the stride
``downsample`` over the quadrature points of all its stacks, as native does;
it is a static Python integer. Source groups (native's coarse and fine source
coils) may have quadrature counts different from the targets'.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants

from .biotsavart import biot_savart_A

__all__ = [
    "b2energy",
    "lp_force",
    "lp_torque",
    "net_flux",
    "regularized_self_field",
    "squared_mean_force",
    "squared_mean_torque",
]

CoilGroup = tuple[jax.Array, jax.Array, jax.Array]

# mu_0 / (4 pi) as selffield.py evaluates it (CODATA mu_0, not exactly 1e-7).
_SELF_FIELD_PREFACTOR = constants.mu_0 / (4 * np.pi)
# Offset native force.py adds to each component of a distance vector.
_DISTANCE_OFFSET = 1e-10


def _self_field_singularity_term(rc_prime, rc_prime_prime, regularization):
    norm_rc_prime = jnp.linalg.norm(rc_prime, axis=1)
    return jnp.cross(rc_prime, rc_prime_prime) * (
        0.5 * (-2 + jnp.log(64 * norm_rc_prime * norm_rc_prime / regularization)) / (norm_rc_prime**3)
    )[:, None]


def regularized_self_field(gamma, gammadash, gammadashdash, quadpoints, current, regularization):
    """Return the regularized self field ``(n, 3)`` of one coil (Landreman and Hurwitz).

    ``quadpoints`` is the curve parameter in ``[0, 1)`` and ``regularization``
    the cross-section term of ``regularization_circ``/``regularization_rect``.
    """
    phi = quadpoints * 2 * jnp.pi
    rc = gamma
    rc_prime = gammadash / 2 / jnp.pi
    rc_prime_prime = gammadashdash / 4 / jnp.pi**2
    dphi = 2 * jnp.pi / phi.shape[0]
    analytic_term = _self_field_singularity_term(rc_prime, rc_prime_prime, regularization)
    dr = rc[:, None] - rc[None, :]
    first_term = jnp.cross(rc_prime[None, :], dr) / (
        (jnp.sum(dr * dr, axis=2) + regularization) ** 1.5
    )[:, :, None]
    cos_fac = 2.0 - 2.0 * jnp.cos(phi[None, :] - phi[:, None])
    second_term = jnp.cross(rc_prime_prime, rc_prime)[:, None, :] * (
        0.5 * cos_fac / (cos_fac * jnp.sum(rc_prime * rc_prime, axis=1)[:, None] + regularization) ** 1.5
    )[:, :, None]
    integral_term = dphi * jnp.sum(first_term + second_term, 1)
    return current * _SELF_FIELD_PREFACTOR * (analytic_term + integral_term)


def _sampled(group: CoilGroup, downsample: int) -> CoilGroup:
    gammas, gammadashs, currents = group
    return gammas[:, ::downsample], gammadashs[:, ::downsample], currents


def _group_field_at_point(point, group: CoilGroup, excluded):
    """Field of ``group`` at ``point``, without coil ``excluded`` (an index or ``None``)."""
    gammas, gammadashs, currents = group
    deltas = point - gammas
    if excluded is not None:
        # Native evaluates the excluded coil (the target itself) in an untaken
        # branch of a conditional, so its near-singular terms never reach the
        # gradient: give it safe inputs before the norm, then drop it.
        is_excluded = jnp.arange(gammas.shape[0]) == excluded
        deltas = jnp.where(is_excluded[:, None, None], 1.0, deltas)

    def from_coil(delta, gammadash, current):
        return jnp.sum(
            jnp.cross(gammadash, delta)
            / (jnp.linalg.norm(delta + _DISTANCE_OFFSET, axis=1) ** 3)[:, None],
            axis=0,
        ) * current

    contributions = jax.vmap(from_coil)(deltas, gammadashs, currents)
    if excluded is not None:
        contributions = jnp.where(is_excluded[:, None], 0.0, contributions)
    return jnp.sum(contributions, axis=0) / gammas.shape[1] * 1e-7


def _mutual_field_at_point(index, point, targets: CoilGroup, sources: tuple[CoilGroup, ...]):
    """Field at ``point`` of target ``index`` from the other targets and every source."""
    field = _group_field_at_point(point, targets, index)
    for group in sources:
        field = field + _group_field_at_point(point, group, None)
    return field


def _target_mutual_fields(targets: CoilGroup, sources: tuple[CoilGroup, ...]):
    """Mutual field ``(ntargets, n, 3)`` at every point of every target coil."""
    gammas = targets[0]

    def on_coil(index, gamma):
        return jax.vmap(lambda point: _mutual_field_at_point(index, point, targets, sources))(gamma)

    return jax.vmap(on_coil)(jnp.arange(gammas.shape[0]), gammas)


def _self_fields(targets: CoilGroup, gammadashdashs, quadpoints, regularizations):
    gammas, gammadashs, currents = targets
    return jax.vmap(regularized_self_field, in_axes=(0, 0, 0, None, 0, 0))(
        gammas, gammadashs, gammadashdashs, quadpoints, currents, regularizations
    )


def _centroid(gamma, gammadash):
    arclength = jnp.linalg.norm(gammadash, axis=-1)
    return jnp.sum(gamma * arclength[:, None], axis=0) / jnp.sum(arclength)


def _thresholded_lp(densities, gammadash_norms, p, threshold):
    """``(1/p) sum_i (1/n) sum_k max(density - threshold, 0)^p |gammadash|``."""
    npoints = densities.shape[1]
    return jnp.sum(jnp.sum(jnp.maximum(densities - threshold, 0) ** p * gammadash_norms)) / npoints * (1.0 / p)


def lp_force(
    targets: CoilGroup,
    gammadashdashs,
    quadpoints,
    regularizations,
    sources: tuple[CoilGroup, ...],
    p,
    threshold,
    downsample: int,
):
    """Native ``lp_force_pure``: the Lp norm of the force per unit length (MN/m)^p.

    The force on each target uses its regularized self field (quadrature
    parameter ``quadpoints`` of the first target, as native) and the field of
    the other targets and the sources.
    """
    targets = _sampled(targets, downsample)
    gammadashdashs = gammadashdashs[:, ::downsample]
    quadpoints = quadpoints[::downsample]
    sources = tuple(_sampled(group, downsample) for group in sources)
    _gammas, gammadashs, currents = targets
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)
    tangents = gammadashs / gammadash_norms[:, :, None]
    fields = _target_mutual_fields(targets, sources) + _self_fields(
        targets, gammadashdashs, quadpoints, regularizations
    )
    forces = currents[:, None, None] * jnp.cross(tangents, fields)
    return _thresholded_lp(jnp.linalg.norm(forces, axis=-1) / 1e6, gammadash_norms, p, threshold)


def lp_torque(
    targets: CoilGroup,
    gammadashdashs,
    quadpoints,
    regularizations,
    sources: tuple[CoilGroup, ...],
    p,
    threshold,
    downsample: int,
):
    """Native ``lp_torque_pure``: the Lp norm of the torque per unit length (MN)^p,
    about each target's arclength centroid."""
    targets = _sampled(targets, downsample)
    gammadashdashs = gammadashdashs[:, ::downsample]
    quadpoints = quadpoints[::downsample]
    sources = tuple(_sampled(group, downsample) for group in sources)
    gammas, gammadashs, currents = targets
    centers = jax.vmap(_centroid)(gammas, gammadashs)
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)
    tangents = gammadashs / gammadash_norms[:, :, None]
    fields = _target_mutual_fields(targets, sources) + _self_fields(
        targets, gammadashdashs, quadpoints, regularizations
    )
    forces = currents[:, None, None] * jnp.cross(tangents, fields)
    torques = jnp.cross(gammas - centers[:, None, :], forces)
    return _thresholded_lp(jnp.linalg.norm(torques, axis=-1) / 1e6, gammadash_norms, p, threshold)


def squared_mean_force(targets: CoilGroup, sources: tuple[CoilGroup, ...], downsample: int):
    """Native ``squared_mean_force_pure``: ``sum_i |mean force per unit length_i|^2`` in (MN/m)^2.

    Only the mutual field enters; a coil's own field exerts no net force.
    """
    targets = _sampled(targets, downsample)
    sources = tuple(_sampled(group, downsample) for group in sources)
    gammas, gammadashs, currents = targets
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
    tangents = gammadashs / gammadash_norms
    force_densities = currents[:, None, None] * jnp.cross(
        tangents, _target_mutual_fields(targets, sources)
    )
    mean_forces = jnp.sum(force_densities * gammadash_norms, axis=1) / gammas.shape[1]
    return jnp.sum(jnp.linalg.norm(mean_forces, axis=-1) ** 2) * 1e-12


def squared_mean_torque(targets: CoilGroup, sources: tuple[CoilGroup, ...], downsample: int):
    """Native ``squared_mean_torque``: ``sum_i |mean torque per unit length_i|^2`` in MN^2."""
    targets = _sampled(targets, downsample)
    sources = tuple(_sampled(group, downsample) for group in sources)
    gammas, gammadashs, currents = targets
    centers = jax.vmap(_centroid)(gammas, gammadashs)
    arclengths = jnp.linalg.norm(gammadashs, axis=-1)
    tangents = gammadashs / arclengths[:, :, None]
    forces = currents[:, None, None] * jnp.cross(tangents, _target_mutual_fields(targets, sources))
    torques = jnp.cross(gammas - centers[:, None, :], forces) * arclengths[:, :, None]
    mean_torques = jnp.sum(torques, axis=1) / gammas.shape[1]
    return jnp.sum(jnp.linalg.norm(mean_torques, axis=-1) ** 2) * 1e-12


def _inductance_kernel_sums(gammas_a, gammadashs_a, gammas_b, gammadashs_b, regularization):
    """``sum_k sum_l (gd_a[k] . gd_b[l]) / sqrt(|r|^2 + regularization)`` per coil pair."""
    r = gammas_b[..., None, :, :] - gammas_a[..., :, None, :] + _DISTANCE_OFFSET
    r_norm = jnp.linalg.norm(r, axis=-1)
    gammadash_products = jnp.sum(gammadashs_b[..., None, :, :] * gammadashs_a[..., :, None, :], axis=-1)
    if regularization is not None:
        r_norm = jnp.sqrt(r_norm**2 + regularization[..., None, None])
    return jnp.sum(jnp.sum(gammadash_products / r_norm, axis=-1), axis=-1)


def coil_inductances(gammas, gammadashs, regularizations, downsample: int):
    """Native ``_coil_coil_inductances_pure``: the inductance matrix in henries.

    Mutual terms use the unregularized kernel; the diagonal uses each coil's
    regularization (native evaluates both kernels on every pair and keeps the
    regularized diagonal; only the diagonal blocks are needed for it).
    """
    gammas = gammas[:, ::downsample]
    gammadashs = gammadashs[:, ::downsample]
    npoints_squared = gammas.shape[1] ** 2
    mutual = _inductance_kernel_sums(
        gammas[:, None], gammadashs[:, None], gammas[None, :], gammadashs[None, :], None
    ) / npoints_squared
    self_terms = _inductance_kernel_sums(
        gammas, gammadashs, gammas, gammadashs, regularizations
    ) / npoints_squared
    inductances = jnp.where(jnp.eye(gammas.shape[0], dtype=bool), jnp.diag(self_terms), mutual)
    return 1e-7 * inductances


def b2energy(gammas, gammadashs, currents, regularizations, downsample: int):
    """Native ``b2energy_pure``: the vacuum field energy ``(1/2) I^T L I`` in MJ."""
    current_products = currents[:, None] * currents[None, :]
    inductances = coil_inductances(gammas, gammadashs, regularizations, downsample)
    return 0.5 * jnp.sum(current_products * inductances) / 1e6


def net_flux(target_gamma, target_gammadash, sources: CoilGroup, downsample: int):
    """Native ``NetFluxes``: the mean of ``A . gammadash`` over the target's points (Wb).

    ``A`` is the Biot-Savart vector potential of the sources at full
    quadrature, evaluated at the target's ``downsample``-strided points.
    """
    gammadash = target_gammadash[::downsample]
    vector_potential = biot_savart_A(target_gamma[::downsample], *sources)
    return jnp.sum(jnp.sum(vector_potential * gammadash, axis=-1), axis=-1) / gammadash.shape[0]
