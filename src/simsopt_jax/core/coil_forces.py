"""Pure JAX kernels of the native coil force, torque and energy objectives.

Shared field, Lorentz density, inductance, energy and flux arithmetic uses
:mod:`simsopt.field.force`; self fields use
``simsopt.field.selffield.B_regularized_pure``: the same constants, the
``1e-10`` offset added to every component of the distance vectors of the
mutual field and the inductances, and native quadrature normalization. Values
and gradients therefore agree with native to round-off, including the NaNs of
degenerate geometry (a zero tangent) and of a zero regularization.

Coil stacks have shape ``(ncoils, nquadpoints, 3)``; currents and
regularizations have shape ``(ncoils,)``. Force and torque reductions retain
safe masking before excluded self-field terms are differentiated; the upstream
complete objectives cannot supply that masking through their public operands.
A coil group is a ``(gammas, gammadashs, currents)`` triple. Every kernel takes the stride
``downsample`` over the quadrature points of all its stacks, as native does;
it is a static Python integer. Source groups (native's coarse and fine source
coils) may have quadrature counts different from the targets'.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from simsopt.field.force import (
    _B_at_point_from_coil_set_pure,
    _coil_coil_inductances_pure,
    _lorentz_force_density_pure,
    b2energy_pure,
    net_ext_fluxes_pure,
)
from simsopt.field.selffield import B_regularized_pure
from simsopt.geo.curve import centroid_pure

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

# Offset native force.py adds to each component of a distance vector.
_DISTANCE_OFFSET = 1e-10


def regularized_self_field(gamma, gammadash, gammadashdash, quadpoints, current, regularization):
    """Return the regularized self field ``(n, 3)`` of one coil.

    ``quadpoints`` is the curve parameter in ``[0, 1)`` and ``regularization``
    the cross-section term of ``regularization_circ``/``regularization_rect``.
    The method is the one of ``simsopt.field.selffield``, from:

    Hurwitz, Siena, Matt Landreman, and Thomas M. Antonsen.
    "Efficient calculation of the self magnetic field, self-force, and self-inductance for
    electromagnetic coils." IEEE Transactions on Magnetics 60 (2024).
    https://doi.org/10.1109/TMAG.2024.3455946

    Landreman, Matt, Siena Hurwitz, and Thomas M. Antonsen.
    "Efficient calculation of self magnetic field, self-force, and self-inductance for
    electromagnetic coils with rectangular cross-section."
    Nuclear Fusion 65.3 (2025): 036008. https://doi.org/10.1088/1741-4326/adb04e

    Args:
        gamma: Array of shape (n, 3), coil positions in m.
        gammadash: Array of shape (n, 3), first parameter derivatives in m.
        gammadashdash: Array of shape (n, 3), second parameter derivatives in m.
        quadpoints: Array of shape (n,), unit-period parameter coordinates in [0, 1).
        current: float scalar, current in A.
        regularization: float scalar, cross-section regularization in m^2.

    Returns:
        Array of shape (n, 3): regularized self magnetic field in T.
    """
    return B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)


def _sampled(group: CoilGroup, downsample: int) -> CoilGroup:
    gammas, gammadashs, currents = group
    return gammas[:, ::downsample], gammadashs[:, ::downsample], currents


def _group_field_at_point(point, group: CoilGroup, excluded):
    """Field of ``group`` at ``point``, without coil ``excluded`` (an index or ``None``)."""
    gammas, gammadashs, currents = group
    deltas = point - gammas
    if excluded is not None:
        # Native's vmapped conditional still traces the excluded coil's singular
        # arithmetic. Make its relative geometry safe before differentiating.
        is_excluded = jnp.arange(gammas.shape[0]) == excluded
        deltas = jnp.where(is_excluded[:, None, None], 1.0, deltas)
    # Translating the evaluation point to zero preserves the precomputed
    # relative vectors, including the safe excluded distances, exactly.
    return _B_at_point_from_coil_set_pure(
        jnp.zeros_like(point), -deltas, gammadashs, currents,
        exclude_index=-1 if excluded is None else excluded, eps=_DISTANCE_OFFSET,
    )


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


def _regularized_force_densities(
    targets: CoilGroup,
    gammadashdashs,
    quadpoints,
    regularizations,
    sources: tuple[CoilGroup, ...],
    downsample: int,
) -> tuple[CoilGroup, jax.Array, jax.Array]:
    """Return sampled targets, speeds and Lorentz densities including self fields.

    Apply the same quadrature stride to every geometry input and source group.
    """
    targets = _sampled(targets, downsample)
    sources = tuple(_sampled(group, downsample) for group in sources)
    _gammas, gammadashs, currents = targets
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)
    tangents = gammadashs / gammadash_norms[:, :, None]
    fields = _target_mutual_fields(targets, sources) + _self_fields(
        targets, gammadashdashs[:, ::downsample], quadpoints[::downsample], regularizations
    )
    forces = _lorentz_force_density_pure(tangents, currents[:, None, None], fields)
    return targets, gammadash_norms, forces


def _thresholded_lp(densities, gammadash_norms, p, threshold):
    """``(1/p) sum_i (1/n) sum_k max(density - threshold, 0)^p |gammadash|``."""
    npoints = densities.shape[1]
    return jnp.sum(jnp.maximum(densities - threshold, 0) ** p * gammadash_norms) / npoints * (1.0 / p)


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
    """Native lp_force_pure integral penalty in (MN/m)^p m.

    Compute (1/p) sum_i mean(max(|dF_i/dl| - threshold, 0)^p |gammadash_i|),
    including regularized self fields. There is no p-th root or coil-length
    normalization; all derivatives use the unit-period curve parameter.

    Args:
        targets: tuple of arrays (gamma, gammadash, currents), shapes (m, n, 3), (m, n, 3), (m,), in m, m per unit-period parameter, and A.
        gammadashdashs: Array of shape (m, n, 3), target second parameter derivatives in m.
        quadpoints: Array of shape (n,), first target parameter coordinates in [0, 1).
        regularizations: Array of shape (m,), cross-section regularization in m^2 from regularization_circ/regularization_rect.
        sources: tuple of coil-group tuples with array shapes (m, n, 3), (m, n, 3), (m,); quadrature counts may differ between groups.
        p: float scalar, dimensionless exponent.
        threshold: float scalar, force-density threshold in MN/m.
        downsample: int, positive static stride dividing every nonempty group quadrature count; validation belongs to callers.

    Returns:
        Array: scalar integral penalty in (MN/m)^p m.
    """
    _, gammadash_norms, forces = _regularized_force_densities(
        targets, gammadashdashs, quadpoints, regularizations, sources, downsample
    )
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
    """Native lp_torque_pure integral penalty in MN^p m.

    Compute the same thresholded integral as lp_force for torque density
    about each target arclength centroid, in MN. There is no p-th root or
    coil-length normalization.

    Args:
        targets: tuple of arrays (gamma, gammadash, currents), shapes (m, n, 3), (m, n, 3), (m,), in m, m per unit-period parameter, and A.
        gammadashdashs: Array of shape (m, n, 3), target second parameter derivatives in m.
        quadpoints: Array of shape (n,), first target parameter coordinates in [0, 1).
        regularizations: Array of shape (m,), cross-section regularization in m^2 from regularization_circ/regularization_rect.
        sources: tuple of coil-group tuples with array shapes (m, n, 3), (m, n, 3), (m,); quadrature counts may differ between groups.
        p: float scalar, dimensionless exponent.
        threshold: float scalar, torque-density threshold in MN.
        downsample: int, positive static stride dividing every nonempty group quadrature count; validation belongs to callers.

    Returns:
        Array: scalar integral penalty in MN^p m.
    """
    targets, gammadash_norms, forces = _regularized_force_densities(
        targets, gammadashdashs, quadpoints, regularizations, sources, downsample
    )
    gammas, gammadashs, _currents = targets
    centers = jax.vmap(centroid_pure)(gammas, gammadashs)
    torques = jnp.cross(gammas - centers[:, None, :], forces)
    return _thresholded_lp(jnp.linalg.norm(torques, axis=-1) / 1e6, gammadash_norms, p, threshold)


def squared_mean_force(targets: CoilGroup, sources: tuple[CoilGroup, ...], downsample: int):
    """Native squared_mean_force_pure: sum_i |integral dF_i/dl dl|^2 in MN^2.

    Quadrature computes mean(force_density * |gammadash|), without division
    by coil length. Only mutual fields enter.

    Args:
        targets: tuple of arrays (gamma, gammadash, currents), shapes (m, n, 3), (m, n, 3), (m,), in m, m per unit-period parameter, and A.
        sources: tuple of coil-group tuples with array shapes (m, n, 3), (m, n, 3), (m,); quadrature counts may differ between groups.
        downsample: int, positive static stride dividing every nonempty group quadrature count; validation belongs to callers.

    Returns:
        Array: scalar squared integrated force in MN^2.
    """
    targets = _sampled(targets, downsample)
    sources = tuple(_sampled(group, downsample) for group in sources)
    gammas, gammadashs, currents = targets
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
    tangents = gammadashs / gammadash_norms
    force_densities = _lorentz_force_density_pure(
        tangents, currents[:, None, None], _target_mutual_fields(targets, sources)
    )
    mean_forces = jnp.sum(force_densities * gammadash_norms, axis=1) / gammas.shape[1]
    return jnp.sum(jnp.linalg.norm(mean_forces, axis=-1) ** 2) * 1e-12


def squared_mean_torque(targets: CoilGroup, sources: tuple[CoilGroup, ...], downsample: int):
    """Native squared_mean_torque: sum_i |integral dT_i/dl dl|^2 in (MN m)^2.

    Torque density is about each target arclength centroid. Quadrature
    integrates it without dividing by coil length; only mutual fields enter.

    Args:
        targets: tuple of arrays (gamma, gammadash, currents), shapes (m, n, 3), (m, n, 3), (m,), in m, m per unit-period parameter, and A.
        sources: tuple of coil-group tuples with array shapes (m, n, 3), (m, n, 3), (m,); quadrature counts may differ between groups.
        downsample: int, positive static stride dividing every nonempty group quadrature count; validation belongs to callers.

    Returns:
        Array: scalar squared integrated torque in (MN m)^2.
    """
    targets = _sampled(targets, downsample)
    sources = tuple(_sampled(group, downsample) for group in sources)
    gammas, gammadashs, currents = targets
    centers = jax.vmap(centroid_pure)(gammas, gammadashs)
    arclengths = jnp.linalg.norm(gammadashs, axis=-1)
    tangents = gammadashs / arclengths[:, :, None]
    forces = _lorentz_force_density_pure(
        tangents, currents[:, None, None], _target_mutual_fields(targets, sources)
    )
    torques = jnp.cross(gammas - centers[:, None, :], forces) * arclengths[:, :, None]
    mean_torques = jnp.sum(torques, axis=1) / gammas.shape[1]
    return jnp.sum(jnp.linalg.norm(mean_torques, axis=-1) ** 2) * 1e-12


def coil_inductances(gammas, gammadashs, regularizations, downsample: int):
    """Native ``_coil_coil_inductances_pure``: the inductance matrix in henries.

    Mutual terms use the unregularized kernel; the diagonal uses each coil's
    regularization. The upstream kernel owns this arithmetic.

    Args:
        gammas: Array of shape (m, n, 3), coil positions in m.
        gammadashs: Array of shape (m, n, 3), unit-period parameter derivatives in m.
        regularizations: Array of shape (m,), cross-section regularization in m^2 from regularization_circ/regularization_rect.
        downsample: int, positive static stride dividing every nonempty group quadrature count; validation belongs to callers.

    Returns:
        Array of shape (m, m): inductance matrix in H.
    """
    return _coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations)


def b2energy(gammas, gammadashs, currents, regularizations, downsample: int):
    """Native ``b2energy_pure``: the vacuum field energy ``(1/2) I^T L I`` in MJ.

    Args:
        gammas: Array of shape (m, n, 3), coil positions in m.
        gammadashs: Array of shape (m, n, 3), unit-period parameter derivatives in m.
        currents: Array of shape (m,), coil currents in A.
        regularizations: Array of shape (m,), cross-section regularization in m^2 from regularization_circ/regularization_rect.
        downsample: int, positive static stride dividing every nonempty group quadrature count; validation belongs to callers.

    Returns:
        Array: scalar vacuum field energy in MJ.
    """
    return b2energy_pure(gammas, gammadashs, currents, downsample, regularizations)


def net_flux(target_gamma, target_gammadash, sources: CoilGroup, downsample: int):
    """Native ``NetFluxes``: the mean of ``A . gammadash`` over the target's points (Wb).

    ``A`` is the Biot-Savart vector potential of the sources at full
    quadrature, evaluated at the target's ``downsample``-strided points.

    Args:
        target_gamma: Array of shape (n, 3), target positions in m.
        target_gammadash: Array of shape (n, 3), target parameter derivatives in m.
        sources: tuple of arrays (gamma, gammadash, currents), shapes (m, n, 3), (m, n, 3), (m,), in m, m per unit-period parameter, and A.
        downsample: int, positive static stride dividing every nonempty group quadrature count; validation belongs to callers.

    Returns:
        Array: scalar net external flux in Wb.
    """
    vector_potential = biot_savart_A(target_gamma[::downsample], *sources)
    return net_ext_fluxes_pure(target_gammadash, vector_potential, downsample)
