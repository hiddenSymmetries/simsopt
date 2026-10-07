"""Boozer residual of native ``boozer_surface_residual`` and its derivatives.

At each surface quadrature point the residual is

.. math::

    \\mathbf r = w \\bigl(G \\mathbf B - |\\mathbf B|^2
        (\\mathbf x_\\varphi + \\iota \\mathbf x_\\theta)\\bigr),
    \\qquad w = 1/|\\mathbf B| \\text{ if weight_inv_modB, else } 1,

with :math:`\\mathbf B` the field at the point and :math:`\\mathbf x_\\varphi`,
:math:`\\mathbf x_\\theta` the native ``gammadash1`` and ``gammadash2``. The
unknowns are ``x = [surface DOFs, iota, G]``, or ``[surface DOFs, iota]`` when
``G`` is not optimized and is a constant.

Derivatives with respect to ``x`` are assembled as native
``simsoptpp.boozer_residual_ds``/``boozer_residual_ds2`` assemble them, so no
derivative passes through the field kernel. A point's residual is a function
of seven local values ``u = (B, x_phi + iota x_theta, G)``; its derivatives
with respect to ``u`` are exact (JAX) and are chained with ``du/dx``, built
from ``dB/dX`` and the coefficient derivatives of the position and the
tangents. Position and tangents are linear in the surface coefficients, so
the only second derivatives of ``u`` are ``d2B/dXdX`` along the position's
coefficient derivatives and the cross term of ``iota x_theta``.

Every function is traceable and meant to run inside a jitted program.
Residual rows are point-major and component-minor, as natively.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from simsopt_jax.pytree import pytree_dataclass

__all__ = ["BoozerPoints", "boozer_least_squares", "boozer_residual"]

_LOCAL_SIZE = 7  # B (3), x_phi + iota x_theta (3), G


@pytree_dataclass(
    data=(
        "B",
        "xphi",
        "xtheta",
        "dB_by_dX",
        "d2B_by_dXdX",
        "dgamma_by_dcoeff",
        "dgammadash1_by_dcoeff",
        "dgammadash2_by_dcoeff",
    )
)
class BoozerPoints:
    """Field and surface at the quadrature points, flattened over the points.

    ``B``, ``xphi`` and ``xtheta`` are ``(npoints, 3)``; ``dB_by_dX``
    ``(npoints, 3, 3)`` and ``d2B_by_dXdX`` ``(npoints, 3, 3, 3)`` have the
    native layout (derivative directions first, field component last); the
    ``*_by_dcoeff`` derivatives of ``gamma``, ``gammadash1`` and ``gammadash2``
    are ``(npoints, 3, nsurface)``. Values need only the first three; first
    derivatives add ``dB_by_dX`` and the coefficient derivatives, second
    derivatives ``d2B_by_dXdX``. Unneeded fields may be ``None``.
    """

    B: jax.Array
    xphi: jax.Array
    xtheta: jax.Array
    dB_by_dX: jax.Array | None = None
    d2B_by_dXdX: jax.Array | None = None
    dgamma_by_dcoeff: jax.Array | None = None
    dgammadash1_by_dcoeff: jax.Array | None = None
    dgammadash2_by_dcoeff: jax.Array | None = None


def _required(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"these derivatives need BoozerPoints.{name}.")
    return value


def _local_residual(local: jax.Array, weight_inv_modB: bool) -> jax.Array:
    """The residual ``(3,)`` of one point's local values ``(B, tangent, G)``."""
    B, tangent, G = local[:3], local[3:6], local[6]
    B2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2]
    residual = G * B - B2 * tangent
    return residual / jnp.sqrt(B2) if weight_inv_modB else residual


def _local_values(G, iota, points: BoozerPoints) -> jax.Array:
    G_column = jnp.broadcast_to(G, (points.B.shape[0], 1))
    return jnp.concatenate((points.B, points.xphi + iota * points.xtheta, G_column), axis=1)


def _local_derivatives(
    local: jax.Array, weight_inv_modB: bool, order: int
) -> tuple[jax.Array, ...]:
    """Pointwise residual ``(npoints, 3)`` and, up to ``order``, its derivatives
    with respect to the local values, ``(npoints, 3, 7)`` and ``(npoints, 3, 7, 7)``."""
    residual_of_local = partial(_local_residual, weight_inv_modB=weight_inv_modB)
    derivatives = [residual_of_local, jax.jacfwd(residual_of_local), jax.hessian(residual_of_local)]
    return tuple(jax.vmap(derivative)(local) for derivative in derivatives[: order + 1])


def _linearization(iota, points: BoozerPoints, optimize_G: bool) -> jax.Array:
    """``du/dx`` per point, ``(npoints, 7, nx)``."""
    dgamma = _required(points.dgamma_by_dcoeff, "dgamma_by_dcoeff")
    npoints, nsurface = dgamma.shape[0], dgamma.shape[-1]
    dB = jnp.einsum("pjl,pjs->pls", _required(points.dB_by_dX, "dB_by_dX"), dgamma)
    dtangent = _required(points.dgammadash1_by_dcoeff, "dgammadash1_by_dcoeff") + iota * _required(
        points.dgammadash2_by_dcoeff, "dgammadash2_by_dcoeff"
    )
    surface_columns = jnp.concatenate(
        (dB, dtangent, jnp.zeros((npoints, 1, nsurface), dgamma.dtype)), axis=1
    )
    iota_column = jnp.concatenate(
        (jnp.zeros_like(points.xtheta), points.xtheta, jnp.zeros((npoints, 1), dgamma.dtype)),
        axis=1,
    )
    columns = [surface_columns, iota_column[..., None]]
    if optimize_G:
        G_column = jnp.zeros((npoints, _LOCAL_SIZE, 1), dgamma.dtype).at[:, 6, 0].set(1.0)
        columns.append(G_column)
    return jnp.concatenate(columns, axis=-1)


def _second_order_terms(
    covector: jax.Array, points: BoozerPoints, nx: int, *, per_point: bool
) -> jax.Array:
    """``sum_v covector_v d2u_v/dx2``, ``(npoints, ..., nx, nx)`` per point or
    ``(..., nx, nx)`` summed over the points.

    ``covector`` is ``(npoints, ..., 7)``. The field curvature fills the
    surface block, the ``iota x_theta`` cross term the surface-iota entries.
    """
    dgamma = _required(points.dgamma_by_dcoeff, "dgamma_by_dcoeff")
    nsurface = dgamma.shape[-1]
    points_axis = "p" if per_point else ""
    field_hessian = jnp.einsum(
        "p...l,pjkl->p...jk", covector[..., :3], _required(points.d2B_by_dXdX, "d2B_by_dXdX")
    )
    surface_block = jnp.einsum(
        f"pjs,p...jk,pkt->{points_axis}...st", dgamma, field_hessian, dgamma
    )
    cross = jnp.einsum(
        f"p...c,pcs->{points_axis}...s",
        covector[..., 3:6],
        _required(points.dgammadash2_by_dcoeff, "dgammadash2_by_dcoeff"),
    )
    terms = jnp.zeros(cross.shape[:-1] + (nx, nx), dgamma.dtype)
    terms = terms.at[..., :nsurface, :nsurface].set(surface_block)
    terms = terms.at[..., :nsurface, nsurface].set(cross)
    return terms.at[..., nsurface, :nsurface].set(cross)


def boozer_residual(
    G, iota, points: BoozerPoints, *, derivatives: int, optimize_G: bool, weight_inv_modB: bool
) -> tuple[jax.Array, ...]:
    """Native ``boozer_surface_residual``: ``(r,)``, ``(r, J)`` or ``(r, J, H)``.

    ``r`` is ``(3 npoints,)``, ``J = dr/dx`` ``(3 npoints, nx)`` and ``H`` the
    second derivative of every residual, ``(3 npoints, nx, nx)``; ``nx`` counts
    ``G`` only if ``optimize_G``.
    """
    local = _local_derivatives(_local_values(G, iota, points), weight_inv_modB, derivatives)
    residual = local[0]
    nresiduals = residual.size
    if derivatives == 0:
        return (residual.reshape(nresiduals),)
    linearization = _linearization(iota, points, optimize_G)
    nx = linearization.shape[-1]
    jacobian = jnp.einsum("pkv,pvx->pkx", local[1], linearization)
    if derivatives == 1:
        return residual.reshape(nresiduals), jacobian.reshape(nresiduals, nx)
    hessian = jnp.einsum(
        "pvx,pkvw,pwy->pkxy", linearization, local[2], linearization
    ) + _second_order_terms(local[1], points, nx, per_point=True)
    return (
        residual.reshape(nresiduals),
        jacobian.reshape(nresiduals, nx),
        hessian.reshape(nresiduals, nx, nx),
    )


def boozer_least_squares(
    G, iota, points: BoozerPoints, *, derivatives: int, optimize_G: bool, weight_inv_modB: bool
) -> tuple[jax.Array, ...]:
    """``0.5 |r|^2`` over all residuals and, up to ``derivatives``, its gradient
    and Hessian in ``x``: native ``boozer_residual``, ``boozer_residual_ds`` or
    ``boozer_residual_ds2`` as a tuple (with the ``G`` entries only if
    ``optimize_G``), not divided by the number of residuals.
    """
    local = _local_derivatives(_local_values(G, iota, points), weight_inv_modB, derivatives)
    residual = local[0]
    value = 0.5 * jnp.sum(residual * residual)
    if derivatives == 0:
        return (value,)
    local_gradient = jnp.einsum("pk,pkv->pv", residual, local[1])
    linearization = _linearization(iota, points, optimize_G)
    gradient = jnp.einsum("pv,pvx->x", local_gradient, linearization)
    if derivatives == 1:
        return value, gradient
    local_curvature = jnp.einsum("pkv,pkw->pvw", local[1], local[1]) + jnp.einsum(
        "pk,pkvw->pvw", residual, local[2]
    )
    curved_linearization = jnp.einsum("pvw,pwy->pvy", local_curvature, linearization)
    hessian = jnp.einsum("pvx,pvy->xy", linearization, curved_linearization)
    hessian = hessian + _second_order_terms(
        local_gradient, points, linearization.shape[-1], per_point=False
    )
    return value, gradient, hessian
