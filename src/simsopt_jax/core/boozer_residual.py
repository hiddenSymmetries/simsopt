"""JAX Boozer residuals and derivatives for surface optimization.

Coordinates are normalized turns in [0, 1); residuals are flattened in
point-major, component-minor order. Functions support use in jitted programs.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax



def _local_residual(local: jax.Array, weight_inv_modB: bool) -> jax.Array:
    """Return one point's Boozer residual, shape (3,), from local B, tangent, and G."""
    B, tangent, G = local[:3], local[3:6], local[6]
    B2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2]
    residual = G * B - B2 * tangent
    return residual / jnp.sqrt(B2) if weight_inv_modB else residual


def boozer_point_residual(
    G, iota, B: jax.Array, xphi: jax.Array, xtheta: jax.Array, *, weight_inv_modB: bool
) -> jax.Array:
    """Evaluate Boozer residuals at all points in point-major order.

    Args:
        G (float | jax.Array): Scalar shape () covariant Boozer field constant in tesla
            meters.
        iota (float | jax.Array): Scalar shape () dimensionless rotational transform.
        B (jax.Array): Magnetic field in teslas, shape (npoints, 3).
        xphi (jax.Array): Toroidal tangent in meters per turn, shape (npoints, 3).
        xtheta (jax.Array): Surface poloidal tangent in meters per turn, shape (npoints,
            3).
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        jax.Array: Residual shape (3 * npoints,), in tesla squared meters, or tesla
            meters when weighted.
    """
    G_column = jnp.broadcast_to(G, (B.shape[0], 1))
    local = jnp.concatenate((B, xphi + iota * xtheta, G_column), axis=1)
    residual_of_local = partial(_local_residual, weight_inv_modB=weight_inv_modB)
    return jax.vmap(residual_of_local)(local).reshape(-1)


def taylor_field(
    coefficients_at: Callable[[jax.Array], tuple[jax.Array, ...]], points: jax.Array
) -> jax.Array:
    """Evaluate a field using its Taylor coefficients at the evaluation points.

    Point derivatives match the field through the supplied order; coefficients
    retain their dependence on coils while expansion points are held fixed.

    Args:
        coefficients_at (Callable): Map points (npoints, 3) to B and optionally
            dB/dX and d2B/dXdX, with shapes (npoints, 3), (npoints, 3, 3),
            and (npoints, 3, 3, 3), respectively.
        points (jax.Array): Evaluation points in meters, shape (npoints, 3).

    Returns:
        jax.Array: Magnetic field in teslas, shape (npoints, 3).
    """
    expansion = lax.stop_gradient(points)
    coefficients = coefficients_at(expansion)
    displacement = points - expansion
    field = coefficients[0]
    if len(coefficients) > 1:
        field = field + jnp.einsum("pj,pjl->pl", displacement, coefficients[1])
    if len(coefficients) > 2:
        # Multiply and sum rather than einsum: as a dot, the GPU compiler of JAX 0.12
        # can read d2B with a field and a derivative axis swapped (openxla/xla issue 51186),
        # which is wrong wherever d2B is not symmetric in those axes.
        field = field + 0.5 * jnp.sum(
            displacement[:, :, None, None] * displacement[:, None, :, None] * coefficients[2],
            axis=(1, 2),
        )
    return field


def _require_derivative_order(derivatives: int, highest: int) -> None:
    """Check that ``derivatives`` is an integer order from 0 to ``highest``.

    Raises:
        ValueError: If ``derivatives`` is a bool, not an integer, or outside
            ``0..highest``.
    """
    # Native asserts the range; a ValueError also holds under python -O, and a
    # bool or float equal to an order is not one.
    if (
        isinstance(derivatives, bool)
        or not isinstance(derivatives, numbers.Integral)
        or derivatives not in range(highest + 1)
    ):
        raise ValueError(
            f"derivatives must be an integer from 0 to {highest}; got {derivatives!r}."
        )


def residual_derivatives(
    residual: Callable[[jax.Array], jax.Array], x: jax.Array, derivatives: int
) -> tuple[jax.Array, ...]:
    """Evaluate a residual vector and optional Jacobian and per-residual Hessians.

    Args:
        residual (Callable): Shape (nx,) to the residual, shape (nr,).
        x (jax.Array): Shape (nx,) point of evaluation.
        derivatives (int): Derivative order, 0, 1 or 2.

    Returns:
        tuple[jax.Array, ...]: Residual (nr,), optionally Jacobian (nr, nx) and
            Hessians (nr, nx, nx).

    Raises:
        ValueError: If ``derivatives`` is not 0, 1 or 2.
    """
    _require_derivative_order(derivatives, 2)
    if derivatives == 0:
        return (residual(x),)

    def residual_and_aux(point):
        value = residual(point)
        return value, value

    if derivatives == 1:
        jacobian, value = jax.jacfwd(residual_and_aux, has_aux=True)(x)
        return value, jacobian

    def jacobian_and_aux(point):
        jacobian, value = jax.jacfwd(residual_and_aux, has_aux=True)(point)
        return jacobian, (value, jacobian)

    hessian, (value, jacobian) = jax.jacfwd(jacobian_and_aux, has_aux=True)(x)
    return value, jacobian, hessian


def objective_derivatives(
    objective: Callable[[jax.Array], jax.Array], x: jax.Array, derivatives: int
) -> tuple[jax.Array, ...]:
    """Evaluate a scalar objective and optional gradient and Hessian.

    Args:
        objective (Callable): Shape (nx,) to a scalar shape ().
        x (jax.Array): Shape (nx,) point of evaluation.
        derivatives (int): Derivative order, 0, 1 or 2.

    Returns:
        tuple[jax.Array, ...]: Value (), optionally gradient (nx,) and Hessian (nx, nx).

    Raises:
        ValueError: If ``derivatives`` is not 0, 1 or 2.
    """
    _require_derivative_order(derivatives, 2)
    if derivatives == 0:
        return (objective(x),)
    if derivatives == 1:
        return jax.value_and_grad(objective)(x)

    def gradient_and_aux(point):
        value, gradient = jax.value_and_grad(objective)(point)
        return gradient, (value, gradient)

    hessian, (value, gradient) = jax.jacfwd(gradient_and_aux, has_aux=True)(x)
    return value, gradient, hessian
