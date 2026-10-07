"""JAX geometry of native simsopt surfaces, evaluated from their specs.

``surface_<method>(spec)`` returns native ``surface.<method>()`` for the
``SurfaceRZFourier``, ``SurfaceXYZFourier`` or ``SurfaceXYZTensorFourier``
whose state the spec holds, with the same conventions and array shapes:
``gamma``, its derivatives ``gammadash1``, ``gammadash2``, ``gammadash1dash1``,
``gammadash1dash2``, ``gammadash2dash2`` with respect to ``quadpoints_phi``
and ``quadpoints_theta``, ``normal``, ``unitnormal``, ``area`` and ``volume``.
Each is jitted with the spec's arrays as traced operands, so new coefficient
values reuse the compiled program; pass arrays already on the intended device
(:func:`simsopt_jax_adapters.geo.surface_spec_from_surface` does).

Coefficient derivatives are JAX transforms of :func:`surface_quantity_of_dofs`
at the native DOF vector, fixed DOFs included, which
:func:`~simsopt_jax.core.surface_fourier_series.surface_get_dofs` reads from a
spec. The DOF axes come last, as natively, so ``Derivative({surface: ...})``
projects them onto the free DOFs. Jit them with the spec as an argument::

    @jax.jit
    def dgamma_by_dcoeff(spec):
        dofs = surface_get_dofs(spec)
        return jax.jacfwd(surface_quantity_of_dofs(surface_gamma, spec))(dofs)

``jax.vjp`` gives ``dgamma_by_dcoeff_vjp``; ``jax.grad`` and ``jax.hessian``
of ``surface_area`` give ``darea_by_dcoeff`` and ``d2area_by_dcoeffdcoeff``.

Clamped ``SurfaceXYZTensorFourier`` second derivatives differ from native's,
which are wrong (:mod:`simsopt_jax.core.surface_fourier_series`). Where
squared normal components, or derivative intermediates such as the cube of
the normal's length, underflow or overflow float64, unit normals, areas and
their derivatives can differ from native in finiteness: XLA on CPU flushes
subnormal results to zero, and autodiff orders the arithmetic differently from
native's hand-written derivatives.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from .specs import SurfaceSpec
from .surface_fourier_series import (
    _rotating_frame_derivative,
    surface_spec_with_dofs,
)

__all__ = [
    "surface_area",
    "surface_gamma",
    "surface_gammadash1",
    "surface_gammadash1dash1",
    "surface_gammadash1dash2",
    "surface_gammadash2",
    "surface_gammadash2dash2",
    "surface_normal",
    "surface_quantity_of_dofs",
    "surface_unitnormal",
    "surface_volume",
]

_TWO_PI = 2.0 * np.pi


def _position_derivative(
    spec: SurfaceSpec, phi_order: int, theta_order: int
) -> jax.Array:
    """``d^(phi_order + theta_order) gamma / dquadpoints_phi^phi_order
    dquadpoints_theta^theta_order``, shape ``(nphi, ntheta, 3)``.

    Leibniz rule for the rotation ``x = xhat cos(phi) - yhat sin(phi)``,
    ``y = xhat sin(phi) + yhat cos(phi)``; each derivative of ``(cos(phi),
    sin(phi))`` turns it by a quarter.
    """
    phi = _TWO_PI * spec.quadpoints_phi
    cos_phi, sin_phi = jnp.cos(phi)[:, None], jnp.sin(phi)[:, None]
    turns = ((cos_phi, sin_phi), (-sin_phi, cos_phi), (-cos_phi, -sin_phi))
    x_terms, y_terms = [], []
    for rotation_order in range(phi_order + 1):
        x_hat, y_hat, z_hat = _rotating_frame_derivative(
            spec, phi_order - rotation_order, theta_order
        )
        if rotation_order == 0:
            z = z_hat
        cos_turned, sin_turned = turns[rotation_order]
        weight = comb(phi_order, rotation_order)
        if y_hat is None:
            x_terms.append(weight * (x_hat * cos_turned))
            y_terms.append(weight * (x_hat * sin_turned))
        else:
            x_terms.append(weight * (x_hat * cos_turned - y_hat * sin_turned))
            y_terms.append(weight * (x_hat * sin_turned + y_hat * cos_turned))
    position = jnp.stack((sum(x_terms), sum(y_terms), z), axis=-1)
    return _TWO_PI ** (phi_order + theta_order) * position


def _norm(vectors: jax.Array) -> jax.Array:
    """``sqrt(x*x + y*y + z*z)`` over the last axis, the native expression.

    Zero normals, and normals whose squares underflow past the subnormal range,
    therefore give native's non-finite unit normals and area gradients.
    """
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    return jnp.sqrt(x * x + y * y + z * z)


@jax.jit
def surface_gamma(spec: SurfaceSpec) -> jax.Array:
    return _position_derivative(spec, 0, 0)


@jax.jit
def surface_gammadash1(spec: SurfaceSpec) -> jax.Array:
    return _position_derivative(spec, 1, 0)


@jax.jit
def surface_gammadash2(spec: SurfaceSpec) -> jax.Array:
    return _position_derivative(spec, 0, 1)


@jax.jit
def surface_gammadash1dash1(spec: SurfaceSpec) -> jax.Array:
    return _position_derivative(spec, 2, 0)


@jax.jit
def surface_gammadash1dash2(spec: SurfaceSpec) -> jax.Array:
    return _position_derivative(spec, 1, 1)


@jax.jit
def surface_gammadash2dash2(spec: SurfaceSpec) -> jax.Array:
    return _position_derivative(spec, 0, 2)


@jax.jit
def surface_normal(spec: SurfaceSpec) -> jax.Array:
    return jnp.cross(surface_gammadash1(spec), surface_gammadash2(spec))


@jax.jit
def surface_unitnormal(spec: SurfaceSpec) -> jax.Array:
    normal = surface_normal(spec)
    return normal / _norm(normal)[..., None]


@jax.jit
def surface_area(spec: SurfaceSpec) -> jax.Array:
    norms = _norm(surface_normal(spec))
    return jnp.sum(norms) / norms.size


@jax.jit
def surface_volume(spec: SurfaceSpec) -> jax.Array:
    gamma = surface_gamma(spec)
    normal = surface_normal(spec)
    gamma_dot_normal = (
        gamma[..., 0] * normal[..., 0]
        + gamma[..., 1] * normal[..., 1]
        + gamma[..., 2] * normal[..., 2]
    )
    return jnp.sum(gamma_dot_normal / 3.0) / gamma_dot_normal.size


def surface_quantity_of_dofs(
    quantity: Callable[[SurfaceSpec], jax.Array], spec: SurfaceSpec
) -> Callable[[jax.Array], jax.Array]:
    """``quantity`` (e.g. :func:`surface_gamma`) as a function of the native
    DOF vector, the rest of ``spec`` held fixed, for JAX transforms."""
    return lambda dofs: quantity(surface_spec_with_dofs(spec, dofs))
