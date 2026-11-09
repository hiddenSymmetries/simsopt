"""JAX geometry of native simsopt surfaces, evaluated from their specs.

``surface_<method>(spec)`` returns native ``surface.<method>()`` for the
``SurfaceRZFourier``, ``SurfaceXYZFourier`` or ``SurfaceXYZTensorFourier``
whose state the spec holds, with the same conventions and array shapes:
``gamma``, its derivatives ``gammadash1``, ``gammadash2``, ``gammadash1dash1``,
``gammadash1dash2``, ``gammadash2dash2`` with respect to ``quadpoints_phi``
and ``quadpoints_theta``, ``normal``, ``unitnormal``, ``area`` and ``volume``.
The angle derivatives are nested ``jax.jvp`` of ``gamma`` in the angles
``phi = 2 pi quadpoints_phi`` and ``theta = 2 pi quadpoints_theta``, in
radians, scaled once by ``(2 pi)**k`` for a derivative of order ``k``, as
native scales its sums. Each is jitted with the spec's arrays as traced
operands, so new coefficient values reuse the compiled program; pass arrays
already on the intended device
(:func:`simsopt_jax_adapters.geo.surface_spec_from_surface` does).

Coefficient derivatives are JAX transforms of :func:`surface_quantity_of_dofs`
at the native DOF vector, fixed DOFs included, which
:func:`~simsopt_jax.core.surface_fourier_series.surface_get_dofs` reads from a
spec. The DOF axes come last, as natively, so ``Derivative({surface: ...})``
projects them onto the free DOFs. Jit them with ``simsopt_jax.backend.jit.jit``,
passing the spec as an argument::

    @jit
    def dgamma_by_dcoeff(spec):
        dofs = surface_get_dofs(spec)
        return jax.jacfwd(surface_quantity_of_dofs(surface_gamma, spec))(dofs)

``jax.vjp`` gives ``dgamma_by_dcoeff_vjp``; ``jax.grad`` and ``jax.hessian``
of ``surface_area`` give ``darea_by_dcoeff`` and ``d2area_by_dcoeffdcoeff``.

Where squared normal components, or derivative intermediates such as the cube of
the normal's length, underflow or overflow float64, unit normals, areas and
their derivatives can differ from native in finiteness: XLA on CPU flushes
subnormal results to zero, and autodiff orders the arithmetic differently from
native's hand-written derivatives. At or next to a singular point the two can
also differ by round-off: one side's normal can cancel to exactly zero, giving
non-finite unit normals and area gradients, while the other's is a tiny nonzero
vector with finite ones.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..backend.jit import jit
from .specs import SurfaceSpec
from .surface_fourier_series import (
    _rotating_frame_components,
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
_AngleFunction = Callable[[jax.Array, jax.Array], jax.Array]


def _position(spec: SurfaceSpec, phi: jax.Array, theta: jax.Array) -> jax.Array:
    """gamma at the toroidal angles ``phi``, shape ``(nphi,)``, and poloidal
    angles ``theta``, shape ``(ntheta,)``, in radians; shape ``(nphi, ntheta,
    3)``: the rotating-frame components turned by ``phi``."""
    cos_phi, sin_phi = jnp.cos(phi)[:, None], jnp.sin(phi)[:, None]
    x_hat, y_hat, z = _rotating_frame_components(spec, phi, theta)
    if y_hat is None:
        return jnp.stack((x_hat * cos_phi, x_hat * sin_phi, z), axis=-1)
    return jnp.stack(
        (x_hat * cos_phi - y_hat * sin_phi, x_hat * sin_phi + y_hat * cos_phi, z),
        axis=-1,
    )


def _position_derivative(
    spec: SurfaceSpec, phi_order: int, theta_order: int
) -> jax.Array:
    """``d^(phi_order + theta_order) gamma / dquadpoints_phi^phi_order
    dquadpoints_theta^theta_order``, shape ``(nphi, ntheta, 3)``.

    Nested forward-mode JAX derivatives of :func:`_position` in the angles
    ``phi = 2 pi quadpoints_phi`` and ``theta = 2 pi quadpoints_theta``, times
    the chain factor ``(2 pi)**(phi_order + theta_order)``. Each grid point
    depends on one ``phi`` and one ``theta`` entry, so a tangent of ones gives
    the partial derivative at every point.
    """

    def position(phi: jax.Array, theta: jax.Array) -> jax.Array:
        return _position(spec, phi, theta)

    def d_phi(f: _AngleFunction) -> _AngleFunction:
        return lambda phi, theta: jax.jvp(
            lambda p: f(p, theta), (phi,), (jnp.ones_like(phi),)
        )[1]

    def d_theta(f: _AngleFunction) -> _AngleFunction:
        return lambda phi, theta: jax.jvp(
            lambda t: f(phi, t), (theta,), (jnp.ones_like(theta),)
        )[1]

    derivative = position
    for _ in range(theta_order):
        derivative = d_theta(derivative)
    for _ in range(phi_order):
        derivative = d_phi(derivative)
    angle_derivative = derivative(
        _TWO_PI * spec.quadpoints_phi, _TWO_PI * spec.quadpoints_theta
    )
    # Native's order (sum in radians, then scale) keeps a sum that cancels exactly
    # in radians exact after scaling. The sums themselves round differently from
    # native's, so a near-cancellation can be exactly zero on one side only.
    return _TWO_PI ** (phi_order + theta_order) * angle_derivative


def _norm(vectors: jax.Array) -> jax.Array:
    """``sqrt(x*x + y*y + z*z)`` over the last axis, the native expression.

    Zero normals, and normals whose squares underflow past the subnormal range,
    therefore give non-finite unit normals and area gradients, as natively for
    the same normal.
    """
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    return jnp.sqrt(x * x + y * y + z * z)


@jit
def surface_gamma(spec: SurfaceSpec) -> jax.Array:
    """Evaluate gamma on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Cartesian positions in meters, shape (nphi, ntheta, 3).
    """
    return _position_derivative(spec, 0, 0)


@jit
def surface_gammadash1(spec: SurfaceSpec) -> jax.Array:
    """Evaluate gammadash1 on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Position derivative with respect to toroidal turns, in meters per
            turn, shape (nphi, ntheta, 3).
    """
    return _position_derivative(spec, 1, 0)


@jit
def surface_gammadash2(spec: SurfaceSpec) -> jax.Array:
    """Evaluate gammadash2 on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Position derivative with respect to poloidal turns, in meters per
            turn, shape (nphi, ntheta, 3).
    """
    return _position_derivative(spec, 0, 1)


@jit
def surface_gammadash1dash1(spec: SurfaceSpec) -> jax.Array:
    """Evaluate gammadash1dash1 on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Second position derivative with respect to toroidal turns, in meters
            per turn squared, shape (nphi, ntheta, 3).
    """
    return _position_derivative(spec, 2, 0)


@jit
def surface_gammadash1dash2(spec: SurfaceSpec) -> jax.Array:
    """Evaluate gammadash1dash2 on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Mixed position derivative with respect to toroidal and poloidal
            turns, in meters per turn squared, shape (nphi, ntheta, 3).
    """
    return _position_derivative(spec, 1, 1)


@jit
def surface_gammadash2dash2(spec: SurfaceSpec) -> jax.Array:
    """Evaluate gammadash2dash2 on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Second position derivative with respect to poloidal turns, in meters
            per turn squared, shape (nphi, ntheta, 3).
    """
    return _position_derivative(spec, 0, 2)


@jit
def surface_normal(spec: SurfaceSpec) -> jax.Array:
    """Evaluate normal on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Oriented cross product of the toroidal and poloidal tangents, in
            square meters per turn squared, shape (nphi, ntheta, 3).
    """
    return jnp.cross(surface_gammadash1(spec), surface_gammadash2(spec))


@jit
def surface_unitnormal(spec: SurfaceSpec) -> jax.Array:
    """Evaluate unitnormal on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Dimensionless unit normal; zero normals produce nonfinite entries,
            shape (nphi, ntheta, 3).
    """
    normal = surface_normal(spec)
    return normal / _norm(normal)[..., None]


@jit
def surface_area(spec: SurfaceSpec) -> jax.Array:
    """Evaluate area on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Quadrature mean of the normal magnitude, in square meters, shape ().
    """
    norms = _norm(surface_normal(spec))
    return jnp.sum(norms) / norms.size


@jit
def surface_volume(spec: SurfaceSpec) -> jax.Array:
    """Evaluate volume on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Signed quadrature mean of gamma dot normal divided by three, in cubic
            meters, shape ().
    """
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
    DOF vector, the rest of ``spec`` held fixed, for JAX transforms.

    Args:
        quantity (Callable[[SurfaceSpec], jax.Array]): Geometry kernel with its own
            output shape and units.
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        Callable[[jax.Array], jax.Array]: Function of the full shape (ndofs,)
            coefficient vector, with the kernel's output shape and units; the remaining
            spec state is fixed.
    """
    return lambda dofs: quantity(surface_spec_with_dofs(spec, dofs))
