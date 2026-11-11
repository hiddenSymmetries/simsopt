"""JAX positions, tangents, normals, areas, and volumes from Fourier surface specs.

Supports RZ, XYZ, and XYZ tensor Fourier surfaces on their device.
Finiteness can differ near singularities and float64 range limits.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..backend.jit import jit
from .specs import SurfaceSpec
from .surface_fourier_series import _rotating_frame_components

_TWO_PI = 2.0 * np.pi
_AngleFunction = Callable[[jax.Array, jax.Array], jax.Array]


def _position(spec: SurfaceSpec, phi: jax.Array, theta: jax.Array) -> jax.Array:
    """Return Cartesian positions, shape (nphi, ntheta, 3), at radian angle grids."""
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
    """Return position derivatives in turns, shape (nphi, ntheta, 3)."""

    def d_phi(f: _AngleFunction) -> _AngleFunction:
        return lambda phi, theta: jax.jvp(
            lambda p: f(p, theta), (phi,), (jnp.ones_like(phi),)
        )[1]

    def d_theta(f: _AngleFunction) -> _AngleFunction:
        return lambda phi, theta: jax.jvp(
            lambda t: f(phi, t), (theta,), (jnp.ones_like(theta),)
        )[1]

    derivative: _AngleFunction = lambda phi, theta: _position(spec, phi, theta)
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
    """Return Euclidean norms over the last axis; squared components may underflow."""
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    return jnp.sqrt(x * x + y * y + z * z)


@jit
def surface_gamma(spec: SurfaceSpec) -> jax.Array:
    """Evaluate Cartesian positions on the surface quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Cartesian positions in meters, shape (nphi, ntheta, 3).
    """
    return _position_derivative(spec, 0, 0)


@jit
def surface_gammadash1(spec: SurfaceSpec) -> jax.Array:
    """Evaluate toroidal position derivatives on the surface quadrature grid.

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
    """Evaluate poloidal position derivatives on the surface quadrature grid.

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
    """Evaluate second toroidal position derivatives on the quadrature grid.

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
    """Evaluate mixed toroidal/poloidal position derivatives on the quadrature grid.

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
    """Evaluate second poloidal position derivatives on the quadrature grid.

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
    """Evaluate oriented surface normals on the quadrature grid.

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
    """Evaluate unit surface normals on the quadrature grid.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Dimensionless unit normal; zero or underflowing normals
            produce non-finite entries,
            shape (nphi, ntheta, 3).
    """
    normal = surface_normal(spec)
    return normal / _norm(normal)[..., None]


@jit
def surface_area(spec: SurfaceSpec) -> jax.Array:
    """Evaluate the surface area by quadrature.

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
    """Evaluate the signed enclosed volume by quadrature.

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
