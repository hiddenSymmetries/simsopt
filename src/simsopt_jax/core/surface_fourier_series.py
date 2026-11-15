"""Fourier surface coefficient evaluation and native DOF layouts in JAX.

Provides rotating-frame components and immutable DOF updates for the
RZ, XYZ, and XYZ tensor Fourier surface specifications.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ..backend.jit import jit
from .specs import (
    SurfaceRZFourierSpec,
    SurfaceSpec,
    SurfaceSpecT,
    SurfaceXYZFourierSpec,
    SurfaceXYZTensorFourierSpec,
)


def _angle_difference_series(
    spec: SurfaceRZFourierSpec | SurfaceXYZFourierSpec,
    phi: jax.Array,
    theta: jax.Array,
    coefficients: jax.Array,
    sine: bool,
) -> jax.Array:
    """Evaluate sine/cosine modes on radian grids phi (nphi,) and theta (ntheta,).

    Returns shape (nphi, ntheta).
    """
    mpol, ntor = coefficients.shape[0] - 1, (coefficients.shape[1] - 1) // 2
    m = np.arange(mpol + 1, dtype=np.float64)
    nfp_n = spec.nfp * np.arange(-ntor, ntor + 1, dtype=np.float64)

    cos_theta = jnp.cos(theta[:, None] * m[None, :])
    sin_theta = jnp.sin(theta[:, None] * m[None, :])
    cos_phi = jnp.cos(phi[:, None] * nfp_n[None, :])
    sin_phi = jnp.sin(phi[:, None] * nfp_n[None, :])
    # cos(a - b) = cos a cos b + sin a sin b; sin(a - b) = sin a cos b - cos a sin b.
    if sine:
        return (
            cos_phi @ (sin_theta @ coefficients).T
            - sin_phi @ (cos_theta @ coefficients).T
        )
    return (
        cos_phi @ (cos_theta @ coefficients).T + sin_phi @ (sin_theta @ coefficients).T
    )


def _series_pair(
    spec: SurfaceRZFourierSpec | SurfaceXYZFourierSpec,
    phi: jax.Array,
    theta: jax.Array,
    cosine: jax.Array,
    sine: jax.Array,
) -> jax.Array:
    return _angle_difference_series(
        spec, phi, theta, cosine, False
    ) + _angle_difference_series(spec, phi, theta, sine, True)


def _rz_fourier_components(
    spec: SurfaceRZFourierSpec, phi: jax.Array, theta: jax.Array
) -> tuple[jax.Array, None, jax.Array]:
    if spec.stellsym:
        return (
            _angle_difference_series(spec, phi, theta, spec.rc, False),
            None,
            _angle_difference_series(spec, phi, theta, spec.zs, True),
        )
    return (
        _series_pair(spec, phi, theta, spec.rc, spec.rs),
        None,
        _series_pair(spec, phi, theta, spec.zc, spec.zs),
    )


def _harmonic_basis(angles: jax.Array, count: int, frequency: int) -> jax.Array:
    """Return constant, cosine, then sine modes of frequency-scaled radian angles.

    The basis has shape (npoints, 2 * count + 1).
    """
    cosine_modes = frequency * np.arange(count + 1, dtype=np.float64)
    sine_modes = frequency * np.arange(1, count + 1, dtype=np.float64)
    return jnp.concatenate(
        (
            jnp.cos(angles[:, None] * cosine_modes[None, :]),
            jnp.sin(angles[:, None] * sine_modes[None, :]),
        ),
        axis=1,
    )


def _tensor_keep_masks(
    spec: SurfaceXYZTensorFourierSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return component masks selecting active tensor Fourier coefficients."""
    shape = spec.xcs.shape
    if not spec.stellsym:
        everything = np.ones(shape, dtype=bool)
        return everything, everything, everything
    mpol, ntor = (shape[0] - 1) // 2, (shape[1] - 1) // 2
    cosine_theta = np.arange(shape[0])[:, None] <= mpol
    cosine_phi = np.arange(shape[1])[None, :] <= ntor
    keep_x = cosine_theta == cosine_phi
    return keep_x, ~keep_x, ~keep_x


def _clamping_factor(
    spec: SurfaceXYZTensorFourierSpec, phi: jax.Array, theta: jax.Array
) -> jax.Array:
    """Return the tensor-surface clamping factor, shape (nphi, ntheta)."""
    sin_phi = jnp.sin(spec.nfp * phi / 2.0)[:, None]
    sin_theta = jnp.sin(theta / 2.0)[None, :]
    return sin_phi * sin_phi + sin_theta * sin_theta


def _xyz_tensor_fourier_components(
    spec: SurfaceXYZTensorFourierSpec, phi: jax.Array, theta: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mpol, ntor = (spec.xcs.shape[0] - 1) // 2, (spec.xcs.shape[1] - 1) // 2
    phi_basis = _harmonic_basis(phi, ntor, spec.nfp)
    theta_basis = _harmonic_basis(theta, mpol, 1)

    def series(coefficients: jax.Array) -> jax.Array:
        return (phi_basis @ coefficients.T) @ theta_basis.T

    clamped_block = np.zeros(spec.xcs.shape, dtype=bool)
    clamped_block[: mpol + 1, : ntor + 1] = True
    components = []
    for coefficients, keep, clamped in zip(
        (spec.xcs, spec.ycs, spec.zcs), _tensor_keep_masks(spec), spec.clamped_dims
    ):
        summed = jnp.where(keep, coefficients, 0.0)
        if not clamped:
            components.append(series(summed))
            continue
        components.append(
            series(jnp.where(clamped_block, 0.0, summed))
            + _clamping_factor(spec, phi, theta)
            * series(jnp.where(clamped_block, summed, 0.0))
        )
    return tuple(components)


def _rotating_frame_components(
    spec: SurfaceSpec, phi: jax.Array, theta: jax.Array
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    """Return rotating-frame components at phi (nphi,) and theta (ntheta,) in radians.

    Each has shape (nphi, ntheta); yhat is None for SurfaceRZFourier.
    """
    if isinstance(spec, SurfaceRZFourierSpec):
        return _rz_fourier_components(spec, phi, theta)
    if isinstance(spec, SurfaceXYZFourierSpec):
        return (
            _series_pair(spec, phi, theta, spec.xc, spec.xs),
            _series_pair(spec, phi, theta, spec.yc, spec.ys),
            _series_pair(spec, phi, theta, spec.zc, spec.zs),
        )
    return _xyz_tensor_fourier_components(spec, phi, theta)


# The arrays holding the native DOFs of the angle-difference series, in
# get_dofs() order, by spec class and stellsym.
_DOF_ARRAYS = {
    (SurfaceRZFourierSpec, True): ("rc", "zs"),
    (SurfaceRZFourierSpec, False): ("rc", "rs", "zc", "zs"),
    (SurfaceXYZFourierSpec, True): ("xc", "ys", "zs"),
    (SurfaceXYZFourierSpec, False): ("xc", "xs", "yc", "ys", "zc", "zs"),
}


def _dof_layout(spec: SurfaceSpec) -> tuple[tuple[str, np.ndarray], ...]:
    """Return coefficient fields and row-major DOF indices in native get_dofs order."""
    if isinstance(spec, SurfaceXYZTensorFourierSpec):
        return tuple(
            (name, np.flatnonzero(keep))
            for name, keep in zip(("xcs", "ycs", "zcs"), _tensor_keep_masks(spec))
        )
    # Every row-major entry from m = 1 on, and the m = 0 entries with n >= 0
    # (cosine arrays, names ending in c) or n > 0 (sine arrays, ending in s).
    names = _DOF_ARRAYS[(type(spec), spec.stellsym)]
    size = getattr(spec, names[0]).size
    ntor = (getattr(spec, names[0]).shape[1] - 1) // 2
    return tuple(
        (name, np.arange(ntor if name.endswith("c") else ntor + 1, size)) for name in names
    )


@jit
def surface_get_dofs(spec: SurfaceSpec) -> jax.Array:
    """Return the full native coefficient vector, including fixed DOFs.

    Args:
        spec (SurfaceSpec): Immutable Fourier coefficients in meters and quadrature
            points in turns.

    Returns:
        jax.Array: Shape (ndofs,) native coefficient vector in meters, including fixed
            DOFs.
    """
    return jnp.concatenate(
        [getattr(spec, name).reshape(-1)[indices] for name, indices in _dof_layout(spec)]
    )


@jit
def surface_spec_with_dofs(spec: SurfaceSpecT, dofs: jax.Array) -> SurfaceSpecT:
    """Return a surface spec with its full native DOF vector replaced.

    Args:
        spec (SurfaceSpecT): Original immutable surface state.
        dofs (jax.Array): Shape (ndofs,) full native coefficient vector in meters,
            including fixed DOFs.

    Returns:
        SurfaceSpecT: New spec of the same class; non-DOF coefficients and quadrature
            are unchanged.
    """
    layout = _dof_layout(spec)
    blocks = jnp.split(dofs, np.cumsum([indices.size for _, indices in layout])[:-1].tolist())
    updated = {}
    for (name, indices), block in zip(layout, blocks):
        coefficients = getattr(spec, name)
        updated[name] = (
            coefficients.reshape(-1)
            .at[indices]
            .set(block.reshape(indices.shape), indices_are_sorted=True, unique_indices=True)
            .reshape(coefficients.shape)
        )
    return replace(spec, **updated)
