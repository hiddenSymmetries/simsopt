"""Coefficient series of the native Fourier surface classes, from their specs.

:func:`_rotating_frame_derivative` differentiates a surface's components
``(xhat, yhat, z)`` in the frame that turns with the toroidal angle
(``x = xhat cos(phi) - yhat sin(phi)``, ``y = xhat sin(phi) + yhat cos(phi)``;
``SurfaceRZFourier`` has ``xhat = r``, ``yhat = 0``), summing exactly the
coefficient entries the native class sums: ``SurfaceRZFourier`` drops ``rs``
and ``zc`` when stellarator symmetric, ``SurfaceXYZFourier`` sums all six
arrays, ``SurfaceXYZTensorFourier`` skips the stellarator-symmetry entries it
holds no DOF for. :func:`surface_get_dofs` and :func:`surface_spec_with_dofs`
are native ``get_dofs`` and ``set_dofs``.

``SurfaceXYZTensorFourier`` clamping multiplies the cosine-cosine block of a
clamped component by ``sin(nfp phi / 2)**2 + sin(theta / 2)**2``. Native
``simsoptpp`` differentiates that factor twice with integer division
(``nfp / 2`` and ``1 / 2``), so its clamped ``gammadash1dash1`` (odd ``nfp``)
and ``gammadash2dash2`` are not the derivatives of its ``gammadash1`` and
``gammadash2``; the derivatives here are.
"""

from __future__ import annotations

from dataclasses import replace
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from .specs import (
    SurfaceRZFourierSpec,
    SurfaceSpec,
    SurfaceSpecT,
    SurfaceXYZFourierSpec,
    SurfaceXYZTensorFourierSpec,
)

__all__ = [
    "surface_get_dofs",
    "surface_spec_with_dofs",
]

_TWO_PI = 2.0 * np.pi
# Phases p of f_p(a) = cos(a + p pi / 2): f_0 = cos, f_3 = sin, d f_p / da = f_{p+1}.
_COSINE = 0
_SINE = 3


def _phase_sign(phase: int) -> float:
    """``f_phase = sign * (cos if phase is even else sin)``."""
    return 1.0 if phase % 4 in (0, 3) else -1.0


def _angle_difference_series(
    spec: SurfaceRZFourierSpec | SurfaceXYZFourierSpec,
    coefficients: jax.Array,
    phase: int,
    phi_order: int,
    theta_order: int,
) -> jax.Array:
    """``d^(phi_order + theta_order) / dphi^phi_order dtheta^theta_order`` of
    ``sum_{m,n} coefficients[m, n + ntor] f_phase(m theta - nfp n phi)``.

    The angles ``phi = 2 pi quadpoints_phi`` and ``theta = 2 pi
    quadpoints_theta`` are in radians; the result has shape ``(nphi,
    ntheta)``. Each derivative multiplies a mode by ``m`` (theta) or
    ``-nfp n`` (phi) and advances its phase; the angle difference expands
    into separate theta and phi tables.
    """
    mpol, ntor = coefficients.shape[0] - 1, (coefficients.shape[1] - 1) // 2
    m = np.arange(mpol + 1, dtype=np.float64)
    nfp_n = spec.nfp * np.arange(-ntor, ntor + 1, dtype=np.float64)
    phase = (phase + phi_order + theta_order) % 4
    weights = (
        _phase_sign(phase) * m[:, None] ** theta_order * (-nfp_n)[None, :] ** phi_order
    )
    weighted = coefficients * weights

    theta = _TWO_PI * spec.quadpoints_theta
    phi = _TWO_PI * spec.quadpoints_phi
    cos_theta = jnp.cos(theta[:, None] * m[None, :])
    sin_theta = jnp.sin(theta[:, None] * m[None, :])
    cos_phi = jnp.cos(phi[:, None] * nfp_n[None, :])
    sin_phi = jnp.sin(phi[:, None] * nfp_n[None, :])
    # cos(a - b) = cos a cos b + sin a sin b; sin(a - b) = sin a cos b - cos a sin b.
    if phase % 2 == 0:
        return cos_phi @ (cos_theta @ weighted).T + sin_phi @ (sin_theta @ weighted).T
    return cos_phi @ (sin_theta @ weighted).T - sin_phi @ (cos_theta @ weighted).T


def _series_pair(
    spec: SurfaceRZFourierSpec | SurfaceXYZFourierSpec,
    cosine: jax.Array,
    sine: jax.Array,
    phi_order: int,
    theta_order: int,
) -> jax.Array:
    return _angle_difference_series(
        spec, cosine, _COSINE, phi_order, theta_order
    ) + _angle_difference_series(spec, sine, _SINE, phi_order, theta_order)


def _rz_fourier_components(
    spec: SurfaceRZFourierSpec, phi_order: int, theta_order: int
) -> tuple[jax.Array, None, jax.Array]:
    if spec.stellsym:
        return (
            _angle_difference_series(spec, spec.rc, _COSINE, phi_order, theta_order),
            None,
            _angle_difference_series(spec, spec.zs, _SINE, phi_order, theta_order),
        )
    return (
        _series_pair(spec, spec.rc, spec.rs, phi_order, theta_order),
        None,
        _series_pair(spec, spec.zc, spec.zs, phi_order, theta_order),
    )


def _xyz_fourier_components(
    spec: SurfaceXYZFourierSpec, phi_order: int, theta_order: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        _series_pair(spec, spec.xc, spec.xs, phi_order, theta_order),
        _series_pair(spec, spec.yc, spec.ys, phi_order, theta_order),
        _series_pair(spec, spec.zc, spec.zs, phi_order, theta_order),
    )


def _harmonic_basis(
    quadpoints: jax.Array, count: int, frequency: int, order: int
) -> jax.Array:
    """``d^order / da^order`` of the native tensor basis at ``a = 2 pi quadpoints``.

    The basis is ``1, cos(k a), ..., cos(count k a), sin(k a), ...,
    sin(count k a)`` with ``k = frequency``; shape ``(npoints, 2 count + 1)``.
    """
    angles = _TWO_PI * quadpoints
    cosine_modes = frequency * np.arange(count + 1, dtype=np.float64)
    sine_modes = frequency * np.arange(1, count + 1, dtype=np.float64)
    blocks = []
    for modes, phase in ((cosine_modes, _COSINE + order), (sine_modes, _SINE + order)):
        trig = jnp.cos if phase % 2 == 0 else jnp.sin
        weights = _phase_sign(phase) * modes**order
        blocks.append(weights[None, :] * trig(angles[:, None] * modes[None, :]))
    return jnp.concatenate(blocks, axis=1)


def _tensor_keep_masks(
    spec: SurfaceXYZTensorFourierSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native ``!skip(dim, m, n)`` for ``x``, ``y`` and ``z``: the summed entries."""
    shape = spec.xcs.shape
    if not spec.stellsym:
        everything = np.ones(shape, dtype=bool)
        return everything, everything, everything
    mpol, ntor = (shape[0] - 1) // 2, (shape[1] - 1) // 2
    cosine_theta = np.arange(shape[0])[:, None] <= mpol
    cosine_phi = np.arange(shape[1])[None, :] <= ntor
    keep_x = cosine_theta == cosine_phi
    return keep_x, ~keep_x, ~keep_x


def _clamping_derivative(
    spec: SurfaceXYZTensorFourierSpec, phi_order: int, theta_order: int
) -> jax.Array:
    """``d^(phi_order + theta_order)`` of ``sin(nfp phi / 2)**2 + sin(theta / 2)**2``.

    Broadcastable to ``(nphi, ntheta)``; mixed derivatives vanish.
    """
    half_phi = spec.nfp * (_TWO_PI * spec.quadpoints_phi) / 2.0
    half_theta = (_TWO_PI * spec.quadpoints_theta) / 2.0
    sin_phi, cos_phi = jnp.sin(half_phi)[:, None], jnp.cos(half_phi)[:, None]
    sin_theta, cos_theta = jnp.sin(half_theta)[None, :], jnp.cos(half_theta)[None, :]
    derivatives = {
        (0, 0): sin_phi * sin_phi + sin_theta * sin_theta,
        (1, 0): spec.nfp * cos_phi * sin_phi,
        (2, 0): spec.nfp * (spec.nfp / 2.0) * (cos_phi * cos_phi - sin_phi * sin_phi),
        (0, 1): cos_theta * sin_theta,
        (0, 2): 0.5 * (cos_theta * cos_theta - sin_theta * sin_theta),
    }
    return derivatives[(phi_order, theta_order)]


def _xyz_tensor_fourier_components(
    spec: SurfaceXYZTensorFourierSpec, phi_order: int, theta_order: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mpol, ntor = (spec.xcs.shape[0] - 1) // 2, (spec.xcs.shape[1] - 1) // 2

    def series(coefficients: jax.Array, phi_order: int, theta_order: int) -> jax.Array:
        phi_basis = _harmonic_basis(spec.quadpoints_phi, ntor, spec.nfp, phi_order)
        theta_basis = _harmonic_basis(spec.quadpoints_theta, mpol, 1, theta_order)
        return (phi_basis @ coefficients.T) @ theta_basis.T

    clamped_block = np.zeros(spec.xcs.shape, dtype=bool)
    clamped_block[: mpol + 1, : ntor + 1] = True
    components = []
    for coefficients, keep, clamped in zip(
        (spec.xcs, spec.ycs, spec.zcs), _tensor_keep_masks(spec), spec.clamped_dims
    ):
        summed = jnp.where(keep, coefficients, 0.0)
        if not clamped:
            components.append(series(summed, phi_order, theta_order))
            continue
        # Leibniz rule for the clamping factor times the cosine-cosine block.
        block = jnp.where(clamped_block, summed, 0.0)
        terms = [series(jnp.where(clamped_block, 0.0, summed), phi_order, theta_order)]
        for factor_phi in range(phi_order + 1):
            for factor_theta in range(theta_order + 1):
                if factor_phi and factor_theta:
                    continue
                weight = comb(phi_order, factor_phi) * comb(theta_order, factor_theta)
                terms.append(
                    weight
                    * _clamping_derivative(spec, factor_phi, factor_theta)
                    * series(block, phi_order - factor_phi, theta_order - factor_theta)
                )
        components.append(sum(terms))
    return tuple(components)


def _rotating_frame_derivative(
    spec: SurfaceSpec, phi_order: int, theta_order: int
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    """``d^(phi_order + theta_order) / dphi^phi_order dtheta^theta_order`` of
    ``(xhat, yhat, z)`` in radians, each ``(nphi, ntheta)``; ``yhat`` is
    ``None`` for ``SurfaceRZFourier``. Orders up to 2 in each angle.
    """
    if isinstance(spec, SurfaceRZFourierSpec):
        return _rz_fourier_components(spec, phi_order, theta_order)
    if isinstance(spec, SurfaceXYZFourierSpec):
        return _xyz_fourier_components(spec, phi_order, theta_order)
    return _xyz_tensor_fourier_components(spec, phi_order, theta_order)


# The arrays holding the native DOFs of the angle-difference series, in
# get_dofs() order, by spec class and stellsym.
_DOF_ARRAYS = {
    (SurfaceRZFourierSpec, True): ("rc", "zs"),
    (SurfaceRZFourierSpec, False): ("rc", "rs", "zc", "zs"),
    (SurfaceXYZFourierSpec, True): ("xc", "ys", "zs"),
    (SurfaceXYZFourierSpec, False): ("xc", "xs", "yc", "ys", "zc", "zs"),
}


def _dof_layout(spec: SurfaceSpec) -> tuple[tuple[str, np.ndarray], ...]:
    """The coefficient fields holding the native DOFs, in ``get_dofs()`` order,
    each with the row-major indices of its DOF entries."""
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


@jax.jit
def surface_get_dofs(spec: SurfaceSpec) -> jax.Array:
    """The native ``get_dofs()`` vector, fixed DOFs included.

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


@jax.jit
def surface_spec_with_dofs(spec: SurfaceSpecT, dofs: jax.Array) -> SurfaceSpecT:
    """``spec`` with its DOFs replaced by ``dofs``, as native ``set_dofs``.

    Coefficient entries that are not DOFs keep their values.

    Args:
        spec (SurfaceSpecT): Original immutable surface state.
        dofs (jax.Array): Shape (ndofs,) full native coefficient vector in meters,
            including fixed DOFs.

    Returns:
        SurfaceSpecT: New surface-spec object of the same class, with non-DOF coefficients and
            quadrature unchanged.
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
