# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
JAX-backed VMEC diagnostics.

This module mirrors a small subset of :mod:`simsopt.mhd.vmec_diagnostics`
without depending on the Fortran VMEC wrapper. The initial scope is a
VMEC-only quasisymmetry metric that can be evaluated from a ``VmecJax``
equilibrium or from a VMEC-style ``wout`` object.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional dependency
    jnp = None
    _import_error = exc
else:
    _import_error = None


__all__ = [
    "QuasisymmetryRatioResidualJax",
    "quasisymmetry_ratio_residual_from_wout_jax",
    "quasisymmetry_ratio_residual_from_state_jax",
]


def _require_jax():
    """Raise a helpful error if the optional JAX dependency is unavailable."""
    if jnp is None:
        raise ImportError(
            "vmec_diagnostics_jax requires JAX. "
            "Install jax/jaxlib before using these helpers."
        ) from _import_error


def _as_surface_array(surfaces) -> jnp.ndarray:
    """Normalize a scalar or iterable surface list into a 1D JAX array."""
    try:
        values = list(surfaces)  # type: ignore[arg-type]
    except Exception:
        values = [surfaces]
    return jnp.asarray(values, dtype=jnp.float64)


def _as_jax_array(values, *, dtype=None):
    """Convert possibly big-endian VMEC arrays into native-endian JAX arrays."""
    try:
        arr = np.asarray(values)
    except Exception:
        return jnp.asarray(values, dtype=dtype)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    elif arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    return jnp.asarray(arr)


def _as_weight_array(weights, nsurf: int) -> jnp.ndarray:
    """Return a 1D weight vector for the requested number of surfaces."""
    if weights is None:
        return jnp.ones((nsurf,), dtype=jnp.float64)
    return jnp.asarray(list(weights), dtype=jnp.float64)


def _half_grid(radial_count: int, dtype) -> jnp.ndarray:
    """Construct the canonical VMEC half-grid for a given radial resolution."""
    if radial_count < 2:
        return jnp.zeros((0,), dtype=dtype)
    s_full = jnp.linspace(0.0, 1.0, radial_count, dtype=dtype)
    return 0.5 * (s_full[:-1] + s_full[1:])


def _interp_half_grid(samples, surfaces, s_half):
    """Linearly interpolate samples tabulated on the VMEC half grid."""
    samples = jnp.asarray(samples)
    surfaces = jnp.asarray(surfaces, dtype=samples.dtype)
    s_half = jnp.asarray(s_half, dtype=samples.dtype)
    if s_half.shape[0] == 0:
        raise ValueError("half-grid interpolation requires at least one radial point")
    if s_half.shape[0] == 1:
        return jnp.broadcast_to(samples[:1], (surfaces.shape[0],) + samples.shape[1:])

    idx_hi = jnp.searchsorted(s_half, surfaces, side="left")
    idx_hi = jnp.clip(idx_hi, 1, s_half.shape[0] - 1)
    idx_lo = idx_hi - 1
    x0 = s_half[idx_lo]
    x1 = s_half[idx_hi]
    denom = jnp.where(x1 != x0, x1 - x0, jnp.ones_like(x1))
    t = ((surfaces - x0) / denom).reshape((surfaces.shape[0],) + (1,) * (samples.ndim - 1))
    y0 = samples[idx_lo]
    y1 = samples[idx_hi]
    return y0 + t * (y1 - y0)


def _radial_mode_matrix(values, *, radial_count: int, mode_count: int) -> jnp.ndarray:
    """Normalize VMEC coefficient storage to shape ``(radial, mode)``."""
    arr = _as_jax_array(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a rank-2 coefficient array, got shape {arr.shape}")
    if arr.shape == (radial_count, mode_count):
        return arr
    if arr.shape == (mode_count, radial_count):
        return jnp.swapaxes(arr, 0, 1)
    raise ValueError(
        f"unexpected coefficient shape {arr.shape}; expected {(radial_count, mode_count)} "
        f"or {(mode_count, radial_count)}"
    )


def quasisymmetry_ratio_residual_from_wout_jax(
    wout,
    *,
    surfaces,
    helicity_m: int = 1,
    helicity_n: int = 0,
    weights=None,
    ntheta: int = 63,
    nphi: int = 64,
):
    r"""Evaluate the VMEC-only quasisymmetry residual from a ``wout`` object.

    This implements the same metric as
    :class:`simsopt.mhd.vmec_diagnostics.QuasisymmetryRatioResidual`, but uses
    JAX arrays and accepts coefficient storage from either the classic VMEC
    Python wrapper or :class:`simsopt.mhd.VmecJax`.
    """
    _require_jax()

    if bool(getattr(wout, "lasym", False)):
        raise RuntimeError("QuasisymmetryRatioResidualJax does not yet support lasym=True")

    surfaces = _as_surface_array(surfaces)
    weights = _as_weight_array(weights, int(surfaces.shape[0]))
    if weights.shape[0] != surfaces.shape[0]:
        raise ValueError("weights must have the same length as surfaces")

    ntheta = int(ntheta)
    nphi = int(nphi)
    nfp = int(getattr(wout, "nfp"))
    helicity_m = int(helicity_m)
    helicity_n = int(helicity_n)

    iotas_full = _as_jax_array(getattr(wout, "iotas"), dtype=np.float64)
    radial_count = int(iotas_full.shape[0])
    s_half = _half_grid(radial_count, iotas_full.dtype)
    half_count = int(s_half.shape[0])
    mode_count = int(_as_jax_array(getattr(wout, "xm_nyq"), dtype=np.float64).shape[0])

    iota = _interp_half_grid(iotas_full[1:], surfaces, s_half)
    G = _interp_half_grid(_as_jax_array(getattr(wout, "bvco"), dtype=np.float64)[1:], surfaces, s_half)
    I = _interp_half_grid(_as_jax_array(getattr(wout, "buco"), dtype=np.float64)[1:], surfaces, s_half)

    gmnc = _interp_half_grid(
        _radial_mode_matrix(getattr(wout, "gmnc"), radial_count=radial_count, mode_count=mode_count)[1:],
        surfaces,
        s_half,
    )
    bmnc = _interp_half_grid(
        _radial_mode_matrix(getattr(wout, "bmnc"), radial_count=radial_count, mode_count=mode_count)[1:],
        surfaces,
        s_half,
    )
    bsubumnc = _interp_half_grid(
        _radial_mode_matrix(getattr(wout, "bsubumnc"), radial_count=radial_count, mode_count=mode_count)[1:],
        surfaces,
        s_half,
    )
    bsubvmnc = _interp_half_grid(
        _radial_mode_matrix(getattr(wout, "bsubvmnc"), radial_count=radial_count, mode_count=mode_count)[1:],
        surfaces,
        s_half,
    )
    bsupumnc = _interp_half_grid(
        _radial_mode_matrix(getattr(wout, "bsupumnc"), radial_count=radial_count, mode_count=mode_count)[1:],
        surfaces,
        s_half,
    )
    bsupvmnc = _interp_half_grid(
        _radial_mode_matrix(getattr(wout, "bsupvmnc"), radial_count=radial_count, mode_count=mode_count)[1:],
        surfaces,
        s_half,
    )

    theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False, dtype=jnp.float64)
    phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nphi, endpoint=False, dtype=jnp.float64)
    theta2d, phi2d = jnp.meshgrid(theta1d, phi1d, indexing="ij")

    xm_nyq = _as_jax_array(getattr(wout, "xm_nyq"), dtype=np.float64)
    xn_nyq = _as_jax_array(getattr(wout, "xn_nyq"), dtype=np.float64)
    angle = theta2d[:, :, None] * xm_nyq[None, None, :] - phi2d[:, :, None] * xn_nyq[None, None, :]
    cosangle = jnp.cos(angle)
    sinangle = jnp.sin(angle)

    modB = jnp.einsum("sm,tpm->stp", bmnc, cosangle)
    d_B_d_theta = jnp.einsum("sm,tpm,m->stp", bmnc, -sinangle, xm_nyq)
    d_B_d_phi = jnp.einsum("sm,tpm,m->stp", bmnc, sinangle, xn_nyq)
    sqrtg = jnp.einsum("sm,tpm->stp", gmnc, cosangle)
    bsubu = jnp.einsum("sm,tpm->stp", bsubumnc, cosangle)
    bsubv = jnp.einsum("sm,tpm->stp", bsubvmnc, cosangle)
    bsupu = jnp.einsum("sm,tpm->stp", bsupumnc, cosangle)
    bsupv = jnp.einsum("sm,tpm->stp", bsupvmnc, cosangle)

    d_psi_d_s = -_as_jax_array(getattr(wout, "phi"), dtype=np.float64)[-1] / (2.0 * jnp.pi)
    sqrtg_safe = jnp.where(sqrtg != 0.0, sqrtg, jnp.ones_like(sqrtg))
    B_dot_grad_B = bsupu * d_B_d_theta + bsupv * d_B_d_phi
    B_cross_grad_B_dot_grad_psi = (
        d_psi_d_s * (bsubu * d_B_d_phi - bsubv * d_B_d_theta) / sqrtg_safe
    )

    dtheta = theta1d[1] - theta1d[0]
    dphi = phi1d[1] - phi1d[0]
    sqrtg_abs = jnp.abs(sqrtg)
    # The QS residual scales like sqrt(|sqrt(g)|). At the magnetic axis this can
    # hit exactly zero, which makes reverse-mode sensitivities undefined even
    # though the residual value itself is finite. Keep a tiny positive floor so
    # the traced objective remains differentiable at the axis surface.
    sqrtg_abs_safe = jnp.maximum(
        sqrtg_abs,
        jnp.asarray(jnp.finfo(sqrtg.dtype).tiny, dtype=sqrtg.dtype),
    )
    modB_safe = jnp.maximum(jnp.abs(modB), jnp.asarray(jnp.finfo(modB.dtype).tiny, dtype=modB.dtype))
    V_prime = nfp * dtheta * dphi * jnp.sum(sqrtg_abs_safe, axis=(1, 2))

    nn = helicity_n * nfp
    prefactor = jnp.sqrt(
        weights[:, None, None]
        * nfp
        * dtheta
        * dphi
        / V_prime[:, None, None]
        * sqrtg_abs_safe
    )
    residuals3d = prefactor * (
        B_cross_grad_B_dot_grad_psi * (nn - iota[:, None, None] * helicity_m)
        - B_dot_grad_B * (helicity_m * G[:, None, None] + nn * I[:, None, None])
    ) / (modB_safe**3)

    residuals1d = jnp.ravel(residuals3d)
    profile = jnp.sum(residuals3d * residuals3d, axis=(1, 2))
    total = jnp.sum(residuals1d * residuals1d)

    return {
        "surfaces": surfaces,
        "weights": weights,
        "residuals3d": residuals3d,
        "residuals1d": residuals1d,
        "profile": profile,
        "total": total,
        "modB": modB,
        "sqrtg": sqrtg,
        "B_dot_grad_B": B_dot_grad_B,
        "B_cross_grad_B_dot_grad_psi": B_cross_grad_B_dot_grad_psi,
        "iota": iota,
        "G": G,
        "I": I,
    }


def quasisymmetry_ratio_residual_from_state_jax(
    vmec,
    state,
    *,
    surfaces,
    helicity_m: int = 1,
    helicity_n: int = 0,
    weights=None,
    ntheta: int = 63,
    nphi: int = 64,
):
    """Evaluate the VMEC-only QS residual directly from a solved ``VmecJax`` state.

    This path avoids constructing an intermediate Python ``wout`` object, so it
    remains usable inside traced JAX optimization loops.
    """
    data = vmec.quasisymmetry_diagnostics_from_state_jax(state)
    return quasisymmetry_ratio_residual_from_wout_jax(
        data,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        weights=weights,
        ntheta=ntheta,
        nphi=nphi,
    )


class QuasisymmetryRatioResidualJax:
    """JAX-backed VMEC-only quasisymmetry residual for :class:`VmecJax`."""

    def __init__(
        self,
        vmec,
        surfaces,
        helicity_m: int = 1,
        helicity_n: int = 0,
        weights=None,
        ntheta: int = 63,
        nphi: int = 64,
    ) -> None:
        """Store the VMEC-JAX source and target surfaces for QS evaluation."""
        self.vmec = vmec
        self.surfaces = list(_as_surface_array(surfaces))
        self.helicity_m = int(helicity_m)
        self.helicity_n = int(helicity_n)
        self.weights = None if weights is None else list(weights)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)

    def compute_from_wout(self, wout):
        """Evaluate all intermediate VMEC-only QS channels from a ``wout`` object."""
        return quasisymmetry_ratio_residual_from_wout_jax(
            wout,
            surfaces=self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
            weights=self.weights,
            ntheta=self.ntheta,
            nphi=self.nphi,
        )

    def compute_from_state(self, state):
        """Evaluate the QS metric from an already solved VMEC-JAX state."""
        return quasisymmetry_ratio_residual_from_state_jax(
            self.vmec,
            state,
            surfaces=self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
            weights=self.weights,
            ntheta=self.ntheta,
            nphi=self.nphi,
        )

    def compute(self, x_free):
        """Evaluate the QS metric for a boundary parameter vector."""
        return self.compute_from_wout(self.vmec.get_wout(x_free))

    def residuals_from_wout(self, wout):
        """Return the flattened least-squares residual vector."""
        return self.compute_from_wout(wout)["residuals1d"]

    def residuals_from_state(self, state):
        """Return the flattened least-squares residual vector from a solved state."""
        return self.compute_from_state(state)["residuals1d"]

    def residuals(self, x_free):
        """Return the flattened least-squares residual vector for ``x_free``."""
        return self.compute(x_free)["residuals1d"]

    def profile_from_wout(self, wout):
        """Return the radial profile of squared QS residuals."""
        return self.compute_from_wout(wout)["profile"]

    def profile_from_state(self, state):
        """Return the radial profile of squared QS residuals from a solved state."""
        return self.compute_from_state(state)["profile"]

    def profile(self, x_free):
        """Return the radial profile of squared QS residuals for ``x_free``."""
        return self.compute(x_free)["profile"]

    def total_from_wout(self, wout):
        """Return the scalar QS objective from a ``wout`` object."""
        return self.compute_from_wout(wout)["total"]

    def total_from_state(self, state):
        """Return the scalar QS objective from a solved VMEC-JAX state."""
        return self.compute_from_state(state)["total"]

    def total(self, x_free):
        """Return the scalar QS objective for ``x_free``."""
        return self.compute(x_free)["total"]
