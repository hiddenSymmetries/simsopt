"""Pure JAX quadratic, normalized and local surface-flux objectives.

These kernels replace ``simsoptpp.integral_BdotN`` for Stage-II coil
optimization; results agree with the native kernel to round-off.
"""

from functools import partial

import jax.numpy as jnp

from simsopt_jax.backend.jit import jit

from ._math_utils import as_jax_float64
from .specs import FixedSurfaceFluxSpec

FLUX_DEFINITIONS = ("quadratic flux", "normalized", "local")


def integral_BdotN(Bcoil, target, normal, definition="quadratic flux"):
    """Compute the quadratic, field-normalized or locally normalized flux integral.

    Quadratic flux averages half the squared normal-field error with surface
    area weights. Normalized flux divides by integrated field strength squared;
    local flux divides pointwise by field strength squared.
    Inputs use float64 with JAX's 64-bit mode enabled; NumPy inputs are copied.
    Zero normals, or zero fields for normalized and local flux, can produce NaN/inf.

    Args:
        Bcoil: Array of shape (nphi, ntheta, 3), magnetic field in T.
        target: Array of shape (nphi, ntheta), target normal field in T, or empty for zero target field.
        normal: Array of shape (nphi, ntheta, 3), unnormalized normals in m^2.
        definition: str, compile-time choice: "quadratic flux", "normalized", or "local".

    Returns:
        Array: scalar objective in T^2 m^2 (quadratic flux), dimensionless (normalized), or m^2 (local).
    """
    return _integral_BdotN(*map(as_jax_float64, (Bcoil, target, normal)), definition)


@partial(jit, static_argnames=("definition",))
def _integral_BdotN(Bcoil, target, normal, definition):
    """Evaluate a flux integral on float64 operands."""
    if len(Bcoil.shape) != 3 or Bcoil.shape[2] != 3:
        raise ValueError(
            f"Bcoil must have shape (nphi, ntheta, 3); got Bcoil.shape={Bcoil.shape}."
        )
    if normal.shape != Bcoil.shape:
        raise ValueError(
            "normal.shape must match Bcoil.shape; "
            f"got normal.shape={normal.shape}, Bcoil.shape={Bcoil.shape}."
        )
    if target.size and target.shape != Bcoil.shape[:2]:
        raise ValueError(
            "target.shape must match Bcoil.shape[:2]; "
            f"got target.shape={target.shape}, Bcoil.shape[:2]={Bcoil.shape[:2]}."
        )
    if definition not in FLUX_DEFINITIONS:
        raise ValueError(f"Unknown definition: {definition!r}")
    norm_n = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
    BdotN = jnp.sum(Bcoil * (normal / norm_n[..., None]), axis=-1)
    if target.size:
        BdotN = BdotN - target
    square = BdotN * BdotN
    mod_B_squared = jnp.sum(Bcoil * Bcoil, axis=-1)
    if definition == "normalized":
        return 0.5 * jnp.sum(square * norm_n) / jnp.sum(mod_B_squared * norm_n)
    if definition == "local":
        square = square / mod_B_squared
    return 0.5 * jnp.sum(square * norm_n) / (Bcoil.shape[0] * Bcoil.shape[1])


def fixed_surface_flux_integral_from_B(B, flux_spec: FixedSurfaceFluxSpec):
    """Evaluate the flux integral on the grid captured by a fixed-surface spec.

    Args:
        B: Array of shape (nphi*ntheta, 3), magnetic field in T in spec point order.
        flux_spec: FixedSurfaceFluxSpec object, surface geometry, target and integral definition.

    Returns:
        Array: scalar objective with units determined by integral_BdotN.
    """
    return integral_BdotN(
        B.reshape((flux_spec.nphi, flux_spec.ntheta, 3)),
        flux_spec.target,
        flux_spec.normal,
        flux_spec.definition,
    )
