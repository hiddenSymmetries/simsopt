"""
Pure JAX replacement for ``simsoptpp.integral_BdotN``.

Computes quadratic-flux-like surface integrals used in Stage-2 coil
optimization, with the formulas of the C++ kernel.  With ``n̂ = n / |n|``
and ``N = nphi·ntheta``:

* ``"quadratic flux"``: ``J = 0.5 / N · Σ (B·n̂ − B_T)² |n|``
* ``"normalized"``:     ``J = 0.5 · Σ (B·n̂ − B_T)² |n|  /  Σ |B|² |n|``
* ``"local"``:          ``J = 0.5 / N · Σ (B·n̂ − B_T)² / |B|² · |n|``

Singular inputs (a zero normal, or a zero field for ``"normalized"`` and
``"local"``) give the same IEEE nan/inf as the C++ kernel. Sums may be
associated in a different order, so results agree to round-off. An empty
target array means no target field, as in the C++ kernel.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .specs import FixedSurfaceFluxSpec

__all__ = [
    "FLUX_DEFINITIONS",
    "fixed_surface_flux_integral_from_B",
    "integral_BdotN",
]

FLUX_DEFINITIONS = ("quadratic flux", "normalized", "local")


def _validate_shapes(Bcoil, target, normal):
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


@partial(jax.jit, static_argnames=("definition",))
def integral_BdotN(Bcoil, target, normal, definition="quadratic flux"):
    """Compute the integral B·n objective.

    Args:
        Bcoil:  (nphi, ntheta, 3) coil magnetic field on the surface.
        target: (nphi, ntheta)    target normal field, or an empty array for none.
        normal: (nphi, ntheta, 3) unnormalized surface normal.
        definition: one of ``"quadratic flux"``, ``"normalized"``,
                    ``"local"``; a compile-time constant.

    Returns:
        J: scalar objective value.
    """
    _validate_shapes(Bcoil, target, normal)
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
    """Evaluate ``integral_BdotN`` for ``B`` at ``flux_spec.points`` (shape ``(n, 3)``)."""
    return integral_BdotN(
        B.reshape((flux_spec.nphi, flux_spec.ntheta, 3)),
        flux_spec.target,
        flux_spec.normal,
        flux_spec.definition,
    )
