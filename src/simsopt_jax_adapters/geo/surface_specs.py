"""Immutable JAX snapshots of native simsopt surfaces for geometry evaluation::

    spec = surface_spec_from_surface(surface)  # rebuild after surface changes
    gamma = surface_gamma(spec)  # Cartesian positions on the device
"""

from __future__ import annotations

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt_jax.backend.dtypes import as_jax_float64
from simsopt_jax.core.specs import (
    SurfaceRZFourierSpec,
    SurfaceSpec,
    SurfaceXYZFourierSpec,
    SurfaceXYZTensorFourierSpec,
)

# Native class -> its spec class and the native arrays the spec copies.
_SPECS = {
    SurfaceRZFourier: (SurfaceRZFourierSpec, ("rc", "rs", "zc", "zs")),
    SurfaceXYZFourier: (SurfaceXYZFourierSpec, ("xc", "xs", "yc", "ys", "zc", "zs")),
    SurfaceXYZTensorFourier: (SurfaceXYZTensorFourierSpec, ("xcs", "ycs", "zcs")),
}


def surface_spec_from_surface(
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
) -> SurfaceSpec:
    """Copy a native surface's coefficients and quadrature grid to the active device.

    Args:
        surface (SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier): Native
            surface of exactly one of these classes; subclasses are unsupported.

    Returns:
        SurfaceSpec: Immutable snapshot of every coefficient entry in meters and
            quadrature points in turns. Rebuild after native state changes.

    Raises:
        TypeError: If the surface is unsupported or a subclass.
    """
    if type(surface) not in _SPECS:
        raise TypeError(
            "surface_spec_from_surface supports SurfaceRZFourier, SurfaceXYZFourier "
            f"and SurfaceXYZTensorFourier, got {type(surface).__name__}."
        )
    spec_class, coefficient_names = _SPECS[type(surface)]
    # Placement snapshots native arrays before set_dofs can rewrite them.
    arrays = {
        name: as_jax_float64(getattr(surface, name))
        for name in (*coefficient_names, "quadpoints_phi", "quadpoints_theta")
    }
    clamped_dims = (
        {"clamped_dims": tuple(bool(flag) for flag in surface.clamped_dims)}
        if type(surface) is SurfaceXYZTensorFourier
        else {}
    )
    return spec_class(
        **arrays, nfp=int(surface.nfp), stellsym=bool(surface.stellsym), **clamped_dims
    )
