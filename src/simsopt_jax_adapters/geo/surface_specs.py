"""Immutable JAX specs of native simsopt surfaces, the inputs of
:mod:`simsopt_jax.core.surface_geometry`::

    spec = surface_spec_from_surface(surface)  # a snapshot: rebuild it after the surface changes
    gamma = surface_gamma(spec)  # surface.gamma() as a device array
"""

from __future__ import annotations

import numpy as np

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

__all__ = ["surface_spec_from_surface"]

# Native class -> its spec class and the native arrays the spec copies.
_SPECS = {
    SurfaceRZFourier: (SurfaceRZFourierSpec, ("rc", "rs", "zc", "zs")),
    SurfaceXYZFourier: (SurfaceXYZFourierSpec, ("xc", "xs", "yc", "ys", "zc", "zs")),
    SurfaceXYZTensorFourier: (SurfaceXYZTensorFourierSpec, ("xcs", "ycs", "zcs")),
}


def surface_spec_from_surface(
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
) -> SurfaceSpec:
    """The spec of ``surface``'s current coefficients and quadrature grid.

    The native coefficient arrays (every entry, as the native geometry reads
    them) and quadrature points are copied and placed explicitly on the active
    JAX device. Only ``SurfaceRZFourier``, ``SurfaceXYZFourier`` and
    ``SurfaceXYZTensorFourier`` themselves are supported: other surface
    classes, and subclasses that may override their geometry, raise
    ``TypeError``.
    """
    if type(surface) not in _SPECS:
        raise TypeError(
            "surface_spec_from_surface supports SurfaceRZFourier, SurfaceXYZFourier "
            f"and SurfaceXYZTensorFourier, got {type(surface).__name__}."
        )
    spec_class, coefficient_names = _SPECS[type(surface)]
    # Copy first: native set_dofs rewrites the coefficient arrays in place, and a
    # CPU device_put may alias the NumPy memory it is given.
    arrays = {
        name: as_jax_float64(np.array(getattr(surface, name), dtype=np.float64))
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
