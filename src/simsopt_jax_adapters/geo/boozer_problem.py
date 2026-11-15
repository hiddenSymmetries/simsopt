"""Build immutable JAX Boozer problems from native simsopt objects.

Snapshots hold surface, coil, and label state for JAX Boozer formulations.
"""

from __future__ import annotations

import jax
import numpy as np

from simsopt.geo.surfaceobjectives import Area, AspectRatio, ToroidalFlux, Volume
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.boozer_problem import (
    BoozerLabelSpec,
    BoozerProblem,
    LabelKind,
    make_boozer_problem,
)
from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

from .surface_specs import surface_spec_from_surface

_LABEL_KINDS: dict[type, LabelKind] = {
    Volume: "volume",
    Area: "area",
    AspectRatio: "aspect_ratio",
    ToroidalFlux: "toroidal_flux",
}


def boozer_problem(
    biotsavart: JaxBiotSavart,
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
    label: Volume | Area | AspectRatio | ToroidalFlux,
    targetlabel: float,
    constraint_weight: float | None = None,
) -> BoozerProblem:
    """Snapshot native surface, coil, and label state as a JAX Boozer problem.

    Args:
        biotsavart (JaxBiotSavart): Field using the same coils as a ToroidalFlux label.
        surface (SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier): Native
            surface to snapshot.
        label (Volume | Area | AspectRatio | ToroidalFlux): Label on this surface or one
            of the same class sharing its DOFs.
        targetlabel (float): Target label in cubic meters (volume), square meters
            (area), webers (toroidal flux), or dimensionless (aspect ratio).
        constraint_weight (float | None): Native penalty coefficient for the squared
            label and z constraints; None for an exact problem.

    Returns:
        BoozerProblem: Immutable device snapshot; rebuild when coils, fixed values,
            label grid, target, or weight change.
    """
    if not isinstance(biotsavart, JaxBiotSavart):
        raise TypeError(f"boozer_problem needs a JaxBiotSavart field, got {type(biotsavart).__name__}.")
    kind = _LABEL_KINDS.get(type(label))
    if kind is None:
        raise TypeError(
            "boozer_problem supports Volume, Area, AspectRatio and ToroidalFlux labels, "
            f"got {type(label).__name__}."
        )
    label_surface = label.surface
    if label_surface is not surface and (
        type(label_surface) is not type(surface) or label_surface.dofs is not surface.dofs
    ):
        raise ValueError("the label must evaluate the Boozer surface or a surface sharing its DOFs.")
    if kind == "toroidal_flux":
        label_coils, coils = list(label.biotsavart.coils), list(biotsavart.coils)
        if len(label_coils) != len(coils) or any(a is not b for a, b in zip(label_coils, coils)):
            raise ValueError("the ToroidalFlux label must use the field's coils.")
    surface_spec = surface_spec_from_surface(surface)
    label_surface_spec = surface_spec if label_surface is surface else surface_spec_from_surface(label_surface)
    return make_boozer_problem(
        surface=surface_spec,
        coils=biotsavart.coil_set_spec(),
        label=BoozerLabelSpec(
            surface=label_surface_spec,
            kind=kind,
            idx=label.idx if kind == "toroidal_flux" else 0,
        ),
        targetlabel=targetlabel,
        constraint_weight=constraint_weight,
    )


def boozer_exact_residual_mask(surface: SurfaceXYZTensorFourier) -> np.ndarray:
    """Return the flattened boolean mask selecting BoozerExact residual equations.

    Other surface classes and unsupported stellarator-symmetric grids raise.

    Args:
        surface (SurfaceXYZTensorFourier): Native surface with a supported exact-solve
            quadrature grid.

    Returns:
        numpy.ndarray: Boolean shape (3 * nphi * ntheta,) native exact residual mask,
            excluding the x equation at (0, 0) under stellarator symmetry.
    """
    if not isinstance(surface, SurfaceXYZTensorFourier):
        raise RuntimeError(
            "Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier"
        )
    mask = np.repeat(surface.get_stellsym_mask()[..., None], 3, axis=2)
    if surface.stellsym:
        mask[0, 0, 0] = False
    return mask.flatten()


def boozer_exact_residual_rows(
    surface: SurfaceXYZTensorFourier, reference: jax.Array | None = None
) -> jax.Array:
    """Return int32 indices of the BoozerExact residual mask on the selected device.

    Other surface classes and unsupported stellarator-symmetric grids raise.

    Args:
        surface (SurfaceXYZTensorFourier): Native surface with a supported exact-solve
            quadrature grid.
        reference (jax.Array | None): Optional device-placement reference, with
            arbitrary shape; None uses the active device.

    Returns:
        jax.Array: Shape (nrows,) int32 flattened residual indices in native mask order,
            placed like reference when supplied.
    """
    rows = np.flatnonzero(boozer_exact_residual_mask(surface))
    return explicit_device_array(rows, dtype=np.int32, reference=reference)
