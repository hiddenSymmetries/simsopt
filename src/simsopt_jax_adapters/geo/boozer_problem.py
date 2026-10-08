"""The JAX Boozer problem of native simsopt objects.

:func:`boozer_problem` takes what native ``BoozerSurface`` takes and returns
the :class:`~simsopt_jax.core.boozer_problem.BoozerProblem` that the JAX
formulations of :mod:`simsopt_jax.core.boozer_problem` evaluate::

    import jax
    import numpy as np
    from simsopt.configs import get_data
    from simsopt.geo import SurfaceXYZTensorFourier, Volume
    from simsopt_jax.core.boozer_problem import boozer_penalty_constraints
    from simsopt_jax_adapters.field import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_problem import boozer_problem

    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    field = BiotSavartJAX(bs.coils)
    surface = SurfaceXYZTensorFourier(
        mpol=3, ntor=3, stellsym=True, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, 12, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 12, endpoint=False),
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)
    volume = Volume(surface)
    problem = boozer_problem(field, surface, volume, volume.J(), constraint_weight=100.0)

    # As BoozerSurface.boozer_penalty_constraints_vectorized(x, derivatives=2,
    # constraint_weight=100.0) at x = [surface DOFs, iota], G from the currents:
    x = jax.device_put(np.concatenate((surface.get_dofs(), [-0.4])))
    value, gradient, hessian = boozer_penalty_constraints(problem, x, derivatives=2)

The problem is a snapshot of the coils, the surface's coefficients outside
its DOFs, the label's grid, the target and the weight: build a new one after
any of them changes; the formulations reuse their compiled programs for it.
This module imports the field adapters, which import this package, so it is
not re-exported from :mod:`simsopt_jax_adapters.geo`.
"""

from __future__ import annotations

import jax
import numpy as np

from simsopt.geo.surfaceobjectives import Area, AspectRatio, ToroidalFlux, Volume
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt_jax.core._math_utils import as_jax_int32
from simsopt_jax.core.boozer_problem import (
    BoozerLabelSpec,
    BoozerProblem,
    LabelKind,
    make_boozer_problem,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

from .surface_specs import surface_spec_from_surface

__all__ = ["boozer_exact_residual_rows", "boozer_problem"]

_LABEL_KINDS: dict[type, LabelKind] = {
    Volume: "volume",
    Area: "area",
    AspectRatio: "aspect_ratio",
    ToroidalFlux: "toroidal_flux",
}


def boozer_problem(
    field: BiotSavartJAX,
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
    label: Volume | Area | AspectRatio | ToroidalFlux,
    targetlabel: float,
    constraint_weight: float | None = None,
) -> BoozerProblem:
    """The Boozer problem of ``BoozerSurface(field, surface, label, targetlabel,
    constraint_weight)``; ``constraint_weight`` is the penalty's weight.

    ``label`` must evaluate ``surface`` itself or a surface of the same class
    that shares its DOFs (native labels built with ``nphi``, ``ntheta`` or
    ``range``); a ``ToroidalFlux`` label must use ``field``'s coils. Unlike
    native ``BoozerSurface``, ``surface`` may also be a ``SurfaceRZFourier``.
    """
    if not isinstance(field, BiotSavartJAX):
        raise TypeError(f"boozer_problem needs a BiotSavartJAX field, got {type(field).__name__}.")
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
        label_coils, coils = list(label.biotsavart.coils), list(field.coils)
        if len(label_coils) != len(coils) or any(a is not b for a, b in zip(label_coils, coils)):
            raise ValueError("the ToroidalFlux label must use the field's coils.")
    return make_boozer_problem(
        surface=surface_spec_from_surface(surface),
        coils=field.coil_set_spec(),
        label=BoozerLabelSpec(
            surface=surface_spec_from_surface(label_surface),
            kind=kind,
            phi_index=label.idx if kind == "toroidal_flux" else 0,
        ),
        target_label=targetlabel,
        constraint_weight=constraint_weight,
    )


def boozer_exact_residual_rows(surface: SurfaceXYZTensorFourier) -> jax.Array:
    """The residual rows of native ``solve_residual_equation_exactly_newton``,
    for :func:`~simsopt_jax.core.boozer_problem.boozer_exact_residual`: native
    ``get_stellsym_mask()`` per component, without the x residual at ``(0, 0)``
    under stellarator symmetry. Raises where native does: for other surface
    classes, and for stellarator-symmetric grids ``get_stellsym_mask()`` rejects.
    """
    if not isinstance(surface, SurfaceXYZTensorFourier):
        raise RuntimeError(
            "Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier"
        )
    mask = np.repeat(surface.get_stellsym_mask()[..., None], 3, axis=2)
    if surface.stellsym:
        mask[0, 0, 0] = False
    return as_jax_int32(np.flatnonzero(mask))
