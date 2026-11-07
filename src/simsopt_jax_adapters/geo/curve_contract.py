"""Shared JAX curve-method contract helpers."""

import jax.numpy as jnp
import numpy as np
from typing import cast

from simsopt._core.optimizable import Optimizable
from simsopt._core.types import RealArray
from simsopt.geo.curveperturbed import CurvePerturbed
from simsopt.geo.finitebuild import CurveFilament
from simsopt_jax.core.specs import make_optimizable_dof_map_spec


def adapter_curve_dof_mode(curve: object) -> str:
    """Choose the DOF vector consumed by an adapter curve's immutable spec.

    Args:
        curve (object): Native or custom curve supplying an immutable geometry contract.

    Returns:
        str: full for perturbed/filament wrappers; otherwise the curve
            override or local by default.
    """
    if isinstance(curve, (CurvePerturbed, CurveFilament)):
        return "full"
    return getattr(curve, "_jax_curve_dof_mode", "local")


def _optimizable_dof_layout(
    opt: Optimizable, *, separate_owners: bool = False,
) -> tuple[RealArray, dict[Optimizable, tuple[int, int]]]:
    """Return full DOFs and slices, retaining actual owners for partial VJPs."""
    if not separate_owners:
        return opt.full_x, opt._full_dof_indices
    indices: dict[Optimizable, tuple[int, int]] = {}
    width = 0
    owners = opt.ancestors + [opt]
    for owner in owners:
        owner_width = int(owner.local_full_dof_size)
        indices[owner] = (width, width + owner_width)
        width += owner_width
    return np.concatenate([cast(np.ndarray, owner.local_full_x) for owner in owners]), indices


def _optimizable_dof_map_components(owner, opt, *, separate_owners: bool = False):
    full_dofs, opt_indices = _optimizable_dof_layout(opt, separate_owners=separate_owners)
    _, owner_indices = _optimizable_dof_layout(owner, separate_owners=separate_owners)
    # Native layouts deduplicate shared DOFs; derivative layouts keep owners apart.
    if separate_owners:
        source_indices = owner_indices
    else:
        slices_by_dofs = {
            dep_opt.dofs: indices for dep_opt, indices in owner_indices.items()
        }
        source_indices = {
            dep_opt: slices_by_dofs[dep_opt.dofs] for dep_opt in opt_indices
        }
    template_full_dofs = jnp.asarray(full_dofs, dtype=jnp.float64)
    owner_segments = tuple(
        (
            int(source_indices[dep_opt][0]),
            int(source_indices[dep_opt][1]),
            int(sub_start),
            int(sub_end),
        )
        for dep_opt, (sub_start, sub_end) in opt_indices.items()
    )
    if adapter_curve_dof_mode(opt) == "full":
        input_mode = "full"
        input_start = 0
        input_end = int(template_full_dofs.shape[0])
    else:
        input_mode = "local"
        input_start, input_end = opt_indices[opt]
    return (
        template_full_dofs,
        owner_segments,
        input_mode,
        int(input_start),
        int(input_end),
    )


def _optimizable_dof_map_spec(owner, opt, *, separate_owners: bool = False):
    (
        template_full_dofs,
        owner_segments,
        input_mode,
        input_start,
        input_end,
    ) = _optimizable_dof_map_components(owner, opt, separate_owners=separate_owners)
    return make_optimizable_dof_map_spec(
        template_full_dofs=template_full_dofs,
        owner_segments=owner_segments,
        input_mode=input_mode,
        input_start=input_start,
        input_end=input_end,
    )
