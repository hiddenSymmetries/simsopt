"""Adapter-owned conversion from native curve objects to immutable JAX specs."""

from __future__ import annotations

import numpy as np
from simsopt_jax.core.specs import CurveSpec

from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveperturbed import CurvePerturbed
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import (
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
    ZeroRotation,
)
from simsopt_jax.core import (
    curve_spec_from_curve as _pure_curve_spec_from_curve,
    make_curve_filament_spec,
    make_curve_helical_spec,
    make_curve_perturbed_spec,
    make_curve_planarfourier_spec,
    make_curve_rzfourier_spec,
    make_curve_xyzfourier_spec,
    make_curve_xyzfouriersymmetries_spec,
    make_frame_rotation_spec,
    make_zero_rotation_spec,
)
from simsopt_jax_adapters.geo.curve_contract import (
    _optimizable_dof_layout,
    _optimizable_dof_map_spec,
    adapter_curve_dof_mode,
)

__all__ = [
    "adapter_curve_dof_mode",
    "curve_spec_from_adapter_curve",
]


def curve_spec_from_adapter_curve(curve, *, separate_owners: bool = False) -> CurveSpec:
    """Capture geometry with shared DOFs or independent actual-owner VJP slots.

    Supported curves: CurveXYZFourier, JaxCurveXYZFourier, CurveXYZFourierSymmetries,
    CurveHelical, CurvePlanarFourier, JaxCurvePlanarFourier, CurveRZFourier,
    CurvePerturbed and CurveFilament (centroid or Frenet frame, FrameRotation or
    ZeroRotation) wrapping any supported curve, and objects exposing to_spec().
    RotatedCurve is supported only outermost, as a coil placement carried by
    CoilSymmetrySpec; a RotatedCurve inside CurvePerturbed or CurveFilament raises
    NotImplementedError.

    Args:
        curve (object): Supported native curve or custom object exposing to_spec().
        separate_owners (bool): Give wrapped geometry independent actual-owner slots for
            VJP projection instead of shared-DOF slots.

    Returns:
        CurveSpec object: Immutable geometry payload with float64
            sampled/coefficient arrays.
    """
    if isinstance(curve, CurveXYZFourierSymmetries):
        return make_curve_xyzfouriersymmetries_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
            nfp=curve.nfp,
            stellsym=curve.stellsym,
            ntor=curve.ntor,
        )
    if isinstance(curve, (CurveXYZFourier, JaxCurveXYZFourier)):
        return make_curve_xyzfourier_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
        )
    if isinstance(curve, CurveHelical):
        return make_curve_helical_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
            m=curve.m,
            ell=curve.ell,
            R0=curve.R0,
            r=curve.r,
        )
    if isinstance(curve, (CurvePlanarFourier, JaxCurvePlanarFourier)):
        return make_curve_planarfourier_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
        )
    if isinstance(curve, CurveRZFourier):
        return make_curve_rzfourier_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
            nfp=curve.nfp,
            stellsym=curve.stellsym,
        )
    if isinstance(curve, CurvePerturbed):
        return _curve_perturbed_spec_from_curve(curve, separate_owners=separate_owners)
    if isinstance(curve, CurveFilament):
        return _curve_filament_spec_from_curve(curve, separate_owners=separate_owners)

    return _pure_curve_spec_from_curve(curve)


def _curve_perturbed_spec_from_curve(curve: CurvePerturbed, *, separate_owners: bool):
    # The spec carries the sample and up to three derivatives; missing ones are
    # zero placeholders that the spec's sample_derivatives marks as absent.
    sample_derivatives = min(len(curve.sample._sample) - 1, 3)
    zero = np.zeros_like(curve.sample[0])
    sample = [curve.sample[k] if k <= max(sample_derivatives, 1) else zero for k in range(4)]
    return make_curve_perturbed_spec(
        dofs=_optimizable_dof_layout(curve, separate_owners=separate_owners)[0],
        quadpoints=curve.quadpoints,
        base_curve=curve_spec_from_adapter_curve(curve.curve, separate_owners=separate_owners),
        base_curve_map=_optimizable_dof_map_spec(curve, curve.curve, separate_owners=separate_owners),
        sample_gamma=sample[0],
        sample_gammadash=sample[1],
        sample_gammadashdash=sample[2],
        sample_gammadashdashdash=sample[3],
        sample_derivatives=sample_derivatives,
    )


def _curve_filament_spec_from_curve(curve: CurveFilament, *, separate_owners: bool):
    return make_curve_filament_spec(
        dofs=_optimizable_dof_layout(curve, separate_owners=separate_owners)[0],
        quadpoints=curve.quadpoints,
        base_curve=curve_spec_from_adapter_curve(curve.curve, separate_owners=separate_owners),
        base_curve_map=_optimizable_dof_map_spec(curve, curve.curve, separate_owners=separate_owners),
        rotation=_rotation_spec_from_curve(curve.rotation, curve.curve.quadpoints),
        rotation_map=_optimizable_dof_map_spec(curve, curve.rotation, separate_owners=separate_owners),
        frame_kind=_frame_kind(curve.framedcurve),
        dn=curve.dn,
        db=curve.db,
    )


def _frame_kind(framedcurve) -> str:
    if isinstance(framedcurve, FramedCurveFrenet):
        return "frenet"
    if isinstance(framedcurve, FramedCurveCentroid):
        return "centroid"
    raise NotImplementedError(
        "CurveFilament JAX spec conversion supports FramedCurveCentroid and "
        f"FramedCurveFrenet frames, got {type(framedcurve).__name__}."
    )


def _rotation_spec_from_curve(rotation, quadpoints):
    if isinstance(rotation, ZeroRotation):
        return make_zero_rotation_spec(quadpoints=quadpoints)
    if isinstance(rotation, FrameRotation):
        return make_frame_rotation_spec(
            dofs=rotation.full_x,
            quadpoints=quadpoints,
            order=rotation.order,
            scale=rotation.scale,
        )
    raise NotImplementedError(
        "CurveFilament JAX spec conversion supports FrameRotation and "
        f"ZeroRotation, got {type(rotation).__name__}."
    )
