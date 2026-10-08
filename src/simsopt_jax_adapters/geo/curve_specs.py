"""Adapter-owned conversion from native curve objects to immutable JAX specs."""

from __future__ import annotations

import numpy as np
from typing import cast
from simsopt_jax.core.specs import CurveSpec

from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveperturbed import CurvePerturbed
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import FrameRotation, FramedCurveFrenet, ZeroRotation
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
    "supports_adapter_curve_spec",
]


def supports_adapter_curve_spec(curve: object) -> bool:
    return isinstance(
        curve,
        (
            CurveXYZFourier,
            CurveXYZFourierSymmetries,
            CurveHelical,
            CurvePlanarFourier,
            CurveRZFourier,
            CurvePerturbed,
            CurveFilament,
        ),
    ) or callable(getattr(curve, "to_spec", None))


def curve_spec_from_adapter_curve(curve, *, separate_owners: bool = False) -> CurveSpec:
    """Capture geometry with shared DOFs or independent actual-owner VJP slots."""
    if isinstance(curve, CurveXYZFourierSymmetries):
        return make_curve_xyzfouriersymmetries_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
            nfp=curve.nfp,
            stellsym=curve.stellsym,
            ntor=curve.ntor,
        )
    if isinstance(curve, CurveXYZFourier):
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
    if isinstance(curve, CurvePlanarFourier):
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

    to_spec = getattr(curve, "to_spec", None)
    if callable(to_spec):
        return cast(CurveSpec, to_spec())

    return _pure_curve_spec_from_curve(curve)


def _curve_perturbed_spec_from_curve(curve: CurvePerturbed, *, separate_owners: bool):
    sample_gamma = curve.sample[0]
    sample_gammadash = curve.sample[1]
    sample_gammadashdash = (
        curve.sample[2]
        if len(curve.sample._sample) > 2
        else np.zeros_like(sample_gamma)
    )
    sample_gammadashdashdash = (
        curve.sample[3]
        if len(curve.sample._sample) > 3
        else np.zeros_like(sample_gamma)
    )

    return make_curve_perturbed_spec(
        dofs=_optimizable_dof_layout(curve, separate_owners=separate_owners)[0],
        quadpoints=curve.quadpoints,
        base_curve=curve_spec_from_adapter_curve(curve.curve, separate_owners=separate_owners),
        base_curve_map=_optimizable_dof_map_spec(curve, curve.curve, separate_owners=separate_owners),
        sample_gamma=sample_gamma,
        sample_gammadash=sample_gammadash,
        sample_gammadashdash=sample_gammadashdash,
        sample_gammadashdashdash=sample_gammadashdashdash,
    )


def _curve_filament_spec_from_curve(curve: CurveFilament, *, separate_owners: bool):
    return make_curve_filament_spec(
        dofs=_optimizable_dof_layout(curve, separate_owners=separate_owners)[0],
        quadpoints=curve.quadpoints,
        base_curve=curve_spec_from_adapter_curve(curve.curve, separate_owners=separate_owners),
        base_curve_map=_optimizable_dof_map_spec(curve, curve.curve, separate_owners=separate_owners),
        rotation=_rotation_spec_from_curve(curve.rotation, curve.curve.quadpoints),
        rotation_map=_optimizable_dof_map_spec(curve, curve.rotation, separate_owners=separate_owners),
        frame_kind="frenet"
        if isinstance(curve.framedcurve, FramedCurveFrenet)
        else "centroid",
        dn=curve.dn,
        db=curve.db,
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
