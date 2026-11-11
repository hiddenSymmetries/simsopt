"""Pure curve-geometry helpers that operate on immutable specs."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.dtypes import explicit_device_array


from .curve_helical import curve_helical_pure
from .curve_planar_fourier import curveplanarfourier_pure
from .curve_rz_fourier import curverzfourier_pure
from .curve_xyz_fourier import jaxfouriercurve_pure
from .curve_xyz_fourier_symmetries import jaxXYZFourierSymmetriescurve_pure
from ._math_utils import as_jax_float64 as _as_jax_float64
from .framedcurve import (
    rotated_centroid_frame,
    rotated_frenet_frame,
    rotation_alpha as jaxrotation_pure,
    rotation_alphadash as jaxrotationdash_pure,
)
from .specs import (
    CurveFilamentSpec,
    CurveHelicalSpec,
    CurvePlanarFourierSpec,
    CurvePerturbedSpec,
    CurveRZFourierSpec,
    CurveSpec,
    CurveXYZFourierSpec,
    CurveXYZFourierSymmetriesSpec,
    OptimizableDofMapSpec,
    RotationSpec,
    ZeroRotationSpec,
    _filament_base_derivative_order,
    _require_sample_derivatives,
)


def _slice_1d_static(array: jax.Array, start: int, end: int) -> jax.Array:
    """Slice axis 0 with static bounds, retaining dtype and placement."""
    return jax.lax.slice_in_dim(array, int(start), int(end), axis=0)


def _update_1d_static(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    """Replace a contiguous axis-0 slice starting at ``start``.

    The replacement is exact for finite entries.
    """
    start = int(start)
    width = int(values.shape[0])
    size = int(array.shape[0])
    if start == 0 and width == size:
        return values
    placement = np.zeros((size, width), dtype=float)
    placement[np.arange(start, start + width), np.arange(width)] = 1.0
    keep = 1.0 - np.sum(placement, axis=1)
    return (
        array * explicit_device_array(keep, dtype=array.dtype, reference=array)
        + explicit_device_array(placement, dtype=array.dtype, reference=array) @ values
    )


def curve_spec_from_curve(curve) -> CurveSpec:
    """Return an immutable JAX spec from a compatible curve.

    ``RotatedCurve`` requires a base spec plus ``CoilSymmetrySpec`` for
    coil placement; standalone rotated geometry uses the CPU wrapper.

    Args:
        curve (object): Curve exposing a compatible to_spec() method; RotatedCurve must
            instead use a base spec with coil symmetry.

    Returns:
        CurveSpec object: Immutable curve geometry.
    """
    to_spec = getattr(curve, "to_spec", None)
    if callable(to_spec):
        return cast(CurveSpec, to_spec())

    if type(curve).__name__ == "RotatedCurve":
        raise NotImplementedError(
            "RotatedCurve is not an immutable JAX CurveSpec. Use the base "
            "curve spec plus CoilSymmetrySpec for coil placement, or evaluate "
            "standalone RotatedCurve geometry through the CPU wrapper."
        )

    raise NotImplementedError(
        f"Curve type {type(curve).__name__} does not expose an immutable JAX spec."
    )


def _curve_gamma_kernel(
    spec: CurveSpec,
    dofs=None,
):
    curve_dofs = (
        spec.dofs
        if dofs is None
        else _as_jax_float64(dofs)
    )
    if isinstance(spec, CurveXYZFourierSpec):
        return lambda quadpoints: jaxfouriercurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if isinstance(spec, CurveRZFourierSpec):
        return lambda quadpoints: curverzfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
        )
    if isinstance(spec, CurvePlanarFourierSpec):
        return lambda quadpoints: curveplanarfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if isinstance(spec, CurveHelicalSpec):
        return lambda quadpoints: curve_helical_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.m,
            spec.ell,
            spec.R0,
            spec.r,
        )
    if isinstance(spec, CurveXYZFourierSymmetriesSpec):
        return lambda quadpoints: jaxXYZFourierSymmetriescurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
            spec.ntor,
        )
    raise TypeError(
        "curve_gamma_kernel only supports direct curve specs, "
        f"got {type(spec).__name__}."
    )


def _parameter_derivative_kernel(kernel, tangents):
    """Return the parameter-derivative function along ``tangents``."""
    return lambda qp: jax.jvp(kernel, (qp,), (tangents,))[1]


def _curve_geometry_terms_from_kernel(gamma_kernel, quadpoints, *, order) -> tuple[jax.Array, ...]:
    """Return curve positions and their first ``order`` parameter derivatives."""
    if order == 0:
        return (gamma_kernel(quadpoints),)
    tangents = jnp.broadcast_to(_as_jax_float64(1.0), quadpoints.shape)
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    terms = [gamma, gammadash]
    derivative_kernel = gamma_kernel
    for _ in range(order - 1):
        derivative_kernel = _parameter_derivative_kernel(derivative_kernel, tangents)
        terms.append(jax.jvp(derivative_kernel, (quadpoints,), (tangents,))[1])
    return tuple(terms)


def _mapped_full_dofs(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
):
    mapped = _as_jax_float64(map_spec.template_full_dofs)
    owner_dofs = _as_jax_float64(owner_dofs)
    # make_optimizable_dof_map_spec guarantees equal owner and target lengths.
    for owner_start, owner_end, target_start, _target_end in map_spec.owner_segments:
        mapped = _update_1d_static(
            mapped,
            target_start,
            _slice_1d_static(owner_dofs, owner_start, owner_end),
        )
    return mapped


def optimizable_input_dofs_from_map_spec(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
):
    """Reconstruct one optimizable input vector from its owner DOFs.

    Args:
        map_spec (OptimizableDofMapSpec): Template of shape (D_full,) and half-open
            owner-to-target copy ranges.
        owner_dofs (array-like): Flat owner DOF vector, shape (D,), matching the
            extraction map; units depend on the owning geometry or current.

    Returns:
        jax.Array: Mapped full vector of shape (D_full,) for full mode,
            otherwise the requested local slice of shape (input_end -
            input_start,).
    """
    mapped_full = _mapped_full_dofs(
        map_spec,
        owner_dofs,
    )
    if map_spec.input_mode == "full":
        return mapped_full
    return _slice_1d_static(mapped_full, map_spec.input_start, map_spec.input_end)


def _rotation_alpha_terms_from_dofs(
    rotation_spec: RotationSpec,
    rotation_map: OptimizableDofMapSpec,
    owner_dofs,
    order: int,
) -> tuple[jax.Array, ...]:
    """Return the frame rotation angle and its first ``order`` parameter derivatives."""
    quadpoints = _as_jax_float64(rotation_spec.quadpoints)
    if isinstance(rotation_spec, ZeroRotationSpec):
        return (jnp.broadcast_to(_as_jax_float64(0.0), quadpoints.shape),) * (order + 1)

    rotation_dofs = optimizable_input_dofs_from_map_spec(rotation_map, owner_dofs)
    rotation_scale = _as_jax_float64(rotation_spec.scale)

    def alphadash(qp):
        return jaxrotationdash_pure(rotation_dofs, qp, rotation_spec.order)

    dash_terms = _curve_geometry_terms_from_kernel(alphadash, quadpoints, order=order - 1)
    terms = (jaxrotation_pure(rotation_dofs, quadpoints, rotation_spec.order), *dash_terms)
    return tuple(rotation_scale * term for term in terms)


def _curve_geometry_terms_from_dofs(spec: CurveSpec, dofs, order: int) -> tuple[jax.Array, ...]:
    """Return positions and ``order`` derivatives for any curve spec.

    Raises ``ValueError`` if a perturbation sample lacks a required derivative.
    """
    if isinstance(spec, CurvePerturbedSpec):
        _require_sample_derivatives(spec, order)
        base_geometry = _curve_geometry_terms_from_dofs(
            spec.base_curve,
            optimizable_input_dofs_from_map_spec(spec.base_curve_map, dofs),
            order,
        )
        return _add_curve_perturbation(spec, *base_geometry)
    if isinstance(spec, CurveFilamentSpec):
        return _curve_filament_terms_from_dofs(spec, dofs, order)
    gamma_kernel = _curve_gamma_kernel(
        spec,
        dofs,
    )
    quadpoints = _as_jax_float64(spec.quadpoints)
    return _curve_geometry_terms_from_kernel(gamma_kernel, quadpoints, order=order)


def _add_curve_perturbation(spec: CurvePerturbedSpec, *geometry_terms) -> tuple[jax.Array, ...]:
    sample_terms = (
        spec.sample_gamma,
        spec.sample_gammadash,
        spec.sample_gammadashdash,
        spec.sample_gammadashdashdash,
    )
    return tuple(
        geometry_term + sample_term
        for geometry_term, sample_term in zip(geometry_terms, sample_terms)
    )


def _curve_filament_input_jets(spec: CurveFilamentSpec, dofs, order: int):
    """Return the rotated-frame function and input derivatives through ``order``."""
    base_order = _filament_base_derivative_order(spec.frame_kind, order)
    base_terms = _curve_geometry_terms_from_dofs(
        spec.base_curve,
        optimizable_input_dofs_from_map_spec(spec.base_curve_map, dofs),
        base_order,
    )
    alpha_terms = _rotation_alpha_terms_from_dofs(
        spec.rotation,
        spec.rotation_map,
        dofs,
        order,
    )
    frame = rotated_frenet_frame if spec.frame_kind == "frenet" else rotated_centroid_frame
    geometry_inputs = base_order - order + 1
    input_jets = tuple(
        (*base_terms[j : j + geometry_inputs], alpha_terms[j]) for j in range(order + 1)
    )
    return frame, input_jets


def _filament_terms(frame, input_jets, dn, db) -> tuple[jax.Array, ...]:
    """Return offset filament positions and derivatives through the input-jet order."""

    def filament(frame_inputs):
        _tangent, normal, binormal = frame(*frame_inputs)
        return (frame_inputs[0] + dn * normal + db * binormal,)

    def differentiated(terms):
        def extended(*jets):
            primal_terms, tangent_terms = jax.jvp(terms, jets[:-1], jets[1:])
            return (*primal_terms, tangent_terms[-1])

        return extended

    terms = filament
    for _ in input_jets[1:]:
        terms = differentiated(terms)
    return terms(*input_jets)


def curve_filament_pack_from_dofs(
    spec: CurveFilamentSpec,
    dofs,
    offsets: tuple[tuple[float, float], ...],
) -> tuple[jax.Array, ...]:
    """Return base-curve geometry and offset filaments sharing a curve and frame.

    Args:
        spec: CurveFilamentSpec object defining base curve, DOF maps, rotation and frame.
        dofs: Array of shape (ndofs,), full filament DOFs in spec mapping order.
        offsets: tuple of (dn, db) normal/binormal offsets in m, one per filament.

    Returns:
        tuple: Base (gamma, gammadash, gammadashdash), each shape (nquadpoints, 3)
            in m, then filament gamma and gammadash, each shape
            (nfilaments, nquadpoints, 3) in m.
    """
    frame, input_jets = _curve_filament_input_jets(spec, dofs, 1)
    # Both frames take (gamma, gammadash, ...) of the base curve, so the frame
    # inputs start with gamma and their parameter derivatives with
    # (gammadash, gammadashdash).
    base_geometry = (input_jets[0][0], *input_jets[1][:2])
    dn, db = (
        _as_jax_float64(np.asarray(values, dtype=np.float64))
        for values in zip(*offsets, strict=True)
    )
    gammas, gammadashs = jax.vmap(
        lambda filament_dn, filament_db: _filament_terms(
            frame, input_jets, filament_dn, filament_db
        )
    )(dn, db)
    return (*base_geometry, gammas, gammadashs)


def _curve_filament_terms_from_dofs(spec: CurveFilamentSpec, dofs, order: int):
    frame, input_jets = _curve_filament_input_jets(spec, dofs, order)
    dn = _as_jax_float64(spec.dn)
    db = _as_jax_float64(spec.db)
    return _filament_terms(frame, input_jets, dn, db)


def curve_spec_with_dofs(
    spec: CurveSpec,
    dofs,
):
    """Replace a curve spec coefficient vector without mutating the original.

    Args:
        spec (CurveSpec): Immutable geometry, quadrature nodes and static parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.

    Returns:
        CurveSpec object: New frozen payload with converted replacement DOFs;
            other fields retain their original values.
    """
    return replace(spec, dofs=_as_jax_float64(dofs))


def curve_gamma_and_gammadash_from_spec(spec: CurveSpec):
    """Evaluate position and tangent using the DOFs stored in the curve spec.

    Args:
        spec (CurveSpec): Immutable geometry, quadrature nodes and static parameters.

    Returns:
        tuple[jax.Array, ...]: Position and normalized-parameter tangent,
            each shape (Q, 3), in meters.
    """
    return curve_gamma_and_gammadash_from_dofs(spec, spec.dofs)


def curve_gamma_and_gammadash_from_dofs(
    spec: CurveSpec,
    dofs,
) -> tuple[jax.Array, ...]:
    """Evaluate curve position and tangent at the supplied DOFs.

    Args:
        spec (CurveSpec): Immutable geometry, quadrature nodes and static parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.

    Returns:
        tuple[jax.Array, ...]: Position and normalized-parameter tangent,
            each shape (Q, 3), in meters.
    """
    return _curve_geometry_terms_from_dofs(spec, dofs, 1)


def curve_geometry_from_dofs(
    spec: CurveSpec,
    dofs,
) -> tuple[jax.Array, ...]:
    """Evaluate curve position and its first two parameter derivatives.

    Args:
        spec (CurveSpec): Immutable geometry, quadrature nodes and static parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.

    Returns:
        tuple[jax.Array, ...]: Position, first and second normalized-parameter
            derivatives, each shape (Q, 3), in meters.

    Raises:
        ValueError: A perturbation sample in the spec provides fewer than two
            parameter derivatives.
    """
    return _curve_geometry_terms_from_dofs(spec, dofs, 2)
