"""Pure curve-geometry helpers that operate on immutable specs."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.dtypes import _has_jax_array_value, explicit_device_array


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
from .oriented_curve import centercurve_pure
from .specs import (
    CurveFilamentSpec,
    CurveHelicalSpec,
    OrientedCurveXYZFourierSpec,
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

__all__ = [
    "curve_gamma_and_gammadash_from_dofs",
    "curve_gamma_and_gammadash_from_spec",
    "curve_gamma_vjp_from_dofs",
    "curve_geometry_from_dofs",
    "curve_gammadash_vjp_from_dofs",
    "curve_gammadashdash_vjp_from_dofs",
    "curve_gammadashdashdash_vjp_from_dofs",
    "curve_pullback_from_dofs",
    "curve_spec_from_curve",
    "curve_spec_with_dofs",
]


def _runtime_scalar(value: float, *, reference=None) -> jax.Array:
    return _as_explicit_array(value, reference=reference)


def _ones_like_runtime(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_runtime_scalar(1.0, reference=array), array.shape)


def _zeros_like_runtime(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_runtime_scalar(0.0, reference=array), array.shape)


def _as_explicit_array(value, *, reference=None) -> jax.Array:
    """Admit spec-backed values, then convert them to float64."""
    if reference is None and not _has_jax_array_value(value):
        raise TypeError(
            "curve_geometry pure helpers require JAX/spec-backed arrays; "
            "materialize an immutable spec or explicit device array first."
        )
    return _as_jax_float64(value)


def _slice_1d_static(array: jax.Array, start: int, end: int) -> jax.Array:
    """``array[start:end]`` along axis 0 with static bounds.

    Exact selection: dtype and placement follow ``array``; no staged constant.
    """
    return jax.lax.slice_in_dim(array, int(start), int(end), axis=0)


def _update_1d_static(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    """``array`` with ``array[start:start + len(values)]`` replaced by ``values``.

    A full-width ``values`` is the result itself. A partial update is the
    masked sum ``array * keep + placement @ values`` with staged one-hot
    constants: exact for finite entries, and bilinear, so a linearization
    never instantiates zero tangents for the untouched entries (a slice
    concatenation or ``dynamic_update_slice`` would, as host constants).
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
    """Return an immutable JAX spec for supported direct curve objects.

    ``RotatedCurve`` is intentionally not represented as a ``CurveSpec``:
    rotation/reflection placement is a wrapper transform with no owned DOFs.
    JAX coil paths should carry that placement through ``CoilSymmetrySpec``;
    standalone rotated-curve geometry remains a documented CPU-only wrapper.

    Args:
        curve (object): Curve exposing a compatible to_spec() method; RotatedCurve must
            instead use a base spec with coil symmetry.

    Returns:
        CurveSpec object: Immutable JAX geometry payload produced by the
            curve.
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
        else _as_explicit_array(
            dofs,
            reference=spec.dofs,
        )
    )
    if isinstance(spec, CurveXYZFourierSpec):
        return lambda quadpoints: jaxfouriercurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if isinstance(spec, OrientedCurveXYZFourierSpec):
        return lambda quadpoints: centercurve_pure(
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
    """Return ``qp -> d kernel(qp) / d qp`` along ``tangents``."""
    return lambda qp: jax.jvp(kernel, (qp,), (tangents,))[1]


def _curve_geometry_terms_from_kernel(gamma_kernel, quadpoints, *, order) -> tuple[jax.Array, ...]:
    """Return ``gamma_kernel`` and its first ``order`` parameter derivatives by nested JVPs."""
    if order == 0:
        return (gamma_kernel(quadpoints),)
    tangents = _ones_like_runtime(quadpoints)
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
    mapped = _as_explicit_array(
        map_spec.template_full_dofs,
        reference=owner_dofs,
    )
    owner_dofs = _as_explicit_array(
        owner_dofs,
        reference=owner_dofs,
    )
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
    quadpoints = _as_explicit_array(
        rotation_spec.quadpoints, reference=owner_dofs
    )
    if isinstance(rotation_spec, ZeroRotationSpec):
        return (_zeros_like_runtime(quadpoints),) * (order + 1)

    rotation_dofs = optimizable_input_dofs_from_map_spec(rotation_map, owner_dofs)
    rotation_scale = _runtime_scalar(rotation_spec.scale, reference=owner_dofs)

    def alphadash(qp):
        return jaxrotationdash_pure(rotation_dofs, qp, rotation_spec.order)

    dash_terms = _curve_geometry_terms_from_kernel(alphadash, quadpoints, order=order - 1)
    terms = (jaxrotation_pure(rotation_dofs, quadpoints, rotation_spec.order), *dash_terms)
    return tuple(rotation_scale * term for term in terms)


def _curve_geometry_terms_from_dofs(spec: CurveSpec, dofs, order: int) -> tuple[jax.Array, ...]:
    """Return gamma and its first ``order`` parameter derivatives for any curve spec.

    A perturbed layer whose sample lacks a needed derivative raises
    ``ValueError``, as native ``PerturbationSample`` does.
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
    quadpoints = _as_explicit_array(spec.quadpoints, reference=dofs)
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
    """Return a filament's rotated-frame function and the frame inputs with their parameter derivatives.

    The result is ``(frame, input_jets)``: ``frame(*input_jets[0])`` is the
    base curve's rotated frame ``(t, n, b)`` and ``input_jets[j]`` is the
    ``j``-th parameter derivative of those inputs, for ``j`` up to ``order``.
    Every input derivative comes from the base geometry and rotation terms
    themselves, so perturbation samples (constant arrays, not functions of the
    curve parameter) contribute their own derivatives.
    """
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


def _filament_offset(
    gamma: jax.Array,
    normal: jax.Array,
    binormal: jax.Array,
    dn: float | jax.Array,
    db: float | jax.Array,
) -> jax.Array:
    """Apply the normal/binormal offset to the base curve, in native arithmetic order."""
    return gamma + dn * normal + db * binormal


def _filament_terms(frame, input_jets, dn, db) -> tuple[jax.Array, ...]:
    """Return the filament ``gamma + dn * normal + db * binormal`` and its first ``len(input_jets) - 1`` parameter derivatives.

    The ``k``-th derivative map takes the input jets ``0..k``; the next one is
    its JVP along jets ``1..k+1``, the chain rule for a curve parameter that
    moves every input along its own derivative (``J x'' + H[x', x']`` at
    order 2).
    """

    def filament(frame_inputs):
        _tangent, normal, binormal = frame(*frame_inputs)
        return (_filament_offset(frame_inputs[0], normal, binormal, dn, db),)

    def differentiated(terms):
        def extended(*jets):
            primal_terms, tangent_terms = jax.jvp(terms, jets[:-1], jets[1:])
            return (*primal_terms, tangent_terms[-1])

        return extended

    terms = filament
    for _ in input_jets[1:]:
        terms = differentiated(terms)
    return terms(*input_jets)


def _curve_filament_terms_from_dofs(spec: CurveFilamentSpec, dofs, order: int):
    frame, input_jets = _curve_filament_input_jets(spec, dofs, order)
    dn = _runtime_scalar(spec.dn, reference=input_jets[0][0])
    db = _runtime_scalar(spec.db, reference=input_jets[0][0])
    return _filament_terms(frame, input_jets, dn, db)


def curve_spec_with_dofs(
    spec: CurveSpec,
    dofs,
):
    """Replace a curve spec coefficient vector without mutating the original.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
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
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.

    Returns:
        tuple[jax.Array, ...]: Position and its first 1 normalized-parameter
            derivative(s), each shape (Q, 3), in meters.
    """
    return curve_gamma_and_gammadash_from_dofs(spec, spec.dofs)


def curve_gamma_and_gammadash_from_dofs(
    spec: CurveSpec,
    dofs,
) -> tuple[jax.Array, ...]:
    """Return (gamma, gammadash) from a single kernel build and JVP call.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.

    Returns:
        tuple[jax.Array, ...]: Position and its first 1 normalized-parameter
            derivative(s), each shape (Q, 3), in meters.
    """
    return _curve_geometry_terms_from_dofs(spec, dofs, 1)


def curve_geometry_from_dofs(
    spec: CurveSpec,
    dofs,
) -> tuple[jax.Array, ...]:
    """Return (gamma, gammadash, gammadashdash) from a single kernel build.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.

    Returns:
        tuple[jax.Array, ...]: Position and its first 2 normalized-parameter
            derivative(s), each shape (Q, 3), in meters.

    Raises:
        ValueError: A perturbation sample in the spec provides fewer than two
            parameter derivatives, as native ``PerturbationSample`` raises.
    """
    return _curve_geometry_terms_from_dofs(spec, dofs, 2)


def _curve_geometry_term_from_dofs(spec: CurveSpec, dofs, term_index: int):
    return _curve_geometry_terms_from_dofs(spec, dofs, max(term_index, 1))[term_index]


def _curve_geometry_term_vjp_from_dofs(
    spec: CurveSpec,
    dofs,
    cotangent,
    *,
    term_index: int,
):
    curve_dofs = _as_jax_float64(dofs)
    cotangent_jax = _as_jax_float64(cotangent)

    def output(curve_x):
        return _curve_geometry_term_from_dofs(spec, curve_x, term_index)

    _, pullback = jax.vjp(output, curve_dofs)
    (coeff_cotangent,) = pullback(cotangent_jax)
    return coeff_cotangent


def curve_gamma_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    """Pull back the curve position to its coefficient vector.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.
        cotangent (array-like): Cotangent of the position, shape (Q, 3); units follow
            the contracted scalar objective.

    Returns:
        jax.Array: Coefficient cotangent, shape (D,), in the input DOF
            ordering; quadrature nodes are held fixed.
    """
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=0,
    )


def curve_gammadash_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    """Pull back the curve first parameter derivative to its coefficient vector.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.
        cotangent (array-like): Cotangent of the first parameter derivative, shape (Q,
            3); units follow the contracted scalar objective.

    Returns:
        jax.Array: Coefficient cotangent, shape (D,), in the input DOF
            ordering; quadrature nodes are held fixed.
    """
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=1,
    )


def curve_gammadashdash_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    """Pull back the curve second parameter derivative to its coefficient vector.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.
        cotangent (array-like): Cotangent of the second parameter derivative, shape (Q,
            3); units follow the contracted scalar objective.

    Returns:
        jax.Array: Coefficient cotangent, shape (D,), in the input DOF
            ordering; quadrature nodes are held fixed.

    Raises:
        ValueError: A perturbation sample in the spec provides fewer than two
            parameter derivatives, as native ``PerturbationSample`` raises.
    """
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=2,
    )


def curve_gammadashdashdash_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    """Pull back the curve third parameter derivative to its coefficient vector.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.
        cotangent (array-like): Cotangent of the third parameter derivative, shape (Q,
            3); units follow the contracted scalar objective.

    Returns:
        jax.Array: Coefficient cotangent, shape (D,), in the input DOF
            ordering; quadrature nodes are held fixed.

    Raises:
        ValueError: A perturbation sample in the spec provides fewer than three
            parameter derivatives, as native ``PerturbationSample`` raises.
    """
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=3,
    )


def curve_pullback_from_dofs(spec: CurveSpec, dofs, dg, dgd):
    """Return the coefficient cotangent of ``(gamma, gammadash)`` for one curve spec.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        dofs (array-like): Flat curve DOFs, shape (D,), in the coefficient ordering
            required by the spec.
        dg (array-like): Position cotangent, shape (Q, 3).
        dgd (array-like): First parameter-derivative cotangent, shape (Q, 3).

    Returns:
        jax.Array: Cotangent of sum(dg*gamma + dgd*gammadash), shape (D,), in
            input DOF order; units depend on the contracted objective.
    """
    curve_dofs = _as_jax_float64(dofs)
    dg_jax = _as_jax_float64(dg)
    dgd_jax = _as_jax_float64(dgd)

    def outputs(curve_x):
        return curve_gamma_and_gammadash_from_dofs(spec, curve_x)

    _, pullback = jax.vjp(outputs, curve_dofs)
    (coeff_cotangent,) = pullback((dg_jax, dgd_jax))
    return coeff_cotangent
