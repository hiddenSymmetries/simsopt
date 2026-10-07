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
from .curve_xyz_fourier import (
    jaxfouriercurve_geometry_pure,
    jaxfouriercurve_pure,
)
from .curve_xyz_fourier_symmetries import jaxXYZFourierSymmetriescurve_pure
from ._math_utils import (
    as_compute_array as _as_compute_array,
    as_runtime_array as _as_runtime_array,
)
from .framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotated_frenet_frame,
    rotated_frenet_frame_dash,
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
    curve_spec_kind,
)

__all__ = [
    "curve_filament_frame_from_dofs",
    "curve_gamma_and_dash_from_dofs",
    "curve_gamma_and_dash_from_spec",
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
    return _as_explicit_runtime_array(value, reference=reference)


def _ones_like_runtime(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_runtime_scalar(1.0, reference=array), array.shape)


def _zeros_like_runtime(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_runtime_scalar(0.0, reference=array), array.shape)


def _as_explicit_runtime_array(value, *, reference=None) -> jax.Array:
    if reference is not None:
        return _as_runtime_array(value)
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return _as_runtime_array(value)
    if isinstance(value, (list, tuple)):
        leaves = jax.tree.leaves(value)
        if any(isinstance(leaf, jax.Array) or hasattr(leaf, "aval") for leaf in leaves):
            return _as_runtime_array(value)
    raise TypeError(
        "curve_geometry pure helpers require JAX/spec-backed arrays; "
        "materialize an immutable spec or explicit device array first."
    )


def _as_explicit_compute_array(value, *, reference=None) -> jax.Array:
    if reference is not None:
        return _as_compute_array(value)
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return _as_compute_array(value)
    if isinstance(value, (list, tuple)):
        leaves = jax.tree.leaves(value)
        if any(isinstance(leaf, jax.Array) or hasattr(leaf, "aval") for leaf in leaves):
            return _as_compute_array(value)
    raise TypeError(
        "curve_geometry compute helpers require JAX/spec-backed arrays; "
        "materialize an immutable spec or explicit device array first."
    )


def _as_explicit_array(value, *, reference=None, use_compute_dtype: bool = False):
    if use_compute_dtype:
        return _as_explicit_compute_array(value, reference=reference)
    return _as_explicit_runtime_array(value, reference=reference)


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
    *,
    use_compute_dtype: bool = False,
):
    curve_dofs = (
        spec.dofs
        if dofs is None
        else _as_explicit_array(
            dofs,
            reference=spec.dofs,
            use_compute_dtype=use_compute_dtype,
        )
    )
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "xyz_fourier":
        spec = cast(CurveXYZFourierSpec, spec)
        return lambda quadpoints: jaxfouriercurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if spec_kind == "oriented_xyz_fourier":
        spec = cast(OrientedCurveXYZFourierSpec, spec)
        return lambda quadpoints: centercurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if spec_kind == "rz_fourier":
        spec = cast(CurveRZFourierSpec, spec)
        return lambda quadpoints: curverzfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
        )
    if spec_kind == "planar_fourier":
        spec = cast(CurvePlanarFourierSpec, spec)
        return lambda quadpoints: curveplanarfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if spec_kind == "helical":
        spec = cast(CurveHelicalSpec, spec)
        return lambda quadpoints: curve_helical_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.m,
            spec.ell,
            spec.R0,
            spec.r,
        )
    if spec_kind == "xyz_fourier_symmetries":
        spec = cast(CurveXYZFourierSymmetriesSpec, spec)
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


def _curve_quadpoints(spec: CurveSpec, *, reference):
    quadpoints = _as_explicit_runtime_array(spec.quadpoints, reference=reference)
    return quadpoints, _ones_like_runtime(quadpoints)


def _curve_geometry_terms_from_kernel(gamma_kernel, quadpoints, tangents, *, order) -> tuple[jax.Array, ...]:
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    if order == 1:
        return gamma, gammadash

    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    if order == 2:
        return gamma, gammadash, gammadashdash

    gammadashdash_kernel = lambda qp: jax.jvp(
        gammadash_kernel,
        (qp,),
        (tangents,),
    )[1]
    _, gammadashdashdash = jax.jvp(
        gammadashdash_kernel,
        (quadpoints,),
        (tangents,),
    )
    return gamma, gammadash, gammadashdash, gammadashdashdash


def _direct_curve_geometry_terms(spec: CurveSpec, dofs, *, order):
    if curve_spec_kind(spec) != "xyz_fourier":
        return None
    spec = cast(CurveXYZFourierSpec, spec)
    curve_dofs = spec.dofs if dofs is None else dofs
    geometry = jaxfouriercurve_geometry_pure(
        curve_dofs,
        spec.quadpoints,
        spec.order,
    )
    return geometry[: order + 1]


def _mapped_full_dofs(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
    *,
    use_compute_dtype: bool = False,
):
    mapped = _as_explicit_array(
        map_spec.template_full_dofs,
        reference=owner_dofs,
        use_compute_dtype=use_compute_dtype,
    )
    owner_dofs = _as_explicit_array(
        owner_dofs,
        reference=owner_dofs,
        use_compute_dtype=use_compute_dtype,
    )
    for owner_start, owner_end, target_start, target_end in map_spec.owner_segments:
        del target_end
        mapped = _update_1d_static(
            mapped,
            target_start,
            _slice_1d_static(owner_dofs, owner_start, owner_end),
        )
    return mapped


def _mapped_input_dofs(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
    *,
    use_compute_dtype: bool = False,
):
    mapped_full = _mapped_full_dofs(
        map_spec,
        owner_dofs,
        use_compute_dtype=use_compute_dtype,
    )
    if map_spec.input_mode == "full":
        return mapped_full
    return _slice_1d_static(mapped_full, map_spec.input_start, map_spec.input_end)


def optimizable_input_dofs_from_map_spec(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
    *,
    use_compute_dtype: bool = False,
):
    return _mapped_input_dofs(
        map_spec,
        owner_dofs,
        use_compute_dtype=use_compute_dtype,
    )


def _rotation_alpha_and_dash_from_dofs(
    rotation_spec: RotationSpec,
    rotation_map: OptimizableDofMapSpec,
    owner_dofs,
):
    quadpoints = _as_explicit_runtime_array(
        rotation_spec.quadpoints, reference=owner_dofs
    )
    if isinstance(rotation_spec, ZeroRotationSpec):
        zeros = _zeros_like_runtime(quadpoints)
        return zeros, zeros

    rotation_dofs = _mapped_input_dofs(rotation_map, owner_dofs)
    rotation_scale = _runtime_scalar(rotation_spec.scale, reference=owner_dofs)
    return (
        rotation_scale
        * jaxrotation_pure(rotation_dofs, quadpoints, rotation_spec.order),
        rotation_scale
        * jaxrotationdash_pure(rotation_dofs, quadpoints, rotation_spec.order),
    )


def _curve_geometry_with_third_derivative_from_dofs(
    spec: CurveSpec,
    dofs,
    *,
    use_compute_dtype: bool = False,
) -> tuple[jax.Array, ...]:
    """Return (gamma, gammadash, gammadashdash, gammadashdashdash) in one pass."""
    if isinstance(spec, CurvePerturbedSpec):
        base_geometry = _curve_geometry_with_third_derivative_from_dofs(
            spec.base_curve,
            _curve_perturbed_base_dofs(spec, dofs),
            use_compute_dtype=use_compute_dtype,
        )
        return _add_curve_perturbation(spec, *base_geometry)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=3)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(
        spec,
        dofs,
        use_compute_dtype=use_compute_dtype,
    )
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=3,
    )


def _curve_perturbed_base_dofs(spec: CurvePerturbedSpec, dofs):
    return _mapped_input_dofs(spec.base_curve_map, dofs)


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


def _curve_perturbed_gamma_and_dash_from_dofs(spec: CurvePerturbedSpec, dofs) -> tuple[jax.Array, ...]:
    base_geometry = curve_gamma_and_dash_from_dofs(
        spec.base_curve,
        _curve_perturbed_base_dofs(spec, dofs),
    )
    return _add_curve_perturbation(spec, *base_geometry)


def _curve_perturbed_geometry_from_dofs(spec: CurvePerturbedSpec, dofs) -> tuple[jax.Array, ...]:
    base_geometry = curve_geometry_from_dofs(
        spec.base_curve,
        _curve_perturbed_base_dofs(spec, dofs),
    )
    return _add_curve_perturbation(spec, *base_geometry)


def _curve_spec_with_quadpoints(spec: CurveSpec, quadpoints):
    quadpoints_jax = _as_explicit_runtime_array(quadpoints, reference=spec.dofs)
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return replace(
            spec,
            quadpoints=quadpoints_jax,
            base_curve=_curve_spec_with_quadpoints(spec.base_curve, quadpoints_jax),
        )
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return replace(
            spec,
            quadpoints=quadpoints_jax,
            base_curve=_curve_spec_with_quadpoints(spec.base_curve, quadpoints_jax),
            rotation=replace(spec.rotation, quadpoints=quadpoints_jax),
        )
    return replace(spec, quadpoints=quadpoints_jax)


def _curve_filament_geometry_from_dofs(spec: CurveFilamentSpec, dofs):
    def gamma_kernel(qp):
        quad_spec = cast(CurveFilamentSpec, _curve_spec_with_quadpoints(spec, qp))
        base_dofs = _mapped_input_dofs(quad_spec.base_curve_map, dofs)
        alpha, _alphadash = _rotation_alpha_and_dash_from_dofs(
            quad_spec.rotation,
            quad_spec.rotation_map,
            dofs,
        )
        gamma, gammadash = curve_gamma_and_dash_from_dofs(
            quad_spec.base_curve, base_dofs
        )
        if quad_spec.frame_kind == "frenet":
            _gamma, _gammadash, gammadashdash = curve_geometry_from_dofs(
                quad_spec.base_curve,
                base_dofs,
            )
            _tangent, normal, binormal = rotated_frenet_frame(
                gamma,
                gammadash,
                gammadashdash,
                alpha,
            )
        else:
            _tangent, normal, binormal = rotated_centroid_frame(
                gamma,
                gammadash,
                alpha,
            )
        return _filament_offset(gamma, normal, binormal, quad_spec.dn, quad_spec.db)

    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    return gamma, gammadash, gammadashdash


def curve_filament_frame_from_dofs(spec: CurveFilamentSpec, dofs) -> tuple[jax.Array, ...]:
    """Return the underlying curve and rotated frame of a finite-build filament.

    The result is ``(gamma, gammadash, gammadashdash, normal, binormal,
    normal_dash, binormal_dash)`` of the filament's base curve; the filament
    is ``gamma + dn * normal + db * binormal`` and its tangent
    ``gammadash + dn * normal_dash + db * binormal_dash``.
    """
    base_dofs = _mapped_input_dofs(spec.base_curve_map, dofs)
    alpha, alphadash = _rotation_alpha_and_dash_from_dofs(
        spec.rotation,
        spec.rotation_map,
        dofs,
    )

    if spec.frame_kind == "frenet":
        gamma, gammadash, gammadashdash, gammadashdashdash = (
            _curve_geometry_with_third_derivative_from_dofs(spec.base_curve, base_dofs)
        )
        _tangent, normal, binormal = rotated_frenet_frame(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
        )
        _tangent_dash, normal_dash, binormal_dash = rotated_frenet_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
        )
    else:
        gamma, gammadash, gammadashdash = curve_geometry_from_dofs(
            spec.base_curve, base_dofs
        )
        _tangent, normal, binormal = rotated_centroid_frame(
            gamma,
            gammadash,
            alpha,
        )
        _tangent_dash, normal_dash, binormal_dash = rotated_centroid_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
        )
    return gamma, gammadash, gammadashdash, normal, binormal, normal_dash, binormal_dash


def _filament_offset(
    gamma: jax.Array,
    normal: jax.Array,
    binormal: jax.Array,
    dn: float | jax.Array,
    db: float | jax.Array,
) -> jax.Array:
    """Apply the normal/binormal offset to a frame component, in native arithmetic order."""
    return gamma + dn * normal + db * binormal


def _curve_filament_gamma_and_dash_from_dofs(spec: CurveFilamentSpec, dofs):
    gamma, gammadash, _gammadashdash, normal, binormal, normal_dash, binormal_dash = (
        curve_filament_frame_from_dofs(spec, dofs)
    )
    dn = _runtime_scalar(spec.dn, reference=normal)
    db = _runtime_scalar(spec.db, reference=binormal)
    return (
        _filament_offset(gamma, normal, binormal, dn, db),
        _filament_offset(gammadash, normal_dash, binormal_dash, dn, db),
    )


def curve_spec_with_dofs(
    spec: CurveSpec,
    dofs,
    *,
    use_compute_dtype: bool = False,
):
    if use_compute_dtype:
        return replace(spec, dofs=_as_compute_array(dofs))
    return replace(spec, dofs=_as_runtime_array(dofs))


def curve_gamma_and_dash_from_spec(spec: CurveSpec):
    return curve_gamma_and_dash_from_dofs(spec, spec.dofs)


def curve_gamma_and_dash_from_dofs(
    spec: CurveSpec,
    dofs,
    *,
    use_compute_dtype: bool = False,
) -> tuple[jax.Array, ...]:
    """Return (gamma, gammadash) from a single kernel build and JVP call."""
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return _curve_perturbed_gamma_and_dash_from_dofs(spec, dofs)
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return _curve_filament_gamma_and_dash_from_dofs(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=1)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(
        spec,
        dofs,
        use_compute_dtype=use_compute_dtype,
    )
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=1,
    )


def curve_geometry_from_dofs(
    spec: CurveSpec,
    dofs,
    *,
    use_compute_dtype: bool = False,
) -> tuple[jax.Array, ...]:
    """Return (gamma, gammadash, gammadashdash) from a single kernel build."""
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return _curve_perturbed_geometry_from_dofs(spec, dofs)
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return _curve_filament_geometry_from_dofs(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=2)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(
        spec,
        dofs,
        use_compute_dtype=use_compute_dtype,
    )
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=2,
    )


def _curve_geometry_term_from_dofs(spec: CurveSpec, dofs, term_index: int):
    if term_index < 2:
        return curve_gamma_and_dash_from_dofs(spec, dofs)[term_index]
    if term_index == 2:
        return curve_geometry_from_dofs(spec, dofs)[2]
    return _curve_geometry_with_third_derivative_from_dofs(spec, dofs)[3]


def _curve_geometry_term_vjp_from_dofs(
    spec: CurveSpec,
    dofs,
    cotangent,
    *,
    term_index: int,
):
    curve_dofs = _as_runtime_array(dofs)
    cotangent_jax = _as_runtime_array(cotangent)

    def output(curve_x):
        return _curve_geometry_term_from_dofs(spec, curve_x, term_index)

    _, pullback = jax.vjp(output, curve_dofs)
    (coeff_cotangent,) = pullback(cotangent_jax)
    return coeff_cotangent


def curve_gamma_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=0,
    )


def curve_gammadash_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=1,
    )


def curve_gammadashdash_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=2,
    )


def curve_gammadashdashdash_vjp_from_dofs(spec: CurveSpec, dofs, cotangent):
    return _curve_geometry_term_vjp_from_dofs(
        spec,
        dofs,
        cotangent,
        term_index=3,
    )


def curve_pullback_from_dofs(spec: CurveSpec, dofs, dg, dgd):
    """Return the coefficient cotangent of ``(gamma, gammadash)`` for one curve spec."""
    curve_dofs = _as_runtime_array(dofs)
    dg_jax = _as_runtime_array(dg)
    dgd_jax = _as_runtime_array(dgd)

    def outputs(curve_x):
        return curve_gamma_and_dash_from_dofs(spec, curve_x)

    _, pullback = jax.vjp(outputs, curve_dofs)
    (coeff_cotangent,) = pullback((dg_jax, dgd_jax))
    return coeff_cotangent
