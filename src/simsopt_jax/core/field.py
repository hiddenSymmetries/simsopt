"""Pure grouped-field helpers that operate on immutable specs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
)
from ._math_utils import (
    runtime_device_put,
)
from .biotsavart import (
    _tree_add,
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_B_vjp,
    biot_savart_d2A_by_dXdX,
    biot_savart_d2B_by_dXdX,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
    group_coil_data,
)
from .curve_geometry import (
    curve_gamma_and_gammadash_from_spec,
    curve_spec_with_dofs,
    optimizable_input_dofs_from_map_spec,
)
from .specs import (
    CoilDofExtractionSpec,
    CoilSetDofExtractionSpec,
    CoilSpec,
    CurrentValueSpec,
    CurveSpec,
    GroupedCoilSetSpec,
    apply_coil_symmetry,
    make_grouped_coil_set_spec,
)


def _zeros_float64(shape):
    return runtime_device_put(np.zeros(shape, dtype=np.float64), dtype=np.float64)


def _empty_grouped_field_result(points: jax.Array, kernel):
    point_count = points.shape[0]
    if kernel in {biot_savart_B, biot_savart_A}:
        return _zeros_float64((point_count, 3))
    if kernel in {biot_savart_dA_by_dX, biot_savart_dB_by_dX}:
        return _zeros_float64((point_count, 3, 3))
    if kernel in {biot_savart_d2A_by_dXdX, biot_savart_d2B_by_dXdX}:
        return _zeros_float64((point_count, 3, 3, 3))
    if kernel is biot_savart_B_and_dB:
        return (
            _zeros_float64((point_count, 3)),
            _zeros_float64((point_count, 3, 3)),
        )
    raise ValueError(f"Unsupported grouped-field kernel: {kernel!r}")


def _accumulate_grouped_field(points: object, coil_spec: GroupedCoilSetSpec, kernel):
    points = _as_jax_float64(points)
    coil_arrays = coil_spec.field_inputs()
    if not coil_arrays:
        return _empty_grouped_field_result(points, kernel)
    result = kernel(points, *coil_arrays[0])
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = _tree_add(result, kernel(points, gammas, gammadashs, currents))
    return result


def group_biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    """Return the ``B`` pullback for one coil group in float64.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        v (array-like): B cotangent, shape (P, 3), contracting the output as sum(v * B);
            units depend on the scalar objective.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Cotangents for positions,
            tangents and currents, shapes (C, Q, 3), (C, Q, 3), (C,). Units are
            those of the contracted scalar per meter or per ampere; points are
            held fixed.
    """
    return biot_savart_B_vjp(points, v, gammas, gammadashs, currents)


def grouped_coil_set_spec_from_coil_specs(
    coil_specs: tuple[CoilSpec, ...] | list[CoilSpec],
) -> GroupedCoilSetSpec:
    """Sample coil specs and group their transformed geometry and currents.

    Args:
        coil_specs (tuple[CoilSpec, ...] or list[CoilSpec]): Immutable coil payloads in
            public coil order.

    Returns:
        GroupedCoilSetSpec object: Sampled geometry and currents after
            symmetry transforms, grouped by quadrature count.
    """
    gammas = []
    gammadashs = []
    currents = []
    geometry_by_curve: dict[int, tuple[jax.Array, jax.Array]] = {}
    for coil_spec in coil_specs:
        curve_id = id(coil_spec.curve)
        geometry = geometry_by_curve.get(curve_id)
        if geometry is None:
            geometry = cast(tuple[jax.Array, jax.Array], curve_gamma_and_gammadash_from_spec(coil_spec.curve))
            geometry_by_curve[curve_id] = geometry
        gamma, gammadash = geometry
        gamma, gammadash, current = apply_coil_symmetry(
            gamma,
            gammadash,
            coil_spec.current.value[0],
            coil_spec.symmetry,
        )
        gammas.append(gamma)
        gammadashs.append(gammadash)
        currents.append(current)
    return make_grouped_coil_set_spec(group_coil_data(gammas, gammadashs, currents))


def _coil_current_value_from_dofs(
    extraction_spec: CoilDofExtractionSpec,
    owner_dofs: object,
) -> CurrentValueSpec:
    if extraction_spec.current_term_maps:
        current_terms = []
        for term_map, scale in zip(
            extraction_spec.current_term_maps,
            extraction_spec.current_term_scales,
            strict=True,
        ):
            term_dofs = optimizable_input_dofs_from_map_spec(
                term_map,
                owner_dofs,
            )
            if term_dofs.shape[0] != 1:
                raise RuntimeError(
                    "affine coil current terms must resolve to scalar Current "
                    "degrees of freedom."
                )
            current_terms.append(_as_jax_float64(scale) * term_dofs[0])
        current_value = jnp.sum(jnp.stack(current_terms))
        return CurrentValueSpec(value=jnp.reshape(current_value, (1,)))

    current_dofs = optimizable_input_dofs_from_map_spec(
        extraction_spec.current_map,
        owner_dofs,
    )
    if current_dofs.shape[0] != 1:
        raise RuntimeError(
            "coil_specs_from_dof_extraction_spec() only supports scalar Current "
            "degrees of freedom."
        )
    return CurrentValueSpec(value=current_dofs[:1])


def coil_specs_from_dof_extraction_spec(
    extraction_spec: CoilSetDofExtractionSpec,
    owner_dofs: object,
) -> tuple[CoilSpec, ...]:
    """Reconstruct immutable coil inputs from an explicit owner DOF vector.

    Args:
        extraction_spec (CoilSetDofExtractionSpec): Frozen owner-to-curve/current
            reconstruction contracts.
        owner_dofs (array-like): Flat owner DOF vector, shape (D,), matching the
            extraction map; units depend on the owning geometry or current.

    Returns:
        tuple[CoilSpec, ...]: Per-coil immutable payloads in public coil
            order, sharing reconstructed curves when their source keys match.
    """
    owner_dofs = _as_jax_float64(owner_dofs)
    curves_by_source: dict[int, CurveSpec] = {}
    coil_specs = []
    for coil_spec in extraction_spec.coils:
        source_index = coil_spec.curve_source_index
        if source_index is None or source_index not in curves_by_source:
            curve = curve_spec_with_dofs(
                coil_spec.curve,
                optimizable_input_dofs_from_map_spec(
                    coil_spec.curve_map,
                    owner_dofs,
                ),
            )
            if source_index is not None:
                curves_by_source[source_index] = curve
        else:
            curve = curves_by_source[source_index]
        coil_specs.append(
            CoilSpec(
                curve=curve,
                current=_coil_current_value_from_dofs(
                    coil_spec,
                    owner_dofs,
                ),
                symmetry=coil_spec.symmetry,
            )
        )
    return tuple(coil_specs)


def coil_set_spec_from_dof_extraction_spec(
    extraction_spec: CoilSetDofExtractionSpec,
    owner_dofs: object,
) -> GroupedCoilSetSpec:
    """Reconstruct immutable coil inputs from an explicit owner DOF vector.

    Args:
        extraction_spec (CoilSetDofExtractionSpec): Frozen owner-to-curve/current
            reconstruction contracts.
        owner_dofs (array-like): Flat owner DOF vector, shape (D,), matching the
            extraction map; units depend on the owning geometry or current.

    Returns:
        GroupedCoilSetSpec object: Reconstructed geometry and currents grouped
            by quadrature count with original coil indices.
    """
    return grouped_coil_set_spec_from_coil_specs(
        coil_specs_from_dof_extraction_spec(
            extraction_spec,
            owner_dofs,
        )
    )


def grouped_coil_set_spec_from_inputs(coil_arrays: Iterable[tuple[jax.Array, jax.Array, jax.Array]]) -> GroupedCoilSetSpec:
    """Wrap pregrouped geometry and currents in immutable coil specs.

    Args:
        coil_arrays (Iterable[tuple]): Groups of (gammas, gammadashs, currents) arrays
            with shapes (C, Q, 3), (C, Q, 3), (C,); Q may differ between groups.
            Positions and tangents are in meters; currents are in amperes.

    Returns:
        GroupedCoilSetSpec object: Immutable groups with sequential coil
            indices assigned in input group/row order.
    """
    groups = []
    coil_offset = 0
    for gammas, gammadashs, currents in coil_arrays:
        group_size = currents.shape[0]
        groups.append(
            (
                gammas,
                gammadashs,
                currents,
                tuple(range(coil_offset, coil_offset + group_size)),
            )
        )
        coil_offset += group_size
    return make_grouped_coil_set_spec(groups)


def grouped_biot_savart_B_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    """Evaluate total B over all quadrature groups.

    An empty coil set returns zeros of the matching output shape.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        coil_spec (GroupedCoilSetSpec): Immutable geometry and currents grouped by
            quadrature count.

    Returns:
        jax.Array: Magnetic field in tesla, shape (P, 3), Cartesian component
            last.
    """
    return _accumulate_grouped_field(points, coil_spec, biot_savart_B)


def grouped_biot_savart_A_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    """Evaluate total A over all quadrature groups.

    An empty coil set returns zeros of the matching output shape.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        coil_spec (GroupedCoilSetSpec): Immutable geometry and currents grouped by
            quadrature count.

    Returns:
        jax.Array: Vector potential in tesla meters, shape (P, 3), Cartesian
            component last.
    """
    return _accumulate_grouped_field(points, coil_spec, biot_savart_A)


def grouped_biot_savart_dA_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    """Return the total vector potential's spatial Jacobian.

    An empty coil set returns zeros of the matching output shape.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        coil_spec (GroupedCoilSetSpec): Immutable geometry and currents grouped by
            quadrature count.

    Returns:
        jax.Array: Shape (P, 3, 3), in tesla; result[p, j, l] = partial_j A_l
            at point p.
    """
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dA_by_dX)


def grouped_biot_savart_d2A_by_dXdX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    """Return the total vector potential's spatial Hessian.

    An empty coil set returns zeros of the matching output shape.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        coil_spec (GroupedCoilSetSpec): Immutable geometry and currents grouped by
            quadrature count.

    Returns:
        jax.Array: Shape (P, 3, 3, 3), in tesla per meter; result[p, i, j, l]
            = partial_i partial_j A_l.
    """
    return _accumulate_grouped_field(points, coil_spec, biot_savart_d2A_by_dXdX)


def grouped_biot_savart_d2B_by_dXdX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    """Return the total magnetic field's spatial Hessian.

    An empty coil set returns zeros of the matching output shape.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        coil_spec (GroupedCoilSetSpec): Immutable geometry and currents grouped by
            quadrature count.

    Returns:
        jax.Array: Shape (P, 3, 3, 3), in tesla per meter squared; result[p,
            i, j, l] = partial_i partial_j B_l.
    """
    return _accumulate_grouped_field(points, coil_spec, biot_savart_d2B_by_dXdX)


def grouped_biot_savart_dB_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    """Return the total magnetic field's spatial Jacobian.

    An empty coil set returns zeros of the matching output shape.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        coil_spec (GroupedCoilSetSpec): Immutable geometry and currents grouped by
            quadrature count.

    Returns:
        jax.Array: Shape (P, 3, 3), in tesla per meter; result[p, j, l] =
            partial_j B_l at point p.
    """
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dB_by_dX)


def grouped_biot_savart_B_and_dB_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    """Return the total magnetic field and its spatial Jacobian.

    An empty coil set returns zeros of the matching output shape.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        coil_spec (GroupedCoilSetSpec): Immutable geometry and currents grouped by
            quadrature count.

    Returns:
        tuple[jax.Array, jax.Array]: B of shape (P, 3) in tesla and its
            Jacobian of shape (P, 3, 3) in tesla per meter, with derivative
            direction before field component.
    """
    return _accumulate_grouped_field(points, coil_spec, biot_savart_B_and_dB)
