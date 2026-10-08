"""Pure grouped-field helpers that operate on immutable specs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from ._math_utils import (
    as_compute_array as _as_compute_array,
)
from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
)
from ._math_utils import (
    runtime_device_put,
)
from .biotsavart import (
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
    curve_gamma_and_dash_from_spec,
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

__all__ = [
    "coil_set_spec_from_dof_extraction_spec",
    "coil_specs_from_dof_extraction_spec",
    "group_biot_savart_B_vjp",
    "grouped_coil_set_spec_from_coil_specs",
    "grouped_biot_savart_A_from_inputs",
    "grouped_biot_savart_A_from_spec",
    "grouped_biot_savart_B_and_dB_from_spec",
    "grouped_biot_savart_B_from_spec",
    "grouped_biot_savart_d2A_by_dXdX_from_spec",
    "grouped_biot_savart_d2B_by_dXdX_from_spec",
    "grouped_biot_savart_dA_by_dX_from_inputs",
    "grouped_biot_savart_dA_by_dX_from_spec",
    "grouped_biot_savart_dB_by_dX_from_inputs",
    "grouped_biot_savart_dB_by_dX_from_spec",
    "grouped_coil_set_spec_from_inputs",
    "grouped_coil_set_spec_from_lists",
    "grouped_field_data_from_spec",
    "grouped_field_inputs_from_spec",
]


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


def _tree_add(left, right):
    return jax.tree.map(lambda x, y: x + y, left, right)


def _compute_group_inputs(points, gammas, gammadashs, currents):
    """Cast one coil group and the points to the points' floating dtype."""
    field_dtype = jnp.asarray(points).dtype
    return (
        _as_compute_array(points, dtype=field_dtype),
        _as_compute_array(gammas, dtype=field_dtype),
        _as_compute_array(gammadashs, dtype=field_dtype),
        _as_compute_array(currents, dtype=field_dtype),
    )


def _evaluate_grouped_field_group(points, gammas, gammadashs, currents, kernel):
    return kernel(*_compute_group_inputs(points, gammas, gammadashs, currents))


def _accumulate_grouped_field(points: object, coil_spec: GroupedCoilSetSpec, kernel):
    coil_arrays = grouped_field_inputs_from_spec(coil_spec)
    if not coil_arrays:
        return _empty_grouped_field_result(cast(jax.Array, points), kernel)
    result = _evaluate_grouped_field_group(points, *coil_arrays[0], kernel)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = _tree_add(
            result,
            _evaluate_grouped_field_group(points, gammas, gammadashs, currents, kernel),
        )
    return result


def group_biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    """Return the ``B`` pullback for one coil group in the points' dtype."""
    compute_points, gammas, gammadashs, currents = _compute_group_inputs(
        points,
        gammas,
        gammadashs,
        currents,
    )
    compute_v = _as_compute_array(v, dtype=compute_points.dtype)
    return biot_savart_B_vjp(compute_points, compute_v, gammas, gammadashs, currents)


def grouped_coil_set_spec_from_lists(
    gammas_list: object,
    gammadashs_list: object,
    currents_list: object,
) -> GroupedCoilSetSpec:
    return make_grouped_coil_set_spec(
        group_coil_data(
            gammas_list,
            gammadashs_list,
            currents_list,
            use_compute_dtype=False,
        )
    )


def grouped_coil_set_spec_from_coil_specs(
    coil_specs: tuple[CoilSpec, ...] | list[CoilSpec],
) -> GroupedCoilSetSpec:
    gammas = []
    gammadashs = []
    currents = []
    geometry_by_curve: dict[int, tuple[jax.Array, jax.Array]] = {}
    for coil_spec in coil_specs:
        curve_id = id(coil_spec.curve)
        geometry = geometry_by_curve.get(curve_id)
        if geometry is None:
            geometry = cast(tuple[jax.Array, jax.Array], curve_gamma_and_dash_from_spec(coil_spec.curve))
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
    return grouped_coil_set_spec_from_lists(gammas, gammadashs, currents)


def _coil_current_value_from_dofs(
    extraction_spec: CoilDofExtractionSpec,
    owner_dofs: object,
    *,
    use_compute_dtype: bool = False,
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
                use_compute_dtype=use_compute_dtype,
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
        use_compute_dtype=use_compute_dtype,
    )
    if current_dofs.shape[0] != 1:
        raise RuntimeError(
            "coil_specs_from_dof_extraction_spec() only supports scalar Current "
            "degrees of freedom."
        )
    return CurrentValueSpec(value=current_dofs[:1])


def _coil_curve_spec_from_dofs(
    extraction_spec: CoilDofExtractionSpec,
    owner_dofs: object,
    *,
    use_compute_dtype: bool = False,
) -> CurveSpec:
    return curve_spec_with_dofs(
        extraction_spec.curve,
        optimizable_input_dofs_from_map_spec(
            extraction_spec.curve_map,
            owner_dofs,
            use_compute_dtype=use_compute_dtype,
        ),
    )


def coil_specs_from_dof_extraction_spec(
    extraction_spec: CoilSetDofExtractionSpec,
    owner_dofs: object,
    *,
    use_compute_dtype: bool = False,
) -> tuple[CoilSpec, ...]:
    if use_compute_dtype:
        owner_dofs = _as_compute_array(owner_dofs)
    else:
        owner_dofs = _as_jax_float64(owner_dofs)
    curves_by_source: dict[int, CurveSpec] = {}
    coil_specs = []
    for coil_spec in extraction_spec.coils:
        source_index = coil_spec.curve_source_index
        if source_index is None or source_index not in curves_by_source:
            curve = _coil_curve_spec_from_dofs(
                coil_spec,
                owner_dofs,
                use_compute_dtype=use_compute_dtype,
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
                    use_compute_dtype=use_compute_dtype,
                ),
                symmetry=coil_spec.symmetry,
            )
        )
    return tuple(coil_specs)


def coil_set_spec_from_dof_extraction_spec(
    extraction_spec: CoilSetDofExtractionSpec,
    owner_dofs: object,
    *,
    use_compute_dtype: bool = False,
) -> GroupedCoilSetSpec:
    return grouped_coil_set_spec_from_coil_specs(
        coil_specs_from_dof_extraction_spec(
            extraction_spec,
            owner_dofs,
            use_compute_dtype=use_compute_dtype,
        )
    )


def grouped_coil_set_spec_from_inputs(coil_arrays: Iterable[tuple[jax.Array, jax.Array, jax.Array]]) -> GroupedCoilSetSpec:
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


def grouped_field_inputs_from_spec(
    coil_spec: GroupedCoilSetSpec,
) -> tuple[tuple[object, object, object], ...]:
    return coil_spec.field_inputs()


def grouped_field_data_from_spec(
    coil_spec: GroupedCoilSetSpec,
) -> tuple[tuple[object, object, object, list[int]], ...]:
    return coil_spec.as_grouped_data()


def grouped_biot_savart_B_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_B)


def grouped_biot_savart_A_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_A)


def grouped_biot_savart_A_from_inputs(points: object, coil_arrays: Iterable[tuple[jax.Array, jax.Array, jax.Array]]):
    return grouped_biot_savart_A_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_dA_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dA_by_dX)


def grouped_biot_savart_dA_by_dX_from_inputs(points: object, coil_arrays: Iterable[tuple[jax.Array, jax.Array, jax.Array]]):
    return grouped_biot_savart_dA_by_dX_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_d2A_by_dXdX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_d2A_by_dXdX)


def grouped_biot_savart_d2B_by_dXdX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_d2B_by_dXdX)


def grouped_biot_savart_dB_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dB_by_dX)


def grouped_biot_savart_dB_by_dX_from_inputs(points: object, coil_arrays: Iterable[tuple[jax.Array, jax.Array, jax.Array]]):
    return grouped_biot_savart_dB_by_dX_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_B_and_dB_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    B, dB = _accumulate_grouped_field(points, coil_spec, biot_savart_B_and_dB)
    return B, dB
