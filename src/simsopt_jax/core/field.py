"""Pure grouped-field helpers that operate on immutable specs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from ._math_utils import (
    as_compute_array as _as_compute_array,
)
from ._math_utils import (
    as_runtime_float64 as _as_runtime_float64,
)
from ._math_utils import (
    pad_axis as _pad_axis,
)
from ._math_utils import (
    runtime_device_put,
)
from .biotsavart import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_B_and_dB_with_point_axis,
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
from .sharding import (
    coil_group_collective_config,
    maybe_shard_grouped_field_inputs,
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


def _tree_trim_axis0(tree, size: int):
    return jax.tree.map(
        lambda leaf: lax.slice_in_dim(leaf, start_index=0, limit_index=size, axis=0),
        tree,
    )


def _compute_group_inputs(reference, gammas, gammadashs, currents):
    field_dtype = jnp.asarray(reference).dtype
    return (
        _as_compute_array(gammas, dtype=field_dtype, reference=reference),
        _as_compute_array(gammadashs, dtype=field_dtype, reference=reference),
        _as_compute_array(currents, dtype=field_dtype, reference=reference),
    )


def _pad_coil_axis_to_device_count(gammas, gammadashs, currents, device_count: int):
    coil_count = int(currents.shape[0])
    pad_count = (-coil_count) % device_count
    if pad_count == 0:
        return gammas, gammadashs, currents
    padded_count = coil_count + pad_count
    # Sharding requires axis sizes divisible by the device count. The padding
    # cost is bounded by device_count - 1 entries; keep this simple unless a
    # JAX device-memory profile shows material peak-memory pressure.
    return (
        _pad_axis(gammas, axis=0, padded_size=padded_count),
        _pad_axis(gammadashs, axis=0, padded_size=padded_count),
        _pad_axis(currents, axis=0, padded_size=padded_count),
    )


def _pad_point_axis_to_device_count(points, device_count: int):
    point_count = int(points.shape[0])
    pad_count = (-point_count) % device_count
    if pad_count == 0:
        return points
    return _pad_axis(points, axis=0, padded_size=point_count + pad_count)


def _field_out_specs(kernel, config):
    point_axis_name = config.point_axis_name
    if point_axis_name is None:
        if kernel is biot_savart_B_and_dB:
            return P(), P()
        return P()
    if kernel is biot_savart_B_and_dB:
        return P(point_axis_name, None), P(point_axis_name, None, None)
    if kernel in {biot_savart_B, biot_savart_A}:
        return P(point_axis_name, None)
    if kernel in {biot_savart_dA_by_dX, biot_savart_dB_by_dX}:
        return P(point_axis_name, None, None)
    if kernel in {biot_savart_d2A_by_dXdX, biot_savart_d2B_by_dXdX}:
        return P(point_axis_name, None, None, None)
    raise ValueError(f"Unsupported grouped-field kernel: {kernel!r}")


def _axis_partition_spec(axis_name: str, ndim: int):
    if ndim <= 0:
        return P()
    return P(axis_name, *([None] * (ndim - 1)))


def _place_collective_group_inputs(points, gammas, gammadashs, currents, config):
    point_spec = P()
    if config.point_axis_name is not None:
        point_spec = _axis_partition_spec(config.point_axis_name, int(points.ndim))
    return (
        runtime_device_put(points, target=NamedSharding(config.mesh, point_spec)),
        runtime_device_put(
            gammas,
            target=NamedSharding(
                config.mesh,
                _axis_partition_spec(config.coil_axis_name, int(gammas.ndim)),
            ),
        ),
        runtime_device_put(
            gammadashs,
            target=NamedSharding(
                config.mesh,
                _axis_partition_spec(config.coil_axis_name, int(gammadashs.ndim)),
            ),
        ),
        runtime_device_put(
            currents,
            target=NamedSharding(
                config.mesh,
                _axis_partition_spec(config.coil_axis_name, int(currents.ndim)),
            ),
        ),
    )


def _collective_kernel(kernel, config):
    if config.point_axis_name is not None and kernel is biot_savart_B_and_dB:
        return partial(
            biot_savart_B_and_dB_with_point_axis,
            point_axis_name=config.point_axis_name,
        )
    return kernel


def _collective_group_field(points, gammas, gammadashs, currents, kernel, config):
    point_count = int(points.shape[0])
    if config.point_axis_name is not None:
        points = _pad_point_axis_to_device_count(points, config.point_device_count)
    group_kernel = _collective_kernel(kernel, config)
    gammas, gammadashs, currents = _pad_coil_axis_to_device_count(
        gammas,
        gammadashs,
        currents,
        config.coil_device_count,
    )
    # ``shard_map`` requires every input to already match its declared
    # mesh/in_spec placement. Make the boundary explicit so single-device JAX
    # arrays do not leak into multi-device collectives.
    points, gammas, gammadashs, currents = _place_collective_group_inputs(
        points,
        gammas,
        gammadashs,
        currents,
        config,
    )
    point_spec = (
        P() if config.point_axis_name is None else P(config.point_axis_name, None)
    )

    @partial(
        jax.shard_map,
        mesh=config.mesh,
        in_specs=(
            point_spec,
            P(config.coil_axis_name, None, None),
            P(config.coil_axis_name, None, None),
            P(config.coil_axis_name),
        ),
        out_specs=_field_out_specs(kernel, config),
        check_vma=True,
    )
    def _group_kernel(points_block, gammas_block, gammadashs_block, currents_block):
        return jax.tree.map(
            lambda value: lax.psum(value, config.reduced_axis_name),
            group_kernel(
                points_block,
                gammas_block,
                gammadashs_block,
                currents_block,
            ),
        )

    result = _group_kernel(points, gammas, gammadashs, currents)
    if config.point_axis_name is None:
        return result
    return _tree_trim_axis0(result, point_count)


def _evaluate_grouped_field_group(points, gammas, gammadashs, currents, kernel):
    point_dtype = jnp.asarray(points).dtype
    compute_points = _as_compute_array(points, dtype=point_dtype)
    gammas, gammadashs, currents = _compute_group_inputs(
        compute_points,
        gammas,
        gammadashs,
        currents,
    )
    config = coil_group_collective_config(currents)
    if config is None:
        return kernel(compute_points, gammas, gammadashs, currents), config
    return (
        _collective_group_field(
            compute_points,
            gammas,
            gammadashs,
            currents,
            kernel,
            config,
        ),
        config,
    )


def _accumulate_grouped_field_with_config(
    points: object,
    coil_spec: GroupedCoilSetSpec,
    kernel,
):
    coil_arrays = grouped_field_inputs_from_spec(coil_spec)
    if not coil_arrays:
        return _empty_grouped_field_result(cast(jax.Array, points), kernel), None
    points, coil_arrays = maybe_shard_grouped_field_inputs(points, coil_arrays)

    result, collective_config = _evaluate_grouped_field_group(
        points,
        *coil_arrays[0],
        kernel,
    )
    for gammas, gammadashs, currents in coil_arrays[1:]:
        group_result, group_config = _evaluate_grouped_field_group(
            points,
            gammas,
            gammadashs,
            currents,
            kernel,
        )
        result = _tree_add(result, group_result)
        if collective_config is None:
            collective_config = group_config
    return result, collective_config


def _accumulate_grouped_field(points: object, coil_spec: GroupedCoilSetSpec, kernel):
    result, _config = _accumulate_grouped_field_with_config(points, coil_spec, kernel)
    return result


def biot_savart_B_vjp_maybe_collective(points, v, gammas, gammadashs, currents):
    """Return B pullback, using the coil-axis collective path when active."""
    point_dtype = jnp.asarray(points).dtype
    compute_points = _as_compute_array(points, dtype=point_dtype)
    compute_v = _as_compute_array(v, dtype=point_dtype, reference=compute_points)
    gammas, gammadashs, currents = _compute_group_inputs(
        compute_points,
        gammas,
        gammadashs,
        currents,
    )
    config = coil_group_collective_config(currents)
    if config is None:
        return biot_savart_B_vjp(
            compute_points, compute_v, gammas, gammadashs, currents
        )

    coil_count = int(currents.shape[0])
    padded_gammas, padded_gammadashs, padded_currents = _pad_coil_axis_to_device_count(
        gammas,
        gammadashs,
        currents,
        config.coil_device_count,
    )

    def _collective_forward(group_gammas, group_gammadashs, group_currents):
        return _collective_group_field(
            compute_points,
            group_gammas,
            group_gammadashs,
            group_currents,
            biot_savart_B,
            config,
        )

    _, pullback = jax.vjp(
        _collective_forward,
        padded_gammas,
        padded_gammadashs,
        padded_currents,
    )
    return _tree_trim_axis0(
        pullback(compute_v),
        coil_count,
    )


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
            current_terms.append(
                _as_runtime_float64(scale, reference=term_dofs) * term_dofs[0]
            )
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
        owner_dofs = _as_compute_array(owner_dofs, reference=owner_dofs)
    else:
        owner_dofs = _as_runtime_float64(owner_dofs, reference=owner_dofs)
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
