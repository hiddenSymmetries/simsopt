"""JAX field evaluation for native Python objectives.

The mutable Optimizable adapter projects native coil graphs into immutable
specs consumed by pure JAX geometry and Biot-Savart kernels. It does not
implement the simsoptpp.MagneticField interface.
"""

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, NoReturn, cast

import jax
from simsopt_jax.backend import get_field_kernel_tuning
import jax.numpy as jnp
import numpy as np

from simsopt._core.derivative import Derivative, OptimizableDefaultDict
from simsopt._core.json import GSONDecoder
from simsopt.field.coil import Current, CurrentSum, ScaledCurrent
from simsopt.field.magneticfield import MagneticField
from simsopt.geo.curveperturbed import CurvePerturbed
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.runtime.host_boundary import host_array
from simsopt._core.optimizable import Optimizable
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.core.state_tokens import make_state_token_factory
from simsopt_jax.backend.dtypes import (
    runtime_device_put,
    runtime_device_put_tree,
)
from simsopt_jax.core import (
    coil_specs_from_dof_extraction_spec,
    curve_pullback_from_dofs,
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_optimizable_dof_map_spec,
)
from simsopt_jax.core.curve_xyz_fourier import jaxfouriercurve_geometry_pure
from simsopt_jax.core.biotsavart import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_d2A_by_dXdX,
    biot_savart_d2B_by_dXdX,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
)
from simsopt_jax.core.field import (
    biot_savart_B_vjp_maybe_collective,
    grouped_biot_savart_A_from_inputs,
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_d2A_by_dXdX_from_spec,
    grouped_biot_savart_d2B_by_dXdX_from_spec,
    grouped_biot_savart_dA_by_dX_from_inputs,
    grouped_biot_savart_dA_by_dX_from_spec,
    grouped_biot_savart_dB_by_dX_from_inputs,
    grouped_biot_savart_dB_by_dX_from_spec,
    grouped_coil_set_spec_from_coil_specs,
    grouped_coil_set_spec_from_lists,
    grouped_field_inputs_from_spec,
)
from simsopt_jax.core._math_utils import (
    as_compute_array as _as_compute_array,
    as_jax_float64 as _as_jax_float64,
)
from simsopt_jax.core._device_scalars import two_pi as _two_pi
from simsopt_jax.core.specs import (
    make_field_eval_spec,
)
from simsopt_jax_adapters.field._coil_graph import (
    _unwrap_coil_curve_and_current_objects,
)
from simsopt_jax_adapters.geo.curve_contract import _optimizable_dof_layout
from simsopt_jax_adapters.geo.curve_specs import (
    adapter_curve_dof_mode,
    curve_spec_from_adapter_curve,
    supports_adapter_curve_spec,
)

_new_coil_dof_state_token = make_state_token_factory()
def _place_array_tree_on_device(tree, device):
    """Place dynamic array leaves on one device while preserving static metadata."""
    return runtime_device_put_tree(tree, device=device)


__all__ = [
    "BiotSavartJAX",
    "BiotSavartFieldPullback",
]


@jax.jit
def _cyl_points_to_cart(points_cyl):
    points = _as_jax_float64(points_cyl)
    r = points[:, 0]
    phi = points[:, 1]
    z = points[:, 2]
    return jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=1)


@jax.jit
def _canonical_set_points_cyl(points_cyl):
    points = _as_jax_float64(points_cyl)
    return points.at[:, 1].set(jnp.fmod(points[:, 1], _two_pi(points)))


@jax.jit
def _cart_points_to_cyl(points_cart):
    points = _as_jax_float64(points_cart)
    x = points[:, 0]
    y = points[:, 1]
    phi = jnp.arctan2(y, x)
    zero = jnp.sum(points - points)
    phi = jnp.where(phi < zero, phi + _two_pi(points), phi)
    return jnp.stack(
        (
            jnp.sqrt(x * x + y * y),
            phi,
            points[:, 2],
        ),
        axis=1,
    )


def _points_cyl_for_basis(points_cart, points_cyl):
    if points_cyl is not None:
        return points_cyl
    return _cart_points_to_cyl(points_cart)


@jax.jit
def _cart_vectors_to_cyl(vectors_cart, points_cyl):
    vectors = _as_jax_float64(vectors_cart)
    points = _as_jax_float64(points_cyl)
    phi = points[:, 1]
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    return jnp.stack(
        (
            cos_phi * vectors[:, 0] + sin_phi * vectors[:, 1],
            -sin_phi * vectors[:, 0] + cos_phi * vectors[:, 1],
            vectors[:, 2],
        ),
        axis=1,
    )


@jax.jit
def _grad_absB_from_B_and_dB(B, dB_by_dX):
    B_jax = _as_jax_float64(B)
    dB_jax = _as_jax_float64(dB_by_dX)
    absB = jnp.linalg.norm(B_jax, axis=1)
    return jnp.sum(dB_jax * B_jax[:, None, :], axis=2) / absB[:, None]


def _per_coil_unit_field_with_batch_size(points, coil_set_spec, kernel, *, batch_size):
    """Per-coil unit-current field as a list of JAX arrays.

    For each coil ``k`` in the public coil ordering, evaluates ``kernel``
    on the single-coil view with ``currents = [1.0]``. Biot-Savart is
    exactly linear in ``I``, so the resulting array equals
    ``∂F/∂I_k`` for the matching spatial-derivative kernel.

    The output is a list of ``ncoils`` separate JAX arrays — one per
    coil — so coil-axis collective reduction does not apply. The
    ``SIMSOPT_JAX_SHARDING=coil_groups`` collective path (used by
    ``grouped_biot_savart_*_from_spec``) is bypassed here by design,
    relying instead on the JAX kernel cache for compile-time reuse
    within a quadrature group.
    """
    compute_points = _as_compute_array(points)
    ncoils = sum(len(group.coil_indices) for group in coil_set_spec.groups)
    result_by_index: dict[int, jax.Array] = {}
    for group in coil_set_spec.groups:
        compute_gammas = _as_compute_array(group.gammas)
        compute_gammadashs = _as_compute_array(group.gammadashs)
        unit_current = _as_compute_array(jnp.ones((1,), dtype=group.currents.dtype))

        def evaluate_single(coil_geometry):
            gamma, gammadash = coil_geometry
            return kernel(
                compute_points,
                gamma[jnp.newaxis, ...],
                gammadash[jnp.newaxis, ...],
                unit_current,
            )

        if batch_size <= 0:
            group_results = jax.vmap(
                lambda gamma, gammadash: evaluate_single((gamma, gammadash))
            )(compute_gammas, compute_gammadashs)
        else:
            group_results = jax.lax.map(
                evaluate_single,
                (compute_gammas, compute_gammadashs),
                batch_size=batch_size,
            )
        for position, coil_index in enumerate(group.coil_indices):
            result_by_index[int(coil_index)] = group_results[position]
    return [result_by_index[index] for index in range(ncoils)]


def _per_coil_unit_field(points, coil_set_spec, kernel):
    return _per_coil_unit_field_with_batch_size(
        points,
        coil_set_spec,
        kernel,
        batch_size=get_field_kernel_tuning().coil_chunk_size,
    )


@pytree_dataclass(data=("d_coil_arrays",), meta=("coil_indices",))
@dataclass(frozen=True)
class BiotSavartFieldPullback:
    """Native grouped cotangent payload for ``BiotSavartJAX`` fields.

    ``d_coil_arrays`` mirrors the grouped field-input structure:
    one ``(d_gammas, d_gammadashs, d_currents)`` tuple per quadrature group.
    ``coil_indices`` maps each group row back to the public coil list.
    """

    d_coil_arrays: tuple[tuple[jax.Array, jax.Array, jax.Array], ...]
    coil_indices: tuple[tuple[int, ...], ...]


def _set_biot_savart_points(field, points):
    # Host inputs are mutable; device placement can alias their storage on CPU.
    if not isinstance(points, jax.Array):
        points = np.array(points, copy=True, order="C")
    field._points_jax = _as_jax_float64(points)
    field._points_cyl_jax = None
    field._points_version += 1
    return field


def _set_biot_savart_points_cyl(field, points_cyl):
    field._points_cyl_jax = _canonical_set_points_cyl(_as_jax_float64(points_cyl))
    field._points_jax = _cyl_points_to_cart(field._points_cyl_jax)
    field._points_version += 1
    return field


def _get_biot_savart_points_cyl(field):
    if field._points_cyl_jax is not None:
        return host_array(field._points_cyl_jax, dtype=np.float64)
    return host_array(_cart_points_to_cyl(field._points_jax), dtype=np.float64)


def _supports_native_curve_geometry(curve):
    return supports_adapter_curve_spec(curve)


def _require_native_curve_geometry(curve):
    if not _supports_native_curve_geometry(curve):
        raise TypeError(
            "BiotSavartJAX coil cotangent projection requires immutable JAX "
            f"curve specs; unsupported type {type(curve).__name__}. "
            "Provide a native curve spec."
        )


def _curve_dof_mode(curve):
    return adapter_curve_dof_mode(curve)


def _curve_quadpoints_jax(curve):
    return _as_jax_float64(curve.quadpoints)


def _slice_1d(array: jax.Array, start: int, end: int) -> jax.Array:
    return jax.lax.slice_in_dim(array, int(start), int(end), axis=0)


def _update_1d(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    start = int(start)
    stop = start + int(values.shape[0])
    return jnp.concatenate(
        (
            _slice_1d(array, 0, start),
            values,
            _slice_1d(array, stop, int(array.shape[0])),
        )
    )


def _add_update_1d(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    start = int(start)
    stop = start + int(values.shape[0])
    return _update_1d(array, start, _slice_1d(array, start, stop) + values)


def _take_positions_1d(array: jax.Array, positions) -> jax.Array:
    indexer = runtime_device_put(positions, dtype=np.int32)
    return jnp.take(_as_jax_float64(array), indexer, axis=0)


def _owner_segments_from_free_positions(
    owner_start: int,
    free_positions,
) -> tuple[tuple[int, int, int, int], ...]:
    """Copy ranges ``owner[o0:o1] -> target[t0:t1]`` for one block of free dofs.

    ``free_positions[i]`` is where owner dof ``owner_start + i`` lands in the
    target's local full vector; each run of consecutive positions is one range.
    """
    free_positions = np.asarray(free_positions, dtype=np.int64)
    run_breaks = np.flatnonzero(np.diff(free_positions) != 1) + 1
    run_bounds = np.concatenate(([0], run_breaks, [free_positions.size]))
    return tuple(
        (
            int(owner_start + first),
            int(owner_start + last),
            int(free_positions[first]),
            int(free_positions[last - 1]) + 1,
        )
        for first, last in zip(run_bounds[:-1], run_bounds[1:])
        if last > first
    )


def _scatter_free_values(template: jax.Array, free_positions, free_values: jax.Array):
    free_positions = np.asarray(free_positions, dtype=np.int64)
    if np.array_equal(free_positions, np.arange(int(template.shape[0]))):
        return free_values
    mask = np.ones(int(template.shape[0]), dtype=np.float64)
    mask[free_positions] = 0.0
    indexer = runtime_device_put(free_positions[:, None], dtype=np.int32)
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return jax.lax.scatter(
        template * runtime_device_put(mask, dtype=np.float64),
        indexer,
        free_values,
        dnums,
        indices_are_sorted=True,
        unique_indices=True,
    )


def _dof_map_cotangent_to_owner_gradient(map_spec, input_cotangent, owner_dofs):
    owner_dofs = _as_jax_float64(owner_dofs)
    input_cotangent = _as_jax_float64(input_cotangent)
    if map_spec.input_mode == "full":
        full_cotangent = input_cotangent
    else:
        zero = jnp.sum(input_cotangent, dtype=input_cotangent.dtype)
        full_cotangent = jnp.broadcast_to(
            zero - zero,
            map_spec.template_full_dofs.shape,
        )
        full_cotangent = _add_update_1d(
            full_cotangent,
            map_spec.input_start,
            input_cotangent,
        )

    owner_gradient = owner_dofs - owner_dofs
    for owner_start, _owner_end, target_start, target_end in map_spec.owner_segments:
        owner_gradient = _add_update_1d(
            owner_gradient,
            owner_start,
            _slice_1d(full_cotangent, target_start, target_end),
        )
    return owner_gradient


def _add_extraction_cotangent_to_dofs_gradient(
    dofs_gradient,
    extraction_spec,
    coil_spec,
    dg,
    dgd,
    dc,
    coil_dofs,
):
    if extraction_spec.symmetry.has_rotation:
        rotmat_t = _as_jax_float64(extraction_spec.symmetry.rotmat).T
        dg = _as_jax_float64(dg) @ rotmat_t
        dgd = _as_jax_float64(dgd) @ rotmat_t

    coeff_cotangent = curve_pullback_from_dofs(
        coil_spec.curve,
        coil_spec.curve.dofs,
        dg,
        dgd,
    )
    current_cotangent = jnp.atleast_1d(
        _as_jax_float64(extraction_spec.symmetry.scale) * _as_jax_float64(dc)
    )
    dofs_gradient = dofs_gradient + _dof_map_cotangent_to_owner_gradient(
        extraction_spec.curve_map,
        coeff_cotangent,
        coil_dofs,
    )
    current_maps = extraction_spec.current_term_maps or (extraction_spec.current_map,)
    current_scales = extraction_spec.current_term_scales or (1.0,)
    for current_map, scale in zip(current_maps, current_scales, strict=True):
        dofs_gradient = dofs_gradient + _dof_map_cotangent_to_owner_gradient(
            current_map,
            _as_jax_float64(scale) * current_cotangent,
            coil_dofs,
        )
    return dofs_gradient


def _coil_cotangents_to_dofs_gradient_from_extraction_spec(
    coil_dof_extraction_spec,
    d_coil_arrays,
    coil_indices,
    coil_dofs,
    *,
    projection_spec=None,
    owner_width=None,
):
    coil_dofs = _as_jax_float64(coil_dofs)
    dofs_gradient = (
        coil_dofs - coil_dofs
        if owner_width is None
        else jnp.zeros((owner_width,), dtype=coil_dofs.dtype)
    )
    owner_template = dofs_gradient
    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    extraction_specs = (
        coil_dof_extraction_spec if projection_spec is None else projection_spec
    ).coils
    for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
        for local_i, global_i in enumerate(indices):
            dofs_gradient = _add_extraction_cotangent_to_dofs_gradient(
                dofs_gradient,
                extraction_specs[global_i],
                coil_specs[global_i],
                jax.lax.index_in_dim(d_g, local_i, axis=0, keepdims=False),
                jax.lax.index_in_dim(d_gd, local_i, axis=0, keepdims=False),
                jax.lax.index_in_dim(d_c, local_i, axis=0, keepdims=False),
                owner_template,
            )
    return dofs_gradient


@partial(jax.jit, static_argnames=("coil_indices", "owner_width"))
def _jitted_coil_cotangents_to_owner_partials(
    extraction_spec, projection_spec, d_coil_arrays, coil_indices, coil_dofs,
    owner_width,
):
    return _coil_cotangents_to_dofs_gradient_from_extraction_spec(
        extraction_spec, d_coil_arrays, coil_indices, coil_dofs,
        projection_spec=projection_spec, owner_width=owner_width,
    )


@partial(jax.jit, static_argnames=("coil_indices",))
def _jitted_coil_cotangents_to_dofs_gradient(
    coil_dof_extraction_spec,
    d_coil_arrays,
    coil_indices,
    coil_dofs,
):
    return _coil_cotangents_to_dofs_gradient_from_extraction_spec(
        coil_dof_extraction_spec,
        d_coil_arrays,
        coil_indices,
        coil_dofs,
    )


def _canonical_coil_indices(coil_indices):
    return tuple(tuple(int(index) for index in indices) for indices in coil_indices)


def _coil_cotangent_arrays_are_jax_compatible(d_coil_arrays):
    leaves = jax.tree.leaves(d_coil_arrays)
    return all(
        isinstance(leaf, (jax.Array, np.ndarray)) or hasattr(leaf, "aval")
        for leaf in leaves
    )


def _add_local_cotangent_to_dofs_gradient(
    dofs_gradient: jax.Array,
    opt,
    full_cotangent,
    dof_indices,
    *,
    free_positions,
):
    if opt.local_dof_size == 0:
        return dofs_gradient
    start, end = dof_indices[opt]
    free_cotangent = _take_positions_1d(full_cotangent, free_positions)
    return _add_update_1d(dofs_gradient, start, free_cotangent)


def _add_full_curve_cotangent_to_dofs_gradient(
    dofs_gradient: jax.Array,
    curve,
    full_cotangent,
    dof_indices,
    *,
    free_positions_for_opt,
):
    full_cotangent = _as_jax_float64(full_cotangent)
    for opt, (start, end) in curve._full_dof_indices.items():
        dofs_gradient = _add_local_cotangent_to_dofs_gradient(
            dofs_gradient,
            opt,
            _slice_1d(full_cotangent, start, end),
            dof_indices,
            free_positions=free_positions_for_opt(opt),
        )
    return dofs_gradient


def _unwrap_coil_curve_and_current(coil):
    curve, rotmat, current, scale = _unwrap_coil_curve_and_current_objects(
        coil.curve,
        coil.current,
    )
    return (
        curve,
        (None if rotmat is None else _as_jax_float64(rotmat)),
        current,
        scale,
    )


def _affine_current_terms(current, coefficient=1.0):
    """Return scalar current owners and their forward/reverse coefficients."""
    if isinstance(current, Current):
        return ((current, coefficient),)
    if isinstance(current, ScaledCurrent):
        return _affine_current_terms(
            current.current_to_scale, coefficient * float(current.scale),
        )
    if isinstance(current, CurrentSum):
        return _affine_current_terms(
            current.current_a, coefficient,
        ) + _affine_current_terms(current.current_b, coefficient)
    raise NotImplementedError(
        "BiotSavartJAX only supports affine expressions of scalar "
        f"Current objects; got {type(current).__name__}."
    )


class BiotSavartJAX(Optimizable):
    """JAX Biot-Savart for Python objectives and their derivatives.

    Supports B, A, spatial derivatives and VJPs through the Optimizable coil
    graph, including native SquaredFlux, CurveLength and scipy minimizers.
    This is not a simsoptpp.MagneticField: tracing, InterpolatedField and
    native field arithmetic require simsopt.field.BiotSavart. Native compute,
    cache and export interfaces are not provided. Arithmetic on
    this adapter raises TypeError rather than constructing an objective. Native
    field sums and scaling wrappers also reject this adapter as a dependency.

    Supported curves are XYZ Fourier (including Fourier symmetries), RZ
    Fourier, planar Fourier and helical curves; rotated, perturbed and Frenet
    filament wrappers are supported through immutable curve specs. A custom
    curve may supply a compatible to_spec() method. Unsupported curves raise.

    Before constructing the adapter, call simsopt_jax.backend.set_backend
    with the desired device and intent to apply debug, JIT, transfer-guard,
    dtype, chunking and cache settings. Environment variables are resolved by
    that call; importing or constructing the adapter does not apply
    process-global JAX settings. Choose the device before JAX initializes it.

    Coil extraction and kernels use immutable arrays. This Optimizable wrapper
    owns mutable point/cache state and is confined to one evaluation thread.

    Args:
        coils: native simsopt.field.Coil objects.
    """

    def clear_points(self) -> None:
        """Clear mutable point buffers without changing source geometry."""
        self._points_jax = None
        self._points_cyl_jax = None
        self._points_version += 1

    def _per_coil_unit_current_derivative(self, kernel):
        """Evaluate a unit-current derivative kernel for this field state."""
        return _per_coil_unit_field(
            self._points_jax,
            self.coil_set_spec(),
            kernel,
        )

    def B(self):
        """Magnetic field B at the evaluation points."""
        return grouped_biot_savart_B_from_spec(self._points_jax, self.coil_set_spec())

    def A(self):
        """Vector potential A at the evaluation points."""
        return grouped_biot_savart_A_from_spec(self._points_jax, self.coil_set_spec())

    def dA_by_dX(self):
        """Spatial Jacobian dA/dX at the evaluation points."""
        return grouped_biot_savart_dA_by_dX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def d2A_by_dXdX(self):
        """Spatial Hessian d2A/dXdX at the evaluation points."""
        return grouped_biot_savart_d2A_by_dXdX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def dB_by_dX(self):
        """Spatial Jacobian dB/dX at the evaluation points."""
        return grouped_biot_savart_dB_by_dX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def d2B_by_dXdX(self):
        """Spatial Hessian d2B/dXdX at the evaluation points."""
        return grouped_biot_savart_d2B_by_dXdX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def B_and_dB(self):
        """Combined B and dB/dX."""
        return grouped_biot_savart_B_and_dB_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def AbsB(self):
        """Magnetic-field magnitude at the evaluation points."""
        return jnp.linalg.norm(self.B(), axis=1)[:, None]

    def GradAbsB(self):
        """Cartesian gradient of ``|B|`` at the evaluation points."""
        return _grad_absB_from_B_and_dB(*self.B_and_dB())

    def B_cyl(self):
        """Magnetic field components in the cylindrical basis."""
        return _cart_vectors_to_cyl(
            self.B(),
            _points_cyl_for_basis(self._points_jax, self._points_cyl_jax),
        )

    def A_cyl(self):
        """Vector potential components in the cylindrical basis."""
        return _cart_vectors_to_cyl(
            self.A(),
            _points_cyl_for_basis(self._points_jax, self._points_cyl_jax),
        )

    def GradAbsB_cyl(self):
        """``GradAbsB`` components in the cylindrical basis."""
        return _cart_vectors_to_cyl(
            self.GradAbsB(),
            _points_cyl_for_basis(self._points_jax, self._points_cyl_jax),
        )

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        """Per-coil B at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_B)

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        """Per-coil ``dB/dX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_dB_by_dX)

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        """Per-coil ``d2B/dXdX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_d2B_by_dXdX)

    def dA_by_dcoilcurrents(self, compute_derivatives=0):
        """Per-coil A at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_A)

    def d2A_by_dXdcoilcurrents(self, compute_derivatives=1):
        """Per-coil ``dA/dX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_dA_by_dX)

    def d3A_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        """Per-coil ``d2A/dXdX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_d2A_by_dXdX)


    def _add_child(self, child: Optimizable) -> None:
        # Native sums/scalings also attach operands through the Optimizable graph.
        if isinstance(child, MagneticField):
            self._unsupported_field_arithmetic(child)
        super()._add_child(child)

    def _unsupported_field_arithmetic(self, other=None) -> NoReturn:
        raise TypeError(
            "BiotSavartJAX supports Python objectives, not native field arithmetic; "
            "use simsopt.field.BiotSavart for field sums and scaling."
        )

    __add__ = _unsupported_field_arithmetic
    __radd__ = _unsupported_field_arithmetic
    __mul__ = _unsupported_field_arithmetic
    __rmul__ = _unsupported_field_arithmetic

    def as_dict(self, serial_objs_dict=None) -> dict:
        serialized = super().as_dict(serial_objs_dict=serial_objs_dict)
        serialized["points"] = (
            None if self._points_jax is None else self.get_points_cart()
        )
        return serialized

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        coils = decoder.process_decoded(d["coils"], serial_objs_dict, recon_objs)
        field = cls(coils)
        points = decoder.process_decoded(d.get("points"), serial_objs_dict, recon_objs)
        if points is not None:
            field.set_points(points)
        return field

    def __init__(self, coils):
        self._coils = list(coils)
        self._points_jax = None
        self._points_cyl_jax = None
        self._points_version = 0
        self._dof_layout_version = 0
        self._coil_dofs_generation = 0
        self._coil_dof_state_token = _new_coil_dof_state_token()
        self._free_dof_layout_ready = False
        self._suppress_dependency_coil_dof_state = False
        self._local_free_positions_by_opt = {}
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=self._coils)

        # Uniform CurveXYZFourier fast-path metadata (populated by _introspect_coils)
        self._uses_uniform_curve_xyz_fourier_fastpath = False
        self._unique_base_curves = []
        self._unique_base_currents = []
        self._coil_descs = []  # list of (curve_idx, current_idx, rotmat_jax, scale)
        self._curve_order = 0
        self._curve_quadpoints_jax = None
        self._introspect_coils()
        self._free_dof_layout_ready = True
        self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()
        self._captured_coil_state_fingerprint = (
            self._current_captured_coil_state_fingerprint()
        )

    def update_free_dof_size_indices(self) -> None:
        super().update_free_dof_size_indices()
        self._local_free_positions_by_opt.clear()
        if self._free_dof_layout_ready:
            self._dof_layout_version += 1
            self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()
            self._captured_coil_state_fingerprint = (
                self._current_captured_coil_state_fingerprint()
            )

    def _current_captured_coil_state_fingerprint(self) -> tuple[bytes, ...]:
        """Fingerprint fixed coordinates on DOF notifications, never on reads."""
        return tuple(
            np.ascontiguousarray(
                np.asarray(opt.local_full_x, dtype=np.float64)[
                    ~np.asarray(opt.local_dofs_free_status, dtype=bool)
                ]
            ).tobytes()
            for opt in self.unique_dof_lineage
        )

    def _perturbation_samples_changed(self) -> bool:
        return any(
            curve.sample is not sample
            or sample._sample is not samples
            or any(current is not previous for current, previous in
                   zip(sample._sample, sample_arrays, strict=True))
            for curve, sample, samples, sample_arrays in self._captured_perturbation_samples
        )

    def _refresh_captured_coil_state(self, *, check_fixed=True) -> None:
        fingerprint = (
            self._current_captured_coil_state_fingerprint()
            if check_fixed else self._captured_coil_state_fingerprint
        )
        if (fingerprint == self._captured_coil_state_fingerprint
                and not self._perturbation_samples_changed()):
            return
        self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()
        self._captured_coil_state_fingerprint = fingerprint

    def _advance_coil_dof_state(self) -> None:
        self._coil_dofs_generation += 1
        self._coil_dof_state_token = _new_coil_dof_state_token()

    def set_recompute_flag(self, parent=None):
        if (
            parent is not None
            and self._free_dof_layout_ready
            and not self._suppress_dependency_coil_dof_state
        ):
            self._advance_coil_dof_state()
            self._refresh_captured_coil_state()
        super().set_recompute_flag(parent=parent)

    def _set_global_coil_dofs(
        self,
        optimizable_setter,
        coil_dofs,
        *,
        rebuild_extraction_spec,
    ):
        self._suppress_dependency_coil_dof_state = True
        try:
            optimizable_setter(self, coil_dofs)
        finally:
            self._suppress_dependency_coil_dof_state = False
        self._advance_coil_dof_state()
        if rebuild_extraction_spec:
            self._refresh_captured_coil_state()

    @property
    def x(self):
        return cast(Callable[[Optimizable], np.ndarray], Optimizable.x.fget)(self)

    @x.setter
    def x(self, coil_dofs):
        self._set_global_coil_dofs(
            Optimizable.x.fset,
            coil_dofs,
            rebuild_extraction_spec=False,
        )

    @property
    def full_x(self):
        return cast(Callable[[Optimizable], np.ndarray], Optimizable.full_x.fget)(self)

    @full_x.setter
    def full_x(self, coil_dofs):
        self._set_global_coil_dofs(
            Optimizable.full_x.fset,
            coil_dofs,
            rebuild_extraction_spec=True,
        )

    def _local_free_positions(self, opt):
        cached = self._local_free_positions_by_opt.get(opt)
        if cached is None:
            cached = np.flatnonzero(opt.local_dofs_free_status)
            self._local_free_positions_by_opt[opt] = cached
        return cached

    def _introspect_coils(self):
        """Walk coil tree to identify unique base curves/currents.

        Enables the JAX-native path when all curves are
        ``CurveXYZFourier`` (possibly wrapped in ``RotatedCurve``)
        with uniform Fourier order and quadrature point count.
        """
        base_curve_ids = {}  # id(obj) → index
        base_current_ids = {}
        base_curves = []
        base_currents = []
        descs = []

        for coil in self._coils:
            curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)

            if not isinstance(curve, CurveXYZFourier):
                return

            cid = id(curve)
            if cid not in base_curve_ids:
                base_curve_ids[cid] = len(base_curves)
                base_curves.append(curve)

            # Must resolve to a single-DOF Current (not CurrentSum etc.)
            if not isinstance(current, Current):
                return

            kid = id(current)
            if kid not in base_current_ids:
                base_current_ids[kid] = len(base_currents)
                base_currents.append(current)

            descs.append(
                (
                    base_curve_ids[cid],
                    base_current_ids[kid],
                    _as_jax_float64(rotmat) if rotmat is not None else None,
                    scale,
                )
            )

        # All curves must share the same Fourier order and quadrature grid
        orders = {c.order for c in base_curves}
        if len(orders) != 1:
            return
        ref_qp = np.asarray(base_curves[0].quadpoints)
        for c in base_curves[1:]:
            if not np.array_equal(ref_qp, np.asarray(c.quadpoints)):
                return

        self._uses_uniform_curve_xyz_fourier_fastpath = True
        self._unique_base_curves = base_curves
        self._unique_base_currents = base_currents
        self._coil_descs = descs
        self._curve_order = orders.pop()
        self._curve_quadpoints_jax = _curve_quadpoints_jax(base_curves[0])

    def _build_coil_dof_extraction_spec(self):
        curve_source_ids = {}
        # Reconstruction is keyed by shared DOFs; partials retain actual owners.
        free_slices_by_dofs = {
            opt.dofs: self.dof_indices[opt] for opt in self.unique_dof_lineage
            if int(opt.local_dof_size) > 0
        }
        self._coil_dof_indices = {
            opt: free_slices_by_dofs[opt.dofs] for opt in self.ancestors
            if int(opt.local_dof_size) > 0
        }

        def coil_extraction_spec(coil):
            curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
            curve_id = id(curve)
            if curve_id not in curve_source_ids:
                curve_source_ids[curve_id] = len(curve_source_ids)
            current_terms = (
                () if isinstance(current, Current) else _affine_current_terms(current)
            )
            return make_coil_dof_extraction_spec(
                curve=curve_spec_from_adapter_curve(curve),
                curve_map=self._free_vector_dof_map_spec(
                    curve,
                    full_graph=_curve_dof_mode(curve) == "full",
                ),
                current_map=self._free_vector_dof_map_spec(
                    current,
                    full_graph=False,
                ),
                current_term_maps=tuple(
                    self._free_vector_dof_map_spec(term, full_graph=False)
                    for term, _coefficient in current_terms
                ),
                current_term_scales=tuple(
                    coefficient for _term, coefficient in current_terms
                ),
                curve_source_index=curve_source_ids[curve_id],
                rotmat=rotmat,
                scale=scale,
            )

        extraction_spec = make_coil_set_dof_extraction_spec(
            coil_extraction_spec(coil) for coil in self._coils
        )
        self._captured_perturbation_samples = tuple(
            (opt, opt.sample, opt.sample._sample, tuple(opt.sample._sample))
            for opt in self.ancestors if isinstance(opt, CurvePerturbed)
        )
        self._build_owner_partial_projection_spec(extraction_spec)
        return extraction_spec

    def _build_owner_partial_projection_spec(self, extraction_spec):
        """Cache full partial destinations by actual owner, including fixed DOFs.

        Reconstruction uses the free-vector maps; projection uses full local
        owner slices. Shared DOFs retain separate native Derivative owner keys.
        """
        owner_slices = {}
        width = 0

        def projection_map(opt, source_map, *, full_graph):
            nonlocal width
            owners = (
                _optimizable_dof_layout(opt, separate_owners=True)[1].items() if full_graph
                else ((opt, (0, opt.local_full_dof_size)),)
            )
            segments = []
            for owner, (start, end) in owners:
                if owner.local_full_dof_size == 0:
                    continue
                if owner not in owner_slices:
                    owner_slices[owner] = (width, width + owner.local_full_dof_size)
                    width += owner.local_full_dof_size
                owner_start, owner_end = owner_slices[owner]
                segments.append((owner_start, owner_end, start, end))
            return replace(source_map, owner_segments=tuple(segments))

        owner_extraction_coils = []
        projection_coils = []
        for coil, spec in zip(self._coils, extraction_spec.coils, strict=True):
            curve, _rotation, current, _scale = _unwrap_coil_curve_and_current(coil)
            full_graph = _curve_dof_mode(curve) == "full"
            owner_spec = (
                replace(
                    spec,
                    curve=curve_spec_from_adapter_curve(curve, separate_owners=True),
                    curve_map=self._free_vector_dof_map_spec(
                        curve, full_graph=True, separate_owners=True,
                    ),
                ) if full_graph else spec
            )
            owner_extraction_coils.append(owner_spec)
            terms = _affine_current_terms(current)
            term_maps = spec.current_term_maps or (spec.current_map,)
            projection_coils.append(replace(
                owner_spec,
                curve_map=projection_map(
                    curve, owner_spec.curve_map, full_graph=full_graph,
                ),
                current_term_maps=tuple(
                    projection_map(term, source_map, full_graph=False)
                    for (term, _coefficient), source_map in zip(terms, term_maps, strict=True)
                ),
                current_term_scales=tuple(coefficient for _term, coefficient in terms),
            ))
        self._owner_partial_extraction_spec = make_coil_set_dof_extraction_spec(owner_extraction_coils)
        self._owner_partial_projection_spec = make_coil_set_dof_extraction_spec(projection_coils)
        self._owner_partial_slices = tuple(owner_slices.items())
        self._owner_partial_width = width
        self._device_projection_contracts = {}

    def coil_dof_extraction_spec(self):
        """Return the cached immutable owner-DOF reconstruction contract."""
        # Direct sample replacement can bypass the native curve notification.
        previous_spec = self._coil_dof_extraction_spec
        self._refresh_captured_coil_state(check_fixed=False)
        if self._coil_dof_extraction_spec is not previous_spec:
            self._advance_coil_dof_state()
            self.set_recompute_flag()
        return self._coil_dof_extraction_spec

    @property
    def dof_layout_version(self) -> int:
        """Return the monotonic free/fixed DOF-layout version."""
        return self._dof_layout_version

    def _local_full_dofs_from_free_vector(self, opt, coil_dofs):
        """Rebuild one Optimizable's full local DOF vector from ``coil_dofs``.

        ``Optimizable.x`` is ordered by unique ancestor name, not by the
        JAX-native coil grouping used below. Reconstruct each curve/current
        block from its own free-DOF slice so mixed free-current / free-curve
        graphs decode correctly.
        """
        full_x = _as_jax_float64(opt.local_full_x)
        if opt.local_dof_size == 0:
            return full_x

        start, end = self._coil_dof_indices[opt]
        free_positions = self._local_free_positions(opt)
        coil_slice = _slice_1d(coil_dofs, start, end)
        return _scatter_free_values(full_x, free_positions, coil_slice)

    def _full_dofs_from_free_vector(self, opt, coil_dofs):
        """Rebuild one Optimizable graph's full DOF vector from ``coil_dofs``."""
        full_x = _as_jax_float64(opt.full_x)
        for dep_opt, (start, end) in opt._full_dof_indices.items():
            dep_full_x = _as_jax_float64(dep_opt.local_full_x)
            if dep_opt.local_dof_size > 0:
                dep_start, dep_end = self._coil_dof_indices[dep_opt]
                free_positions = self._local_free_positions(dep_opt)
                dep_slice = _slice_1d(coil_dofs, dep_start, dep_end)
                dep_full_x = _scatter_free_values(
                    dep_full_x,
                    free_positions,
                    dep_slice,
                )
            full_x = _update_1d(full_x, start, dep_full_x)
        return full_x

    def _curve_dofs_from_free_vector(self, curve, coil_dofs):
        if _curve_dof_mode(curve) == "full":
            return self._full_dofs_from_free_vector(curve, coil_dofs)
        return self._local_full_dofs_from_free_vector(curve, coil_dofs)

    def _free_vector_dof_map_spec(self, opt, *, full_graph, separate_owners=False):
        if full_graph:
            full_dofs, full_indices = _optimizable_dof_layout(opt, separate_owners=separate_owners)
            owner_segments = tuple(
                (
                    owner_start,
                    owner_end,
                    int(target_start + local_start),
                    int(target_start + local_end),
                )
                for dep_opt, (target_start, _target_end) in full_indices.items()
                if int(dep_opt.local_dof_size) > 0
                for owner_start, owner_end, local_start, local_end in
                _owner_segments_from_free_positions(
                    self._coil_dof_indices[dep_opt][0], self._local_free_positions(dep_opt),
                )
            )
            template_full_dofs = _as_jax_float64(full_dofs)
            return self._full_input_dof_map_spec(template_full_dofs, owner_segments)

        template_full_dofs = _as_jax_float64(opt.local_full_x)
        if opt.local_dof_size == 0:
            return self._full_input_dof_map_spec(template_full_dofs, ())

        owner_start, _owner_end = self._coil_dof_indices[opt]
        owner_segments = _owner_segments_from_free_positions(
            owner_start,
            self._local_free_positions(opt),
        )
        return self._full_input_dof_map_spec(template_full_dofs, owner_segments)

    def _full_input_dof_map_spec(self, template_full_dofs, owner_segments):
        return make_optimizable_dof_map_spec(
            template_full_dofs=template_full_dofs,
            owner_segments=owner_segments,
            input_mode="full",
            input_start=0,
            input_end=int(template_full_dofs.shape[0]),
        )

    def _normalize_explicit_coil_dofs(self, coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        expected_dofs = self.dof_size
        if coil_dofs.shape[0] != expected_dofs:
            raise ValueError(
                f"Expected {expected_dofs} coil DOFs, got {coil_dofs.shape[0]}."
            )
        return coil_dofs

    def coil_specs_from_dofs(self, coil_dofs):
        """Build immutable per-coil specs from an explicit flat DOF vector."""
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        return coil_specs_from_dof_extraction_spec(
            self.coil_dof_extraction_spec(),
            coil_dofs,
        )

    def _coil_set_spec_from_dofs_immutable_specs(self, coil_dofs):
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        return grouped_coil_set_spec_from_coil_specs(
            self.coil_specs_from_dofs(coil_dofs),
        )

    def _scalar_current_value_from_dofs(self, current, coil_dofs, lane_label):
        current_full_x = self._local_full_dofs_from_free_vector(current, coil_dofs)
        if current_full_x.shape[0] != 1:
            raise RuntimeError(
                "grouped_coil_arrays_from_dofs() only supports scalar Current "
                f"degrees of freedom on the {lane_label}."
            )
        return current_full_x[0]

    def _coil_arrays_in_order_from_dofs(self, coil_dofs):
        """Build per-coil ``(gamma, gammadash, current)`` arrays from DOFs.

        This is the pure-array counterpart to reading geometry from the live
        ``Optimizable`` graph: it reconstructs coil data from the explicit
        flat ``coil_dofs`` vector without assigning ``self.x``.

        Used only for uniform ``CurveXYZFourier`` coils; other curve families
        reconstruct their immutable coil specs in ``_coil_set_spec_from_explicit_state``.
        """
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)

        quadpoints = self._curve_quadpoints_jax

        curve_dofs = []
        for curve in self._unique_base_curves:
            curve_dofs.append(self._local_full_dofs_from_free_vector(curve, coil_dofs))

        current_values = []
        for current in self._unique_base_currents:
            current_values.append(
                self._scalar_current_value_from_dofs(
                    current,
                    coil_dofs,
                    "JAX-native lane",
                )
            )

        base_gammas = []
        base_gammadashs = []
        for curve_x in curve_dofs:
            gamma, gammadash, _, _ = jaxfouriercurve_geometry_pure(
                curve_x,
                quadpoints,
                self._curve_order,
            )
            base_gammas.append(gamma)
            base_gammadashs.append(gammadash)

        coil_gammas = []
        coil_gammadashs = []
        coil_currents = []
        for curve_idx, current_idx, rotmat, scale in self._coil_descs:
            gamma = base_gammas[curve_idx]
            gammadash = base_gammadashs[curve_idx]
            if rotmat is not None:
                gamma = gamma @ rotmat
                gammadash = gammadash @ rotmat
            coil_gammas.append(gamma)
            coil_gammadashs.append(gammadash)
            coil_currents.append(_as_jax_float64(scale) * current_values[current_idx])

        return coil_gammas, coil_gammadashs, coil_currents

    def grouped_coil_arrays_from_dofs(self, coil_dofs):
        """Build grouped coil arrays from an explicit flat DOF vector."""
        return list(
            grouped_field_inputs_from_spec(self.coil_set_spec_from_dofs(coil_dofs))
        )

    def coil_set_spec_from_dofs(self, coil_dofs):
        """Build an immutable grouped coil spec from an explicit flat DOF vector."""
        return self._coil_set_spec_from_dofs_immutable_specs(coil_dofs)

    @property
    def coils(self):
        return self._coils

    def set_points(self, points):
        """Set evaluation points (converted to a JAX array once).

        Accepts both NumPy and JAX arrays.  JAX arrays stay on device
        without a host round-trip. Mutates the cached point buffer on this
        instance, so callers should not share one ``BiotSavartJAX`` across
        concurrent evaluation threads.
        """
        return _set_biot_savart_points(self, points)

    def set_points_cart(self, points):
        return self.set_points(points)

    def set_points_cyl(self, points_cyl):
        return _set_biot_savart_points_cyl(self, points_cyl)

    def get_points_cart_ref(self):
        """Return the current JAX point buffer for point-preserving callers."""
        return self._points_jax

    def get_points_cart(self):
        return host_array(self._points_jax, dtype=np.float64)

    def get_points_cyl(self):
        return _get_biot_savart_points_cyl(self)

    def set_points_from_spec(self, field_eval_spec):
        """Set evaluation points from an immutable field-evaluation spec.

        This still mutates the receiving ``BiotSavartJAX`` instance.
        """
        return _set_biot_savart_points(self, field_eval_spec.points)

    def field_eval_spec(self):
        """Build the immutable field-evaluation spec for the current points."""
        return make_field_eval_spec(self._points_jax)


    def _coil_set_spec_from_explicit_state(self):
        if self._uses_uniform_curve_xyz_fourier_fastpath:
            return grouped_coil_set_spec_from_lists(
                *self._coil_arrays_in_order_from_dofs(_as_jax_float64(self.x))
            )
        return self.coil_set_spec_from_dofs(_as_jax_float64(self.x))


    def coil_set_spec(self):
        """Build the grouped coil spec for the current coil graph.

        The path stays in immutable-spec space: reconstruct from the live
        free-DOF vector with the cached explicit grouped-spec contract.
        """
        return self._coil_set_spec_from_explicit_state()

    def coil_specs(self):
        """Build immutable per-coil specs from the live coil graph."""
        return self.coil_specs_from_dofs(_as_jax_float64(self.x))

    # ------------------------------------------------------------------
    # VJP (reverse-mode gradient w.r.t. coil DOFs)
    # ------------------------------------------------------------------


    def B_pullback_native(self, v):
        r"""Return the native grouped cotangents for ``B``.

        This is the JAX-native pullback boundary. It returns cotangents with
        respect to grouped coil geometry/current arrays, without projecting
        them into SIMSOPT's public :class:`Derivative` object graph.
        """
        points = self._points_jax
        v_jax = _as_jax_float64(v)
        coil_set_spec = self._coil_set_spec_from_explicit_state()
        d_coil_arrays = tuple(
            biot_savart_B_vjp_maybe_collective(
                points,
                v_jax,
                group.gammas,
                group.gammadashs,
                group.currents,
            )
            for group in coil_set_spec.groups
        )
        return BiotSavartFieldPullback(
            d_coil_arrays=d_coil_arrays,
            coil_indices=coil_set_spec.coil_index_lists(),
        )

    def _pullback_to_derivative(self, pullback) -> Derivative:
        return self.coil_cotangents_to_derivative(
            pullback.d_coil_arrays,
            pullback.coil_indices,
        )

    def B_vjp(self, v) -> Derivative:
        r"""Vector-Jacobian product of B w.r.t. coil DOFs.

        Given a cotangent vector ``v`` (typically ``dJ/dB``), returns
        a :class:`Derivative` mapping every coil DOF, including fixed DOFs, to its
        contribution to the scalar objective.

        Uses ``jax.vjp`` through the pure Biot-Savart kernel, then
        projects each coil's geometry/current cotangents through immutable
        curve specs. Unsupported curves are rejected explicitly.

        Args:
            v: (npoints, 3) cotangent, same shape as ``B()``.

        Returns:
            :class:`Derivative` (sum over all coils).
        """
        return self._pullback_to_derivative(self.B_pullback_native(v))

    def _field_pullback_native(
        self,
        grouped_forward,
        cotangent,
    ):
        coil_set_spec = self._coil_set_spec_from_explicit_state()
        coil_arrays = coil_set_spec.field_inputs()
        if not coil_arrays:
            return BiotSavartFieldPullback((), ())

        _, pullback = jax.vjp(
            lambda grouped_inputs: grouped_forward(self._points_jax, grouped_inputs),
            coil_arrays,
        )
        d_coil_arrays = pullback(_as_jax_float64(cotangent))[0]
        return BiotSavartFieldPullback(
            d_coil_arrays=tuple(d_coil_arrays),
            coil_indices=coil_set_spec.coil_index_lists(),
        )

    def A_pullback_native(self, v):
        r"""Return native grouped cotangents for ``A``."""
        return self._field_pullback_native(grouped_biot_savart_A_from_inputs, v)

    def dA_by_dX_pullback_native(self, vgrad):
        r"""Return native grouped cotangents for ``dA/dX``."""
        return self._field_pullback_native(
            grouped_biot_savart_dA_by_dX_from_inputs,
            vgrad,
        )

    def dB_by_dX_pullback_native(self, vgrad):
        r"""Return native grouped cotangents for ``dB/dX``."""
        return self._field_pullback_native(
            grouped_biot_savart_dB_by_dX_from_inputs,
            vgrad,
        )

    def A_and_dA_pullback_native(self, v, vgrad):
        r"""Return separate native grouped cotangents for ``A`` and ``dA/dX``."""
        return (
            self.A_pullback_native(v),
            self.dA_by_dX_pullback_native(vgrad),
        )

    def B_and_dB_pullback_native(self, v, vgrad):
        r"""Return separate native grouped cotangents for ``B`` and ``dB/dX``."""
        return (
            self.B_pullback_native(v),
            self.dB_by_dX_pullback_native(vgrad),
        )

    def A_vjp(self, v):
        r"""Vector-Jacobian product of A w.r.t. coil DOFs."""
        return self._pullback_to_derivative(self.A_pullback_native(v))

    def A_and_dA_vjp(self, v, vgrad):
        r"""Separate vector-Jacobian products for A and dA/dX."""
        a_pullback, da_pullback = self.A_and_dA_pullback_native(v, vgrad)
        return (
            self._pullback_to_derivative(a_pullback),
            self._pullback_to_derivative(da_pullback),
        )

    def B_and_dB_vjp(self, v, vgrad):
        r"""Separate vector-Jacobian products for B and dB/dX."""
        b_pullback, db_pullback = self.B_and_dB_pullback_native(v, vgrad)
        return (
            self._pullback_to_derivative(b_pullback),
            self._pullback_to_derivative(db_pullback),
        )

    def _add_single_coil_cotangent_to_dofs_gradient(
        self,
        dofs_gradient,
        coil,
        dg,
        dgd,
        dc,
        coil_dofs,
    ):
        curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
        _require_native_curve_geometry(curve)

        if rotmat is not None:
            rotmat_t = _as_jax_float64(rotmat).T
            dg = _as_jax_float64(dg) @ rotmat_t
            dgd = _as_jax_float64(dgd) @ rotmat_t

        coeff_cotangent = curve_pullback_from_dofs(
            curve_spec_from_adapter_curve(curve),
            self._curve_dofs_from_free_vector(curve, coil_dofs),
            dg,
            dgd,
        )
        if _curve_dof_mode(curve) == "full":
            dofs_gradient = _add_full_curve_cotangent_to_dofs_gradient(
                dofs_gradient,
                curve,
                coeff_cotangent,
                self._coil_dof_indices,
                free_positions_for_opt=self._local_free_positions,
            )
        else:
            dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                dofs_gradient,
                curve,
                coeff_cotangent,
                self._coil_dof_indices,
                free_positions=self._local_free_positions(curve),
            )

        current_cotangent = jnp.atleast_1d(_as_jax_float64(scale) * _as_jax_float64(dc))
        for owner, block in current.vjp(current_cotangent).data.items():
            dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                dofs_gradient,
                owner,
                block,
                self._coil_dof_indices,
                free_positions=self._local_free_positions(owner),
            )
        return dofs_gradient

    def coil_cotangents_to_dofs_gradient(
        self,
        d_coil_arrays,
        coil_indices,
        *,
        coil_dofs=None,
    ):
        """Project grouped coil cotangents to the flat free-DOF gradient."""
        if coil_dofs is None:
            coil_dofs = self.x.copy()
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        if _coil_cotangent_arrays_are_jax_compatible(d_coil_arrays):
            extraction_spec = _place_array_tree_on_device(
                self.coil_dof_extraction_spec(),
                coil_dofs.device,
            )
            return _jitted_coil_cotangents_to_dofs_gradient(
                extraction_spec,
                d_coil_arrays,
                _canonical_coil_indices(coil_indices),
                coil_dofs,
            )

        dofs_gradient = coil_dofs - coil_dofs
        for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
            for local_i, global_i in enumerate(indices):
                dofs_gradient = self._add_single_coil_cotangent_to_dofs_gradient(
                    dofs_gradient,
                    self._coils[global_i],
                    jax.lax.index_in_dim(d_g, local_i, axis=0, keepdims=False),
                    jax.lax.index_in_dim(d_gd, local_i, axis=0, keepdims=False),
                    jax.lax.index_in_dim(d_c, local_i, axis=0, keepdims=False),
                    coil_dofs,
                )
        return dofs_gradient

    def coil_cotangents_to_derivative(self, d_coil_arrays, coil_indices):
        """Project grouped coil cotangent arrays to a :class:`Derivative`.

        Curves are projected through immutable specs into full owner partials.
        Free/fixed filtering belongs to ``Derivative.__call__``.

        Args:
            d_coil_arrays: list of ``(d_gammas, d_gammadashs, d_currents)``
                cotangent tuples, one per quadrature group.
            coil_indices: list of index lists, one per group, mapping
                local position to global coil index.

        Returns:
            :class:`Derivative` over all coil DOFs.
        """
        self.coil_dof_extraction_spec()
        coil_dofs = self._normalize_explicit_coil_dofs(self.x)
        device = coil_dofs.device
        contract = self._device_projection_contracts.get(device)
        if contract is None:
            contract = _place_array_tree_on_device(
                (self._owner_partial_extraction_spec, self._owner_partial_projection_spec), device,
            )
            self._device_projection_contracts[device] = contract
        partials = host_array(
            _jitted_coil_cotangents_to_owner_partials(
                *contract, d_coil_arrays, _canonical_coil_indices(coil_indices),
                coil_dofs, self._owner_partial_width,
            ),
            dtype=np.float64,
        )
        return Derivative(OptimizableDefaultDict({
            owner: partials[start:end].copy()
            for owner, (start, end) in self._owner_partial_slices
        }))
