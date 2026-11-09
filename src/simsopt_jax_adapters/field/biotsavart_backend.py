"""JAX field evaluation for native Python objectives.

The mutable Optimizable adapter projects native coil graphs into immutable
specs consumed by pure JAX geometry and Biot-Savart kernels. It does not
implement the simsoptpp.MagneticField interface.
"""

import copy
from dataclasses import dataclass, replace
from functools import partial
from typing import NoReturn, cast

import jax
from simsopt_jax.backend import get_field_kernel_tuning, register_backend_cache_clear
from simsopt_jax.backend.jit import jit
import jax.numpy as jnp
import numpy as np

from simsopt._core.derivative import Derivative, OptimizableDefaultDict
from simsopt._core.json import GSONDecoder
from simsopt.field.coil import Current, CurrentSum, ScaledCurrent
from simsopt.field.magneticfield import MagneticField
from simsopt.geo.curveperturbed import CurvePerturbed
from simsopt_jax.runtime.host_boundary import host_array
from simsopt._core.optimizable import Optimizable
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.core.state_tokens import make_state_token_factory
from simsopt_jax.backend.dtypes import runtime_device_put_tree
from simsopt_jax.core import (
    coil_specs_from_dof_extraction_spec,
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_optimizable_dof_map_spec,
)
from simsopt_jax.core.biotsavart import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_d2A_by_dXdX,
    biot_savart_d2B_by_dXdX,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
)
from simsopt_jax.core.field import (
    coil_set_spec_from_dof_extraction_spec,
    group_biot_savart_B_vjp,
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
    grouped_field_inputs_from_spec,
)
from simsopt_jax.core._math_utils import (
    as_jax_float64 as _as_jax_float64,
)
from simsopt_jax.core.curve_geometry import optimizable_input_dofs_from_map_spec
from simsopt_jax.core._device_scalars import (
    placement_zero as _placement_zero,
    two_pi as _two_pi,
)
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    GroupedCoilSetSpec,
    OptimizableDofMapSpec,
    make_field_eval_spec,
)
from simsopt_jax_adapters.field._coil_graph import (
    _unwrap_coil_curve_and_current_objects,
    coil_graph_parameters,
)
from simsopt_jax_adapters.geo.curve_contract import _optimizable_dof_layout
from simsopt_jax_adapters.geo.curve_specs import (
    adapter_curve_dof_mode,
    curve_spec_from_adapter_curve,
)

_new_backend_epoch = make_state_token_factory()
_backend_epoch = _new_backend_epoch()


def _advance_backend_epoch() -> None:
    """Retire every adapter's device coil state when the backend is reconfigured."""
    global _backend_epoch
    _backend_epoch = _new_backend_epoch()


register_backend_cache_clear(_advance_backend_epoch)


def _place_array_tree_on_device(tree, device):
    """Place dynamic array leaves on one device while preserving static metadata."""
    return runtime_device_put_tree(tree, device=device)


@dataclass(frozen=True)
class _DeviceCoilContract:
    """The adapter's immutable extraction contracts, placed on one device."""

    extraction: CoilSetDofExtractionSpec
    owner_full_map: OptimizableDofMapSpec
    owner_projection: CoilSetDofExtractionSpec


@dataclass(frozen=True, eq=False)
class _CoilEvaluationState:
    """Device inputs of every field evaluation at one coil-DOF state.

    ``coil_dofs`` is the free-DOF vector and ``coil_set_spec`` the grouped
    coil geometry and currents built from it. Compared by identity: a new
    state is built after each recompute notification, extraction-contract
    rebuild or backend reconfiguration (``backend_epoch``).
    """

    backend_epoch: int
    coil_dofs: jax.Array
    coil_set_spec: GroupedCoilSetSpec


@jit
def _jitted_coil_set_spec_from_extraction_spec(coil_dof_extraction_spec, coil_dofs):
    """Owner free-DOF vector -> grouped coil geometry and currents, as one program."""
    return coil_set_spec_from_dof_extraction_spec(coil_dof_extraction_spec, coil_dofs)


__all__ = [
    "JaxBiotSavart",
    "BiotSavartFieldPullback",
]


@jit
def _cyl_points_to_cart(points_cyl):
    points = _as_jax_float64(points_cyl)
    r = points[:, 0]
    phi = points[:, 1]
    z = points[:, 2]
    return jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=1)


@jit
def _canonical_set_points_cyl(points_cyl):
    points = _as_jax_float64(points_cyl)
    return points.at[:, 1].set(jnp.fmod(points[:, 1], _two_pi(points)))


@jit
def _cart_points_to_cyl(points_cart):
    points = _as_jax_float64(points_cart)
    x = points[:, 0]
    y = points[:, 1]
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < _placement_zero(points), phi + _two_pi(points), phi)
    return jnp.stack(
        (
            jnp.sqrt(x * x + y * y),
            phi,
            points[:, 2],
        ),
        axis=1,
    )


@jit
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


@partial(jit, static_argnames=("kernel_tuning",))
def _jitted_grad_absB(points, coil_set_spec, kernel_tuning):
    """``jax.grad`` of ``|B|`` in each point, tiled like the field kernels.

    ``kernel_tuning`` is static so a retuned backend retraces the B kernels
    this program calls; a zero point tile evaluates every point at once.
    """

    def abs_B(point):
        return jnp.linalg.norm(
            grouped_biot_savart_B_from_spec(point[jnp.newaxis, :], coil_set_spec)[0]
        )

    grad_abs_B = jax.grad(abs_B)
    point_chunk_size = kernel_tuning.point_chunk_size
    if point_chunk_size <= 0:
        return jax.vmap(grad_abs_B)(points)
    return jax.lax.map(grad_abs_B, points, batch_size=point_chunk_size)


def _grouped_grad_absB_from_spec(points, coil_set_spec):
    if not coil_set_spec.groups:
        # No coils: B is identically zero, and native's (B . dB/dX) / |B| is 0/0.
        # The 0/0 is formed from a placed zero, so no host literal crosses the
        # strict transfer guard.
        zero = _placement_zero(points)
        return jnp.broadcast_to(zero / zero, points.shape)
    return _jitted_grad_absB(
        points, coil_set_spec, kernel_tuning=get_field_kernel_tuning(),
    )


@partial(jit, static_argnames=("kernel", "batch_size"))
def _per_coil_unit_field_with_batch_size(points, coil_set_spec, kernel, *, batch_size):
    """Per-coil unit-current field as a list of JAX arrays.

    For each coil ``k`` in the public coil ordering, evaluates ``kernel``
    on the single-coil view with ``currents = [1.0]``. Biot-Savart is
    exactly linear in ``I``, so the resulting array equals
    ``∂F/∂I_k`` for the matching spatial-derivative kernel.

    The output is a list of ``ncoils`` separate JAX arrays — one per
    coil. One compiled program per kernel and coil chunk size builds the
    unit current and splits the per-coil rows, so no host value is staged
    at evaluation.
    """
    compute_points = _as_jax_float64(points)
    ncoils = sum(len(group.coil_indices) for group in coil_set_spec.groups)
    result_by_index: dict[int, jax.Array] = {}
    for group in coil_set_spec.groups:
        compute_gammas = _as_jax_float64(group.gammas)
        compute_gammadashs = _as_jax_float64(group.gammadashs)
        unit_current = _as_jax_float64(jnp.ones((1,), dtype=group.currents.dtype))

        def evaluate_single(coil_geometry):
            gamma, gammadash = coil_geometry
            return kernel(
                compute_points,
                gamma[jnp.newaxis, ...],
                gammadash[jnp.newaxis, ...],
                unit_current,
            )

        if batch_size <= 0:
            group_results = jax.vmap(evaluate_single)((compute_gammas, compute_gammadashs))
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


def _per_coil_unit_current_outputs(points, coil_set_spec, *, kernel):
    """Immutable per-coil outputs for the adapter's state-and-point cache."""
    return tuple(_per_coil_unit_field(points, coil_set_spec, kernel))


_PER_COIL_UNIT_CURRENT_OUTPUTS = {
    kernel: partial(_per_coil_unit_current_outputs, kernel=kernel)
    for kernel in (
        biot_savart_B, biot_savart_dB_by_dX, biot_savart_d2B_by_dXdX,
        biot_savart_A, biot_savart_dA_by_dX, biot_savart_d2A_by_dXdX,
    )
}


@pytree_dataclass(data=("d_coil_arrays",), meta=("coil_indices",))
@dataclass(frozen=True)
class BiotSavartFieldPullback:
    """Native grouped cotangent payload for ``JaxBiotSavart`` fields.

    ``d_coil_arrays`` mirrors the grouped field-input structure:
    one ``(d_gammas, d_gammadashs, d_currents)`` tuple per quadrature group.
    ``coil_indices`` maps each group row back to the public coil list.

    Args:
        d_coil_arrays (tuple[tuple[jax.Array, jax.Array, jax.Array], ...]): Per-group
            (d_gammas, d_gammadashs, d_currents) cotangent arrays of shapes (C, Q, 3),
            (C, Q, 3), (C,); units are contracted-objective units per meter or ampere.
        coil_indices (tuple[tuple[int, ...], ...]): Original public coil indices for
            each group, matching its cotangent row order.
    """

    d_coil_arrays: tuple[tuple[jax.Array, jax.Array, jax.Array], ...]
    coil_indices: tuple[tuple[int, ...], ...]


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


def _coil_cotangent_tree(d_coil_arrays):
    """Grouped ``(d_gammas, d_gammadashs, d_currents)`` cotangents as float64 tuples."""
    return tuple(
        tuple(_as_jax_float64(leaf) for leaf in group) for group in d_coil_arrays
    )


def _extraction_input_cotangent(extraction_spec, d_coil_arrays, coil_indices, input_dofs):
    """Pull grouped coil cotangents back to an extraction spec's input DOFs.

    One ``jax.vjp`` of the forward reconstruction: the transposes of the DOF
    maps, symmetry rotations, current scales and curve geometry all come from
    autodiff of ``coil_set_spec_from_dof_extraction_spec``.
    """

    def coil_field_inputs(dofs):
        coil_set_spec = coil_set_spec_from_dof_extraction_spec(extraction_spec, dofs)
        if coil_set_spec.coil_index_lists() != coil_indices:
            raise ValueError(
                "Coil cotangent groups do not match the adapter's coil grouping: "
                f"got {coil_indices}, expected {coil_set_spec.coil_index_lists()}."
            )
        return coil_set_spec.field_inputs()

    _, pullback = jax.vjp(coil_field_inputs, input_dofs)
    return pullback(d_coil_arrays)[0]


@partial(jit, static_argnames=("coil_indices",))
def _jitted_coil_cotangents_to_owner_partials(
    projection_spec, owner_full_map, d_coil_arrays, coil_indices, coil_dofs,
):
    owner_full_dofs = optimizable_input_dofs_from_map_spec(owner_full_map, coil_dofs)
    return _extraction_input_cotangent(
        projection_spec, d_coil_arrays, coil_indices, owner_full_dofs,
    )


_jitted_coil_cotangents_to_dofs_gradient = jit(
    _extraction_input_cotangent, static_argnames=("coil_indices",),
)


def _canonical_coil_indices(coil_indices):
    return tuple(tuple(int(index) for index in indices) for indices in coil_indices)


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
        "JaxBiotSavart only supports affine expressions of scalar "
        f"Current objects; got {type(current).__name__}."
    )


class JaxBiotSavart(Optimizable):
    r"""JAX version of :class:`simsopt.field.BiotSavart`: the field of closed coils.

    Computes the magnetic field induced by closed curves :math:`\Gamma_k` with
    electric currents :math:`I_k`,

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi

    and the vector potential

    .. math::

        A(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{\Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|} d\phi

    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant. As in the native
    class, each integral is the mean of the integrand over the curve's
    quadrature points. Spatial derivatives of :math:`B` and :math:`A` are
    provided, and vector-Jacobian products with respect to the coil DOFs
    propagate through the Optimizable graph.

    Args:
        coils (Sequence[Coil]): Native simsopt coil graph; geometry is in meters and
            physical currents in amperes.

    Supported curves are those
    :func:`~simsopt_jax_adapters.geo.curve_specs.curve_spec_from_adapter_curve`
    converts: XYZ Fourier (including Fourier symmetries and
    ``JaxCurveXYZFourier``), RZ Fourier, planar Fourier (including
    ``JaxCurvePlanarFourier``) and helical curves; perturbed curves and
    filaments with a centroid or Frenet frame wrapping any of them; a
    ``RotatedCurve`` as the outermost wrapper; and other curves that supply a
    compatible ``to_spec()`` method.

    This is not a ``simsoptpp.MagneticField``: field-line tracing,
    :class:`~simsopt.field.InterpolatedField` and native field arithmetic need
    :class:`simsopt.field.BiotSavart`. Call
    :func:`simsopt_jax.backend.set_backend` before constructing it to choose
    the device. An instance holds mutable point state, so do not share it
    across threads. As in the native class, the evaluation points start as
    one point at the origin.
    """

    def clear_points(self) -> None:
        """Clear mutable point buffers without changing source geometry.

        Evaluating the field afterwards raises until points are set again.
        """
        self._points_jax = None
        self._points_cyl_jax = None
        self._invalidate_point_outputs()

    def _evaluation_points(self) -> jax.Array:
        """The Cartesian points every evaluation needs, shape (P, 3), in meters."""
        if self._points_jax is None:
            raise RuntimeError(
                f"{type(self).__name__} has no evaluation points after clear_points(); "
                "call set_points() before evaluating it."
            )
        return self._points_jax

    def _invalidate_point_outputs(self) -> None:
        """Release outputs of the previous point set as soon as it is replaced."""
        self._points_version += 1
        self._field_outputs = {}
        self._field_outputs_key = None

    def _per_coil_unit_current_derivative(self, kernel):
        """Evaluate a unit-current derivative kernel for this field state."""
        # Preserve the public list's mutation isolation; cached arrays are immutable.
        return list(self._field_output(_PER_COIL_UNIT_CURRENT_OUTPUTS[kernel]))

    def _field_output(self, grouped_field):
        """``grouped_field(points, coil_set_spec)`` at the live coil state and points.

        Evaluated once per coil state and point set; JAX arrays are immutable,
        so the cached value is returned as is.
        """
        state = self._coil_evaluation_state()
        if self._field_outputs_key != (state, self._points_version):
            self._field_outputs = {}
            self._field_outputs_key = (state, self._points_version)
        value = self._field_outputs.get(grouped_field)
        if value is None:
            value = grouped_field(self._evaluation_points(), state.coil_set_spec)
            self._field_outputs[grouped_field] = value
        return value

    def B(self) -> jax.Array:
        """Magnetic field B at the evaluation points.

        Returns:
            jax.Array: Magnetic field in tesla, shape (P, 3), Cartesian component
                last.
        """
        return cast(jax.Array, self._field_output(grouped_biot_savart_B_from_spec))

    def A(self):
        """Vector potential A at the evaluation points.

        Returns:
            jax.Array: Vector potential in tesla meters, shape (P, 3), Cartesian
                component last.
        """
        return self._field_output(grouped_biot_savart_A_from_spec)

    def dA_by_dX(self):
        """Spatial Jacobian dA/dX at the evaluation points.

        Returns:
            jax.Array: Shape (P, 3, 3), in tesla; result[p, j, l] = partial_j A_l
                at point p.
        """
        return self._field_output(grouped_biot_savart_dA_by_dX_from_spec)

    def d2A_by_dXdX(self):
        """Spatial Hessian d2A/dXdX at the evaluation points.

        Returns:
            jax.Array: Shape (P, 3, 3, 3), in tesla per meter; result[p, i, j, l]
                = partial_i partial_j A_l.
        """
        return self._field_output(grouped_biot_savart_d2A_by_dXdX_from_spec)

    def dB_by_dX(self):
        """Spatial Jacobian dB/dX at the evaluation points.

        Returns:
            jax.Array: Shape (P, 3, 3), in tesla per meter; result[p, j, l] =
                partial_j B_l at point p.
        """
        return self._field_output(grouped_biot_savart_dB_by_dX_from_spec)

    def d2B_by_dXdX(self):
        """Spatial Hessian d2B/dXdX at the evaluation points.

        Returns:
            jax.Array: Shape (P, 3, 3, 3), in tesla per meter squared; result[p,
                i, j, l] = partial_i partial_j B_l.
        """
        return self._field_output(grouped_biot_savart_d2B_by_dXdX_from_spec)

    def B_and_dB(self):
        """Combined B and dB/dX.

        Returns:
            tuple[jax.Array, jax.Array]: B of shape (P, 3) in tesla and its
                Jacobian of shape (P, 3, 3) in tesla per meter, with derivative
                direction before field component.
        """
        return self._field_output(grouped_biot_savart_B_and_dB_from_spec)

    def AbsB(self):
        """Magnetic-field magnitude at the evaluation points.

        Returns:
            jax.Array: Magnetic-field magnitude in tesla, shape (P, 1).
        """
        return jnp.linalg.norm(self.B(), axis=1)[:, None]

    def GradAbsB(self):
        """Cartesian gradient of ``|B|`` at the evaluation points.

        Reverse-mode ``jax.grad`` of ``|B|`` through the Biot-Savart B kernel,
        one point at a time; it does not form dB/dX.

        Returns:
            jax.Array: Cartesian gradient of field magnitude, shape (P, 3), in
                tesla per meter; zero field retains the division singularity,
                so an empty coil set gives NaN as native does.
        """
        return self._field_output(_grouped_grad_absB_from_spec)

    def B_cyl(self):
        """Magnetic field components in the cylindrical basis.

        Returns:
            jax.Array: Components (B_R, B_phi, B_Z), shape (P, 3), in tesla, in
                the local orthonormal cylindrical basis.
        """
        return _cart_vectors_to_cyl(
            self.B(),
            self._basis_points_cyl(),
        )

    def A_cyl(self):
        """Vector potential components in the cylindrical basis.

        Returns:
            jax.Array: Components (A_R, A_phi, A_Z), shape (P, 3), in tesla
                meters, in the local orthonormal cylindrical basis.
        """
        return _cart_vectors_to_cyl(
            self.A(),
            self._basis_points_cyl(),
        )

    def GradAbsB_cyl(self):
        """``GradAbsB`` components in the cylindrical basis.

        Returns:
            jax.Array: Components (partial_R |B|, (1/R)*partial_phi |B|, partial_Z
                |B|), shape (P, 3), in tesla per meter.
        """
        return _cart_vectors_to_cyl(
            self.GradAbsB(),
            self._basis_points_cyl(),
        )

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        """Per-coil B at unit current.

        Args:
            compute_derivatives (int): Native compatibility argument, ignored; the
                method name determines the spatial derivative order.

        Returns:
            list[jax.Array]: One unit-physical-current B per public coil, each
                shape (P, 3), in tesla per ampere; derivative axes precede the field
                component. Current-expression scaling is not included.
        """
        return self._per_coil_unit_current_derivative(biot_savart_B)

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        """Per-coil ``dB/dX`` at unit current.

        Args:
            compute_derivatives (int): Native compatibility argument, ignored; the
                method name determines the spatial derivative order.

        Returns:
            list[jax.Array]: One unit-physical-current dB_by_dX per public coil,
                each shape (P, 3, 3), in tesla per meter per ampere; derivative axes
                precede the field component. Current-expression scaling is not
                included.
        """
        return self._per_coil_unit_current_derivative(biot_savart_dB_by_dX)

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        """Per-coil ``d2B/dXdX`` at unit current.

        Args:
            compute_derivatives (int): Native compatibility argument, ignored; the
                method name determines the spatial derivative order.

        Returns:
            list[jax.Array]: One unit-physical-current d2B_by_dXdX per public
                coil, each shape (P, 3, 3, 3), in tesla per meter squared per ampere;
                derivative axes precede the field component. Current-expression
                scaling is not included.
        """
        return self._per_coil_unit_current_derivative(biot_savart_d2B_by_dXdX)

    def dA_by_dcoilcurrents(self, compute_derivatives=0):
        """Per-coil A at unit current.

        Args:
            compute_derivatives (int): Native compatibility argument, ignored; the
                method name determines the spatial derivative order.

        Returns:
            list[jax.Array]: One unit-physical-current A per public coil, each
                shape (P, 3), in tesla meters per ampere; derivative axes precede the
                field component. Current-expression scaling is not included.
        """
        return self._per_coil_unit_current_derivative(biot_savart_A)

    def d2A_by_dXdcoilcurrents(self, compute_derivatives=1):
        """Per-coil ``dA/dX`` at unit current.

        Args:
            compute_derivatives (int): Native compatibility argument, ignored; the
                method name determines the spatial derivative order.

        Returns:
            list[jax.Array]: One unit-physical-current dA_by_dX per public coil,
                each shape (P, 3, 3), in tesla per ampere; derivative axes precede the
                field component. Current-expression scaling is not included.
        """
        return self._per_coil_unit_current_derivative(biot_savart_dA_by_dX)

    def d3A_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        """Per-coil ``d2A/dXdX`` at unit current.

        Args:
            compute_derivatives (int): Native compatibility argument, ignored; the
                method name determines the spatial derivative order.

        Returns:
            list[jax.Array]: One unit-physical-current d2A_by_dXdX per public
                coil, each shape (P, 3, 3, 3), in tesla per meter per ampere;
                derivative axes precede the field component. Current-expression
                scaling is not included.
        """
        return self._per_coil_unit_current_derivative(biot_savart_d2A_by_dXdX)


    def _add_child(self, child: Optimizable) -> None:
        # Native sums/scalings also attach operands through the Optimizable graph.
        if isinstance(child, MagneticField):
            self._unsupported_field_arithmetic(child)
        super()._add_child(child)

    def _unsupported_field_arithmetic(self, other=None) -> NoReturn:
        raise TypeError(
            "JaxBiotSavart supports Python objectives, not native field arithmetic; "
            "use simsopt.field.BiotSavart for field sums and scaling."
        )

    __add__ = _unsupported_field_arithmetic
    __radd__ = _unsupported_field_arithmetic
    __mul__ = _unsupported_field_arithmetic
    __rmul__ = _unsupported_field_arithmetic

    def as_dict(self, serial_objs_dict=None) -> dict:
        """Serialize the coil graph and current evaluation points.

        Args:
            serial_objs_dict (dict or None): Shared GSON serialization registry passed
                to Optimizable.as_dict.

        Returns:
            dict: Serialized coil graph and Cartesian points, including None when
                points have been cleared.
        """
        serialized = super().as_dict(serial_objs_dict=serial_objs_dict)
        serialized["points"] = (
            None if self._points_jax is None else self.get_points_cart()
        )
        return serialized

    def __copy__(self):
        """A new field over the same coils and points, registered for their notifications."""
        return self._with_coils(self._coils)

    def __deepcopy__(self, memo):
        """A new field over deep copies of the coils (which simsopt coils do not support)."""
        return self._with_coils(copy.deepcopy(self._coils, memo))

    def _with_coils(self, coils):
        field = type(self)(coils)
        field._points_jax = self._points_jax
        field._points_cyl_jax = self._points_cyl_jax
        return field

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        """Reconstruct a field and its points from a GSON payload.

        Args:
            d (dict): Decoded GSON field payload with coils and points; as in
                native BiotSavart the points key is required, and None in it
                restores cleared points.
            serial_objs_dict (dict): Serialized-object registry used by GSONDecoder.
            recon_objs (dict): Registry of objects already reconstructed by GSONDecoder.

        Returns:
            JaxBiotSavart object: Reconstructed field with its serialized
                evaluation points restored.

        Raises:
            KeyError: The payload has no points key.
        """
        decoder = GSONDecoder()
        coils = decoder.process_decoded(d["coils"], serial_objs_dict, recon_objs)
        field = cls(coils)
        points = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        if points is None:
            field.clear_points()
        else:
            field.set_points(points)
        return field

    def __init__(self, coils):
        """Build the field over a native coil graph, evaluated at the origin.

        Args:
            coils (Sequence[Coil]): Native simsopt coils; geometry in meters and
                physical currents in amperes. The field depends on them as an
                Optimizable and refreshes when they notify a change.
        """
        self._coils = list(coils)
        self._points_version = 0
        # Native fields start with one evaluation point at the origin.
        self.set_points(np.zeros((1, 3)))
        self._free_dof_layout_ready = False
        self._fixed_dofs_maybe_changed = False
        self._local_free_positions_by_opt = {}
        self._device_contracts = {}
        self._coil_state = None
        self._field_outputs = {}
        self._field_outputs_key = None
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=self._coils)
        self._free_dof_layout_ready = True
        self._rebuild_coil_dof_extraction_spec()
        self._captured_coil_state_fingerprint = (
            self._current_captured_coil_state_fingerprint()
        )

    def update_free_dof_size_indices(self) -> None:
        """Refresh free-DOF layout and invalidate its captured extraction contract.

        Free-DOF indices and extraction contracts are updated.
        """
        super().update_free_dof_size_indices()
        self._local_free_positions_by_opt.clear()
        if self._free_dof_layout_ready:
            self._rebuild_coil_dof_extraction_spec()
            self._captured_coil_state_fingerprint = (
                self._current_captured_coil_state_fingerprint()
            )
            self._fixed_dofs_maybe_changed = False

    def _current_captured_coil_state_fingerprint(self) -> tuple[bytes, ...]:
        """Fixed coordinates, non-DOF coil parameters (current scales, filament
        offsets, frame-rotation scales) and perturbation sample contents,
        fingerprinted once at the first read after a dependency notification."""
        return tuple(
            np.ascontiguousarray(
                np.asarray(opt.local_full_x, dtype=np.float64)[
                    ~np.asarray(opt.local_dofs_free_status, dtype=bool)
                ]
            ).tobytes()
            for opt in self.unique_dof_lineage
        ) + (coil_graph_parameters(self.ancestors).tobytes(),) + tuple(
            np.ascontiguousarray(array).tobytes()
            for opt in self.ancestors if isinstance(opt, CurvePerturbed)
            for array in opt.sample._sample
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
        if check_fixed:
            self._fixed_dofs_maybe_changed = False
        if (fingerprint == self._captured_coil_state_fingerprint
                and not self._perturbation_samples_changed()):
            return
        self._rebuild_coil_dof_extraction_spec()
        self._captured_coil_state_fingerprint = fingerprint

    def _rebuild_coil_dof_extraction_spec(self) -> None:
        """Rebuild the extraction contracts; retire what was placed or built from the old ones."""
        self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()
        self._device_contracts = {}
        self._drop_coil_state()

    def _drop_coil_state(self) -> None:
        self._coil_state = None
        self._field_outputs = {}
        self._field_outputs_key = None

    def recompute_bell(self, parent=None):
        """Drop the cached coil state and field values on simsopt's recompute notification.

        Args:
            parent (Optimizable or None): Dependency that triggered the recompute
                notification; recompute_bell does not inspect it.
        """
        self._drop_coil_state()

    def _device_contract(self, device) -> _DeviceCoilContract:
        """The extraction contracts placed on ``device``, placed once per contract build."""
        contract = self._device_contracts.get(device)
        if contract is None:
            contract = _DeviceCoilContract(*_place_array_tree_on_device(
                (
                    self._coil_dof_extraction_spec,
                    self._owner_full_dof_map_spec,
                    self._owner_partial_projection_spec,
                ),
                device,
            ))
            self._device_contracts[device] = contract
        return contract

    def _coil_evaluation_state(self) -> _CoilEvaluationState:
        """Device coil DOFs and grouped coil geometry at the live coil-DOF state.

        Built by one compiled program at the first evaluation after a recompute
        notification, contract rebuild or backend reconfiguration; every field
        value, pullback and per-coil derivative at that state reuses it.
        """
        self.coil_dof_extraction_spec()
        state = self._coil_state
        if state is None or state.backend_epoch != _backend_epoch:
            coil_dofs = self._normalize_explicit_coil_dofs(self.x)
            state = _CoilEvaluationState(
                backend_epoch=_backend_epoch,
                coil_dofs=coil_dofs,
                coil_set_spec=_jitted_coil_set_spec_from_extraction_spec(
                    self._device_contract(coil_dofs.device).extraction,
                    coil_dofs,
                ),
            )
            self._coil_state = state
        return state

    def set_recompute_flag(self, parent=None):
        """Mark dependent field state stale and defer the coil-parameter re-check to the next read.

        Native fields read every coil parameter on their next evaluation after
        any notification, so every notification, with or without a parent and
        including the ones a DOF setter sends, marks fixed DOFs and non-DOF
        coil parameters for re-check on the next read.

        Args:
            parent (Optimizable or None): Dependency that triggered the recompute
                notification, or None for a direct call.
        """
        if self._free_dof_layout_ready:
            # Every ancestor notifies separately; compare once, on the next read.
            self._fixed_dofs_maybe_changed = True
        super().set_recompute_flag(parent=parent)

    def _local_free_positions(self, opt):
        cached = self._local_free_positions_by_opt.get(opt)
        if cached is None:
            cached = np.flatnonzero(opt.local_dofs_free_status)
            self._local_free_positions_by_opt[opt] = cached
        return cached

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
                    full_graph=adapter_curve_dof_mode(curve) == "full",
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
        The owner-full map rebuilds the concatenated owner vectors from the
        free vector, as the point where the projection is differentiated.
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
            # The validating constructor keeps owner and target lengths equal.
            return make_optimizable_dof_map_spec(
                template_full_dofs=source_map.template_full_dofs,
                owner_segments=segments,
                input_mode=source_map.input_mode,
                input_start=source_map.input_start,
                input_end=source_map.input_end,
            )

        projection_coils = []
        for coil, spec in zip(self._coils, extraction_spec.coils, strict=True):
            curve, _rotation, current, _scale = _unwrap_coil_curve_and_current(coil)
            full_graph = adapter_curve_dof_mode(curve) == "full"
            owner_spec = (
                replace(
                    spec,
                    curve=curve_spec_from_adapter_curve(curve, separate_owners=True),
                    curve_map=self._free_vector_dof_map_spec(
                        curve, full_graph=True, separate_owners=True,
                    ),
                ) if full_graph else spec
            )
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
        self._owner_partial_projection_spec = make_coil_set_dof_extraction_spec(projection_coils)
        self._owner_partial_slices = tuple(owner_slices.items())
        self._owner_full_dof_map_spec = self._full_input_dof_map_spec(
            _as_jax_float64(np.concatenate(
                [np.zeros(0)] + [
                    np.asarray(owner.local_full_x, dtype=np.float64)
                    for owner in owner_slices
                ]
            )),
            tuple(
                segment
                for owner, (owner_start, _owner_end) in owner_slices.items()
                if int(owner.local_dof_size) > 0
                for segment in self._free_owner_segments(owner, owner_start)
            ),
        )

    def coil_dof_extraction_spec(self):
        """Return the cached immutable owner-DOF reconstruction contract.

        Returns:
            CoilSetDofExtractionSpec object: Cached reconstruction contract,
                refreshed when captured fixed DOFs, non-DOF coil parameters,
                samples or layout change.
        """
        # Direct sample replacement can bypass the native curve notification.
        previous_spec = self._coil_dof_extraction_spec
        self._refresh_captured_coil_state(check_fixed=self._fixed_dofs_maybe_changed)
        if self._coil_dof_extraction_spec is not previous_spec:
            # The contract was just built from live state: notify dependents only.
            super().set_recompute_flag()
        return self._coil_dof_extraction_spec

    def _free_vector_dof_map_spec(self, opt, *, full_graph, separate_owners=False):
        if full_graph:
            full_dofs, full_indices = _optimizable_dof_layout(opt, separate_owners=separate_owners)
            owner_segments = tuple(
                segment
                for dep_opt, (target_start, _target_end) in full_indices.items()
                if int(dep_opt.local_dof_size) > 0
                for segment in self._free_owner_segments(dep_opt, target_start)
            )
            template_full_dofs = _as_jax_float64(full_dofs)
            return self._full_input_dof_map_spec(template_full_dofs, owner_segments)

        template_full_dofs = _as_jax_float64(opt.local_full_x)
        if opt.local_dof_size == 0:
            return self._full_input_dof_map_spec(template_full_dofs, ())

        return self._full_input_dof_map_spec(
            template_full_dofs, self._free_owner_segments(opt),
        )

    def _free_owner_segments(self, opt, target_offset=0):
        """Copy ranges from ``opt``'s free-vector slice to its full local DOFs.

        Target bounds are shifted by ``target_offset``, the start of ``opt``'s
        local full vector in the destination.
        """
        return tuple(
            (owner_start, owner_end, int(target_offset + local_start), int(target_offset + local_end))
            for owner_start, owner_end, local_start, local_end in _owner_segments_from_free_positions(
                self._coil_dof_indices[opt][0], self._local_free_positions(opt),
            )
        )

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
        """Build immutable per-coil specs from an explicit flat DOF vector.

        Args:
            coil_dofs (array-like): Global free coil/current DOFs, shape
                (self.dof_size,), in self.x order; units depend on the corresponding
                owner.

        Returns:
            tuple[CoilSpec, ...]: Reconstructed immutable payloads in public coil
                order.
        """
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        return coil_specs_from_dof_extraction_spec(
            self.coil_dof_extraction_spec(),
            coil_dofs,
        )

    def grouped_coil_arrays_from_dofs(self, coil_dofs):
        """Build grouped coil arrays from an explicit flat DOF vector.

        Args:
            coil_dofs (array-like): Global free coil/current DOFs, shape
                (self.dof_size,), in self.x order; units depend on the corresponding
                owner.

        Returns:
            list[tuple]: Geometry/tangent/current triples with shapes (C, Q, 3),
                (C, Q, 3), (C,), grouped by quadrature count.
        """
        return list(
            grouped_field_inputs_from_spec(self.coil_set_spec_from_dofs(coil_dofs))
        )

    def coil_set_spec_from_dofs(self, coil_dofs):
        """Build an immutable grouped coil spec from an explicit flat DOF vector.

        Args:
            coil_dofs (array-like): Global free coil/current DOFs, shape
                (self.dof_size,), in self.x order; units depend on the corresponding
                owner.

        Returns:
            GroupedCoilSetSpec object: Immutable sampled geometry and currents
                grouped by quadrature count.
        """
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        return coil_set_spec_from_dof_extraction_spec(
            self.coil_dof_extraction_spec(), coil_dofs,
        )

    @property
    def coils(self):
        """Return the native source coils in public order.

        Returns:
            Sequence[Coil]: Stored public native coil sequence, retaining its
                object identities.
        """
        return self._coils

    def set_points(self, points):
        """Set evaluation points (converted to a JAX array once).

        Accepts both NumPy and JAX arrays.  JAX arrays stay on device
        without a host round-trip. Mutates the cached point buffer on this
        instance, so callers should not share one ``JaxBiotSavart`` across
        concurrent evaluation threads.

        Args:
            points (array-like): Cartesian evaluation points, shape (P, 3), in meters.

        Returns:
            JaxBiotSavart object: This instance after replacing owned point
                buffers and invalidating cached point outputs.
        """
        # Field values are cached per point set, so the field owns its points: host
        # inputs are mutable, and placement snapshots them at the shared boundary.
        if isinstance(points, jax.Array):
            points = jnp.array(points, copy=True)
        self._points_jax = _as_jax_float64(points)
        self._points_cyl_jax = None
        self._invalidate_point_outputs()
        return self

    def set_points_cart(self, points):
        """Set evaluation points and invalidate cached field outputs.

        Args:
            points (array-like): Cartesian evaluation points, shape (P, 3), in meters.

        Returns:
            JaxBiotSavart object: This instance after replacing owned point
                buffers and invalidating cached point outputs.
        """
        return self.set_points(points)

    def set_points_cyl(self, points_cyl):
        """Set evaluation points and invalidate cached field outputs.

        Args:
            points_cyl (array-like): Cylindrical (R, phi, Z) coordinates, shape (P, 3),
                with lengths in meters and phi in radians; angles are reduced using
                fmod(phi, 2*pi).

        Returns:
            JaxBiotSavart object: This instance after replacing owned point
                buffers and invalidating cached point outputs.
        """
        self._points_cyl_jax = _canonical_set_points_cyl(_as_jax_float64(points_cyl))
        self._points_jax = _cyl_points_to_cart(self._points_cyl_jax)
        self._invalidate_point_outputs()
        return self

    def get_points_cart_ref(self):
        """Return the current JAX point buffer for point-preserving callers.

        Returns:
            jax.Array or None: Stored Cartesian point buffer, shape (P, 3), in
                meters, or None after clearing points.
        """
        return self._points_jax

    def get_points_cart(self):
        """Materialize the current Cartesian evaluation points on the host.

        Returns:
            numpy.ndarray: Writable Cartesian coordinates, shape (P, 3), in
                meters, materialized at the explicit host boundary.
        """
        return host_array(self._evaluation_points(), dtype=np.float64)

    def get_points_cyl(self):
        """Materialize the current cylindrical evaluation points on the host.

        Returns:
            numpy.ndarray: Writable (R, phi, Z) coordinates, shape (P, 3), with
                lengths in meters and angle in radians. Cartesian-derived phi is in
                [0, 2*pi); explicitly set angles retain signed fmod values.
        """
        return host_array(self._basis_points_cyl(), dtype=np.float64)

    def _basis_points_cyl(self):
        """Explicitly set cylindrical points, else those derived from the Cartesian ones."""
        if self._points_cyl_jax is not None:
            return self._points_cyl_jax
        return _cart_points_to_cyl(self._evaluation_points())

    def set_points_from_spec(self, field_eval_spec):
        """Set evaluation points from an immutable field-evaluation spec.

        This still mutates the receiving ``JaxBiotSavart`` instance.

        Args:
            field_eval_spec (FieldEvalSpec): Immutable Cartesian point cloud of shape
                (P, 3), in meters.

        Returns:
            JaxBiotSavart object: This instance after replacing owned point
                buffers and invalidating cached point outputs.
        """
        return self.set_points(field_eval_spec.points)

    def field_eval_spec(self):
        """Build the immutable field-evaluation spec for the current points.

        Returns:
            FieldEvalSpec object: Float64 Cartesian point payload of
                shape (P, 3), in meters.
        """
        return make_field_eval_spec(self._evaluation_points())


    def coil_set_spec(self):
        """Return the grouped coil spec for the current coil graph.

        Reconstructed from the live free-DOF vector with the cached explicit
        extraction contract, once per coil-DOF state.

        Returns:
            GroupedCoilSetSpec object: Cached sampled geometry and physical
                currents for the current owner DOFs.
        """
        return self._coil_evaluation_state().coil_set_spec

    def coil_specs(self):
        """Build immutable per-coil specs from the live coil graph.

        Returns:
            tuple[CoilSpec, ...]: Immutable per-coil payloads reconstructed from
                the current free DOFs, with each coefficient leaf matching its curve
                spec shape.
        """
        return self.coil_specs_from_dofs(_as_jax_float64(self.x))

    # ------------------------------------------------------------------
    # VJP (reverse-mode gradient w.r.t. coil DOFs)
    # ------------------------------------------------------------------


    def B_pullback_native(self, v):
        """Return the native grouped cotangents for ``B``.

        This is the JAX-native pullback boundary. It returns cotangents with
        respect to grouped coil geometry/current arrays, without projecting
        them into SIMSOPT's public :class:`Derivative` object graph.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.

        Returns:
            BiotSavartFieldPullback object: Grouped geometry/tangent/current
                cotangents with leaf shapes (C, Q, 3), (C, Q, 3), (C,) and static
                public coil indices.
        """
        points = self._evaluation_points()
        v_jax = _as_jax_float64(v)
        coil_set_spec = self.coil_set_spec()
        d_coil_arrays = tuple(
            group_biot_savart_B_vjp(
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
        """Vector-Jacobian product of B w.r.t. coil DOFs.

        Given a cotangent vector ``v`` (typically ``dJ/dB``), returns
        a :class:`Derivative` mapping every coil DOF, including fixed DOFs, to its
        contribution to the scalar objective.

        Uses ``jax.vjp`` through the pure Biot-Savart kernel, then pulls the
        grouped geometry/current cotangents back to the owner DOFs with one
        ``jax.vjp`` of the coil-spec reconstruction.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.

        Returns:
            Derivative object: Full owner partials, including fixed DOFs, after
                projecting geometry and current cotangents; free/fixed filtering
                occurs when the Derivative is called.
        """
        return self._pullback_to_derivative(self.B_pullback_native(v))

    def _field_pullback_native(
        self,
        grouped_forward,
        cotangent,
    ):
        coil_set_spec = self.coil_set_spec()
        coil_arrays = coil_set_spec.field_inputs()
        if not coil_arrays:
            return BiotSavartFieldPullback((), ())

        points = self._evaluation_points()
        _, pullback = jax.vjp(
            lambda grouped_inputs: grouped_forward(points, grouped_inputs),
            coil_arrays,
        )
        d_coil_arrays = pullback(_as_jax_float64(cotangent))[0]
        return BiotSavartFieldPullback(
            d_coil_arrays=tuple(d_coil_arrays),
            coil_indices=coil_set_spec.coil_index_lists(),
        )

    def A_pullback_native(self, v):
        """Return native grouped cotangents for ``A``.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.

        Returns:
            BiotSavartFieldPullback object: Grouped geometry/tangent/current
                cotangents with leaf shapes (C, Q, 3), (C, Q, 3), (C,) and static
                public coil indices.
        """
        return self._field_pullback_native(grouped_biot_savart_A_from_inputs, v)

    def dA_by_dX_pullback_native(self, vgrad):
        """Return native grouped cotangents for ``dA/dX``.

        Args:
            vgrad (array-like): Spatial-Jacobian cotangent, shape (P, 3, 3), with
                derivative direction before field component; units follow the scalar
                objective.

        Returns:
            BiotSavartFieldPullback object: Grouped geometry/tangent/current
                cotangents with leaf shapes (C, Q, 3), (C, Q, 3), (C,) and static
                public coil indices.
        """
        return self._field_pullback_native(
            grouped_biot_savart_dA_by_dX_from_inputs,
            vgrad,
        )

    def dB_by_dX_pullback_native(self, vgrad):
        """Return native grouped cotangents for ``dB/dX``.

        Args:
            vgrad (array-like): Spatial-Jacobian cotangent, shape (P, 3, 3), with
                derivative direction before field component; units follow the scalar
                objective.

        Returns:
            BiotSavartFieldPullback object: Grouped geometry/tangent/current
                cotangents with leaf shapes (C, Q, 3), (C, Q, 3), (C,) and static
                public coil indices.
        """
        return self._field_pullback_native(
            grouped_biot_savart_dB_by_dX_from_inputs,
            vgrad,
        )

    def A_and_dA_pullback_native(self, v, vgrad):
        """Return separate native grouped cotangents for ``A`` and ``dA/dX``.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.
            vgrad (array-like): Spatial-Jacobian cotangent, shape (P, 3, 3), with
                derivative direction before field component; units follow the scalar
                objective.

        Returns:
            tuple[BiotSavartFieldPullback, BiotSavartFieldPullback]: Separate
                value and spatial-Jacobian grouped cotangents, in that order;
                contributions are not added.
        """
        return (
            self.A_pullback_native(v),
            self.dA_by_dX_pullback_native(vgrad),
        )

    def B_and_dB_pullback_native(self, v, vgrad):
        """Return separate native grouped cotangents for ``B`` and ``dB/dX``.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.
            vgrad (array-like): Spatial-Jacobian cotangent, shape (P, 3, 3), with
                derivative direction before field component; units follow the scalar
                objective.

        Returns:
            tuple[BiotSavartFieldPullback, BiotSavartFieldPullback]: Separate
                value and spatial-Jacobian grouped cotangents, in that order;
                contributions are not added.
        """
        return (
            self.B_pullback_native(v),
            self.dB_by_dX_pullback_native(vgrad),
        )

    def A_vjp(self, v):
        """Vector-Jacobian product of A w.r.t. coil DOFs.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.

        Returns:
            Derivative object: Full owner partials, including fixed DOFs, after
                projecting geometry and current cotangents; free/fixed filtering
                occurs when the Derivative is called.
        """
        return self._pullback_to_derivative(self.A_pullback_native(v))

    def A_and_dA_vjp(self, v, vgrad):
        """Separate vector-Jacobian products for A and dA/dX.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.
            vgrad (array-like): Spatial-Jacobian cotangent, shape (P, 3, 3), with
                derivative direction before field component; units follow the scalar
                objective.

        Returns:
            tuple[Derivative, Derivative]: Separate value and spatial-Jacobian
                owner partials, including fixed DOFs; contributions are not added.
        """
        a_pullback, da_pullback = self.A_and_dA_pullback_native(v, vgrad)
        return (
            self._pullback_to_derivative(a_pullback),
            self._pullback_to_derivative(da_pullback),
        )

    def B_and_dB_vjp(self, v, vgrad):
        """Separate vector-Jacobian products for B and dB/dX.

        Args:
            v (array-like): Field-value cotangent, shape (P, 3), contracting the output
                as sum(v * F); units follow the scalar objective.
            vgrad (array-like): Spatial-Jacobian cotangent, shape (P, 3, 3), with
                derivative direction before field component; units follow the scalar
                objective.

        Returns:
            tuple[Derivative, Derivative]: Separate value and spatial-Jacobian
                owner partials, including fixed DOFs; contributions are not added.
        """
        b_pullback, db_pullback = self.B_and_dB_pullback_native(v, vgrad)
        return (
            self._pullback_to_derivative(b_pullback),
            self._pullback_to_derivative(db_pullback),
        )

    def coil_cotangents_to_dofs_gradient(
        self,
        d_coil_arrays,
        coil_indices,
        *,
        coil_dofs=None,
    ):
        """Project grouped coil cotangents to the flat free-DOF gradient.

        Args:
            d_coil_arrays (Sequence[tuple]): Per-group (d_gammas, d_gammadashs,
                d_currents) cotangent arrays of shapes (C, Q, 3), (C, Q, 3), (C,); units
                are contracted-objective units per meter or ampere.
            coil_indices (Sequence[Sequence[int]]): Original public coil indices for
                each group, matching its cotangent row order.
            coil_dofs (array-like or None): Global free coil/current DOFs, shape
                (self.dof_size,), in self.x order; units depend on the corresponding
                owner. None snapshots the current self.x.

        Returns:
            jax.Array: Flat free-DOF gradient, shape (self.dof_size,), in self.x
                order, in objective units per corresponding DOF unit.
        """
        if coil_dofs is None:
            coil_dofs = np.array(self.x, dtype=np.float64)
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        self.coil_dof_extraction_spec()
        return _jitted_coil_cotangents_to_dofs_gradient(
            self._device_contract(coil_dofs.device).extraction,
            _coil_cotangent_tree(d_coil_arrays),
            _canonical_coil_indices(coil_indices),
            coil_dofs,
        )

    def coil_cotangents_to_derivative(self, d_coil_arrays, coil_indices):
        """Project grouped coil cotangent arrays to a :class:`Derivative`.

        The partials of every owner, fixed DOFs included, are one ``jax.vjp``
        of the coil-spec reconstruction from the concatenated owner vectors.
        Free/fixed filtering belongs to ``Derivative.__call__``.

        Args:
            d_coil_arrays (Sequence[tuple]): Per-group (d_gammas, d_gammadashs,
                d_currents) cotangent arrays of shapes (C, Q, 3), (C, Q, 3), (C,); units
                are contracted-objective units per meter or ampere.
            coil_indices (Sequence[Sequence[int]]): Original public coil indices for
                each group, matching its cotangent row order.

        Returns:
            Derivative object: Full owner partials, including fixed DOFs,
                materialized on the host; free/fixed filtering occurs when called.
        """
        self.coil_dof_extraction_spec()
        coil_dofs = self._normalize_explicit_coil_dofs(self.x)
        contract = self._device_contract(coil_dofs.device)
        partials = host_array(
            _jitted_coil_cotangents_to_owner_partials(
                contract.owner_projection, contract.owner_full_map,
                _coil_cotangent_tree(d_coil_arrays),
                _canonical_coil_indices(coil_indices), coil_dofs,
            ),
            dtype=np.float64,
        )
        return Derivative(OptimizableDefaultDict({
            owner: partials[start:end].copy()
            for owner, (start, end) in self._owner_partial_slices
        }))
