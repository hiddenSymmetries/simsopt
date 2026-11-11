"""JIT-compilable, autodiff-compatible Biot-Savart kernels for CPU and GPU.

Public kernels cast inputs to float64, compute in float64 and return JAX
arrays without requiring a native field object.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

import jax
from simsopt_jax.backend.runtime import register_backend_cache_clear
from simsopt_jax.backend.dtypes import as_jax_array as _as_jax_array
from simsopt_jax.backend.jit import jit
from jax import lax
import jax.numpy as jnp
import numpy as np


from ._device_scalars import device_one as _device_one
from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
    pairwise_sum_axis as _pairwise_sum_axis,
    zeros as _zeros,
)

_MU0_OVER_4PI = 1e-7

# Field tiling of the public kernels: coils per chunk, quadrature nodes per
# block and points per chunk; a count within one tile is not split. The kernel
# factories ``_make_kernel`` and ``_make_B_vjp_kernel`` take any other sizes
# explicitly, zero disabling a level.
COIL_CHUNK_SIZE = 64
QUADRATURE_BLOCK_SIZE = 64
POINT_CHUNK_SIZE = 1024


# ── Array slicing primitives ──────────────────────────────────────────


def _slice_coil_chunk(array, start, chunk_size: int):
    zero = _as_int32_scalar(0)
    return lax.dynamic_slice(
        array,
        (_as_int32_scalar(start),) + (zero,) * (array.ndim - 1),
        (chunk_size,) + tuple(array.shape[1:]),
    )


def _slice_quadrature_block(array, start, block_size: int):
    zero = _as_int32_scalar(0)
    return lax.dynamic_slice(
        array,
        (zero, _as_int32_scalar(start)) + (zero,) * (array.ndim - 2),
        (array.shape[0], block_size) + tuple(array.shape[2:]),
    )


def _slice_point_chunk(points, start, chunk_size: int):
    zero = _as_int32_scalar(0)
    return lax.dynamic_slice(
        points,
        (_as_int32_scalar(start),) + (zero,) * (points.ndim - 1),
        (chunk_size,) + tuple(points.shape[1:]),
    )


def _float64_scalar(reference, value):
    return _device_one(reference) * np.float64(value)


def _as_int32_scalar(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _radius_squared(diff):
    """Return squared separation without regularizing point-on-coil singularities."""

    return jnp.sum(diff * diff, axis=-1)


def _cross_product(left, right):
    left_x = left[..., 0]
    left_y = left[..., 1]
    left_z = left[..., 2]
    right_x = right[..., 0]
    right_y = right[..., 1]
    right_z = right[..., 2]
    return jnp.stack(
        (
            left_y * right_z - left_z * right_y,
            left_z * right_x - left_x * right_z,
            left_x * right_y - left_y * right_x,
        ),
        axis=-1,
    )


# ── Tree utilities ────────────────────────────────────────────────────


def _tree_dynamic_update(prefix_tree, chunk_tree, start_index: int):
    return jax.tree.map(
        lambda acc, update: lax.dynamic_update_slice(
            acc,
            update,
            (_as_int32_scalar(start_index),) + (_as_int32_scalar(0),) * (acc.ndim - 1),
        ),
        prefix_tree,
        chunk_tree,
    )


def _tree_zeros_like_prefix(reference_tree, prefix_size: int):
    return jax.tree.map(
        lambda leaf: _zeros(
            (prefix_size,) + tuple(leaf.shape[1:]),
            dtype=leaf.dtype,
        ),
        reference_tree,
    )


def _tree_add(left, right):
    return jax.tree.map(
        lambda left_leaf, right_leaf: left_leaf + right_leaf, left, right
    )


# ── Tiling primitives ────────────────────────────────────────────────


def _two_chunk_sum(first_chunk, second_chunk, reduce_chunk):
    return _tree_add(reduce_chunk(*first_chunk), reduce_chunk(*second_chunk))


def _coil_chunk_reduce(
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    zero,
    reduce_chunk,
):
    """Reduce coil chunks assuming a concrete leading coil dimension.

    This helper is only safe on call paths where ``currents.shape[0]`` is known
    at trace time. The surrounding kernel builders satisfy that contract by
    specializing each JIT closure to a fixed grouped-coil layout before calling
    into this reducer.
    """
    coil_count = currents.shape[0]
    if coil_count == 0:
        return zero
    if chunk_size <= 0 or coil_count <= chunk_size:
        return reduce_chunk(gammas, gammadashs, currents)

    chunk_count = (coil_count + chunk_size - 1) // chunk_size
    if chunk_count == 2:
        # Exact two-chunk shapes avoid padded loop overhead in B/B+dB hot paths.
        second_chunk_size = coil_count - chunk_size
        return _two_chunk_sum(
            (
                _slice_coil_chunk(gammas, 0, chunk_size),
                _slice_coil_chunk(gammadashs, 0, chunk_size),
                _slice_coil_chunk(currents, 0, chunk_size),
            ),
            (
                _slice_coil_chunk(gammas, chunk_size, second_chunk_size),
                _slice_coil_chunk(gammadashs, chunk_size, second_chunk_size),
                _slice_coil_chunk(currents, chunk_size, second_chunk_size),
            ),
            reduce_chunk,
        )

    full_chunk_count, tail_size = divmod(coil_count, chunk_size)

    # Keep the outer coil accumulation serial until parity data implicates it;
    # the quadrature-axis sum dominates the known reduction-order drift, while
    # tree-reducing chunk outputs would add another staged hot-path combine.
    def body(chunk_index: int, acc):
        start = chunk_index * chunk_size
        chunk_gammas = _slice_coil_chunk(gammas, start, chunk_size)
        chunk_gammadashs = _slice_coil_chunk(gammadashs, start, chunk_size)
        chunk_currents = _slice_coil_chunk(currents, start, chunk_size)
        return _tree_add(
            acc,
            reduce_chunk(chunk_gammas, chunk_gammadashs, chunk_currents),
        )

    reduced = lax.fori_loop(0, full_chunk_count, body, zero)
    if tail_size:
        tail_start = full_chunk_count * chunk_size
        reduced = _tree_add(
            reduced,
            reduce_chunk(
                gammas[tail_start:],
                gammadashs[tail_start:],
                currents[tail_start:],
            ),
        )
    return reduced


def _quadrature_block_integral(
    x,
    gammas,
    gammadashs,
    *,
    block_size: int,
    integrand,
):
    quadrature_count = gammas.shape[1]
    # The single-block, exact two-block, and full-block-plus-tail >=3 paths use
    # different quadrature reduction-tree shapes. Keep parity tests pinned
    # across block boundaries when changing this routine.
    if block_size <= 0 or quadrature_count <= block_size:
        values = integrand(x, gammas, gammadashs)
        return _pairwise_sum_axis(values, axis=1) * _as_jax_array(1.0 / values.shape[1], dtype=values.dtype)

    block_count = (quadrature_count + block_size - 1) // block_size
    if block_count == 2:
        # Exact two-block shapes avoid padded loop overhead in B/B+dB hot paths.
        second_block_size = quadrature_count - block_size
        return (
            _two_chunk_sum(
                (
                    _slice_quadrature_block(gammas, 0, block_size),
                    _slice_quadrature_block(gammadashs, 0, block_size),
                ),
                (
                    _slice_quadrature_block(gammas, block_size, second_block_size),
                    _slice_quadrature_block(
                        gammadashs,
                        block_size,
                        second_block_size,
                    ),
                ),
                lambda gg, ggd: _pairwise_sum_axis(integrand(x, gg, ggd), axis=1),
            )
            / quadrature_count
        )

    full_block_count, tail_size = divmod(quadrature_count, block_size)
    zero = _zeros((gammas.shape[0], 3), dtype=jnp.float64)

    def body(block_index: int, acc):
        start = block_index * block_size
        block_gammas = _slice_quadrature_block(gammas, start, block_size)
        block_gammadashs = _slice_quadrature_block(
            gammadashs,
            start,
            block_size,
        )
        block_integrand = integrand(x, block_gammas, block_gammadashs)
        return acc + _pairwise_sum_axis(block_integrand, axis=1)

    integral_sum = lax.fori_loop(0, full_block_count, body, zero)
    if tail_size:
        tail_start = full_block_count * block_size
        tail_integrand = integrand(
            x,
            gammas[:, tail_start:],
            gammadashs[:, tail_start:],
        )
        integral_sum = integral_sum + _pairwise_sum_axis(tail_integrand, axis=1)
    return integral_sum / quadrature_count


# The Hessian integrand carries 27 doubles per (point, quadrature node) where
# ``B`` and ``dB`` together carry 12, and its ``jacfwd(jacfwd)`` trace keeps
# both tangent levels live.  The point tile is sized for the ``B``/``B+dB``
# hot paths, so the Hessian kernel takes one eighth of it:
# 2048 points at 18 NCSX coils x 250 nodes otherwise materializes >20 GB and
# fails on a 32 GB device, while 256 points stay near 2 GB.
_HESSIAN_POINT_CHUNK_DIVISOR = 8


def hessian_point_chunk_size(point_chunk_size: int) -> int:
    """Return the Hessian point tile size for a field-value tile size.

    Args:
        point_chunk_size (int): Point tile size for field values; nonpositive sizes
            disable tiling.

    Returns:
        int: Zero when tiling is disabled; otherwise the value tile divided by
            the Hessian divisor and floored at one.
    """
    if point_chunk_size <= 0:
        return 0
    return max(1, point_chunk_size // _HESSIAN_POINT_CHUNK_DIVISOR)


def _point_chunk_reduce(points, chunk_kernel, chunk_size):
    """Evaluate ``chunk_kernel`` over points, optionally tiled by ``chunk_size``."""
    point_count = points.shape[0]
    if point_count == 0 or chunk_size <= 0 or point_count <= chunk_size:
        return chunk_kernel(points)

    full_chunk_count, tail_size = divmod(point_count, chunk_size)
    # Keep the default remat policy until CUDA profiling shows that saving dot
    # residuals is a better memory/runtime tradeoff for this kernel.
    remat_chunk_kernel = jax.checkpoint(chunk_kernel)
    first_chunk_points = _slice_point_chunk(points, 0, chunk_size)
    first_result = remat_chunk_kernel(first_chunk_points)
    result = _tree_dynamic_update(
        _tree_zeros_like_prefix(first_result, point_count),
        first_result,
        0,
    )

    def body(chunk_index: int, acc):
        start = chunk_index * chunk_size
        chunk_points = _slice_point_chunk(points, start, chunk_size)
        chunk_result = remat_chunk_kernel(chunk_points)
        return _tree_dynamic_update(acc, chunk_result, start)

    result = lax.fori_loop(1, full_chunk_count, body, result)
    if tail_size:
        tail_start = full_chunk_count * chunk_size
        tail_result = remat_chunk_kernel(points[tail_start:])
        result = _tree_dynamic_update(result, tail_result, tail_start)
    return result


# ── Physics integrands ────────────────────────────────────────────────


def _biot_savart_B_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = _radius_squared(diff)
    r_inv = lax.rsqrt(r2)
    r_inv3 = r_inv * jnp.reciprocal(r2)
    cross = _cross_product(diff, gammadashs)
    return cross * r_inv3[..., None]


def _biot_savart_A_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = _radius_squared(diff)
    r_inv = lax.rsqrt(r2)
    return gammadashs * r_inv[..., None]


def _biot_savart_B_and_dB_integrand(x, gammas, gammadashs):
    diff = x - gammas
    r2 = _radius_squared(diff)
    r_inv = lax.rsqrt(r2)
    r_inv3 = r_inv * jnp.reciprocal(r2)
    r_inv4 = r_inv3 * r_inv
    norm = r2 * r_inv

    cross = _cross_product(gammadashs, diff)
    B_integrand = cross * r_inv3[..., None]

    dg_norm = gammadashs * norm[..., None]
    zero = _zeros(diff.shape[:-1], dtype=diff.dtype)
    numerator0 = jnp.stack((zero, dg_norm[..., 2], -dg_norm[..., 1]), axis=-1)
    numerator1 = jnp.stack((-dg_norm[..., 2], zero, dg_norm[..., 0]), axis=-1)
    numerator2 = jnp.stack((dg_norm[..., 1], -dg_norm[..., 0], zero), axis=-1)

    three_cross_inv = cross * (3.0 * r_inv)[..., None]
    temp0 = numerator0 - three_cross_inv * diff[..., 0, None]
    temp1 = numerator1 - three_cross_inv * diff[..., 1, None]
    temp2 = numerator2 - three_cross_inv * diff[..., 2, None]
    dB_integrand = jnp.stack((temp0, temp1, temp2), axis=-2) * r_inv4[..., None, None]
    return B_integrand, dB_integrand


# ── Dense one-point kernels ──────────────────────────────────────────


class _Integrand(Enum):
    B = "B"
    A = "A"


class _DiffMode(Enum):
    VALUE = "value"
    JACOBIAN = "jacobian"
    HESSIAN = "hessian"
    VALUE_AND_JACOBIAN = "value_and_jacobian"


_INTEGRANDS = {
    _Integrand.B: _biot_savart_B_integrand,
    _Integrand.A: _biot_savart_A_integrand,
}


def _one_point_dense(
    x, gammas, gammadashs, currents, *, integrand, quadrature_block_size=0
):
    integral = _quadrature_block_integral(
        x,
        gammas,
        gammadashs,
        block_size=quadrature_block_size,
        integrand=integrand,
    )
    return _float64_scalar(currents, _MU0_OVER_4PI) * jnp.einsum(
        "c,cj->j", currents, integral, precision=lax.Precision.HIGHEST
    )


def _quadrature_block_B_and_dB_integral(x, gammas, gammadashs, *, block_size: int):
    quadrature_count = gammas.shape[1]

    def reduce_values(block_gammas, block_gammadashs):
        return jax.tree.map(
            lambda values: _pairwise_sum_axis(values, axis=1),
            _biot_savart_B_and_dB_integrand(x, block_gammas, block_gammadashs),
        )

    def average_sum(integral_sum):
        return jax.tree.map(lambda values: values / quadrature_count, integral_sum)

    if block_size <= 0 or quadrature_count <= block_size:
        return average_sum(reduce_values(gammas, gammadashs))

    block_count = (quadrature_count + block_size - 1) // block_size
    if block_count == 2:
        second_block_size = quadrature_count - block_size
        first = reduce_values(
            _slice_quadrature_block(gammas, 0, block_size),
            _slice_quadrature_block(gammadashs, 0, block_size),
        )
        second = reduce_values(
            _slice_quadrature_block(gammas, block_size, second_block_size),
            _slice_quadrature_block(gammadashs, block_size, second_block_size),
        )
        return average_sum(_tree_add(first, second))

    full_block_count, tail_size = divmod(quadrature_count, block_size)
    zero = (
        _zeros((gammas.shape[0], 3), dtype=jnp.float64),
        _zeros((gammas.shape[0], 3, 3), dtype=jnp.float64),
    )

    def body(block_index: int, acc):
        start = block_index * block_size
        block_gammas = _slice_quadrature_block(gammas, start, block_size)
        block_gammadashs = _slice_quadrature_block(
            gammadashs,
            start,
            block_size,
        )
        return _tree_add(acc, reduce_values(block_gammas, block_gammadashs))

    integral_sum = lax.fori_loop(0, full_block_count, body, zero)
    if tail_size:
        tail_start = full_block_count * block_size
        integral_sum = _tree_add(
            integral_sum,
            reduce_values(
                gammas[:, tail_start:],
                gammadashs[:, tail_start:],
            ),
        )
    return average_sum(integral_sum)


def _one_point_dense_B_and_dB(
    x, gammas, gammadashs, currents, *, quadrature_block_size
):
    B_integral, dB_integral = _quadrature_block_B_and_dB_integral(
        x,
        gammas,
        gammadashs,
        block_size=quadrature_block_size,
    )
    scale = _float64_scalar(currents, _MU0_OVER_4PI)
    B = scale * jnp.einsum(
        "c,cj->j",
        currents,
        B_integral,
        precision=lax.Precision.HIGHEST,
    )
    dB = scale * jnp.einsum(
        "c,cjl->jl",
        currents,
        dB_integral,
        precision=lax.Precision.HIGHEST,
    )
    return B, dB


# ── Kernel factory ────────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _make_kernel(
    integrand_key,
    diff_mode,
    coil_cs,
    quad_bs,
    point_cs,
):
    """Build and JIT-compile a Biot-Savart kernel for the given tiling."""
    integrand = _INTEGRANDS[integrand_key]

    def one_point(x, gammas, gammadashs, currents):
        return _coil_chunk_reduce(
            gammas,
            gammadashs,
            currents,
            chunk_size=coil_cs,
            zero=_zeros((3,), dtype=jnp.float64),
            reduce_chunk=lambda cg, cgd, cc: _one_point_dense(
                x,
                cg,
                cgd,
                cc,
                integrand=integrand,
                quadrature_block_size=quad_bs,
            ),
        )

    if diff_mode is _DiffMode.VALUE:
        per_point = one_point
    elif diff_mode is _DiffMode.JACOBIAN:

        def jacobian_point(x, gammas, gammadashs, currents):
            return jnp.swapaxes(
                jax.jacfwd(one_point, argnums=0)(x, gammas, gammadashs, currents),
                -1,
                -2,
            )
        per_point = jacobian_point

    elif diff_mode is _DiffMode.HESSIAN:

        def hessian_point(x, gammas, gammadashs, currents):
            # Transpose (1,2,0): raw jacfwd² yields [component, d1, d2],
            # upstream convention is [d1, d2, component].
            # jacfwd² (not jacrev∘jacfwd) preserves exact current-linearity.
            return jnp.transpose(
                jax.jacfwd(jax.jacfwd(one_point, argnums=0), argnums=0)(
                    x,
                    gammas,
                    gammadashs,
                    currents,
                ),
                (1, 2, 0),
            )
        per_point = hessian_point

    elif diff_mode is _DiffMode.VALUE_AND_JACOBIAN:
        # Fused B and dB/dX kernel, built for ``_Integrand.B`` only.

        def B_and_jacobian_point(x, gammas, gammadashs, currents):
            return _coil_chunk_reduce(
                gammas,
                gammadashs,
                currents,
                chunk_size=coil_cs,
                zero=(
                    _zeros((3,), dtype=jnp.float64),
                    _zeros((3, 3), dtype=jnp.float64),
                ),
                reduce_chunk=lambda cg, cgd, cc: _one_point_dense_B_and_dB(
                    x,
                    cg,
                    cgd,
                    cc,
                    quadrature_block_size=quad_bs,
                ),
            )
        per_point = B_and_jacobian_point

    kernel_point_cs = (
        hessian_point_chunk_size(point_cs)
        if diff_mode is _DiffMode.HESSIAN
        else point_cs
    )

    def kernel(points, gammas, gammadashs, currents):
        def chunk_fn(chunk_points):
            return jax.vmap(
                lambda x: per_point(x, gammas, gammadashs, currents),
                in_axes=(0,),
            )(chunk_points)

        return _point_chunk_reduce(points, chunk_fn, kernel_point_cs)

    kernel.__name__ = "biotsavart_forward"
    return jit(kernel)


def _get_kernel(integrand_key, diff_mode):
    """Return the cached JIT-compiled kernel at the public tiling."""
    return _make_kernel(
        integrand_key, diff_mode, COIL_CHUNK_SIZE, QUADRATURE_BLOCK_SIZE, POINT_CHUNK_SIZE
    )


@lru_cache(maxsize=64)
def _make_B_vjp_kernel(coil_cs, quad_bs, point_cs):
    """Build the compiled field pullback for the given tiling."""
    forward_kernel = _make_kernel(
        _Integrand.B,
        _DiffMode.VALUE,
        coil_cs,
        quad_bs,
        point_cs,
    )

    def kernel(points, v, gammas, gammadashs, currents):
        def fwd(group_gammas, group_gammadashs, group_currents):
            return forward_kernel(
                points,
                group_gammas,
                group_gammadashs,
                group_currents,
            )

        _, pullback = jax.vjp(fwd, gammas, gammadashs, currents)
        return pullback(v)

    kernel.__name__ = "biotsavart_vjp"
    return jit(kernel)


def _get_B_vjp_kernel():
    """Return the cached JIT-compiled VJP kernel at the public tiling."""
    return _make_B_vjp_kernel(COIL_CHUNK_SIZE, QUADRATURE_BLOCK_SIZE, POINT_CHUNK_SIZE)


def invalidate_kernel_cache() -> None:
    """Clear cached forward and pullback kernels, rebuilt on their next use."""
    _make_kernel.cache_clear()
    _make_B_vjp_kernel.cache_clear()


register_backend_cache_clear(invalidate_kernel_cache)


# ── Public API ────────────────────────────────────────────────────────


def _float64_inputs(*values):
    """Cast to float64, copying host inputs and reusing float64 JAX arrays."""
    return tuple(_as_jax_float64(value) for value in values)


def _apply_forward_kernel(
    integrand_key,
    diff_mode,
    points,
    gammas,
    gammadashs,
    currents,
):
    """Apply one cached field kernel to its inputs cast to float64."""
    inputs = _float64_inputs(points, gammas, gammadashs, currents)
    return _get_kernel(integrand_key, diff_mode)(*inputs)


def biot_savart_B(points, gammas, gammadashs, currents):
    """Evaluate the magnetic field for one rectangular coil group.

    Averages Q uniform samples of the normalized curve parameter using
    ``mu0/(4*pi) = 1e-7``; evaluation on a filament is singular.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Magnetic field in tesla, shape (P, 3), Cartesian component
            last.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.VALUE,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    """Return the magnetic field's spatial Jacobian.

    Averages Q uniform samples of the normalized curve parameter using
    ``mu0/(4*pi) = 1e-7``; evaluation on a filament is singular.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3), in tesla per meter; result[p, j, l] =
            partial_j B_l at point p.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_d2B_by_dXdX(points, gammas, gammadashs, currents):
    """Return the dense spatial Hessian of the magnetic field.

    This kernel requires large intermediates; use ``biot_savart_B_and_dB``
    when second spatial derivatives are not needed.

    Averages Q uniform samples of the normalized curve parameter using
    ``mu0/(4*pi) = 1e-7``; evaluation on a filament is singular.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3, 3), in tesla per meter squared; result[p,
            i, j, l] = partial_i partial_j B_l.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.HESSIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_B_and_dB(points, gammas, gammadashs, currents):
    """Return the magnetic field and its spatial Jacobian.

    Averages Q uniform samples of the normalized curve parameter using
    ``mu0/(4*pi) = 1e-7``; evaluation on a filament is singular.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        tuple[jax.Array, jax.Array]: B of shape (P, 3) in tesla and its
            Jacobian of shape (P, 3, 3) in tesla per meter, with derivative
            direction before field component.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.VALUE_AND_JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_A(points, gammas, gammadashs, currents):
    """Evaluate the vector potential for one rectangular coil group.

    Averages Q uniform samples of the normalized curve parameter using
    ``mu0/(4*pi) = 1e-7``; evaluation on a filament is singular.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Vector potential in tesla meters, shape (P, 3), Cartesian
            component last.
    """
    return _apply_forward_kernel(
        _Integrand.A,
        _DiffMode.VALUE,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_dA_by_dX(points, gammas, gammadashs, currents):
    """Return the vector potential's spatial Jacobian.

    Averages Q uniform samples of the normalized curve parameter using
    ``mu0/(4*pi) = 1e-7``; evaluation on a filament is singular.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3), in tesla; result[p, j, l] = partial_j A_l
            at point p.
    """
    return _apply_forward_kernel(
        _Integrand.A,
        _DiffMode.JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_d2A_by_dXdX(points, gammas, gammadashs, currents):
    """Return the vector potential's spatial Hessian for one coil group.

    Averages Q uniform samples of the normalized curve parameter using
    ``mu0/(4*pi) = 1e-7``; evaluation on a filament is singular.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3, 3), in tesla per meter; result[p, i, j, l]
            = partial_i partial_j A_l.
    """
    return _apply_forward_kernel(
        _Integrand.A,
        _DiffMode.HESSIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    """VJP of ``biot_savart_B`` w.r.t. coil data.

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
    inputs = _float64_inputs(points, v, gammas, gammadashs, currents)
    return _get_B_vjp_kernel()(*inputs)


# ── Grouped coil utilities ───────────────────────────────────────────


def group_coil_data(
    gammas_list,
    gammadashs_list,
    currents_list,
):
    """Batch coils by quadrature count while preserving original index mappings.

    Args:
        gammas_list (Sequence[array-like]): Per-coil positions, each shape (Q_i, 3), in
            meters.
        gammadashs_list (Sequence[array-like]): Matching parameter derivatives, each
            shape (Q_i, 3), in meters.
        currents_list (Sequence[scalar]): Matching coil currents in amperes, each scalar
            shape ().

    Returns:
        list[tuple]: One (gammas, gammadashs, currents, indices) tuple per
            quadrature count, with shapes (C_g, Q_g, 3), (C_g, Q_g, 3), (C_g,) and
            original index lists. Groups are ordered by their first coil index.
    """
    indices_by_nquad = {}
    for i, gamma in enumerate(gammas_list):
        indices_by_nquad.setdefault(gamma.shape[0], []).append(i)

    groups = []
    for indices in sorted(indices_by_nquad.values(), key=lambda group: group[0]):
        groups.append(
            (
                jnp.stack([_as_jax_float64(gammas_list[i]) for i in indices]),
                jnp.stack([_as_jax_float64(gammadashs_list[i]) for i in indices]),
                jnp.stack([_as_jax_float64(currents_list[i]) for i in indices]),
                indices,
            )
        )
    return groups
