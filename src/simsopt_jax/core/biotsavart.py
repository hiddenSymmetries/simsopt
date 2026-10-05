"""
Pure JAX implementation of the Biot-Savart magnetic field computation.

This module provides JIT-compilable, autodiff-compatible functions that
evaluate Biot-Savart on CPU or GPU without a native field object.

All functions accept and return JAX arrays and are fully traceable
by ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.hessian``.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import cast

import jax
from simsopt_jax.backend import get_field_kernel_tuning, register_backend_cache_clear
from jax import lax
import jax.numpy as jnp
import numpy as np


from ._device_scalars import device_one as _device_one
from ._math_utils import (
    as_compute_array as _as_compute_array,
    as_jax_float64 as _as_jax_float64,
    axis0_entries as _axis0_entries,
    explicit_inv as _explicit_inv,
    explicit_rsqrt as _explicit_rsqrt,
    eye as _eye,
    scalar_like as _scalar_like,
    zeros as _zeros,
)
from .reductions import pairwise_sum_axis as _pairwise_sum_axis

__all__ = [
    "biot_savart_B",
    "biot_savart_B_vjp",
    "biot_savart_dB_by_dX",
    "biot_savart_d2B_by_dXdX",
    "biot_savart_B_and_dB",
    "biot_savart_B_and_dB_with_point_axis",
    "biot_savart_A",
    "biot_savart_dA_by_dX",
    "biot_savart_d2A_by_dXdX",
    "group_coil_data",
    "invalidate_kernel_cache",
]

_MU0_OVER_4PI = 1e-7


# ── Config reader (monkeypatchable by tests) ──────────────────────────


def _read_tuning_config() -> tuple:
    """Return ``(coil_chunk_size, quadrature_block_size, point_chunk_size)``.

    Single indirection point for all tuning knobs consumed by the kernel
    factory.  Tests override this one function (+ ``invalidate_kernel_cache``)
    instead of patching three separate stubs.

    Resolves the backend config once to avoid repeated mode/policy lookups.
    """
    fkt = get_field_kernel_tuning()
    return fkt.coil_chunk_size, fkt.quadrature_block_size, fkt.point_chunk_size


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
    """Return exact ``r^2`` so point-on-coil singularities stay loud."""

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
                lax.dynamic_slice(currents, (0,), (chunk_size,)),
            ),
            (
                _slice_coil_chunk(gammas, chunk_size, second_chunk_size),
                _slice_coil_chunk(gammadashs, chunk_size, second_chunk_size),
                lax.dynamic_slice(currents, (chunk_size,), (second_chunk_size,)),
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
        chunk_currents = lax.dynamic_slice(
            currents,
            (start,),
            (chunk_size,),
        )
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
        return _pairwise_sum_axis(values, axis=1) * _scalar_like(
            values,
            1.0 / values.shape[1],
        )

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
# both tangent levels live.  The tuning table sizes ``point_chunk_size`` for
# the ``B``/``B+dB`` hot paths, so the Hessian kernel takes one eighth of it:
# 2048 points at 18 NCSX coils x 250 nodes otherwise materializes >20 GB and
# fails on a 32 GB device, while 256 points stay near 2 GB.
_HESSIAN_POINT_CHUNK_DIVISOR = 8


def hessian_point_chunk_size(point_chunk_size: int) -> int:
    """Point tile for ``d2B/dXdX`` derived from the ``B``-sized tuning value.

    ``0`` (tiling disabled) is preserved so audit policies that pin every
    chunk to the dense layout keep their contract; any positive size is
    divided by ``_HESSIAN_POINT_CHUNK_DIVISOR`` and floored at one point.
    """
    if point_chunk_size <= 0:
        return 0
    return max(1, point_chunk_size // _HESSIAN_POINT_CHUNK_DIVISOR)


def _point_chunk_reduce(points, chunk_kernel, chunk_size):
    """Evaluate *chunk_kernel* over *points* with optional point-axis tiling.

    Three-arg signature: ``chunk_size`` is passed explicitly by the kernel
    factory (not read from module state) so that the value is part of the
    ``lru_cache`` key and triggers recompilation when it changes.
    """
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
    r_inv = _explicit_rsqrt(r2)
    r_inv3 = r_inv * _explicit_inv(r2)
    cross = _cross_product(diff, gammadashs)
    return cross * r_inv3[..., None]


def _biot_savart_A_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = _radius_squared(diff)
    r_inv = _explicit_rsqrt(r2)
    return gammadashs * r_inv[..., None]


def _biot_savart_B_and_dB_integrand(x, gammas, gammadashs):
    diff = x - gammas
    r2 = _radius_squared(diff)
    r_inv = _explicit_rsqrt(r2)
    r_inv3 = r_inv * _explicit_inv(r2)
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
    point_vma_axis_name,
):
    """Build and JIT-compile a Biot-Savart kernel for the given tuning config.

    All tiling parameters are captured in closures — callers never thread them.
    ``lru_cache`` ensures the same config returns the same compiled function.

    Cache keyed on ``(integrand_key, diff_mode, coil_cs, quad_bs, point_cs,
    point_vma_axis_name)``.  JAX owns per-backend executable caching below the
    Python closure, so backend identity is not part of this LRU key.
    """
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

    elif diff_mode is _DiffMode.VALUE_AND_JACOBIAN and integrand_key is _Integrand.B:

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

    elif diff_mode is _DiffMode.VALUE_AND_JACOBIAN:

        def value_and_jacobian_point(x, gammas, gammadashs, currents):
            f = lambda xx: one_point(xx, gammas, gammadashs, currents)
            primals, tangents_fn = jax.linearize(f, x)
            basis = _eye(3, dtype=jnp.float64)
            if point_vma_axis_name is not None:
                basis = lax.pcast(basis, point_vma_axis_name, to="varying")
            return primals, jax.vmap(tangents_fn, in_axes=(0,))(basis)
        per_point = value_and_jacobian_point

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
    return jax.jit(kernel)


def _get_kernel(integrand_key, diff_mode, *, point_vma_axis_name=None):
    """Read tuning config and return the cached JIT-compiled kernel."""
    coil_cs, quad_bs, point_cs = _read_tuning_config()
    return _make_kernel(
        integrand_key,
        diff_mode,
        coil_cs,
        quad_bs,
        point_cs,
        point_vma_axis_name,
    )


@lru_cache(maxsize=64)
def _make_B_vjp_kernel(coil_cs, quad_bs, point_cs):
    """Build the tuning-keyed compiled VJP kernel for ``biot_savart_B``.

    ``biot_savart_B_vjp`` differentiates through the forward field kernel, so it
    must be rebuilt when the backend tuning changes in the same process.
    Keying this factory on the same tuning tuple as the forward kernels keeps
    the backend/performance contract consistent across both paths.
    """
    forward_kernel = _make_kernel(
        _Integrand.B,
        _DiffMode.VALUE,
        coil_cs,
        quad_bs,
        point_cs,
        None,
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
    return jax.jit(kernel)


def _get_B_vjp_kernel():
    """Read tuning config and return the cached JIT-compiled VJP kernel."""
    coil_cs, quad_bs, point_cs = _read_tuning_config()
    return _make_B_vjp_kernel(coil_cs, quad_bs, point_cs)


def invalidate_kernel_cache() -> None:
    """Drop all cached JIT-compiled Biot-Savart kernels and tuning config.

    Call after overriding ``_read_tuning_config`` (e.g. via ``monkeypatch``)
    to ensure the next ``biot_savart_*`` call rebuilds with the new config.
    """
    _make_kernel.cache_clear()
    _make_B_vjp_kernel.cache_clear()


register_backend_cache_clear(invalidate_kernel_cache)


# ── Public API ────────────────────────────────────────────────────────


def _apply_forward_kernel(
    integrand_key,
    diff_mode,
    points,
    gammas,
    gammadashs,
    currents,
    *,
    point_vma_axis_name=None,
):
    """Apply one cached field kernel under the canonical forward trace scope."""

    return _get_kernel(
        integrand_key,
        diff_mode,
        point_vma_axis_name=point_vma_axis_name,
    )(points, gammas, gammadashs, currents)


def biot_savart_B(points, gammas, gammadashs, currents):
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.VALUE,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    """First spatial gradient of the Biot-Savart magnetic field.

    Returns
    -------
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, j, l] = ∂_j B_l(x_p)``. Axis 1 is the spatial derivative
        direction; axis 2 is the B-field component.
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
    """Return the dense Hessian of ``B`` with shape ``(npoints, 3, 3, 3)``.

    This is an opt-in diagnostics kernel: each point materializes three dense
    derivative planes per field component, so callers should prefer
    ``biot_savart_B_and_dB`` unless second point derivatives are required.
    The pre-reduction integrand is large: for example, ``P_chunk=512``,
    ``C=16``, ``Q=128`` materializes roughly 226 MB (216 MiB) of Hessian
    intermediates before quadrature reduction.
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
    """Return ``(B, dB_by_dX)`` for the Biot-Savart field.

    Returns
    -------
    B : jax.Array
        Shape ``(n_points, 3)``.
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, j, l] = ∂_j B_l(x_p)``. Axis 1 is the spatial derivative
        direction; axis 2 is the B-field component.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.VALUE_AND_JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_B_and_dB_with_point_axis(
    points,
    gammas,
    gammadashs,
    currents,
    point_axis_name: str,
):
    """``biot_savart_B_and_dB`` with a named vmap axis over points.

    Returns
    -------
    B : jax.Array
        Shape ``(n_points, 3)``.
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, j, l] = ∂_j B_l(x_p)``. Axis 1 is the spatial derivative
        direction; axis 2 is the B-field component.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.VALUE_AND_JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
        point_vma_axis_name=point_axis_name,
    )


def biot_savart_A(points, gammas, gammadashs, currents):
    return _apply_forward_kernel(
        _Integrand.A,
        _DiffMode.VALUE,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_dA_by_dX(points, gammas, gammadashs, currents):
    """First spatial gradient of the Biot-Savart vector potential.

    Returns
    -------
    dA : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dA[p, j, l] = ∂_j A_l(x_p)``. Axis 1 is the spatial derivative
        direction; axis 2 is the A-field component.
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

    Returns raw cotangents for ``(gammas, gammadashs, currents)``. The CPU
    ``BiotSavart.B_vjp`` wrapper instead pushes geometry and current
    cotangents through each ``Coil.vjp(...)`` and returns a ``Derivative``
    over coil/current dofs, so downstream comparisons must combine the raw
    JAX leaves through the same coil mapping.

    Uses a dedicated tuning-keyed kernel factory so backend mode and chunking
    changes rebuild the compiled closure in the same process, matching the
    cache invalidation behavior of the forward ``biot_savart_B`` kernels.
    """
    return _get_B_vjp_kernel()(points, v, gammas, gammadashs, currents)


# ── Grouped coil utilities ───────────────────────────────────────────


def _coil_entry_sequence(values: object) -> tuple[jax.Array, ...]:
    if isinstance(values, tuple):
        return cast(tuple[jax.Array, ...], values)
    if isinstance(values, list):
        return cast(tuple[jax.Array, ...], tuple(values))
    return _axis0_entries(values)


def group_coil_data(
    gammas_list,
    gammadashs_list,
    currents_list,
    *,
    use_compute_dtype: bool = True,
):
    gamma_entries = _coil_entry_sequence(gammas_list)
    gammadash_entries = _coil_entry_sequence(gammadashs_list)
    current_entries = _coil_entry_sequence(currents_list)
    indices_by_nquad = {}
    for i, gamma in enumerate(gamma_entries):
        indices_by_nquad.setdefault(gamma.shape[0], []).append(i)

    array_for_group = _as_compute_array if use_compute_dtype else _as_jax_float64
    groups = []
    for indices in sorted(indices_by_nquad.values(), key=lambda group: group[0]):
        groups.append(
            (
                jnp.stack([array_for_group(gamma_entries[i]) for i in indices]),
                jnp.stack([array_for_group(gammadash_entries[i]) for i in indices]),
                jnp.stack([array_for_group(current_entries[i]) for i in indices]),
                indices,
            )
        )
    return groups
