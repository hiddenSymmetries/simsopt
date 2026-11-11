"""JIT-compilable, autodiff-compatible Biot-Savart kernels for CPU and GPU.

Public kernels cast inputs to float64, compute in float64 and return JAX
arrays without requiring a native field object.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from types import MappingProxyType

import jax
from simsopt_jax.backend.dtypes import _uncommitted_default_device
from simsopt_jax.backend.runtime import register_backend_cache_clear
from simsopt_jax.backend.jit import jit
from jax import core as jax_core
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
# Point tile of the d2B pullback ``biot_savart_d2B_by_dXdX_vjp`` by the platform
# of the device the call runs on, independent of ``POINT_CHUNK_SIZE``; the
# factory ``_make_d2B_vjp_kernel`` takes any other size. Measured with 16 coils
# x 128 nodes: GPU 2048 (RTX 5090, 16384 points) took 74 ms at 0.91 GB peak,
# versus 84 ms at 0.54 GB for 512 and 70 ms at 1.82 GB for 4096; CPU 128 (4096
# points, 4 threads) took 115 ms, versus 360 ms for 8, 253 ms for 32 and 240 ms
# for 512. Live memory is about 220 bytes per (point, coil node) pair in a tile.
HESSIAN_VJP_POINT_CHUNK_SIZE = MappingProxyType({"cpu": 128, "gpu": 2048})


# ── Array slicing primitives ──────────────────────────────────────────


def _slice_block(array, axis: int, start, size: int):
    """Return ``size`` entries of ``array`` along ``axis`` from ``start`` on."""
    zero = _as_int32_scalar(0)
    return lax.dynamic_slice(
        array,
        (zero,) * axis + (_as_int32_scalar(start),) + (zero,) * (array.ndim - axis - 1),
        array.shape[:axis] + (size,) + array.shape[axis + 1 :],
    )


def _slice_point_chunk(points, start, chunk_size: int):
    return _slice_block(points, 0, start, chunk_size)


def _float64_scalar(reference, value):
    return _device_one(reference) * np.float64(value)


def _as_int32_scalar(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _radius_squared(diff):
    """Return squared separation without regularizing point-on-coil singularities."""

    return jnp.sum(diff * diff, axis=-1)


def _components(vectors):
    """Return the x, y and z components of ``vectors``, shape (..., 3)."""
    return vectors[..., 0], vectors[..., 1], vectors[..., 2]


def _cross_components(left, right):
    """Return the components of ``left x right`` from component tuples."""
    left_x, left_y, left_z = left
    right_x, right_y, right_z = right
    return (
        left_y * right_z - left_z * right_y,
        left_z * right_x - left_x * right_z,
        left_x * right_y - left_y * right_x,
    )


def _cross_product(left, right):
    return jnp.stack(_cross_components(_components(left), _components(right)), axis=-1)


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


def _block_sum(arrays, *, axis: int, block_size: int, zero, reduce_block):
    """Sum block reductions of ``arrays`` along ``axis``.

    A nonpositive ``block_size`` or one covering the axis uses a single block.
    """
    count = arrays[0].shape[axis]
    if block_size <= 0 or count <= block_size:
        return reduce_block(*arrays)
    if count <= 2 * block_size:
        first = tuple(_slice_block(a, axis, 0, block_size) for a in arrays)
        second = tuple(
            _slice_block(a, axis, block_size, count - block_size) for a in arrays
        )
        return _tree_add(reduce_block(*first), reduce_block(*second))

    full_block_count, tail_size = divmod(count, block_size)

    def body(block_index: int, acc):
        start = block_index * block_size
        block = (_slice_block(a, axis, start, block_size) for a in arrays)
        return _tree_add(acc, reduce_block(*block))

    total = lax.fori_loop(0, full_block_count, body, zero)
    if tail_size:
        tail = (slice(None),) * axis + (slice(full_block_count * block_size, None),)
        total = _tree_add(total, reduce_block(*(a[tail] for a in arrays)))
    return total


def _coil_chunk_reduce(
    gammas, gammadashs, currents, *, chunk_size: int, zero, reduce_chunk
):
    """Sum coil-chunk reductions with a concrete coil count.

    The coil count is the shared leading axis of the geometry and currents.
    """
    if currents.shape[0] == 0:
        return zero
    return _block_sum(
        (gammas, gammadashs, currents),
        axis=0,
        block_size=chunk_size,
        zero=zero,
        reduce_block=reduce_chunk,
    )


def _quadrature_block_sum(
    x, gammas, gammadashs, currents, *, block_size: int, integrand
):
    """Return current-weighted quadrature sums for x, y and z."""

    def block_sum(block_gammas, block_gammadashs):
        # Contract each block's per-coil sums with the currents before any stack,
        # block add or 1/Q scale; otherwise XLA fuses them with the reduce (A100,
        # dB/dX: the reduce moves into the stack's concatenate fusion; XLA:CPU:
        # the add or scale becomes its epilogue, about 2x slower).
        return tuple(
            jnp.einsum(
                "c,c->",
                currents,
                _pairwise_sum_axis(component, axis=1),
                precision=lax.Precision.HIGHEST,
            )
            for component in integrand(x, block_gammas, block_gammadashs)
        )

    return _block_sum(
        (gammas, gammadashs),
        axis=1,
        block_size=block_size,
        zero=(_zeros((), dtype=jnp.float64),) * 3,
        reduce_block=block_sum,
    )


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

# The B and A integrands are component-major: three (C, Qb) arrays, no length-3
# axis. Since jaxlib 0.10.1, XLA's GPU layout assignment can make that axis the
# minor dimension of a per-coil reduction that sums another axis, which runs
# several times slower (openxla/xla#52437).


def _separation(x, gammas):
    """Return components of ``gammas - x`` and its unregularized squared length.

    For ``x`` of shape (3,) and ``gammas`` of shape (C, Qb, 3), each returned
    array has shape (C, Qb).
    """
    dx, dy, dz = (
        gamma - point for gamma, point in zip(_components(gammas), _components(x))
    )
    return (dx, dy, dz), dx * dx + dy * dy + dz * dz


def _biot_savart_B_integrand(x, gammas, gammadashs):
    diff, r2 = _separation(x, gammas)
    r_inv = lax.rsqrt(r2)
    r_inv3 = r_inv * jnp.reciprocal(r2)
    return tuple(
        component * r_inv3
        for component in _cross_components(diff, _components(gammadashs))
    )


def _biot_savart_A_integrand(x, gammas, gammadashs):
    _, r2 = _separation(x, gammas)
    r_inv = lax.rsqrt(r2)
    return tuple(tangent * r_inv for tangent in _components(gammadashs))


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
    # Stack, block add and 1/Q scale act on contracted sums; see _quadrature_block_sum.
    weighted_sum = _quadrature_block_sum(
        x,
        gammas,
        gammadashs,
        currents,
        block_size=quadrature_block_size,
        integrand=integrand,
    )
    return _float64_scalar(currents, _MU0_OVER_4PI / gammas.shape[1]) * jnp.stack(
        weighted_sum
    )


def _quadrature_block_B_and_dB_integral(x, gammas, gammadashs, *, block_size: int):
    # _block_sum's three paths, written out: its two-block split slices both
    # blocks before reducing either, which would change this kernel's program.
    # This fused kernel also still adds the blocks and applies 1/Q before
    # contracting with the currents.
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
            _slice_block(gammas, 1, 0, block_size),
            _slice_block(gammadashs, 1, 0, block_size),
        )
        second = reduce_values(
            _slice_block(gammas, 1, block_size, second_block_size),
            _slice_block(gammadashs, 1, block_size, second_block_size),
        )
        return average_sum(_tree_add(first, second))

    full_block_count, tail_size = divmod(quadrature_count, block_size)
    zero = (
        _zeros((gammas.shape[0], 3), dtype=jnp.float64),
        _zeros((gammas.shape[0], 3, 3), dtype=jnp.float64),
    )

    def body(block_index: int, acc):
        start = block_index * block_size
        block_gammas = _slice_block(gammas, 1, start, block_size)
        block_gammadashs = _slice_block(gammadashs, 1, start, block_size)
        return _tree_add(acc, reduce_values(block_gammas, block_gammadashs))

    integral_sum = lax.fori_loop(0, full_block_count, body, zero)
    if tail_size:
        tail_start = full_block_count * block_size
        integral_sum = _tree_add(
            integral_sum,
            reduce_values(gammas[:, tail_start:], gammadashs[:, tail_start:]),
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


# The six symmetric direction pairs ``(j, k)``, ``j <= k``, of a Hessian seed.
_SYMMETRIC_DIRECTION_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _symmetric_hessian_seed(vgradgrad):
    """Combine Hessian seeds into six symmetric pairs, adding off-diagonal entries."""
    return jnp.stack(
        [
            vgradgrad[:, j, k] if j == k else vgradgrad[:, j, k] + vgradgrad[:, k, j]
            for j, k in _SYMMETRIC_DIRECTION_PAIRS
        ],
        axis=1,
    )


def _pair_B_components(x, gamma, gammadash, current, quadrature_count):
    """Return one coil node's magnetic-field contribution at one point."""
    d0, d1, d2 = gamma[0] - x[0], gamma[1] - x[1], gamma[2] - x[2]
    r_inv = lax.rsqrt(d0 * d0 + d1 * d1 + d2 * d2)
    weight = (_MU0_OVER_4PI / quadrature_count) * current * (r_inv * r_inv * r_inv)
    return (
        (d1 * gammadash[2] - d2 * gammadash[1]) * weight,
        (d2 * gammadash[0] - d0 * gammadash[2]) * weight,
        (d0 * gammadash[1] - d1 * gammadash[0]) * weight,
    )


def _replace_scalar(values, index, value):
    return tuple(value if position == index else values[position] for position in range(len(values)))


def _scalar_partial(fn, values, index):
    """Return the function value and its partial for ``values[index]``."""
    return jax.jvp(
        lambda value: fn(_replace_scalar(values, index, value)),
        (values[index],),
        (jnp.ones_like(values[index]),),
    )


def _scalar_second_partial(fn, values, j, k):
    return jax.jvp(
        lambda value: _scalar_partial(fn, _replace_scalar(values, j, value), k)[1],
        (values[j],),
        (jnp.ones_like(values[j]),),
    )[1]


def _pair_hessian_contraction(seed, x, local, quadrature_count):
    """Contract the seed with six symmetric field-Hessian pairs at one coil node."""
    gamma, gammadash, current = local[0:3], local[3:6], local[6]

    def field(y):
        return _pair_B_components(y, gamma, gammadash, current, quadrature_count)

    return sum(
        seed[pair, 0] * second[0] + seed[pair, 1] * second[1] + seed[pair, 2] * second[2]
        for pair, second in enumerate(
            _scalar_second_partial(field, x, j, k) for j, k in _SYMMETRIC_DIRECTION_PAIRS
        )
    )


def _pair_hessian_pullback(seed, x0, x1, x2, g0, g1, g2, t0, t1, t2, current, *, quadrature_count):
    """Return the pair contraction's derivatives with respect to seven local scalars."""
    local = (g0, g1, g2, t0, t1, t2, current)

    def contraction(values):
        return _pair_hessian_contraction(seed, (x0, x1, x2), values, quadrature_count)

    return tuple(_scalar_partial(contraction, local, index)[1] for index in range(7))


@lru_cache(maxsize=64)
def _make_d2B_vjp_kernel(hessian_vjp_point_cs):
    """Build the float64 magnetic-field Hessian pullback with point tiling.

    A nonpositive ``hessian_vjp_point_cs`` evaluates all points in one tile.
    """
    chunk_size = hessian_vjp_point_cs

    def float64_kernel(points, vgradgrad, gammas, gammadashs, currents):
        quadrature_count = gammas.shape[1]
        pair_pullback = jax.vmap(
            lambda *args: _pair_hessian_pullback(*args, quadrature_count=quadrature_count),
            in_axes=(None, None, None, None, 0, 0, 0, 0, 0, 0, None),
        )
        coil_pullback = jax.vmap(pair_pullback, in_axes=(None,) * 4 + (0,) * 7)
        point_pullback = jax.vmap(coil_pullback, in_axes=(0,) * 4 + (None,) * 7)

        def tile_node_sums(tile_points, tile_seed):
            derivatives = point_pullback(
                tile_seed,
                tile_points[:, 0], tile_points[:, 1], tile_points[:, 2],
                gammas[..., 0], gammas[..., 1], gammas[..., 2],
                gammadashs[..., 0], gammadashs[..., 1], gammadashs[..., 2],
                currents,
            )
            sums = tuple(jnp.sum(derivative, axis=0) for derivative in derivatives)
            return jnp.stack(sums[0:3], axis=-1), jnp.stack(sums[3:6], axis=-1), sums[6]

        point_count = points.shape[0]
        if point_count == 0:
            return jax.tree.map(jnp.zeros_like, (gammas, gammadashs, currents))
        seed = _symmetric_hessian_seed(vgradgrad)
        if chunk_size <= 0 or point_count <= chunk_size:
            node_sums = tile_node_sums(points, seed)
        else:
            full_chunk_count, tail_size = divmod(point_count, chunk_size)

            def tile_at(start):
                return tile_node_sums(
                    _slice_point_chunk(points, start, chunk_size),
                    _slice_point_chunk(seed, start, chunk_size),
                )

            node_sums = lax.fori_loop(
                1,
                full_chunk_count,
                lambda chunk_index, sums: _tree_add(sums, tile_at(chunk_index * chunk_size)),
                tile_at(0),
            )
            if tail_size:
                tail_start = full_chunk_count * chunk_size
                node_sums = _tree_add(
                    node_sums, tile_node_sums(points[tail_start:], seed[tail_start:]),
                )
        dgammas, dgammadashs, dcurrent_nodes = node_sums
        return dgammas, dgammadashs, _pairwise_sum_axis(dcurrent_nodes, axis=1)

    def kernel(points, vgradgrad, gammas, gammadashs, currents):
        return float64_kernel(*(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (points, vgradgrad, gammas, gammadashs, currents)
        ))

    kernel.__name__ = "biotsavart_d2B_vjp"
    return jit(kernel)


def invalidate_kernel_cache() -> None:
    """Clear cached forward and pullback kernels, rebuilt on their next use."""
    _make_kernel.cache_clear()
    _make_B_vjp_kernel.cache_clear()
    _make_d2B_vjp_kernel.cache_clear()


register_backend_cache_clear(invalidate_kernel_cache)


# ── Public API ────────────────────────────────────────────────────────


def _float64_inputs(*values):
    """Cast to float64, copying host inputs and reusing float64 JAX arrays."""
    return tuple(_as_jax_float64(value) for value in values)


def _call_device(arrays):
    """Return the committed input device, or JAX's default device if none."""
    committed = (array for array in arrays if not isinstance(array, jax_core.Tracer) and array.committed)
    return next((next(iter(array.devices())) for array in committed), None) or _uncommitted_default_device()


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


def biot_savart_d2B_by_dXdX_vjp(points, vgradgrad, gammas, gammadashs, currents):
    """Pull back a magnetic-field Hessian seed to coil geometry and currents.

    The seed need not be symmetric; points are held fixed.

    Args:
        points (jax.Array or numpy.ndarray): Cartesian positions of shape
            ``(npoints, 3)``, in meters.
        vgradgrad (jax.Array or numpy.ndarray): Hessian cotangent of shape
            ``(npoints, 3, 3, 3)`` in ``[point, d1, d2, component]`` order.
            For a scalar objective with units U, the seed has units U m²/T.
        gammas (jax.Array or numpy.ndarray): Coil positions of shape
            ``(ncoils, nquad, 3)``, in meters.
        gammadashs (jax.Array or numpy.ndarray): Coil tangents of shape
            ``(ncoils, nquad, 3)``, in meters per unit dimensionless curve
            parameter; quadrature uses uniformly spaced nodes on [0, 1).
        currents (jax.Array or numpy.ndarray): Coil currents of shape
            ``(ncoils,)``, in amperes.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Cotangents of ``gammas``,
            ``gammadashs`` and ``currents``, with their respective input shapes
            ``(ncoils, nquad, 3)``, ``(ncoils, nquad, 3)`` and ``(ncoils,)``.
            For an objective in U, geometry cotangents have units U/m and current
            cotangents have units U/A.
    """
    inputs = _float64_inputs(points, vgradgrad, gammas, gammadashs, currents)
    platform = _call_device(inputs).platform
    return _make_d2B_vjp_kernel(HESSIAN_VJP_POINT_CHUNK_SIZE[platform])(*inputs)


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
