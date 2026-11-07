"""Shared reduction helpers for parity-sensitive JAX kernels."""

from __future__ import annotations

import jax.numpy as jnp

from ._math_utils import pad_axis as _pad_axis


__all__ = [
    "pairwise_sum_axis",
]


def _next_power_of_two(size: int) -> int:
    if size <= 1:
        return 1
    return 1 << (size - 1).bit_length()


def _pairwise_reduce_axis0(array):
    reduced = array
    while reduced.shape[0] > 1:
        pair_shape = (reduced.shape[0] // 2, 2) + tuple(reduced.shape[1:])
        paired = jnp.reshape(reduced, pair_shape)
        reduced = jnp.sum(paired, axis=1)
    return reduced


def pairwise_sum_axis(array, *, axis: int):
    """Reduce ``array`` along ``axis`` using a fixed binary addition tree.

    Args:
        array (jax.Array): Input of any nonscalar shape; finite values are assumed for
            zero padding.
        axis (int): Reduction axis; negative values count from the end.

    Returns:
        jax.Array: Sum of shape equal to the input shape with axis removed,
            retaining the input dtype for nonempty reductions; empty axes use
            jnp.sum dtype rules.
    """
    axis_index = axis if axis >= 0 else array.ndim + axis
    axis_size = array.shape[axis_index]
    if axis_size == 0:
        return jnp.sum(array, axis=axis_index)

    reduced = jnp.moveaxis(array, axis_index, 0)
    reduced = _pad_axis(reduced, axis=0, padded_size=_next_power_of_two(axis_size))
    return jnp.squeeze(_pairwise_reduce_axis0(reduced), axis=0)
