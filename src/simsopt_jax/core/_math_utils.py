"""Placement-aware array and scalar helpers, and the fixed-order pairwise sum, for pure kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.backend.dtypes import (
    _shape_tuple,
    as_jax_array,
    as_jax_float64,
    as_jax_int32,
    explicit_device_array as _explicit_device_array,
    runtime_device_put,
)

from ._device_scalars import placement_zero

__all__ = [
    "_explicit_device_array",
    "as_jax_array",
    "as_jax_float64",
    "as_jax_int32",
    "eye",
    "pad_axis",
    "pairwise_sum_axis",
    "runtime_device_put",
    "zero_padding_like",
    "zeros",
]


def zero_padding_like(array, *, axis: int, pad_width: int):
    """Construct trailing padding with the input dtype and execution placement.

    Args:
        array (jax.Array): Input of any nonscalar shape supplying dtype and
            placement; its values are not read.
        axis (int): Axis to replace with a padding extent; negative axes count from the
            end.
        pad_width (int): Nonnegative number of padding entries.

    Returns:
        jax.Array: Exact zeros with the input shape except for axis extent
            pad_width, including for nonfinite or overflowing inputs.
    """
    axis_index = int(axis) if axis >= 0 else array.ndim + int(axis)
    target_shape = (
        array.shape[:axis_index] + (int(pad_width),) + array.shape[axis_index + 1 :]
    )
    return jnp.broadcast_to(placement_zero(array), target_shape)


def pad_axis(array, *, axis: int, padded_size: int):
    """Extend an array axis to the requested size without truncating it.

    Args:
        array (jax.Array): Input of any nonscalar shape.
        axis (int): Axis to extend; negative axes count from the end.
        padded_size (int): Desired axis extent; sizes no greater than the current extent
            return the original input.

    Returns:
        jax.Array: Input with trailing zeros, retaining dtype and placement;
            the axis grows to padded_size if larger than its current extent.
    """
    axis_index = int(axis) if axis >= 0 else array.ndim + int(axis)
    pad_width = int(padded_size) - int(array.shape[axis_index])
    if pad_width <= 0:
        return array
    return jnp.concatenate(
        (
            array,
            zero_padding_like(array, axis=axis_index, pad_width=pad_width),
        ),
        axis=axis_index,
    )


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
        array (jax.Array): Input of any nonscalar shape.
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
    reduced = pad_axis(reduced, axis=0, padded_size=_next_power_of_two(axis_size))
    return jnp.squeeze(_pairwise_reduce_axis0(reduced), axis=0)


def zeros(shape, dtype=jnp.float64) -> jax.Array:
    """Allocate an explicitly placed zero array of the requested dtype.

    Args:
        shape (int or Sequence[int]): Output dimensions; an integer requests a one-
            dimensional array.
        dtype (dtype-like): Exact output dtype; defaults to float64.

    Returns:
        jax.Array: Zero array of the requested shape and dtype at
            default/runtime placement.
    """
    return _explicit_device_array(
        np.zeros(_shape_tuple(shape), dtype=np.dtype(dtype)),
        dtype=dtype,
    )


def eye(size: int, dtype=jnp.float64) -> jax.Array:
    """Allocate an explicitly placed identity matrix.

    Args:
        size (int): Number of rows and columns.
        dtype (dtype-like): Exact output dtype; defaults to float64.

    Returns:
        jax.Array: Identity matrix, shape (size, size), at default/runtime
            placement.
    """
    return _explicit_device_array(
        np.eye(int(size), dtype=np.dtype(dtype)),
        dtype=dtype,
    )
