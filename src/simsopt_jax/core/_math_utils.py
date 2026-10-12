"""Placement-aware array and scalar helpers for pure field and curve kernels."""

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

__all__ = [
    "_explicit_device_array",
    "as_jax_array",
    "as_jax_float64",
    "as_jax_int32",
    "axis0_entries",
    "explicit_inv",
    "explicit_rsqrt",
    "eye",
    "iter_axis0_entries",
    "pad_axis",
    "runtime_device_put",
    "scalar_like",
    "zero_padding_like",
    "zeros",
]


def iter_axis0_entries(array):
    """Yield axis-0 slices from a shaped JAX array.

    Args:
        array (array-like): Input of shape (N, ...); axis0_entries also accepts a
            scalar.

    Returns:
        jax.Array: Successive slices of shape array.shape[1:] in axis-zero
            order; an empty leading axis yields no entries.
    """
    axis0_size = int(array.shape[0])
    if axis0_size == 0:
        return
    for entry in jnp.split(array, axis0_size, axis=0):
        yield jnp.squeeze(entry, axis=0)


def axis0_entries(array: object) -> tuple[jax.Array, ...]:
    """Return axis-zero entries after float64 conversion.

    Args:
        array (array-like): Input of shape (N, ...); axis0_entries also accepts a
            scalar.

    Returns:
        tuple[jax.Array, ...]: Axis-zero slices with shape array.shape[1:]; a
            scalar becomes a one-entry tuple of scalar shape ().
    """
    array_jax = as_jax_float64(array)
    if array_jax.ndim == 0:
        return (array_jax,)
    return tuple(iter_axis0_entries(array_jax))


def scalar_like(reference, value) -> jax.Array:
    """Convert a value using the reference dtype and the shared boundary.

    Args:
        reference (jax.Array): Reference array of any shape whose dtype and execution
            placement are used.
        value (array-like): Value, usually scalar shape (), to convert to the reference
            dtype.

    Returns:
        jax.Array: Converted value with its original shape; host floats still
            take runtime placement and become float64.
    """
    return as_jax_array(value, dtype=reference.dtype)


def zero_padding_like(array, *, axis: int, pad_width: int):
    """Construct trailing padding with the input dtype and execution placement.

    Args:
        array (jax.Array): Finite input of any nonscalar shape supplying dtype and
            placement.
        axis (int): Axis to replace with a padding extent; negative axes count from the
            end.
        pad_width (int): Nonnegative number of padding entries.

    Returns:
        jax.Array: Zeros with the input shape except for axis extent
            pad_width; formed from the input reduction, so nonfinite/overflowing
            inputs can yield NaNs.
    """
    axis_index = int(axis) if axis >= 0 else array.ndim + int(axis)
    zero_slice = jnp.sum(array, axis=axis_index, keepdims=True, dtype=array.dtype)
    zero_slice = zero_slice - zero_slice
    target_shape = (
        array.shape[:axis_index] + (int(pad_width),) + array.shape[axis_index + 1 :]
    )
    return jnp.broadcast_to(zero_slice, target_shape)


def pad_axis(array, *, axis: int, padded_size: int):
    """Extend an array axis to the requested size without truncating it.

    Args:
        array (jax.Array): Input of any nonscalar shape.
        axis (int): Axis to extend; negative axes count from the end.
        padded_size (int): Desired axis extent; sizes no greater than the current extent
            return the original input.

    Returns:
        jax.Array: Input with trailing zero_padding_like entries, shape equal
            to the input except that axis has extent max(original_size,
            padded_size), retaining dtype.
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


def _explicit_inv_impl(x):
    return jnp.divide(scalar_like(x, 1.0), x)


@jax.custom_jvp
def explicit_inv(x):
    """Evaluate the elementwise reciprocal with an explicit custom JVP.

    Args:
        x (jax.Array): Floating input of arbitrary shape.

    Returns:
        jax.Array: Same shape as x containing 1/x; the JVP is -x_dot/x**2 and
            singularities are not regularized.
    """
    return _explicit_inv_impl(x)


@explicit_inv.defjvp
def _explicit_inv_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = _explicit_inv_impl(x)
    tangent_out = jnp.negative(x_dot * primal_out * primal_out)
    return primal_out, tangent_out


def _explicit_rsqrt_impl(x):
    return jnp.divide(scalar_like(x, 1.0), jnp.sqrt(x))


@jax.custom_jvp
def explicit_rsqrt(x):
    """Evaluate the elementwise reciprocal square root with an explicit custom JVP.

    Args:
        x (jax.Array): Floating input of arbitrary shape.

    Returns:
        jax.Array: Same shape as x containing 1/sqrt(x); the JVP is
            -0.5*x_dot/(x*sqrt(x)) and singularities are not regularized.
    """
    return _explicit_rsqrt_impl(x)


@explicit_rsqrt.defjvp
def _explicit_rsqrt_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = _explicit_rsqrt_impl(x)
    tangent_out = x_dot * scalar_like(x, -0.5) / (x * jnp.sqrt(x))
    return primal_out, tangent_out
