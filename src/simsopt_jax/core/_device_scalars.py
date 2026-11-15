"""Helpers for constructing scalar values on the same device as a reference array."""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Hashable
from typing import cast
from functools import partial

import jax
from jax import core as jax_core
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.jit import jit


@lru_cache(maxsize=None)
def _staged_scalar_builder(host_value: object, dtype_string: str):
    resolved_dtype = np.dtype(dtype_string)

    @jit
    def build(reference):
        zero = placement_zero(reference).astype(resolved_dtype)
        literal = jnp.asarray(host_value, dtype=resolved_dtype)
        return zero + literal

    return build


def device_one(reference: jax.Array) -> jax.Array:
    """Return scalar one with the reference dtype and placement.

    Args:
        reference (jax.Array): Reference array of any shape whose dtype and execution
            placement are used.

    Returns:
        jax.Array: Dimensionless scalar 1, shape (), with zero cotangent, for
            any reference values, finite or not.
    """
    return jax.lax.stop_gradient(jnp.exp(placement_zero(reference)))


@partial(jit, keep_unused=True)
def _placed_zero(reference: jax.Array) -> jax.Array:
    """Return a placed positive zero, including for nonfinite inputs and derivatives."""
    empty = jax.lax.broadcast_in_dim(
        reference, (0,) + reference.shape, tuple(range(1, reference.ndim + 1))
    )
    return jnp.sum(empty, dtype=reference.dtype)


@jax.custom_jvp
def placement_zero(reference: jax.Array) -> jax.Array:
    """Return exact scalar zero with the reference dtype and placement.

    Args:
        reference (jax.Array): Reference array of any shape whose dtype and execution
            placement are used.

    Returns:
        jax.Array: Exact scalar +0, shape (), with the reference dtype and
            placement and exact placed zero tangents/cotangents,
            including for nonfinite reference values and derivative seeds.
    """
    return _placed_zero(reference)


@placement_zero.defjvp
def _placement_zero_jvp(primals, tangents):
    (reference,), (tangent,) = primals, tangents
    zero = placement_zero(reference)
    return zero, _placed_zero(tangent) + zero


def two_pi(reference: jax.Array) -> jax.Array:
    """Return scalar 2*pi with the reference dtype and placement.

    Args:
        reference (jax.Array): Reference array of any shape whose dtype and execution
            placement are used.

    Returns:
        jax.Array: Dimensionless scalar 2*pi, shape (), with the reference
            dtype and placement, even for nonfinite reference values.
    """
    pi = jnp.arccos(-device_one(reference))
    return pi + pi


def staged_like(reference: jax.Array, host_value, *, dtype=None) -> jax.Array:
    """Explicitly stage a value with reference-compatible placement.

    Args:
        reference (jax.Array): Reference array of any shape whose dtype and execution
            placement are used.
        host_value (array-like): Value to stage, of any shape; may also be a concrete
            JAX array or tracer.
        dtype (dtype-like or None): Requested dtype; None uses reference.dtype.

    Returns:
        jax.Array: Staged value with the input value shape and resolved dtype.
            Concrete placement follows the reference; under tracing placement
            belongs to the compiled program.
    """
    reference = jnp.asarray(reference)
    resolved_dtype = reference.dtype if dtype is None else np.dtype(dtype)
    if isinstance(host_value, jax.Array):
        if isinstance(host_value, jax_core.Tracer) or isinstance(
            reference, jax_core.Tracer
        ):
            return jnp.asarray(host_value, dtype=resolved_dtype)
        return explicit_device_array(
            host_value,
            dtype=resolved_dtype,
            reference=reference,
        )
    if isinstance(reference, jax_core.Tracer) and np.ndim(host_value) == 0:
        typed_host_value = np.asarray(host_value, dtype=resolved_dtype)[()]
        return _staged_scalar_builder(
            cast(Hashable, typed_host_value),
            resolved_dtype.str,
        )(reference)
    if isinstance(reference, jax_core.Tracer):
        return jnp.asarray(host_value, dtype=resolved_dtype)
    return explicit_device_array(
        host_value,
        dtype=resolved_dtype,
        reference=reference,
    )
