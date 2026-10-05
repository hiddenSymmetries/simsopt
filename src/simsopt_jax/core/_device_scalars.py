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


@lru_cache(maxsize=None)
def _staged_scalar_builder(host_value: object, dtype_string: str):
    resolved_dtype = np.dtype(dtype_string)

    @jax.jit
    def build(reference):
        zero = jnp.sum(reference - reference).astype(resolved_dtype)
        literal = jnp.asarray(host_value, dtype=resolved_dtype)
        return zero + literal

    return build


def device_one(reference: jax.Array) -> jax.Array:
    """A 1.0 placed and typed like ``reference``, with no derivative path.

    The value is built from ``reference`` only for placement.  Its derivative
    is zero, and it must also be computed as zero: left differentiable, reverse
    mode sends the full cotangent of every product ``device_one(r) * y`` into
    ``r`` twice, as ``+c`` and ``-c``, and accumulates them beside ``r``'s own
    cotangent.  When ``c`` dwarfs that cotangent -- ``mu0/4pi`` scaling a field
    whose currents are its reference -- the sum ``(g + c) - c`` keeps only
    ``u |c|`` of ``g``'s accuracy.
    """
    return jax.lax.stop_gradient(jnp.exp(jnp.sum(reference - reference)))


@partial(jax.jit, keep_unused=True)
def _placed_zero(reference: jax.Array) -> jax.Array:
    """A literal 0.0 of ``reference``'s dtype, computed where ``reference`` lives.

    No arithmetic reads ``reference``'s values: the zero is a constant of the
    compiled program, and the unused argument (kept) only places the program
    on ``reference``'s device, so no host-to-device transfer creates it.
    """
    return jnp.zeros((), dtype=reference.dtype)


@jax.custom_jvp
def placement_zero(reference: jax.Array) -> jax.Array:
    """An exact 0.0 placed and typed like ``reference`` whose tangent is also a placed 0.0.

    Adding it to a value that does not depend on ``reference`` (a fixed coil
    current) gives that value a tangent on ``reference``'s device, so
    ``jax.linearize`` never materializes a symbolic-zero tangent from a host
    literal under the strict transfer guard. ``stop_gradient`` cannot do this:
    its tangent is that symbolic zero. Nor can a tangent that ignores the input
    tangent, which linearization also treats as a symbolic zero.

    The zero reads no value of ``reference``:
    ``sum(r) - sum(r)`` overflows to NaN for finite ``r = [1e308, 1e308]``
    and couples every entry of ``r`` into every sum it joins. The tangent is a
    select of the zero over the input tangent's sum: it depends on the input
    tangent (so it stays on device), its value is exactly 0.0 for any tangent,
    NaN and infinities included, and its transpose sends exactly 0.0 back.
    """
    return _placed_zero(reference)


@placement_zero.defjvp
def _placement_zero_jvp(primals, tangents):
    (reference,), (tangent,) = primals, tangents
    zero = placement_zero(reference)
    # ``zero`` is exactly 0.0 by construction, so the mask is a placed False and
    # the select always yields the zero; it exists only to make the tangent
    # depend on ``tangent``.
    return zero, jnp.where(jnp.isnan(zero), jnp.sum(tangent), zero)


def two_pi(reference: jax.Array) -> jax.Array:
    pi = jnp.arccos(-device_one(reference))
    return pi + pi


def staged_like(reference: jax.Array, host_value, *, dtype=None) -> jax.Array:
    """Explicitly stage a value with reference-compatible placement.

    A host value is placed with the reference; so is a concrete device array
    held elsewhere (an explicit transfer), so the result always joins the
    reference in one program. Under a trace the value is converted in place.
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
