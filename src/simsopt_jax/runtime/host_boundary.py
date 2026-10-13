"""Explicit host materialization and scoped transfer guards for JAX fields.

Adapters materialize values through host_array, host_value or host_tree;
snapshot_host_tree owns host input buffers before backend.dtypes places them.
The scope helpers permit or disallow implicit transfers without changing the
surrounding guard.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, TypeVar

import jax
from jax import core as jax_core
import numpy as np


_TreeT = TypeVar("_TreeT")


@contextmanager
def disallow_host_transfers() -> Iterator[None]:
    """Refuse IMPLICIT host-to-device transfers for the duration of the block.

    That refusal holds on every backend.  Explicit ``device_put``/``device_get``
    always pass, and on CPU -- verified on jax 0.10.0, where no copy actually
    happens -- implicit device-to-host (``np.sum(x)``, ``x.tolist()``) passes
    too.

    Returns:
        contextlib.AbstractContextManager[None] object: Scoped transfer guard
            that restores the surrounding setting on exit.
    """

    with jax.transfer_guard("disallow"):
        yield


@contextmanager
def allow_host_transfers() -> Iterator[None]:
    """Permit implicit transfers inside one explicit host-driven boundary.

    For host callers that hand device arrays to compiled code while an outer
    strict guard may be active: lowering a program that closes over device
    arrays reads them back, which the strict guard would refuse. The counterpart
    of :func:`disallow_host_transfers`.

    Returns:
        contextlib.AbstractContextManager[None] object: Scoped transfer guard
            that restores the surrounding setting on exit.
    """

    with jax.transfer_guard("allow"):
        yield


def host_value(value: _TreeT) -> _TreeT:
    """Materialize a JAX value or pytree while preserving its Python structure.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.

    Returns:
        pytree object: Same structure with JAX leaves materialized as host
            values of matching shapes and dtypes.
    """
    return jax.device_get(value)


def host_array(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> np.ndarray:
    """Materialize ``value`` to a writeable NumPy array at an explicit D2H boundary.

    Args:
        value (array-like): Value of any shape to materialize on the host.
        dtype (dtype-like or None): Host conversion dtype; None preserves the
            materialized dtype.

    Returns:
        numpy.ndarray: Writable host array with the input shape; already
            writable NumPy storage may be reused.
    """
    array = np.asarray(host_value(value))
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return array


def block_until_ready(value: _TreeT) -> _TreeT:
    """Wait for every JAX leaf and return the same pytree structure and values.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.

    Returns:
        pytree object: Same structure and values after all asynchronous JAX
            leaves are ready.
    """
    return jax.block_until_ready(value)


def snapshot_host_tree(value: _TreeT, *, dtype=None) -> _TreeT:
    """Own NumPy leaves synchronously before asynchronous JAX consumption.

    Device arrays and tracers pass through unchanged. A private contiguous
    NumPy copy prevents both CPU buffer aliasing and delayed transfer reads.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.
        dtype (dtype-like or None): Optional dtype for host NumPy leaves and host
            scalars; None preserves dtypes. Device arrays and tracers are not coerced by
            snapshotting.

    Returns:
        pytree object: Same structure with private C-contiguous copies of
            NumPy array leaves, preserving each leaf shape. Device arrays and
            tracers remain unchanged.
    """
    if isinstance(value, (jax.Array, jax_core.Tracer)):
        return value

    def _hostify_leaf(leaf):
        if isinstance(leaf, jax_core.Tracer):
            return leaf
        if isinstance(leaf, np.ndarray):
            leaf_dtype = leaf.dtype if dtype is None else dtype
            return np.array(leaf, dtype=leaf_dtype, copy=True, order="C")
        if dtype is not None and (isinstance(leaf, np.generic) or np.isscalar(leaf)):
            return np.asarray(leaf, dtype=dtype)
        return leaf

    return jax.tree.map(_hostify_leaf, value)


def host_tree(value, *, dtype=None):
    """Materialize a pytree on the host and snapshot its NumPy storage.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.
        dtype (dtype-like or None): Conversion dtype for materialized NumPy leaves
            and host scalars; default None preserves dtypes. Device arrays are
            materialized before this conversion; each leaf keeps its shape.

    Returns:
        pytree: Host leaves with the same shape per leaf and privately owned
            NumPy array buffers.
    """
    return snapshot_host_tree(host_value(value), dtype=dtype)


def host_tree_after_ready(value, *, dtype=None):
    """Wait for a pytree, then materialize its array leaves on the host.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.
        dtype (dtype-like or None): Conversion dtype for materialized NumPy leaves
            and host scalars; default None preserves dtypes. Device arrays are
            materialized before this conversion; each leaf keeps its shape.

    Returns:
        pytree object: Ready host leaves with the same shape per leaf and
            privately owned NumPy array buffers.
    """
    return host_tree(block_until_ready(value), dtype=dtype)
