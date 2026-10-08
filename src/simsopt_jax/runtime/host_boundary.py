"""Explicit host materialization and scoped transfer guards for JAX fields.

Adapters materialize values through host_array, host_value or host_tree;
backend.dtypes owns device placement. The scope helpers permit or disallow
implicit transfers without changing the surrounding guard.
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
    """

    with jax.transfer_guard("allow"):
        yield


def host_value(value: _TreeT) -> _TreeT:
    """Materialize a JAX value or pytree while preserving its Python structure."""
    return jax.device_get(value)


def host_array(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> np.ndarray:
    """Materialize ``value`` to a writeable NumPy array at an explicit D2H boundary."""
    array = np.asarray(host_value(value))
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return array


def block_until_ready(value: _TreeT) -> _TreeT:
    """Wait for every JAX leaf and return the same pytree structure and values."""
    return jax.block_until_ready(value)


def host_tree(value, *, dtype=None):
    materialized = host_value(value)

    def _hostify_leaf(leaf):
        if isinstance(leaf, jax_core.Tracer):
            return leaf
        if isinstance(leaf, np.ndarray):
            leaf_dtype = leaf.dtype if dtype is None else dtype
            return np.array(leaf, dtype=leaf_dtype, copy=True)
        if dtype is not None and (isinstance(leaf, np.generic) or np.isscalar(leaf)):
            return np.asarray(leaf, dtype=dtype)
        return leaf

    return jax.tree.map(_hostify_leaf, materialized)


def host_tree_after_ready(value, *, dtype=None):
    """Wait for a pytree, then materialize its array leaves on the host."""
    return host_tree(block_until_ready(value), dtype=dtype)
