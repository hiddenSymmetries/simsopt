"""JAX dtype conversion and device-placement helpers.

Floating values use float64; exact-dtype placement rejects other float dtypes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, cast

import jax
from jax import core as jax_core
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.runtime import (
    get_runtime_jax_device,
    host_value,
    snapshot_host_tree,
)

_TreeT = TypeVar("_TreeT")

_FLOAT64 = np.dtype(np.float64)


def _shape_tuple(shape: int | Sequence[int]) -> tuple[int, ...]:
    if np.isscalar(shape):
        return (int(cast(int, shape)),)
    return tuple(int(dim) for dim in cast(Sequence[int], shape))


def _contains_jax_leaves(value) -> bool:
    return any(
        isinstance(leaf, jax.Array) or hasattr(leaf, "aval")
        for leaf in jax.tree.leaves(value)
    )


def _has_jax_array_value(value) -> bool:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return True
    return isinstance(value, (list, tuple)) and _contains_jax_leaves(value)


def _reference_placement(reference):
    # A tracer is a ``jax.Array`` with no concrete placement; probing it raises an
    # ``AttributeError`` whose message walks the whole jaxpr. Once per placement
    # across a trace, that is an O(jaxpr^2) cost, so tracers are skipped.
    if isinstance(reference, jax_core.Tracer):
        return None
    if isinstance(reference, jax.Array):
        return _committed_placement(reference)
    if isinstance(reference, (list, tuple)):
        for leaf in jax.tree.leaves(reference):
            if isinstance(leaf, jax.Array) and not isinstance(leaf, jax_core.Tracer):
                placement = _committed_placement(leaf)
                if placement is not None:
                    return placement
    return None


def _committed_placement(array: jax.Array):
    """Return a concrete array's committed device, or ``None`` if uncommitted."""
    if not array.committed:
        return None
    (device,) = array.devices()
    return device


def _array_like_dtype(value) -> np.dtype | None:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)
    if isinstance(value, (list, tuple)):
        if _contains_jax_leaves(value):
            return np.dtype(jnp.asarray(value).dtype)
        return np.asarray(value).dtype
    if np.isscalar(value):
        return np.asarray(value).dtype
    return None


def _device_put_target(target, device):
    if target is not None and device is not None:
        raise TypeError("runtime_device_put accepts either target or device, not both.")
    return device if device is not None else target


def _runtime_device_put_dtype(
    value,
    dtype,
    *,
    preserve_float_dtype: bool,
) -> np.dtype | None:
    """Placement dtype: floats become float64; an exact request must already be float64."""
    requested = None if dtype is None else np.dtype(dtype)
    resolved = _array_like_dtype(value) if requested is None else requested
    if resolved is None or resolved.kind != "f":
        return requested
    if preserve_float_dtype and resolved != _FLOAT64:
        raise TypeError(
            f"Floating dtype must be one of {(_FLOAT64.name,)}; got {resolved.name!r}."
        )
    return _FLOAT64


def _placement_platforms(placement) -> frozenset[str]:
    """Platforms an explicit placement (a device or a sharding) spans."""
    devices = (
        placement.device_set
        if isinstance(placement, jax.sharding.Sharding)
        else (placement,)
    )
    return frozenset(device.platform for device in devices)


def _explicit_backend_hop(leaf, placement):
    """Materialize cross-backend array transfers on the host; pass other leaves through."""
    if placement is None or not isinstance(leaf, jax.Array) or isinstance(leaf, jax_core.Tracer):
        return leaf
    leaf_platforms = frozenset(device.platform for device in leaf.devices())
    if leaf_platforms == _placement_platforms(placement):
        return leaf
    return host_value(leaf)


def _explicit_device_put_leaves(leaves: list, placements: list) -> list:
    """Place leaves with explicit cross-backend transfers."""
    return jax.device_put(  # noqa: TID251 placement owner; callers snapshot NumPy leaves
        [
            _explicit_backend_hop(leaf, placement)
            for leaf, placement in zip(leaves, placements, strict=True)
        ],
        placements,
    )


def _explicit_device_put(value, placement):
    """Place a value with explicit cross-backend transfers."""
    return jax.device_put(  # noqa: TID251 placement owner; callers snapshot NumPy leaves
        jax.tree.map(lambda leaf: _explicit_backend_hop(leaf, placement), value),
        placement,
    )


def _uncommitted_default_device():
    """The device an uncommitted put lands on: a ``jax.default_device`` scope, else JAX's."""
    scoped = jax.config.values["jax_default_device"]
    if scoped is None:
        return jax.local_devices()[0]
    if isinstance(scoped, str):
        return jax.local_devices(backend=scoped)[0]
    return scoped


def _unplaced_leaf_placement(leaf, *, home, runtime_device):
    """Where an unplaced leaf goes; ``None`` leaves it uncommitted on ``home``."""
    if not isinstance(leaf, jax.Array) or isinstance(leaf, jax_core.Tracer):
        return None
    if leaf.committed:
        return runtime_device
    if leaf.devices() == {home}:
        return None
    return home


def _unplaced_device_put(value):
    """Place a value or pytree using default-device/runtime policy.

    An active ``jax.default_device`` scope takes precedence for host and
    uncommitted values; committed arrays move to the runtime device.
    Host values and arrays already at home stay uncommitted when JAX's
    default placement agrees; otherwise placement is committed.
    """
    runtime_device = get_runtime_jax_device()
    if runtime_device is None:
        return _explicit_device_put(value, None)
    default_device = _uncommitted_default_device()
    scoped = jax.config.values["jax_default_device"] is not None
    home = default_device if scoped else runtime_device
    if home != default_device:
        return _explicit_device_put(value, runtime_device)
    leaves, treedef = jax.tree.flatten(value)
    placements = [
        _unplaced_leaf_placement(leaf, home=home, runtime_device=runtime_device)
        for leaf in leaves
    ]
    return jax.tree.unflatten(treedef, _explicit_device_put_leaves(leaves, placements))


def _device_put(
    value,
    *,
    dtype,
    target,
    device,
    preserve_float_dtype: bool,
) -> jax.Array:
    placement = _device_put_target(target, device)
    resolved_dtype = _runtime_device_put_dtype(
        value,
        dtype,
        preserve_float_dtype=preserve_float_dtype,
    )
    if _has_jax_array_value(value):
        array = jnp.asarray(snapshot_host_tree(value), dtype=resolved_dtype)
    else:
        # One private C-contiguous copy, as ``snapshot_host_tree`` makes.
        array = np.array(value, dtype=resolved_dtype, copy=True, order="C")
    if placement is None:
        return _unplaced_device_put(array)
    return _explicit_device_put(array, placement)


def runtime_device_put(value, *, dtype=None, target=None, device=None) -> jax.Array:
    """Copy host inputs and place values with floating values converted to float64.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are copied.
        dtype (dtype-like or None): Requested dtype; None infers it. Floats become float64.
        target (jax.Device, jax.sharding.Sharding or None): Explicit placement; omit to
            use runtime/default-device policy. Mutually exclusive with device.
        device (jax.Device or None): Explicit device, mutually exclusive with target.

    Returns:
        jax.Array: Same shape at the resolved placement; floats use float64.
    """
    return _device_put(
        value,
        dtype=dtype,
        target=target,
        device=device,
        preserve_float_dtype=False,
    )


def runtime_device_put_tree(
    value: _TreeT,
    *,
    target: object | None = None,
    device: object | None = None,
) -> _TreeT:
    """Place dynamic leaves without changing dtypes or structure.

    Args:
        value (pytree): Dynamic array leaves of any shape; host NumPy leaves are copied
            before placement.
        target (jax.Device, jax.sharding.Sharding or None): Explicit placement; omit to
            use runtime/default-device policy. Mutually exclusive with device.
        device (jax.Device or None): Explicit device, mutually exclusive with target.

    Returns:
        pytree object: Same structure and leaf shapes/dtypes, with dynamic
            leaves placed on devices.
    """
    placement = _device_put_target(target, device)
    value = snapshot_host_tree(value)
    if placement is None:
        return _unplaced_device_put(value)
    return _explicit_device_put(value, placement)


def as_jax_array(value, *, dtype) -> jax.Array:
    """Convert host or JAX values with dtype conversion.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are copied.
        dtype (dtype-like): Conversion dtype for JAX values; host floating values
            become float64.

    Returns:
        jax.Array: Same shape; existing JAX placement is retained.
    """
    if _has_jax_array_value(value):
        return jnp.asarray(snapshot_host_tree(value), dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return runtime_device_put(value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype)


def as_jax_float64(value) -> jax.Array:
    """Convert host or JAX values to float64.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are copied.

    Returns:
        jax.Array: Same shape as value in float64; JAX values retain their
            placement and host values take runtime placement.
    """
    return as_jax_array(value, dtype=jnp.float64)


def as_jax_int32(value) -> jax.Array:
    """Convert host or JAX values to signed 32-bit integers.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are copied.

    Returns:
        jax.Array: Same shape as value with int32 entries.
    """
    return as_jax_array(value, dtype=jnp.int32)


def explicit_device_array(
    value,
    *,
    dtype,
    reference=None,
    target=None,
    device=None,
) -> jax.Array:
    """Place an exact-dtype array using one explicit or reference placement.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are copied.
        dtype (dtype-like): Exact requested dtype; a floating dtype must be float64.
        reference (jax.Array, sequence or None): Reference arrays of any shape; use the
            first committed concrete placement, ignoring tracers and uncommitted leaves.
            Mutually exclusive with target/device.
        target (jax.Device, jax.sharding.Sharding or None): Explicit placement; omit to
            use runtime/default-device policy. Mutually exclusive with device.
        device (jax.Device or None): Explicit device, mutually exclusive with target.

    Returns:
        jax.Array: Same shape as value in the exact dtype at the resolved
            placement.
    """
    if reference is not None and (target is not None or device is not None):
        raise TypeError(
            "explicit_device_array accepts reference or explicit target/device, not both."
        )
    return _device_put(
        value,
        dtype=dtype,
        target=_reference_placement(reference) if reference is not None else target,
        device=device,
        preserve_float_dtype=True,
    )
