"""Policy-owned JAX dtype and device-placement helpers (H2D / on-device SSOT).

Ownership split:

* This module — **device placement and dtype policy**: ``runtime_device_put``
  (floating values become float64), ``explicit_device_array`` (exact requested
  dtype, optionally on a reference array's device; float64 is the only floating
  dtype), ``as_jax_float64`` (float64 conversion). Do not reimplement
  ``device_put`` with ad-hoc float coercion.
* ``simsopt_jax.runtime.host_boundary`` — **host ownership and materialization**:
  ``snapshot_host_tree`` before placement; ``host_array`` / ``host_tree`` (D2H).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, cast

import jax
from jax import core as jax_core
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.runtime.host_boundary import snapshot_host_tree

__all__ = [
    "as_jax_array",
    "as_jax_float64",
    "as_jax_int32",
    "commit_in_place",
    "explicit_device_array",
    "runtime_device_put",
    "runtime_device_put_tree",
]

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


def _is_jax_tracer(value) -> bool:
    return isinstance(value, jax_core.Tracer)


def _has_jax_array_value(value) -> bool:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return True
    return isinstance(value, (list, tuple)) and _contains_jax_leaves(value)


def _reference_placement(reference):
    # A tracer is a ``jax.Array`` but carries no concrete placement; probing its
    # placement raises ``AttributeError`` whose message eagerly walks the entire
    # jaxpr (jax's ``_origin_msg``/``find_progenitors``). Paid once per
    # placement across an O(jaxpr) trace, that is an O(jaxpr^2) construction
    # cost that scales with resolution, so tracers are skipped.
    if _is_jax_tracer(reference):
        return None
    if isinstance(reference, jax.Array):
        return _committed_placement(reference)
    if isinstance(reference, (list, tuple)):
        for leaf in jax.tree.leaves(reference):
            if isinstance(leaf, jax.Array) and not _is_jax_tracer(leaf):
                placement = _committed_placement(leaf)
                if placement is not None:
                    return placement
    return None


def _committed_placement(array: jax.Array):
    """The device a concrete array is committed to, or ``None`` if it claims none.

    An uncommitted array (placed by nobody, JAX's default device) makes no
    claim, so a value placed with it stays unplaced too
    (``_unplaced_device_put``) and joins whatever committed data it meets.
    """
    if not array.committed:
        return None
    return _array_device(array)


def _array_device(array: jax.Array):
    """The bare device a concrete single-device array lives on.

    A bare device and the array's own placement object put identically when
    ``device_put`` runs eagerly: the result is committed to the same device.
    They differ when the put is staged into a ``jit`` trace, which happens
    whenever a host literal is placed next to a *concrete* reference captured
    by a traced function. The placement object carries
    ``memory_kind='device'``, so jax both wraps the staged constant in a
    single-device placement op and folds it into the computation's device
    assignment (jax's dispatch and ``_tpu_gpu_device_put_lowering`` key on
    ``memory_kind is not None``).
    That pins the whole jaxpr to one device. A bare device is ignored by both,
    leaving the constant's placement to XLA.
    """
    (device,) = array.devices()
    return device


def _array_like_dtype(value) -> np.dtype | None:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)
    if isinstance(value, (np.ndarray, np.generic)):
        return np.asarray(value).dtype
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
    if not isinstance(leaf, jax.Array) or _is_jax_tracer(leaf):
        return None
    if leaf.committed:
        return runtime_device
    if leaf.devices() == {home}:
        return None
    return home


def _unplaced_device_put(value):
    """Place a value (or pytree) its caller did not place, as JAX itself would.

    Its home is an active ``jax.default_device`` scope, else the runtime
    device. When JAX's own uncommitted placement lands there, a host value
    (or an uncommitted array already there) stays uncommitted, as JAX leaves
    it, and joins the committed data it meets: committing it would claim a
    placement no caller made and refuse every computation with data committed
    elsewhere (for example an array built under a CPU ``jax.default_device`` scope).
    An uncommitted array elsewhere is moved home, a committed array onto the
    runtime device, and everything is committed when the runtime device is
    one JAX would not choose (a jax-cpu policy in a CUDA process).
    """
    runtime_device = get_runtime_jax_device()
    if runtime_device is None:
        return jax.device_put(value)
    default_device = _uncommitted_default_device()
    scoped = jax.config.values["jax_default_device"] is not None
    home = default_device if scoped else runtime_device
    if home != default_device:
        return jax.device_put(value, runtime_device)
    leaves, treedef = jax.tree.flatten(value)
    placements = [
        _unplaced_leaf_placement(leaf, home=home, runtime_device=runtime_device)
        for leaf in leaves
    ]
    return jax.tree.unflatten(treedef, jax.device_put(leaves, placements))


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
    elif resolved_dtype is None:
        array = np.asarray(value)
    else:
        array = np.asarray(value, dtype=resolved_dtype)
    array = snapshot_host_tree(array)
    if placement is None:
        return _unplaced_device_put(array)
    return jax.device_put(array, placement)


def runtime_device_put(value, *, dtype=None, target=None, device=None) -> jax.Array:
    """Snapshot host values, then place them using runtime policy for float dtypes.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are snapshotted
            before device consumption.
        dtype (dtype-like or None): Optional requested dtype; None infers it. Floating
            types are converted to float64.
        target (jax.Device, jax.sharding.Sharding or None): Explicit placement; omit to
            use runtime/default-device policy. Mutually exclusive with device.
        device (jax.Device or None): Explicit device, mutually exclusive with target.

    Returns:
        jax.Array: Same shape as value, floating values in float64, at the
            resolved placement.
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
    preserve_placement: bool = False,
) -> _TreeT:
    """Place dynamic leaves without changing dtypes or structure.

    With no explicit target, ``preserve_placement`` keeps existing JAX arrays
    unchanged and applies runtime placement only to host leaves.

    Args:
        value (pytree): Dynamic array leaves of any shape; host NumPy leaves are copied
            before placement.
        target (jax.Device, jax.sharding.Sharding or None): Explicit placement; omit to
            use runtime/default-device policy. Mutually exclusive with device.
        device (jax.Device or None): Explicit device, mutually exclusive with target.
        preserve_placement (bool): With no explicit placement, keep existing JAX leaves
            unchanged and place only host leaves.

    Returns:
        pytree object: Same structure and leaf shapes/dtypes, with dynamic
            leaves placed on devices.
    """
    placement = _device_put_target(target, device)
    value = snapshot_host_tree(value)
    if placement is None:
        if preserve_placement:
            return jax.tree.map(
                lambda leaf: leaf if isinstance(leaf, jax.Array) else _unplaced_device_put(leaf),
                value,
            )
        return _unplaced_device_put(value)
    return jax.device_put(value, placement)


def _device_put_preserving_dtype(
    value,
    *,
    dtype,
    target=None,
    device=None,
) -> jax.Array:
    """Place a value in its exact dtype; a floating dtype must be float64."""
    return _device_put(
        value,
        dtype=dtype,
        target=target,
        device=device,
        preserve_float_dtype=True,
    )


def as_jax_array(value, *, dtype) -> jax.Array:
    """Convert host or JAX values using the shared dtype and placement boundary.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are snapshotted
            before device consumption.
        dtype (dtype-like): Conversion dtype for JAX values; host floating values
            become float64.

    Returns:
        jax.Array: Converted value with the same shape. Existing JAX placement
            is retained by conversion.
    """
    if _has_jax_array_value(value):
        return jnp.asarray(snapshot_host_tree(value), dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return runtime_device_put(value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype)


def as_jax_float64(value) -> jax.Array:
    """Convert host or JAX values to float64 through the shared boundary.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are snapshotted
            before device consumption.

    Returns:
        jax.Array: Same shape as value in float64; JAX values retain their
            placement and host values take runtime placement.
    """
    return as_jax_array(value, dtype=jnp.float64)


def as_jax_int32(value) -> jax.Array:
    """Convert host or JAX values to signed 32-bit integers.

    Args:
        value (array-like): Host or JAX value of any shape; NumPy inputs are snapshotted
            before device consumption.

    Returns:
        jax.Array: Same shape as value with int32 entries.
    """
    return as_jax_array(value, dtype=jnp.int32)


def commit_in_place(array: jax.Array) -> jax.Array:
    """Commit a concrete array to the device it already lives on.

    For a loop that feeds a jitted step its own outputs: ``jit`` keys
    committed and uncommitted arguments separately, and a step's outputs are
    committed whenever an input is, so the first input is committed up front
    to compile the executable every later call reuses.

    Args:
        array (jax.Array): Concrete single-device array of any shape and dtype.

    Returns:
        jax.Array: Same values, shape and dtype, committed to the original
            device.
    """
    return _device_put_preserving_dtype(
        array,
        dtype=array.dtype,
        target=_array_device(array),
    )


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
        value (array-like): Host or JAX value of any shape; NumPy inputs are snapshotted
            before device consumption.
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
    return _device_put_preserving_dtype(
        value,
        dtype=dtype,
        target=_reference_placement(reference) if reference is not None else target,
        device=device,
    )
