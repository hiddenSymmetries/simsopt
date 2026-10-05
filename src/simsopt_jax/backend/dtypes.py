"""Policy-owned JAX dtype and device-placement helpers (H2D / on-device SSOT).

Ownership split:

* This module — **device placement and dtype policy**: ``runtime_device_put``
  (runtime float policy), ``explicit_device_array`` (exact requested dtype),
  ``as_runtime_array`` / ``as_compute_array`` (policy + optional reference
  sharding). Do not reimplement ``device_put`` with ad-hoc float coercion.
* ``simsopt_jax.runtime.host_boundary`` — **host materialization** (D2H):
  ``host_array`` / ``host_tree`` and ready variants.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, cast

import jax
from jax import core as jax_core
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import Sharding

from simsopt_jax.backend.runtime import (
    get_backend_policy,
    get_compute_dtype,
    get_runtime_jax_device,
)

__all__ = [
    "as_compute_array",
    "as_jax_array",
    "as_jax_float64",
    "as_jax_int32",
    "as_runtime_array",
    "as_runtime_float64",
    "commit_in_place",
    "compute_jnp_dtype",
    "explicit_device_array",
    "runtime_device_put",
    "runtime_device_put_tree",
    "runtime_jnp_dtype",
    "runtime_np_dtype",
]

_TreeT = TypeVar("_TreeT")

_DTYPE_BY_NAME = {
    "float64": jnp.float64,
    "float32": jnp.float32,
}
_HOST_DTYPE_BY_NAME = {
    "float64": np.dtype(np.float64),
    "float32": np.dtype(np.float32),
}


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


def _contains_traced_jax_leaves(value) -> bool:
    return any(_is_jax_tracer(leaf) for leaf in jax.tree.leaves(value))


def _has_jax_array_value(value) -> bool:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return True
    return isinstance(value, (list, tuple)) and _contains_jax_leaves(value)


def _contains_concrete_jax_leaves(value) -> bool:
    return any(
        isinstance(leaf, jax.Array) and not _is_jax_tracer(leaf)
        for leaf in jax.tree.leaves(value)
    )


def _has_only_traced_jax_leaves(value) -> bool:
    return _has_jax_array_value(value) and not _contains_concrete_jax_leaves(value)


def _reference_placement(reference, *, ndim: int | None = None):
    # A tracer is a ``jax.Array`` but carries no concrete sharding; probing
    # ``tracer.sharding`` raises ``AttributeError`` whose message eagerly walks
    # the entire jaxpr (jax's ``_origin_msg``/``find_progenitors``) only to be
    # discarded by ``getattr(..., None)``. Paid once per ``as_runtime_array``
    # call across an O(jaxpr) trace, that is an O(jaxpr^2) construction cost that
    # scales with resolution. Skip tracers: their reference sharding is always
    # ``None`` and ``as_runtime_array`` bypasses reference placement for traced
    # values regardless (see ``_has_only_traced_jax_leaves`` guard there).
    if _is_jax_tracer(reference):
        return None
    if isinstance(reference, jax.Array):
        return _committed_placement(reference, ndim=ndim)
    if isinstance(reference, (list, tuple)):
        for leaf in jax.tree.leaves(reference):
            if isinstance(leaf, jax.Array) and not _is_jax_tracer(leaf):
                placement = _committed_placement(leaf, ndim=ndim)
                if placement is not None:
                    return placement
    return None


def _committed_placement(array: jax.Array, *, ndim: int | None):
    """The placement a concrete array claims, or ``None`` if it claims none.

    An uncommitted array (placed by nobody, JAX's default device) makes no
    claim, so a value placed with it stays unplaced too
    (``_unplaced_device_put``) and joins whatever committed data it meets.
    """
    if not array.committed:
        return None
    sharding = array.sharding
    if isinstance(sharding, NamedSharding):
        return _compatible_reference_sharding(sharding, ndim=ndim)
    return _single_device_placement(sharding)


def _single_device_placement(sharding):
    """Reduce a single-device reference sharding to the bare device it names.

    Both forms place identically when ``device_put`` runs eagerly: the result is
    committed to the same device with the same sharding. They differ when the
    put is staged into a ``jit`` trace, which happens whenever a host literal is
    placed next to a *concrete* reference captured by a traced function. A
    concrete ``Sharding`` carries ``memory_kind='device'``, so jax
    both wraps the staged constant in a single-device sharding op and folds that
    sharding into the computation's device assignment
    (``dispatch.get_intermediate_shardings`` and ``_tpu_gpu_device_put_lowering``
    key on ``isinstance(device, Sharding) and device.memory_kind is not None``).
    That pins the whole jaxpr to one device and is rejected outright when an
    argument is replicated or point-axis sharded across several. A bare device
    is ignored by both, leaving the constant's placement to XLA.
    """
    if not isinstance(sharding, Sharding) or len(sharding.device_set) != 1:
        return sharding
    (device,) = sharding.device_set
    return device


def _reference_sharding(reference, *, ndim: int | None = None):
    placement = _reference_placement(reference, ndim=ndim)
    if placement is None or isinstance(placement, NamedSharding):
        return placement
    return _compatible_reference_sharding(placement, ndim=ndim)


def _compatible_reference_sharding(sharding, *, ndim: int | None):
    if not isinstance(sharding, NamedSharding):
        return None
    if ndim is None or len(sharding.spec) <= ndim:
        return sharding
    return NamedSharding(sharding.mesh, P())


def _value_ndim(value) -> int | None:
    if isinstance(value, (list, tuple)) and _contains_jax_leaves(value):
        return None
    if _contains_traced_jax_leaves(value):
        return None
    if isinstance(value, jax.Array):
        return int(value.ndim)
    if hasattr(value, "aval"):
        return None
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return int(np.ndim(value))
    return None


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


def _dtype_name(dtype, *, source: str) -> str:
    if isinstance(dtype, str):
        name = dtype
    else:
        name = np.dtype(dtype).name
    if name not in _DTYPE_BY_NAME:
        accepted = tuple(_DTYPE_BY_NAME)
        raise TypeError(f"{source} must be one of {accepted}; got {name!r}.")
    return name


def _jnp_dtype_from_name(name: str, *, source: str):
    if name not in _DTYPE_BY_NAME:
        accepted = tuple(_DTYPE_BY_NAME)
        raise TypeError(f"{source} must be one of {accepted}; got {name!r}.")
    return _DTYPE_BY_NAME[name]


def _np_dtype_from_name(name: str, *, source: str) -> np.dtype:
    if name not in _HOST_DTYPE_BY_NAME:
        accepted = tuple(_HOST_DTYPE_BY_NAME)
        raise TypeError(f"{source} must be one of {accepted}; got {name!r}.")
    return _HOST_DTYPE_BY_NAME[name]


def runtime_jnp_dtype():
    dtype_name = get_backend_policy().runtime_dtype
    return _jnp_dtype_from_name(dtype_name, source="BackendPolicy.runtime_dtype")


def compute_jnp_dtype():
    dtype_name = get_compute_dtype()
    return _jnp_dtype_from_name(dtype_name, source="BackendPolicy.compute_dtype")


def runtime_np_dtype() -> np.dtype:
    dtype_name = get_backend_policy().runtime_dtype
    return _np_dtype_from_name(dtype_name, source="BackendPolicy.runtime_dtype")


def _resolve_jnp_dtype(dtype, *, source: str):
    if dtype is None:
        return runtime_jnp_dtype()
    return _jnp_dtype_from_name(_dtype_name(dtype, source=source), source=source)


def _resolve_np_dtype(dtype, *, source: str) -> np.dtype:
    if dtype is None:
        return runtime_np_dtype()
    return _np_dtype_from_name(_dtype_name(dtype, source=source), source=source)


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
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype.kind == "f" and not preserve_float_dtype:
            return runtime_np_dtype()
        return dtype
    value_dtype = _array_like_dtype(value)
    if value_dtype is not None and value_dtype.kind == "f" and not preserve_float_dtype:
        return runtime_np_dtype()
    return None


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
    elsewhere (an array built under
    ``with_cpu_device_for_construction``, an active mesh on other devices).
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
        array = jnp.asarray(value, dtype=resolved_dtype)
    elif resolved_dtype is None:
        array = np.asarray(value)
    else:
        array = np.asarray(value, dtype=resolved_dtype)
    if placement is None:
        return _unplaced_device_put(array)
    return jax.device_put(array, placement)


def runtime_device_put(value, *, dtype=None, target=None, device=None) -> jax.Array:
    """Place host values on a JAX device using runtime policy for float dtypes."""
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
    """
    placement = _device_put_target(target, device)
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
    """Place an explicitly typed value without applying runtime float policy."""
    return _device_put(
        value,
        dtype=dtype,
        target=target,
        device=device,
        preserve_float_dtype=True,
    )


def _compute_device_put(value, *, dtype, target=None, device=None) -> jax.Array:
    resolved_dtype = _resolve_np_dtype(dtype, source="dtype")
    return _device_put_preserving_dtype(
        value,
        dtype=resolved_dtype,
        target=target,
        device=device,
    )


def as_jax_array(value, *, dtype) -> jax.Array:
    if _has_jax_array_value(value):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return runtime_device_put(value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype)


def as_jax_float64(value) -> jax.Array:
    return as_runtime_array(value)


def as_jax_int32(value) -> jax.Array:
    return as_jax_array(value, dtype=jnp.int32)


def as_runtime_array(value, *, dtype=None, reference=None):
    resolved_dtype = _resolve_jnp_dtype(dtype, source="dtype")
    reference_sharding = _reference_sharding(reference, ndim=_value_ndim(value))
    if reference_sharding is not None and not _has_only_traced_jax_leaves(value):
        return runtime_device_put(
            value, dtype=resolved_dtype, target=reference_sharding
        )
    return as_jax_array(value, dtype=resolved_dtype)


def as_compute_array(value, *, dtype=None, reference=None) -> jax.Array:
    """Place proposal data using compute dtype and optional reference sharding."""
    resolved_dtype = (
        compute_jnp_dtype()
        if dtype is None
        else _resolve_jnp_dtype(dtype, source="dtype")
    )
    if _has_only_traced_jax_leaves(value):
        return jnp.asarray(value, dtype=resolved_dtype)
    reference_sharding = _reference_sharding(reference, ndim=_value_ndim(value))
    if reference_sharding is not None:
        return _compute_device_put(
            value,
            dtype=resolved_dtype,
            target=reference_sharding,
        )
    if _has_jax_array_value(value):
        return jnp.asarray(value, dtype=resolved_dtype)
    return _compute_device_put(value, dtype=resolved_dtype)


def as_runtime_float64(value, *, reference):
    return as_runtime_array(value, reference=reference)


def commit_in_place(array: jax.Array) -> jax.Array:
    """Commit a concrete array to the device or mesh it already lives on.

    For a loop that feeds a jitted step its own outputs: ``jit`` keys
    committed and uncommitted arguments separately, and a step's outputs are
    committed whenever an input is, so the first input is committed up front
    to compile the executable every later call reuses.
    """
    sharding = array.sharding
    placement = (
        sharding
        if isinstance(sharding, NamedSharding)
        else _single_device_placement(sharding)
    )
    return _device_put_preserving_dtype(array, dtype=array.dtype, target=placement)


def explicit_device_array(
    value,
    *,
    dtype,
    reference=None,
    target=None,
    device=None,
) -> jax.Array:
    """Place an exact-dtype array using one explicit or reference placement."""
    if reference is not None and (target is not None or device is not None):
        raise TypeError(
            "explicit_device_array accepts reference or explicit target/device, not both."
        )
    reference_placement = _reference_placement(reference, ndim=_value_ndim(value))
    return _device_put_preserving_dtype(
        value,
        dtype=dtype,
        target=reference_placement if reference is not None else target,
        device=device,
    )
