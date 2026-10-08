"""Central registration for frozen dataclasses and custom JAX pytree nodes."""

from collections.abc import Callable
from dataclasses import Field, dataclass, fields
from typing import ClassVar, Protocol, TypeVar, cast, dataclass_transform

import jax

class _DataclassFields(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[object]]]


_T = TypeVar("_T")
_REGISTERED_CLASSES: list[type[object]] = []


def registered_pytree_classes() -> tuple[type[object], ...]:
    """Return an immutable snapshot of classes registered through this helper."""
    return tuple(_REGISTERED_CLASSES)


def _frozen_dataclass(cls: type[_T]) -> type[_T]:
    if "__dataclass_fields__" in cls.__dict__:
        if not getattr(cls, "__dataclass_params__").frozen:
            raise TypeError(f"{cls.__name__} must be a frozen dataclass")
        return cls
    return dataclass(frozen=True)(cls)


@dataclass_transform(frozen_default=True)
def pytree_node(cls: type[_T]) -> type[_T]:
    """Freeze a class and register its custom flatten/unflatten contract.

    Use this when constructor or reconstruction guarantees cannot be expressed
    by a dataclass data/meta partition. Class options and methods are preserved.
    """
    cls = _frozen_dataclass(cls)
    jax.tree_util.register_pytree_node_class(cls)
    _REGISTERED_CLASSES.append(cls)
    return cls


@dataclass_transform(frozen_default=True)
def pytree_dataclass(
    *, data: tuple[str, ...], meta: tuple[str, ...] = ()
) -> Callable[[type[_T]], type[_T]]:
    """Freeze a class and register its complete, disjoint init-field partition.

    Existing frozen dataclasses retain their options. Leaves follow ``data``
    order; ``meta`` fields form static JAX metadata.
    """

    def decorate(cls: type[_T]) -> type[_T]:
        cls = _frozen_dataclass(cls)
        data_names, meta_names = set(data), set(meta)
        if len(data_names) != len(data):
            raise ValueError(f"{cls.__name__}: duplicate names in data")
        if len(meta_names) != len(meta):
            raise ValueError(f"{cls.__name__}: duplicate names in meta")
        overlap = data_names & meta_names
        if overlap:
            raise ValueError(f"{cls.__name__}: data and meta overlap: {sorted(overlap)}")
        init_names = {field.name for field in fields(cast(type[_DataclassFields], cls)) if field.init}
        partition_names = data_names | meta_names
        if partition_names != init_names:
            missing = sorted(init_names - partition_names)
            unknown = sorted(partition_names - init_names)
            raise ValueError(
                f"{cls.__name__}: data/meta must partition init fields; "
                f"missing={missing}, unknown={unknown}"
            )
        jax.tree_util.register_dataclass(
            cls, data_fields=list(data), meta_fields=list(meta)
        )
        _REGISTERED_CLASSES.append(cls)
        return cls

    return decorate
