"""Registration of frozen dataclasses as JAX pytrees."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, dataclass_transform

import jax

_T = TypeVar("_T")


def _frozen_dataclass(cls: type[_T]) -> type[_T]:
    if "__dataclass_fields__" in cls.__dict__:
        if not getattr(cls, "__dataclass_params__").frozen:
            raise TypeError(f"{cls.__name__} must be a frozen dataclass")
        return cls
    return dataclass(frozen=True)(cls)


@dataclass_transform(frozen_default=True)
def pytree_dataclass(
    *, data: tuple[str, ...], meta: tuple[str, ...] = ()
) -> Callable[[type[_T]], type[_T]]:
    """Freeze a class and register it as a JAX pytree.

    Leaves follow ``data`` order; ``meta`` fields are static metadata.

    Args:
        data (tuple[str, ...]): Ordered names of dynamic data fields.
        meta (tuple[str, ...]): Names of static, hashable metadata fields; together with
            data, a complete disjoint partition of constructor fields.

    Returns:
        Callable[[type], type]: Decorator that freezes and registers the class
            while preserving existing frozen-dataclass options.
    """

    def decorate(cls: type[_T]) -> type[_T]:
        cls = _frozen_dataclass(cls)
        data_names, meta_names = set(data), set(meta)
        if len(data_names) != len(data):
            raise ValueError(f"{cls.__name__}: duplicate names in data")
        if len(meta_names) != len(meta):
            raise ValueError(f"{cls.__name__}: duplicate names in meta")
        jax.tree_util.register_dataclass(
            cls, data_fields=list(data), meta_fields=list(meta)
        )
        return cls

    return decorate
