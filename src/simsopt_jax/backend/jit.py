"""Device-neutral ``jax.jit`` that owns its NumPy arguments before dispatch.

simsopt overwrites cached NumPy arrays (``gamma()``, field caches, evaluation
points) in place when DOFs or points change. ``jax.jit`` dispatches
asynchronously, and on CPU it may read an aligned NumPy argument's memory
directly, or copy an unaligned one in the background, after the call has
returned. A cache overwritten in that window silently changes the result.
:func:`jit` copies every NumPy array leaf of the dynamic arguments
synchronously, through ``snapshot_host_tree`` (the host-ownership owner),
before calling the compiled function.

Placement is left to JAX: no device is pinned, so NumPy and uncommitted
inputs land on JAX's default device (the one ``set_backend`` selects) and
committed inputs keep their device, exactly as with ``jax.jit``.
"""

from __future__ import annotations

import inspect
import operator
import types
from collections.abc import Callable, Iterable
from functools import update_wrapper
from itertools import repeat
from typing import overload

import jax
import numpy as np

from simsopt_jax.runtime.host_boundary import snapshot_host_tree

__all__ = ["CopyingJit", "jit"]

# Argument types that never hold a NumPy array; JAX arrays include tracers.
_NUMPY_FREE_TYPES = (jax.Array, bool, int, float, complex, str, type(None))


def _index_tuple(argnums) -> tuple[int, ...]:
    """``static_argnums`` as a tuple of ints, accepting one index or several."""
    try:
        return (operator.index(argnums),)
    except TypeError:
        return tuple(map(operator.index, argnums))


def _str_tuple(argnames: str | Iterable[str]) -> tuple[str, ...]:
    return (argnames,) if isinstance(argnames, str) else tuple(argnames)


def _static_selectors(
    fun: Callable,
    argnums: int | Iterable[int] | None,
    argnames: str | Iterable[str] | None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """The positional indices and keyword names ``jax.jit`` treats as static.

    Resolved once, as JAX 0.12 does (``resolve_argnums``) and as
    ``simsopt.geo.jit`` does: a selector given alone implies the other through
    ``fun``'s positional-or-keyword parameters, and a negative index names its
    parameter in the signature, so it is made non-negative here. Only for a
    ``fun`` with ``*args`` does it stay negative and count from each call's
    last positional argument. A callable without an inspectable signature
    (some builtins) implies nothing.
    """
    nums = () if argnums is None else _index_tuple(argnums)
    names = () if argnames is None else _str_tuple(argnames)
    try:
        parameters = tuple(inspect.signature(fun).parameters.values())
    except (TypeError, ValueError):
        return nums, names
    named = [
        (i, parameter.name)
        for i, parameter in enumerate(parameters)
        if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    if argnums is None and argnames is not None:
        nums = tuple(i for i, name in named if name in names)
    elif argnames is None and argnums is not None:
        names = tuple(name for i, name in named if i in nums or i - len(parameters) in nums)
    if not any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        positional = sum(
            parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in parameters
        )
        # Out-of-range indices stay as given so that jax.jit rejects them.
        nums = tuple(i + positional if -positional <= i < 0 else i for i in nums)
    return nums, names


def _holds_numpy(value) -> bool:
    """Whether ``value`` is, or is a pytree with, a NumPy array leaf."""
    if isinstance(value, _NUMPY_FREE_TYPES):
        return False
    if isinstance(value, np.ndarray):
        return True
    return any(map(isinstance, jax.tree.leaves(value), repeat(np.ndarray)))


class CopyingJit:
    """A function compiled with ``jax.jit`` whose NumPy arguments are copied per call.

    Calls copy the NumPy array leaves of dynamic arguments, rebuilding the
    containers that hold them, and pass static arguments and every other leaf
    (JAX arrays, tracers, Python scalars) through by identity, so the
    compilation cache keys are those of ``jax.jit``. Calls without NumPy
    leaves skip the copy. Attributes of the
    compiled function (``lower``, ``trace``, ``eval_shape``,
    ``clear_cache``, ...) are forwarded unchanged, and the wrapper binds as a
    method like the compiled function does.

    Not protected: NumPy arrays that ``fun`` captures from an enclosing scope
    and JAX arrays that already alias NumPy memory.
    """

    def __init__(self, fun: Callable, **options) -> None:
        """Wrap ``jax.jit(fun, **options)``, which compiles at the first call.

        Args:
            fun (callable): Function to compile.
            **options: Keyword arguments of :func:`jax.jit`, for example
                ``static_argnums``, ``static_argnames`` or ``donate_argnums``.
        """
        # First, so the metadata copied from ``fun`` cannot replace the state below.
        update_wrapper(self, fun)
        argnums = options.get("static_argnums")
        argnames = options.get("static_argnames")
        nums, names = _static_selectors(fun, argnums, argnames)
        if argnums is not None or argnames is not None:
            # jax.jit receives the selectors this wrapper copies by, so both
            # agree by construction, and an iterator selector is read once.
            options = {**options, "static_argnums": nums, "static_argnames": names}
        self._compiled = jax.jit(fun, **options)  # noqa: TID251 the one raw jit; this wrapper owns NumPy inputs
        self._static_argnums = frozenset(nums)
        self._static_argnames = frozenset(names)

    def __call__(self, *args, **kwargs):
        """Copy NumPy leaves of dynamic arguments, then call the compiled function.

        Args:
            *args: Positional arguments of ``fun``; arrays of any shape.
            **kwargs: Keyword arguments of ``fun``; arrays of any shape.

        Returns:
            pytree: The compiled function's outputs, placed by JAX.
        """
        static, count = self._static_argnums, len(args)
        if any(
            _holds_numpy(value)
            for i, value in enumerate(args)
            if i not in static and i - count not in static
        ):
            args = tuple(
                value if i in static or i - count in static else snapshot_host_tree(value)
                for i, value in enumerate(args)
            )
        names = self._static_argnames
        if kwargs and any(
            _holds_numpy(value) for name, value in kwargs.items() if name not in names
        ):
            kwargs = {
                name: value if name in names else snapshot_host_tree(value)
                for name, value in kwargs.items()
            }
        return self._compiled(*args, **kwargs)

    def __get__(self, instance, owner=None):
        """Bind to ``instance`` as a method, as ``jax.jit`` functions do.

        Args:
            instance (object or None): Instance the attribute is read from;
                None for a read from the class.
            owner (type or None): Class the attribute is read from.

        Returns:
            CopyingJit or types.MethodType: This wrapper for a class read, or
                this wrapper bound to ``instance``.
        """
        return self if instance is None else types.MethodType(self, instance)

    def __getattr__(self, name: str):
        """Forward attributes of the compiled function, such as ``lower``.

        Args:
            name (str): Attribute missing from this wrapper.

        Returns:
            object: The compiled function's attribute ``name``.

        Raises:
            AttributeError: The compiled function has no attribute ``name``.
        """
        # ``_compiled`` reaches here only on an instance made without
        # ``__init__`` (``copy.copy``); raising avoids infinite recursion.
        if name == "_compiled":
            raise AttributeError(name)
        return getattr(self._compiled, name)


@overload
def jit(fun: Callable, /, **options) -> CopyingJit: ...


@overload
def jit(fun: None = None, /, **options) -> Callable[[Callable], CopyingJit]: ...


def jit(
    fun: Callable | None = None, /, **options
) -> CopyingJit | Callable[[Callable], CopyingJit]:
    """``jax.jit`` that copies NumPy arguments before asynchronous dispatch.

    Use it wherever ``jax.jit`` would be used: as ``@jit``,
    ``@partial(jit, static_argnames=...)``, ``@jit(static_argnames=...)`` or
    ``jit(f, static_argnums=...)``. See :class:`CopyingJit` for what is copied.

    Args:
        fun (callable or None): Function to compile; None returns a decorator
            that applies ``options``.
        **options: Keyword arguments of :func:`jax.jit`.

    Returns:
        CopyingJit or callable: The compiled wrapper, or a decorator producing
            one when ``fun`` is None.
    """
    if fun is None:
        return lambda f: CopyingJit(f, **options)
    return CopyingJit(fun, **options)
