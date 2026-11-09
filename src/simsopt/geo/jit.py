import inspect
import operator
import os
from collections import OrderedDict, defaultdict
from functools import wraps

import jax
import numpy as np
from jax import jit as jaxjit

from .config import parameters

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    jax.config.update("jax_platform_name", "cpu")


def native_jax_device():
    """Device on which native simsopt runs the functions compiled by :func:`jit`.

    Returns:
        jax.Device object: First CPU device when CPU is enabled, otherwise the
            first default device (for example with ``JAX_PLATFORMS=cuda``).
    """
    platforms = jax.config.jax_platforms
    if platforms and "cpu" not in platforms.split(","):
        return jax.devices()[0]
    return jax.devices("cpu")[0]


def _owned(value):
    # Native caches can overwrite NumPy views while JAX still reads them.
    return value.copy(order="K") if isinstance(value, np.ndarray) else value


def _owned_tree(value):
    # Bare arrays are the common leaf case; skip pytree traversal for them.
    if type(value) is np.ndarray:
        return value.copy(order="K")
    if isinstance(value, jax.Array):
        return value
    return jax.tree_util.tree_map(_owned, value)


def _owned_plain_leaf(value):
    if type(value) in (dict, defaultdict, OrderedDict):
        # These built-in containers have known copy/setitem contracts.
        result = value.copy()
        for key, item in value.items():
            result[key] = _snapshot_plain(item)
        return result
    return _owned(value)


def _snapshot_plain(value):
    # Treat dicts as leaves so JAX does not sort their keys in plain Python calls.
    return jax.tree_util.tree_map(
        _owned_plain_leaf,
        value,
        is_leaf=lambda item: type(item) in (dict, defaultdict, OrderedDict),
    )


def _static_arguments(fun, options):
    """Resolve both static selectors once and hand both to :func:`jax.jit`.

    Given one selector, JAX infers the other from the signature, and JAX 0.12
    changed that inference (a negative index now names its parameter and is
    resolved against the signature rather than the call). The selectors are
    resolved here as JAX 0.12 does and both are passed explicitly, so every
    supported JAX compiles exactly the arguments this wrapper leaves uncopied.
    """
    argnums = options.get("static_argnums")
    argnames = options.get("static_argnames")
    if argnums is not None:
        # Match JAX's scalar __index__ protocol before considering an iterable.
        try:
            argnums = (operator.index(argnums),)
        except TypeError:
            argnums = tuple(map(operator.index, argnums))
    if argnames is not None:
        argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)

    try:
        signature = inspect.signature(fun)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        parameters = tuple(signature.parameters.values())
        named = {
            i: param.name
            for i, param in enumerate(parameters)
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        }
        if argnums is None and argnames is not None:
            argnums = tuple(i for i, name in named.items() if name in argnames)
        elif argnames is None and argnums is not None:
            argnames = tuple(
                name
                for i, name in named.items()
                if i in argnums or i - len(parameters) in argnums
            )
        positional = sum(
            param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for param in parameters
        )
        if argnums and not any(
            param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters
        ):
            # Out-of-range indices stay as given so that JAX rejects them.
            argnums = tuple(
                i + positional if -positional <= i < 0 else i for i in argnums
            )
    argnums, argnames = argnums or (), argnames or ()
    compile_options = {**options, "static_argnums": argnums, "static_argnames": argnames}
    return frozenset(argnums), frozenset(argnames), compile_options


def jit(fun, **args):
    """Compile ``fun`` with :func:`jax.jit` for native simsopt.

    Dynamic NumPy pytree leaves are private copies, preserving ndarray subclasses.
    Static compiled arguments retain their identity and compilation cache keys.
    Calls run under ``jax.default_device(native_jax_device())``, also with jit
    disabled, so uncommitted inputs land there; committed JAX inputs keep their
    placement. With jit disabled, all NumPy leaves are copied.
    Plain dict, defaultdict and OrderedDict inputs preserve their key order;
    other registered pytrees use JAX traversal. Opaque unregistered container
    leaves retain their identity and contents. Captured arrays and JAX inputs
    already aliasing external mutable storage are outside this ownership boundary.
    Disabling jit still returns a wrapper, with
    ``fun`` available as ``__wrapped__`` and its metadata retained.

    Args:
        fun (callable): Function to compile.
        **args: Keyword arguments forwarded to :func:`jax.jit`.

    Returns:
        callable: The input-owning wrapper, compiled when jit is enabled.
    """
    use_jit = parameters["jit"]
    if use_jit:
        static_argnums, static_argnames, compile_options = _static_arguments(fun, args)
        run = jaxjit(fun, **compile_options)
    else:
        static_argnums, static_argnames = (), ()
        run = fun

    @wraps(fun)
    def on_native_device(*fun_args, **fun_kwargs):
        if not use_jit:
            with jax.default_device(native_jax_device()):
                return run(
                    *tuple(_snapshot_plain(value) for value in fun_args),
                    **{name: _snapshot_plain(value) for name, value in fun_kwargs.items()},
                )
        if static_argnums:
            count = len(fun_args)
            fun_args = tuple(
                value
                if i in static_argnums or i - count in static_argnums
                else _owned_tree(value)
                for i, value in enumerate(fun_args)
            )
        else:
            fun_args = tuple(map(_owned_tree, fun_args))
        fun_kwargs = {
            name: value if name in static_argnames else _owned_tree(value)
            for name, value in fun_kwargs.items()
        }
        with jax.default_device(native_jax_device()):
            return run(*fun_args, **fun_kwargs)

    return on_native_device
