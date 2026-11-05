import functools

import jax
import numpy as np
jax.config.update('jax_platform_name', 'cpu')
from jax import jit as jaxjit
from .config import parameters


def _copy_numpy_leaves(x):
    if isinstance(x, np.ndarray):
        return np.array(x)
    if isinstance(x, (list, tuple, dict)):
        return jax.tree_util.tree_map(_copy_numpy_leaves, x)
    return x


def jit(fun, **args):
    """
    Wrap ``fun`` with ``jax.jit`` (if ``parameters['jit']`` is set), copying
    every ``np.ndarray`` argument before the call.

    The copy prevents a silent race between JAX and simsopt's caches. simsopt
    caches derived arrays such as ``gamma()``, ``gammadash()`` and ``kappa()``,
    and when the DOFs or evaluation points change it writes the new values into
    the same arrays in place. JAX dispatches computations asynchronously: a call
    returns as soon as the work is queued, and the result is filled in later.
    On CPU, JAX also does not always copy a NumPy argument when the call is
    made. It may read an aligned array's memory directly while the computation
    runs, and it may copy a large unaligned array in the background. So if a
    caller passes a cached array, keeps the result, and then changes the DOFs
    before the computation finishes, the computation can read the new values.
    The result then reflects some or all of the new state, without any error
    or warning.

    This wrapper copies every NumPy array argument before it calls ``fun``, so
    the computation reads private memory that no simsopt cache will overwrite.
    It copies arrays passed as positional or keyword arguments, including
    arrays nested in lists, tuples and dicts, and passes everything else
    through unchanged: JAX arrays, tracers and static Python values. The copy
    has limits. It costs time and memory in proportion to the argument sizes on
    every call. It does not reach NumPy arrays that ``fun`` captures from an
    enclosing scope or module, or arrays held inside other kinds of objects;
    code that captures a mutable array must copy it itself. It also cannot
    protect a JAX array that already shares memory with a NumPy array, such as
    one created with ``jax.device_put`` from an aligned NumPy array.

    Results such as ``LpCurveCurvature.J()``, ``LpCurveTorsion.J()`` and the
    vector-Jacobian products of ``JaxCurve``, for example
    ``dkappa_by_dcoeff_vjp()``, are JAX arrays that may still be computing
    when they are returned. The wrapper deliberately does not convert them to
    NumPy, because that would wait for every computation to finish at this
    boundary. Because the inputs were copied, a caller can hold such a result
    across later DOF or point changes, and it keeps the value computed from
    the state at the time of the call.
    """
    f = jaxjit(fun, **args) if parameters['jit'] else fun

    @functools.wraps(fun)
    def wrapped(*a, **kw):
        a = [_copy_numpy_leaves(x) for x in a]
        if kw:
            kw = {k: _copy_numpy_leaves(v) for k, v in kw.items()}
        return f(*a, **kw)

    return wrapped
