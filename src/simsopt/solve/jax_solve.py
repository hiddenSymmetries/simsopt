# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
JAX-native least squares solver.

This module avoids finite differences and relies on JAX autodiff.
If jaxopt is available, it will be used for LBFGS; otherwise a simple
gradient-descent fallback is used.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional dependency
    jax = None
    jnp = None
    _import_error = exc
else:
    _import_error = None

try:
    import jaxopt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    jaxopt = None

try:
    from scipy.optimize import least_squares as _scipy_least_squares  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _scipy_least_squares = None

__all__ = ["least_squares_jax_solve"]


def _require_jax():
    """Raise a helpful error when the optional JAX dependency is unavailable."""
    if jax is None or jnp is None:
        raise ImportError(
            "least_squares_jax_solve requires JAX. "
            "Install jax/jaxlib before using this solver."
        ) from _import_error


def _normalize_scale(x0, x_scale):
    """Return a nonzero parameter scaling vector in solver coordinates."""
    if x_scale is None:
        return jnp.ones_like(x0)
    scale = jnp.asarray(x_scale, dtype=x0.dtype)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    return scale


def least_squares_jax_solve(
    residual_fun: Callable[[jnp.ndarray], jnp.ndarray],
    x0,
    *,
    method: str = "scipy",
    max_nfev: int = 50,
    gtol: float = 1e-7,
    step_size: float = 1e-2,
    x_scale=None,
    jit: bool = True,
    verbose: int = 1,
    jac: str | Callable | None = None,
) -> Dict[str, object]:
    """Solve a nonlinear least-squares problem with JAX derivatives.

    The function is intentionally lightweight: it accepts a residual callback
    defined in terms of the physical parameter vector ``x`` and dispatches to
    one of three backends:

    - ``method="scipy"`` uses :func:`scipy.optimize.least_squares`, with either
      a JAX Jacobian or SciPy finite differences.
    - ``method="lbfgs"`` uses :mod:`jaxopt` when available.
    - ``method="gradient_descent"`` runs a simple JAX-native fallback loop.

    Parameters
    ----------
    residual_fun:
        Callable returning the residual vector for a given parameter vector.
    x0:
        Initial parameter vector in physical coordinates.
    method:
        Solver backend name.
    max_nfev:
        Maximum number of residual or objective evaluations.
    gtol:
        Termination tolerance applied to the gradient norm, and passed through
        to SciPy's ``xtol`` / ``ftol`` / ``gtol`` for consistency.
    step_size:
        Gradient-descent step size when using the fallback backend.
    x_scale:
        Optional parameter scaling, matching the role of ``x_scale`` in
        :func:`scipy.optimize.least_squares`.
    jit:
        Whether to JIT-compile the residual / Jacobian path where supported.
    verbose:
        Verbosity level. ``0`` disables backend progress printing.
    jac:
        For ``method="scipy"``, either ``None`` / ``"jax"`` for a JAX Jacobian
        or any SciPy finite-difference keyword such as ``"2-point"``.

    Returns
    -------
    dict
        Dictionary containing the optimized parameters, success flag, cost,
        gradient, and evaluation count.
    """
    _require_jax()

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    scale = _normalize_scale(x0, x_scale)
    y0 = x0 / scale

    def residuals_y(y):
        return residual_fun(y * scale)

    def objective_y(y):
        r = residuals_y(y)
        return 0.5 * jnp.sum(r * r)

    if method in ("scipy", "least_squares"):
        if _scipy_least_squares is None:
            if jaxopt is not None and verbose:
                print("scipy not available; falling back to jaxopt LBFGS.")
                method = "lbfgs"
            else:
                raise ImportError("scipy is required for method='scipy'.")
        else:
            jac_scipy = jac
            if jac_scipy is None:
                jac_scipy = "jax"
            if jac_scipy == "jax":
                if jit:
                    residual_fun = jax.jit(residual_fun)
                jac_builder = jax.jacfwd if jit else jax.jacrev
                jac_fun = jac_builder(residual_fun)
                if jit:
                    jac_fun = jax.jit(jac_fun)
            else:
                jac_fun = None

            def residuals_y(y):
                return residual_fun(y * scale)

            def jac_y(y):
                J = jac_fun(y * scale)
                return J * scale[None, :]

            verbose_scipy = 2 if verbose else 0
            res = _scipy_least_squares(
                lambda y: np.asarray(residuals_y(jnp.asarray(y))),
                np.asarray(y0),
                jac=(lambda y: np.asarray(jac_y(jnp.asarray(y)))) if jac_scipy == "jax" else jac_scipy,
                max_nfev=int(max_nfev),
                xtol=float(gtol),
                ftol=float(gtol),
                gtol=float(gtol),
                verbose=verbose_scipy,
            )
            y_opt = jnp.asarray(res.x)
            grad = jnp.asarray(res.jac.T @ res.fun) if res.jac is not None else jnp.zeros_like(y_opt)
            nfev = int(res.nfev)
            status = 1 if res.success else 0
            cost = float(0.5 * np.sum(res.fun * res.fun))
    if method == "lbfgs" and jaxopt is not None:
        solver = jaxopt.LBFGS(fun=objective_y, maxiter=int(max_nfev), tol=float(gtol))
        result = solver.run(y0)
        y_opt = result.params
        grad = result.state.grad
        nfev = int(result.state.iter_num)
        status = 1 if result.state.error <= gtol else 0
        cost = float(objective_y(y_opt))
    elif method not in ("scipy", "least_squares"):
        if method not in ("lbfgs", "gradient_descent"):
            raise ValueError("method must be 'lbfgs' or 'gradient_descent'")
        if method == "lbfgs" and jaxopt is None and verbose:
            print("jaxopt not available; falling back to gradient descent.")

        value_and_grad = jax.value_and_grad(objective_y)
        if jit:
            value_and_grad = jax.jit(value_and_grad)
        y = y0
        status = 0
        cost = float(objective_y(y))
        grad = jnp.zeros_like(y)
        nfev = 0
        for k in range(int(max_nfev)):
            cost, grad = value_and_grad(y)
            grad_norm = float(jnp.linalg.norm(grad))
            nfev = k + 1
            if verbose:
                print(f"iter {k:3d}  cost={float(cost):.6e}  grad_norm={grad_norm:.3e}")
            if grad_norm < gtol:
                status = 1
                break
            y = y - float(step_size) * grad
        y_opt = y
        cost = float(objective_y(y_opt))

    x_opt = np.asarray(y_opt * scale)
    return {
        "x": x_opt,
        "status": int(status),
        "success": bool(status == 1),
        "nfev": int(nfev),
        "cost": float(cost),
        "grad": np.asarray(grad),
    }
