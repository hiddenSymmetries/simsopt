"""The compiled solver loops of ``simsopt_jax.core.boozer_solvers`` at a singular second step.

No Boozer problem gives an exactly singular matrix after a regular first
step, so each loop is driven by a synthetic two-unknown system substituted
for its formulation: the matrix is the identity at the start and loses its
second pivot once the first unknown reaches zero, which the first step
does exactly. Native's ``np.linalg.solve`` raises on that matrix; the loop
must stop there, flag ``singular`` and keep the iterate it reached (not
step past it), on each lane.
"""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    fixture_parity_lane,  # noqa: F401
    host_array,
    parity_default_device,
)

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax.core import boozer_solvers

_START = np.array([1.0, 0.0])
_TOL = 1e-12
_MAXITER = 10.0


def _matrix(x: jax.Array) -> jax.Array:
    """``diag(1, x[0])``: the identity at the start, singular once ``x[0] = 0``."""
    return jnp.diag(jnp.stack((jnp.ones_like(x[0]), x[0])))


def _exact_system(problem, x, residual_rows, *, derivatives=0):
    return jnp.ones(2, x.dtype), _matrix(x)


def _exact_final_residual(problem, x, *, optimize_G, weight_inv_modB=False):
    return (x,)


def _penalty_derivatives(problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True):
    return jnp.zeros((), x.dtype), jnp.ones(2, x.dtype), _matrix(x)


def _penalty_residuals(problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True):
    # J^T r = [2, x[0]]: the damped first step (lam = 1) is [1, 1/2].
    return jnp.array([2.0, 1.0], x.dtype), _matrix(x)


# Per loop: the formulation(s) substituted, the iterate after the first step and
# the matrix native's np.linalg.solve factorises at the second step.
_CASES = {
    "exact": (
        {"boozer_exact_residual": _exact_system, "boozer_surface_residual": _exact_final_residual},
        np.array([0.0, -1.0]),
        np.diag([1.0, 0.0]),
    ),
    "penalty-newton": (
        {"boozer_penalty_constraints": _penalty_derivatives},
        np.array([0.0, -1.0]),
        np.diag([1.0, 0.0]),
    ),
    "gauss-newton": (
        {"boozer_penalty_residual": _penalty_residuals},
        np.array([0.0, -0.5]),
        np.diag([1.0 + 1.0 / 3.0, 0.0]),
    ),
}


def _run(name: str, x: jax.Array):
    """The loop ``name`` compiled afresh (so it traces the substituted formulation)."""
    tol, maxiter = jnp.asarray(_TOL), jnp.asarray(_MAXITER)
    if name == "exact":
        loop = jax.jit(inspect.unwrap(boozer_solvers.boozer_exact_newton), static_argnames=("G_from_currents",))
        return loop(None, x, jnp.arange(2, dtype=jnp.int32), tol, maxiter)
    options = {"optimize_G": True, "weight_inv_modB": False}
    if name == "penalty-newton":
        loop = jax.jit(inspect.unwrap(boozer_solvers.boozer_penalty_newton), static_argnames=tuple(options))
        return loop(None, x, tol, maxiter, jnp.asarray(0.0), **options)
    loop = jax.jit(inspect.unwrap(boozer_solvers.boozer_penalty_gauss_newton), static_argnames=tuple(options))
    return loop(None, x, tol, maxiter, **options)


@pytest.mark.parametrize("name", _CASES)
def test_loop_stops_at_a_singular_second_step_and_keeps_its_iterate(name, parity_lane, monkeypatch):
    formulations, reached, second_matrix = _CASES[name]
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        np.linalg.solve(second_matrix, np.ones(2))
    for attribute, formulation in formulations.items():
        monkeypatch.setattr(boozer_solvers, attribute, formulation)
    with parity_default_device(parity_lane):
        result = _run(name, jnp.asarray(_START))
        singular, iterations, x = (host_array(value) for value in (result.singular, result.iterations, result.x))
    assert bool(singular), f"{name}: the singular second step was not flagged"
    assert int(iterations) == 1, f"{name}: {int(iterations)} steps counted; native raises during the second"
    np.testing.assert_array_equal(x, reached, err_msg=f"{name}: the loop did not keep the iterate it reached")
