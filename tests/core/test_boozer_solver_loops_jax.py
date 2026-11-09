"""The compiled solver loops of ``simsopt_jax.core.boozer_solvers`` at a singular
second step and on a growing walk.

No Boozer problem gives an exactly singular matrix after a regular first
step, so each loop is driven by a synthetic two-unknown system substituted
for its formulation: the matrix is the identity at the start and loses its
second pivot once the first unknown reaches zero, which the first step
does exactly. Native's ``np.linalg.solve`` raises on that matrix; the loop
must stop there, flag ``singular`` and keep the iterate it reached (not
step past it, and not restore its start although its norm grew, since
native raises before that), on each lane. A second synthetic system, ``x`` with the
constant matrix ``0.05 I``, multiplies the norm at every step (native's
``test_penalty_newton_divergence_factor`` uses it): as natively, the loop
runs to ``maxiter``, reports the final norm and restores its start with the
values there.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core import boozer_solvers
from unittest import mock

from unittest_jax_support import host_array, parity_default_device

from collections.abc import Callable
import inspect

import numpy as np


_START = np.array([1.0, 0.0])
_TOL = 1e-12
_MAXITER = 10.0


def _matrix(x: jax.Array) -> jax.Array:
    """``diag(1, x[0])``: the identity at the start, singular once ``x[0] = 0``."""
    return jnp.diag(jnp.stack((jnp.ones_like(x[0]), x[0])))


def _growth(x: jax.Array) -> jax.Array:
    """``1 + x[1]^2``: 1 at the start, 2 at the first exact and Newton iterate,
    so those walks end at a larger norm than they started with."""
    return 1 + x[1] ** 2


def _exact_system(problem, x, residual_rows, *, derivatives=0):
    return _growth(x) * jnp.ones(2, x.dtype), _matrix(x)


def _exact_final_residual(problem, x, *, optimize_G, weight_inv_modB=False):
    return (x,)


def _penalty_derivatives(
    problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True
):
    return jnp.zeros((), x.dtype), _growth(x) * jnp.ones(2, x.dtype), _matrix(x)


def _penalty_residuals(
    problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True
):
    # J^T r = [2, x[0]] at the start: the damped first step (lam = 1) is [1, 1/2];
    # at the iterate it reaches, |J^T r| = 2.5 > |[2, 1]|.
    return _growth(x) * jnp.array([2.0, 1.0], x.dtype), _matrix(x)


# Per loop: the formulation(s) substituted, the iterate after the first step and
# the matrix native's np.linalg.solve factorises at the second step.
_CASES = {
    "exact": (
        {
            "boozer_exact_residual": _exact_system,
            "boozer_surface_residual": _exact_final_residual,
        },
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


_GROWTH = 0.05


def _growing_exact_system(problem, x, residual_rows, *, derivatives=0):
    # Refined Newton steps x - x / 0.05 = -19 x.
    return x, _GROWTH * jnp.eye(2, dtype=x.dtype)


def _growing_penalty_derivatives(
    problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True
):
    # Newton steps x - x / 0.05 = -19 x.
    return 0.5 * x @ x, x, _GROWTH * jnp.eye(2, dtype=x.dtype)


def _growing_penalty_residuals(
    problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True
):
    # J^T r = 0.05 x and J^T J = 0.0025 I: the damped first step (lam = 1) gives -9 x.
    return x, _GROWTH * jnp.eye(2, dtype=x.dtype)


# Per loop: the formulation substituted, the norm after two steps (BoozerExact
# and penalty Newton: x = 361 x0; Gauss-Newton: steps 10 x and 15 x give x = 126 x0, where
# |J^T r| = 0.05 * 126) and the values at the start that the restored result carries.
_GROWING_CASES = {
    "exact": (
        {
            "boozer_exact_residual": _growing_exact_system,
            "boozer_surface_residual": _exact_final_residual,
        },
        361.0,
        {"residual": _START, "jacobian": _GROWTH * np.eye(2)},
    ),
    "penalty-newton": (
        {"boozer_penalty_constraints": _growing_penalty_derivatives},
        361.0,
        {"gradient": _START, "hessian": _GROWTH * np.eye(2)},
    ),
    "gauss-newton": (
        {"boozer_penalty_residual": _growing_penalty_residuals},
        _GROWTH * 126.0,
        {
            "residual": _START,
            "gradient": _GROWTH * _START,
            "normal_matrix": _GROWTH**2 * np.eye(2),
        },
    ),
}


def _compiled_afresh(solver, static_argnames: tuple[str, ...]):
    """``solver``'s body compiled anew, so it traces the substituted formulation."""
    body: Callable[..., object] = inspect.unwrap(solver)
    return jit(body, static_argnames=static_argnames)


def _substitute(patches, formulations):
    """Patch each ``boozer_solvers`` formulation for the rest of the case."""
    for attribute, formulation in formulations.items():
        patches.enter_context(mock.patch.object(boozer_solvers, attribute, formulation))


def _run(name: str, x: jax.Array, maxiter: float = _MAXITER, divergence_factor=None):
    """The loop ``name`` compiled afresh (so it traces the substituted formulation);
    ``divergence_factor`` goes to the penalty Newton."""
    tol, cap = jnp.asarray(_TOL), jnp.asarray(maxiter)
    if name == "exact":
        loop = _compiled_afresh(boozer_solvers.boozer_exact_newton, ("G_from_currents",))
        return loop(None, x, jnp.arange(2, dtype=jnp.int32), tol, cap)
    options = {"optimize_G": True, "weight_inv_modB": False}
    if name == "penalty-newton":
        loop = _compiled_afresh(boozer_solvers.boozer_penalty_newton, tuple(options))
        factor = None if divergence_factor is None else jnp.asarray(divergence_factor)
        return loop(None, x, tol, cap, jnp.asarray(0.0), factor, **options)
    loop = _compiled_afresh(boozer_solvers.boozer_penalty_gauss_newton, tuple(options))
    return loop(None, x, tol, cap, **options)


class TestBoozerSolverLoopsJax(JaxTestCase):
    def test_loop_stops_at_a_singular_second_step_and_keeps_its_iterate(self):
        """Every device solver stops on a singular second step and retains its last completed iterate."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for name in _CASES:
                with self.subTest(
                    case_id_0=case_id_0, parity_lane=parity_lane, name=name
                ), self.case() as patches:
                    self._case_loop_stops_at_a_singular_second_step_and_keeps_its_iterate(
                        parity_lane, name, patches
                    )

    def _case_loop_stops_at_a_singular_second_step_and_keeps_its_iterate(
        self, parity_lane, name, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        formulations, reached, second_matrix = _CASES[name]
        with self.assertRaisesRegex(np.linalg.LinAlgError, "Singular matrix"):
            np.linalg.solve(second_matrix, np.ones(2))
        _substitute(patches, formulations)
        with parity_default_device(parity_lane):
            result = _run(name, jnp.asarray(_START))
            singular, iterations, x = (
                host_array(value)
                for value in (result.singular, result.iterations, result.x)
            )
        self.assertTrue(
            bool(singular), f"{name}: the singular second step was not flagged"
        )
        self.assertTrue(
            int(iterations) == 1,
            f"{name}: {int(iterations)} steps counted; native raises during the second",
        )
        np.testing.assert_array_equal(
            x, reached, err_msg=f"{name}: the loop did not keep the iterate it reached"
        )

    def test_growing_walk_runs_to_maxiter_and_restores_its_start(self):
        """A walk whose norm grows at every step ends at ``maxiter`` with its final
        norm and returns its start and the values there, on each lane."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for name in _GROWING_CASES:
                with self.subTest(
                    case_id_0=case_id_0, parity_lane=parity_lane, name=name
                ), self.case() as patches:
                    self._case_growing_walk_runs_to_maxiter_and_restores_its_start(
                        parity_lane, name, patches
                    )

    def _case_growing_walk_runs_to_maxiter_and_restores_its_start(
        self, parity_lane, name, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        formulations, norm, restored = _GROWING_CASES[name]
        _substitute(patches, formulations)
        with parity_default_device(parity_lane):
            result = _run(name, jnp.asarray(_START), maxiter=2.0)
            values = {
                key: host_array(getattr(result, key))
                for key in ("x", "iterations", "norm", "singular", *restored)
            }
        self.assertEqual(int(values["iterations"]), 2, name)
        self.assertFalse(bool(values["singular"]), name)
        np.testing.assert_allclose(values["norm"], norm, rtol=1e-12, err_msg=name)
        np.testing.assert_array_equal(values["x"], _START, err_msg=f"{name}: start not restored")
        for key, expected in restored.items():
            np.testing.assert_array_equal(values[key], expected, err_msg=f"{name} {key}")

    def test_divergence_factor_stops_a_growing_penalty_newton_walk(self):
        """As in native's ``test_penalty_newton_divergence_factor``: the gradient
        norm grows 19-fold per step, so a factor of 1e3 stops the walk after three
        steps (norm 6859); None runs to ``maxiter``. Both restore the start."""
        formulations, _, restored = _GROWING_CASES["penalty-newton"]
        for factor, steps in ((1e3, 3), (None, 40)):
            with self.subTest(factor=factor), self.case() as patches:
                _substitute(patches, formulations)
                result = _run(
                    "penalty-newton", jnp.asarray(_START), maxiter=40.0, divergence_factor=factor
                )
                self.assertEqual(int(host_array(result.iterations)), steps)
                np.testing.assert_allclose(host_array(result.norm), 19.0**steps, rtol=1e-12)
                np.testing.assert_array_equal(host_array(result.x), _START)
                np.testing.assert_array_equal(host_array(result.gradient), restored["gradient"])
