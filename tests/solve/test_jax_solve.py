import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from simsopt.solve import least_squares_jax_solve


def test_least_squares_jax_solve_quadratic():
    def residual(x):
        return jnp.array([x[0] - 2.0, 2.0 * x[1] + 1.0])

    result = least_squares_jax_solve(
        residual,
        np.array([0.0, 0.0]),
        method="gradient_descent",
        max_nfev=200,
        step_size=0.2,
        gtol=1e-8,
        jit=False,
        verbose=0,
    )

    np.testing.assert_allclose(result["x"], np.array([2.0, -0.5]), atol=1e-3)


def test_least_squares_jax_solve_gradient_descent_avoids_redundant_startup_objectives():
    def residual(x):
        return jnp.array([x[0] - 2.0, 2.0 * x[1] + 1.0])

    result = least_squares_jax_solve(
        residual,
        np.array([0.0, 0.0]),
        method="gradient_descent",
        max_nfev=1,
        step_size=0.2,
        gtol=1e-8,
        jit=False,
        verbose=0,
        profile=True,
    )

    assert result["profile"]["value_and_grad_calls"] == 1
    assert result["profile"]["objective_calls"] == 1


def test_least_squares_jax_solve_scipy_numeric_jacobian():
    def residual(x):
        return jnp.array([x[0] - 3.0, x[1] + 4.0])

    result = least_squares_jax_solve(
        residual,
        np.array([0.0, 0.0]),
        method="scipy",
        max_nfev=20,
        gtol=1e-10,
        jit=False,
        jac="2-point",
        verbose=0,
    )

    np.testing.assert_allclose(result["x"], np.array([3.0, -4.0]), atol=1e-8)
