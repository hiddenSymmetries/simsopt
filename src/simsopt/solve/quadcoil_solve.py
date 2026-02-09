"""
QUADCOIL optimization solvers using JAX and optax.

This module provides solvers for Quadratically Constrained Quadratic Programs
(QCQPs) arising in the QUADCOIL current potential optimization framework.
It includes unconstrained least-squares solvers, augmented Lagrangian
constrained solvers, and implicit differentiation through the optimality
conditions via the Cauchy Implicit Function Theorem.

References:
    - Optax L-BFGS documentation: https://optax.readthedocs.io/
    - Augmented Lagrangian method: Bertsekas, *Constrained Optimization
      and Lagrange Multiplier Methods*, 1982.
    - Implicit differentiation: arXiv:1911.02590
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
try:
    import optax
    import optax.tree_utils as otu
except ImportError:
    optax = None
    otu = None
from jax import jit, grad, jacrev, jvp
from jax.lax import while_loop, scan

__all__ = [
    'run_opt_lbfgs', 'run_opt_bfgs', 'run_opt_optax', 'solve_quad_unconstrained',
    'solve_constrained', 'intialize_quad', 'dfdy_from_qcqp'
]

# lstsq_vmap = vmap(jnp.linalg.lstsq)

# def wl_debug(cond_fun, body_fun, init_val):
#     val = init_val
#     iter_num_wl = 1
#     while cond_fun(val):
#         val = body_fun(val)
#     return val

# Wrapper for optax.lbfgs


def run_opt_lbfgs(
    init_params: jax.Array,
    fun: Callable[[jax.Array], float],
    max_iter: int,
    ftol: float,
    xtol: float,
    gtol: float,
) -> jax.Array:
    """Run QUADCOIL optimization using the optax L-BFGS optimizer.

    Thin wrapper around :func:`run_opt_optax` with ``opt=optax.lbfgs()``.

    Args:
        init_params: Initial parameter vector of shape ``(n,)``.
        fun: Scalar objective function ``f(x) -> float`` to minimize.
        max_iter: Maximum number of L-BFGS iterations.
        ftol: Convergence tolerance on the change in objective value
            ``|f_{k+1} - f_k|``.
        xtol: Convergence tolerance on the parameter step size
            ``||x_{k+1} - x_k||`` and update norm ``||u_{k+1} - u_k||``.
        gtol: Convergence tolerance on the L2 norm of the gradient.

    Returns:
        Optimized parameter vector of shape ``(n,)``.
    """
    return run_opt_optax(init_params, fun, max_iter, ftol, xtol, gtol, opt=optax.lbfgs())


def run_opt_bfgs(
    init_params: jax.Array,
    fun: Callable[[jax.Array], float],
    max_iter: int,
    ftol: float,
    xtol: float,
    gtol: float,
) -> jax.Array:
    """Run QUADCOIL optimization using ``jax.scipy.optimize.minimize`` (BFGS).

    .. warning::
        This solver is not robust and is included only as a backup.
        Prefer :func:`run_opt_lbfgs` for production use.

    Args:
        init_params: Initial parameter vector of shape ``(n,)``.
        fun: Scalar objective function ``f(x) -> float`` to minimize.
        max_iter: Maximum number of BFGS iterations.
        ftol: Unused (kept for API compatibility with other optimizers).
        xtol: Unused (kept for API compatibility with other optimizers).
        gtol: Gradient tolerance passed to ``jax.scipy.optimize.minimize``.

    Returns:
        Optimized parameter vector of shape ``(n,)``.
    """
    return jax.scipy.optimize.minimize(fun=fun, x0=init_params, method='BFGS', tol=gtol, options={'maxiter': max_iter, }).x


def run_opt_optax(
    init_params: jax.Array,
    fun: Callable[[jax.Array], float],
    max_iter: int,
    ftol: float,
    xtol: float,
    gtol: float,
    opt: optax.GradientTransformation,
) -> jax.Array:
    """Run a QUADCOIL optimization using an arbitrary optax optimizer.

    Uses ``jax.lax.while_loop`` for JIT-compatible iteration. The loop
    terminates when *any* of the following conditions is met:

    * The iteration count reaches *max_iter*.
    * The L2 gradient norm drops below *gtol*.
    * The parameter step ``||x_{k+1} - x_k||`` drops below *xtol*.
    * The update step ``||u_{k+1} - u_k||`` drops below *xtol*.
    * The objective change ``|f_{k+1} - f_k|`` drops below *ftol*.

    Args:
        init_params: Initial parameter vector of shape ``(n,)``.
        fun: Scalar objective function ``f(x) -> float`` to minimize.
        max_iter: Maximum number of optimizer iterations.
        ftol: Convergence tolerance on the absolute change in objective value.
        xtol: Convergence tolerance on the parameter step norm and
            update step norm.
        gtol: Convergence tolerance on the L2 norm of the gradient.
        opt: An optax ``GradientTransformation`` optimizer instance,
            e.g. ``optax.lbfgs()`` or ``optax.adam(1e-3)``.

    Returns:
        Optimized parameter vector of shape ``(n,)``, extracted from
        the final optimizer state.
    """
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    # Carry is params, update, value, dx, du, df, state1

    def step(carry):
        params1, updates1, value1, _, _, _, state1 = carry
        value2, grad2 = value_and_grad_fun(params1, state=state1)
        updates2, state2 = opt.update(
            grad2, state1, params1, value=value2, grad=grad2, value_fn=fun
        )
        params2 = optax.apply_updates(params1, updates2)
        return (
            params2, updates2, value2,
            jnp.linalg.norm(params2 - params1),
            jnp.linalg.norm(updates2 - updates1),
            jnp.abs(value2 - value1),
            state2
        )

    def continuing_criterion(carry):
        params, _, _, dx, du, df, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        # Quit if exceed the max number of iterations
        # or if the tolerances have been met.
        return (iter_num == 0) | (
            (iter_num < max_iter)
            & (err >= gtol)
            & (dx >= xtol)
            & (du >= xtol)
            & (df >= ftol)
        )
    init_carry = (
        init_params,
        jnp.zeros_like(init_params, dtype=jnp.float32),
        0., 0., 0., 0.,
        opt.init(init_params)
    )
    final_params, final_updates, final_value, final_dx, final_du, final_df, final_state = while_loop(
        continuing_criterion, step, init_carry
    )
    return (otu.tree_get(final_state, 'params'))


def eval_quad_scaled(
    phi_scaled: jax.Array,
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    current_scale: float | jax.Array,
) -> jax.Array:
    """Evaluate a scaled quadratic form ``phi^T A phi + b^T phi + c``.

    The input ``phi_scaled`` is first divided by *current_scale* to
    recover the physical current potential ``phi``. Rescaling is
    critical because the physical units of ``phi`` can cause gradients
    to be either too flat or too steep for the optimizer.

    The relationship is::

        phi = phi_scaled / current_scale
        result = phi^T A phi + b^T phi + c

    Args:
        phi_scaled: Scaled (dimensionless, order-1) parameter vector
            of shape ``(n,)``.
        A: Quadratic coefficient matrix of shape ``(n, n)`` (or
            ``(n_constraints, n, n)`` for batched constraints).
        b: Linear coefficient vector of shape ``(n,)`` (or
            ``(n_constraints, n)``).
        c: Constant scalar (or vector of shape ``(n_constraints,)``).
        current_scale: Scaling factor such that
            ``phi_scaled / current_scale`` yields the physical current
            potential with proper units.

    Returns:
        Scalar value (or array for batched constraints) of the
        quadratic form.
    """
    phi = phi_scaled / current_scale
    return ((A @ phi) @ phi + b @ phi + c)


@jit
def solve_quad_unconstrained(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve the unconstrained QUADCOIL problem via least-squares.

    Minimizes the quadratic ``x^T A x + b^T x + c`` by solving
    the linear system ``2 A x = -b`` using ``jnp.linalg.lstsq``.
    This is equivalent to the NESCOIL solution.

    Args:
        A: Quadratic coefficient matrix of shape ``(n, n)``.
        b: Linear coefficient vector of shape ``(n,)``.
        c: Constant offset (scalar).

    Returns:
        A tuple ``(x, f)`` where:

        - **x** -- Optimized parameter vector of shape ``(n,)``.
        - **f** -- Scalar objective value at the optimum.
    """
    x, _, _, _ = jnp.linalg.lstsq(2 * A, -b)  # Solve the least-squares problem
    f = eval_quad_scaled(x, A, b, c, 1)  # rescale the objective
    return (x, f)


def solve_constrained(
    x_init: jax.Array,
    f_obj: Callable[[jax.Array], float],
    run_opt: Callable,
    # No constraints by default
    c_init: float = 0.1,
    lam_init: jax.Array = jnp.zeros(1, dtype=jnp.float32),
    h_eq: Callable[[jax.Array], jax.Array] = lambda x: jnp.zeros(1, dtype=jnp.float32),
    mu_init: jax.Array = jnp.zeros(1, dtype=jnp.float32),
    g_ineq: Callable[[jax.Array], jax.Array] = lambda x: jnp.zeros(1, dtype=jnp.float32),
    c_growth_rate: float = 1.1,
    tol_outer: float = 1e-7,
    ftol_inner: float = 1e-7,
    xtol_inner: float = 1e-7,
    gtol_inner: float = 1e-7,
    max_iter_inner: int = 500,
    max_iter_outer: int = 20,
    scan_mode: bool = False,
) -> dict | tuple[dict, dict]:
    """Solve a constrained optimization problem via the augmented Lagrangian method.

    Solves::

        min  f(x)
        s.t. h(x) = 0   (equality constraints)
             g(x) <= 0   (inequality constraints)

    The outer loop updates the penalty parameter ``c`` and the Lagrange
    multipliers ``lam`` (equality) and ``mu`` (inequality) using
    first-order multiplier updates. Each inner iteration minimizes the
    augmented Lagrangian sub-problem using the supplied optimizer.

    When ``scan_mode=False`` (default), the outer loop uses
    ``jax.lax.while_loop`` with convergence checking. When
    ``scan_mode=True``, it uses ``jax.lax.scan`` for a fixed number
    of iterations, which enables forward-mode differentiation and
    records iteration history at the cost of no early stopping.

    Args:
        x_init: Initial parameter vector of shape ``(n,)``.
        f_obj: Scalar objective function ``f(x) -> float``.
        run_opt: Optimizer callable with signature
            ``run_opt(x0, fun, max_iter, ftol, xtol, gtol) -> x_opt``.
        c_init: Initial penalty parameter for the augmented Lagrangian.
        lam_init: Initial Lagrange multipliers for equality constraints,
            shape ``(n_eq,)``. Defaults to a single zero (no constraints).
        h_eq: Equality constraint function ``h(x) -> array(n_eq,)``.
            The problem enforces ``h(x) = 0``.
        mu_init: Initial Lagrange multipliers for inequality constraints,
            shape ``(n_ineq,)``. Defaults to a single zero (no constraints).
        g_ineq: Inequality constraint function ``g(x) -> array(n_ineq,)``.
            The problem enforces ``g(x) <= 0``.
        c_growth_rate: Factor by which the penalty parameter ``c`` is
            multiplied after each outer iteration.
        tol_outer: Convergence tolerance for the outer loop. The loop
            terminates when all constraint violations are below this value.
        ftol_inner: Objective change tolerance for each inner optimization.
        xtol_inner: Parameter step tolerance for each inner optimization.
        gtol_inner: Gradient norm tolerance for each inner optimization.
        max_iter_inner: Maximum iterations for each inner sub-problem.
        max_iter_outer: Maximum outer augmented Lagrangian iterations.
        scan_mode: If ``True``, use ``jax.lax.scan`` instead of
            ``while_loop``. Enables history tracking and forward-mode
            differentiation but disables early convergence stopping.

    Returns:
        If ``scan_mode=False``:
            A dict with keys ``'x_k'``, ``'f_k'``, ``'c_k'``,
            ``'lam_k'``, ``'mu_k'``, ``'conv'``, ``'current_niter'``.

        If ``scan_mode=True``:
            A tuple ``(result, history)`` where *result* is the final
            state dict and *history* is a dict of per-iteration arrays
            with keys ``'conv'``, ``'x_k'``, ``'objective'``.
    """

    # Has shape n_cons_ineq
    def gplus(x, mu, c): return jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)

    # True when non-convergent.
    @jit
    def outer_convergence_criterion(dict_in):
        # conv = dict_in['conv']
        x_k = dict_in['x_k']
        return (
            # This is the convergence condition (True when not converged yet)
            jnp.logical_and(
                dict_in['current_niter'] <= max_iter_outer,
                jnp.any(jnp.array([
                    # if gradient is too large, continue
                    jnp.any(g_ineq(x_k) >= tol_outer),
                    # if equality constraints are too large or small, continue
                    jnp.any(h_eq(x_k) >= tol_outer),
                    jnp.any(h_eq(x_k) <= -tol_outer),
                    # if inequality constraints are too large, continue
                    jnp.any(g_ineq(x_k) >= tol_outer)
                ]))
            )
        )

    # Recursion
    @jit
    def body_fun_augmented_lagrangian(dict_in, x_dummy=None):
        x_km1 = dict_in['x_k']
        c_k = dict_in['c_k']
        lam_k = dict_in['lam_k']
        mu_k = dict_in['mu_k']
        # Construct the augmented lagrangian objective function

        def l_k(x): return (
            f_obj(x)
            + lam_k@h_eq(x)
            + c_k/2 * (
                jnp.sum(h_eq(x)**2)
                + jnp.sum(gplus(x, mu_k, c_k)**2)
            )
        )
        # Eq (10) on p160 of Constrained Optimization and Multiplier Method (citation?)
        # Solving a stage of the problem
        x_k = run_opt(x_km1, l_k, max_iter_inner, ftol_inner, xtol_inner, gtol_inner)

        # Compute the lagrange multipliers update
        lam_k_first_order = lam_k + c_k * h_eq(x_k)
        mu_k_first_order = mu_k + c_k * gplus(x_k, mu_k, c_k)

        dict_out = {
            'conv': jnp.linalg.norm(x_km1-x_k)/jnp.linalg.norm(x_k),
            'x_k': x_k,
            'c_k': c_k * c_growth_rate,
            'lam_k': lam_k_first_order,
            'mu_k': mu_k_first_order,
            'current_niter': dict_in['current_niter']+1,
        }
        # When using jax.lax.scan for outer iteration,
        # the body fun also records history.
        if scan_mode:
            history_out = {
                'conv': jnp.linalg.norm(x_km1-x_k)/jnp.linalg.norm(x_k),
                'x_k': x_k,
                'objective': f_obj(x_k),
            }
            return (dict_out, history_out)
        return (dict_out)

    init_dict = body_fun_augmented_lagrangian({
        'conv': 100,
        'x_k': x_init,
        'c_k': c_init,
        'lam_k': lam_init,
        'mu_k': mu_init,
        'current_niter': 1,
    })
    if scan_mode:
        result, history = scan(
            f=body_fun_augmented_lagrangian,
            init=init_dict,
            length=max_iter_outer
        )
        result['f_k'] = f_obj(result['x_k'])
        # Now we recover the l_k of the last augmented lagrangian iteration.
        c_k = result['c_k']
        lam_k = result['lam_k']
        mu_k = result['mu_k']

        def l_k(x): return (
            f_obj(x)
            + lam_k@h_eq(x)
            + c_k/2 * (
                jnp.sum(h_eq(x)**2)
                + jnp.sum(gplus(x, mu_k, c_k)**2)
            )
        )
        return (result, history)

    result = while_loop(
        cond_fun=outer_convergence_criterion,
        body_fun=body_fun_augmented_lagrangian,
        init_val=init_dict,
    )
    result['f_k'] = f_obj(result['x_k'])
    # Now we recover the l_k of the last augmented lagrangian iteration.
    c_k = result['c_k']
    lam_k = result['lam_k']
    mu_k = result['mu_k']

    def l_k(x): return (
        f_obj(x)
        + lam_k@h_eq(x)
        + c_k/2 * (
            jnp.sum(h_eq(x)**2)
            + jnp.sum(gplus(x, mu_k, c_k)**2)
        )
    )
    return (result)


def intialize_quad(
    A_f: jax.Array,
    b_f: jax.Array,
    c_f: jax.Array,
    A_eq: Optional[jax.Array],
    b_eq: Optional[jax.Array],
    c_eq: Optional[jax.Array],
    A_ineq: Optional[jax.Array],
    b_ineq: Optional[jax.Array],
    c_ineq: Optional[jax.Array],
    x_scale: float | jax.Array,
    x_memory: Optional[jax.Array] = None,
) -> tuple[
    Callable[[jax.Array], jax.Array],
    Callable[[jax.Array], jax.Array],
    jax.Array,
    Callable[[jax.Array], jax.Array],
    jax.Array,
]:
    """Convert QCQP quadratic coefficients into callable functions.

    Constructs the objective ``f_obj(x)``, equality constraint ``h_eq(x)``,
    and inequality constraint ``g_ineq(x)`` as callables that evaluate
    the corresponding quadratic forms using :func:`eval_quad_scaled`.
    Also initializes the Lagrange multiplier vectors ``lam_init`` and
    ``mu_init`` to zero vectors of the appropriate size.

    When constraint coefficient arrays are ``None``, the corresponding
    constraint function returns a zero scalar and the multiplier is a
    single-element zero vector (i.e., no constraint is active).

    Args:
        A_f: Quadratic coefficient matrix for the objective,
            shape ``(n, n)``.
        b_f: Linear coefficient vector for the objective,
            shape ``(n,)``.
        c_f: Constant offset for the objective (scalar).
        A_eq: Quadratic coefficient matrices for equality constraints,
            shape ``(n_eq, n, n)``, or ``None`` if no equality constraints.
        b_eq: Linear coefficient vectors for equality constraints,
            shape ``(n_eq, n)``, or ``None``.
        c_eq: Constant offsets for equality constraints,
            shape ``(n_eq,)``, or ``None``.
        A_ineq: Quadratic coefficient matrices for inequality constraints,
            shape ``(n_ineq, n, n)``, or ``None`` if no inequality constraints.
        b_ineq: Linear coefficient vectors for inequality constraints,
            shape ``(n_ineq, n)``, or ``None``.
        c_ineq: Constant offsets for inequality constraints,
            shape ``(n_ineq,)``, or ``None``.
        x_scale: Scaling factor applied inside :func:`eval_quad_scaled`
            to make the decision variable dimensionless.
        x_memory: Unused. Reserved for future warm-start functionality.

    Returns:
        A tuple ``(f_obj, h_eq, lam_init, g_ineq, mu_init)`` where:

        - **f_obj** -- Callable ``f_obj(x) -> scalar`` evaluating the objective.
        - **h_eq** -- Callable ``h_eq(x) -> array(n_eq,)`` evaluating
          equality constraints (``h(x) = 0`` at feasibility).
        - **lam_init** -- Initial Lagrange multipliers for equality
          constraints, shape ``(n_eq,)`` (zeros).
        - **g_ineq** -- Callable ``g_ineq(x) -> array(n_ineq,)`` evaluating
          inequality constraints (``g(x) <= 0`` at feasibility).
        - **mu_init** -- Initial Lagrange multipliers for inequality
          constraints, shape ``(n_ineq,)`` (zeros).
    """
    # Defining objective and constraint functions
    def f_obj(x): return eval_quad_scaled(x, A_f, b_f, c_f, x_scale)
    if A_eq is None:
        def h_eq(x): return jnp.zeros(1, dtype=jnp.float32)
        lam_init = jnp.zeros(1, dtype=jnp.float32)  # No constraints by default
    else:
        # First normalize A, B, c by their size (the total constraint number).
        def h_eq(x): return eval_quad_scaled(x, A_eq, b_eq, c_eq, x_scale)
        lam_init = jnp.zeros_like(c_eq)
    if A_ineq is None:
        def g_ineq(x): return jnp.zeros(1, dtype=jnp.float32)
        mu_init = jnp.zeros(1, dtype=jnp.float32)  # No constraints by default
    else:
        def g_ineq(x): return eval_quad_scaled(x, A_ineq, b_ineq, c_ineq, x_scale)
        mu_init = jnp.zeros_like(c_ineq)
    return (f_obj, h_eq, lam_init, g_ineq, mu_init)


def dfdy_from_qcqp(
    y_to_qcqp_coefs: Callable[[jax.Array], tuple],
    f_metric_list: list[Callable[[jax.Array, jax.Array], float]],
    y_params: jax.Array,
    init_mode: str = 'zero',
    x_init: Optional[jax.Array] = None,
    run_opt: Callable = run_opt_lbfgs,
    **kwargs,
) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]], dict]:
    """Differentiate the QCQP optimum with respect to external parameters.

    Solves a (nearly convex) QCQP whose coefficients depend on an
    external parameter vector ``y_params``, then computes the total
    derivative ``df_i/dy`` for each metric function ``f_i`` in
    *f_metric_list* using the **Cauchy Implicit Function Theorem**.

    The QCQP has the form::

        min  x^T A_f x + b_f^T x + c_f
        s.t. x^T A_eq[j] x + b_eq[j]^T x + c_eq[j] = 0   for all j
             x^T A_ineq[k] x + b_ineq[k]^T x + c_ineq[k] <= 0  for all k

    where all coefficient matrices ``(A_f, b_f, c_f, ...)`` are
    functions of ``y_params`` via the mapping *y_to_qcqp_coefs*.

    The total derivative is computed as::

        df/dy = nabla_x f * J(x*, y) + nabla_y f

    where the Jacobian ``J = dx*/dy`` is obtained from the implicit
    function theorem applied to the stationarity condition of the
    augmented Lagrangian.

    Args:
        y_to_qcqp_coefs: Callable mapping ``y_params`` to a tuple::

            (A_f, b_f, c_f,
             A_eq, b_eq, c_eq,
             A_ineq, b_ineq, c_ineq,
             x_scale)

            Use ``None`` for constraint arrays when no constraints
            are present.
        f_metric_list: List of scalar metric functions, each with
            signature ``f(x_with_unit, y_params) -> scalar``. These
            are the quantities whose derivatives ``df/dy`` are computed.
        y_params: External parameter vector at which to evaluate.
        init_mode: Strategy for the initial guess of the inner QCQP:

            - ``'zero'``: Start from the zero vector.
            - ``'unconstrained'``: Start from the unconstrained
              (NESCOIL) solution.
            - ``'memory'``: Start from a user-supplied *x_init*.
        x_init: Initial guess for ``'memory'`` mode (with physical
            units). Required when ``init_mode='memory'``.
        run_opt: Optimizer callable with signature
            ``(x0, fun, max_iter, ftol, xtol, gtol) -> x_opt``.
            Defaults to :func:`run_opt_lbfgs`.
        **kwargs: Additional keyword arguments forwarded to
            :func:`solve_constrained` (e.g. ``c_init``,
            ``max_iter_inner``, ``tol_outer``).

    Returns:
        A tuple ``(x_with_unit, out_list, solve_results)`` where:

        - **x_with_unit** -- Optimized current potential with physical
          units, shape ``(n,)``.
        - **out_list** -- List of ``(f_value, df_dy)`` tuples, one per
          metric in *f_metric_list*. ``f_value`` is the scalar metric
          evaluated at the optimum, and ``df_dy`` is its total
          derivative with respect to ``y_params``.
        - **solve_results** -- Dict of augmented Lagrangian solver
          state (see :func:`solve_constrained`).
    """
    # Calculating quadratic coefficients
    (
        A_f, b_f, c_f,
        A_eq, b_eq, c_eq,
        A_ineq, b_ineq, c_ineq,
        x_scale
    ) = y_to_qcqp_coefs(y_params)

    # Defining the objective and constraint callables
    f_obj, h_eq, lam_init, g_ineq, mu_init = intialize_quad(
        A_f, b_f, c_f,
        A_eq, b_eq, c_eq,
        A_ineq, b_ineq, c_ineq,
        x_scale,
    )

    # Calculating initial guess
    if init_mode == 'zero':  # start from zeros
        x_init_scaled = jnp.zeros_like(b_f, dtype=jnp.float32)
    elif init_mode == 'unconstrained':  # start from solution of the unconstrained problem
        phi_nescoil, _ = solve_quad_unconstrained(A_f, b_f, c_f)
        x_init_scaled = phi_nescoil*x_scale
    elif init_mode == 'memory':  # start from a given solution
        if x_init is None:
            raise ValueError('x_init must be provided for memory mode')
        x_init_scaled = x_init*x_scale

    # Solving the constrained QUADCOIL problem
    # A dictionary containing augmented lagrangian info
    # and the last augmented lagrangian objective function for
    # implicit differentiation.
    solve_results = solve_constrained(
        x_init=x_init_scaled,
        f_obj=f_obj,
        run_opt=run_opt,
        lam_init=lam_init,
        mu_init=mu_init,
        h_eq=h_eq,
        g_ineq=g_ineq,
        **kwargs
    )
    # Rescale the f_B value if needed
    # quadcoil_f_B = eval_quad_scaled(solve_results['x_k'], A_f_B, b_f_B, c_f_B, x_scale)
    x_k = solve_results['x_k']
    x_with_unit = x_k / x_scale  # Rescale the solution back to the original unit

    ''' Recover the l_k in the last iteration for dx_k/dy '''
    # First, we reproduce the augmented lagrangian objective l_k that
    # led to the optimum.
    c_k = solve_results['c_k']
    lam_k = solve_results['lam_k']
    mu_k = solve_results['mu_k']
    # @jit

    def l_k(x, y):
        (
            A_f, b_f, c_f,
            A_eq, b_eq, c_eq,
            A_ineq, b_ineq, c_ineq,
            x_scale
        ) = y_to_qcqp_coefs(y)
        f_obj, h_eq, lam_init, g_ineq, mu_init = intialize_quad(
            A_f, b_f, c_f,
            A_eq, b_eq, c_eq,
            A_ineq, b_ineq, c_ineq,
            x_scale,
        )
        def gplus(x, mu, c): return jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)
        # l_k = lambda x: (
        #     f_obj(x)
        #     + lam_k@h_eq(x)
        #     + c_k/2 * (
        #         jnp.sum(h_eq(x)**2)
        #         + jnp.sum(gplus(x, mu_k, c_k)**2)
        #     )
        # )
        return (
            f_obj(x)
            + lam_k@h_eq(x)
            + c_k/2 * (
                jnp.sum(h_eq(x)**2)
                + jnp.sum(gplus(x, mu_k, c_k)**2)
            )
        )

    ''' Apply Cauchy Implicit Function Therorem '''
    # Now we begin to calculate the Jacobian of the QUADCOIL
    # optimum wrt the plasma dofs using Cauchy Implicite Function Theorem
    # See (IFT) of arXiv:1911.02590
    # Here, we call the equilibrium properties y and
    # the current potential dofs x. The optimum is x*.
    # df/dy = \nabla_{x_k} f J(x_k, y) + \nabla_{y} f.
    # df/dy = \nabla_{x_k} f [-H(l_k, x_k)^-1 \nabla_{x_k}\nabla_{y} l_k] + \nabla_{y} f.
    # J(x_k, y) can be calculated from implicit function theorem.
    # The x gradient of f (not l_k)
    out_list = []
    nabla_x_l_k = grad(l_k, argnums=0)
    nabla_y_l_k = grad(l_k, argnums=1)
    # For ca
    def nabla_y_l_k_for_hess(x): return nabla_y_l_k(x, y_params)
    # The vector-inverse-hessian product
    # \nabla_{x_k} f [H(l_k, x_k)^-1]
    hess_l_k = jacrev(nabla_x_l_k)(x_k, y_params)
    for f_metric_with_unit in f_metric_list:
        def f_metric(x, y): return f_metric_with_unit(x/x_scale, y)
        nabla_x_f = grad(f_metric, argnums=0)(x_k, y_params)
        nabla_y_f = grad(f_metric, argnums=1)(x_k, y_params)
        vihp = jnp.linalg.solve(hess_l_k, nabla_x_f)
        # Now we calculate df/dy using vjp
        # \nabla_{x_k} f [-H(l_k, x_k)^-1 \nabla_{x_k}\nabla_{y} l_k]
        _, dfdy1 = jvp(nabla_y_l_k_for_hess, primals=[x_k], tangents=[vihp])
        # \nabla_{y} f
        dfdy2 = nabla_y_f
        out_list.append((f_metric(x_k, y_params), - dfdy1 + dfdy2))

    return x_with_unit, out_list, solve_results
