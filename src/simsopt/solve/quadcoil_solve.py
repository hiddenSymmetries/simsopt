
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax import jit, vmap, grad, jacrev, jvp
import jax
from jax.lax import while_loop, scan
from functools import partial
import numpy as np

# lstsq_vmap = vmap(jnp.linalg.lstsq)

# def wl_debug(cond_fun, body_fun, init_val):
#     val = init_val
#     iter_num_wl = 1
#     while cond_fun(val):
#         val = body_fun(val)
#     return val

# Wrapper for optax.lbfgs


def run_opt_lbfgs(init_params, fun, max_iter, ftol, xtol, gtol): return \
    run_opt_optax(init_params, fun, max_iter, ftol, xtol, gtol, opt=optax.lbfgs())

# Not robust. In here for backup purposes.


def run_opt_bfgs(init_params, fun, max_iter, ftol, xtol, gtol):
    return jax.scipy.optimize.minimize(fun=fun, x0=init_params, method='BFGS', tol=gtol, options={'maxiter': max_iter, }).x


def run_opt_optax(init_params, fun, max_iter, ftol, xtol, gtol, opt):
    """
    Run a QUADCOIL optimization using optax. 

    Args:
        init_params: jax array containing initial parameters for optax.
        fun: The function to optimize.
        max_iter: Maximum number of iterations.
        ftol: Tolerance for the function value.
        xtol: Tolerance for the parameter change.
        gtol: Tolerance for the gradient.
        opt: optax optimizer (e.g. ADAM or BFGS).
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


def eval_quad_scaled(phi_scaled, A, b, c, current_scale):
    """
    Rescaling the objective (the quadratic in phi) by 
    a scale factor making phi unit-free and order-1 is important
    because otherwise the gradient may be too shallow.
    We can't normalize A, b, c to ~1 but not do the same to Phi.
    current_scale is very necessary - if the unit for 
    phi is not removed, the gradients of this function 
    may be too flat or steep.

    Args:
        phi_scaled: jax array containing the scaled parameters.
        A: jax array containing the quadratic coefficients.
        b: jax array containing the linear coefficients.
        c: jax array containing the constant coefficients.
        current_scale: jax array containing the current
    """
    phi = phi_scaled / current_scale
    return ((A @ phi) @ phi + b @ phi + c)


@jit
def solve_quad_unconstrained(A, b, c):
    """
    Solve the unconstrained QUADCOIL problem.

    Args:
        A: jax array containing the quadratic coefficients.
        b: jax array containing the linear coefficients.
        c: jax array containing the constant coefficients.
    """
    x, _, _, _ = jnp.linalg.lstsq(2 * A, -b)  # Solve the least-squares problem
    f = eval_quad_scaled(x, A, b, c, 1)  # rescale the objective
    return (x, f)


def solve_constrained(
    x_init,
    f_obj,
    run_opt,
    # No constraints by default
    c_init=0.1,
    lam_init=jnp.zeros(1, dtype=jnp.float32),
    h_eq=lambda x: jnp.zeros(1, dtype=jnp.float32),
    mu_init=jnp.zeros(1, dtype=jnp.float32),
    g_ineq=lambda x: jnp.zeros(1, dtype=jnp.float32),
    c_growth_rate=1.1,
    tol_outer=1e-7,
    ftol_inner=1e-7,
    xtol_inner=1e-7,
    gtol_inner=1e-7,
    max_iter_inner=500,
    max_iter_outer=20,
    scan_mode=False,
):
    """
    A simple augmented Lagrangian implementation for solving 
    min f(x)
        subject to 
        h(x) = 0, g(x) <= 0

    May want a temporary jit flag, because we want 
    derivatives wrt f and g's contents too.

    Args:
        x_init: jax array containing the initial parameters.
        f_obj: jax function computing the objective function.
        run_opt: jax function to run the optiization.
        c_init: jax array containing the initial penalty parameter.
        lam_init: jax array containing the initial Lagrange multipliers.
        h_eq: jax array containing the equality constraints.
        mu_init: jax array containing the initial Lagrange multipliers.
        g_ineq: jax array containing the inequality constraints.
        c_growth_rate: jax array containing the growth rate of the penalty parameter.
        tol_outer: jax array containing the tolerance for the outer loop.
        ftol_inner: jax array containing the tolerance for the inner loop.
        xtol_inner: jax array containing the tolerance for the inner loop.
        gtol_inner: jax array containing the tolerance for the inner loop.
        max_iter_inner: jax array containing the maximum number of iterations for the inner loop.
        max_iter_outer: jax array containing the maximum number of iterations for the outer loop.
        scan_mode: jax array containing the scan mode flag. If true, uses jax.lax.scan 
            instead of while_loop. Enables history and forward diff but disables 
            convergence test.
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
    A_f, b_f, c_f,
    A_eq, b_eq, c_eq,
    A_ineq, b_ineq, c_ineq,
    x_scale,
    x_memory=None
):
    """
    Convert quadratic coefficients to callables.
    reused later for reconstructing l_k as a 
    function of both the current potential and 
    plasma parameters. Also initializes lam and mu,
    the two multipliers.

    Args:
        A_f: jax array containing the quadratic coefficients for the objective.
        b_f: jax array containing the linear coefficients for the objective.
        c_f: jax array containing the constant coefficients for the objective.
        A_eq: jax array containing the quadratic coefficients for the equality constraints.
        b_eq: jax array containing the linear coefficients for the equality constraints.
        c_eq: jax array containing the constant coefficients for the equality constraints.
        A_ineq: jax array containing the quadratic coefficients for the inequality constraints.
        b_ineq: jax array containing the linear coefficients for the inequality constraints.
        c_ineq: jax array containing the constant coefficients for the inequality constraints.
        x_scale: jax array containing the scaling factor for the parameters.
        x_memory: jax array containing the memory for the initial guess.
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
    y_to_qcqp_coefs,
    f_metric_list,
    y_params,
    init_mode='zero',
    x_init=None,
    run_opt=run_opt_lbfgs,
    **kwargs
):
    """
    Differentiate the result of a nearly convex QCQP, wrt 
    parameter "y_param" that determines:
    A_f, b_f, c_f, (coeffs of the objective "f". Takes x WITH UNIT)
    A_eq, b_eq, c_eq, (coeffs of the equality constraints. Takes x WITH UNIT)
    A_ineq, b_ineq, c_ineq, (coeffs of the inequality constraints. Takes x WITH UNIT)
    x_scale, (a scaling factor based on the UNIT OF x, so that x_with_unit * x_scale ~ 1.)

    Uses Cauchy Implicit Function Theorem.

    Args:
        y_to_qcqp_coefs: maps
            maps y_params 
            -> 
            A_f, b_f, c_f,          # shape:         (n,n),         (n), 0
            A_eq, b_eq, c_eq,       # shape: (n_eq,   n,n), (n_eq,   n), n_eq
            A_ineq, b_ineq, c_ineq, # shape: (n_ineq, n,n), (n_ineq, n), n_ineq
            x_scale                 # scalar
            When there are no constraints present, use None.

        f_metric_list: list containing [f1, f2, ...] that maps
            x_with_unit, y_params,  
            -> 
            scalar

        y_params: Location to evaluate at.
        init_mode: Initial guess choice. Available options are 'zero' or 'unconstrained'. 
        x_init: 
        run_opt: Optimizer choice. Must have the signature (init_params, fun, max_iter, ftol, xtol, gtol)
        **kwargs: Will be passed directly into solve_constrained.

    Returns: (x, list)
        x_with_unit:    The optimum of the QCQP, WITH UNIT
        list: A list containing:
            [
                (f1, df1/dy),
                (f2, df2/dy),
                ...
            ] 
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
