# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the least_squares_serial_solve
function. Eventually I can also put a serial_solve function here for
general optimization problems.
"""

from datetime import datetime
from time import time
from typing import Union, Callable
import logging

import numpy as np
from scipy.optimize import least_squares, minimize

from .._core.types import RealArray
from ..objectives.least_squares import LeastSquaresProblem
from .._core.optimizable import Optimizable
from .._core.finite_difference import FiniteDifference


logger = logging.getLogger(__name__)

__all__ = ['least_squares_serial_solve', 'serial_solve']


def finite_difference_jac_wrapper(fd, problem_type = 'least_squares', verbose = "legacy", comment = ""):
    """Wrapper for the `jac` method of the `MPIFiniteDifference` and `FiniteDifference` classes.
    For logging the jacobian calculation when used with `scipy.optimize.least_squares`.
    Also handles scipy.optimize.minimize.
    verbose - controls the amount of information logged.
            'legacy': Only the points used by the jacobian calculation is logged. The default. 
            'dfdx': The derivative of the scalar objective is logged.
            'dRdx': For least_squares problems. Logs the matrix dR_i/dx_j where R_i are the components of the residual vector. WARNING: can result in huge jacobian logs."""

    if comment != "":
        comment = " " + comment
    
    if verbose not in ["legacy", 'dfdx', 'dRdx']:
        raise ValueError("Unrecognized verbose flag: '" + verbose + "' Recognized values are 'legacy', dfdx', 'dRdx'.")
    if (verbose == 'dRdx') and (problem_type != 'least_squares'):
        raise ValueError("Verbose flag 'dRdx' is only supported for problem_type 'lest_squares'.")
    
    def jac(x: RealArray = None, *args, **kwargs):
        ret = fd.jac(x, *args, **kwargs)
        log_file = fd.log_file
        nparams = fd.nparams
        # WRITE HEADER
        if not fd.log_header_written:
            log_file.write(f'Problem type:\n{problem_type}{comment}\nnparams:\n{nparams}\n')
            log_file.write('function_evaluation, seconds')
            if verbose == "dRdx":
                log_file.write(', d(residual_j)/d(x_i)')
            elif verbose == "dfdx":
                log_file.write(', d(f)/d(x_i)')
            elif verbose == "legacy":
                for j in range(nparams):
                    log_file.write(f', x({j})')
            log_file.write('\n')
            fd.log_header_written = True
        # WRITE DATA
        if verbose == "dfdx":
            del_t = time() - fd.start_time
            j_eval = fd.eval_cnt//fd.nevals_jac
            log_file.write(f'{j_eval:6d},{del_t:12.4e}')
        
            if problem_type == 'least_squares':
                f = fd.f0 # function value at x0
                total_jac = np.sum(2 * ret * f[:, None],axis=0)                
            else:
                total_jac = ret
            for total_jacj in total_jac:
                log_file.write(f',{total_jacj:24.16e}')
            log_file.write('\n')
            log_file.flush()
        elif verbose == "dRdx":
            # only least squares can use verbose = 'dRdx'
            del_t = time() - fd.start_time
            j_eval = fd.eval_cnt//fd.nevals_jac
            log_file.write(f'{j_eval:6d},{del_t:12.4e}')
            with np.printoptions(threshold=np.inf):
                log_file.write(", " + np.array_str(ret, max_line_width = np.inf, precision = None).replace('\n',','))
                log_file.write('\n')
                log_file.flush()
        elif verbose == "legacy":
            for j in range(fd.nevals_jac):
                del_t = time() - fd.start_time
                j_eval = j + fd.eval_cnt - fd.nevals_jac - 1
                log_file.write(f'{j_eval:6d},{del_t:12.4e}')
                for xj in fd.xs[:, j]:
                    log_file.write(f',{xj:24.16e}')
                log_file.write('\n')
                log_file.flush()
        return ret
        
    return jac


def least_squares_serial_solve(prob: LeastSquaresProblem,
                               grad: bool = None,
                               abs_step: float = 1.0e-7,
                               rel_step: float = 0.0,
                               diff_method: str = "forward",
                               jac_verbose: str = "legacy", 
                               **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem using
    scipy.optimize, and without using any parallelization.

    Args:
        prob: LeastSquaresProblem object defining the objective function(s)
             and parameter space.
        grad: Whether to use a gradient-based optimization algorithm, as
             opposed to a gradient-free algorithm. If unspecified, a
             a gradient-free algorithm
             will be used by default. If you set ``grad=True`` for a problem,
             finite-difference gradients will be used.
        abs_step: Absolute step size for finite difference jac evaluation
        rel_step: Relative step size for finite difference jac evaluation
        diff_method: Differentiation strategy. Options are ``"centered"``, and
             ``"forward"``. If ``"centered"``, centered finite differences will
             be used. If ``"forward"``, one-sided finite differences will
             be used. Else, error is raised.
        kwargs: Any arguments to pass to
                `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_.
                For instance, you can supply ``max_nfev=100`` to set
                the maximum number of function evaluations (not counting
                finite-difference gradient evaluations) to 100. Or, you
                can supply ``method`` to choose the optimization algorithm.
    """

    datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    objective_file = open(f"objective_{datestr}.dat", 'w')
    residuals_file = open(f"residuals_{datestr}.dat", 'w')

    nevals = 0
    start_time = time()
    datalogging_started = False

    def objective(x):
        nonlocal datalogging_started, objective_file, residuals_file, nevals
        #success = True
        try:
            residuals = prob.residuals(x)
        except:
            logger.info("Exception caught during function evaluation")
            residuals = np.full(prob.parent_return_fns_no, 1.0e12)
            #success = False

        objective_val = prob.objective()

        # Check that 2 ways of computing the objective give same
        # answer within roundoff:
        #if success:
        #    objective2 = prob.objective()
        #    logger.info("objective_from_f={} objective={} diff={}".format(
        #        objective_val, objective2, objective_val - objective2))
        #    abs_diff = np.abs(objective_val - objective2)
        #    rel_diff = abs_diff / (1e-12 + np.abs(objective_val + objective2))
        #    assert (abs_diff < 1e-12) or (rel_diff < 1e-12)

        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not datalogging_started:
            # Initialize log file
            datalogging_started = True
            ndofs = prob.dof_size
            objective_file.write(
                f"Problem type:\nleast_squares\nnparams:\n{ndofs}\n")
            objective_file.write("function_evaluation,seconds")
            for j in range(ndofs):
                objective_file.write(f",x({j})")
            objective_file.write(",objective_function\n")

            residuals_file.write(
                f"Problem type:\nleast_squares\nnparams:\n{ndofs}\n")
            residuals_file.write("function_evaluation,seconds")
            for j in range(ndofs):
                residuals_file.write(f",x({j})")
            residuals_file.write(",objective_function")
            for j in range(len(residuals)):
                residuals_file.write(f",F({j})")
            residuals_file.write("\n")

        elapsed_t = time() - start_time
        objective_file.write(f"{nevals:6d},{elapsed_t:12.4e}")
        for xj in x:
            objective_file.write(f",{xj:24.16e}")
        objective_file.write(f",{objective_val:24.16e}")
        objective_file.write("\n")
        objective_file.flush()

        residuals_file.write(f"{nevals:6d},{elapsed_t:12.4e}")
        for xj in x:
            residuals_file.write(f",{xj:24.16e}")
        residuals_file.write(f",{objective_val:24.16e}")
        for fj in residuals:
            residuals_file.write(f",{fj:24.16e}")
        residuals_file.write("\n")
        residuals_file.flush()

        nevals += 1
        return residuals

    logger.info("Beginning solve.")
    #if grad is None:
    #    grad = prob.dofs.grad_avail

    #if not 'verbose' in kwargs:

    print('prob is ', prob)
    x0 = np.copy(prob.x)
    if grad:
        fd = FiniteDifference(prob.residuals, abs_step=abs_step,
                              rel_step=rel_step, diff_method=diff_method)
        logger.info("Using derivatives")
        jac = finite_difference_jac_wrapper(fd, verbose=jac_verbose, comment="serial")
        result = least_squares(objective, x0, verbose=2, jac=jac, **kwargs)
    else:
        logger.info("Using derivative-free method")
        result = least_squares(objective, x0, verbose=2, **kwargs)

    datalogging_started = False
    objective_file.close()
    residuals_file.close()
    logger.info("Completed solve.")

    prob.x = result.x


def serial_solve(prob: Union[Optimizable, Callable],
                 grad: bool = None,
                 abs_step: float = 1.0e-7,
                 rel_step: float = 0.0,
                 diff_method: str = "centered",
                 jac_verbose: str = "legacy",
                 **kwargs):
    """
    Solve a general minimization problem (i.e. one that need not be of
    least-squares form) using scipy.optimize.minimize, and without using any
    parallelization.

    Args:
        prob: Optimizable object defining the objective function(s)
             and parameter space.
        grad: Whether to use a gradient-based optimization algorithm, as
             opposed to a gradient-free algorithm. If unspecified, a
             gradient-based algorithm will be used if ``prob`` has gradient
             information available, otherwise a gradient-free algorithm
             will be used by default. If you set ``grad=True``
             in which gradient information is not available,
             finite-difference gradients will be used.
        abs_step: Absolute step size for finite difference jac evaluation
        rel_step: Relative step size for finite difference jac evaluation
        diff_method: Differentiation strategy. Options are ``"centered"``, and
             ``"forward"``. If ``"centered"``, centered finite differences will
             be used. If ``"forward"``, one-sided finite differences will
             be used. Else, error is raised.
        kwargs: Any arguments to pass to
                `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
                For instance, you can supply ``method``
                to choose the optimization algorithm.
    """

    filename = "objective_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") \
               + ".dat"
    with open(filename, 'w') as objective_file:
        datalogging_started = False
        nevals = 0
        start_time = time()

        def objective(x):
            nonlocal datalogging_started, objective_file, nevals
            prob.x = x
            try:
                result = prob.J()
            except:
                logger.info("Exception caught during function evaluation")
                result = 1.0e12
            # Since the number of terms is not known until the first
            # evaluation of the objective function, we cannot write the
            # header of the output file until this first evaluation is
            # done.
            if not datalogging_started:
                # Initialize log file
                datalogging_started = True
                objective_file.write(
                    f"Problem type:\nminimize\nnparams:\n{prob.dof_size}\n")
                objective_file.write("function_evaluation,seconds")
                for j in range(prob.dof_size):
                    objective_file.write(f",x({j})")
                objective_file.write(",objective_function")
                objective_file.write("\n")

            del_t = time() - start_time
            objective_file.write(f"{nevals:6d},{del_t:12.4e}")
            for xj in x:
                objective_file.write(f",{xj:24.16e}")
            # objective_file.write(f",{result:24.16e}")
            objective_file.write(f",{result}")
            objective_file.write("\n")
            objective_file.flush()

            nevals += 1
            return result

        # Need to fix up this next line for non-least-squares problems:
        #if grad is None:
        #    grad = prob.dofs.grad_avail

        #if not 'verbose' in kwargs:

        logger.info("Beginning solve.")
        x0 = np.copy(prob.x)
        if grad:
            logger.info("Using derivatives")
            fd = FiniteDifference(prob.J, abs_step=abs_step,
                                  rel_step=rel_step, diff_method=diff_method, flatten_out = True)
            jac = finite_difference_jac_wrapper(fd, problem_type='minimize', verbose = jac_verbose, comment="serial")
            result = minimize(objective, x0, options={'disp': True}, jac=jac,
                                   **kwargs)
        else:
            logger.info("Using derivative-free method")
            result = minimize(objective, x0, options={'disp': True}, **kwargs)

        datalogging_started = False
        logger.info("Completed solve.")

    prob.x = result.x
