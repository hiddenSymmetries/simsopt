# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides functions for solving least-squares and general optimization
problems, without parallelization in the optimization algorithm itself,
and without parallelized finite-difference gradients. 
These functions could still be used for cases in which there is parallelization within the objective
function evaluations.
These functions essentially
are interfaces between a :obj:`simsopt.core.least_squares_problem.LeastSquaresProblem`
object and `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_.
The functions here also create a log file with history of the objective function evaluations.

If you want parallelized finite difference gradient evaluations, you should instead use
:meth:`simsopt.solve.mpi_solve.least_squares_mpi_solve()`. If not, the methods here may be preferable
due to their greater simplicity.
"""

import logging
from datetime import datetime
from time import time
import traceback

import numpy as np
from scipy.optimize import least_squares, minimize

from ..objectives.least_squares import LeastSquaresProblem
from .graph_serial import least_squares_serial_solve as glsss
from ..util.dev import deprecated

logger = logging.getLogger(__name__)


@deprecated(replacement=glsss,
            message="This class has been deprecated from v0.6.0 and will be "
                    "deleted from future versions of simsopt. Use graph "
                    "framework to define the optimization problem. Use "
                    "simsopt.objectives.graph_least_squares.LeastSquaresProblem"
                    " class in conjunction with"
                    " simsopt.solve.graph_serial.least_squares_serial_solve")
def least_squares_serial_solve(prob: LeastSquaresProblem,
                               grad: bool = None,
                               **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem.

    Args:
        prob: An instance of LeastSquaresProblem, defining the objective
                function(s) and parameter space.
        grad: Whether to use a gradient-based optimization algorithm, as
                opposed to a gradient-free algorithm. If unspecified, a
                gradient-based algorithm will be used if ``prob`` has
                gradient information available, otherwise a gradient-free
                algorithm will be used by default. If you set ``grad=True``
                for a problem in which gradient information is not available,
                finite-difference gradients will be used.
        kwargs: Any arguments to pass to 
                `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_.
                For instance, you can supply ``max_nfev=100`` to set the
                maximum number of function evaluations (not counting
                finite-difference gradient evaluations) to 100.
    """

    objective_file = None
    datalogging_started = False
    residuals_file = None
    nevals = 0
    start_time = time()

    def objective(x):
        nonlocal datalogging_started, objective_file, residuals_file, nevals

        f_unshifted = prob.dofs.f(x)
        f_shifted = prob.f_from_unshifted(f_unshifted)
        objective_val = prob.objective_from_shifted_f(f_shifted)

        # Check that 2 ways of computing the objective give same
        # answer within roundoff:
        objective2 = prob.objective()
        logger.info("objective_from_f={} objective={} diff={}".format(
            objective_val, objective2, objective_val - objective2))
        abs_diff = np.abs(objective_val - objective2)
        rel_diff = abs_diff / (1e-12 + np.abs(objective_val + objective2))
        assert (abs_diff < 1e-12) or (rel_diff < 1e-12)

        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not datalogging_started:
            # Initialize log file
            datalogging_started = True
            datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = "simsopt_" + datestr + ".dat"
            objective_file = open(filename, 'w')
            objective_file.write("Problem type:\nleast_squares\nnparams:\n{}\n".format(prob.dofs.nparams))
            objective_file.write("function_evaluation,seconds")
            for j in range(prob.dofs.nparams):
                objective_file.write(",x({})".format(j))
            objective_file.write(",objective_function")
            objective_file.write("\n")

            filename = "residuals_" + datestr + ".dat"
            residuals_file = open(filename, 'w')
            residuals_file.write("Problem type:\nleast_squares\nnparams:\n{}\n".format(prob.dofs.nparams))
            residuals_file.write("function_evaluation,seconds")
            for j in range(prob.dofs.nparams):
                residuals_file.write(",x({})".format(j))
            residuals_file.write(",objective_function")
            for j in range(prob.dofs.nvals):
                residuals_file.write(",F({})".format(j))
            residuals_file.write("\n")

        objective_file.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            objective_file.write(",{:24.16e}".format(xj))
        objective_file.write(",{:24.16e}".format(objective_val))
        objective_file.write("\n")
        objective_file.flush()

        residuals_file.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            residuals_file.write(",{:24.16e}".format(xj))
        residuals_file.write(",{:24.16e}".format(objective_val))
        for fj in f_unshifted:
            residuals_file.write(",{:24.16e}".format(fj))
        residuals_file.write("\n")
        residuals_file.flush()

        nevals += 1
        return f_shifted

    logger.info("Beginning solve.")
    prob._init()  # In case 'fixed', 'mins', etc have changed since the problem was created.
    if grad is None:
        grad = prob.dofs.grad_avail

    #if not 'verbose' in kwargs:

    x0 = np.copy(prob.x)
    if grad:
        logger.info("Using derivatives")
        result = least_squares(objective, x0, verbose=2, jac=prob.jac, **kwargs)
    else:
        logger.info("Using derivative-free method")
        result = least_squares(objective, x0, verbose=2, **kwargs)

    datalogging_started = False
    objective_file.close()
    residuals_file.close()
    logger.info("Completed solve.")

    #print("optimum x:",result.x)
    #print("optimum residuals:",result.fun)
    #print("optimum cost function:",result.cost)
    # Set Parameters to their values for the optimum
    prob.x = result.x


def serial_solve(prob, grad=None, **kwargs):
    """
    Solve a general minimization problem (i.e. one that need not be of
    least-squares form) using scipy.optimize.minimize, and without using any
    parallelization.

    prob should be a simsopt problem.

    kwargs allows you to pass any arguments to scipy.optimize.minimize.
    """

    objective_file = None
    datalogging_started = False
    nevals = 0
    start_time = time()

    def objective(x):
        nonlocal datalogging_started, objective_file, nevals

        result = prob.objective(x)

        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not datalogging_started:
            # Initialize log file
            datalogging_started = True
            filename = "simsopt_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".dat"
            objective_file = open(filename, 'w')
            objective_file.write("Problem type:\ngeneral\nnparams:\n{}\n".format(prob.dofs.nparams))
            objective_file.write("function_evaluation,seconds")
            for j in range(prob.dofs.nparams):
                objective_file.write(",x({})".format(j))
            objective_file.write(",objective_function")
            objective_file.write("\n")

        objective_file.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            objective_file.write(",{:24.16e}".format(xj))
        objective_file.write(",{:24.16e}".format(result))
        objective_file.write("\n")
        objective_file.flush()

        nevals += 1
        return result

    logger.info("Beginning solve.")
    # Not sure if the next line has an analog for non-least-squares problems:
    # prob._init() # In case 'fixed', 'mins', etc have changed since the problem was created.

    # Need to fix up this next line for non-least-squares problems:
    #if grad is None:
    #    grad = prob.dofs.grad_avail

    #if not 'verbose' in kwargs:

    x0 = np.copy(prob.x)
    if grad:
        raise RuntimeError("Need to convert least-squares Jacobian to gradient of the scalar objective function")
        logger.info("Using derivatives")
        print("Using derivatives")
        result = least_squares(objective, x0, verbose=2, jac=prob.jac, **kwargs)
    else:
        logger.info("Using derivative-free method")
        print("Using derivative-free method")
        result = minimize(objective, x0, options={'disp': True}, **kwargs)

    datalogging_started = False
    objective_file.close()
    logger.info("Completed solve.")

    #print("optimum x:",result.x)
    #print("optimum residuals:",result.fun)
    #print("optimum cost function:",result.cost)
    # Set Parameters to their values for the optimum
    prob.x = result.x
