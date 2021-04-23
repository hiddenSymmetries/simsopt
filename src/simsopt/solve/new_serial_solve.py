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
import numpy as np
from scipy.optimize import least_squares, minimize
import logging

logger = logging.getLogger(__name__)

def least_squares_solve(prob, grad=None, **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem using
    scipy.optimize, and without using any parallelization.

    prob should be a LeastSquaresProblem object.

    kwargs allows you to pass any arguments to scipy.optimize.least_squares.
    """

    logfile = None
    logfile_started = False
    residuals_file = None
    nevals = 0
    start_time = time()
    
    def objective(x):
        nonlocal logfile_started, logfile, residuals_file, nevals
        #success = True
        try:
            residuals = prob.residuals(x)
        except:
            logger.info("Exception caught during function evaluation")
            residuals = np.full(prob.get_parent_return_fns_no(), 1.0e12)
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
        if not logfile_started:
            # Initialize log file
            logfile_started = True
            datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = "simsopt_" + datestr + ".dat"
            logfile = open(filename, 'w')
            ndofs = prob.dof_size
            logfile.write(
                f"Problem type:\nleast_squares\nnparams:\n{ndofs}\n")
            logfile.write("function_evaluation,seconds")
            for j in range(ndofs):
                logfile.write(f",x({j})")
            logfile.write(",objective_function\n\n")

            filename = "residuals_" + datestr + ".dat"
            residuals_file = open(filename, 'w')
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
        logfile.write(f"{nevals:6d},{elapsed_t:12.4e}")
        for xj in x:
            logfile.write(f",{xj:24.16e}")
        logfile.write(f",{objective_val:24.16e}")
        logfile.write("\n")
        logfile.flush()

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
        logger.info("Using derivatives")
        print("Using derivatives")
        result = least_squares(objective, x0, verbose=2, jac=prob.jac, **kwargs)
    else:
        logger.info("Using derivative-free method")
        print("Using derivative-free method")
        result = least_squares(objective, x0, verbose=2, **kwargs)

    logfile_started = False
    logfile.close()
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

    logfile = None
    logfile_started = False
    nevals = 0
    start_time = time()
    
    def objective(x):
        nonlocal logfile_started, logfile, nevals
        try:
            result = prob.objective(x)
        except:
            result = 1e+12
        
        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not logfile_started:
            # Initialize log file
            logfile_started = True
            filename = "simsopt_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".dat"
            logfile = open(filename, 'w')
            logfile.write(f"Problem type:\ngeneral\nnparams:\n{prob.dof_size}\n")
            logfile.write("function_evaluation,seconds")
            for j in range(prob.dof_size):
                logfile.write(f",x({j})")
            logfile.write(",objective_function")
            logfile.write("\n")

        del_t = time() - start_time
        logfile.write(f"{nevals:6d},{del_t:12.4e}")
        for xj in x:
            logfile.write(f",{xj:24.16e}")
        logfile.write(f",{result:24.16e}")
        logfile.write("\n")
        logfile.flush()

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
        result = minimize(objective, x0, options={'disp':True}, **kwargs)

    logfile_started = False
    logfile.close()
    logger.info("Completed solve.")
    
    #print("optimum x:",result.x)
    #print("optimum residuals:",result.fun)
    #print("optimum cost function:",result.cost)
    # Set Parameters to their values for the optimum
    prob.x = result.x
