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

def least_squares_serial_solve(prob, grad=None, **kwargs):
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
        try:
            result = prob.f(x)
            objective_val = prob.objective()
        except:
            logger.info("Exception caught during function evaluation")
            result = np.full(prob.dofs.nvals, 1.0e12)
            objective_val = prob.dofs.nvals * 1e24
        
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
            logfile.write("Problem type:\nleast_squares\nnparams:\n{}\n".format(prob.dofs.nparams))
            logfile.write("function_evaluation,seconds")
            for j in range(prob.dofs.nparams):
                logfile.write(",x({})".format(j))
            logfile.write(",objective_function")
            logfile.write("\n")
            
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
            
        logfile.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            logfile.write(",{:24.16e}".format(xj))
        logfile.write(",{:24.16e}".format(objective_val))
        logfile.write("\n")
        logfile.flush()

        residuals_file.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            residuals_file.write(",{:24.16e}".format(xj))
        residuals_file.write(",{:24.16e}".format(objective_val))
        for fj in result:
            residuals_file.write(",{:24.16e}".format(fj))
        residuals_file.write("\n")
        residuals_file.flush()

        nevals += 1
        return result

    logger.info("Beginning solve.")
    prob._init() # In case 'fixed', 'mins', etc have changed since the problem was created.
    if grad is None:
        grad = prob.dofs.grad_avail
        
    #if not 'verbose' in kwargs:
        
    x0 = np.copy(prob.x)
    if grad:
        logger.info("Using derivatives")
        print("Using derivatives")
        result = least_squares(objective, x0, verbose=2, jac=prob.jac, **kwargs)
    else:
        logger.info("Using derivative-free method")
        print("Using derivative-free method")
        result = least_squares(objective, x0, verbose=2, **kwargs)

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
            logfile.write("Problem type:\ngeneral\nnparams:\n{}\n".format(prob.dofs.nparams))
            logfile.write("function_evaluation,seconds")
            for j in range(prob.dofs.nparams):
                logfile.write(",x({})".format(j))
            logfile.write(",objective_function")
            logfile.write("\n")
            
        logfile.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            logfile.write(",{:24.16e}".format(xj))
        logfile.write(",{:24.16e}".format(result))
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

    logfile.close()
    logger.info("Completed solve.")
    
    #print("optimum x:",result.x)
    #print("optimum residuals:",result.fun)
    #print("optimum cost function:",result.cost)
    # Set Parameters to their values for the optimum
    prob.x = result.x
