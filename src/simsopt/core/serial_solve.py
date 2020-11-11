# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the least_squares_serial_solve
function. Eventually I can also put a serial_solve function here for
general optimization problems.
"""

import numpy as np
from scipy.optimize import least_squares
import logging

logger = logging.getLogger(__name__)

def least_squares_serial_solve(prob, grad=None, **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem using
    scipy.optimize, and without using any parallelization.

    prob should be a LeastSquaresProblem object.

    kwargs allows you to pass any arguments to scipy.optimize.least_squares.
    """
    logger.info("Beginning solve.")
    prob._init() # In case 'fixed', 'mins', etc have changed since the problem was created.
    if grad is None:
        grad = prob.dofs.grad_avail
        
    #if not 'verbose' in kwargs:
        
    x0 = np.copy(prob.x)
    if grad:
        logger.info("Using derivatives")
        print("Using derivatives")
        result = least_squares(prob.f, x0, verbose=2, jac=prob.jac, **kwargs)
    else:
        logger.info("Using derivative-free method")
        print("Using derivative-free method")
        result = least_squares(prob.f, x0, verbose=2, **kwargs)

    logger.info("Completed solve.")

    #print("optimum x:",result.x)
    #print("optimum residuals:",result.fun)
    #print("optimum cost function:",result.cost)
    # Set Parameters to their values for the optimum
    prob.x = result.x
