"""
This module provides functions for computing the finite-difference
gradient of a function, and for checking the correctness of
user-supplied analytic gradients.
"""

import numpy as np
import logging
from .optimizable import function_from_user
from .dofs import Dofs

def finite_difference(func, eps=1e-7):
    """
    Compute the finite-difference gradient of the function func with
    respect to all non-fixed degrees of freedom. A centered-difference
    approximation is used, with step size eps. func can be any
    optimizable object that could be supplied to LeastSquaresTerm.
    """

    logger = logging.getLogger(__name__)
    
    # If func is an object rather than a function, get the function:
    func = function_from_user(func)
    logger.info('Beginning finite difference gradient calculation for function ' + str(func))

    # Get the non-fixed degrees of freedom:
    dofs = Dofs([func])

    x0 = dofs.x
    n = len(x0)
    logger.info('  n: ' + str(n))
    logger.info('  x0: ' + str(x0))

    grad = np.zeros(n)
    for j in range(n):
        x = np.copy(x0)
        
        x[j] = x0[j] + eps
        dofs.set(x)
        fplus = func()
        
        x[j] = x0[j] - eps
        dofs.set(x)
        fminus = func()

        # Centered differences:
        grad[j] = (fplus - fminus) / (2 * eps)

    return grad

