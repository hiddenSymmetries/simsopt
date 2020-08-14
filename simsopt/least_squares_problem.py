"""
This module provides the LeastSquaresProblem class.
"""

import numpy as np
from .least_squares_term import LeastSquaresTerm
from scipy.optimize import least_squares
import logging

class LeastSquaresProblem:
    """
    This class represents a nonlinear-least-squares optimization
    problem. The class stores a list of LeastSquaresTerm objects.
    """

    def __init__(self, terms):
        """
        The argument "terms" must be convertable to a list by the
        list() subroutine. Each entry of the resulting list must have
        type LeastSquaresTerm.
        """
        try:
            terms = list(terms)
        except:
            raise ValueError("terms must be convertable to a list by the " \
                                 + "list(terms) command.")
        if len(terms) == 0:
            raise ValueError("At least 1 LeastSquaresTerm must be provided " \
                                 "in terms")
        for term in terms:
            if not isinstance(term, LeastSquaresTerm):
                raise ValueError("Each term in terms must be an instance of " \
                                     "LeastSquaresTerm.")
        self._terms = terms
        # Get a list of all Parameters
        params = set()
        for j in range(len(terms)):
            params = params.union(terms[j].in_target.parameters)
        self._parameters = list(params)

    @property
    def parameters(self):
        """
        Return a list of all Parameter objects upon which the
        objective function depends.
        """
        return self._parameters

    @property
    def objective(self):
        """
        Return the value of the total objective function, by summing
        the terms.
        """
        logger = logging.getLogger(__name__)
        logger.info("objective called.")
        sum = 0
        for term in self._terms:
            sum += term.out_val
        return sum

    def solve(self):
        """
        Solve the nonlinear-least-squares minimization problem.
        """
        logger = logging.getLogger(__name__)
        logger.info("Beginning solve.")
        # Get vector of initial values for the parameters:
        #print("Parameters for solve:",self._parameters)
        x0 = [param.val for param in self._parameters if not param.fixed]
        #print("x0:",x0)
        # Call scipy.optimize:
        result = least_squares(self._residual_func, x0, verbose=2)
        logger.info("Completed solve.")
        #print("optimum x:",result.x)
        #print("optimum residuals:",result.fun)
        #print("optimum cost function:",result.cost)
        # Set Parameters to their values for the optimum
        index = 0
        for j in range(len(x0)):
            if not self._parameters[j].fixed:
                self._parameters[j].val = result.x[index]
                index += 1

    def _residual_func(self, x):
        """
        This private method is passed to scipy.optimize.
        """
        logger = logging.getLogger(__name__)
        logger.info("_residual_func called.")
        #print("_residual_func called with x=",x)
        index = 0
        for j in range(len(self._parameters)):
            if not self._parameters[j].fixed:
                self._parameters[j].val = x[index]
                index += 1
        assert index == len(x)
        return [(term.in_val - term.goal) / term.sigma for term in self._terms]
        
