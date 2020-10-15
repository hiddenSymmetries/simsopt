# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides the LeastSquaresProblem class, as well as the
associated class LeastSquaresTerm.
"""

from mpi4py import MPI
import numpy as np
from scipy.optimize import least_squares
import logging
from .dofs import Dofs
from .util import isnumber
from .optimizable import function_from_user
from .mpi import proc0, worker_loop, mobilize_workers, stop_workers, CALCULATE_F, CALCULATE_JAC
#from simsopt import mpi
#import mpi

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

class LeastSquaresTerm:
    """
    This class represents one term in a nonlinear-least-squares
    problem. A LeastSquaresTerm instance has 3 basic attributes: a
    function (called f_in), a goal value (called goal), and a weight
    (sigma).  The overall value of the term is:

    f_out = weight * (f_in - goal) ** 2.

    You are also free to specify sigma = 1 / sqrt(weight) instead of
    weight, so

    f_out = ((f_in - goal) / sigma) ** 2.
    """

    def __init__(self, f_in, goal, weight=None, sigma=None):
        if not isnumber(goal):
            raise TypeError('goal must be a float or int')
        if (weight is None) and (sigma is None):
            raise ValueError('You must specify either weight or sigma.')
        if (weight is not None) and (sigma is not None):
            raise ValueError('You cannot specify both sigma and weight.')
        if sigma == 0:
            raise ValueError('sigma cannot be 0')
        if weight is not None:
            # Weight was specified, sigma was not
            if not isnumber(weight):
                raise TypeError('Weight must be a float or int')
            if weight < 0:
                raise ValueError('Weight cannot be negative')
            self.weight = float(weight)
        else:
            # Sigma was specified, weight was not
            if not isnumber(sigma):
                raise TypeError('sigma must be a float or int')
            self.weight = 1.0 / float(sigma * sigma)

        self.f_in = function_from_user(f_in)
        self.goal = float(goal)
        self.fixed = np.full(0, False)

    def f_out(self):
        """
        Return the overall value of this least-squares term.
        """
        temp = self.f_in() - self.goal
        return self.weight * temp * temp 

    
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
        self.terms = terms
        self._init()

    def _init(self):
        """
        Call collect_dofs() on the list of terms to set x, mins, maxs, names, etc.
        This is done both when the object is created, so 'objective' works immediately,
        and also at the start of solve()
        """
        self.dofs = Dofs([t.f_in for t in self.terms])

    @property
    def x(self):
        """
        Return the state vector.
        """
        # Delegate to Dofs:
        return self.dofs.x

    def set(self, x):
        """
        Sets the global state vector to x.
        """
        # Delegate to Dofs:
        self.dof.set(x)
    
    def objective(self, x=None):
        """
        Return the value of the total objective function, by summing
        the terms.

        If the argument x is not supplied, the objective will be
        evaluated for the present state vector. If x is supplied, then
        first set_dofs() will be called for each object to set the
        global state vector to x.
        """
        logger.info("objective() called with x=" + str(x))
        if x is not None:
            self.dofs.set(x)
            
        sum = 0
        for term in self.terms:
            sum += term.f_out()
        return sum

    def f(self, x=None):
        """
        This method returns the vector of residuals for a given state
        vector x.  This function is passed to scipy.optimize, and
        could be passed to other optimization algorithms too.  This
        function differs from Dofs.function() because it shifts and
        scales the terms.

        If the argument x is not supplied, the residuals will be
        evaluated for the present state vector. If x is supplied, then
        first set_dofs() will be called for each object to set the
        global state vector to x.
        """
        logger.info("residuals() called with x=" + str(x))
        if x is not None:
            self.dofs.set(x)

        # Importantly for MPI, the next line calls the functions in
        # the same order that Dofs.f() does. Proc0 calls this function
        # whereas worker procs call Dofs.f().
        residuals = [(term.f_in() - term.goal) * np.sqrt(term.weight) for term in self.terms]
        return np.array(residuals)
        
    def jac(self, x=None):
        """
        This method gives the Jacobian of the residuals with respect to
        the parameters, if it is available, given the state vector
        x. This function is passed to scipy.optimize, and could be
        passed to other optimization algorithms too. This Jacobian
        differs from the one returned by Dofs() because it accounts
        for the 'weight' scale factors.

        If the argument x is not supplied, the Jacobian will be
        evaluated for the present state vector. If x is supplied, then
        first set_dofs() will be called for each object to set the
        global state vector to x.
        """
        logger.info("jac() called with x=" + str(x))

        if x is not None:
            self.dofs.set(x)
            
        # This next line does the hard work of evaluating the Jacobian:
        jac = self.dofs.jac()
        # Scale rows by sqrt(weight):
        for j in range(self.dofs.nfuncs):
            jac[j, :] = jac[j, :] * np.sqrt(self.terms[j].weight)
            
        return np.array(jac)
        
    def solve(self):
        """
        Solve the nonlinear-least-squares minimization problem.
        """
        logger.info("Beginning solve.")
        self._init()
        if not proc0():
            worker_loop(self.dofs)
            x = np.copy(self.x)
        else:
            # proc 0 does this block.
            x0 = np.copy(self.dofs.x)
            #print("x0:",x0)
            # Call scipy.optimize:
            if self.dofs.grad_avail:
                logger.info("Using analytic derivatives")
                print("Using analytic derivatives")
                result = least_squares(self.f_proc0, x0, verbose=2, jac=self.jac_proc0)
            else:
                logger.info("Using derivative-free method")
                print("Using derivative-free method")
                result = least_squares(self.f_proc0, x0, verbose=2)

            stop_workers()
            logger.info("Completed solve.")
            x = result.x

        MPI.COMM_WORLD.Bcast(x)
        #print("optimum x:",result.x)
        #print("optimum residuals:",result.fun)
        #print("optimum cost function:",result.cost)
        # Set Parameters to their values for the optimum
        self.dofs.set(x)
                
    def f_proc0(self, x):
        """
        Similar to f, except this version is called only by proc 0 while
        workers are in the worker loop.
        """
        mobilize_workers(x, CALCULATE_F)
        return self.f(x)

    def jac_proc0(self, x):
        """
        Similar to jac, except this version is called only by proc 0 while
        workers are in the worker loop.
        """
        mobilize_workers(x, CALCULATE_JAC)
        return self.jac(x)
