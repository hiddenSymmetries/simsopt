# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

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
from .optimizable import function_from_user, Target
from .mpi import MpiPartition, CALCULATE_F, CALCULATE_JAC
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
        self.goal = goal
        #self.fixed = np.full(0, False) # What is this line for?

    def f_out(self):
        """
        Return the overall value of this least-squares term.
        """
        temp = self.f_in() - self.goal
        # Below, np.dot works with both scalars and vectors.
        return self.weight * np.dot(temp, temp)
    

    
class LeastSquaresProblem:
    """
    This class represents a nonlinear-least-squares optimization
    problem. The class stores a list of LeastSquaresTerm objects.
    """

    def __init__(self, terms, mpi=None):
        """
        The argument "terms" must be convertable to a list by the list()
        subroutine. Each entry of the resulting list must either have
        type LeastSquaresTerm or else be a list or tuple of the form
        (function, goal, weight) or (object, attribute_str, goal,
        weight).
        """

        try:
            terms = list(terms)
        except:
            raise ValueError("terms must be convertable to a list by the " \
                                 + "list(terms) command.")
        if len(terms) == 0:
            raise ValueError("At least 1 LeastSquaresTerm must be provided " \
                                 "in terms")

        # For each item provided in the list, either convert to a
        # LeastSquaresTerm or, if it is already a LeastSquaresTerm,
        # use it directly.
        self.terms = []
        msg = 'Each term must be either (1) a LeastSquaresTerm or (2) a list ' \
            'or tuple of the form (function, goal, weight) or (object, ' \
            'attribute_str, goal, weight)'
        for term in terms:
            if isinstance(term, LeastSquaresTerm):
                self.terms.append(term)
            else:
                # Then term should be a list or tuple
                try:
                    n = len(term)
                except:
                    raise ValueError(msg)
                
                if n == 3:
                    self.terms.append(LeastSquaresTerm(term[0], term[1], term[2]))
                elif n == 4:
                    self.terms.append(LeastSquaresTerm(Target(term[0], term[1]), term[2], term[3]))
                else:
                    raise ValueError(msg)
                
        if mpi is None:
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi
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
        function differs from Dofs.f() because it shifts and scales
        the terms.

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
        f_unscaled = self.dofs.f()
        residuals = np.zeros(len(f_unscaled))
        start_index = 0
        for j in range(self.dofs.nfuncs):
            term = self.terms[j]
            end_index = start_index + self.dofs.nvals_per_func[j]
            residuals[start_index:end_index] = (f_unscaled[start_index:end_index] - term.goal) \
                * np.sqrt(term.weight)
            start_index = end_index
        #residuals = [(term.f_in() - term.goal) * np.sqrt(term.weight) for term in self.terms]
        #return np.array(residuals)
        return residuals
        
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

        if self.dofs.grad_avail:
            # This next line does the hard work of evaluating the Jacobian:
            logger.debug('Calling analytic Jacobian')
            jac = self.dofs.jac()
        else:
            logger.debug('Calling parallel finite-difference Jacobian')
            jac = self.dofs.fd_jac_par(self.mpi)
            
        # Scale rows by sqrt(weight):
        start_index = 0
        for j in range(self.dofs.nfuncs):
            end_index = start_index + self.dofs.nvals_per_func[j]
            #jac[j, :] = jac[j, :] * np.sqrt(self.terms[j].weight)
            jac[start_index:end_index, :] = jac[start_index:end_index, :] \
                * np.sqrt(self.terms[j].weight)
            start_index = end_index
            
        return np.array(jac)
        
    def solve(self, grad=None):
        """
        Solve the nonlinear-least-squares minimization problem.
        """
        logger.info("Beginning solve.")
        self._init()
        if grad is None:
            grad = self.dofs.grad_avail

        x = np.copy(self.x) # For use in Bcast later.

        # Send group leaders and workers into their respective loops
        self.mpi.together = False
        if self.mpi.proc0_world:
            pass
        elif self.mpi.proc0_groups:
            self.mpi.leaders_loop(self.dofs)
        else:
            self.mpi.worker_loop(self.dofs)
            
        if self.mpi.proc0_world:
            # proc0_world does this block, running the optimization.
            x0 = np.copy(self.dofs.x)
            #print("x0:",x0)
            # Call scipy.optimize:
            if grad:
                logger.info("Using derivatives")
                print("Using derivatives")
                result = least_squares(self.f_proc0, x0, verbose=2, jac=self.jac_proc0)
            else:
                logger.info("Using derivative-free method")
                print("Using derivative-free method")
                result = least_squares(self.f_proc0, x0, verbose=2)

            logger.info("Completed solve.")
            x = result.x
            self.mpi.stop_leaders() # Proc0_world stops the leaders.

        if self.mpi.proc0_groups:
            self.mpi.stop_workers() # All group leaders stop their workers.

        self.mpi.together = True
        # Finally, make sure all procs get the optimal state vector.
        self.mpi.comm_world.Bcast(x)
        logger.debug('After Bcast, x={}'.format(x))
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
        self.mpi.mobilize_workers(x, CALCULATE_F)
        return self.f(x)

    def jac_proc0(self, x):
        """
        Similar to jac, except this version is called only by proc 0 while
        workers are in the worker loop.
        """
        if self.dofs.grad_avail:
            # proc0_world calling mobilize_workers will mobilize only group 0.
            self.mpi.mobilize_workers(x, CALCULATE_JAC)
        else:
            # fd_jac_par will be called
            self.mpi.mobilize_leaders(x)
            
        return self.jac(x)
