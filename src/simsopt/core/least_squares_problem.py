# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the LeastSquaresProblem class, as well as the
associated class LeastSquaresTerm.
"""

import numpy as np
import logging
import warnings

from scipy.optimize import least_squares
from mpi4py import MPI
from .dofs import Dofs
from .util import isnumber
from .optimizable import function_from_user, Target


logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)


class LeastSquaresTerm:
    """
    This class represents one term in a nonlinear-least-squares
    problem. A LeastSquaresTerm instance has 3 basic attributes: a
    function (called f_in), a goal value (called goal), and a weight
    (sigma).  The overall value of the term is:

    f_out = weight * (f_in - goal) ** 2.
    """

    def __init__(self, f_in, goal, weight): #=None, sigma=None):
        self.f_in = function_from_user(f_in)
        self.goal = goal
        if not isnumber(weight): # Bharat's comment: Do we need this check?
            raise TypeError('Weight must be a float or int')
        if weight < 0:
            raise ValueError('Weight cannot be negative')
        self.weight = float(weight)
        # self.fixed = np.full(0, False) # What is this line for?

    @classmethod
    def from_sigma(cls, f_in, goal, sigma):
        """
        Define the LeastSquaresTerm with sigma = 1 / sqrt(weight), so

        f_out = ((f_in - goal) / sigma) ** 2.
        """
        if sigma == 0:
            raise ValueError('sigma cannot be 0')
        if not isnumber(sigma): # Bharat's comment: Do we need this check?
            raise TypeError('sigma must be a float or int')
        return cls(f_in, goal, 1.0 / float(sigma * sigma))

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

    def __init__(self, terms):
        """
        The argument "terms" must be convertable to a list by the list()
        subroutine. Each entry of the resulting list must either have
        type LeastSquaresTerm or else be a list or tuple of the form
        (function, goal, weight) or (object, attribute_str, goal,
        weight).
        """

        #try:
        #   terms = list(terms)
        #except:
        #    raise ValueError("terms must be convertable to a list by the "
        #                     "list(terms) command.")

        # For each item provided in the list, either convert to a
        # LeastSquaresTerm or, if it is already a LeastSquaresTerm,
        # use it directly.
        self.terms = []
        msg = 'Each term must be either (1) a LeastSquaresTerm or (2) a list '\
              'or tuple of the form (function, goal, weight) or (object, '    \
              'attribute_str, goal, weight)'
        for term in terms:
            if isinstance(term, LeastSquaresTerm):
                self.terms.append(term)
            else: # Expect the term to be an Iterable, but don't check
                if len(term) == 4: # 4 item list is a special case
                    lst = LeastSquaresTerm(Target(*term[:2]), *term[2:])
                else:
                    lst = LeastSquaresTerm(*term)
                self.terms.append(lst)

        if not len(self.terms):
            raise ValueError("At least 1 LeastSquaresTerm must be as argument")

        self._init()

    def _init(self):
        """
        Call collect_dofs() on the list of terms to set x, mins, maxs, names,
        etc. This is done both when the object is created, so 'objective' 
        works immediately, and also at the start of solve()
        """
        self.dofs = Dofs([t.f_in for t in self.terms])

    @property
    def x(self):
        """
        Return the state vector.
        """
        # Delegate to Dofs:
        return self.dofs.x

    @x.setter
    def x(self, x):
        """
        Sets the global state vector to x.
        """
        # Delegate to Dofs:
        if x is not None:
            self.dofs.set(x)
        else:
            warnings.warn("Supplied a null object as state vector. Ignoring it")

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
        self.x = x

        return sum(t.f_out() for t in self.terms)

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
        self.x = x

        # Importantly for MPI, the next line calls the functions in
        # the same order that Dofs.f() does. Proc0 calls this function
        # whereas worker procs call Dofs.f().
        f_unscaled = self.dofs.f()
        residuals = np.zeros(len(f_unscaled))
        start_index = 0
        for j in range(self.dofs.nfuncs):
            term = self.terms[j]
            end_index = start_index + self.dofs.nvals_per_func[j]
            residuals[start_index:end_index] = \
                (f_unscaled[start_index:end_index] - term.goal) * \
                np.sqrt(term.weight)
            start_index = end_index
        # residuals = [(term.f_in() - term.goal) * np.sqrt(term.weight) for \
        #               term in self.terms]
        # return np.array(residuals)
        return residuals
        
    def scale_dofs_jac(self, jmat):
        """
        Given a Jacobian matrix j for the Dofs() associated to this
        least-squares problem, return the scaled Jacobian matrix for
        the least-squares residual terms. This function does not
        actually compute the Dofs() Jacobian, since sometimes we would
        compute that directly whereas other times we might compute it
        with serial or parallel finite differences. The provided jmat
        is scaled in-place.
        """
        logger.info("scale_dofs_jac() called")

        # Scale rows by sqrt(weight):
        start_index = 0
        for j in range(self.dofs.nfuncs):
            end_index = start_index + self.dofs.nvals_per_func[j]
            #jac[j, :] = jac[j, :] * np.sqrt(self.terms[j].weight)
            jmat[start_index:end_index, :] = jmat[start_index:end_index, :] \
                * np.sqrt(self.terms[j].weight)
            start_index = end_index
            
        return np.array(jmat)
    
    def jac(self, x=None, **kwargs):
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

        kwargs is passed to Dofs.fd_jac().
        """
        logger.info("jac() called with x=" + str(x))

        self.x = x

        # This next bit does the hard work of evaluating the
        # Jacobian.
        # Bharat's comment: The conditional logic should be delegated to
        # Bharat's comment: Dofs class
        if self.dofs.grad_avail:
            logger.debug('Calling analytic Jacobian')
            jmat = self.dofs.jac()
        else:
            logger.debug('Calling finite_difference Jacobian')
            jmat = self.dofs.fd_jac(**kwargs)

        # Scale by sqrt(weight) factor:
        return self.scale_dofs_jac(jmat)
