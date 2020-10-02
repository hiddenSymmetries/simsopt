"""
This module provides the LeastSquaresProblem class, as well as the
associated class LeastSquaresTerm.
"""

import numpy as np
from scipy.optimize import least_squares
import logging
from .dofs import Dofs
from .util import isnumber
from .optimizable import function_from_user

class LeastSquaresTerm:
    """
    This class represents one term in a nonlinear-least-squares
    problem. A LeastSquaresTerm instance has 3 basic attributes: a
    function (called f_in), a goal value (called goal), and a weight
    (sigma).  The overall value of the term is:

    f_out = ((f_in - goal) / sigma) ** 2.
    """

    def __init__(self, f_in, goal, sigma):
        if not isnumber(goal):
            raise ValueError('goal must be a float or int')
        if not isnumber(sigma):
            raise ValueError('sigma must be a float or int')
        if sigma == 0:
            raise ValueError('sigma cannot be 0')
        self.f_in = function_from_user(f_in)
        self.goal = float(goal)
        self.sigma = float(sigma)
        self.fixed = np.full(0, False)

    def f_out(self):
        """
        Return the overall value of this least-squares term.
        """
        temp = (self.f_in() - self.goal) / self.sigma
        return temp * temp 

    
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

        self.logger = logging.getLogger(__name__)

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
    def objective(self):
        """
        Return the value of the total objective function, by summing
        the terms.
        """
        self.logger.info("objective called.")
        sum = 0
        for term in self.terms:
            sum += term.f_out()
        return sum

    @property
    def x(self):
        """
        Return the state vector.
        """
        return self.dofs.x

    def solve(self):
        """
        Solve the nonlinear-least-squares minimization problem.
        """
        self.logger.info("Beginning solve.")
        self._init()
        x0 = np.copy(self.dofs.x)
        #print("x0:",x0)
        # Call scipy.optimize:
        result = least_squares(self._residual_func, x0, verbose=2)
        self.logger.info("Completed solve.")
        #print("optimum x:",result.x)
        #print("optimum residuals:",result.fun)
        #print("optimum cost function:",result.cost)
        # Set Parameters to their values for the optimum
        self.dofs.set(result.x)

    #def _set_dofs(self, x):
    #    """
    #    Call set_dofs() for each object, given a state vector x.
    #    """
    #    # Idea behind the following loops: call set_dofs exactly once
    #    # once for each object, in case that improves performance at
    #    # all for the optimizable objects.
    #    for owner in self.all_owners:
    #        # In the next line, we make sure to cast the type to a
    #        # float. Otherwise get_dofs might return an array with
    #        # integer type.
    #        objx = np.array(owner.get_dofs(), dtype=np.dtype(float))
    #        for j in range(self.nparams):
    #            if self.dof_owners[j] == owner:
    #                objx[self.indices[j]] = x[j]
    #        owner.set_dofs(objx)
                
    def _residual_func(self, x):
        """
        This private method is passed to scipy.optimize.
        """
        self.logger.info("_residual_func called with x=" + str(x))
        self.dofs.set(x)
        residuals = [(term.f_in() - term.goal) / term.sigma for term in self.terms]
        return residuals
        
