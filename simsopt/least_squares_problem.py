"""
This module provides the LeastSquaresProblem class.
"""

import numpy as np
from scipy.optimize import least_squares
import logging
from .least_squares_term import LeastSquaresTerm
from .collect_dofs import collect_dofs

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
        dofs = collect_dofs([t.f_in for t in self.terms])
        self.nparams = len(dofs.x)
        # Transfer all non-builtin attributes of dofs to self:
        for att in dir(dofs):
            if not att.startswith('_'):
                setattr(self, att, getattr(dofs, att))

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

    def solve(self):
        """
        Solve the nonlinear-least-squares minimization problem.
        """
        self.logger.info("Beginning solve.")
        self._init()
        x0 = np.copy(self.x)
        #print("x0:",x0)
        # Call scipy.optimize:
        result = least_squares(self._residual_func, x0, verbose=2)
        self.logger.info("Completed solve.")
        #print("optimum x:",result.x)
        #print("optimum residuals:",result.fun)
        #print("optimum cost function:",result.cost)
        # Set Parameters to their values for the optimum
        self._set_dofs(result.x)

    def _set_dofs(self, x):
        """
        Call set_dofs() for each object, given a state vector x.
        """
        # Idea behind the following loops: call set_dofs no more than
        # once for each object, in case that improves performance at
        # all for the optimizable objects.
        for owner in self.owners:
            # In the next line, we make sure to cast the type to a
            # float. Otherwise get_dofs might return an array with
            # integer type.
            objx = np.array(owner.get_dofs(), dtype=np.dtype(float))
            for j in range(self.nparams):
                if self.owners[j] == owner:
                    objx[self.indices[j]] = x[j]
            owner.set_dofs(objx)
                
    def _residual_func(self, x):
        """
        This private method is passed to scipy.optimize.
        """
        self.logger.info("_residual_func called with x=" + str(x))
        self._set_dofs(x)
        residuals = [(term.f_in() - term.goal) / term.sigma for term in self.terms]
        return residuals
        
