# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
Provides the ConstrainedProblem class implemented using the new graph based
optimization framework.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence as ABC_Sequence
from typing import Union, Callable, Tuple, Sequence
from numbers import Real

import numpy as np

from .._core.optimizable import Optimizable
from .._core.util import ObjectiveFailure
from .._core.types import RealArray, IntArray, BoolArray

__all__ = ['ConstrainedProblem']

logger = logging.getLogger(__name__)

StrSeq = Union[Sequence, Sequence[Sequence[str]]]


class ConstrainedProblem(Optimizable):
    """
    Represents a nonlinear, constrained optimization problem implemented using the 
    graph based optimization framework. A ConstrainedProblem instance has
    4 basic attributes: an objective (`f`), nonlinear constraints (`c`), 
    linear constraints, and bound constraints. Problems take the general form:

    .. math::

        \min_x f(x) 
        s.t. 
          l_c \le c(x) \le u_c
          Ax \le b
          l_x \le x \le u_x
        

    Args:
        f_obj: objective function handle (Generally one of the output functions of
                  the Optimizable instances
        tuples_nlc: Nonlinear constraints as a sequence of triples containing 
                    the nonlinear constraint function c with lower and upper bounds
                    i.e. (c,l_c,u_c). Use +- np.inf to indicate unbounded components.
        tuple_lc: Linear constraints as a tuple containing the 2d-array A, and 1d array
                  b, i.e. (A,b).
        lb: 1d-array of lower bounds, -np.inf can be used if an entry is unconstrained.
            Set a componenent equal to the upper bound to enforce an equality constraints.
        ub: 1d-array of upper bounds, np.inf can be used if an entry is unconstrained
            Set a componenent equal to the lower bound to enforce an equality constraints.
    """

    def __init__(self,
                 f_obj: Callable,
                 tuples_nlc: Sequence[Tuple[Callable, Real,Real]] = None,
                 tuple_lc: Tuple[RealArray, Union[RealArray, Real]] = None,
                 lb: RealArray = None,
                 ub: RealArray = None,
                 fail: Union[None, float] = 1.0e12):

        self.fail = fail

        # Attributes for function evaluation
        self.nvals = 0
        self.first_eval_obj = True
        self.first_eval_con = True

        self.has_bounds = False
        if lb is None:
            self.lb = -np.inf
        else:
            self.lb = np.asarray(lb, np.double)
            self.has_bounds = True

        if ub is None:
            self.ub = np.inf
        else:
            self.ub = np.asarray(ub, np.double)
            self.has_bounds = True


        # unpack the nonlinear constraints
        if tuples_nlc is not None:
            f_nlc, lhs_nlc, rhs_nlc = zip(*tuples_nlc)
            funcs_in = [f_obj, *f_nlc]
            self.has_nlc = True
            lhs_nlc = np.array(lhs_nlc,dtype=float)
            rhs_nlc = np.array(rhs_nlc,dtype=float)
            self.lhs_nlc = lhs_nlc
            self.rhs_nlc = rhs_nlc
        else:
            funcs_in = [f_obj]  
            self.has_nlc = False

        # unpack the linear constraints
        if tuple_lc:
            A_lc = np.atleast_2d(tuple_lc[0])
            b_lc = np.atleast_1d(tuple_lc[1])
            if np.shape(A_lc)[0] != len(b_lc):
                raise ValueError(f"Linear constraint A and b do not have compatible shapes.")
            
            self.A_lc = A_lc
            self.b_lc = b_lc
            self.has_lc = True
        else:
            self.has_lc = False

        super().__init__(funcs_in=funcs_in)


    def nonlinear_constraints(self, x=None, *args, **kwargs):
        """
        Return the nonlinear constraint left hand side, c(x)

        Args:
            x: Degrees of freedom or state
            args: Any additional arguments
            kwargs: Keyword arguments
        """
        if x is not None:
            # only change x if different than last evaluated
            if np.any(self.x != x):
                self.x = x

        if self.new_x:
            # empty the cache for objective and constraint
            self.objective_cache = None
            self.constraint_cache = None

        # get the constraint funcs
        fn_nlc = self.funcs_in[1:]
        if len(fn_nlc) == 0:
            # No nonlinear constraints to evaluate
            raise RuntimeError

        if (self.constraint_cache is None):
            outputs = []
            for i, fn in enumerate(fn_nlc):

                try:
                    out = fn(*args, **kwargs)
                except ObjectiveFailure:
                    logger.warning(f"Function evaluation failed for {fn}")
                    if self.fail is None or self.first_eval_con:
                        raise

                    break

                output = np.array([out]) if not np.ndim(out) else np.asarray(out)
                if self.first_eval_con:
                    self.nvals += len(output)
                    logger.debug(f"{i}: first eval {self.nvals}")
                outputs += [output]
            else:
                if self.first_eval_con:
                    self.first_eval_con = False
                self.constraint_cache = np.concatenate(outputs)
                self.new_x = False
                return self.constraint_cache

            # Reached here after encountering break in for loop
            self.constraint_cache = np.full(self.nvals, self.fail)
            self.new_x = False
            return self.constraint_cache
        else:
            return self.constraint_cache


    def objective(self, x=None, *args, **kwargs):
        """
        Return the objective function

        Args:
            x: Degrees of freedom or state
            args: Any additional arguments
            kwargs: Keyword arguments
        """
        if x is not None:
            # only change x if different than last evaluated
            if np.any(self.x != x):
                self.x = x

        if self.new_x:
            # empty the cache for objective and constraint
            self.objective_cache = None
            self.constraint_cache = None

        if (self.objective_cache is None):
            fn = self.funcs_in[0]
            try:
                out = fn(*args, **kwargs)
            except ObjectiveFailure:
                logger.warning(f"Function evaluation failed for {fn}")
                if self.fail is None or self.first_eval_obj:
                    raise
                out = self.fail

            self.objective_cache = out
            self.new_x = False

            return self.objective_cache
        else:
            return self.objective_cache


    def all_funcs(self, x=None, *args, **kwargs):
        """
        Evaluate the objective and nonlinear constraints.

        Args:
            x: Degrees of freedom or state
            args: Any additional arguments
            kwargs: Keyword arguments
        """
        f_obj = self.objective(x,*args,**kwargs)
        out = np.array([f_obj])
        if self.has_nlc:
            f_nlc = self.nonlinear_constraints(x,*args,**kwargs)
            out = np.concatenate((out,f_nlc))
        return out


    #return_fn_map = {'residuals': residuals, 'objective': objective}

