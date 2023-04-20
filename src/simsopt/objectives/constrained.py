# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
Provides the ConstrainedProblem class implemented using the graph based
optimization framework.
"""

from __future__ import annotations

import logging
import warnings
from typing import Union, Callable, Tuple, Sequence
from numbers import Real

import numpy as np

from .._core.optimizable import Optimizable
from .._core.util import ObjectiveFailure
from .._core.types import RealArray

__all__ = ['ConstrainedProblem']

logger = logging.getLogger(__name__)


class ConstrainedProblem(Optimizable):
    """
    Represents a nonlinear, constrained optimization problem implemented using the 
    graph based optimization framework. A ConstrainedProblem instance has
    4 basic attributes: an objective (`f`), nonlinear constraints (`c`), 
    linear constraints, and bound constraints. Problems take the general form:

    .. math::

        \min_x f(x) 
        s.t. 
          l_{nlc} \le c(x) \le u_{nlc}
          l_{lc} \le Ax \le u_{lc}

    Bound constraints should be specified directly through the Optimizable objects. 
    For instance, with an optimizable object `v` we can set the 
    upper bounds of the free DOFs associated with the current Optimizable object 
    and those of its ancestors via `v.upper_bounds = ub` where ub is a 1d-array. 
    To set the upper bounds on the free dofs of a single optimizable object (and not
    it's ancestors) use `v.local_upper_bounds = ub`.
    The upper bound of a single dof can be set with `v.set_upper_bound(dof_name,value)`.

    Args:
        f_obj: objective function handle (generally one of the output functions of
            the Optimizable instances)
        tuples_nlc: Nonlinear constraints as a sequence of triples containing 
                    the nonlinear constraint function c with lower and upper bounds
                    i.e. `[(c,l_{nlc},u_{nlc}), ...]`.
                    Constraint handle can (`c`) can be vector-valued or scalar-valued.
                    Constraint bounds can also be array or scalar.
                    Use +- np.inf to indicate unbounded components.
                    Define equality constraints by using equal upper and lower bounds.
        tuple_lc: Linear constraints as a triple containing the 2d-array A,
                  lower bound `l_{lc}`, and upper bound `u_{lc}`, 
                  i.e. `(A,l_{lc},u_{lc})`.
                  Constraint bounds can be 1d arrays or scalars.
                  Use +- np.inf in the bounds to indicate unbounded components.
                  Define equality constraints by using equal upper and lower bounds.
    """

    def __init__(self,
                 f_obj: Callable,
                 tuples_nlc: Sequence[Tuple[Callable, Real, Real]] = None,
                 tuple_lc: Tuple[RealArray, Union[RealArray, Real], Union[RealArray, Real]] = None,
                 fail: Union[None, float] = 1.0e12):

        self.fail = fail

        # Attributes for function evaluation
        self.nvals = 0
        self.first_eval_obj = True
        self.first_eval_con = True

        # unpack the nonlinear constraints
        if tuples_nlc is not None:
            f_nlc, lhs_nlc, rhs_nlc = zip(*tuples_nlc)
            funcs_in = [f_obj, *f_nlc]
            self.has_nlc = True
            self.lhs_nlc = lhs_nlc
            self.rhs_nlc = rhs_nlc
        else:
            funcs_in = [f_obj]  
            self.has_nlc = False

        # unpack the linear constraints
        if tuple_lc:
            self.A_lc = np.asarray(tuple_lc[0])
            self.l_lc = np.asarray(tuple_lc[1]) if np.ndim(tuple_lc[1]) else float(tuple_lc[1])
            self.u_lc = np.asarray(tuple_lc[2]) if np.ndim(tuple_lc[2]) else float(tuple_lc[2])
            self.has_lc = True
        else:
            self.has_lc = False

        # make our class Optimizable
        super().__init__(funcs_in=funcs_in)

    def nonlinear_constraints(self, x=None, *args, **kwargs):
        """
        Evaluates the Nonlinear constraints, l_c <= c(x) <= u_c.
        Returns an array [l_c - c(x), c(x) - u_c,...].

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
        if not self.has_nlc:
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

                # evaluate lhs as lhs - c(x) <= 0
                if np.any(np.isfinite(self.lhs_nlc[i])):
                    diff = np.array(self.lhs_nlc[i]) - out
                    output = np.array([diff]) if not np.ndim(diff) else np.asarray(diff)
                    outputs += [output]
                    if self.first_eval_con:
                        self.nvals += len(output)
                        logger.debug(f"{i}: first eval {self.nvals}")

                # evaluate rhs as c(x) - rhs <= 0
                if np.any(np.isfinite(self.rhs_nlc[i])):
                    diff = out - np.array(self.rhs_nlc[i]) 
                    output = np.array([diff]) if not np.ndim(diff) else np.asarray(diff)
                    outputs += [output]
                    if self.first_eval_con:
                        self.nvals += len(output)
                        logger.debug(f"{i}: first eval {self.nvals}")

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

            if self.first_eval_obj:
                self.first_eval_obj = False

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
        f_obj = self.objective(x, *args, **kwargs)
        out = np.array([f_obj])
        if self.has_nlc:
            f_nlc = self.nonlinear_constraints(x, *args, **kwargs)
            out = np.concatenate((out, f_nlc))
        return out


