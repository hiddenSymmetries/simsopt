# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
Provides the ConstrainedProblem class implemented using the graph based
optimization framework.
"""

from __future__ import annotations

import logging
from typing import Union, Callable, Tuple, Sequence, Optional
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
    graph based optimization framework. A ``ConstrainedProblem`` instance has
    4 basic attributes: an objective, nonlinear constraints, 
    linear constraints, and bound constraints. Problems take the general form:

        .. math::

            \min_x f(x) 

            \\text{s.t.} 

            l_{\\text{nlc}} \leq c(x) \leq u_{\\text{nlc}}

            l_{\\text{lc}} \leq Ax \leq u_{\\text{lc}}

    Constrained optimization problems can be solved with the ``constrained_mpi_solve`` or
    ``constrained_serial_solve`` functions. Typically, this class is used for Stage-I optimization.

    Whereas linear and nonlinear constraints are passed as arguments to this class, bound constraints
    should be specified directly through the Optimizable objects. For instance, with an optimizable 
    object ``v`` we can set the upper bounds of the free DOFs associated with the current Optimizable object 
    and those of its ancestors via ``v.upper_bounds = ub`` where ``ub`` is a 1d-array. 
    To set the upper bounds on the free dofs of a single optimizable object (and not
    it's ancestors) use ``v.local_upper_bounds = ub``.
    The upper bound of a single dof can be set with ``v.set_upper_bound(dof_name,value)``.

    Args:
        f_obj (Callable): objective function handle (generally a method of an Optimizable instance)
        tuples_nlc (list): Nonlinear constraints as a sequence of triples containing 
            the nonlinear constraint function, :math:`c`, with lower and upper bounds
            i.e. ``[(c,l_{nlc},u_{nlc}), ...]``.
            Each constraint handle, :math:`c`, can be scalar-valued or be vector valued. Similarly, 
            the constraint bounds :math:`l_{\\text{nlc}}`, :math:`u_{\\text{nlc}}` can be scalars or 1d-arrays.
            Use ``+-np.inf`` to indicate unbounded components, and define equality constraints 
            by using equal upper and lower bounds.
        tuple_lc (tuple): A tuple containing the matrix :math:`A`, lower and upper bounds, for the linear
            constraints i.e. :math:`(A, l_{\\text{lc}}, u_{\\text{lc}})`. Constraint bounds can be 1d-arrays or scalars.
            Use ``+-np.inf`` in the bounds to indicate unbounded components, define equality constraints 
            by using equal upper and lower bounds.
        fail (float, optional): If an objective or nonlinear constraint evaluation fails, the value returned
            is set to this value.
    """

    def __init__(self,
                 f_obj: Callable,
                 tuples_nlc: Sequence[Tuple[Callable, Real, Real]] = None,
                 tuple_lc: Tuple[RealArray, Union[RealArray, Real], Union[RealArray, Real]] = None,
                 fail: Optional[float] = 1.0e12):

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
        Evaluates the nonlinear constraints, :math:`l_{\\text{nlc}} \leq c(x) \leq u_{\\text{nlc}}`.

        Args:
            x (array, Optional): Degrees of freedom. If not provided, the current degrees of freedom are used.
            args: Any additional arguments passed to the nonlinear constraint functions.
            kwargs: Keyword arguments passed to the nonlinear constraint functions.

        Returns:
            Array containing the nonlinear constraints, ordered as
            :math:`[l_{\\text{nlc}} - c(x), c(x) - u_{\\text{nlc}},...]`.
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
        Evaluate the objective function, :math:`f(x)`.

        Args:
            x (array, Optional): Degrees of freedom. If not provided, the current degrees of freedom are used.
            args: Any additional arguments passed to the objective function.
            kwargs: Keyword arguments passed to the objective function.

        Returns:
            `(float)` Objective function value.
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
            x (array, Optional): Degrees of freedom. If not provided, the current degrees of freedom are used.
            args: Any additional arguments passed to the objective and nonlinear constraint functions.
            kwargs: Keyword arguments passed to the objective and nonlinear constraint functions.

        Returns:
            Array containing the objective and nonlinear constraints, ordered as 
            :math:`[f(x), l_{\\text{nlc}} - c(x), c(x) - u_{\\text{nlc}},...]`
        """
        f_obj = self.objective(x, *args, **kwargs)
        out = np.array([f_obj])
        if self.has_nlc:
            f_nlc = self.nonlinear_constraints(x, *args, **kwargs)
            out = np.concatenate((out, f_nlc))
        return out
