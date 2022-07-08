# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
Provides the LeastSquaresProblem class implemented using the new graph based
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

__all__ = ['LeastSquaresProblem']

logger = logging.getLogger(__name__)

StrSeq = Union[Sequence, Sequence[Sequence[str]]]


class LeastSquaresProblem(Optimizable):
    """
    Represents a nonlinear-least-squares problem implemented using the 
    graph based optimization framework. A LeastSquaresProblem instance has
    3 basic attributes: a set of functions (`f_in`), target values for each
    of the functions (`goal`), and weights.  The residual
    (`f_out`) for each of the `f_in` is defined as:

    .. math::

        f_{out} = weight * (f_{in} - goal) ^ 2

    Args:
        goals: Targets for residuals in optimization
        weights: Weight associated with each of the residual
        funcs_in: Input functions (Generally one of the output functions of
                  the Optimizable instances
        depends_on: (Alternative initialization) Instead of specifying funcs_in,
                one could specify the Optimizable objects
        opt_return_fns:  (Alternative initialization) If using *depends_on*,
                specify the return functions associated with each Optimizable
                object
    """

    def __init__(self,
                 goals: Union[Real, RealArray],
                 weights: Union[Real, RealArray],
                 funcs_in: Sequence[Callable] = None,
                 depends_on: Union[Optimizable, Sequence[Optimizable]] = None,
                 opt_return_fns: StrSeq = None,
                 fail: Union[None, float] = 1.0e12):

        if isinstance(goals, Real):
            goals = [goals]
        if isinstance(weights, Real):
            weights = [weights]
        if np.any(np.asarray(weights) < 0):
            raise ValueError('Weight cannot be negative')
        self.goals = np.asarray(goals)
        self.inp_weights = np.asarray(weights)
        self.fail = fail

        # Attributes for function evaluation
        self.nvals = 0
        self.first_eval = True

        if depends_on is not None:
            if not isinstance(depends_on, ABC_Sequence):
                depends_on = [depends_on]
                if opt_return_fns is not None:
                    opt_return_fns = [opt_return_fns]

        super().__init__(depends_on=depends_on, opt_return_fns=opt_return_fns,
                         funcs_in=funcs_in)

    @classmethod
    def from_sigma(cls,
                   goals: Union[Real, RealArray],
                   sigma: Union[Real, RealArray],
                   funcs_in: Sequence[Callable] = None,
                   depends_on: Union[Optimizable, Sequence[Optimizable]] = None,
                   opt_return_fns: StrSeq = None,
                   fail: Union[None, float] = 1.0e12) -> LeastSquaresProblem:
        r"""
        Define the LeastSquaresProblem with

        .. math::
            \sigma = 1/\sqrt{weight}, \text{so} \\
            f_{out} = \left(\frac{f_{in} - goal}{\sigma}\right) ^ 2.

        Args:
            goals: Targets for residuals in optimization
            sigma: Inverse of the sqrt of the weight associated with each
                of the residual
            funcs_in: Input functions (Generally one of the output functions of
                the Optimizable instances
            depends_on: (Alternative initialization) Instead of specifying
                funcs_in, one could specify the Optimizable objects
            opt_return_fns: (Alternative initialization) If using *depends_on*,
                specify the return functions associated with each Optimizable
                object
        """
        if np.any(np.array(sigma) == 0):
            raise ValueError('sigma cannot be 0')
        if not isinstance(sigma, Real):
            sigma = np.array(sigma)

        return cls(goals, 1.0 / (sigma * sigma),
                   depends_on=depends_on,
                   opt_return_fns=opt_return_fns,
                   funcs_in=funcs_in,
                   fail=fail)

    @classmethod
    def from_tuples(cls,
                    tuples: Sequence[Tuple[Callable, Real, Real]],
                    fail: Union[None, float] = 1.0e12) -> LeastSquaresProblem:
        """
        Initializes graph based LeastSquaresProblem from a sequence of tuples
        containing *f_in*, *goal*, and *weight*.

        Args:
            tuples: A sequence of tuples containing (f_in, goal, weight) in
                each tuple (the specified order matters).
        """
        funcs_in, goals, weights = zip(*tuples)
        return cls(goals, weights, funcs_in=funcs_in, fail=fail)

    def unweighted_residuals(self, x=None, *args, **kwargs):
        """
        Return the unweighted residuals (f_in - goal)

        Args:
            x: Degrees of freedom or state
            args: Any additional arguments
            kwargs: Keyword arguments
        """
        if x is not None:
            self.x = x

        if self.new_x:
            outputs = []
            new_weights = []
            for i, fn in enumerate(self.funcs_in):
                try:
                    out = fn(*args, **kwargs)
                except ObjectiveFailure:
                    logger.warning(f"Function evaluation failed for {fn}")
                    if self.fail is None or self.first_eval:
                        raise

                    break

                output = np.array([out]) if not np.ndim(out) else np.asarray(out)
                output -= self.goals[i]
                if self.first_eval:
                    self.nvals += len(output)
                    logger.debug(f"{i}: first eval {self.nvals}")
                new_weights += [self.inp_weights[i]] * len(output)
                outputs += [output]
            else:
                if self.first_eval:
                    self.first_eval = False
                self.weights = np.asarray(new_weights)
                self.cache = np.concatenate(outputs)
                self.new_x = False
                return self.cache

            # Reached here after encountering break in for loop
            self.cache = np.full(self.nvals, self.fail)
            self.new_x = False
            return self.cache
        else:
            return self.cache

    def residuals(self, x=None, *args, **kwargs):
        """
        Return the weighted residuals

        Args:
            x: Degrees of freedom or state
            args: Any additional arguments
            kwargs: Keyword arguments
        """
        unweighted_residuals = self.unweighted_residuals(x, *args, **kwargs)
        return unweighted_residuals * np.sqrt(self.weights)

    def objective(self, x=None, *args, **kwargs):
        """
        Return the least squares sum

        Args:
            x: Degrees of freedom or state
            args: Any additional arguments
            kwargs: Keyword arguments
        """
        logger.info(f"objective() called with x={x}")
        unweighted_residuals = self.unweighted_residuals(x, *args, **kwargs)

        s = 0
        for i, val in enumerate(unweighted_residuals):
            s += np.dot(val, val) * self.weights[i]

        logger.info(f"objective(): {s}")
        return s

    return_fn_map = {'residuals': residuals, 'objective': objective}

    def __add__(self, other: LeastSquaresProblem) -> LeastSquaresProblem:
        return LeastSquaresProblem(
            np.concatenate([self.goals, other.goals]),
            np.concatenate([self.inp_weights, other.inp_weights]),
            depends_on=(self.parents + other.parents),
            opt_return_fns=(self.get_parent_return_fns_list() +
                            other.get_parent_return_fns_list()),
            fail=max(self.fail, other.fail)
        )
