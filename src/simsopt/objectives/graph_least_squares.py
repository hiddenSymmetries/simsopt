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

from .._core.graph_optimizable import DOFs, Optimizable
from ..util.types import RealArray, IntArray, BoolArray


logger = logging.getLogger(__name__)

StrSeq = Union[Sequence, Sequence[Sequence[str]]]


class LeastSquaresProblem(Optimizable):
    """
    Represents a nonlinear-least-squares problem implemented using the new
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
        opts_in: (Alternative initialization) Instead of specifying funcs_in,
                one could specify the Optimizable objects
        opt_return_fns:  (Alternative initialization) If using *opts_in*,
                specify the return functions associated with each Optimizable
                object
    """

    def __init__(self,
                 goals: Union[Real, RealArray],
                 weights: Union[Real, RealArray],
                 funcs_in: Sequence[Callable] = None,
                 opts_in: Union[Optimizable, Sequence[Optimizable]] = None,
                 opt_return_fns: StrSeq = None):

        if isinstance(goals, Real):
            goals = [goals]
        if isinstance(weights, Real):
            weights = [weights]
        if np.any(np.array(weights) < 0):
            raise ValueError('Weight cannot be negative')
        self.goals = np.array(goals)
        self.weights = np.array(weights)

        if opts_in is not None:
            if not isinstance(opts_in, ABC_Sequence):
                opts_in = [opts_in]
                #goals = [goals]
                #weights = [weights]
                if opt_return_fns is not None:
                    opt_return_fns = [opt_return_fns]

        super().__init__(opts_in=opts_in, opt_return_fns=opt_return_fns,
                         funcs_in=funcs_in)

    @classmethod
    def from_sigma(cls,
                   goals: Union[Real, RealArray],
                   sigma: Union[Real, RealArray],
                   funcs_in: Sequence[Callable] = None,
                   opts_in: Union[Optimizable, Sequence[Optimizable]] = None,
                   opt_return_fns: StrSeq = None) -> LeastSquaresProblem:
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
            opts_in: (Alternative initialization) Instead of specifying
                funcs_in, one could specify the Optimizable objects
            opt_return_fns: (Alternative initialization) If using *opts_in*,
                specify the return functions associated with each Optimizable
                object
        """
        if np.any(np.array(sigma) == 0):
            raise ValueError('sigma cannot be 0')
        if not isinstance(sigma, Real):
            sigma = np.array(sigma)

        return cls(goals, 1.0 / (sigma * sigma),
                   opts_in=opts_in,
                   opt_return_fns=opt_return_fns,
                   funcs_in=funcs_in)

    @classmethod
    def from_tuples(cls,
                    tuples: Sequence[Tuple[Callable, Real, Real]]
                    ) -> LeastSquaresProblem:
        """
        Initializes graph based LeastSquaresProblem from a sequence of tuples
        containing *f_in*, *goal*, and *weight*.

        Args:
            tuples: A sequence of tuples containing (f_in, goal, weight) in
                each tuple (the specified order matters).
        """
        funcs_in, goals, weights = zip(*tuples)
        #funcs_in = list(funcs_in)
        #goals = list(goals)
        #weights = list(weights)
        return cls(goals, weights, funcs_in=funcs_in)

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

        outputs = []
        for i, opt in enumerate(self.parents):
            out = opt(child=self, *args, **kwargs)
            output = np.array([out]) if np.isscalar(out) else np.array(out)
            outputs += [output]

        return np.concatenate(outputs) - self.goals

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

        # if x is not None:
        #    self.x = x
        #outputs = []
        #for i, opt in enumerate(self.parents):
        #    out = opt(child=self, *args, **kwargs)
        #    output = np.array([out]) if np.isscalar(out) else np.array(out)
        #    if self.opt_return_fns is None:
        #       fn_value = output
        #    elif self.func_masks[i] is None:
        #       fn_value = output
        #    else:
        #       fn_value = output[self.func_masks[i]]
        #    outputs += [output]
        #outputs = np.concatenate(outputs)
        #residuals = (outputs - self.goals) * np.sqrt(self.weights)
        #return residuals

        #print('terms at line 104', terms)
        #print('goals', self.goals )
        #terms = np.concatenate(terms) - self.goals
        #print('after subtraction', terms)
        #s = np.dot(terms, terms)
        #print('dot product', s)
        #objective = np.sum(np.multiply(s, self.weights))
        #print('final objective', objective)
        #return np.concatenate(residuals)

    def objective(self, x=None, *args, **kwargs):
        """
        Return the least squares sum

        Args:
            x: Degrees of freedom or state
            args: Any additional arguments
            kwargs: Keyword arguments
        """
        #if x is not None:
        #    self.x = x

        #outputs = []
        #for i, opt in enumerate(self.parents):
        #    out = opt(child=self, *args, **kwargs)
        #    output = np.array([out]) if np.isscalar(out) else np.array(out)
        #    outputs += [output]
        #outputs = np.concatenate(outputs)
        #diff_values = outputs - self.goals
        unweighted_residuals = self.unweighted_residuals(x, *args, **kwargs)

        s = 0
        for i, val in enumerate(unweighted_residuals):
            s += np.dot(val, val) * self.weights[i]

        return s

    return_fn_map = {'residuals': residuals, 'objective': objective}

    def __add__(self, other: LeastSquaresProblem) -> LeastSquaresProblem:
        # TODO: This could be buggy with respect to x-order after addition

        return LeastSquaresProblem(
            np.concatenate([self.goals, other.goals]),
            np.concatenate([self.weights, other.weights]),
            opts_in=(self.parents + other.parents),
            opt_return_fns=(self.get_parent_return_fns_list() +
                            other.get_parent_return_fns_list()),
        )

    #def residuals(self, x: Union[RealArray, IntArray] = None):
    #    if x is not None:
    #        self.x = x

    #    temp = np.append([f() for f in self.parents]) - self.goal
    #    return np.sqrt(self.weights) * temp
