# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the LeastSquaresProblem class.
"""

from __future__ import annotations

import numpy as np
import logging
import warnings

from collections.abc import Sequence
from typing import Union, Callable, Tuple
from numbers import Real
from mpi4py import MPI
from .new_optimizable import DOFs, Optimizable
from .util import RealArray, IntArray, BoolArray


logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

StrSeq = Union[Sequence, Sequence[Sequence[str]]]

class LeastSquaresProblem(Optimizable):
    """
    This class represents a nonlinear-least-squares
    problem. A LeastSquaresProblem instance has 3 basic attributes: a
    set of functions (called f_in), target values for each of the functions
    (called goal), and weights (sigma).  The function returns f_out for each
    of the term defined as:

    f_out = weight * (f_in - goal) ** 2.
    """

    def __init__(self,
                 goals: Union[Real, RealArray],
                 weights: Union[Real, RealArray],
                 opts_in: Union[Optimizable, Sequence[Optimizable]] = None,
                 opt_return_fns: StrSeq = None,
                 funcs_in: Sequence[Callable] = None):
        """

        Args:
            funcs_in:
            goals:
            weights:
        """
        if isinstance(goals, Real):
            goals = [goals]
        if isinstance(weights, Real):
            weights = [weights]
        if np.any(np.array(weights) < 0):
            raise ValueError('Weight cannot be negative')
        self.goals = np.array(goals)
        self.weights = np.array(weights)

        if opts_in is not None:
            if not isinstance(opts_in, Sequence):
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
                   opts_in: Union[Optimizable, Sequence[Optimizable]] = None,
                   opt_return_fns: StrSeq = None,
                   funcs_in: Sequence[Callable] = None) -> LeastSquaresProblem:
        """
        Define the LeastSquaresProblem with sigma = 1 / sqrt(weight), so

        f_out = ((f_in - goal) / sigma) ** 2.
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
        funcs_in, goals, weights = zip(*tuples)
        #funcs_in = list(funcs_in)
        #goals = list(goals)
        #weights = list(weights)
        return cls(goals, weights, funcs_in=funcs_in)

    def residuals(self, x=None, *args, **kwargs):
        """
        Return the residuals
        """
        if x is not None:
            self.x = x
        outputs = []
        for i, opt in enumerate(self.parents):
            out = opt(child=self)
            output = np.array([out]) if np.isscalar(out) else np.array(out)
            #if self.opt_return_fns is None:
            #    fn_value = output
            #elif self.func_masks[i] is None:
            #    fn_value = output
            #else:
            #    fn_value = output[self.func_masks[i]]
            outputs += [output]
        outputs = np.concatenate(outputs)
        residuals = (outputs - self.goals) * np.sqrt(self.weights)
        return residuals

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
        """
        if x is not None:
            self.x = x

        outputs = []
        for i, opt in enumerate(self.parents):
            out = opt(child=self)
            output = np.array([out]) if np.isscalar(out) else np.array(out)
            outputs += [output]
        outputs = np.concatenate(outputs)
        diff_values = outputs - self.goals
        s = 0
        for i, val in enumerate(diff_values):
            s += np.dot(val, val) * self.weights[i]

        return s

    return_fn_map = {'residuals': residuals}

    def __add__(self, other: LeastSquaresProblem) -> LeastSquaresProblem:

        return LeastSquaresProblem(
            np.concatenate([self.goals, other.goals]),
            np.concatenate([self.weights, other.weights]),
            self.parents + other.parents,
            self.get_parent_return_fns_list() + other.get_parent_return_fns_list(),
            )

    #def residuals(self, x: Union[RealArray, IntArray] = None):
    #    if x is not None:
    #        self.x = x

    #    temp = np.append([f() for f in self.parents]) - self.goal
    #    return np.sqrt(self.weights) * temp
