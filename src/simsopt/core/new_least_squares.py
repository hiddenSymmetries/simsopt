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
from typing import Union
from numbers import Real
from mpi4py import MPI
from .new_optimizable import DOFs, Optimizable
from .util import RealArray, IntArray, BoolArray


logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)


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
                 funcs_in: Union[Optimizable, Sequence[Optimizable]],
                 goals: Union[Real, RealArray],
                 weights: Union[Real, RealArray],
                 func_masks: Union[bool, BoolArray, Sequence[BoolArray]] = None):
        """

        Args:
            funcs_in:
            goals:
            weights:
        """
        if isinstance(goals, Real):
            self.goals = np.array([goals])
        else:
            self.goals = np.array(goals)

        if np.any(weights < 0):
            raise ValueError('Weight cannot be negative')
        if isinstance(weights, Real):
            self.weights = np.array([weights])
        else:
            self.weights = np.array(weights)

        if not isinstance(funcs_in, Sequence):
            funcs_in = [funcs_in]

        if isinstance(func_masks, bool):
            self.func_masks = [np.array([func_masks])]
        else:
            self.func_masks = func_masks

        super().__init__(funcs_in=funcs_in)

    @classmethod
    def from_sigma(cls,
                   funcs_in: Union[Optimizable, Sequence[Optimizable]],
                   goals: Union[Real, RealArray],
                   sigma: Union[Real, RealArray],
                   func_masks: Union[bool, BoolArray, Sequence[BoolArray]] = None)\
            -> LeastSquaresProblem:
        """
        Define the LeastSquaresProblem with sigma = 1 / sqrt(weight), so

        f_out = ((f_in - goal) / sigma) ** 2.
        """
        if np.any(sigma == 0):
            raise ValueError('sigma cannot be 0')
        if isinstance(sigma, Real):
            sigma = np.array([sigma])
        else:
            sigma = np.array(sigma)

        return cls(funcs_in, goals, 1.0 / (sigma * sigma), func_masks)

    def f(self):
        """
        Return the overall value of this least-squares term.
        """
        s = 0
        residuals = []
        for i, opt in enumerate(self.parents):
            out = opt()
            output = np.array([out]) if np.isscalar(out) else np.array(out)
            if self.func_masks is None:
                fn_value = output
            elif self.func_masks[i] is None:
                fn_value = output
            else:
                fn_value = output[self.func_masks[i]]
            residuals += [(fn_value - self.goals[i]) * np.sqrt(self.weights[i])]

        #print('terms at line 104', terms)
        #print('goals', self.goals )
        #terms = np.concatenate(terms) - self.goals
        #print('after subtraction', terms)
        #s = np.dot(terms, terms)
        #print('dot product', s)
        #objective = np.sum(np.multiply(s, self.weights))
        #print('final objective', objective)
        return np.concatenate(residuals)

    def __add__(self, other: LeastSquaresProblem) -> LeastSquaresProblem:

        return LeastSquaresProblem(self.parents + other.parents,
                                   np.concatenate([self.goals, other.goals]),
                                   np.concatenate([self.weights, other.weights])
                                   )

    #def residuals(self, x: Union[RealArray, IntArray] = None):
    #    if x is not None:
    #        self.x = x

    #    temp = np.append([f() for f in self.parents]) - self.goal
    #    return np.sqrt(self.weights) * temp