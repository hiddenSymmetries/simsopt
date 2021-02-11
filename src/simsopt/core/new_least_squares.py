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
    problem. A LeastSquaresTerm instance has 3 basic attributes: a
    function (called f_in), a goal value (called goal), and a weight
    (sigma).  The overall value of the term is:

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
        terms = []
        for i, opt in enumerate(self.parents):
            output = np.array(opt())
            if self.func_masks is None:
                terms += [output]
            else:
                terms += [output[self.func_masks[i]]]

        terms = np.concatenate(terms)
        s = np.multiply(terms, terms)
        return np.sum(np.multiply(s, self.weights))

    def __add__(self, other: LeastSquaresProblem) -> LeastSquaresProblem:

        return LeastSquaresProblem(self.parents + other.parents,
                                   np.concatenate([self.goals, other.goals]),
                                   np.concatenate([self.weights, other.weights])
                                   )

    def residuals(self, x: Union[RealArray, IntArray] = None):
        if x is not None:
            self.x = x

        temp = np.append([f() for f in self.parents]) - self.goal
        return np.sqrt(self.weights) * temp

