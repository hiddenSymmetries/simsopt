"""
This module provides the LeastSquaresTerm class.
"""

import numpy as np
from .util import isnumber, function_from_user

class LeastSquaresTerm:
    """
    This class represents one term in a nonlinear-least-squares
    problem. A LeastSquaresTerm instance has 3 basic attributes: a
    function (called target), a goal value (called goal), and a weight
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
