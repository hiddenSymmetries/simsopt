"""
This module provides the LeastSquaresTerm class.
"""

from .parameter import Parameter, isnumber
from .target import Target

class LeastSquaresTerm:
    """
    This class represents one term in a nonlinear-least-squares
    problem. A LeastSquaresTerm instance has 3 basic attributes: a
    function (called target), a goal value (called goal), and a weight
    (sigma).  The overall value of the term is:

    ((target - goal) / sigma) ** 2.
    """

    def __init__(self, target, goal, sigma):
        if not isinstance(target, Target):
            raise ValueError('target must be an instance of Target')
        if not isnumber(goal):
            raise ValueError('goal must be a float or int')
        if not isnumber(sigma):
            raise ValueError('sigma must be a float or int')
        if sigma == 0:
            raise ValueError('sigma cannot be 0')
        self._in_target = target
        # If goal or sigma is an int, convert to a float so we don't
        # have integer division by mistake:
        self._goal = float(goal)
        self._sigma = float(sigma)
        self._out_target = Target(self._in_target.parameters, self._out_function)

    @property
    def in_target(self):
        """
        Return the Target object used for the input to this
        least-squares term.  For simplicity, target is read-only.
        """
        return self._in_target

    @property
    def out_target(self):
        """
        Return a Target object representing the output of this
        least-squares term, i.e. a shifted and scaled version of the
        input Target.  For simplicity, out_target is read-only.
        """
        return self._out_target

    @property
    def goal(self):
        """
        For simplicity, goal is read-only.
        """
        return self._goal

    @property
    def sigma(self):
        """
        For simplicity, sigma is read-only.
        """
        return self._sigma

    @property
    def in_val(self):
        """
        This property is a shorthand for target.evaluate().
        """
        return self._in_target.evaluate()

    @property
    def out_val(self):
        """
        Return the overall value of this least-squares term.
        """
        temp = (self._in_target.evaluate() - self._goal) / self._sigma
        return temp * temp 

    def _out_function(self):
        """
        _out_function is the same as out_val but is a method instead
        of a property.
        """
        return self.out_val
