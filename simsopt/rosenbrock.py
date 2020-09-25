"""
This module provides a minimal object that possesses Parameters and
Targets.
"""

import numpy as np
from .parameter import Parameter
from .target import Target
import logging

class Rosenbrock:
    """
    This class defines a minimal object that possesses Parameters and
    Targets.
    """

    def __init__(self, a=1, b=100):
        self.a = a
        self.sqrtb = np.sqrt(b)
        owner = " for Rosenbrock object " + str(hex(id(self)))
        self.x1 = Parameter(0.0, self.reset, name="x1" + owner)
        self.x2 = Parameter(0.0, self.reset, name="x2" + owner)
        params = {self.x1, self.x2}
        self.need_to_run_code = True
        self.target1 = Target(params, self.evaluate_target1)
        self.target2 = Target(params, self.evaluate_target2)

    def reset(self):
        logger = logging.getLogger(__name__)
        logger.info("Resetting")
        self.need_to_run_code = True

    def long_computation(self):
        logger = logging.getLogger(__name__)
        if self.need_to_run_code:
            logger.info("Running long computation...")
            self.code_output = (self.x2.val - self.x1.val * self.x1.val) * self.sqrtb
        self.need_to_run_code = False

    def evaluate_target1(self):
        """
        First term in the 2D Rosenbrock function.
        """
        return self.a - self.x1.val

    def evaluate_target2(self):
        """
        Second term in the 2D Rosenbrock function.
        """
        self.long_computation()
        return self.code_output
