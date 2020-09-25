"""
This module provides the Target class for quantities that can be
targeted in optimization.
"""

from .parameter import Parameter

class Target:
    """
    Target is an abstract base class for any scalar quantity that can
    be part of an objective function for optimization.
    """

    def __init__(self, parameters, function):
        """
        When constructing a Target, you should supply a python set in
        which the elements are the Parameter objects upon which this
        Target depends.

        reset_function can be None or something callable.
        """
        if type(parameters) is not set:
            raise ValueError("Argument to Target.__init__ must have type 'set'")
        for param in parameters:
            if type(param) is not Parameter:
                raise ValueError("In parameters argument to Target.__init__, " \
                                     "each element must have type Parameter.")
        if not callable(function):
            raise ValueError("function must be callable.")
        self._function = function

        self._parameters = parameters

    def __repr__(self):
        return "Target(parameters=" + self._parameters.__repr__() + \
            "; evaluate=" + self._function.__repr__() + ")"

    @property
    def parameters(self):
        """
        Return a python set containing all the Parameter objects from
        which the value of this Target can be calculated. There should
        be no arguments.
        """
        return self._parameters

    def evaluate(self):
        """
        Return a float, the scalar value that can be part of an
        objective function. Doing this generally requires running a
        physics code. There should be no arguments.
        """
        return self._function()

class Identity:
    """
    Identity is a minimal object for displaying the behaviors of the
    Target class. An Identity instance has one Parameter named x, and
    one Target named target. The target just returns the value of the
    Parameter.
    """
    def __init__(self):
        """
        There are 2 protected variables, _x and _target, both
        publically accessible through the @property decorator.
        """
        self._x = Parameter(name="x for Identity " + str(hex(id(self))))
        self._target = Target({self._x}, self.eval)

    @property
    def x(self):
        """
        x is read-only.
        """
        return self._x

    @property
    def target(self):
        """
        target is read-only.
        """
        return self._target

    def eval(self):
        """
        Just return the value of the sole Parameter.
        """
        return self._x.val
