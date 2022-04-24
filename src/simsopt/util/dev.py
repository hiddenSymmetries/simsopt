# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

import functools
import warnings

from monty.dev import requires


class SimsoptRequires(requires):
    def __call__(self, _callable):
        """
        :param _callable: Callable function or class.
        """
        self._callable = _callable

        @functools.wraps(_callable)
        def decorated(*args, **kwargs):
            if not self.condition:
                raise RuntimeError(self.message)
            return _callable(*args, **kwargs)

        return decorated

    @property
    def __class(self):
        return self._callable

    def __instancecheck__(self, other):
        return isinstance(other, self._callable)


def deprecated(replacement=None, message=None, category=FutureWarning):
    """
    Decorator to mark classes or functions as deprecated,
    with a possible replacement.
    Credits: monty.dev package from Materials Virtual Lab

    Args:
        replacement (callable): A replacement class or method.
        message (str): A warning message to be displayed.
        category (Warning): Choose the category of the warning to issue. Defaults
            to FutureWarning. Another choice can be DeprecationWarning. NOte that
            FutureWarning is meant for end users and is always shown unless silenced.
            DeprecationWarning is meant for developers and is never shown unless
            python is run in developmental mode or the filter is changed. Make
            the choice accordingly.

    Returns:
        Original function, but with a warning to use the updated class.
    """

    def wrap(old):
        @functools.wraps(old)
        def wrapped(*args, **kwargs):
            msg = "%s is deprecated" % old.__name__
            if replacement is not None:
                if isinstance(replacement, property):
                    r = replacement.fget
                elif isinstance(replacement, (classmethod, staticmethod)):
                    r = replacement.__func__
                else:
                    r = replacement
                msg += "; use %s in %s instead." % (r.__name__, r.__module__)
            if message is not None:
                msg += "\n" + message
            warnings.warn(msg, category=category, stacklevel=2)
            return old(*args, **kwargs)

        return wrapped

    return wrap
