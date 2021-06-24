# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

import functools

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

    def __instancecheck__(self, other):
        return isinstance(other, self._callable)
