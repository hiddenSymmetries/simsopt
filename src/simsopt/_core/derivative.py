import numpy as np
import numbers
import collections

__all__ = ['Derivative']


class OptimizableDefaultDict(collections.defaultdict):
    """
    Custom defaultdict that automatically returns a numpy array of zeros of
    size equal to the number of free dofs when the key wasn't found.
    """

    def __init__(self, d):
        super().__init__(None, d)

    def __missing__(self, key):
        from .optimizable import Optimizable  # Import here to avoid circular import
        assert isinstance(key, Optimizable)
        self[key] = value = np.zeros((key.local_full_dof_size, ))
        return value


def copy_numpy_dict(d):
    res = OptimizableDefaultDict({})
    for k, v in d.items():
        res[k] = v.copy()
    return res


class Derivative():

    """
    This class stores the derivative of a scalar output wrt to the individual
    ``Optimizable`` classes that are required to compute this output.

    The idea of this class is as follows:

    Consider a situation

    .. code-block::

        inA = OptimA()
        inB = OptimB()
        inter1 = Intermediate1(inA, inB)
        inter2 = Intermediate2(inA, inB)
        obj = Objective(inter1, inter2)

    Then ``obj.dJ(partials=True)`` will return a ``Derivative`` object containing a dictionary

    .. code-block::

        {
            inA : dobj/dinA,
            inB : dobj/dinB,
        }

    with

    .. code-block::

        dobj/dinA = dobj/dinter1 * dinter1/dinA + dobj/dinter2 * dinter2/dinA
        dobj/dinB = dobj/dinter1 * dinter1/dinB + dobj/dinter2 * dinter2/dinB

    SIMSOPT computes these derivatives by first computing ``dobj/dinter1`` and ``dobj/dinter2``
    and then passing this vector to ``Intermediate1.vjp`` and ``Intermediate2.vjp``, which returns

    .. code-block::

        {
            inA: dobj/dinter1 * dinter1/dinA
            inB: dobj/dinter1 * dinter1/dinA
        }

    and 

    .. code-block::

        {
            inA: dobj/dinter2 * dinter2/dinA
            inB: dobj/dinter2 * dinter2/dinA
        }

    respectively. Due to the overloaded ``__add__`` and ``__iadd__`` functions adding the ``Derivative`` objects than results in the desired

    .. code-block::

        {
            inA: dobj/dinter1 * dinter1/dinA + dobj/dinter2 * dinter2/dinA
            inB: dobj/dinter1 * dinter1/dinA + dobj/dinter2 * dinter2/dinA
        }

    This ``Derivative`` can then be used to obtain partial derivatives or the full gradient of ``J``, via

    .. code-block::

        dJ = obj.dJ(partials=True)
        dJ_by_dinA = dJ(inA) # derivative of Objective w.r.t. to OptimA
        dJ_by_dinB = dJ(inB) # derivative of Objective w.r.t. to OptimB
        gradJ = dJ(obj) # gradient of Objective

    For the common case in which you just want the gradient of
    ``obj.J`` and do not need the individual partial derivatives, the
    argument ``partials=True`` can be omitted in ``obj.dJ()``. In this
    case, ``obj.dJ()`` directly returns the gradient rather than
    returning the ``Derivative`` object, acting as a shorthand for
    ``obj.dJ(partials=True)(obj)``. This behavior is implemented with
    the decorator :obj:`derivative_dec`.
    """

    def __init__(self, data=OptimizableDefaultDict({})):
        self.data = OptimizableDefaultDict(data)

    def __add__(self, other):
        x = self.data
        y = other.data
        z = copy_numpy_dict(x)
        for k in y:
            if k in z:
                z[k] += y[k]
            else:
                z[k] = y[k].copy()
        return Derivative(z)

    def __sub__(self, other):
        x = self.data
        y = other.data
        z = copy_numpy_dict(x)
        for k, yk in y.items():
            if k in z:
                z[k] -= yk
            else:
                z[k] = -yk
        return Derivative(z)

    def __iadd__(self, other):
        x = self.data
        y = other.data
        for k, yk in y.items():
            if k in x:
                x[k] += yk
            else:
                x[k] = yk.copy()
        return self

    def __isub__(self, other):
        x = self.data
        y = other.data
        for k, yk in y.items():
            if k in x:
                x[k] -= yk
            else:
                x[k] = -yk
        return self

    def __mul__(self, other):
        assert isinstance(other, numbers.Number)
        x = copy_numpy_dict(self.data)
        for k in x:
            x[k] *= other
        return Derivative(x)

    def __rmul__(self, other):
        assert isinstance(other, numbers.Number)
        x = copy_numpy_dict(self.data)
        for k in x:
            x[k] *= other
        return Derivative(x)

    def __call__(self, optim):
        """
        Get the derivative with respect to all DOFs that ``optim`` depends on.

        Args:
            optim: An Optimizable object
        """
        from .optimizable import Optimizable  # Import here to avoid circular import
        assert isinstance(optim, Optimizable)
        deps = optim.ancestors + [optim]
        derivs = []
        for k in deps:
            if np.any(k.dofs_free_status):
                derivs.append(self.data[k][k.local_dofs_free_status])
        return np.concatenate(derivs)

    # https://stackoverflow.com/questions/11624955/avoiding-python-sum-default-start-arg-behavior
    def __radd__(self, other):
        # This allows sum() to work (the default start value is zero)
        if other == 0:
            return self
        return self.__add__(other)


def derivative_dec(func):
    """
    This decorator is applied to functions of Optimizable objects that
    return a derivative, typically named ``dJ()``. This allows
    ``obj.dJ()`` to provide a shorthand for the full gradient,
    equivalent to ``obj.dJ(partials=True)(obj)``. If
    ``partials=True``, the underlying :obj:`Derivative` object will be
    returned, so partial derivatives can be accessed and combined to
    assemble gradients.
    """

    def _derivative_dec(self, *args, partials=False, **kwargs):
        if partials:
            return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)(self)
    return _derivative_dec
