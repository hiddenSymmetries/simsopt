import numpy as np
from .graph_optimizable import Optimizable


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

    Then ``obj.dJ()`` will return a ``Derivative`` object containing a dictionary
    
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
    and then passing this vector to `Intermediate1.vjp` and `Intermediate2.vjp`, which returns 

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

    This `Derivative` can then be used to obtain partial derivatives or the full gradient of `J`, via

    .. code-block::

        dJ = obj.dJ()
        dJ_by_dinA = dJ(inA) # derivative of Objective w.r.t. to OptimA
        dJ_by_dinB = dJ(inB) # derivative of Objective w.r.t. to OptimB
        gradJ = dJ(obj) # gradient of Objective

    
    """

    def __init__(self, data={}):
        self.data = data

    def __add__(self, other):
        x = self.data
        y = other.data
        z = x.copy()
        for k in y.keys():
            if k in z:
                z[k] += y[k]
            else:
                z[k] = y[k]

        return Derivative(z)

    def __iadd__(self, other):
        x = self.data
        y = other.data
        for k in y.keys():
            if k in x:
                x[k] += y[k]
            else:
                x[k] = y[k]
        return self

    def __mul__(self, other):
        assert isinstance(other, float)
        x = self.data.copy()
        for k in x.keys():
            x[k] *= other
        return Derivative(x)

    def __rmul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        x = self.data.copy()
        for k in x.keys():
            x[k] *= other
        return Derivative(x)

    def __call__(self, optim: Optimizable):
        """
        Get the derivative with respect to all DOFs that `optim` depends on.
        """
        deps = optim.ancestors + [optim]
        derivs = []
        for k in deps:
            if k in self.data.keys() and np.any(k.dofs_free_status):
                derivs.append(self.data[k][k.dofs_free_status])
        return np.concatenate(derivs)
