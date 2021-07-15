from .graph_optimizable import Optimizable


class Derivative():

    """
    This class stores the derivative of a scalar output wrt to the individual
    Optimizable classes that are required to compute this output.
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
        assert isinstance(other, float)
        x = self.data.copy()
        for k in x.keys():
            x[k] *= other
        return Derivative(x)

    def __call__(self, key: Optimizable):
        """
        Get the derivative with respect to the Optimizable object key
        """
        return self.data[key]
