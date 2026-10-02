from functools import wraps

__all__ = ['derivative_dec']


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

    @wraps(func)
    def _derivative_dec(self, *args, partials=False, **kwargs):
        if partials:
            return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)(self)
    return _derivative_dec
