from deprecated import deprecated

from jax import grad
import jax.numpy as jnp

from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp

__all__ = ['CurrentPenalty']


@jit
def current_penalty_pure(I, p, threshold):
    return jnp.maximum(abs(I) - threshold, 0)**p

class CurrentPenalty(Optimizable):
    """
    A :obj:`CurrentPenalty` can be used to penalize
    large currents in coils.
    """
    def __init__(self, current, p, threshold=0):
        self.current = current
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[current])
        self.J_jax = jit(lambda c, p, t: current_penalty_pure(c, p, t))
        self.this_grad = jit(lambda c, p, t: grad(self.J_jax, argnums=0)(c, p, t))

    def J(self):
        return self.J_jax(self.current.get_value(), self.p, self.threshold)

    @derivative_dec
    def dJ(self):
        grad0 = self.this_grad(self.current.get_value(), self.p, self.threshold)
        return self.current.vjp(grad0)
