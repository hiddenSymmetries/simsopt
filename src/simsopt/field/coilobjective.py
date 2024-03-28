from deprecated import deprecated

from jax import grad
import jax.numpy as jnp

from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp

__all__ = ['CurrentPenalty']


def current_penalty_pure(I, threshold):
    return jnp.maximum(abs(I) - threshold, 0)**2

class CurrentPenalty(Optimizable):
    """
    A :obj:`CurrentPenalty` can be used to penalize
    large currents in coils.
    """
    def __init__(self, current, threshold=0):
        self.current = current
        self.threshold = threshold
        super().__init__(depends_on=[current])
        self.J_jax = lambda I: current_penalty_pure(I, self.threshold)
        self.this_grad = lambda I: grad(self.J_jax, argnums=0)(I)

    def J(self):
        return self.J_jax(self.current.get_value())

    @derivative_dec
    def dJ(self):
        grad0 = self.this_grad(self.current.get_value())
        return self.current.vjp(grad0)
