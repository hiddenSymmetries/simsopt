from .curve import JaxCurve, Curve
from math import pi
from jax.ops import index, index_add
import jax.numpy as jnp
import numpy as np
import simsgeopp as sgpp


class CurveRZFourier(sgpp.CurveRZFourier, Curve):

    def __init__(self, quadpoints, order, nfp, stellsym):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1./nfp, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        Curve.__init__(self)
        sgpp.CurveRZFourier.__init__(self, quadpoints, order, nfp, stellsym)

    def get_dofs(self):
        return np.asarray(sgpp.CurveRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        sgpp.CurveRZFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()
