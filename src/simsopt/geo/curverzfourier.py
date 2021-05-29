import numpy as np

import simsgeopp as sgpp
from .curve import Curve


class CurveRZFourier(sgpp.CurveRZFourier, Curve):
    r"""
    CurveRZFourier is a curve that is represented in cylindrical
       coordinates using the following Fourier series:

           r(phi) = \sum_{n=0}^{order} x_{c,n}cos(n*nfp*phi) + \sum_{n=1}^order x_{s,n}sin(n*nfp*phi)
           z(phi) = \sum_{n=0}^{order} z_{c,n}cos(n*nfp*phi) + \sum_{n=1}^order z_{s,n}sin(n*nfp*phi)

       If stellsym = true, then the sin terms for r and the cos terms for z are zero.

       For the stellsym = False case, the dofs are stored in the order

           [r_{c,0},...,r_{c,order},r_{s,1},...,r_{s,order},z_{c,0},....]

       or in the stellsym = true case they are stored

           [r_{c,0},...,r_{c,order},z_{s,1},...,z_{s,order}]
    """

    def __init__(self, quadpoints, order, nfp, stellsym):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1./nfp, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sgpp.CurveRZFourier.__init__(self, quadpoints, order, nfp, stellsym)
        Curve.__init__(self)

    def get_dofs(self):
        return np.asarray(sgpp.CurveRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        sgpp.CurveRZFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()
