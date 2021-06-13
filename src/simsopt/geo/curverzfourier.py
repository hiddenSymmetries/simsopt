import numpy as np

import simsgeopp as sgpp
from .curve import Curve


class CurveRZFourier(sgpp.CurveRZFourier, Curve):
    r"""
    CurveRZFourier is a curve that is represented in cylindrical
       coordinates using the following Fourier series:

        .. math::
           r(\phi) &= \sum_{m=0}^{\text{order}} x_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} x_{s,m}\sin(n_{\text{fp}} m \phi) \\
           z(\phi) &= \sum_{m=0}^{\text{order}} z_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} z_{s,m}\sin(n_{\text{fp}} m \phi)

       If ``stellsym = True``, then the :math:`\sin` terms for :math:`r` and the :math:`\cos` terms for :math:`z` are zero.

       For the ``stellsym = False`` case, the dofs are stored in the order

        .. math::
           [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, z_{c,0},....]

       or in the ``stellsym = True`` case they are stored

        .. math::
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
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sgpp.CurveRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        sgpp.CurveRZFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()
