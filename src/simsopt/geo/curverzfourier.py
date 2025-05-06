import numpy as np

import simsoptpp as sopp
from .curve import Curve

__all__ = ['CurveRZFourier']


class CurveRZFourier(sopp.CurveRZFourier, Curve):
    r"""
    ``CurveRZFourier`` is a curve that is represented in cylindrical
    coordinates using the following Fourier series:

    .. math::
       r(\phi) &= \sum_{m=0}^{\text{order}} r_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(n_{\text{fp}} m \phi) \\
       z(\phi) &= \sum_{m=0}^{\text{order}} z_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} z_{s,m}\sin(n_{\text{fp}} m \phi)

    If ``stellsym = True``, then the :math:`\sin` terms for :math:`r` and the :math:`\cos` terms for :math:`z` are zero.

    For the ``stellsym = False`` case, the dofs are stored in the order

    .. math::
       [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, z_{c,0},....]

    or in the ``stellsym = True`` case they are stored

    .. math::
       [r_{c,0},...,r_{c,order},z_{s,1},...,z_{s,order}]
    """

    def __init__(self, quadpoints, order, nfp, stellsym, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1./nfp, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurveRZFourier.__init__(self, quadpoints, order, nfp, stellsym)
        if dofs is None:
            Curve.__init__(self, x0=self.get_dofs(), names=self._make_names(order, stellsym),
                           external_dof_setter=CurveRZFourier.set_dofs_impl)
        else:
            Curve.__init__(self, external_dof_setter=CurveRZFourier.set_dofs_impl,
                           dofs=dofs)

    def _make_names(self, order, stellsym):
        r_names = [f'rc({i})' for i in range(0, order + 1)]
        z_names = [f'zs({i})' for i in range(1, order + 1)]
        if not stellsym:
            r_names += [f'rs({i})' for i in range(1, order + 1)]
            z_names = [f'zc({i})' for i in range(0, order + 1)] + z_names
        return r_names + z_names

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurveRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurveRZFourier.set_dofs(self, dofs)
