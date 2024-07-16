import numpy as np

import simsoptpp as sopp
from .curve import Curve

__all__ = ['CurvePlanarFourier']


class CurvePlanarFourier(sopp.CurvePlanarFourier, Curve):
    r"""
    ``CurvePlanarFourier`` is a curve that is restricted to lie in a plane. The
    shape of the curve within the plane is represented by a Fourier series in
    polar coordinates. The resulting planar curve is then rotated in three
    dimensions using a quaternion, and finally a translation is applied. The
    Fourier series in polar coordinates is

    .. math::

       r(\phi) = \sum_{m=0}^{\text{order}} r_{c,m}\cos(m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(m \phi).

    The rotation quaternion is

    .. math::

       \bf{q} &= [q_0,q_i,q_j,q_k]

       &= [\cos(\theta / 2), \hat{x}\sin(\theta / 2), \hat{y}\sin(\theta / 2), \hat{z}\sin(\theta / 2)]

    where :math:`\theta` is the counterclockwise rotation angle about a unit axis
    :math:`(\hat{x},\hat{y},\hat{z})`. Details of the quaternion rotation can be
    found for example in pages 575-576 of
    https://www.cis.upenn.edu/~cis5150/ws-book-Ib.pdf.


    A quaternion is used for rotation rather than other methods for rotation to
    prevent gimbal locking during optimization. The quaternion is normalized
    before being applied to prevent scaling of the curve. The dofs themselves are not normalized. This
    results in a redundancy in the optimization, where several different sets of 
    dofs may correspond to the same normalized quaternion. Normalizing the dofs 
    directly would create a dependence between the quaternion dofs, which may cause 
    issues during optimization.

    The dofs are stored in the order

    .. math::
       [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, q_0, q_i, q_j, q_k, x_{\text{center}}, y_{\text{center}}, z_{\text{center}}]


    """

    def __init__(self, quadpoints, order, nfp, stellsym, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1., quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurvePlanarFourier.__init__(self, quadpoints, order, nfp, stellsym)
        if dofs is None:
            Curve.__init__(self, external_dof_setter=CurvePlanarFourier.set_dofs_impl,
                           x0=self.get_dofs())
        else:
            Curve.__init__(self, external_dof_setter=CurvePlanarFourier.set_dofs_impl,
                           dofs=dofs)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurvePlanarFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurvePlanarFourier.set_dofs(self, dofs)
