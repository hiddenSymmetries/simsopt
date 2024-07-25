import numpy as np

import simsoptpp as sopp
from .curve import Curve, RotatedCurve
from .._core.derivative import Derivative

__all__ = ['CurvePlanarFourier', 'PSCCurve']


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

class PSCCurve(CurvePlanarFourier):
    def __init__(self, quadpoints, order, nfp, stellsym, index, npsc, dofs=None):
        CurvePlanarFourier.__init__(self, quadpoints, order, nfp, stellsym, dofs=dofs)
        self._index = index
        names = self.local_dof_names
        if len(names) > 0: 
            for i in range(2 * self.order + 1):
                self.fix(names[i])
            self.fix(names[2 * self.order + 5])
            self.fix(names[2 * self.order + 6])
            self.fix(names[2 * self.order + 7])
        self.npsc = npsc
        
    def dkappa_dcoef_vjp(self, v_current, dpsi, index):
        dofs = self.get_dofs()  
        dofs_unnormalized = dofs[2 * self.order + 1:2 * self.order + 5]
        normalization = np.sqrt(np.sum(dofs_unnormalized ** 2))
        dofs = dofs_unnormalized / normalization  # normalize the quaternion
        w = dofs[0]
        x = dofs[1]
        y = dofs[2]
        z = dofs[3]
        dalpha_dw = (2 * x * (-2 * (x ** 2 + y ** 2) + 1)) / \
            (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
        dalpha_dx = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
            (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
        dalpha_dy = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
            (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
        dalpha_dz = (2 * y * (1 - 2 * (x ** 2 + y ** 2))) / \
            (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
        ddelta_dw = y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dx = -z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dy = w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dz = -x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
    
        # So far this is dalpha with respect to the normalized variables 
        # so need to convert to the original dof derivatives
        dnormalization = np.zeros((4, 4))
        for i in range(4):
            eye = np.zeros(4)
            eye[i] = 1.0
            dnormalization[:, i] = eye / normalization - dofs_unnormalized * dofs_unnormalized[i] / normalization ** 3
        dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz]) @ dnormalization
        ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz]) @ dnormalization

        # dalpha_total = self._psc_array.dpsi_full[index] * dalpha
        # ddelta_total = self._psc_array.dpsi_full[index + self.npsc] * ddelta
        
        # dalpha_total = dpsi[self._index] * dalpha
        # ddelta_total = dpsi[self._index + self.npsc] * ddelta
        dalpha_total = dpsi[index] * dalpha
        ddelta_total = dpsi[index + self.npsc] * ddelta
        
        dalpha = np.hstack((np.zeros(2 * self.order + 1), dalpha_total))
        dalpha = np.hstack((dalpha, np.zeros(3)))
        ddelta = np.hstack((np.zeros(2 * self.order + 1), ddelta_total))
        ddelta = np.hstack((ddelta, np.zeros(3)))
        deriv = dalpha + ddelta
        # print(deriv, v_current)
        # exit()
        return deriv * v_current[0]