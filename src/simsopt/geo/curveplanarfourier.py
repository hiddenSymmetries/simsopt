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
    def __init__(self, quadpoints, order, nfp, stellsym, psc_array, index, dofs=None):
        CurvePlanarFourier.__init__(self, quadpoints, order, nfp, stellsym, dofs=None)
        self._psc_array = psc_array
        self._index = index
        names = self.local_dof_names
        if len(names) > 0: 
            for i in range(2 * self.order + 1):
                self.fix(names[i])
            self.fix(names[2 * self.order + 5])
            self.fix(names[2 * self.order + 6])
            self.fix(names[2 * self.order + 7])
        self.npsc = self._psc_array.num_psc
        
    def psc_current_contribution_vjp(self, v_current):
        indices = np.hstack((self._index, self._index + self.npsc))
        Linv_partial = self._psc_array.L_inv[self._index % self.npsc, self._index % self.npsc]
        # Linv = np.vstack((Linv_partial, Linv_partial)).T
        # psi_deriv = self._psc_array.dpsi_full[indices]
        # print(v_curre.shape)
        # print(Linv, psi_deriv, v_current)
        # print('deriv info = ', -Linv, psi_deriv, v_current, np.ravel((-Linv * psi_deriv) @ v_current))
        # print(Linv.shape, indices, psi_deriv.shape, v_current.shape, ((-Linv * psi_deriv) @ v_current).shape)
        # print(np.ravel((-Linv * psi_deriv) @ v_current).shape)
        # print(v_current, Linv_partial, self._psc_array.L_inv.shape, self._index)
        return Derivative({self: np.ravel(-Linv_partial * self.dkappa_dcoef_vjp(v_current)) })  #  np.ravel((-Linv * psi_deriv) @ v_current).tolist()  # 
    
    def dkappa_dcoef_vjp(self, v_current):
        dofs = self.get_dofs()  # should already be only the orientation dofs
        
        # For the rotated coils, need to compute dalpha_prime / dcoef
        if len(dofs) == 0:
            dofs = self.curve.get_dofs()

        dofs = dofs[2 * self.order + 1:2 * self.order + 5]  # don't need the coordinate variables
        # print('dofs = ', dofs)
        normalization = np.sqrt(np.sum(dofs ** 2))
        dofs = dofs / normalization  # normalize the quaternion
        w = dofs[0]
        x = dofs[1]
        y = dofs[2]
        z = dofs[3]
        # alphas = 2.0 * np.arcsin(np.sqrt(dofs[:, 1] ** 2 + dofs[:, 3] ** 2))
        # deltas = 2.0 * np.arccos(np.sqrt(dofs[:, 0] ** 2 + dofs[:, 1] ** 2))
        # dofs[3] = cos(alpha / 2.0) * cos(delta / 2.0)
        # dofs[4] = sin(alpha / 2.0) * cos(delta / 2.0)
        # dofs[5] = cos(alpha / 2.0) * sin(delta / 2.0)
        # dofs[6] = -sin(alpha / 2.0) * sin(delta / 2.0)
        dalpha_dw = (2 * x * (2 * (x ** 2 + y ** 2) - 1)) / \
            (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
        dalpha_dx = (w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
            (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
        dalpha_dy = (z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
            (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
        dalpha_dz = (2 * y * (2 * (x ** 2 + y ** 2) - 1)) / \
            (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
        ddelta_dw = -y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dx = z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dy = -w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dz = x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
    
        dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz] / normalization)
        ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz] / normalization)
        
        # if len(dofs) == 0:
        dalpha = self._psc_array.dpsi_daa[self._index] * dalpha + self._psc_array.dpsi_dad[self._index] * ddelta
        ddelta = self._psc_array.dpsi_ddd[self._index] * ddelta + self._psc_array.dpsi_dda[self._index] * dalpha
             
        dalpha = np.hstack((np.zeros(2 * self.order + 1), dalpha))
        dalpha = np.hstack((dalpha, np.zeros(3)))
        ddelta = np.hstack((np.zeros(2 * self.order + 1), ddelta))
        ddelta = np.hstack((ddelta, np.zeros(3)))
        deriv = dalpha + ddelta
        # print(deriv.shape)
        # exit()
        return deriv * v_current[0]  # Derivative({self: deriv})
    
# class RotatedPSCCurve(RotatedCurve, PSCCurve):
#     """
#     RotatedCurve inherits from the Curve base class.  It takes an
#     input a Curve, rotates it about the ``z`` axis by a toroidal angle
#     ``phi``, and optionally completes a reflection when ``flip=True``.
#     """

#     def __init__(self, curve, phi, flip):
#         RotatedCurve.__init__(self, curve, phi, flip)
#         PSCCurve.__init__(self, curve.quadpoints, curve.order, curve.nfp, curve.stellsym, curve._psc_array, curve._index, curve.get_dofs())
#         print(self, self.order, self.curve)
#         #self.curve = curve
#         #self.order = curve.order
        