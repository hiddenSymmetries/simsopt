import numpy as np

import simsoptpp as sopp
from .curve import Curve, RotatedCurve
from .._core.derivative import Derivative

__all__ = ['CurvePlanarFourier', 'PSCCurve']  # , 'RotatedPSCCurve']


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
        CurvePlanarFourier.__init__(self, quadpoints, order, nfp, stellsym, dofs=dofs)
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
        indices = np.arange(self._index, self.npsc * self._psc_array.symmetry, self.npsc)
        Linv_partial = self._psc_array.L_inv[self._index, self._index]  # % self.npsc, self._index % self.npsc]
        print(self._psc_array.I[self._index], 
              -self._psc_array.L_inv[self._index, :] @ self._psc_array.psi_total * 1e7,
              -self._psc_array.L_inv[self._index, indices] @ self._psc_array.psi_total[indices] * 1e7)
        return Derivative({self: np.ravel(-Linv_partial * self.dkappa_dcoef_vjp(v_current)) })  #  np.ravel((-Linv * psi_deriv) @ v_current).tolist()  # 
    
    def dkappa_dcoef_vjp(self, v_current, index):
        dofs = self.get_dofs()  # should already be only the orientation dofs
        
        # For the rotated coils, need to compute dalpha_prime / dcoef
        # if len(dofs) == 0:
        #     dofs = self.curve.get_dofs()
            # rotated curves need the orientation dofs of the rotated curve

        dofs_unnormalized = dofs[2 * self.order + 1:2 * self.order + 5]  # don't need the coordinate variables
        normalization = np.sqrt(np.sum(dofs_unnormalized ** 2))
        # print('dofs = ', dofs, normalization)
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
    
        # dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz])
        # ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz])

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
        
        # dalpha_total = self._psc_array.dpsi_daa[self._index] * dalpha + self._psc_array.dpsi_dad[self._index] * ddelta
        # ddelta_total = self._psc_array.dpsi_dda[self._index] * dalpha + self._psc_array.dpsi_ddd[self._index] * ddelta
        
        # indices = np.arange(self._index, 2 * self.npsc * self._psc_array.symmetry, 2 * self.npsc)
        # indices2 = np.arange(self._index + self.npsc, 2 * self.npsc * self._psc_array.symmetry, 2 * self.npsc)
        # dalpha_total = self._psc_array.dpsi_full[indices] * dalpha
        # ddelta_total = self._psc_array.dpsi_full[indices2] * ddelta
        
        dalpha_total = self._psc_array.dpsi[self._index] * dalpha
        ddelta_total = self._psc_array.dpsi[self._index + self.npsc] * ddelta
        
        # if len(dofs) == 0:
        # print(self._index)
        # dalpha_total = np.zeros(dalpha.shape)
        # ddelta_total = np.zeros(ddelta.shape)
        # for i in range(self._psc_array.symmetry):
        #     index = self._index + self.npsc * i
        #     dalpha_total += self._psc_array.dpsi_daa[index] * dalpha + self._psc_array.dpsi_dad[index] * ddelta
        #     ddelta_total += self._psc_array.dpsi_ddd[index] * ddelta + self._psc_array.dpsi_dda[index] * dalpha
        
        dalpha = np.hstack((np.zeros(2 * self.order + 1), dalpha_total))
        dalpha = np.hstack((dalpha, np.zeros(3)))
        ddelta = np.hstack((np.zeros(2 * self.order + 1), ddelta_total))
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
#         # sopp.Curve.__init__(self, curve.quadpoints)
#         # print(curve)
#         # print(curve.quadpoints, curve.order, curve.nfp, curve.stellsym, curve._psc_array, curve._index, curve.get_dofs(), phi, flip)
#         # print(self)
#         # PSCCurve.__init__(self, curve.quadpoints, curve.order, curve.nfp, curve.stellsym, curve._psc_array, curve._index, curve.get_dofs())
#         RotatedCurve.__init__(self, curve, phi, flip)
#         exit()

#         print(self, self.order, self.curve)
#         # rotcurve.order = base_curves[i].order
#         # rotcurve._psc_array = base_curves[i]._psc_array
#         # rotcurve._index = base_curves[i]._index + base_curves[i].npsc * q
#         # rotcurve.psc_current_contribution_vjp = base_curves[i].psc_current_contribution_vjp
#         # rotcurve.dkappa_dcoef_vjp = base_curves[i].dkappa_dcoef_vjp
#         #self.curve = curve
#         #self.order = curve.order
        