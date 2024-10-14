import numpy as np
import time
import jax.numpy as jnp

import simsoptpp as sopp
from .curve import Curve, RotatedCurve, JaxCurve
from .._core.derivative import Derivative

__all__ = ['CurvePlanarFourier', 'JaxCurvePlanarFourier']


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

def jaxplanarcurve_pure(dofs, quadpoints, order):
    coeffs = dofs[:2 * order + 1]
    q = dofs[2 * order + 1: 2 * order + 5]
    q_norm = q / jnp.linalg.norm(q)
    center = dofs[2 * order + 5:]
    phi = 2 * np.pi * quadpoints  # points is an angle in [0, 1]
    jrange = jnp.arange(1, order + 1)[:, None]
    jphi = jrange * phi[None, :]
    r_curve = coeffs[0] + jnp.sum(coeffs[1:order + 1, None] * jnp.cos(jphi) \
        + coeffs[order + 1: 2 * order + 1, None] * jnp.sin(jphi), axis=0)

    x_curve_in_plane = r_curve * jnp.cos(phi)
    y_curve_in_plane = r_curve * jnp.sin(phi)
    return jnp.transpose(jnp.vstack((jnp.vstack(((1.0 - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3])) * x_curve_in_plane \
        + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * y_curve_in_plane \
        + center[0], (1.0 - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3])) * y_curve_in_plane \
        + 2 * (q_norm[0] * q_norm[3] + q_norm[1] * q_norm[2]) * x_curve_in_plane \
        + center[1])), 2 * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * x_curve_in_plane \
        + 2 * (q_norm[0] * q_norm[1] + q_norm[2] * q_norm[3]) * y_curve_in_plane \
        + center[2])))

class JaxCurvePlanarFourier(JaxCurve):

    """
    A Python+Jax implementation of the CurvePlanarFourier class.  There is
    actually no reason why one should use this over the C++ implementation in
    :mod:`simsoptpp`, but the point of this class is to illustrate how jax can be used
    to define a geometric object class and calculate all the derivatives (both
    with respect to dofs and with respect to the angle :math:`\theta`) automatically.

    [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, 
    q_0, q_i, q_j, q_k, 
    x_{\text{center}}, y_{\text{center}}, z_{\text{center}}]
    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        pure = lambda dofs, points: jaxplanarcurve_pure(dofs, points, order)
        self.order = order
        self.dof_list = np.zeros(2 * order + 1 + 4 + 3)
        if dofs is None:
            super().__init__(quadpoints, pure, x0=self.dof_list,
                             external_dof_setter=JaxCurvePlanarFourier.set_dofs_impl)
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=JaxCurvePlanarFourier.set_dofs_impl)

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return (2 * self.order + 1 + 4 + 3)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return self.dof_list

    def set_dofs_impl(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.dof_list = dofs


# class JaxCurvePlanarFourier(JaxCurve):

#     """
#     A Python+Jax implementation of the CurvePlanarFourier class.  There is
#     actually no reason why one should use this over the C++ implementation in
#     :mod:`simsoptpp`, but the point of this class is to illustrate how jax can be used
#     to define a geometric object class and calculate all the derivatives (both
#     with respect to dofs and with respect to the angle :math:`\theta`) automatically.

#     [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, 
#     q_0, q_i, q_j, q_k, 
#     x_{\text{center}}, y_{\text{center}}, z_{\text{center}}]
#     """

#     def __init__(self, quadpoints, order, dofs=None):
#         if isinstance(quadpoints, int):
#             quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
#         pure = lambda dofs, points: jaxplanarcurve_pure(dofs, points, order)
#         self.order = order
#         self.coefficients = [np.zeros((2 * order + 1,)), np.zeros((4,)), np.zeros((3,))]
#         if dofs is None:
#             super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
#                              external_dof_setter=JaxCurvePlanarFourier.set_dofs_impl)
#         else:
#             super().__init__(quadpoints, pure, dofs=dofs,
#                              external_dof_setter=JaxCurvePlanarFourier.set_dofs_impl)
#         # self.passive_current_jax = jit(lambda dofs: self.passive_current_pure(dofs, points))
#         # self.passive_current_impl_jax = jit(lambda dofs, p: self.passive_current_pure(dofs, p))
#         # self.dpassive_current_by_dcoeff_jax = jit(jacfwd(self.passive_current_jax))
#         # self.dpassive_current_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.passive_current_jax, x)[1](v)[0])

#     # def passive_current_pure(self, dofs, points):
#     #     # To get the I_i passive current, need to compute (L^{-1} * psi)_i = Linv_{ij}psi_j
#     #     # so just need the jth column of Linv here. 
#     #     normal = self.normal
#     #     psi = 
       

#     def num_dofs(self):
#         """
#         This function returns the number of dofs associated to this object.
#         """
#         return (2 * self.order + 1 + 4 + 3)

#     def get_dofs(self):
#         """
#         This function returns the dofs associated to this object.
#         """
#         return np.concatenate(self.coefficients)

#     def set_dofs_impl(self, dofs):
#         """
#         This function sets the dofs associated to this object.
#         """
#         # self.coefficients = dofs
#         for j in range(2 * self.order + 1):
#             self.coefficients[0][j] = dofs[j]
#         self.coefficients[1][:] = dofs[2 * self.order + 1:2 * self.order + 5]
#         self.coefficients[2][:] = dofs[2 * self.order + 5:]