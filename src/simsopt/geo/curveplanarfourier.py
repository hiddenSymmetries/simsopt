import numpy as np
import jax.numpy as jnp

import simsoptpp as sopp
from .curve import Curve, JaxCurve

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

    def center(self, gamma, gammadash):
        # Compute the centroid of the curve
        arclength = jnp.linalg.norm(gammadash, axis=-1)
        barycenter = jnp.sum(gamma * arclength[:, None], axis=0) / jnp.sum(arclength)
        return barycenter


def jaxplanarcurve_pure(dofs, quadpoints, order):
    coeffs = dofs[:2 * order + 1]
    q = dofs[2 * order + 1: 2 * order + 5]
    norm_q = jnp.linalg.norm(q)
    q_norm = jnp.where(norm_q < 1e-8,
                       q / (norm_q + 1e-8),  # safe division when norm is small
                       q / norm_q)  # this shouldn't happen if the quaternion dofs are properly initialized
    center = dofs[2 * order + 5:]
    phi = 2 * np.pi * quadpoints  # points is an angle in [0, 1]
    jrange = jnp.arange(1, order + 1)[:, None]
    jphi = jrange * phi[None, :]
    r_curve = coeffs[0] + jnp.sum(coeffs[1:order + 1, None] * jnp.cos(jphi)
                                  + coeffs[order + 1: 2 * order + 1, None] * jnp.sin(jphi), axis=0)

    x_curve_in_plane = r_curve * jnp.cos(phi)
    y_curve_in_plane = r_curve * jnp.sin(phi)
    return jnp.transpose(jnp.vstack((jnp.vstack(((1.0 - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3])) * x_curve_in_plane
                                                 + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * y_curve_in_plane
                                                 + center[0], (1.0 - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3])) * y_curve_in_plane
                                                 + 2 * (q_norm[0] * q_norm[3] + q_norm[1] * q_norm[2]) * x_curve_in_plane
                                                 + center[1])), 2 * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * x_curve_in_plane
                                     + 2 * (q_norm[0] * q_norm[1] + q_norm[2] * q_norm[3]) * y_curve_in_plane
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

        def pure(dofs, points): return jaxplanarcurve_pure(dofs, points, order)
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
        return np.array(self.dof_list)

    def set_dofs_impl(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.dof_list = np.array(dofs)

    def set_quadpoints(self, quadpoints):
        self.quadpoints = quadpoints
