import numpy as np

import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

__all__ = ['SurfaceXYZFourier']


class SurfaceXYZFourier(sopp.SurfaceXYZFourier, Surface):
    r"""`SurfaceXYZFourier` is a surface that is represented in Cartesian
    coordinates using the following Fourier series:

    .. math::
        \hat x(\phi,\theta) &= \sum_{m=0}^{m_\text{pol}} \sum_{n=-n_{\text{tor}}}^{n_{tor}} [
              x_{c,m,n} \cos(m \theta - n_\text{ fp } n \phi)
            + x_{s,m,n} \sin(m \theta - n_\text{ fp } n \phi)]\\
        \hat y(\phi,\theta) &= \sum_{m=0}^{m_\text{pol}} \sum_{n=-n_\text{tor}}^{n_\text{tor}} [
              y_{c,m,n} \cos(m \theta - n_\text{fp} n \phi)
            + y_{s,m,n} \sin(m \theta - n_\text{fp} n \phi)]\\
        z(\phi,\theta) &= \sum_{m=0}^{m_\text{pol}} \sum_{n=-n_\text{tor}}^{n_\text{tor}} [
              z_{c,m,n} \cos(m \theta - n_\text{fp}n \phi)
            + z_{s,m,n} \sin(m \theta - n_\text{fp}n \phi)]

    where

    .. math::
        x &= \hat x \cos(\phi) - \hat y \sin(\phi)\\
        y &= \hat x \sin(\phi) + \hat y \cos(\phi)

    Note that for :math:`m=0` we skip the :math:`n<0` term for the cos
    terms, and the :math:`n \leq 0` for the sin terms.

    When enforcing stellarator symmetry, we set the

    .. math::
        x_{s,*,*}, ~y_{c,*,*}, \text{and} ~z_{c,*,*}

    terms to zero.

    For more information about the arguments `quadpoints_phi``, and
    ``quadpoints_theta``, see the general documentation on :ref:`surfaces`.
    Instead of supplying the quadrature point arrays along :math:`\phi` and
    :math:`\theta` directions, one could also specify the number of
    quadrature points for :math:`\phi` and :math:`\theta` using the
    class method :py:meth:`~simsopt.geo.surface.Surface.from_nphi_ntheta`.

    Args:
        nfp: The number of field periods.
        stellsym: Whether the surface is stellarator-symmetric, i.e.
          symmetry under rotation by :math:`\pi` about the x-axis.
        mpol: Maximum poloidal mode number included.
        ntor: Maximum toroidal mode number included, divided by ``nfp``.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0,
                 quadpoints_phi=None, quadpoints_theta=None,
                 dofs=None):

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        sopp.SurfaceXYZFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                        quadpoints_phi, quadpoints_theta)
        self.xc[0, ntor] = 1.0
        self.xc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        if dofs is None:
            Surface.__init__(self, x0=self.get_dofs(), names=self._make_names(),
                             external_dof_setter=SurfaceXYZFourier.set_dofs_impl)
        else:
            Surface.__init__(self, dofs=dofs,
                             external_dof_setter=SurfaceXYZFourier.set_dofs_impl)

    def _make_names(self):
        """
        Form a list of names of the ``xc``, ``ys``, ``zs``, ``xs``,
        ``yc``, or ``zc`` array elements. The order of these four arrays
        here must match the order in ``set_dofs_impl()`` and ``get_dofs()``
        in ``src/simsoptpp/surfacexyzfourier.h``.
        """
        if self.stellsym:
            names = self._make_names_helper('xc', True) \
                + self._make_names_helper('ys', False) \
                + self._make_names_helper('zs', False)
        else:
            names = self._make_names_helper('xc', True) \
                + self._make_names_helper('xs', False) \
                + self._make_names_helper('yc', True) \
                + self._make_names_helper('ys', False) \
                + self._make_names_helper('zc', True) \
                + self._make_names_helper('zs', False)
        return names

    def _make_names_helper(self, prefix, include0):
        """
        Helper function for `_make_names` method. Forms array of coefficients
        for :math:'m = [0, m_{pol}]' and :math:'n = [-n_{tor}, n_{tor}]'. If :math:'m = 0', only
        positive values of :math:'n' are used. If it is a cosine term, the :math:'(0,0)' term is included.

        Args:
            prefix: The prefix for the name of the coefficients.
            include0: Whether to include the (0,0) term.
        """
        if include0:
            names = [prefix + "(0,0)"]
        else:
            names = []

        names += [prefix + '(0,' + str(n) + ')' for n in range(1, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names

    def get_dofs(self):
        """
        Return the dofs associated to this surface.
        """
        return np.asarray(sopp.SurfaceXYZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        Set the dofs associated to this surface.
        """
        self.local_full_x = dofs

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.
        """
        ntor = self.ntor
        mpol = self.mpol
        surf = SurfaceRZFourier(nfp=self.nfp,
                                stellsym=self.stellsym,
                                mpol=mpol,
                                ntor=ntor,
                                quadpoints_phi=self.quadpoints_phi,
                                quadpoints_theta=self.quadpoints_theta)

        gamma = np.zeros((surf.quadpoints_phi.size, surf.quadpoints_theta.size, 3))
        for idx in range(gamma.shape[0]):
            gamma[idx, :, :] = self.cross_section(surf.quadpoints_phi[idx])

        surf.least_squares_fit(gamma)
        return surf

    def extend_via_normal(self, distance):
        """
        Extend the surface in the normal direction by a uniform distance.

        Args:
            distance: The distance to extend the surface.
        """
        self._extend_via_normal_for_nonuniform_phi(distance)


    return_fn_map = {'area': sopp.SurfaceXYZFourier.area,
                     'volume': sopp.SurfaceXYZFourier.volume,
                     'aspect-ratio': Surface.aspect_ratio}
