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
                 quadpoints_phi=None, quadpoints_theta=None):

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        sopp.SurfaceXYZFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                        quadpoints_phi, quadpoints_theta)
        self.xc[0, ntor] = 1.0
        self.xc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        Surface.__init__(self, x0=self.get_dofs(),
                         external_dof_setter=SurfaceXYZFourier.set_dofs_impl)

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
            gamma[idx, :, :] = self.cross_section(surf.quadpoints_phi[idx]*2*np.pi)

        surf.least_squares_fit(gamma)
        return surf

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["mpol"] = self.mpol
        d["ntor"] = self.ntor
        d["stellsym"] = self.stellsym
        return d

    @classmethod
    def from_dict(cls, d):
        surf = cls(nfp=d["nfp"], stellsym=d["stellsym"],
                   mpol=d["mpol"], ntor=d["ntor"],
                   quadpoints_phi=d["quadpoints_phi"],
                   quadpoints_theta=d["quadpoints_theta"])
        surf.set_dofs(d["x0"])
        return surf

    return_fn_map = {'area': sopp.SurfaceXYZFourier.area,
                     'volume': sopp.SurfaceXYZFourier.volume,
                     'aspect-ratio': Surface.aspect_ratio}
