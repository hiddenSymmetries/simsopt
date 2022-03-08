import numpy as np
import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier


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

    For more information about the arguments ``nphi``, ``ntheta``,
    ``range``, ``quadpoints_phi``, and ``quadpoints_theta``, see the
    general documentation on :ref:`surfaces`.

    Args:
        nfp: The number of field periods.
        stellsym: Whether the surface is stellarator-symmetric, i.e.
          symmetry under rotation by :math:`\pi` about the x-axis.
        mpol: Maximum poloidal mode number included.
        ntor: Maximum toroidal mode number included, divided by ``nfp``.
        nphi: Number of grid points :math:`\phi_j` in the toroidal angle :math:`\phi`.
        ntheta: Number of grid points :math:`\theta_j` in the toroidal angle :math:`\theta`.
        range: Toroidal extent of the :math:`\phi` grid.
          Set to ``"full torus"`` (or equivalently ``SurfaceXYZFourier.RANGE_FULL_TORUS``)
          to generate points up to 1 (with no point at 1).
          Set to ``"field period"`` (or equivalently ``SurfaceXYZFourier.RANGE_FIELD_PERIOD``)
          to generate points up to :math:`1/n_{fp}` (with no point at :math:`1/n_{fp}`).
          Set to ``"half period"`` (or equivalently ``SurfaceXYZFourier.RANGE_HALF_PERIOD``)
          to generate points up to :math:`1/(2 n_{fp})`, with all grid points shifted by half
          of the grid spacing in order to provide spectral convergence of integrals.
          If ``quadpoints_phi`` is specified, ``range`` is irrelevant.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0,
                 nphi=None, ntheta=None, range="full torus",
                 quadpoints_phi=None, quadpoints_theta=None):

        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(nfp=nfp,
                                                                  nphi=nphi, ntheta=ntheta, range=range,
                                                                  quadpoints_phi=quadpoints_phi,
                                                                  quadpoints_theta=quadpoints_theta)
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

    return_fn_map = {'area': sopp.SurfaceXYZFourier.area,
                     'volume': sopp.SurfaceXYZFourier.volume,
                     'aspect-ratio': Surface.aspect_ratio}
