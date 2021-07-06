import numpy as np
import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier


class SurfaceXYZTensorFourier(sopp.SurfaceXYZTensorFourier, Surface):

    r"""
    `SurfaceXYZTensorFourier` is a surface that is represented in cartesian
    coordinates using the following Fourier series:

    .. math::
        \hat x(\theta, \phi) &= \sum_{i=0}^{2m_\text{pol}} \sum_{j=0}^{2n_\text{tor}} x_{ij} w_i(\theta)v_j(\phi)\\
        \hat y(\theta, \phi) &= \sum_{i=0}^{2m_\text{pol}} \sum_{j=0}^{2n_\text{tor}} y_{ij} w_i(\theta)v_j(\phi)\\
        x(\phi, \theta) &= \hat x(\phi, \theta)  \cos(\phi) - \hat y(\phi, \theta)  \sin(\phi)\\
        y(\phi, \theta) &= \hat x(\phi, \theta)  \sin(\phi) + \hat y(\phi, \theta)  \cos(\phi)\\
        z(\theta, \phi) &= \sum_{i=0}^{2m_\text{pol}} \sum_{j=0}^{2n_\text{tor}} z_{ij} w_i(\theta)v_j(\phi)

    where the basis functions :math:`{v_j}` are given by

    .. math::
        \{1, \cos(1\,\mathrm{nfp}\,\phi), \ldots, \cos(n_\text{tor}\,\mathrm{nfp}\,\phi), \sin(1\,\mathrm{nfp}\,\phi), \ldots, \sin(n_\text{tor}\,\mathrm{nfp}\,\phi)\}

    and :math:`{w_i}` are given by

    .. math::
        \{1, \cos(1\theta), \ldots, \cos(m_\text{pol}\theta), \sin(1\theta), \ldots, \sin(m_\text{pol}\theta)\}

    When `stellsym=True` the sums above change as follows:

    .. math::
        \hat x(\theta, \phi) &= \sum_{i=0}^{m_\text{pol}} \sum_{j=0}^{n_\text{tor}} x_{ij} w_i(\theta)v_j(\phi) + \sum_{i=m_\text{pol}+1}^{2m_\text{pol}} \sum_{j=n_\text{tor}+1}^{2n_\text{tor}} x_{ij} w_i(\theta)v_j(\phi)\\
        \hat y(\theta, \phi) &= \sum_{i=0}^{m_\text{pol}} \sum_{j=n_\text{tor}+1}^{2n_\text{tor}} y_{ij} w_i(\theta)v_j(\phi) + \sum_{i=m_\text{pol}+1}^{2m_\text{pol}} \sum_{j=0}^{n_\text{tor}} y_{ij} w_i(\theta)v_j(\phi)\\\\
        z(\theta, \phi) &= \sum_{i=0}^{m_\text{pol}} \sum_{j=n_\text{tor}+1}^{2n_\text{tor}} z_{ij} w_i(\theta)v_j(\phi) + \sum_{i=m_\text{pol}+1}^{2m_\text{pol}} \sum_{j=0}^{n_\text{tor}} z_{ij} w_i(\theta)v_j(\phi)

    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=1, clamped_dims=[False, False, False],
                 quadpoints_phi=32, quadpoints_theta=32):
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
            quadpoints_theta = list(quadpoints_theta)
        sopp.SurfaceXYZTensorFourier.__init__(self, mpol, ntor, nfp, stellsym, clamped_dims,
                                              quadpoints_phi, quadpoints_theta)
        self.x[0, 0] = 1.0
        self.x[1, 0] = 0.1
        self.z[mpol+1, 0] = 0.1
        Surface.__init__(self)

    def get_dofs(self):
        """
        Return the dofs associated to this surface.
        """
        return np.asarray(sopp.SurfaceXYZTensorFourier.get_dofs(self))

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.
        """
        surf = SurfaceRZFourier(self.mpol, self.ntor, self.nfp, self.stellsym, self.quadpoints_phi, self.quadpoints_theta)
        gamma = np.zeros((surf.quadpoints_phi.size, surf.quadpoints_theta.size, 3))
        for idx in range(gamma.shape[0]):
            gamma[idx, :, :] = self.cross_section(surf.quadpoints_phi[idx]*2*np.pi)
        surf.least_squares_fit(self.gamma())
        return surf

    def set_dofs(self, dofs):
        """
        Set the dofs associated to this surface.
        """
        sopp.SurfaceXYZTensorFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()

    def get_stellsym_mask(self):
        """
        In the case of stellarator symmetry, some of the information is
        redundant, since the coordinates at (-phi, -theta) are the same (up
        to sign changes) to those at (phi, theta).
        The point of this function is to identify those angles phi and theta
        that we can ignore. This is difficult to do in general, so we focus on
        the following three common cases below:

            phis = np.linspace(0, 1/self.nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

        or

            phis = np.linspace(0, 1/self.nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 0.5, 2*mpol, endpoint=False)

        or

            phis = np.linspace(0, 1/(2*self.nfp), ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

        This function could be extended to be aware of rotational symmetry as
        well.  So far we assume that that redundancy was removed already (hence
        the phis only go to 1/nfp or 1/(2*nfp)).

        """

        phis = self.quadpoints_phi
        thetas = self.quadpoints_theta
        nphi = len(phis)
        ntheta = len(thetas)
        mask = np.ones((nphi, ntheta), dtype=bool)
        if not self.stellsym:
            return mask
        ntor = self.ntor
        mpol = self.mpol

        def npsame(a, b):
            return a.shape == b.shape and np.allclose(a, b)

        if npsame(phis, np.linspace(0, 1/self.nfp, 2*ntor+1, endpoint=False)) and \
                npsame(thetas, np.linspace(0, 1, 2*mpol+1, endpoint=False)):
            mask[:, mpol+1:] = False
            mask[ntor+1:, 0] = False
        if npsame(phis, np.linspace(0, 1/self.nfp, 2*ntor+1, endpoint=False)) and \
                npsame(thetas, np.linspace(0, 0.5, mpol+1, endpoint=False)):
            mask[ntor+1:, 0] = False
        if npsame(phis, np.linspace(0, 1/(2*self.nfp), ntor+1, endpoint=False)) and \
                npsame(thetas, np.linspace(0, 1, 2*mpol+1, endpoint=False)):
            mask[0, mpol+1:] = False
        return mask
