import numpy as np
import simsgeopp as sgpp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier


class SurfaceXYZTensorFourier(sgpp.SurfaceXYZTensorFourier, Surface):

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=1,
                 quadpoints_phi=32, quadpoints_theta=32):
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
            quadpoints_theta = list(quadpoints_theta)
        sgpp.SurfaceXYZTensorFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                              quadpoints_phi, quadpoints_theta)
        self.x[0, 0] = 1.0
        self.x[1, 0] = 0.1
        self.z[mpol+1, 0] = 0.1
        Surface.__init__(self)

    def get_dofs(self):
        return np.asarray(sgpp.SurfaceXYZTensorFourier.get_dofs(self))

    def to_RZFourier(self):
        surf = SurfaceRZFourier(self.mpol, self.ntor, self.nfp, self.stellsym,
                                self.quadpoints_phi, self.quadpoints_theta)
        surf.least_squares_fit(self.gamma())
        return surf

    def set_dofs(self, dofs):
        sgpp.SurfaceXYZTensorFourier.set_dofs(self, dofs)
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
            mask[0, :mpol+1] = False
        return mask
