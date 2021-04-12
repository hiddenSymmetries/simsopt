import numpy as np
import simsgeopp as sgpp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier


class SurfaceXYZFourier(sgpp.SurfaceXYZFourier, Surface):
    r"""
       SurfaceXYZFourier is a surface that is represented in cartesian
       coordinates using the following Fourier series:

           \hat x(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
                 x_{c,m,n} \cos(m \theta - n nfp \phi)
               + x_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

           \hat y(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
                 y_{c,m,n} \cos(m \theta - n nfp \phi)
               + y_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

           x = \hat x * \cos(\phi) - \hat y * \sin(\phi)
           y = \hat x * \sin(\phi) + \hat y * \cos(\phi)

           z(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
               z_{c,m,n} \cos(m \theta - n nfp \phi)
               + z_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

       Note that for m=0 we skip the n<0 term for the cos terms, and the n<=0
       for the sin terms.

       When enforcing stellarator symmetry, we set the

           x_{s,*,*}, y_{c,*,*} and z_{c,*,*}

       terms to zero.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0, quadpoints_phi=32, quadpoints_theta=32):
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
            quadpoints_theta = list(quadpoints_theta)
        sgpp.SurfaceXYZFourier.__init__(self, mpol, ntor, nfp, stellsym, quadpoints_phi, quadpoints_theta)
        self.xc[0, ntor] = 1.0
        self.xc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        Surface.__init__(self)

    def get_dofs(self):
        return np.asarray(sgpp.SurfaceXYZFourier.get_dofs(self))

    def to_RZFourier(self):
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
            gamma[idx, :, :] = self.cross_section(surf.quadpoints_phi[idx]*2*np.pi, theta_resolution=surf.quadpoints_theta.size)
        
        surf.least_squares_fit(gamma)
        return surf

    def set_dofs(self, dofs):
        sgpp.SurfaceXYZFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()
