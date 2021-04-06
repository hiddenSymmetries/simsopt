import numpy as np
import simsgeopp as sgpp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier


class SurfaceXYZFourier(sgpp.SurfaceXYZFourier, Surface):
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
        surf = SurfaceRZFourier(self.mpol, self.ntor, self.nfp, self.stellsym, self.quadpoints_phi, self.quadpoints_theta)
        surf.least_squares_fit(self.gamma())
        return surf


