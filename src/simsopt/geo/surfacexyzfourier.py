import numpy as np
import simsgeopp as sgpp
from .surface import Surface


class SurfaceXYZFourier(sgpp.SurfaceXYZFourier, Surface):
    def __init__(self, mpol, ntor, nfp, stellsym, quadpoints_phi, quadpoints_theta):
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
            quadpoints_theta = list(quadpoints_theta)
        sgpp.SurfaceXYZFourier.__init__(self, mpol, ntor, nfp, stellsym, quadpoints_phi, quadpoints_theta)
        Surface.__init__(self, quadpoints_phi, quadpoints_theta)

    def get_dofs(self):
        return np.asarray(sgpp.SurfaceXYZFourier.get_dofs(self))
