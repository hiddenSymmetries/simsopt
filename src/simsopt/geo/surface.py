import numpy as np
from simsopt.core.optimizable import Optimizable

class Surface(Optimizable):
    """
    Surface is a base class for various representations of toroidal
    surfaces in simsopt.
    """

    def __init__(self):
        self.dependencies = []
    
    def plot(self, ax=None, show=True, plot_normal=False, plot_derivative=False, scalars=None, wireframe=True):
        gamma = self.gamma()
        
        from mayavi import mlab
        mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], scalars=scalars)
        if wireframe:
            mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], representation='wireframe', color = (0,0,0))
        

        if plot_derivative:
            dg1 = 0.05 * self.gammadash1()
            dg2 = 0.05 * self.gammadash2()
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], dg1[:,:,0], dg1[:,:,1], dg1[:,:,2])
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], dg2[:,:,0], dg2[:,:,1], dg2[:,:,2])
        if plot_normal:
            n = 0.005 * self.normal()
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], n[:,:,0], n[:,:,1], n[:,:,2])
        if show:
            mlab.show()

    def darea(self):
        return self.darea_by_dcoeff()

    def dvolume(self):
        return self.dvolume_by_dcoeff()

    def __repr__(self):
        return "Surface " + str(hex(id(self)))

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.  All subclasses should implement this abstract
        method.
        """
        raise NotImplementedError
