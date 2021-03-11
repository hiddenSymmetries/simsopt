import numpy as np

class Surface():
    def __init__(self, quadpoints_phi, quadpoints_theta):
        self.dependencies = []
        self.phi_grid, self.theta_grid = np.meshgrid(quadpoints_phi, quadpoints_theta)
        self.phi_grid = self.phi_grid.T    
        self.theta_grid = self.theta_grid.T 
    
    def print_metadata(self):
        print("Surface area is {:.8f}".format(np.abs(self.surface_area()) ) )
        print("*************************") 

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
