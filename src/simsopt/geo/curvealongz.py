import jax.numpy as jnp
from math import pi
import numpy as np
from .curve import JaxCurve

__all__ = ['CurveAlongZ']


def jaxcurvealongz_pure(dofs, quadpoints, zscale):
    """
    Pure function for the CurveAlongZ, returns the points along the curve given the degrees of freedom and the quadpoints. 

    Args:
        dofs: degrees of freedom of the curve, must be len(3)
        quadpoints: points in [0,1] on which to evaluate the curve
    """
    lenquadpoints = len(quadpoints)
    x = dofs[0]*jnp.ones(lenquadpoints)
    y = dofs[1]*jnp.ones(lenquadpoints)
    halfstep = .5/(lenquadpoints-1)
    z = jnp.tan(((quadpoints + halfstep) - .5)*pi)*zscale
    gamma = jnp.stack((x, y, z), axis=1)
    return gamma


class CurveAlongZ(JaxCurve):
    r'''
    Straight vertical curve, parallel to the z-axis. 

    Useful for quickly generating a toroidal field, comparing to tokamak equilibria
    where an axisymmetric 1/R field is present, and testing. 
    
    Degrees of freedom are the x, y coordinates of the vertical coil. 
    Displacing this from [0,0] can give a 1/1 perturbation if you feel like it. 


    Args:
        quadpoints: number of grid points/resolution along the curve;
        x0: the x-coordinate of the vertical coil
        y0: the y-coordinate of the vertical coil
        zscale: points are closer together at z=0, and spread apart using a zscale*tan(pi*(gamma-.5)) scaling.
    '''

    def __init__(self, quadpoints, x0=0., y0=0., zscale=10):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        
        self.set_dofs_impl([x0, y0])
        
        self.zscale = zscale
        pure = lambda dofs, points: jaxcurvealongz_pure(
            dofs, points, self.zscale)
        
        super().__init__(quadpoints, pure, external_dof_setter=self.make_dof_names)
        # unless you are doing strange things, you don't want to move the coil so we
        # set the dofs fixed. 
        self.fix_all()

    def num_dofs(self):
        return 2

    def get_dofs(self):
        return self.xy

    def set_dofs_impl(self, dofs):
#        if len(dofs != 2):
#            raise ValueError(f"The number of degrees of freedom of a CurveAlongZ are only 2, not {len(dofs)}")
        self.xy = np.array(dofs)
        
    
    def make_dof_names(self):
        return ['x0', 'y0', 'zscale']

