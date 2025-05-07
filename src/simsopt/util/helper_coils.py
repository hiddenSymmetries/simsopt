"""
This file contains a helper function for coils. 
Currently only a simple function that returns a simsopt.field.Coil object that represents a current along the z-axis. 

Useful for adding a toroidal field, for example to perturb a stellarator equilibrium. 
"""

__all__ = ['current_along_z',]

from simsopt.geo import CurveAlongZ
from simsopt.field import Coil, Current

__all__ = ['current_along_z',]


def current_along_z(current, quadpoints=100, x0=0., y0=0., zscale=10, coil_dofs_fixed=True):
    """
    Returns a Coil object that represents a current along the z-axis. 
    Add this to your a coilset before calling simsopt.BiotSavart to 
    add a toroidal field. 

    The dofs of the coil are only the current value, unles you set coil_dofs_fixed=False.

    Args:
        current: the current in Amperes
        quadpoints: number of grid points/resolution along the curve;
        x0: (default 0) the x-coordinate .
        y0: (default 0) the y-coordinate.  
        zscale: (default 10) points are closer together at z=0, and spread apart using a zscale*tan(pi*(gamma-.5)) scaling.
        coil_dofs_fixed: (default True) unless you are doing strange things, you don't want to move the coil from the axis. 
    """
    return Coil(CurveAlongZ(quadpoints, x0, y0, zscale, coil_dofs_fixed=coil_dofs_fixed), Current(current))
