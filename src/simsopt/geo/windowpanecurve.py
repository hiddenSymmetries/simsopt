import jax.numpy as jnp
from math import pi, sin, cos
import numpy as np
from .curve import JaxCurve
from simsopt._core.optimizable import Optimizable

__all__ = ['WindowpaneCurveXYZFourier']

def shift_pure( v, xyz ):
    for ii in range(0,3):
        v = v.at[:,ii].add(xyz[ii])
    return v

def rotate_pure( v, ypr ):        
    yaw = ypr[0]
    pitch = ypr[1]
    roll = ypr[2]

    Myaw = jnp.asarray(
        [[jnp.cos(yaw), -jnp.sin(yaw), 0],
        [jnp.sin(yaw), jnp.cos(yaw), 0],
        [0, 0, 1]]
    )
    Mpitch = jnp.asarray(
        [[jnp.cos(pitch), 0, jnp.sin(pitch)],
        [0, 1, 0],
        [-jnp.sin(pitch), 0, jnp.cos(pitch)]]
    )
    Mroll = jnp.asarray(
        [[1, 0, 0],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll), jnp.cos(roll)]]
    )

    return v @ Myaw @ Mpitch @ Mroll

def centercurve_pure(dofs, quadpoints, order):
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    fmn = dofs[6:]

    k = len(fmn)//3
    coeffs = [fmn[:k], fmn[k:(2*k)], fmn[(2*k):]]
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    for i in range(0,3):
        for j in range(0, order):
            gamma = gamma.at[:, i].add(coeffs[i][2 * j] * jnp.sin(2 * pi * (j+1) * points))
            gamma = gamma.at[:, i].add(coeffs[i][2 * j + 1] * jnp.cos(2 * pi * (j+1) * points))

    return shift_pure( rotate_pure( gamma, ypr ), xyz )
    

class WindowpaneCurveXYZFourier( JaxCurve ):
    """
    WindowpaneCurveXYZFourier is a translated and rotated 
    JaxCurveXYZFourier Curve.
    """
    def __init__(self, quadpoints, order, dofs=None ):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        self.order = order
        pure = lambda dofs, points: centercurve_pure(dofs, points, self.order)

        self.coefficients = [np.zeros((3,)), np.zeros((3,)), np.zeros((2*order,)), np.zeros((2*order,)), np.zeros((2*order,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=WindowpaneCurveXYZFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=WindowpaneCurveXYZFourier.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 3 + 3*(2*self.order)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)
    
    def set_dofs_impl(self, dofs):
        self.coefficients[0][:] = dofs[0:3]
        self.coefficients[1][:] = dofs[3:6]

        counter = 6
        for i in range(0,3):
            for j in range(0, self.order):
                self.coefficients[i+2][2*j] = dofs[counter]
                counter += 1
                self.coefficients[i+2][2*j+1] = dofs[counter]
                counter += 1

        

    def _make_names(self):
        xyc_name = ['x0', 'y0', 'z0']
        ypr_name = ['yaw', 'pitch', 'roll']
        dofs_name = []
        for c in ['x', 'y', 'z']:
            for j in range(0, self.order):
                dofs_name += [f'{c}s({j+1})', f'{c}c({j+1})']
        return xyc_name + ypr_name + dofs_name

