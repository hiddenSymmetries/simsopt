import jax.numpy as jnp
from math import pi, sin, cos
import numpy as np
from .curve import JaxCurve
from simsopt._core.optimizable import Optimizable

__all__ = ['OrientedCurveXYZFourier']

def shift_pure( v, xyz ):
    """Apply translation in cartesian coordinates
    
    Args:
     - v: array to translate. Should have size Nx3.
     - xyz: translation vector. Should have size 3.

    Returns:
     - v+xyz: translated array, size Nx3
    """
    for ii in range(0,3):
        v = v.at[:,ii].add(xyz[ii])
    return v

def rotate_pure( v, ypr ):
    """Apply rotation around x, y, and z axis.
    
    Args:
     - v: set of points to rotate. Should have size Nx3.
     - ypr: rotation angles.
            ypr[0] describes the rotation around the z-axis.
            ypr[1] describes the rotation around the y-axis.
            ypr[2] describes the rotation around the x-axis.

    Returns:
    - v: Rotated set of points
    """ 
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
    """Construct curve centered at the origin
    
    Args:
     - dofs: Set of degrees of freedom
     - quadpoints: Quadrature points. Array of size N, with float values between 0 and 1.
     - order: Maximum Fourier mode number.

    Returns:
     - gamma: Curve that has been translated and rotated to the desired position.
    """
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    fmn = dofs[6:]

    k = len(fmn)//3
    coeffs = [fmn[:k], fmn[k:(2*k)], fmn[(2*k):]]
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    for i in range(0,3):
        for j in range(0, order):
            gamma = gamma.at[:, i].add(coeffs[i][2 * j    ] * jnp.sin(2 * pi * (j+1) * points))
            gamma = gamma.at[:, i].add(coeffs[i][2 * j + 1] * jnp.cos(2 * pi * (j+1) * points))

    return shift_pure( rotate_pure( gamma, ypr ), xyz )
    

class OrientedCurveXYZFourier( JaxCurve ):
    """
    OrientedCurveXYZFourier is a translated and rotated Curve.

    Args:
     - quadpoints: Integer (number of quadrature points), or array of size N, with float values between 0 and 1.
     - order: Maximum mode order
     - dofs (optionnal): Degrees of freedom
    """
    def __init__(self, quadpoints, order, dofs=None ):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        self.order = order
        pure = lambda dofs, points: centercurve_pure(dofs, points, self.order)

        self.coefficients = [np.zeros((3,)), np.zeros((3,)), np.zeros((2*order,)), np.zeros((2*order,)), np.zeros((2*order,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=OrientedCurveXYZFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=OrientedCurveXYZFourier.set_dofs_impl,
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
    
    @classmethod
    def from_curvexyzfourier(cls, xyzcurve):
        oriented_curve = cls(xyzcurve.quadpoints, xyzcurve.order)

        for dname in xyzcurve.local_full_dof_names:
            if dname in ['xc(0)', 'yc(0)', 'zc(0)']:
                continue
            oriented_curve.set(dname, xyzcurve.get(dname))

        oriented_curve.set('x0', xyzcurve.get('xc(0)'))
        oriented_curve.set('y0', xyzcurve.get('yc(0)'))
        oriented_curve.set('z0', xyzcurve.get('zc(0)'))

        return oriented_curve

