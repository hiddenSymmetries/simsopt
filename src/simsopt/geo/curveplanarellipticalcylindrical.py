import jax.numpy as jnp
import numpy as np
from .curve import JaxCurve

__all__ = ['CurvePlanarEllipticalCylindrical', 'create_equally_spaced_cylindrical_curves']


def r_ellipse(a, b, l):
    return a*b / jnp.sqrt((b*jnp.cos(2.*np.pi*l))**2 + (a*jnp.sin(2.*np.pi*l))**2)


def xyz_cyl(a, b, l):
    """
    given curve dofs return a curve centered at the origin
    """
    # l is equally spaced from 0 to 1 - 2pi*l is an equally spaced array in theta from 0 to 2pi
    nl = l.size
    out = jnp.zeros((nl, 3))
    out = out.at[:, 0].set(r_ellipse(a, b, l)*jnp.cos(2.*np.pi*l))  # x
    out = out.at[:, 1].set(0.0)  # y
    out = out.at[:, 2].set(r_ellipse(a, b, l)*jnp.sin(2.*np.pi*l))  # z
    return out


def rotations(curve, a, b, alpha_r, alpha_phi, alpha_z, dr):
    #rotates curves around r,phi,z coordinates

    z_rot = jnp.asarray(
        [[jnp.cos(alpha_z), -jnp.sin(alpha_z), 0],
         [jnp.sin(alpha_z), jnp.cos(alpha_z), 0],
         [0, 0, 1]])

    y_rot = jnp.asarray(
        [[jnp.cos(alpha_phi), 0, jnp.sin(alpha_phi)],
         [0, 1, 0],
         [-jnp.sin(alpha_phi), 0, jnp.cos(alpha_phi)]])

    x_rot = jnp.asarray(
        [[1, 0, 0],
         [0, jnp.cos(alpha_r), -jnp.sin(alpha_r)],
         [0, jnp.sin(alpha_r), jnp.cos(alpha_r)]])

    out = curve

    # I think this is doing the z rotation, so use b
    out = out.at[:, 2].set(out[:, 2] + b)  # move bottom of coil up to z=0
    out = out @ y_rot @ x_rot @ z_rot  # apply rotation in this frame
    out = out.at[:, 0].set(out[:, 0] + dr)  # dr is the major radius shift
    out = out.at[:, 2].set(out[:, 2] - b)  # move bottom of coils back down

    return out


def convert_to_cyl(a):
    #convert to cylindrical - I think this still works for elliptical coils
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(jnp.sqrt(a[:, 0]**2 + a[:, 1]**2))  # R = X^2 + Y^2
    out = out.at[:, 1].set(jnp.arctan2(a[:, 1], a[:, 0]))  # phi = arctan(Y/X)
    out = out.at[:, 2].set(a[:, 2])  # z = z
    return out


def cylindrical_shift(a, dphi, dz):
    #shifting in phi, z
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(a[:, 0])
    out = out.at[:, 1].set(a[:, 1]+dphi)
    out = out.at[:, 2].set(a[:, 2]+dz)
    return out


def cyl_to_cart(a):
    #cylindrical to cartesian
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(a[:, 0] * jnp.cos(a[:, 1]))  # x = R*cos(phi)
    out = out.at[:, 1].set(a[:, 0] * jnp.sin(a[:, 1]))  # y = R*sin(phi)
    out = out.at[:, 2].set(a[:, 2])  # z = z
    return out


def gamma_pure(dofs, points, a, b):
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    g1 = xyz_cyl(a, b, points)  # generate elliptical parameterization
    g2 = rotations(g1, a, b, ypr[0], ypr[1], ypr[2], xyz[0])  # rotate and translate in R
    g3 = convert_to_cyl(g2)  # convert to R, phi, Z coordinates
    g4 = cylindrical_shift(g3, xyz[1], xyz[2])  # shift in phi and Z
    final_gamma = cyl_to_cart(g4)  # convert back to cartesian
    return final_gamma


class CurvePlanarEllipticalCylindrical(JaxCurve):
    """
    CurvePlanarEllipticalCylindrical is a translated and rotated Curve in r, phi, and z coordinates.
    It is parametrized by an ellipse with semi-major axis a and semi-minor axis b, where a lies along 
    the R vector and b along the Z vector. This curve is then translated and rotated in cylindrical coordinates
    to a major radius R0, toroidal angle phi, and vertical position Z0. The curve is then rotated by angles
    r_rotation, phi_rotation, and z_rotation in the r, phi, and z directions, respectively, about the minimum 
    Z location of the curve.
    """

    def __init__(self, quadpoints, a, b, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        def pure(dofs, points): return gamma_pure(dofs, points, a, b)

        self.coefficients = [np.zeros((3,)), np.zeros((3,))]
        self.a = a
        self.b = b
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=CurvePlanarEllipticalCylindrical.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=CurvePlanarEllipticalCylindrical.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 3

    def get_dofs(self):
        return np.concatenate([self.coefficients[0], self.coefficients[1]])

    def set_dofs_impl(self, dofs):
        """
        This function sets the degrees of freedom (DoFs) for this object.
         """
        self.coefficients[0][:] = dofs[0:3]  # R0, phi, Z0
        self.coefficients[1][:] = dofs[3:6]  # theta, constant_phi,

    def _make_names(self):
        """
        Generates names for the degrees of freedom (DoFs) for this object.
        """
        rtpz_name = ['R0', 'phi', 'Z0']
        angle_name = ['r_rotation', 'phi_rotation', 'z_rotation']
        return rtpz_name + angle_name


def create_equally_spaced_cylindrical_curves(ncurves, nfp, stellsym, R0, a, b, numquadpoints=32):
    """
    Create a list of equally spaced cylindrical curves in the toroidal coordinate.
    Parameters:
    ncurves (int): Number of curves to create (per half field period).
    nfp (int): Number of field periods.
    stellsym (bool): Stellarator symmetry flag.
    R0 (float): Major radius of the curve centers.
    a (float): Semi-major axis of the elliptical cross-section. 
    b (float): Semi-minor axis of the elliptical cross-section. 
    numquadpoints (int, optional): Number of quadrature points. 
    Returns:
    list: A list of CurvePlanarEllipticalCylindrical objects representing the cylindrical curves.
    """
    curves = []
    for i in range(ncurves):
        curve = CurvePlanarEllipticalCylindrical(numquadpoints, a, b)
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ncurves)
        curve.set('R0', R0)
        curve.set('phi', angle)
        curve.set('Z0', 0)
        curve.set('r_rotation', 0)
        curve.set('phi_rotation', 0)
        curve.set('z_rotation', 0)
        curves.append(curve)
    return curves
