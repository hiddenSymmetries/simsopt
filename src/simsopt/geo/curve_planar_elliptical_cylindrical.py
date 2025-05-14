import jax.numpy as jnp
import numpy as np
from .curve import JaxCurve

__all__ = ['CurvePlanarEllipticalCylindrical', 'create_equally_spaced_cylindrical_curves',
           'xyz_cyl', 'rotations', 'convert_to_cyl', 'cylindrical_shift', 'cyl_to_cart',
           'gamma_pure']


def r_ellipse(a, b, l):
    """
    Returns the cylindrical coordinate of an ellipse with semi-major axis a and semi-minor axis b
    at a given angle l.
    
    Args:
        a: float
            semi-major axis of the ellipse
        b: float
            semi-minor axis of the ellipse
        l: np.array, shape (nquadpoints,)
            2*pi*l is an equally spaced array in theta from 0 to 2pi

    Returns:
        out: np.array, shape (nquadpoints,)
            Cylindrical coordinate of the ellipse at a given angle l.
    """
    return a*b / jnp.sqrt((b*jnp.cos(2.*np.pi*l))**2 + (a*jnp.sin(2.*np.pi*l))**2)


def xyz_cyl(a, b, l):
    """
    Given curve dofs, return the Cartesian curve coordinates centered at the origin

    Args:
        a: float
            semi-major axis of the ellipse
        b: float
            semi-minor axis of the ellipse
        l: np.array, shape (nquadpoints,)
            2pi*l is an equally spaced array in theta from 0 to 2pi

    Returns:
        out: np.array, shape (nquadpoints, 3)
            xyz coordinates of the curve
    """
    # l is equally spaced from 0 to 1
    nl = l.size
    out = jnp.zeros((nl, 3))
    out = out.at[:, 0].set(r_ellipse(a, b, l)*jnp.cos(2.*np.pi*l))  # x
    out = out.at[:, 1].set(0.0)  # y
    out = out.at[:, 2].set(r_ellipse(a, b, l)*jnp.sin(2.*np.pi*l))  # z
    return out


def rotations(curve, b, alpha_x, alpha_y, alpha_z, dr):
    """
    Rotates a curve in Cartesian coordinates by moving the bottom of the coil up to z=0, 
    then rotating in the x, y, and z directions,
    then moving the bottom of the coil back down to z=b.
    Args:
        curve: np.array, shape (nquadpoints, 3)
            Curve coordinates.
        b: float
            semi-minor axis of the ellipse
        alpha_x: float
            rotation angle in the x direction
        alpha_y: float
            rotation angle in the y direction
        alpha_z: float
            rotation angle in the z direction

    Returns:
        out: np.array, shape (nquadpoints, 3)
            Rotated curve coordinates.
    """
    z_rot = jnp.asarray(
        [[jnp.cos(alpha_z), -jnp.sin(alpha_z), 0],
         [jnp.sin(alpha_z), jnp.cos(alpha_z), 0],
         [0, 0, 1]])

    y_rot = jnp.asarray(
        [[jnp.cos(alpha_y), 0, jnp.sin(alpha_y)],
         [0, 1, 0],
         [-jnp.sin(alpha_y), 0, jnp.cos(alpha_y)]])

    x_rot = jnp.asarray(
        [[1, 0, 0],
         [0, jnp.cos(alpha_x), -jnp.sin(alpha_x)],
         [0, jnp.sin(alpha_x), jnp.cos(alpha_x)]])

    out = curve

    # I think this is doing the z rotation, so use b
    out = out.at[:, 2].set(out[:, 2] + b)  # move bottom of coil up to z=0
    out = out @ y_rot @ x_rot @ z_rot  # apply rotation in this frame
    out = out.at[:, 0].set(out[:, 0] + dr)
    out = out.at[:, 2].set(out[:, 2] - b)  # move bottom of coils back down

    return out


def convert_to_cyl(a):
    """
    Converts a curve to cylindrical coordinates
    
    Args:
        a: np.array, shape (nquadpoints, 3)
            Cartesian coordinates of the curve.

    Returns:
        out: np.array, shape (nquadpoints, 3)
            Cylindrical coordinates of the curve.
    """
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(jnp.sqrt(a[:, 0]**2 + a[:, 1]**2))  # R = X^2 + Y^2
    out = out.at[:, 1].set(jnp.arctan2(a[:, 1], a[:, 0]))  # phi = arctan(Y/X)
    out = out.at[:, 2].set(a[:, 2])  # z = z
    return out


def cylindrical_shift(a, dphi, dz):
    """
    Shifts a curve in cylindrical coordinates.
    
    Args:
        a: np.array, shape (nquadpoints, 3)
            Cylindrical coordinates of the curve.
        dphi: float
            Shift in the toroidal angle.
        dz: float
            Shift in the vertical position.
    Returns:
        out: np.array, shape (nquadpoints, 3)
            Shifted cylindrical coordinates of the curve.
    """
    #shifting in r, phi, z
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(a[:, 0])
    out = out.at[:, 1].set(a[:, 1] + dphi)
    out = out.at[:, 2].set(a[:, 2] + dz)
    return out


def cyl_to_cart(a):
    """
    Converts curve cylindrical coordinates to cartesian coordinates.
    
    Args:
        a: np.array, shape (nquadpoints, 3)
            Cylindrical coordinates of the curve.

    Returns:
        out: np.array, shape (nquadpoints, 3)
            Cartesian coordinates of the curve.
    """
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(a[:, 0] * jnp.cos(a[:, 1]))  # x = R*cos(phi)
    out = out.at[:, 1].set(a[:, 0] * jnp.sin(a[:, 1]))  # y = R*sin(phi)
    out = out.at[:, 2].set(a[:, 2])  # z = z
    return out


def gamma_pure(dofs, points, a, b):
    """
    Generates a curve in cartesian coordinates.
    
    Args:
        dofs: array of dofs
        points: array of points
        a: float
            semi-major axis of the ellipse
        b: float
            semi-minor axis of the ellipse

    Returns:
        out: JaxCurve object
    """
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    g1 = xyz_cyl(a, b, points)  # generate elliptical parameterization
    g2 = rotations(g1, b, ypr[0], ypr[1], ypr[2], xyz[0])  # rotate in local X, Y, Z coordinates
    g3 = convert_to_cyl(g2)  # convert to R, phi, Z coordinates
    g4 = cylindrical_shift(g3, xyz[1], xyz[2])  # shift in r, phi, z
    final_gamma = cyl_to_cart(g4)  # convert back to cartesian
    return final_gamma


class CurvePlanarEllipticalCylindrical(JaxCurve):
    """
    CurvePlanarEllipticalCylindrical is a translated and rotated Curve in r, phi, and z coordinates.
    
    Mathematically, the underlying curve is an ellipse in the (X, Z) plane (Cartesian coordinates),
    parameterized as:

    .. math::
        X(t) = E * cos(2π t)
        Y(t) = 0
        Z(t) = E * sin(2π t)
        E(t) = ab / \sqrt((b \cos(2π t))^2 + (a \sin(2π t))^2)
    for :math:`t in [0, 1)`, where:
        - :math:`a` is the semimajor axis 
        - :math:`b` is the semiminor axis 
       
    The Cartesian coordinates of the curve are then rotated in 3D space by additional degrees of freedom:
        - Rotations about the x, y, and z axes (alpha_x, alpha_y, alpha_z)

    I am not sure the motivation, but then the curve is shifted in the X-direction by R0. 

    Then these Cartesian coordinates are converted to cylindrical coordinates, shifted in phi, and z,
    by additional degrees of freedom (phi, Z0) and then converted back to Cartesian coordinates.
    
    Args:
        quadpoints: number of quadrature points
        a: float
            semi-major axis of the ellipse
        b: float
            semi-minor axis of the ellipse
        dofs: array of dofs

    Returns:
        out: JaxCurve object
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
        """
        Returns the degrees of freedom (DoFs) for this object.
        """
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
        angle_name = ['x_rotation', 'y_rotation', 'z_rotation']
        return rtpz_name + angle_name


def create_equally_spaced_cylindrical_curves(ncurves, nfp, stellsym, R0, a, b, numquadpoints=32):
    """
    Create a list of equally spaced cylindrical curves in the toroidal coordinate.
    Args:
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
        curve.set('x_rotation', 0)
        curve.set('y_rotation', 0)
        curve.set('z_rotation', 0)
        curves.append(curve)
    return curves
