import numpy as np
from scipy.special import ellipk, ellipe
from simsopt.geo.magneticfield import MagneticField
import simsgeopp as sgpp
try:
    from sympy.parsing.sympy_parser import parse_expr
    import sympy as sp
    sympy_found = True
except ImportError:
    sympy_found = False


class ToroidalField(MagneticField):
    """
    Magnetic field purely in the toroidal direction, that is, in the phi
    direction with (R,phi,Z) the standard cylindrical coordinates.
    Its modulus is given by B = B0*R0/R where R0 is the first input and B0 the second input to the function.

    Args:
        B0:  modulus of the magnetic field at R0
        R0:  radius of normalization
    """

    def __init__(self, R0, B0):
        self.R0 = R0
        self.B0 = B0

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi), R), np.divide(np.cos(phi), R), np.zeros(len(phi)))).T
        self._B = np.multiply(self.B0*self.R0, phiUnitVectorOverR)

        x = points[:, 0]
        y = points[:, 1]

        if compute_derivatives >= 1:

            dB_by_dX1 = np.vstack((
                np.multiply(np.divide(self.B0*self.R0, R**4), 2*np.multiply(x, y)),
                np.multiply(np.divide(self.B0*self.R0, R**4), y**2-x**2),
                0*R))
            dB_by_dX2 = np.vstack((
                np.multiply(np.divide(self.B0*self.R0, R**4), y**2-x**2),
                np.multiply(np.divide(self.B0*self.R0, R**4), -2*np.multiply(x, y)),
                0*R))
            dB_by_dX3 = np.vstack((0*R, 0*R, 0*R))

            dToroidal_by_dX = np.array([dB_by_dX1, dB_by_dX2, dB_by_dX3]).T

            self._dB_by_dX = dToroidal_by_dX

        if compute_derivatives >= 2:
            self._d2B_by_dXdX = 2*self.B0*self.R0*np.multiply(
                1/(points[:, 0]**2+points[:, 1]**2)**3, np.array([
                    [[3*points[:, 0]**2+points[:, 1]**3, points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, np.zeros((len(points)))],
                     [points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                        np.zeros((len(points)))],
                     np.zeros((3, len(points)))],
                    [[points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                      np.zeros((len(points)))],
                     [3*points[:, 0]**2*points[:, 1]-points[:, 1]**3, -points[:, 0]**3+3*points[:, 0]*points[:, 1]**2,
                      np.zeros((len(points)))],
                     np.zeros((3, len(points)))],
                    np.zeros((3, 3, len(points)))])).T
        return self

    def compute_A(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        self._A = self.B0*self.R0*np.array([
            points[:, 2]*points[:, 0]/(points[:, 0]**2+points[:, 1]**2),
            points[:, 2]*points[:, 1]/(points[:, 0]**2+points[:, 1]**2),
            0*points[:, 2]]).T

        if compute_derivatives >= 1:
            self._dA_by_dX = self.B0*self.R0*np.array((points[:, 2]/(points[:, 0]**2+points[:, 1]**2)**2)*np.array(
                [[-points[:, 0]**2+points[:, 1]**2, -2*points[:, 0]*points[:, 1], np.zeros((len(points)))],
                 [-2*points[:, 0]*points[:, 1], points[:, 0]**2-points[:, 1]**2, np.zeros((len(points)))],
                 [points[:, 0]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
                  points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2], np.zeros((len(points)))]])).T

        if compute_derivatives >= 2:
            self._d2A_by_dXdX = 2*self.B0*self.R0*np.array(
                (points[:, 2]/(points[:, 0]**2+points[:, 1]**2)**3)*np.array([
                    [[points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                      (-points[:, 0]**4+points[:, 1]**4)/(2*points[:, 2])],
                     [3*points[:, 0]**2*points[:, 1]-points[:, 1]**3, -points[:, 0]**3+3*points[:, 0]*points[:, 1]**2, -points[:, 0]*points[:, 1]*(
                         points[:, 0]**2+points[:, 1]**2)/points[:, 2]],
                     [(-points[:, 0]**4+points[:, 1]**4)/(2*points[:, 2]),
                      -points[:, 0]*points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
                      np.zeros((len(points)))]],
                    [[3*points[:, 0]**2*points[:, 1]-points[:, 1]**3, -points[:, 0]**3+3*points[:, 0]*points[:, 1]**2,
                      -points[:, 0]*points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2]],
                     [-points[:, 0]**3+3*points[:, 0]*points[:, 1]**2, -3*points[:, 0]**2*points[:, 1]+points[:, 1]**3, (
                         points[:, 0]**4-points[:, 1]**4)/(2*points[:, 2])],
                     [-points[:, 0]*points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
                      (points[:, 0]**4-points[:, 1]**4)/(2*points[:, 2]), np.zeros((len(points)))]],
                    np.zeros((3, 3, len(points)))])).transpose((3, 0, 1, 2))


class ScalarPotentialRZMagneticField(MagneticField):
    """
    Vacuum magnetic field as a solution of B = grad(phi) where phi is the
    magnetic field scalar potential.  It takes phi as an input string, which
    should contain an expression involving the standard cylindrical coordinates
    (R, phi, Z) Example: ScalarPotentialRZMagneticField("2*phi") yields a
    magnetic field B = grad(2*phi) = (0,2/R,0).  Note: this function needs
    sympy.

    Args:
        phi_str:  string containing vacuum scalar potential expression as a function of R, Z and phi
    """

    def __init__(self, phi_str):
        if not sympy_found:
            raise RuntimeError("Sympy is required for the ScalarPotentialRZMagneticField class")
        self.phi_str = phi_str
        self.phi_parsed = parse_expr(phi_str)
        R, Z, phi = sp.symbols('R Z phi')
        self.Blambdify = sp.lambdify((R, Z, phi), [self.phi_parsed.diff(R)+1e-30*sp.sin(phi), self.phi_parsed.diff(phi)/R+1e-30*sp.sin(phi), self.phi_parsed.diff(Z)+1e-30*sp.sin(phi)])
        self.dBlambdify_by_dX = sp.lambdify(
            (R, Z, phi),
            [[sp.cos(phi)*self.phi_parsed.diff(R).diff(R)-(sp.sin(phi)/R)*self.phi_parsed.diff(R).diff(phi),
              sp.cos(phi)*(self.phi_parsed.diff(phi)/R).diff(R)-(sp.sin(phi)/R)*(self.phi_parsed.diff(phi)/R).diff(phi),
              sp.cos(phi)*self.phi_parsed.diff(Z).diff(R)-(sp.sin(phi)/R)*self.phi_parsed.diff(Z).diff(phi)],
             [sp.sin(phi)*self.phi_parsed.diff(R).diff(R)+(sp.cos(phi)/R)*self.phi_parsed.diff(R).diff(phi),
              sp.sin(phi)*(self.phi_parsed.diff(phi)/R).diff(R)+(sp.cos(phi)/R)*(self.phi_parsed.diff(phi)/R).diff(phi),
              sp.sin(phi)*self.phi_parsed.diff(Z).diff(R)+(sp.cos(phi)/R)*self.phi_parsed.diff(Z).diff(phi)],
             [self.phi_parsed.diff(R).diff(Z)+1e-30*sp.sin(phi), (self.phi_parsed.diff(phi)/R).diff(Z), self.phi_parsed.diff(Z).diff(Z)+1e-30*sp.sin(phi)]])

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        self._B = np.array(self.Blambdify(r, z, phi)).T

        if compute_derivatives >= 1:
            self._dB_by_dX = np.array(self.dBlambdify_by_dX(r, z, phi)).transpose((2, 0, 1))


class CircularCoil(MagneticField):
    '''
    Magnetic field created by a single circular coil evaluated using analytical
    functions, including complete elliptic integrals of the first and second
    kind.  As inputs, it takes the radius of the coil (r0), its center, current
    (I) and its normal vector [either spherical angle components
    (normal=[theta,phi]) or (x,y,z) components of a vector (normal=[x,y,z])]).

    Args:
        r0: radius of the coil
        center: point at the coil center
        I: current of the coil in Ampere's
        normal: if list with two values treats it as spherical angles theta and
                phi of the normal vector to the plane of the coil centered at the coil
                center, if list with three values treats it a vector
    '''

    def __init__(self, r0=0.1, center=[0, 0, 0], I=5e5/np.pi, normal=[0, 0]):
        self.r0 = r0
        self.Inorm = I*4e-7
        self.center = center
        if len(normal) == 2:
            self.normal = [normal[0], -normal[1]]
        else:
            self.normal = [np.arctan2(np.sqrt(normal[0]**2+normal[1]**2), normal[2]), -np.arctan2(normal[0], normal[1])]
        self.rot_matrix = np.array([[np.cos(self.normal[1]), np.sin(self.normal[0])*np.sin(self.normal[1]),
                                    np.cos(self.normal[0])*np.sin(self.normal[1])],
                                   [0, np.cos(self.normal[0]), -np.sin(self.normal[0])],
                                   [np.sin(self.normal[1]), np.sin(self.normal[0])*np.cos(self.normal[1]),
                                    np.cos(self.normal[0])*np.cos(self.normal[1])]])
        self.rot_matrix_inv = np.array(self.rot_matrix.T)

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        points = np.array(np.dot(self.rot_matrix, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)
        gamma = np.square(points[:, 0]) - np.square(points[:, 1])
        self._B = np.dot(self.rot_matrix_inv, np.array(
            [self.Inorm*points[:, 0]*points[:, 2]/(2*alpha**2*beta*rho**2)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm*points[:, 1]*points[:, 2]/(2*alpha**2*beta*rho**2)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm/(2*alpha**2*beta)*((self.r0**2-r**2)*ellipek2+alpha**2*ellipkk2)])).T

        if compute_derivatives >= 1:

            dBxdx = (self.Inorm*points[:, 2]*(
                ellipkk2*alpha**2*((2*points[:, 0]**4 + gamma*(
                    points[:, 1]**2 + points[:, 2]**2))*r**2 + self.r0**2*(
                        gamma*(self.r0**2 + 2*points[:, 2]**2) - (3*points[:, 0]**2 - 2*points[:, 1]**2)*rho**2))
                + ellipek2*(-((2*points[:, 0]**4 + gamma*(points[:, 1]**2 + points[:, 2]**2))*r**4)
                            + self.r0**4*(-(gamma*(self.r0**2 + 3*points[:, 2]**2)) + (8*points[:, 0]**2 - points[:, 1]**2)*rho**2)
                            - self.r0**2*(
                                3*gamma*points[:, 2]**4 - 2*(2*points[:, 0]**2 + points[:, 1]**2)*points[:, 2]**2 * rho**2
                                + (5*points[:, 0]**2 + points[:, 1]**2)*rho**4
                ))
            ))/(2*alpha**4*beta**3*rho**4)

            dBydx = (self.Inorm*points[:, 0]*points[:, 1]*points[:, 2]*(
                ellipkk2*alpha**2*(
                    2*self.r0**4 + r**2*(2*r**2 + rho**2) - self.r0**2*(-4*points[:, 2]**2 + 5*rho**2))
                + ellipek2*(-2*self.r0**6 - r**4*(2*r**2 + rho**2) + 3*self.r0**4*(-2*points[:, 2]**2 + 3*rho**2) - 2*self.r0**2*(3*points[:, 2]**4 - points[:, 2]**2*rho**2 + 2*rho**4))
            ))/(2*alpha**4*beta**3*rho**4)

            dBzdx = (self.Inorm*points[:, 0]*(
                - (ellipkk2*alpha**2*((-self.r0**2 + rho**2)**2 + points[:, 2]**2*(self.r0**2 + rho**2)))
                + ellipek2*(
                    points[:, 2]**4*(self.r0**2 + rho**2) + (-self.r0**2 + rho**2)**2*(self.r0**2 + rho**2)
                    + 2*points[:, 2]**2*(self.r0**4 - 6*self.r0**2*rho**2 + rho**4))
            ))/(2*alpha**4*beta**3*rho**2)
            dBxdy = dBydx

            dBydy = (self.Inorm*points[:, 2]*(
                ellipkk2*alpha**2*((2*points[:, 1]**4 - gamma*(points[:, 0]**2 + points[:, 2]**2))*r**2 +
                                   self.r0**2*(-(gamma*(self.r0**2 + 2*points[:, 2]**2)) - (-2*points[:, 0]**2 + 3*points[:, 1]**2)*rho**2)) +
                ellipek2*(-((2*points[:, 1]**4 - gamma*(points[:, 0]**2 + points[:, 2]**2))*r**4) +
                          self.r0**4*(gamma*(self.r0**2 + 3*points[:, 2]**2) + (-points[:, 0]**2 + 8*points[:, 1]**2)*rho**2) -
                          self.r0**2*(-3*gamma*points[:, 2]**4 - 2*(points[:, 0]**2 + 2*points[:, 1]**2)*points[:, 2]**2*rho**2 +
                                      (points[:, 0]**2 + 5*points[:, 1]**2)*rho**4))))/(2*alpha**4*beta**3*rho**4)

            dBzdy = dBzdx*points[:, 1]/points[:, 0]

            dBxdz = dBzdx

            dBydz = dBzdy

            dBzdz = (self.Inorm*points[:, 2]*(ellipkk2*alpha**2*(self.r0**2 - r**2) +
                                              ellipek2*(-7*self.r0**4 + r**4 + 6*self.r0**2*(-points[:, 2]**2 + rho**2))))/(2*alpha**4*beta**3)

            dB_by_dXm = np.array([
                [dBxdx, dBydx, dBzdx],
                [dBxdy, dBydy, dBzdy],
                [dBxdz, dBydz, dBzdz]])

            self._dB_by_dX = np.array([
                [np.dot(self.rot_matrix[:, 0], np.dot(self.rot_matrix_inv[0, :], dB_by_dXm)),
                 np.dot(self.rot_matrix[:, 1], np.dot(self.rot_matrix_inv[0, :], dB_by_dXm)),
                 np.dot(self.rot_matrix[:, 2], np.dot(self.rot_matrix_inv[0, :], dB_by_dXm))],
                [np.dot(self.rot_matrix[:, 0], np.dot(self.rot_matrix_inv[1, :], dB_by_dXm)),
                 np.dot(self.rot_matrix[:, 1], np.dot(self.rot_matrix_inv[1, :], dB_by_dXm)),
                 np.dot(self.rot_matrix[:, 2], np.dot(self.rot_matrix_inv[1, :], dB_by_dXm))],
                [np.dot(self.rot_matrix[:, 0], np.dot(self.rot_matrix_inv[2, :], dB_by_dXm)),
                 np.dot(self.rot_matrix[:, 1], np.dot(self.rot_matrix_inv[2, :], dB_by_dXm)),
                 np.dot(self.rot_matrix[:, 2], np.dot(self.rot_matrix_inv[2, :], dB_by_dXm))]]).T

    def compute_A(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        points = np.array(np.dot(self.rot_matrix, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)

        self._A = -self.Inorm/2*np.dot(self.rot_matrix_inv, np.array(
            (2*self.r0+np.sqrt(points[:, 0]**2+points[:, 1]**2)*ellipek2+(self.r0**2+points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)*(ellipek2-ellipkk2)) /
            ((points[:, 0]**2+points[:, 1]**2)*np.sqrt(self.r0**2+points[:, 0]**2+points[:, 1]**2+2*self.r0*np.sqrt(points[:, 0]**2+points[:, 1]**2)+points[:, 2]**2)) *
            np.array([-points[:, 1], points[:, 0], 0*points[:, 0]]))).T


class Dommaschk(MagneticField):
    """
    Vacuum magnetic field created by an explicit representation of the magnetic
    field scalar potential as proposed by W. Dommaschk (1986), Computer Physics
    Communications 40, 203-218. As inputs, it takes the arrays for the harmonics
    m, n and its corresponding coefficients.

    Args:
        m: first harmonic array
        n: second harmonic array
        coeffs: coefficient for Vml for each of the ith index of the harmonics m and n
    """

    def __init__(self, mn=[[0, 0]], coeffs=[[0, 0]]):
        self.m = np.array(mn, dtype=np.int16)[:, 0]
        self.n = np.array(mn, dtype=np.int16)[:, 1]
        self.coeffs = coeffs
        self.Btor = ToroidalField(1, 1)

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2
        self.Btor.set_points(points)
        self._B = np.add.reduce(sgpp.DommaschkB(self.m, self.n, self.coeffs, points))+self.Btor.B()

        if compute_derivatives >= 1:
            self._dB_by_dX = np.add.reduce(sgpp.DommaschkdB(self.m, self.n, self.coeffs, points))+self.Btor.dB_by_dX()


class Reiman(MagneticField):
    '''Magnetic field model in section 5 of Reiman and Greenside, Computer Physics Communications 43 (1986) 157—167. 
    This field allows for an analytical expression of the magnetic island width that can be used for island optimization.
    However, the field is not completely physical as it does not have nested flux surfaces.

    Args:
        iota0: unperturbed rotational transform
        iota1: unperturbed global magnetic shear
        k: integer array specifying the Fourier modes used
        epsilonk: coefficient of the Fourier modes
        m0: toroidal symmetry parameter (normally m0=1)
    '''

    def __init__(self, iota0=0.15, iota1=0.38, k=[6], epsilonk=[0.01], m0=1):
        self.iota0 = iota0
        self.iota1 = iota1
        self.k = k
        self.epsilonk = epsilonk
        self.m0 = m0

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2
        self._B = sgpp.ReimanB(self.iota0, self.iota1, self.k, self.epsilonk, self.m0, points)

        if compute_derivatives >= 1:
            self._dB_by_dX = sgpp.ReimandB(self.iota0, self.iota1, self.k, self.epsilonk, self.m0, points)
