import numpy as np
from scipy.special import ellipk, ellipe
from simsopt.field.magneticfield import MagneticField
import simsoptpp as sopp
import logging
try:
    from sympy.parsing.sympy_parser import parse_expr
    import sympy as sp
    sympy_found = True
except ImportError:
    sympy_found = False

logger = logging.getLogger(__name__)


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
        MagneticField.__init__(self)
        self.R0 = R0
        self.B0 = B0

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi), R), np.divide(np.cos(phi), R), np.zeros(len(phi)))).T
        B[:] = np.multiply(self.B0*self.R0, phiUnitVectorOverR)

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))

        x = points[:, 0]
        y = points[:, 1]

        dB_by_dX1 = np.vstack((
            np.multiply(np.divide(self.B0*self.R0, R**4), 2*np.multiply(x, y)),
            np.multiply(np.divide(self.B0*self.R0, R**4), y**2-x**2),
            0*R))
        dB_by_dX2 = np.vstack((
            np.multiply(np.divide(self.B0*self.R0, R**4), y**2-x**2),
            np.multiply(np.divide(self.B0*self.R0, R**4), -2*np.multiply(x, y)),
            0*R))
        dB_by_dX3 = np.vstack((0*R, 0*R, 0*R))

        dB[:] = np.array([dB_by_dX1, dB_by_dX2, dB_by_dX3]).T

    def _d2B_by_dXdX_impl(self, ddB):
        points = self.get_points_cart_ref()
        x = points[:, 0]
        y = points[:, 1]
        ddB[:] = 2*self.B0*self.R0*np.multiply(
            1/(points[:, 0]**2+points[:, 1]**2)**3, np.array([
                [[3*points[:, 0]**2+points[:, 1]**3, points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, np.zeros((len(points)))], [
                    points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                    np.zeros((len(points)))],
                 np.zeros((3, len(points)))],
                [[points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                  np.zeros((len(points)))],
                 [3*points[:, 0]**2*points[:, 1]-points[:, 1]**3, -points[:, 0]**3+3*points[:, 0]*points[:, 1]**2,
                  np.zeros((len(points)))], np.zeros((3, len(points)))],
                np.zeros((3, 3, len(points)))])).T

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        A[:] = self.B0*self.R0*np.array([
            points[:, 2]*points[:, 0]/(points[:, 0]**2+points[:, 1]**2),
            points[:, 2]*points[:, 1]/(points[:, 0]**2+points[:, 1]**2),
            0*points[:, 2]]).T

    def _dA_by_dX_impl(self, dA):
        points = self.get_points_cart_ref()
        dA[:] = self.B0*self.R0*np.array((points[:, 2]/(points[:, 0]**2+points[:, 1]**2)**2)*np.array(
            [[-points[:, 0]**2+points[:, 1]**2, -2*points[:, 0]*points[:, 1], np.zeros((len(points)))],
             [-2*points[:, 0]*points[:, 1], points[:, 0]**2-points[:, 1]**2, np.zeros((len(points)))],
             [points[:, 0]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
              points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2], np.zeros((len(points)))]])).T

    def _d2A_by_dXdX_impl(self, ddA):
        points = self.get_points_cart_ref()
        ddA[:] = 2*self.B0*self.R0*np.array(
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


class PoloidalField(MagneticField):
    '''
    Magnetic field purely in the poloidal direction, that is, in the
    theta direction of a poloidal-toroidal coordinate system.  Its
    modulus is given by B = B0 * r / (R0 * q) so that, together with
    the toroidal field, it creates a safety factor equals to q

    Args:
        B0: modulus of the magnetic field at R0
        R0: major radius of the magnetic axis
        q: safety factor/pitch angle of the magnetic field lines
    '''

    def __init__(self, R0, B0, q):
        MagneticField.__init__(self)
        self.R0 = R0
        self.B0 = B0
        self.q = q

    def _B_impl(self, B):
        points = self.get_points_cart_ref()

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        phi = np.arctan2(y, x)
        theta = np.arctan2(z, np.sqrt(x**2+y**2)-self.R0)
        r = np.sqrt((np.sqrt(x**2+y**2)-self.R0)**2+z**2)
        thetaUnitVectorOver_times_r = np.vstack((-np.multiply(np.sin(theta), r)*np.cos(phi), -np.multiply(np.sin(theta), r)*np.sin(phi), np.multiply(np.cos(theta), r))).T
        B[:] = self.B0/self.R0/self.q*thetaUnitVectorOver_times_r

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        phi = np.arctan2(y, x)
        theta = np.arctan2(z, np.sqrt(x**2+y**2)-self.R0)
        r = np.sqrt((np.sqrt(x**2+y**2)-self.R0)**2+z**2)

        dtheta_by_dX1 = -((x*z)/(np.sqrt(x**2+y**2)*(x**2+y**2+z**2-2*np.sqrt(x**2+y**2)*self.R0+(self.R0)**2)))
        dtheta_by_dX2 = -((y*z)/(np.sqrt(x**2+y**2)*(x**2+y**2+z**2-2*np.sqrt(x**2+y**2)*self.R0+(self.R0)**2)))
        dtheta_by_dX3 = 1/((-self.R0+np.sqrt(x**2+y**2))*(1+z**2/(self.R0-np.sqrt(x**2+y**2))**2))

        dphi_by_dX1 = -(y/(x**2 + y**2))
        dphi_by_dX2 = x/(x**2 + y**2)
        dphi_by_dX3 = 0.*z

        dthetaunitvector_by_dX1 = np.vstack((
            -np.cos(theta)*np.cos(phi)*dtheta_by_dX1+np.sin(theta)*np.sin(phi)*dphi_by_dX1,
            -np.cos(theta)*np.sin(phi)*dtheta_by_dX1-np.sin(theta)*np.cos(phi)*dphi_by_dX1,
            -np.sin(theta)*dtheta_by_dX1
        )).T
        dthetaunitvector_by_dX2 = np.vstack((
            -np.cos(theta)*np.cos(phi)*dtheta_by_dX2+np.sin(theta)*np.sin(phi)*dphi_by_dX2,
            -np.cos(theta)*np.sin(phi)*dtheta_by_dX2-np.sin(theta)*np.cos(phi)*dphi_by_dX2,
            -np.sin(theta)*dtheta_by_dX2
        )).T
        dthetaunitvector_by_dX3 = np.vstack((
            -np.cos(theta)*np.cos(phi)*dtheta_by_dX3+np.sin(theta)*np.sin(phi)*dphi_by_dX3,
            -np.cos(theta)*np.sin(phi)*dtheta_by_dX3-np.sin(theta)*np.cos(phi)*dphi_by_dX3,
            -np.sin(theta)*dtheta_by_dX3
        )).T

        dB_by_dX1_term1 = np.multiply(dthetaunitvector_by_dX1.T, r)
        dB_by_dX2_term1 = np.multiply(dthetaunitvector_by_dX2.T, r)
        dB_by_dX3_term1 = np.multiply(dthetaunitvector_by_dX3.T, r)

        thetaUnitVector_1 = -np.sin(theta)*np.cos(phi)
        thetaUnitVector_2 = -np.sin(theta)*np.sin(phi)
        thetaUnitVector_3 = np.cos(theta)

        dr_by_dX1 = (x*(-self.R0+np.sqrt(x**2+y**2)))/(np.sqrt(x**2+y**2)*np.sqrt((self.R0-np.sqrt(x**2+y**2))**2+z**2))
        dr_by_dX2 = (y*(-self.R0+np.sqrt(x**2+y**2)))/(np.sqrt(x**2+y**2)*np.sqrt((self.R0-np.sqrt(x**2+y**2))**2+z**2))
        dr_by_dX3 = z/np.sqrt((self.R0-np.sqrt(x**2+y**2))**2+z**2)

        dB_by_dX1_term2 = np.vstack((
            thetaUnitVector_1*dr_by_dX1,
            thetaUnitVector_2*dr_by_dX1,
            thetaUnitVector_3*dr_by_dX1))
        dB_by_dX2_term2 = np.vstack((
            thetaUnitVector_1*dr_by_dX2,
            thetaUnitVector_2*dr_by_dX2,
            thetaUnitVector_3*dr_by_dX2))
        dB_by_dX3_term2 = np.vstack((
            thetaUnitVector_1*dr_by_dX3,
            thetaUnitVector_2*dr_by_dX3,
            thetaUnitVector_3*dr_by_dX3))

        dB[:] = self.B0/self.R0/self.q*np.array([dB_by_dX1_term1+dB_by_dX1_term2, dB_by_dX2_term1+dB_by_dX2_term2, dB_by_dX3_term1+dB_by_dX3_term2]).T


class ScalarPotentialRZMagneticField(MagneticField):
    """
    Vacuum magnetic field as a solution of B = grad(Phi) where Phi is the
    magnetic field scalar potential.  It takes Phi as an input string, which
    should contain an expression involving the standard cylindrical coordinates
    (R, phi, Z) Example: ScalarPotentialRZMagneticField("2*phi") yields a
    magnetic field B = grad(2*phi) = (0,2/R,0). In order for the analytical
    derivatives to be performed by sympy, a term 1e-30*Phi*R*Z is added
    to every entry. Note: this function needs sympy.

    Args:
        phi_str:  string containing vacuum scalar potential expression as a function of R, Z and phi
    """

    ## TRY to add C*R*phi*Z in all entries and then put C=0

    def __init__(self, phi_str):
        MagneticField.__init__(self)
        if not sympy_found:
            raise RuntimeError("Sympy is required for the ScalarPotentialRZMagneticField class")
        self.phi_str = phi_str
        self.phi_parsed = parse_expr(phi_str)
        R, Z, Phi = sp.symbols('R Z phi')
        self.Blambdify = sp.lambdify((R, Z, Phi), [self.phi_parsed.diff(R)+1e-30*Phi*R*Z,\
                                                   self.phi_parsed.diff(Phi)/R+1e-30*Phi*R*Z,\
                                                   self.phi_parsed.diff(Z)+1e-30*Phi*R*Z])
        self.dBlambdify_by_dX = sp.lambdify(
            (R, Z, Phi),
            [[1e-30*Phi*R*Z+sp.cos(Phi)*self.phi_parsed.diff(R).diff(R)-(sp.sin(Phi)/R)*self.phi_parsed.diff(R).diff(Phi),
              1e-30*Phi*R*Z+sp.cos(Phi)*(self.phi_parsed.diff(Phi)/R).diff(R)-(sp.sin(Phi)/R)*(self.phi_parsed.diff(Phi)/R).diff(Phi),
              1e-30*Phi*R*Z+sp.cos(Phi)*self.phi_parsed.diff(Z).diff(R)-(sp.sin(Phi)/R)*self.phi_parsed.diff(Z).diff(Phi)],
             [1e-30*Phi*R*Z+sp.sin(Phi)*self.phi_parsed.diff(R).diff(R)+(sp.cos(Phi)/R)*self.phi_parsed.diff(R).diff(Phi),
              1e-30*Phi*R*Z+sp.sin(Phi)*(self.phi_parsed.diff(Phi)/R).diff(R)+(sp.cos(Phi)/R)*(self.phi_parsed.diff(Phi)/R).diff(Phi),
              1e-30*Phi*R*Z+sp.sin(Phi)*self.phi_parsed.diff(Z).diff(R)+(sp.cos(Phi)/R)*self.phi_parsed.diff(Z).diff(Phi)],
             [1e-30*Phi*R*Z+self.phi_parsed.diff(R).diff(Z),
              1e-30*Phi*R*Z+(self.phi_parsed.diff(Phi)/R).diff(Z),
              1e-30*Phi*R*Z+self.phi_parsed.diff(Z).diff(Z)]])

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        B[:] = np.array(self.Blambdify(r, z, phi)).T

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        dB[:] = np.array(self.dBlambdify_by_dX(r, z, phi)).transpose((2, 0, 1))


class CircularCoil(MagneticField):
    '''
    Magnetic field created by a single circular coil evaluated using analytical
    functions, including complete elliptic integrals of the first and second
    kind.  As inputs, it takes the radius of the coil (r0), its center, current
    (I) and its normal vector [either spherical angle components
    (normal=[theta,phi]) or (x,y,z) components of a vector (normal=[x,y,z])]).
    The (theta,phi) angles are related to the (x,y,z) components of the normal vector via
    theta = np.arctan2(normal[1], normal[0]) and phi = np.arctan2(np.sqrt(normal[0]**2+normal[1]**2), normal[2]).
    Sign convention: CircularCoil with a positive current produces a magnetic field
    vector in the same direction as the normal when evaluated at the center of the coil.a

    Args:
        r0: radius of the coil
        center: point at the coil center
        I: current of the coil in Ampere's
        normal: if list with two values treats it as spherical angles theta and
                phi of the normal vector to the plane of the coil centered at the coil
                center, if list with three values treats it a vector
    '''

    def __init__(self, r0=0.1, center=[0, 0, 0], I=5e5/np.pi, normal=[0, 0]):
        MagneticField.__init__(self)
        self.r0 = r0
        self.Inorm = I*4e-7
        self.center = center
        self.normal = normal
        if len(normal) == 2:
            theta = normal[0]
            phi = normal[1]
        else:
            theta = np.arctan2(normal[1], normal[0])
            phi = np.arctan2(np.sqrt(normal[0]**2+normal[1]**2), normal[2])

        self.rotMatrix = np.array([
            [np.cos(phi) * np.cos(theta)**2 + np.sin(theta)**2,
             -np.sin(phi / 2)**2 * np.sin(2 * theta), 
             np.cos(theta) * np.sin(phi)],
            [-np.sin(phi / 2)**2 * np.sin(2 * theta), 
             np.cos(theta)**2 + np.cos(phi) * np.sin(theta)**2, 
             np.sin(phi) * np.sin(theta)],
            [-np.cos(theta) * np.sin(phi),
             -np.sin(phi) * np.sin(theta), 
             np.cos(phi)]
        ])

        self.rotMatrixInv = np.array(self.rotMatrix.T)

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self.rotMatrixInv, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)
        gamma = np.square(points[:, 0]) - np.square(points[:, 1])
        B[:] = np.dot(self.rotMatrix, np.array(
            [self.Inorm*points[:, 0]*points[:, 2]/(2*alpha**2*beta*rho**2+1e-31)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm*points[:, 1]*points[:, 2]/(2*alpha**2*beta*rho**2+1e-31)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm/(2*alpha**2*beta+1e-31)*((self.r0**2-r**2)*ellipek2+alpha**2*ellipkk2)])).T

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self.rotMatrixInv, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)
        gamma = np.square(points[:, 0]) - np.square(points[:, 1])
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
        ))/(2*alpha**4*beta**3*rho**4+1e-31)

        dBydx = (self.Inorm*points[:, 0]*points[:, 1]*points[:, 2]*(
            ellipkk2*alpha**2*(
                2*self.r0**4 + r**2*(2*r**2 + rho**2) - self.r0**2*(-4*points[:, 2]**2 + 5*rho**2))
            + ellipek2*(-2*self.r0**6 - r**4*(2*r**2 + rho**2) + 3*self.r0**4*(-2*points[:, 2]**2 + 3*rho**2) - 2*self.r0**2*(3*points[:, 2]**4 - points[:, 2]**2*rho**2 + 2*rho**4))
        ))/(2*alpha**4*beta**3*rho**4+1e-31)

        dBzdx = (self.Inorm*points[:, 0]*(
            - (ellipkk2*alpha**2*((-self.r0**2 + rho**2)**2 + points[:, 2]**2*(self.r0**2 + rho**2)))
            + ellipek2*(
                points[:, 2]**4*(self.r0**2 + rho**2) + (-self.r0**2 + rho**2)**2*(self.r0**2 + rho**2)
                + 2*points[:, 2]**2*(self.r0**4 - 6*self.r0**2*rho**2 + rho**4))
        ))/(2*alpha**4*beta**3*rho**2+1e-31)
        dBxdy = dBydx

        dBydy = (self.Inorm*points[:, 2]*(
            ellipkk2*alpha**2*((2*points[:, 1]**4 - gamma*(points[:, 0]**2 + points[:, 2]**2))*r**2 +
                               self.r0**2*(-(gamma*(self.r0**2 + 2*points[:, 2]**2)) - (-2*points[:, 0]**2 + 3*points[:, 1]**2)*rho**2)) +
            ellipek2*(-((2*points[:, 1]**4 - gamma*(points[:, 0]**2 + points[:, 2]**2))*r**4) +
                      self.r0**4*(gamma*(self.r0**2 + 3*points[:, 2]**2) + (-points[:, 0]**2 + 8*points[:, 1]**2)*rho**2) -
                      self.r0**2*(-3*gamma*points[:, 2]**4 - 2*(points[:, 0]**2 + 2*points[:, 1]**2)*points[:, 2]**2*rho**2 +
                                  (points[:, 0]**2 + 5*points[:, 1]**2)*rho**4))))/(2*alpha**4*beta**3*rho**4+1e-31)

        dBzdy = dBzdx*points[:, 1]/(points[:, 0]+1e-31)

        dBxdz = dBzdx

        dBydz = dBzdy

        dBzdz = (self.Inorm*points[:, 2]*(ellipkk2*alpha**2*(self.r0**2 - r**2) +
                                          ellipek2*(-7*self.r0**4 + r**4 + 6*self.r0**2*(-points[:, 2]**2 + rho**2))))/(2*alpha**4*beta**3+1e-31)

        dB_by_dXm = np.array([
            [dBxdx, dBydx, dBzdx],
            [dBxdy, dBydy, dBzdy],
            [dBxdz, dBydz, dBzdz]])

        dB[:] = np.array([
            [np.dot(self.rotMatrixInv[:, 0], np.dot(self.rotMatrix[0, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 1], np.dot(self.rotMatrix[0, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 2], np.dot(self.rotMatrix[0, :], dB_by_dXm))],
            [np.dot(self.rotMatrixInv[:, 0], np.dot(self.rotMatrix[1, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 1], np.dot(self.rotMatrix[1, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 2], np.dot(self.rotMatrix[1, :], dB_by_dXm))],
            [np.dot(self.rotMatrixInv[:, 0], np.dot(self.rotMatrix[2, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 1], np.dot(self.rotMatrix[2, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 2], np.dot(self.rotMatrix[2, :], dB_by_dXm))]]).T

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self.rotMatrixInv, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)

        A[:] = -self.Inorm/2*np.dot(self.rotMatrix, np.array(
            (2*self.r0+np.sqrt(points[:, 0]**2+points[:, 1]**2)*ellipek2+(self.r0**2+points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)*(ellipe(k**2)-ellipkk2)) /
            ((points[:, 0]**2+points[:, 1]**2+1e-31)*np.sqrt(self.r0**2+points[:, 0]**2+points[:, 1]**2+2*self.r0*np.sqrt(points[:, 0]**2+points[:, 1]**2)+points[:, 2]**2+1e-31)) *
            np.array([-points[:, 1], points[:, 0], 0])).T)


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
        MagneticField.__init__(self)
        self.m = np.array(mn, dtype=np.int16)[:, 0]
        self.n = np.array(mn, dtype=np.int16)[:, 1]
        self.coeffs = coeffs
        self.Btor = ToroidalField(1, 1)

    def _set_points_cb(self):
        self.Btor.set_points_cart(self.get_points_cart_ref())

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        B[:] = np.add.reduce(sopp.DommaschkB(self.m, self.n, self.coeffs, points))+self.Btor.B()

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = np.add.reduce(sopp.DommaschkdB(self.m, self.n, self.coeffs, points))+self.Btor.dB_by_dX()


class Reiman(MagneticField):
    '''
    Magnetic field model in section 5 of Reiman and Greenside, Computer Physics Communications 43 (1986) 157—167.
    This field allows for an analytical expression of the magnetic island width
    that can be used for island optimization.  However, the field is not
    completely physical as it does not have nested flux surfaces.

    Args:
        iota0: unperturbed rotational transform
        iota1: unperturbed global magnetic shear
        k: integer array specifying the Fourier modes used
        epsilonk: coefficient of the Fourier modes
        m0: toroidal symmetry parameter (normally m0=1)
    '''

    def __init__(self, iota0=0.15, iota1=0.38, k=[6], epsilonk=[0.01], m0=1):
        MagneticField.__init__(self)
        self.iota0 = iota0
        self.iota1 = iota1
        self.k = k
        self.epsilonk = epsilonk
        self.m0 = m0

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        B[:] = sopp.ReimanB(self.iota0, self.iota1, self.k, self.epsilonk, self.m0, points)

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = sopp.ReimandB(self.iota0, self.iota1, self.k, self.epsilonk, self.m0, points)


class UniformInterpolationRule(sopp.UniformInterpolationRule):
    pass


class ChebyshevInterpolationRule(sopp.ChebyshevInterpolationRule):
    pass


class InterpolatedField(sopp.InterpolatedField, MagneticField):
    r"""
    This field takes an existing field and interpolates it on a regular grid in :math:`r,\phi,z`.
    This resulting interpolant can then be evaluated very quickly.
    """

    def __init__(self, field, degree, rrange, phirange, zrange, extrapolate=True, nfp=1, stellsym=False):
        r"""
        Args:
            field: the underlying :mod:`simsopt.field.magneticfield.MagneticField` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            rrange: a 3-tuple of the form ``(rmin, rmax, nr)``. This mean that the interval :math:`[rmin, rmax]` is
                    split into ``nr`` many subintervals.
            phirange: a 3-tuple of the form ``(phimin, phimax, nphi)``.
            zrange: a 3-tuple of the form ``(zmin, zmax, nz)``.
            extrapolate: whether to extrapolate the field when evaluate outside
                         the integration domain or to throw an error.
            nfp: Whether to exploit rotational symmetry. In this case any angle
                 is always mapped into the interval :math:`[0, 2\pi/\mathrm{nfp})`,
                 hence it makes sense to use ``phimin=0`` and
                 ``phimax=2*np.pi/nfp``.
            stellsym: Whether to exploit stellarator symmetry. In this case
                      ``z`` is always mapped to be positive, hence it makes sense to use
                      ``zmin=0``.
        """
        MagneticField.__init__(self)
        if stellsym and zrange[0] != 0:
            logger.warning(fr"Sure about zrange[0]={zrange[0]}? When exploiting stellarator symmetry, the interpolant is never evaluated for z<0.")
        if nfp > 1 and abs(phirange[1] - 2*np.pi/nfp) > 1e-14:
            logger.warning(fr"Sure about phirange[1]={phirange[1]}? When exploiting rotational symmetry, the interpolant is never evaluated for phi>2\pi/nfp.")

        sopp.InterpolatedField.__init__(self, field, degree, rrange, phirange, zrange, extrapolate, nfp, stellsym)
        self.__field = field

    def to_vtk(self, filename, h=0.1):
        """Export the field evaluated on a regular grid for visualisation with e.g. Paraview."""
        degree = self.rule.degree
        MagneticField.to_vtk(
            self, filename,
            nr=self.r_range[2]*degree+1,
            nphi=self.phi_range[2]*degree+1,
            nz=self.z_range[2]*degree+1,
            rmin=self.r_range[0], rmax=self.r_range[1],
            zmin=self.z_range[0], zmax=self.z_range[1]
        )
