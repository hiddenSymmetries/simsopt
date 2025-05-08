import logging

import numpy as np
import warnings
from scipy.special import ellipk, ellipe
try:
    from sympy.parsing.sympy_parser import parse_expr
    import sympy as sp
    sympy_found = True
except ImportError:
    sympy_found = False

import simsoptpp as sopp
from .magneticfield import MagneticField
from .._core.json import GSONDecoder

logger = logging.getLogger(__name__)

__all__ = ['ToroidalField', 'PoloidalField', 'ScalarPotentialRZMagneticField',
           'CircularCoil', 'Dommaschk', 'Reiman', 'InterpolatedField', 'DipoleField',
           'MirrorModel']


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

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["R0"], d["B0"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


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

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["R0"], d["B0"], d["q"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


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
        self.Blambdify = sp.lambdify((R, Z, Phi), [self.phi_parsed.diff(R)+1e-30*Phi*R*Z,
                                                   self.phi_parsed.diff(Phi)/R+1e-30*Phi*R*Z,
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
        B_cyl = np.array(self.Blambdify(r, z, phi)).T
        # Bx = Br cos(phi) - Bphi sin(phi)
        B[:, 0] = B_cyl[:, 0] * np.cos(phi) - B_cyl[:, 1] * np.sin(phi)
        # By = Br sin(phi) + Bphi cos(phi)
        B[:, 1] = B_cyl[:, 0] * np.sin(phi) + B_cyl[:, 1] * np.cos(phi)
        B[:, 2] = B_cyl[:, 2]

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        dB_cyl = np.array(self.dBlambdify_by_dX(r, z, phi)).transpose((2, 0, 1))
        dBrdx = dB_cyl[:, 0, 0]
        dBrdy = dB_cyl[:, 1, 0]
        dBrdz = dB_cyl[:, 2, 0]
        dBphidx = dB_cyl[:, 0, 1]
        dBphidy = dB_cyl[:, 1, 1]
        dBphidz = dB_cyl[:, 2, 1]
        dB[:, 0, 2] = dB_cyl[:, 0, 2]
        dB[:, 1, 2] = dB_cyl[:, 1, 2]
        dB[:, 2, 2] = dB_cyl[:, 2, 2]
        dcosphidx = -points[:, 0]**2/r**3 + 1/r
        dsinphidx = -points[:, 0]*points[:, 1]/r**3
        dcosphidy = -points[:, 0]*points[:, 1]/r**3
        dsinphidy = -points[:, 1]**2/r**3 + 1/r
        B_cyl = np.array(self.Blambdify(r, z, phi)).T
        Br = B_cyl[:, 0]
        Bphi = B_cyl[:, 1]
        # Bx = Br cos(phi) - Bphi sin(phi)
        dB[:, 0, 0] = dBrdx * np.cos(phi) + Br * dcosphidx - dBphidx * np.sin(phi) \
            - Bphi * dsinphidx
        dB[:, 1, 0] = dBrdy * np.cos(phi) + Br * dcosphidy - dBphidy * np.sin(phi) \
            - Bphi * dsinphidy
        dB[:, 2, 0] = dBrdz * np.cos(phi) - dBphidz * np.sin(phi)
        # By = Br sin(phi) + Bphi cos(phi)
        dB[:, 0, 1] = dBrdx * np.sin(phi) + Br * dsinphidx + dBphidx * np.cos(phi) \
            + Bphi * dcosphidx
        dB[:, 1, 1] = dBrdy * np.sin(phi) + Br * dsinphidy + dBphidy * np.cos(phi) \
            + Bphi * dcosphidy
        dB[:, 2, 1] = dBrdz * np.sin(phi) + dBphidz * np.cos(phi)

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["phi_str"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


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

        super().__init__(x0=self.get_dofs(), names=self._make_names(), external_dof_setter=CircularCoil.set_dofs_impl)

    def _make_names(self):
        if len(self.normal) == 2:
            normal_names = ['theta', 'phi']
        elif len(self.normal) == 3:
            normal_names = ['x', 'y', 'z']
        return ['r0', 'x0', 'y0', 'z0', 'Inorm'] + normal_names

    def num_dofs(self):
        return 5+len(self.normal)

    def get_dofs(self):
        return np.concatenate([np.array([self.r0]), np.array(self.center), np.array([self.Inorm]), np.array(self.normal)])

    def set_dofs_impl(self, dofs):
        self.r0 = dofs[0]
        self.center = dofs[1:4].tolist()
        self.Inorm = dofs[4]
        self.normal = dofs[5:].tolist()

    @property
    def I(self):
        return self.Inorm * 25e5

    def _rotmat(self):
        if len(self.normal) == 2:
            theta = self.get('theta')
            phi = self.get('phi')
        else:
            xn = self.get('x')
            yn = self.get('y')
            zn = self.get('z')
            theta = np.arctan2(yn, xn)
            phi = np.arctan2(np.sqrt(xn**2+yn**2), zn)

        m = np.array([
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
        return m

    def _rotmatinv(self):
        m = self._rotmat()
        minv = np.array(m.T)
        return minv

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self._rotmatinv(), np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)
        B[:] = np.dot(self._rotmat(), np.array(
            [self.Inorm*points[:, 0]*points[:, 2]/(2*alpha**2*beta*rho**2+1e-31)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm*points[:, 1]*points[:, 2]/(2*alpha**2*beta*rho**2+1e-31)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm/(2*alpha**2*beta+1e-31)*((self.r0**2-r**2)*ellipek2+alpha**2*ellipkk2)])).T

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self._rotmatinv(), np.array(np.subtract(points, self.center)).T).T)
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
            [np.dot(self._rotmatinv()[:, 0], np.dot(self._rotmat()[0, :], dB_by_dXm)),
             np.dot(self._rotmatinv()[:, 1], np.dot(self._rotmat()[0, :], dB_by_dXm)),
             np.dot(self._rotmatinv()[:, 2], np.dot(self._rotmat()[0, :], dB_by_dXm))],
            [np.dot(self._rotmatinv()[:, 0], np.dot(self._rotmat()[1, :], dB_by_dXm)),
             np.dot(self._rotmatinv()[:, 1], np.dot(self._rotmat()[1, :], dB_by_dXm)),
             np.dot(self._rotmatinv()[:, 2], np.dot(self._rotmat()[1, :], dB_by_dXm))],
            [np.dot(self._rotmatinv()[:, 0], np.dot(self._rotmat()[2, :], dB_by_dXm)),
             np.dot(self._rotmatinv()[:, 1], np.dot(self._rotmat()[2, :], dB_by_dXm)),
             np.dot(self._rotmatinv()[:, 2], np.dot(self._rotmat()[2, :], dB_by_dXm))]]).T

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self._rotmatinv(), np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)

        num = (2*self.r0+np.sqrt(points[:, 0]**2+points[:, 1]**2)*ellipek2+(self.r0**2+points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)*(ellipe(k**2)-ellipkk2))
        denom = ((points[:, 0]**2+points[:, 1]**2+1e-31)*np.sqrt(self.r0**2+points[:, 0]**2+points[:, 1]**2+2*self.r0*np.sqrt(points[:, 0]**2+points[:, 1]**2)+points[:, 2]**2+1e-31))
        fak = num/denom
        pts = fak[:, None]*np.concatenate((-points[:, 1][:, None], points[:, 0][:, None], np.zeros((points.shape[0], 1))), axis=-1)
        A[:] = -self.Inorm/2*np.dot(self._rotmat(), pts.T).T

    def as_dict(self, serial_objs_dict):
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["r0"], d["center"], d["I"], d["normal"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field

    def gamma(self, points=64):
        """Export points of the coil."""

        angle_points = np.linspace(0, 2*np.pi, points+1)[:-1]

        x = self.r0 * np.cos(angle_points)
        y = self.r0 * np.sin(angle_points)
        z = 0 * angle_points

        coords = np.add(np.dot(self._rotmat(), np.column_stack([x, y, z]).T).T, self.center)
        return coords

    def to_vtk(self, filename, close=False):
        """
        Export circular coil to VTK format

        Args:
            filename: Name of the file to write.
            close: Whether to draw the segment from the last quadrature point back to the first.
        """
        from pyevtk.hl import polyLinesToVTK

        def wrap(data):
            return np.concatenate([data, [data[0]]])

        # get the coordinates
        if close:
            x = wrap(self.gamma()[:, 0])
            y = wrap(self.gamma()[:, 1])
            z = wrap(self.gamma()[:, 2])
            ppl = np.asarray([self.gamma().shape[0]+1])
        else:
            x = self.gamma()[:, 0]
            y = self.gamma()[:, 1]
            z = self.gamma()[:, 2]
            ppl = np.asarray([self.gamma().shape[0]])

        polyLinesToVTK(str(filename), x, y, z, pointsPerLine=ppl)


class DipoleField(MagneticField):
    r"""
    Computes the MagneticField induced by N dipoles. The field is given by

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{i=1}^{N} \left(\frac{3\mathbf{r}_i\cdot \mathbf{m}_i}{|\mathbf{r}_i|^5}\mathbf{r}_i - \frac{\mathbf{m}_i}{|\mathbf{r}_i|^3}\right)

    where :math:`\mu_0=4\pi\times 10^{-7}\;N/A^2` is the permeability of free space
    and :math:`\mathbf{r_i} = \mathbf{x} - \mathbf{x}^{dipole}_i` is the
    vector between the field evaluation point and the dipole :math:`i`
    position.

    Args:
        dipole_grid: 2D numpy array, shape (ndipoles, 3).
            A set of points corresponding to the locations of magnetic dipoles.
        dipole_vectors: 2D numpy array, shape (ndipoles, 3).
            The dipole vectors of each of the dipoles in the grid.
        stellsym: bool (default True).
            Whether or not the dipole grid is stellarator symmetric.
        nfp: int (default 1).
            The field-period symmetry of the dipole-grid.
        coordinate_flag: string (default "cartesian").
            The global coordinate system that should be considered grid-aligned in the calculation.
            The options are "cartesian" (rectangular bricks), "cylindrical" (cylindrical bricks),
            and "toroidal" (uniform grid in simple toroidal coordinates). Note that this ASSUMES
            that the global coordinate system for the dipole locations is one of these three
            choices, so be careful if your permanent magnets are shaped/arranged differently!
        m_maxima: 1D numpy array, shape (ndipoles,).
            The maximum dipole strengths of each magnet in the grid. If not specified, defaults
            to using the largest dipole strength of the magnets in dipole_grid, and using this
            value for all the dipoles. Needed for plotting normalized dipole magnitudes in the
            vtk functionality.
        R0: double.
            The value of the major radius of the stellarator needed only for simple toroidal
            coordinates.
    """

    def __init__(self, dipole_grid, dipole_vectors, stellsym=True, nfp=1, coordinate_flag='cartesian', m_maxima=None, R0=1):
        super().__init__()
        if coordinate_flag == 'toroidal':
            warnings.warn('Note that if using simple toroidal coordinates, '
                          'the major radius must be specified through R0 argument.')
        self.R0 = R0
        self._dipole_fields_from_symmetries(dipole_grid, dipole_vectors, stellsym, nfp, coordinate_flag, m_maxima, R0)

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        B[:] = sopp.dipole_field_B(points, self.dipole_grid, self.m_vec)

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = sopp.dipole_field_dB(points, self.dipole_grid, self.m_vec)

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        A[:] = sopp.dipole_field_A(points, self.dipole_grid, self.m_vec)

    def _dA_by_dX_impl(self, dA):
        points = self.get_points_cart_ref()
        dA[:] = sopp.dipole_field_dA(points, self.dipole_grid, self.m_vec)

    def _dipole_fields_from_symmetries(self, dipole_grid, dipole_vectors, stellsym=True, nfp=1, coordinate_flag='cartesian', m_maxima=None, R0=1):
        """
        Takes the dipoles and grid initialized in a PermanentMagnetOptimizer (for a half-period surface)
        and generates the full dipole manifold so that the call to B() (the magnetic field from
        the dipoles) correctly returns contributions from all the dipoles from symmetries.
        """
        self.dipole_grid = dipole_grid

        # Read in the required fields from pm_opt object
        ndipoles = dipole_grid.shape[0]
        if m_maxima is None:
            m_maxima = np.max(np.linalg.norm(dipole_vectors, axis=-1)) * np.ones(ndipoles)
        if stellsym:
            stell_list = [1, -1]
            nsym = nfp * 2
        else:
            stell_list = [1]
            nsym = nfp
        m = dipole_vectors.reshape(ndipoles, 3)

        # Initialize new grid and dipole vectors for all the dipoles
        # after we account for the symmetries below.
        dipole_grid_x = np.zeros(ndipoles * nsym)
        dipole_grid_y = np.zeros(ndipoles * nsym)
        dipole_grid_z = np.zeros(ndipoles * nsym)
        m_vec = np.zeros((ndipoles * nsym, 3))
        m_max = np.zeros(ndipoles * nsym)

        # Load in the dipole locations for a half-period surface
        ox = dipole_grid[:, 0]
        oy = dipole_grid[:, 1]
        oz = dipole_grid[:, 2]

        # loop through the dipoles and repeat for fp and stellarator symmetries
        index = 0
        n = ndipoles

        # get the components in Cartesian, converting if needed
        mmx = m[:, 0]
        mmy = m[:, 1]
        mmz = m[:, 2]
        if coordinate_flag == 'cylindrical':
            phi_dipole = np.arctan2(oy, ox)
            mmx_temp = mmx * np.cos(phi_dipole) - mmy * np.sin(phi_dipole)
            mmy_temp = mmx * np.sin(phi_dipole) + mmy * np.cos(phi_dipole)
            mmx = mmx_temp
            mmy = mmy_temp
        if coordinate_flag == 'toroidal':
            phi_dipole = np.arctan2(oy, ox)
            theta_dipole = np.arctan2(oz, np.sqrt(ox ** 2 + oy ** 2) - R0)
            mmx_temp = mmx * np.cos(phi_dipole) * np.cos(theta_dipole) - mmy * np.sin(phi_dipole) - mmz * np.cos(phi_dipole) * np.sin(theta_dipole)
            mmy_temp = mmx * np.sin(phi_dipole) * np.cos(theta_dipole) + mmy * np.cos(phi_dipole) - mmz * np.sin(phi_dipole) * np.sin(theta_dipole)
            mmz_temp = mmx * np.sin(theta_dipole) + mmz * np.cos(theta_dipole)
            mmx = mmx_temp
            mmy = mmy_temp
            mmz = mmz_temp

        # Loop over stellarator and field-period symmetry contributions
        for stell in stell_list:
            for fp in range(nfp):
                phi0 = (2 * np.pi / nfp) * fp

                # get new dipoles locations by flipping the y and z components, then rotating by phi0
                dipole_grid_x[index:index + n] = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
                dipole_grid_y[index:index + n] = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
                dipole_grid_z[index:index + n] = oz * stell

                # get new dipole vectors by flipping the x component, then rotating by phi0
                m_vec[index:index + n, 0] = mmx * np.cos(phi0) * stell - mmy * np.sin(phi0)
                m_vec[index:index + n, 1] = mmx * np.sin(phi0) * stell + mmy * np.cos(phi0)
                m_vec[index:index + n, 2] = mmz

                m_max[index:index + n] = m_maxima
                index += n

        contig = np.ascontiguousarray
        self.dipole_grid = contig(np.array([dipole_grid_x, dipole_grid_y, dipole_grid_z]).T)
        self.m_vec = contig(m_vec)
        self.m_maxima = contig(m_max)

    def _toVTK(self, vtkname):
        """
            Write dipole data into a VTK file (acknowledgements to Caoxiang's CoilPy code).

        Args:
            vtkname (str): VTK filename, will be appended with .vts or .vtu.
        """

        # get the coordinates
        ox = np.ascontiguousarray(self.dipole_grid[:, 0])
        oy = np.ascontiguousarray(self.dipole_grid[:, 1])
        oz = np.ascontiguousarray(self.dipole_grid[:, 2])
        ophi = np.arctan2(oy, ox)
        otheta = np.arctan2(oz, np.sqrt(ox ** 2 + oy ** 2) - self.R0)

        # define the m vectors and the normalized m vectors
        # in Cartesian, cylindrical, and simple toroidal coordinates.
        mx = np.ascontiguousarray(self.m_vec[:, 0])
        my = np.ascontiguousarray(self.m_vec[:, 1])
        mz = np.ascontiguousarray(self.m_vec[:, 2])
        mx_normalized = np.ascontiguousarray(mx / self.m_maxima)
        my_normalized = np.ascontiguousarray(my / self.m_maxima)
        mz_normalized = np.ascontiguousarray(mz / self.m_maxima)
        mr = np.ascontiguousarray(mx * np.cos(ophi) + my * np.sin(ophi))
        mrminor = np.ascontiguousarray(mx * np.cos(ophi) * np.cos(otheta) + my * np.sin(ophi) * np.cos(otheta) + np.sin(otheta) * mz)
        mphi = np.ascontiguousarray(-mx * np.sin(ophi) + my * np.cos(ophi))
        mtheta = np.ascontiguousarray(-mx * np.cos(ophi) * np.sin(otheta) - my * np.sin(ophi) * np.sin(otheta) + np.cos(otheta) * mz)
        mr_normalized = np.ascontiguousarray(mr / self.m_maxima)
        mrminor_normalized = np.ascontiguousarray(mrminor / self.m_maxima)
        mphi_normalized = np.ascontiguousarray(mphi / self.m_maxima)
        mtheta_normalized = np.ascontiguousarray(mtheta / self.m_maxima)

        # Save all the data to a vtk file which can be visualized nicely with ParaView
        data = {"m": (mx, my, mz), "m_normalized": (mx_normalized, my_normalized, mz_normalized), "m_rphiz": (mr, mphi, mz), "m_rphiz_normalized": (mr_normalized, mphi_normalized, mz_normalized), "m_rphitheta": (mrminor, mphi, mtheta), "m_rphitheta_normalized": (mrminor_normalized, mphi_normalized, mtheta_normalized)}
        from pyevtk.hl import pointsToVTK
        pointsToVTK(str(vtkname), ox, oy, oz, data=data)


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

    @property
    def mn(self):
        return np.column_stack((self.m, self.n))

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        mn = decoder.process_decoded(d["mn"], serial_objs_dict, recon_objs)
        field = cls(mn, d["coeffs"])
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


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

    def as_dict(self, serial_objs_dict):
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["iota0"], d["iota1"], d["k"], d["epsilonk"], d["m0"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class UniformInterpolationRule(sopp.UniformInterpolationRule):
    pass


class ChebyshevInterpolationRule(sopp.ChebyshevInterpolationRule):
    pass


class InterpolatedField(sopp.InterpolatedField, MagneticField):
    r"""
    This field takes an existing field and interpolates it on a regular grid in :math:`r,\phi,z`.
    This resulting interpolant can then be evaluated very quickly.
    """

    def __init__(self, field, degree, rrange, phirange, zrange, extrapolate=True, nfp=1, stellsym=False, skip=None):
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
            skip: a function that takes in a point (in cylindrical (r,phi,z)
                  coordinates) and returns whether to skip that location when
                  building the interpolant or not. The signature should be

                  .. code-block:: Python

                      def skip(r: double, phi: double, z: double) -> bool:
                          ...

                  See also here
                  https://github.com/hiddenSymmetries/simsopt/pull/227 for a
                  graphical illustration.

        """
        MagneticField.__init__(self)
        if stellsym and zrange[0] != 0:
            logger.warning(fr"Sure about zrange[0]={zrange[0]}? When exploiting stellarator symmetry, the interpolant is never evaluated for z<0.")
        if nfp > 1 and abs(phirange[1] - 2*np.pi/nfp) > 1e-14:
            logger.warning(fr"Sure about phirange[1]={phirange[1]}? When exploiting rotational symmetry, the interpolant is never evaluated for phi>2\pi/nfp.")

        if skip is None:
            def skip(xs, ys, zs):
                return [False for _ in xs]

        sopp.InterpolatedField.__init__(self, field, degree, rrange, phirange, zrange, extrapolate, nfp, stellsym, skip)
        self.__field = field

    def to_vtk(self, filename):
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


class MirrorModel(MagneticField):
    r"""
    Model magnetic field employed in https://arxiv.org/abs/2305.06372 to study
    the magnetic mirror experiment WHAM. The
    magnetic field is given by :math:`\vec{B}=B_R \vec{e}_R + B_Z \vec{e}_Z`, where 
    :math:`\vec{e}_R` and :math:`\vec{e}_Z` are
    the cylindrical radial and axial unit vectors, respectively, and
    :math:`B_R` and :math:`B_Z` are given by

    .. math::

        B_R = -\frac{1}{R} \frac{\partial\psi}{\partial Z}, \; B_Z = \frac{1}{R}.
        \frac{\partial\psi}{\partial R}

    In this model, the magnetic flux function :math:`\psi` is written as a double
    Lorentzian function

    .. math::

        \psi = \frac{R^2 \mathcal{B}}{2 \pi \gamma}\left(\left[1+\left(\frac{Z-Z_m}{\gamma}\right)^2\right]^{-1}+\left[1+\left(\frac{Z+Z_m}{\gamma}\right)^2\right]^{-1}\right).

    Note that this field is neither a vacuum field nor a solution of MHD force balance.
    The input parameters are ``B0``, ``gamma`` and ``Z_m`` with the standard values the
    ones used in https://arxiv.org/abs/2305.06372, that is, ``B0 = 6.51292``,
    ``gamma = 0.124904``, and ``Z_m = 0.98``.

    Args:
        B0:  parameter :math:`\mathcal{B}` of the flux surface function
        gamma:  parameter :math:`\gamma` of the flux surface function
        Z_m:  parameter :math:`Z_m` of the flux surface function
    """

    def __init__(self, B0=6.51292, gamma=0.124904, Z_m=0.98):
        MagneticField.__init__(self)
        self.B0 = B0
        self.gamma = gamma
        self.Z_m = Z_m

    def _psi(self, R, Z):
        factor1 = 1+((Z-self.Z_m)/(self.gamma))**2
        factor2 = 1+((Z+self.Z_m)/(self.gamma))**2
        psi = (R*R*self.B0/(2*np.pi*self.gamma))*(1/factor1+1/factor2)
        return psi

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        # BR = -(1/R)dpsi/dZ, BZ=(1/R)dpsi/dR
        factor1 = (1+((z-self.Z_m)/(self.gamma))**2)**2
        factor2 = (1+((z+self.Z_m)/(self.gamma))**2)**2
        Br = (r*self.B0/(np.pi*self.gamma**3))*((z-self.Z_m)/factor1+(z+self.Z_m)/factor2)
        Bz = self._psi(r, z)*2/r/r
        B[:, 0] = Br * np.cos(phi)
        B[:, 1] = Br * np.sin(phi)
        B[:, 2] = Bz

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])

        factor1 = (1+((z-self.Z_m)/(self.gamma))**2)**2
        factor2 = (1+((z+self.Z_m)/(self.gamma))**2)**2
        Br = (r*self.B0/(np.pi*self.gamma**3))*((z-self.Z_m)/factor1+(z+self.Z_m)/factor2)
        # Bz = self._psi(r,z)*2/r/r
        dBrdr = (self.B0/(np.pi*self.gamma**3))*((z-self.Z_m)/factor1+(z+self.Z_m)/factor2)
        dBzdz = -2*dBrdr
        dBrdz = (self.B0*r/(np.pi*self.gamma**3))*(1/factor1+1/factor2
                                                   - 4*self.gamma**4*((z-self.Z_m)**2/((z-self.Z_m)**2+self.gamma**2)**3+(z+self.Z_m)**2/((z+self.Z_m)**2+self.gamma**2)**3))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        dcosphidx = -points[:, 0]**2/r**3 + 1/r
        dsinphidx = -points[:, 0]*points[:, 1]/r**3
        dcosphidy = -points[:, 0]*points[:, 1]/r**3
        dsinphidy = -points[:, 1]**2/r**3 + 1/r
        drdx = points[:, 0]/r
        drdy = points[:, 1]/r
        dBxdx = dBrdr*drdx*cosphi + Br*dcosphidx
        dBxdy = dBrdr*drdy*cosphi + Br*dcosphidy
        dBxdz = dBrdz*cosphi
        dBydx = dBrdr*drdx*sinphi + Br*dsinphidx
        dBydy = dBrdr*drdy*sinphi + Br*dsinphidy
        dBydz = dBrdz*sinphi

        dB[:, 0, 0] = dBxdx
        dB[:, 1, 0] = dBxdy
        dB[:, 2, 0] = dBxdz
        dB[:, 0, 1] = dBydx
        dB[:, 1, 1] = dBydy
        dB[:, 2, 1] = dBydz
        dB[:, 0, 2] = 0
        dB[:, 1, 2] = 0
        dB[:, 2, 2] = dBzdz

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["B0"], d["gamma"], d["Z_m"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field
