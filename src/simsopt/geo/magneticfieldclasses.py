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
        MagneticField.__init__(self)
        self.R0 = R0
        self.B0 = B0

    def B_impl(self, B):
        points = self.points
        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi), R), np.divide(np.cos(phi), R), np.zeros(len(phi)))).T
        B[:] = np.multiply(self.B0*self.R0, phiUnitVectorOverR)

    def dB_by_dX_impl(self, dB):
        points = self.points
        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi), R), np.divide(np.cos(phi), R), np.zeros(len(phi)))).T

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


    def d2B_by_dXdX_impl(self, ddB):
        points = self.points
        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi), R), np.divide(np.cos(phi), R), np.zeros(len(phi)))).T

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

    def A_impl(self, A):
        points = self.points
        A[:] = self.B0*self.R0*np.array([
            points[:, 2]*points[:, 0]/(points[:, 0]**2+points[:, 1]**2),
            points[:, 2]*points[:, 1]/(points[:, 0]**2+points[:, 1]**2),
            0*points[:, 2]]).T

    def dA_by_dX_impl(self, dA):
        points = self.points
        dA[:] = self.B0*self.R0*np.array((points[:, 2]/(points[:, 0]**2+points[:, 1]**2)**2)*np.array(
            [[-points[:, 0]**2+points[:, 1]**2, -2*points[:, 0]*points[:, 1], np.zeros((len(points)))],
             [-2*points[:, 0]*points[:, 1], points[:, 0]**2-points[:, 1]**2, np.zeros((len(points)))],
             [points[:, 0]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
              points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2], np.zeros((len(points)))]])).T

    def d2A_by_dXdX_impl(self, ddA):
        points = self.points
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


class ScalarPotentialRZMagneticField(MagneticField):
    """
    Vacuum magnetic field as a solution of B = grad(Phi) where Phi is the
    magnetic field scalar potential.  It takes Phi as an input string, which
    should contain an expression involving the standard cylindrical coordinates
    (R, phi, Z) Example: ScalarPotentialRZMagneticField("2*phi") yields a
    magnetic field B = grad(2*phi) = (0,2/R,0).  Note: this function needs
    sympy.

    Args:
        PhiStr:  string containing vacuum scalar potential expression as a function of R, Z and phi
    """

    def __init__(self, PhiStr):
        MagneticField.__init__(self)
        if not sympy_found:
            raise RuntimeError("Sympy is required for the ScalarPotentialRZMagneticField class")
        self.PhiStr = PhiStr
        self.Phiparsed = parse_expr(PhiStr)
        R, Z, Phi = sp.symbols('R Z phi')
        self.Blambdify = sp.lambdify((R, Z, Phi), [self.Phiparsed.diff(R), self.Phiparsed.diff(Phi)/R, self.Phiparsed.diff(Z)])
        self.dBlambdify_by_dX = sp.lambdify(
            (R, Z, Phi),
            [[sp.cos(Phi)*self.Phiparsed.diff(R).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),
              sp.cos(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)-(sp.sin(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),
              sp.cos(Phi)*self.Phiparsed.diff(Z).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
             [sp.sin(Phi)*self.Phiparsed.diff(R).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),
              sp.sin(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)+(sp.cos(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),
              sp.sin(Phi)*self.Phiparsed.diff(Z).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
             [self.Phiparsed.diff(R).diff(Z)+1e-30*sp.sin(Phi), (self.Phiparsed.diff(Phi)/R).diff(Z), self.Phiparsed.diff(Z).diff(Z)+1e-30*sp.sin(Phi)]])

    def B_impl(self, B):
        points = self.points
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        B[:] = np.array(self.Blambdify(r, z, phi)).T

    def dB_by_dX_impl(self, dB):
        points = self.points
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
        if len(normal) == 2:
            self.normal = [normal[0], -normal[1]]
        else:
            self.normal = [np.arctan2(np.sqrt(normal[0]**2+normal[1]**2), normal[2]), -np.arctan2(normal[0], normal[1])]
        self.rotMatrix = np.array([[np.cos(self.normal[1]), np.sin(self.normal[0])*np.sin(self.normal[1]),
                                    np.cos(self.normal[0])*np.sin(self.normal[1])],
                                   [0, np.cos(self.normal[0]), -np.sin(self.normal[0])],
                                   [np.sin(self.normal[1]), np.sin(self.normal[0])*np.cos(self.normal[1]),
                                    np.cos(self.normal[0])*np.cos(self.normal[1])]])
        self.rotMatrixInv = np.array(self.rotMatrix.T)

    def B_impl(self, B):
        points = self.points
        points = np.array(np.dot(self.rotMatrix, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)
        gamma = np.square(points[:, 0]) - np.square(points[:, 1])
        B[:] = np.dot(self.rotMatrixInv, np.array(
            [self.Inorm*points[:, 0]*points[:, 2]/(2*alpha**2*beta*rho**2)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm*points[:, 1]*points[:, 2]/(2*alpha**2*beta*rho**2)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm/(2*alpha**2*beta)*((self.r0**2-r**2)*ellipek2+alpha**2*ellipkk2)])).T

    def dB_by_dX_impl(self, dB):
        points = self.points
        points = np.array(np.dot(self.rotMatrix, np.array(np.subtract(points, self.center)).T).T)
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

        dB[:] = np.array([
            [np.dot(self.rotMatrix[:, 0], np.dot(self.rotMatrixInv[0, :], dB_by_dXm)),
             np.dot(self.rotMatrix[:, 1], np.dot(self.rotMatrixInv[0, :], dB_by_dXm)),
             np.dot(self.rotMatrix[:, 2], np.dot(self.rotMatrixInv[0, :], dB_by_dXm))],
            [np.dot(self.rotMatrix[:, 0], np.dot(self.rotMatrixInv[1, :], dB_by_dXm)),
             np.dot(self.rotMatrix[:, 1], np.dot(self.rotMatrixInv[1, :], dB_by_dXm)),
             np.dot(self.rotMatrix[:, 2], np.dot(self.rotMatrixInv[1, :], dB_by_dXm))],
            [np.dot(self.rotMatrix[:, 0], np.dot(self.rotMatrixInv[2, :], dB_by_dXm)),
             np.dot(self.rotMatrix[:, 1], np.dot(self.rotMatrixInv[2, :], dB_by_dXm)),
             np.dot(self.rotMatrix[:, 2], np.dot(self.rotMatrixInv[2, :], dB_by_dXm))]]).T

    def A_impl(self, A):
        points = self.points
        points = np.array(np.dot(self.rotMatrix, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)

        A[:] = -self.Inorm/2*np.dot(self.rotMatrixInv, np.array(
            (2*self.r0*+np.sqrt(points[:, 0]**2+points[:, 1]**2)*ellipek2+(self.r0**2+points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)*(ellipe(k**2)-ellipkk2)) /
            ((points[:, 0]**2+points[:, 1]**2)*np.sqrt(self.r0**2+points[:, 0]**2+points[:, 1]**2+2*self.r0*np.sqrt(points[:, 0]**2+points[:, 1]**2)+points[:, 2]**2)) *
            np.array([-points[:, 1], points[:, 0], 0])).T)


class Dommaschk(MagneticField):
    """
    Vacuum magnetic field created by an explicit representation of the magnetic
    field scalar potential as proposed by W. Dommaschk (1986), Computer Physics
    Communications 40, 203-218 As inputs, it takes the arrays for the harmonics
    m, n and its corresponding coefficients

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

    def set_points(self, points):
        self.Btor.set_points(points)
        return MagneticField.set_points(self, points)

    def B_impl(self, B):
        points = self.points
        B[:] = np.add.reduce(sgpp.DommaschkB(self.m, self.n, self.coeffs, points))+self.Btor.B()

    def dB_by_dX_impl(self, dB):
        points = self.points
        dB[:] = np.add.reduce(sgpp.DommaschkdB(self.m, self.n, self.coeffs, points))+self.Btor.dB_by_dX()

class InterpolatedField(MagneticField):
    """
    Takes a magnetic field and interpolates it on a regular grid

        [rmin, rmax] x [phimin, phimax] x [zmin, zmax]

    using polynomials of order 1, 2, 3, or 4. The point of this class is that
    evaluation is much faster than evaluating e.g. a Biot Savart field every
    time. This is useful for things like Poincare plots or particle tracing.
    """

    def __init__(self, basefield, order=4, rmin=0.9, rmax=1.1, rsteps=8, phimin=0, phimax=2*np.pi, phisteps=8*32, zmin=-0.1, zmax=0.1, zsteps=8):
        self.basefield = basefield
        self.order = order
        self.rmin = rmin
        self.rmax = rmax
        self.rsteps = rsteps
        self.phimin = phimin
        self.phimax = phimax
        self.phisteps = phisteps
        self.zmin = zmin
        self.zmax = zmax
        self.zsteps = zsteps
        self._last_B = None

        import simsgeopp as sgpp
        if order == 1:
            self.Bh = sgpp.RegularGridInterpolant3D1((rmin, rmax, rsteps), (phimin, phimax, phisteps), (zmin, zmax, zsteps), 3)
        elif order == 4:
            self.Bh = sgpp.RegularGridInterpolant3D4((rmin, rmax, rsteps), (phimin, phimax, phisteps), (zmin, zmax, zsteps), 3)
        else:
            raise NotImplementedError('Only order 1 and 4 supported.')

        def bsfunbatch(r, phi, z):
            r = np.asarray(r)
            phi = np.asarray(phi)
            z = np.asarray(z)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            return self.basefield.set_points(np.vstack((x, y, z)).T).B().flatten()

        self.Bh.interpolate_batch(bsfunbatch)

    def compute(self, points, compute_derivatives=0):
        if compute_derivatives > 0:
            raise NotImplementedError('Only B supported.')
        if self._last_B is not None and self._last_B.shape == points.shape:
            self._B = self._last_B
        else:
            self._B = np.zeros_like(points)
        self.Bh.evaluate_batch_with_transform(points, self._B)
        self._last_B = self._B

    def estimate_error(self, n=1000):
        def bsfun(r, phi, z):
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            return self.basefield.set_points(np.asarray([[x, y, z]])).B()[0, :]
        return self.Bh.estimate_error(bsfun, n)
