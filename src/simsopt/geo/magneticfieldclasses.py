from simsopt.geo.magneticfield import MagneticField
from scipy.special import ellipk, ellipe
import numpy as np
try:
    from sympy.parsing.sympy_parser import parse_expr
    import sympy as sp
    sympy_found = True
except:
    sympy_found = False
    pass

class ToroidalField(MagneticField):
    '''Magnetic field purely in the toroidal direction, that is, in the phi direction with (R,phi,Z) the standard cylindrical coordinates.
       Its modulus is given by B = B0*R0/R where R0 is the first input and B0 the second input to the function.    

    Args:
        B0:  modulus of the magnetic field at R0
        R0:  radius of normalization
    '''
    def __init__(self, R0, B0):
        self.R0=R0
        self.B0=B0

    def compute(self, points, compute_derivatives=0):
        phi = np.arctan2(points[:,1],points[:,0])
        R   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi),R),np.divide(np.cos(phi),R),np.zeros(len(phi)))).T
        self._B = np.multiply(self.B0*self.R0,phiUnitVectorOverR)

        x=points[:,0]
        y=points[:,1]

        if compute_derivatives >= 1:

            dB_by_dX1=np.vstack((
                np.multiply(np.divide(self.B0*self.R0,R**4),2*np.multiply(x,y)),
                np.multiply(np.divide(self.B0*self.R0,R**4),y**2-x**2),
                0*R))
            dB_by_dX2=np.vstack((
                np.multiply(np.divide(self.B0*self.R0,R**4),y**2-x**2),
                np.multiply(np.divide(self.B0*self.R0,R**4),-2*np.multiply(x,y)),
                0*R))
            dB_by_dX3=np.vstack((0*R,0*R,0*R))

            dToroidal_by_dX=np.array([dB_by_dX1,dB_by_dX2,dB_by_dX3]).T

            self._dB_by_dX = dToroidal_by_dX

        if compute_derivatives >= 2:
            self._d2B_by_dXdX = None # To be implemented

        return self

class ScalarPotentialRZMagneticField(MagneticField):
    '''Vacuum magnetic field as a solution of B = grad(Phi) where Phi is the magnetic field scalar potential.
       It takes Phi as an input string, which should contain an expression involving the standard cylindrical coordinates (R, phi, Z)
       Example: ScalarPotentialRZMagneticField("2*phi") yields a magnetic field B = grad(2*phi) = (0,2/R,0).
       Note: this function needs sympy.    

    Args:
        PhiStr:  string containing vacuum scalar potential expression as a function of R, Z and phi
    '''
    def __init__(self, PhiStr):
        if not sympy_found:
            raise RuntimeError("Sympy is required for the ScalarPotentialRZMagneticField class")
        self.PhiStr = PhiStr
        self.Phiparsed = parse_expr(PhiStr)
        R,Z,Phi = sp.symbols('R Z phi')
        self.Blambdify = sp.lambdify((R, Z, Phi), [self.Phiparsed.diff(R), self.Phiparsed.diff(Phi)/R, self.Phiparsed.diff(Z)])
        self.dBlambdify_by_dX = sp.lambdify((R, Z, Phi), 
                            [[sp.cos(Phi)*self.Phiparsed.diff(R).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),sp.cos(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)-(sp.sin(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),sp.cos(Phi)*self.Phiparsed.diff(Z).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
                             [sp.sin(Phi)*self.Phiparsed.diff(R).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),sp.sin(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)+(sp.cos(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),sp.sin(Phi)*self.Phiparsed.diff(Z).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
                             [self.Phiparsed.diff(R).diff(Z),(self.Phiparsed.diff(Phi)/R).diff(Z),self.Phiparsed.diff(Z).diff(Z)]])
        
    def compute(self, points, compute_derivatives=0):
        r   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        z   = points[:,2]
        phi = np.arctan2(points[:,1],points[:,0])
        self._B = [self.Blambdify(r[i],z[i],phi[i]) for i in range(len(r))]

        if compute_derivatives >= 1:
            self._dB_by_dX = [self.dBlambdify_by_dX(r[i],z[i],phi[i]) for i in range(len(r))]


class CircularCoilXY(MagneticField):
    '''Magnetic field created by a single circular coil in the xy plane evaluated using analytical functions, including complete elliptic integrals of the first and second kind.
    As inputs, it takes the radius of the coil (r0), its center and current (I).
    Note: this function needs scipy.

    Args:
        r0: radius of the coil
        center: point at the coil center
        I: current of the coil in Ampere's
    '''
    def __init__(self, r0=0.1, center=[0,0,0], I=5e5/np.pi):
        self.r0     = r0
        self.Inorm  = I*4e-7
        self.center = center

    def compute(self, points, compute_derivatives=0):
        points = np.array([np.subtract(point,self.center) for point in points])
        rho   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        r     = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2) + np.power(points[:,2],2))
        alpha = np.sqrt(self.r0**2 + np.power(r,2) - 2*self.r0*rho)
        beta  = np.sqrt(self.r0**2 + np.power(r,2) + 2*self.r0*rho)
        k     = np.sqrt(1-np.divide(np.power(alpha,2),np.power(beta,2)))
        self._B = np.array([
            [self.Inorm*point[0]*point[2]/(2*alpha[i]**2*beta[i]*rho[i]**2)*((self.r0**2+r[i]**2)*ellipe(k[i]**2)-alpha[i]**2*ellipk(k[i]**2)),
             self.Inorm*point[1]*point[2]/(2*alpha[i]**2*beta[i]*rho[i]**2)*((self.r0**2+r[i]**2)*ellipe(k[i]**2)-alpha[i]**2*ellipk(k[i]**2)),
             self.Inorm/(2*alpha[i]**2*beta[i])*((self.r0**2-r[i]**2)*ellipe(k[i]**2)+alpha[i]**2*ellipk(k[i]**2))]
            for i,point in enumerate(points)])

## Next magnetic field classes to implement
# class ReimanModel(MagneticField):
# class DommaschkPotential(MagneticField):
# class PointDipole(MagneticField): (equation 14 of https://arxiv.org/pdf/2009.06535)