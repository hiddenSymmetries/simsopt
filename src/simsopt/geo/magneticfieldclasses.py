from simsopt.geo.magneticfield import MagneticField
from sympy.parsing.sympy_parser import parse_expr
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.biotsavart import BiotSavart
import simsgeopp as sgpp
import sympy as sp
import numpy as np

class ToroidalField(MagneticField):

    def __init__(self, R0, B0):
        self.R0=R0
        self.B0=B0

    def compute(self, points, compute_derivatives):
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

    def __init__(self, PhiStr):
        self.PhiStr = PhiStr
        self.Phiparsed = parse_expr(PhiStr)
        R,Z,Phi = sp.symbols('R Z phi')
        self.Blambdify = sp.lambdify((R, Z, Phi), [self.Phiparsed.diff(R), self.Phiparsed.diff(Phi)/R, self.Phiparsed.diff(Z)])
        self.dBlambdify_by_dX = sp.lambdify((R, Z, Phi), 
                            [[sp.cos(Phi)*self.Phiparsed.diff(R).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),sp.cos(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)-(sp.sin(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),sp.cos(Phi)*self.Phiparsed.diff(Z).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
                             [sp.sin(Phi)*self.Phiparsed.diff(R).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),sp.sin(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)+(sp.cos(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),sp.sin(Phi)*self.Phiparsed.diff(Z).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
                             [self.Phiparsed.diff(R).diff(Z),(self.Phiparsed.diff(Phi)/R).diff(Z),self.Phiparsed.diff(Z).diff(Z)]])
        
    def compute(self, points, compute_derivatives):
        r   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        z   = points[:,2]
        phi = np.arctan2(points[:,1],points[:,0])
        self._B = [self.Blambdify(r[i],z[i],phi[i]) for i in range(len(r))]

        if compute_derivatives >= 1:
            self._dB_by_dX = [self.dBlambdify_by_dX(r[i],z[i],phi[i]) for i in range(len(r))]

## Next magnetic field classes to work on
# class ReimanModel(MagneticField):
# class DommaschkPotential(MagneticField):
# class CircularCoils(MagneticField):
# class PointDipole(MagneticField):

## Method to add magnetic fields
# class MagneticFieldSum(MagneticField):
#     def __add__(self, other: MagneticField) -> MagneticField:

#         return MagneticField(self._B1 + self._B2)
