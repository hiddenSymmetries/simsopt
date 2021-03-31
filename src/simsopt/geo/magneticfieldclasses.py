from simsopt.geo.magneticfield import MagneticField
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.biotsavart import BiotSavart
import simsgeopp as sgpp
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
            self._d2B_by_dXdX

        return self

## Next magnetic field classes to work on
# class ReimanModel(MagneticField):
# class DommaschkPotential(MagneticField):
# class CircularCoils(MagneticField):
# class PointDipole(MagneticField):

## Method to add magnetic fields
# class MagneticFieldSum(MagneticField):
#     def __add__(self, other: MagneticField) -> MagneticField:

#         return MagneticField(self._B1 + self._B2)
