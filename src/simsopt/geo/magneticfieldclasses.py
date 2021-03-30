from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.magneticfield import MagneticField
from simsopt.geo.biotsavart import BiotSavart
import simsgeopp as sgpp
import numpy as np


class HelicalField(MagneticField):

	def __init__(self, n0=5, l0=2, B0=1, R0=1, r0=0.3, I=[-0.0307 * 1e7, 0.0307 * 1e7], Aarr=[[0], [np.pi/2]], Barr=[[0], [0]], ppp=150):
		self.ncoils = len(Aarr)
		self.order = len(Aarr[0])
		self.n0 = n0
		self.nfp = n0
		self.l0 = l0
		self.B0 = B0
		self.R0 = R0
		self.r0 = r0
		self.I = np.array(I)
		self.Aarr = np.array(Aarr)
		self.Barr = np.array(Barr)
		self.ppp = ppp
		self.update()
		
	def update(self):
		self.coils = [CurveHelical(self.ppp, self.order, self.n0, self.l0, self.R0, self.r0) for i in range(self.ncoils)]
		for i in range(self.ncoils):
			self.coils[i].set_dofs(np.concatenate((self.Aarr[i],self.Barr[i])))
		self.biotsavart = BiotSavart(self.coils, self.I)

	def compute(self, points, compute_derivatives):
		# Helical magnetic field
		self.biotsavart.set_points(points)
		Bhelical = self.biotsavart.B()

		# Toroidal magnetic field
		phi = np.arctan2(points[:,1],points[:,0])
		R   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
		phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi),R),np.divide(np.cos(phi),R),np.zeros(len(phi)))).T
		Btoroidal = np.multiply(self.B0*self.R0,phiUnitVectorOverR)	

		# Sum the two
		self._B = Btoroidal + Bhelical

		if compute_derivatives >= 1:
			# Helical magnetic field derivatives
			dBHelical_by_dX = self.biotsavart.dB_by_dX()

			# Toroidal magnetic field derivatives
			x=points[:,0]
			y=points[:,1]
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

			# Sum the two
			self._dB_by_dX = dBHelical_by_dX + dToroidal_by_dX
		return self

class ToroidalField(MagneticField):

	def __init__(self, R0, B0):
		self.R0=R0
		self.B0=B0

	def compute(self, points, compute_derivatives):
		phi = np.arctan2(points[:,1],points[:,0])
		R   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
		phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi),R),np.divide(np.cos(phi),R),np.zeros(len(phi)))).T
		self._B = np.multiply(self.B0*self.R0,phiUnitVectorOverR)

		if compute_derivatives >= 1:
			x=points[:,0]
			y=points[:,1]
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
