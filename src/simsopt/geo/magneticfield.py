from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.biotsavart import BiotSavart
import simsgeopp as sgpp
import numpy as np

class MagneticField():

	def clear_cached_properties(self):
		self._B = None
		self._dB_by_dX = None

	def set_points(self, points):
		self.points = points
		self.clear_cached_properties()
		return self

	def B(self, compute_derivatives=0):
		if self._B is None:
			self.compute(self.points, compute_derivatives)
		return self._B

	def dB_by_dX(self, compute_derivatives=1):
		if self._dB_by_dX is None:
			assert compute_derivatives >= 1
			self.compute(self.points, compute_derivatives)
		return self._dB_by_dX

	def compute(self, points, compute_derivatives):
		self._B = None
		return self

class HelicalField(MagneticField):

	def __init__(self, m0=5, l0=2, B0=1, R0=1, r=0.3, I=[-0.021 * 1e7, 0.021 * 1e7], A=[np.pi/2, 0], ppp=20):
		self.nHcoils = len(A)
		self.m0 = m0
		self.nfp = m0
		self.l0 = l0
		self.B0 = B0
		self.R0 = R0
		self.r = r
		self.I = np.array(I)
		self.A = np.array(A)
		self.ppp = ppp
		self.update()
		
	def update(self):
		coil_data=np.zeros((self.l0+self.m0+1,self.nHcoils*6))
		for i in range(self.nHcoils):
			coil_data[self.m0+self.l0,0+6*i]=-self.r*np.sin(self.A[i])/2 # sin coeff of x
			coil_data[self.m0-self.l0,0+6*i]=-self.r*np.sin(self.A[i])/2 # sin coeff of x
			coil_data[self.l0,1+6*i]=self.R0							 # cos coeff of x
			coil_data[self.m0+self.l0,1+6*i]= self.r*np.cos(self.A[i])/2 # cos coeff of x
			coil_data[self.m0-self.l0,1+6*i]= self.r*np.cos(self.A[i])/2 # cos coeff of x
			coil_data[self.l0,2+6*i]=self.R0							 # sin coeff of y
			coil_data[self.m0+self.l0,2+6*i]= self.r*np.cos(self.A[i])/2 # sin coeff of y
			coil_data[self.m0-self.l0,2+6*i]=-self.r*np.cos(self.A[i])/2 # sin coeff of y
			coil_data[self.m0+self.l0,3+6*i]= self.r*np.sin(self.A[i])/2 # cos coeff of y
			coil_data[self.m0-self.l0,3+6*i]=-self.r*np.sin(self.A[i])/2 # cos coeff of y
			coil_data[self.m0,4+6*i]        =-self.r*np.cos(self.A[i])   # sin coeff of z
			coil_data[self.m0,5+6*i]        =-self.r*np.sin(self.A[i])   # cos coeff of z

		Nt_coils=len(coil_data)-1
		num_coils = int(len(coil_data[0])/6)
		self.coils = [CurveXYZFourier(Nt_coils*self.ppp, Nt_coils) for i in range(num_coils)]
		for ic in range(num_coils):
			dofs = self.coils[ic].dofs
			dofs[0][0] = coil_data[0, 6*ic + 1]
			dofs[1][0] = coil_data[0, 6*ic + 3]
			dofs[2][0] = coil_data[0, 6*ic + 5]
			for io in range(0, Nt_coils):
				dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
				dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
				dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
				dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
				dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
				dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
			self.coils[ic].set_dofs(np.concatenate(dofs))
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
		return self

		if compute_derivatives >= 1:
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
			self._dB_by_dX = dToroidal_by_dX

# class ReimanModel(MagneticField):
# class DommaschkPotential(MagneticField):
# class CircularCoils(MagneticField):

### DEFINE METHODS TO ADD CLASSES
# class MagneticFieldSum(MagneticField):
#     def __add__(self, other: MagneticField) -> MagneticField:

#         return MagneticField(self._B1 + self._B2)
