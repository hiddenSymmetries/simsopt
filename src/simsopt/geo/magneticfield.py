import numpy as np
from simsopt.geo.poincare import compute_poincare, plot_poincare


class MagneticField():

    '''
    Generic class that represents any magnetic field for which each magnetic
    field class inherits.
    '''

    def clear_cached_properties(self):
        self._B = None
        self._dB_by_dX = None
        self._d2B_by_dXdX = None
        self._A = None
        self._dA_by_dX = None
        self._d2A_by_dXdX = None

    def set_points(self, points):
        self.points = np.array(points)
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

    def d2B_by_dXdX(self, compute_derivatives=2):
        if self._d2B_by_dXdX is None:
            assert compute_derivatives >= 2
            self.compute(self.points, compute_derivatives)
        return self._d2B_by_dXdX

    def A(self, compute_derivatives=0):
        if self._A is None:
            assert compute_derivatives >= 0
            self.compute_A(self.points, compute_derivatives)
        return self._A

    def dA_by_dX(self, compute_derivatives=1):
        if self._dA_by_dX is None:
            assert compute_derivatives >= 1
            self.compute_A(self.points, compute_derivatives)
        return self._dA_by_dX

    def d2A_by_dXdX(self, compute_derivatives=2):
        if self._d2A_by_dXdX is None:
            assert compute_derivatives >= 2
            self.compute_A(self.points, compute_derivatives)
        return self._d2A_by_dXdX

    def compute(self, points, compute_derivatives=0):
        raise NotImplementedError('Computation of B field not implemented. Needs to be overwritten by childclass.')

    def compute_A(self, points, compute_derivatives=0):
        raise NotImplementedError('Computation of potential A not implemented. Needs to be overwritten by childclass.')

    def __add__(self, other):
        return MagneticFieldSum([self, other])

    def __mul__(self, other):
        return MagneticFieldMultiply(other, self)

    def __rmul__(self, other):
        return MagneticFieldMultiply(other, self)

    def BR_Bphi_BZ(self):
        Bxyz = self.B()
        phi = np.arctan2(self.points[:, 1], self.points[:, 0])
        B_phi = np.cos(phi)*Bxyz[:, 1] - np.sin(phi)*Bxyz[:, 0]
        B_r = np.cos(phi)*Bxyz[:, 0] + np.sin(phi)*Bxyz[:, 1]
        B_z = Bxyz[:, 2]
        return np.vstack((B_r, B_phi, B_z)).T

    def dB_by_drphiz(self):
        phi = np.arctan2(self.points[:, 1], self.points[:, 0])
        x = self.points[:, 0]
        y = self.points[:, 1]
        dB = self.dB_by_dX()
        B = self.B()
        drBr = np.cos(phi)*(np.cos(phi)*dB[:, 0, 0]+np.sin(phi)*dB[:, 0, 1])+np.sin(phi)*(np.cos(phi)*dB[:, 1, 0]+np.sin(phi)*dB[:, 1, 1])
        dphiBr = -np.sin(phi)*B[:, 0]+np.cos(phi)*B[:, 1]+x*(-np.sin(phi)*dB[:, 0, 0]+np.cos(phi)*dB[:, 0, 1])+y*(-np.sin(phi)*dB[:, 1, 0]+np.cos(phi)*dB[:, 1, 1])
        dzBr = np.cos(phi)*dB[:, 0, 2]+np.sin(phi)*dB[:, 1, 2]
        drBphi = np.cos(phi)*(np.cos(phi)*dB[:, 1, 0]+np.sin(phi)*dB[:, 1, 1])-np.sin(phi)*(np.cos(phi)*dB[:, 0, 0]+np.sin(phi)*dB[:, 0, 1])
        dphiBphi = -np.sin(phi)*B[:, 1]-np.cos(phi)*B[:, 0]+x*(-np.sin(phi)*dB[:, 1, 0]+np.cos(phi)*dB[:, 1, 1])-y*(-np.sin(phi)*dB[:, 0, 0]+np.cos(phi)*dB[:, 0, 1])
        dzBphi = np.cos(phi)*dB[:, 1, 2]-np.sin(phi)*dB[:, 0, 2]
        drBz = np.cos(phi)*dB[:, 2, 0]+np.sin(phi)*dB[:, 2, 1]
        dphiBz = -y*dB[:, 2, 0]+x*dB[:, 2, 1]
        dzBz = dB[:, 2, 2]
        return np.transpose(np.array([[drBr, drBphi, dzBr], [dphiBr, dphiBphi, dphiBz], [drBz, dzBphi, dzBz]]), [2, 1, 0])

    def d_RZ_d_phi(self, phi, RZ):
        R = RZ[0]
        Z = RZ[1]
        x = R*np.cos(phi)
        y = R*np.sin(phi)
        self.set_points([[x, y, Z]])
        BR, Bphi, BZ = self.BR_Bphi_BZ()[0]
        return [R * BR / Bphi, R * BZ / Bphi]

    def compute_poincare(self, R0, Z0, NFP=1, npoints=20, rtol=1e-6, atol=1e-9):
        self.Poincare_data = compute_poincare(self, R0, Z0, NFP=NFP, npoints=npoints, rtol=rtol, atol=atol)
        return self.Poincare_data

    def plot_poincare(self, R0, Z0, NFP=1, npoints=20, rtol=1e-6, atol=1e-9, marker_size=2, extra_str=""):
        self.Poincare_data = compute_poincare(self, R0, Z0, NFP=NFP, npoints=npoints, rtol=rtol, atol=atol)
        plot_poincare(self.Poincare_data, marker_size=2, extra_str="")


class MagneticFieldMultiply(MagneticField):
    '''
    Class used to multiply a magnetic field by a scalar.
    It takes as input a MagneticField class and a scalar and multiplies B, A and their derivatives by that value
    '''

    def __init__(self, scalar, Bfield):
        self.scalar = scalar
        self.Bfield = Bfield

    def set_points(self, points):
        self.points = np.array(points)
        self.clear_cached_properties()
        self.Bfield.set_points(points)
        return self

    def compute(self, points, compute_derivatives=0):
        self.Bfield.compute(points, compute_derivatives)
        self._B = self.scalar*self.Bfield._B
        if compute_derivatives >= 1:
            self._dB_by_dX = self.scalar*self.Bfield._dB_by_dX
        if compute_derivatives >= 2:
            self._d2B_by_dXdX = self.scalar*self.Bfield._d2B_by_dXdX

    def compute_A(self, points, compute_derivatives=0):
        self.Bfield.compute_A(points, compute_derivatives)
        self._A = self.scalar*self.Bfield._A
        if compute_derivatives >= 1:
            self._dA_by_dX = self.scalar*self.Bfield._dA_by_dX
        if compute_derivatives >= 2:
            self._d2A_by_dXdX = self.scalar*self.Bfield._d2A_by_dXdX


class MagneticFieldSum(MagneticField):
    '''
    Class used to sum two or more magnetic field together.  It can either be
    called directly with a list of magnetic fields given as input and outputing
    another magnetic field with B, A and its derivatives added together or it
    can be called by summing magnetic fields classes as Bfield1 + Bfield1
    '''

    def __init__(self, Bfields):
        self.Bfields = Bfields

    def set_points(self, points):
        self.points = np.array(points)
        self.clear_cached_properties()
        [Bfield.set_points(points) for Bfield in self.Bfields]
        return self

    def compute(self, points, compute_derivatives=0):
        [Bfield.compute(points, compute_derivatives) for Bfield in self.Bfields]
        self._B = np.sum([Bfield._B for Bfield in self.Bfields], 0)
        if compute_derivatives >= 1:
            self._dB_by_dX = np.sum([Bfield._dB_by_dX for Bfield in self.Bfields], 0)
        if compute_derivatives >= 2:
            self._d2B_by_dXdX = np.sum([Bfield._d2B_by_dXdX for Bfield in self.Bfields], 0)

    def compute_A(self, points, compute_derivatives=0):
        [Bfield.compute_A(points, compute_derivatives) for Bfield in self.Bfields]
        self._A = np.sum([Bfield._A for Bfield in self.Bfields], 0)
        if compute_derivatives >= 1:
            self._dA_by_dX = np.sum([Bfield._dA_by_dX for Bfield in self.Bfields], 0)
        if compute_derivatives >= 2:
            self._d2A_by_dXdX = np.sum([Bfield._d2A_by_dXdX for Bfield in self.Bfields], 0)
