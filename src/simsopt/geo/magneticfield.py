import numpy as np
import simsgeopp as sgpp


class MagneticField(sgpp.MagneticField):

    '''
    Generic class that represents any magnetic field for which each magnetic
    field class inherits.
    '''
    def __init__(self):
        sgpp.MagneticField.__init__(self)

    def clear_cached_properties(self):
        sgpp.MagneticField.invalidate_cache(self)

    def __add__(self, other):
        return MagneticFieldSum([self, other])

    def __mul__(self, other):
        return MagneticFieldMultiply(other, self)

    def __rmul__(self, other):
        return MagneticFieldMultiply(other, self)


class MagneticFieldMultiply(MagneticField):
    '''
    Class used to multiply a magnetic field by a scalar.
    It takes as input a MagneticField class and a scalar and multiplies B, A and their derivatives by that value
    '''

    def __init__(self, scalar, Bfield):
        MagneticField.__init__(self)
        self.scalar = scalar
        self.Bfield = Bfield

    def set_points(self, points):
        self.Bfield.set_points(points)
        return MagneticField.set_points(self, points)

    def B_impl(self, B):
        B[:] = self.scalar*self.Bfield.B()

    def dB_by_dX_impl(self, dB):
        dB[:] = self.scalar*self.Bfield.dB_by_dX()

    def d2B_by_dXdX_impl(self, ddB):
        ddB[:] = self.scalar*self.Bfield.d2B_by_dXdX()

    def A_impl(self, A):
        A[:] = self.scalar*self.Bfield.A()

    def dA_by_dX_impl(self, dA):
        dA[:] = self.scalar*self.Bfield.dA_by_dX()

    def d2A_by_dXdX_impl(self, ddA):
        ddA[:] = self.scalar*self.Bfield.d2A_by_dXdX()


class MagneticFieldSum(MagneticField):
    '''
    Class used to sum two or more magnetic field together.  It can either be
    called directly with a list of magnetic fields given as input and outputing
    another magnetic field with B, A and its derivatives added together or it
    can be called by summing magnetic fields classes as Bfield1 + Bfield1
    '''

    def __init__(self, Bfields):
        MagneticField.__init__(self)
        self.Bfields = Bfields

    def set_points(self, points):
        for bf in self.Bfields:
            bf.set_points(points)
        return MagneticField.set_points(self, points)

    def B_impl(self, B):
        B[:] = np.sum([bf.B() for bf in self.Bfields], axis=0)

    def dB_by_dX_impl(self, dB):
        dB[:] = np.sum([bf.dB_by_dX() for bf in self.Bfields], axis=0)

    def d2B_by_dXdX_impl(self, ddB):
        ddB[:] = np.sum([bf.d2B_by_dXdX() for bf in self.Bfields], axis=0)

    def A_impl(self, A):
        A[:] = np.sum([bf.A() for bf in self.Bfields], axis=0)

    def dA_by_dX_impl(self, dA):
        dA[:] = np.sum([bf.dA_by_dX() for bf in self.Bfields], axis=0)

    def d2A_by_dXdX_impl(self, ddA):
        ddA[:] = np.sum([bf.d2A_by_dXdX() for bf in self.Bfields], axis=0)
