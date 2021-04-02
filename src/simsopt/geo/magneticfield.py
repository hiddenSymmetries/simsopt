import numpy as np

class MagneticField():
    '''Generic class that represents any magnetic field for which each magnetic field class inherits.'''
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

    def compute(self, points, compute_derivatives=0):
        self._B = None
        if compute_derivatives >= 1:
            self._dB_by_dX = None
        if compute_derivatives >= 2:
            self._d2B_by_dXdX = None
        return self

    def compute_A(self, points, compute_derivatives=0):
        self._A = None
        if compute_derivatives >= 1:
            self._dA_by_dX = None
        if compute_derivatives >= 2:
            self._d2A_by_dXdX = None
        return self

    def __add__(self, other):
        return MagneticFieldSum([self,other])

class MagneticFieldSum(MagneticField):
    '''Class that is called when two or more magnetic fields are added together.
    It can either be called directly with a list of magnetic fields given as input and outputing another magnetic field with B, A and its derivatives added together or it can be called by summing magnetic fields classes as Bfield1 + Bfield1
    '''
    def __init__(self,Bfields):
        self.Bfields = Bfields
        for Bfield in self.Bfields:
            if hasattr(Bfield, 'coils'):
                self.coils = Bfield.coils
                break

    def set_points(self, points):
        self.points = np.array(points)
        self.clear_cached_properties()
        [Bfield.set_points(points) for Bfield in self.Bfields]
        return self

    def compute(self, points, compute_derivatives=0):
        [Bfield.compute(points, compute_derivatives) for Bfield in self.Bfields]
        self._B             = np.sum([Bfield._B for Bfield in self.Bfields],0)
        if compute_derivatives >= 1:
            self._dB_by_dX  = np.sum([Bfield._dB_by_dX for Bfield in self.Bfields],0)
        if compute_derivatives >= 2:
            self._d2B_by_dX = np.sum([Bfield._d2B_by_dX for Bfield in self.Bfields],0)

    def compute_A(self, points, compute_derivatives=0):
        [Bfield.compute_A(points, compute_derivatives) for Bfield in self.Bfields]
        self._A             = np.sum([Bfield._A for Bfield in self.Bfields],0)
        if compute_derivatives >= 1:
            self._dA_by_dX  = np.sum([Bfield._dA_by_dX for Bfield in self.Bfields],0)
        if compute_derivatives >= 2:
            self._d2A_by_dX = np.sum([Bfield._d2A_by_dX for Bfield in self.Bfields],0)