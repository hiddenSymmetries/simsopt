
class MagneticField():

    def clear_cached_properties(self):
        self._B = None
        self._dB_by_dX = None
        self._d2B_by_dXdX = None
        self._A = None
        self._dA_by_dX = None
        self._d2A_by_dXdX = None

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

class MagneticFieldSum(MagneticField):

    def __init__(self,Bfield1,Bfield2):
        self.Bfield1 = Bfield1
        self.Bfield2 = Bfield2
        if hasattr(Bfield1, 'coils'):
            self.coils = Bfield1.coils
        elif hasattr(Bfield2, 'coils'):
            self.coils = Bfield2.coils
        else:
            self.coils = None

    def set_points(self, points):
        self.points = points
        self.clear_cached_properties()
        self.Bfield1.set_points(points)
        self.Bfield2.set_points(points)
        return self

    def compute(self, points, compute_derivatives=0):
        self.Bfield1.compute(points, compute_derivatives)
        self.Bfield2.compute(points, compute_derivatives)
        self._B = self.Bfield1._B + self.Bfield2._B
        if compute_derivatives >= 1:
            self._dB_by_dX = self.Bfield1._dB_by_dX + self.Bfield2._dB_by_dX
        if compute_derivatives >= 2:
            self._d2B_by_dXdX = self.Bfield1._d2B_by_dXdX + self.Bfield2._d2B_by_dXdX

    def compute_A(self, points, compute_derivatives=0):
        self.Bfield1.compute_A(points, compute_derivatives)
        self.Bfield2.compute_A(points, compute_derivatives)
        self._A = self.Bfield1._A + self.Bfield2._A
        if compute_derivatives >= 1:
            self._dA_by_dX = self.Bfield1._dA_by_dX + self.Bfield2._dA_by_dX
        if compute_derivatives >= 2:
            self._d2A_by_dXdX = self.Bfield1._d2A_by_dXdX + self.Bfield2._d2A_by_dXdX