
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

	def compute(self, points, compute_derivatives):
		self._B = None
		return self