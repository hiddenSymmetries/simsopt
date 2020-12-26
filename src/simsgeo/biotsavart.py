import numpy as np
import simsgeopp as sgpp


class BiotSavart():

    def __init__(self, coils, coil_currents):
        assert len(coils) == len(coil_currents)
        self.coils = coils
        self.coil_currents = coil_currents

    def clear_cached_properties(self):
        self._B = None
        self._dB_by_dX = None
        self._d2B_by_dXdX = None

    def set_points(self, points):
        self.points = points
        self.clear_cached_properties()
        return self

    def B(self, compute_derivatives=0):
        if self._B is None:
            assert compute_derivatives >= 0
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
        assert compute_derivatives <= 2

        self._dB_by_dcoilcurrents    = [np.zeros((len(points), 3)) for coil in self.coils]

        if compute_derivatives >= 1:
            self._d2B_by_dXdcoilcurrents = [np.zeros((len(points), 3, 3)) for coil in self.coils]
        else:
            self._d2B_by_dXdcoilcurrents = []

        if compute_derivatives >= 2:
            self._d3B_by_dXdXdcoilcurrents = [np.zeros((len(points), 3, 3, 3)) for coil in self.coils]
        else:
            self._d3B_by_dXdXdcoilcurrents = []

        gammas                 = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis        = [coil.gammadash() for coil in self.coils]

        sgpp.biot_savart(points, gammas, dgamma_by_dphis, self._dB_by_dcoilcurrents, self._d2B_by_dXdcoilcurrents, self._d3B_by_dXdXdcoilcurrents)

        self._B = sum(self.coil_currents[i] * self._dB_by_dcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 1:
            self._dB_by_dX = sum(self.coil_currents[i] * self._d2B_by_dXdcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 2:
            self._d2B_by_dXdX = sum(self.coil_currents[i] * self._d3B_by_dXdXdcoilcurrents[i] for i in range(len(self.coil_currents)))

        return self

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        if self._dB_by_dcoilcurrents is None:
            assert compute_derivatives >= 0
            self.compute(self.points, compute_derivatives)
        return self._dB_by_dcoilcurrents

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        if self._d2B_by_dXdcoilcurrents is None:
            assert compute_derivatives >= 1
            self.compute(self.points, compute_derivatives)
        return self._d2B_by_dXdcoilcurrents

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        if self._d3B_by_dXdXdcoilcurrents is None:
            assert compute_derivatives >= 2
            self.compute(self.points, compute_derivatives)
        return self._d3B_by_dXdXdcoilcurrents

    def B_and_dB_vjp(self, v, vgrad):
        gammas                 = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis        = [coil.gammadash() for coil in self.coils]
        currents = self.coil_currents
        dgamma_by_dcoeffs      = [coil.dgamma_by_dcoeff() for coil in self.coils]
        d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
        n = len(self.coils)
        coils = self.coils
        res_B = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        res_dB = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        sgpp.biot_savart_by_dcoilcoeff_all_vjp_full(self.points, gammas, dgamma_by_dphis, currents, v, vgrad, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, res_dB)
        return (res_B, res_dB)

    # def compute_by_dcoilcoeff(self, points):
    #     self.dB_by_dcoilcoeffs    = [np.zeros((len(points), 3, coil.num_dofs())) for coil in self.coils]
    #     self.d2B_by_dXdcoilcoeffs = [np.zeros((len(points), 3, 3, coil.num_dofs())) for coil in self.coils]
    #     gammas                 = [coil.gamma() for coil in self.coils]
    #     dgamma_by_dphis        = [coil.gammadash() for coil in self.coils]
    #     dgamma_by_dcoeffs      = [coil.dgamma_by_dcoeff() for coil in self.coils]
    #     d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
    #     cpp.biot_savart_by_dcoilcoeff_all(points, gammas, dgamma_by_dphis, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, self.coil_currents, self.dB_by_dcoilcoeffs, self.d2B_by_dXdcoilcoeffs)
    #     return self
