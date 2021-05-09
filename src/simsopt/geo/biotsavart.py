import numpy as np
import simsgeopp as sgpp
from simsopt.geo.magneticfield import MagneticField


class BiotSavart(MagneticField):

    def __init__(self, coils, coil_currents):
        assert len(coils) == len(coil_currents)
        self.coils = coils
        self.coil_currents = coil_currents

    def compute_A(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        self._dA_by_dcoilcurrents = [np.zeros((len(points), 3)) for coil in self.coils]
        if compute_derivatives >= 1:
            self._d2A_by_dXdcoilcurrents = [np.zeros((len(points), 3, 3)) for coil in self.coils]
        if compute_derivatives >= 2:
            self._d3A_by_dXdXdcoilcurrents = [np.zeros((len(points), 3, 3, 3)) for coil in self.coils]

        gammas = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis = [coil.gammadash() for coil in self.coils]
        for l in range(len(self.coils)):
            coil = self.coils[l]
            current = self.coil_currents[l]
            gamma = gammas[l]
            dgamma_by_dphi = dgamma_by_dphis[l] 
            num_coil_quadrature_points = gamma.shape[0]
            for i, point in enumerate(points):
                diff = point-gamma
                dist = np.linalg.norm(diff, axis=1)
                self._dA_by_dcoilcurrents[l][i, :] += np.sum((1./dist)[:, None] * dgamma_by_dphi, axis=0)

                if compute_derivatives >= 1:
                    for j in range(3):
                        self._d2A_by_dXdcoilcurrents[l][i, j, :] = np.sum(-(diff[:, j]/dist**3)[:, None] * dgamma_by_dphi, axis=0)

                if compute_derivatives >= 2:
                    for j1 in range(3):
                        for j2 in range(3):
                            term1 = 3 * (diff[:, j1] * diff[:, j2]/dist**5)[:, None] * dgamma_by_dphi
                            if j1 == j2:
                                term2 = - (1./dist**3)[:, None] * dgamma_by_dphi
                            else:
                                term2 = 0
                            self._d3A_by_dXdXdcoilcurrents[l][i, j1, j2, :] = np.sum(term1 + term2, axis=0)

            self._dA_by_dcoilcurrents[l] *= (1e-7/num_coil_quadrature_points)
            if compute_derivatives >= 1:
                self._d2A_by_dXdcoilcurrents[l] *= (1e-7/num_coil_quadrature_points)
            if compute_derivatives >= 2:
                self._d3A_by_dXdXdcoilcurrents[l] *= (1e-7/num_coil_quadrature_points)

        self._A = sum(self.coil_currents[i] * self._dA_by_dcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 1:
            self._dA_by_dX = sum(self.coil_currents[i] * self._d2A_by_dXdcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 2:
            self._d2A_by_dXdX = sum(self.coil_currents[i] * self._d3A_by_dXdXdcoilcurrents[i] for i in range(len(self.coil_currents)))
        return self

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        self._dB_by_dcoilcurrents = [np.zeros((len(points), 3)) for coil in self.coils]

        if compute_derivatives >= 1:
            self._d2B_by_dXdcoilcurrents = [np.zeros((len(points), 3, 3)) for coil in self.coils]
        else:
            self._d2B_by_dXdcoilcurrents = []

        if compute_derivatives >= 2:
            self._d3B_by_dXdXdcoilcurrents = [np.zeros((len(points), 3, 3, 3)) for coil in self.coils]
        else:
            self._d3B_by_dXdXdcoilcurrents = []

        gammas = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis = [coil.gammadash() for coil in self.coils]

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

    def B_vjp(self, v):
        gammas = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis = [coil.gammadash() for coil in self.coils]
        currents = self.coil_currents
        dgamma_by_dcoeffs = [coil.dgamma_by_dcoeff() for coil in self.coils]
        d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
        n = len(self.coils)
        coils = self.coils
        res_B = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        sgpp.biot_savart_vjp(self.points, gammas, dgamma_by_dphis, currents, v, [], dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, [])
        return res_B

    def B_and_dB_vjp(self, v, vgrad):
        gammas = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis = [coil.gammadash() for coil in self.coils]
        currents = self.coil_currents
        dgamma_by_dcoeffs = [coil.dgamma_by_dcoeff() for coil in self.coils]
        d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
        n = len(self.coils)
        coils = self.coils
        res_B = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        res_dB = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        sgpp.biot_savart_vjp(self.points, gammas, dgamma_by_dphis, currents, v, vgrad, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, res_dB)
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
