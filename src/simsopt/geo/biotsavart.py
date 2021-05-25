import numpy as np
import simsgeopp as sgpp
from simsopt.geo.magneticfield import MagneticField


class BiotSavart(sgpp.BiotSavart, MagneticField):

    def __init__(self, coils, coil_currents):
        assert len(coils) == len(coil_currents)
        self.currents_optim = [sgpp.Current(c) for c in coil_currents]
        self.coils_optim = [sgpp.Coil(curv, curr) for curv, curr in zip(coils, self.currents_optim)]
        MagneticField.__init__(self)
        sgpp.BiotSavart.__init__(self, self.coils_optim)
        self.coils = coils
        self.coil_currents = coil_currents

    def compute_A(self, compute_derivatives=0):
        points = self.get_points_cart_ref()
        assert compute_derivatives <= 2

        npoints = len(points)
        ncoils = len(self.coils)

        self._dA_by_dcoilcurrents = [self.cache_get_or_create(f'A_{i}', [npoints, 3]) for i in range(ncoils)]
        if compute_derivatives >= 1:
            self._d2A_by_dXdcoilcurrents = [self.cache_get_or_create(f'dA_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        if compute_derivatives >= 2:
            self._d3A_by_dXdXdcoilcurrents = [self.cache_get_or_create(f'ddA_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]

        gammas = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis = [coil.gammadash() for coil in self.coils]
        for l in range(ncoils):
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

        A = self.cache_get_or_create('A', [npoints, 3])
        A[:] = sum(self.coil_currents[i] * self._dA_by_dcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 1:
            dA = self.cache_get_or_create('dA_by_dX', [npoints, 3, 3])
            dA[:] = sum(self.coil_currents[i] * self._d2A_by_dXdcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 2:
            ddA = self.cache_get_or_create('d2A_by_dXdX', [npoints, 3, 3, 3])
            ddA[:] = sum(self.coil_currents[i] * self._d3A_by_dXdXdcoilcurrents[i] for i in range(len(self.coil_currents)))
        return self

    def A_impl(self, A):
        self.compute_A(compute_derivatives=0)

    def dA_by_dX_impl(self, dA_by_dX):
        self.compute_A(compute_derivatives=1)

    def d2A_by_dXdX_impl(self, dA_by_dX):
        self.compute_A(compute_derivatives=2)

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        if any([not self.cache_get_or_create(f'B_{i}' for i in range(len(self.coils)))]):
            assert compute_derivatives >= 0
            self.compute(compute_derivatives)
        self._dB_by_dcoilcurrents = [self.check_the_cache(f'B_{i}', [len(self.points), 3]) for i in range(len(self.coils))]
        return self._dB_by_dcoilcurrents

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        if any([not self.cache_get_or_create(f'dB_{i}' for i in range(len(self.coils)))]):
            assert compute_derivatives >= 1
            self.compute(compute_derivatives)
        self._d2B_by_dXdcoilcurrents = [self.check_the_cache(f'dB_{i}', [len(self.points), 3, 3]) for i in range(len(self.coils))]
        return self._d2B_by_dXdcoilcurrents

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        if any([not self.cache_get_or_create(f'ddB_{i}' for i in range(len(self.coils)))]):
            assert compute_derivatives >= 2
            self.compute(compute_derivatives)
        self._d3B_by_dXdXdcoilcurrents = [self.check_the_cache(f'ddB_{i}', [len(self.points), 3, 3, 3]) for i in range(len(self.coils))]
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
        sgpp.biot_savart_vjp(self.get_points_cart_ref(), gammas, dgamma_by_dphis, currents, v, [], dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, [])
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
        sgpp.biot_savart_vjp(self.get_points_cart_ref(), gammas, dgamma_by_dphis, currents, v, vgrad, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, res_dB)
        return (res_B, res_dB)
