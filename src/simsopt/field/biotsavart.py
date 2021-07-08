import numpy as np
import simsoptpp as sopp
from simsopt.field.magneticfield import MagneticField
from simsopt.geo.curve import Curve


class BiotSavart(sopp.BiotSavart, MagneticField):
    r"""
    Computes the MagneticField induced by a list of closed curves :math:`\Gamma_k` with electric currents :math:`I_k`.
    The field is given by

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi

    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant.

    """

    def __init__(self, coils, coil_currents):
        assert len(coils) == len(coil_currents)
        assert all(isinstance(item, Curve) for item in coils)
        assert all(isinstance(item, float) for item in coil_currents)
        self.currents_optim = [sopp.Current(c) for c in coil_currents]
        self.coils_optim = [sopp.Coil(curv, curr) for curv, curr in zip(coils, self.currents_optim)]
        self.coils = coils
        self.coil_currents = coil_currents
        MagneticField.__init__(self)
        sopp.BiotSavart.__init__(self, self.coils_optim)

    def compute_A(self, compute_derivatives=0):
        r"""
        Compute the potential of a BiotSavart field

        .. math::

            A(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{\Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|} d\phi

        as well as the derivatives of `A` if requested.
        """
        assert compute_derivatives <= 2

        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self.coils)

        self._dA_by_dcoilcurrents = [self.fieldcache_get_or_create(f'A_{i}', [npoints, 3]) for i in range(ncoils)]
        if compute_derivatives >= 1:
            self._d2A_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dA_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        if compute_derivatives >= 2:
            self._d3A_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddA_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]

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

        self._A = sum(self.coil_currents[i] * self._dA_by_dcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 1:
            self._dA = sum(self.coil_currents[i] * self._d2A_by_dXdcoilcurrents[i] for i in range(len(self.coil_currents)))
        if compute_derivatives >= 2:
            self._ddA = sum(self.coil_currents[i] * self._d3A_by_dXdXdcoilcurrents[i] for i in range(len(self.coil_currents)))
        return self

    def _A_impl(self, A):
        self.compute_A(compute_derivatives=0)
        A[:] = self._A

    def _dA_by_dX_impl(self, dA_by_dX):
        self.compute_A(compute_derivatives=1)
        dA_by_dX[:] = self._dA

    def _d2A_by_dXdX_impl(self, d2A_by_dXdX):
        self.compute_A(compute_derivatives=2)
        d2A_by_dXdX[:] = self._ddA

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self.coils)
        if any([not self.fieldcache_get_status(f'B_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 0
            self.compute(compute_derivatives)
        self._dB_by_dcoilcurrents = [self.fieldcache_get_or_create(f'B_{i}', [npoints, 3]) for i in range(ncoils)]
        return self._dB_by_dcoilcurrents

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self.coils)
        if any([not self.fieldcache_get_status(f'dB_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 1
            self.compute(compute_derivatives)
        self._d2B_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dB_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        return self._d2B_by_dXdcoilcurrents

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self.coils)
        if any([not self.fieldcache_get_status(f'ddB_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 2
            self.compute(compute_derivatives)
        self._d3B_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddB_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]
        return self._d3B_by_dXdXdcoilcurrents

    def B_vjp(self, v):
        r"""
        Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
        :math:`\{\mathbf{B}_i\}_{i=1}^n`.
        These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
        This function returns the vector Jacobian product of this dependency, i.e.

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k.

        """

        gammas = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis = [coil.gammadash() for coil in self.coils]
        currents = self.coil_currents
        dgamma_by_dcoeffs = [coil.dgamma_by_dcoeff() for coil in self.coils]
        d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
        n = len(self.coils)
        coils = self.coils
        res_B = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        sopp.biot_savart_vjp(self.get_points_cart_ref(), gammas, dgamma_by_dphis, currents, v, [], dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, [])
        return res_B

    def B_and_dB_vjp(self, v, vgrad):
        r"""
        Same as :obj:`simsopt.field.biotsavart.BiotSavart.B_vjp` but returns the vector Jacobian product for :math:`B` and :math:`\nabla B`, i.e. it returns

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k, \{ \sum_{i=1}^{n} {\mathbf{v}_\mathrm{grad}}_i \cdot \partial_{\mathbf{c}_k} \nabla \mathbf{B}_i \}_k.
        """

        gammas = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis = [coil.gammadash() for coil in self.coils]
        currents = self.coil_currents
        dgamma_by_dcoeffs = [coil.dgamma_by_dcoeff() for coil in self.coils]
        d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
        n = len(self.coils)
        coils = self.coils
        res_B = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        res_dB = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        sopp.biot_savart_vjp(self.get_points_cart_ref(), gammas, dgamma_by_dphis, currents, v, vgrad, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, res_dB)
        return (res_B, res_dB)
