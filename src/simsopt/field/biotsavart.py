import numpy as np

import simsoptpp as sopp
from ..geo.curve import Curve
from .._core.derivative import Derivative
from .magneticfield import MagneticField


class BiotSavart(sopp.BiotSavart, MagneticField):
    r"""
    Computes the MagneticField induced by a list of closed curves :math:`\Gamma_k` with electric currents :math:`I_k`.
    The field is given by

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi

    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant.

    Args:
        coils: A list of :obj:`simsopt.field.coil.Coil` objects.
    """

    def __init__(self, coils):
        self.__coils = coils
        sopp.BiotSavart.__init__(self, coils)
        MagneticField.__init__(self, depends_on=coils)

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
        coils = self.__coils
        ncoils = len(coils)

        self._dA_by_dcoilcurrents = [self.fieldcache_get_or_create(f'A_{i}', [npoints, 3]) for i in range(ncoils)]
        if compute_derivatives >= 1:
            self._d2A_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dA_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        if compute_derivatives >= 2:
            self._d3A_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddA_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]

        gammas = [coil.curve.gamma() for coil in coils]
        dgamma_by_dphis = [coil.curve.gammadash() for coil in coils]
        for l in range(ncoils):
            coil = coils[l]
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

        self._A = sum(coils[i].current.get_value() * self._dA_by_dcoilcurrents[i] for i in range(ncoils))
        if compute_derivatives >= 1:
            self._dA = sum(coils[i].current.get_value() * self._d2A_by_dXdcoilcurrents[i] for i in range(ncoils))
        if compute_derivatives >= 2:
            self._ddA = sum(coils[i].current.get_value() * self._d3A_by_dXdXdcoilcurrents[i] for i in range(ncoils))
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
        ncoils = len(self.__coils)
        if any([not self.fieldcache_get_status(f'B_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 0
            self.compute(compute_derivatives)
        self._dB_by_dcoilcurrents = [self.fieldcache_get_or_create(f'B_{i}', [npoints, 3]) for i in range(ncoils)]
        return self._dB_by_dcoilcurrents

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self.__coils)
        if any([not self.fieldcache_get_status(f'dB_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 1
            self.compute(compute_derivatives)
        self._d2B_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dB_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        return self._d2B_by_dXdcoilcurrents

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self.__coils)
        if any([not self.fieldcache_get_status(f'ddB_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 2
            self.compute(compute_derivatives)
        self._d3B_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddB_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]
        return self._d3B_by_dXdXdcoilcurrents

    def B_and_dB_vjp(self, v, vgrad):
        r"""
        Same as :obj:`simsopt.geo.biotsavart.BiotSavart.B_vjp` but returns the vector Jacobian product for :math:`B` and :math:`\nabla B`, i.e. it returns

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k, \{ \sum_{i=1}^{n} {\mathbf{v}_\mathrm{grad}}_i \cdot \partial_{\mathbf{c}_k} \nabla \mathbf{B}_i \}_k.
        """

        coils = self.__coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
        res_grad_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_grad_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
                                   res_gamma, res_gammadash, vgrad, res_grad_gamma, res_grad_gammadash)

        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        d2B_by_dXdcoilcurrents = self.d2B_by_dXdcoilcurrents()
        res_grad_current = [np.sum(vgrad * d2B_by_dXdcoilcurrents[i]) for i in range(len(d2B_by_dXdcoilcurrents))]

        res = (
            sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))]),
            sum([coils[i].vjp(res_grad_gamma[i], res_grad_gammadash[i], np.asarray([res_grad_current[i]])) for i in range(len(coils))])
        )

        return res

    def B_vjp(self, v):
        r"""
        Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
        :math:`\{\mathbf{B}_i\}_{i=1}^n`.
        These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
        This function returns the vector Jacobian product of this dependency, i.e.

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k.

        """

        coils = self.__coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
                                   res_gamma, res_gammadash, [], [], [])
        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])
