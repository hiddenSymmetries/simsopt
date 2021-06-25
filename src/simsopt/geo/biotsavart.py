import numpy as np
import simsoptpp as sopp
from simsopt.geo.magneticfield import MagneticField
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
        self._dA_by_dcoilcurrents = [np.zeros((len(points), 3)) for coil in self.coils]

        if compute_derivatives >= 1:
            self._d2A_by_dXdcoilcurrents = [np.zeros((len(points), 3, 3)) for coil in self.coils]
        else:
            self._d2A_by_dXdcoilcurrents = []

        if compute_derivatives >= 2:
            self._d3A_by_dXdXdcoilcurrents = [np.zeros((len(points), 3, 3, 3)) for coil in self.coils]
        else:
            self._d3A_by_dXdXdcoilcurrents = []

        gammas                 = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis        = [coil.gammadash() for coil in self.coils]

        sopp.biot_savart_vector_potential(points, gammas, dgamma_by_dphis, self._dA_by_dcoilcurrents, self._d2A_by_dXdcoilcurrents, self._d3A_by_dXdXdcoilcurrents)

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
        Same as :obj:`simsopt.geo.biotsavart.BiotSavart.B_vjp` but returns the vector Jacobian product for :math:`B` and :math:`\nabla B`, i.e. it returns

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

    def A_vjp(self, v):
        gammas                 = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis        = [coil.gammadash() for coil in self.coils]
        currents = self.coil_currents
        dgamma_by_dcoeffs      = [coil.dgamma_by_dcoeff() for coil in self.coils]
        d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
        n = len(self.coils)
        coils = self.coils
        res_A = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        sopp.biot_savart_vector_potential_vjp(self.get_points_cart_ref(), gammas, dgamma_by_dphis, currents, v, [], dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_A, [])
        return res_A

    def A_and_dA_vjp(self, v, vgrad):
        gammas                 = [coil.gamma() for coil in self.coils]
        dgamma_by_dphis        = [coil.gammadash() for coil in self.coils]
        currents = self.coil_currents
        dgamma_by_dcoeffs      = [coil.dgamma_by_dcoeff() for coil in self.coils]
        d2gamma_by_dphidcoeffs = [coil.dgammadash_by_dcoeff() for coil in self.coils]
        n = len(self.coils)
        coils = self.coils
        res_A = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        res_dA = [np.zeros((coils[i].num_dofs(), )) for i in range(n)]
        sopp.biot_savart_vector_potential_vjp(self.get_points_cart_ref(), gammas, dgamma_by_dphis, currents, v, vgrad, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_A, res_dA)
        return (res_A, res_dA)
