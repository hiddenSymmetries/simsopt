import numpy as np

import simsoptpp as sopp
from .magneticfield import MagneticField
from .._core.json import GSONDecoder
from .._core.derivative import Derivative

__all__ = ['BiotSavart', 'PSC_BiotSavart']


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
        self._coils = coils
        sopp.BiotSavart.__init__(self, coils)
        MagneticField.__init__(self, depends_on=coils)

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'B_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 0
            self.compute(compute_derivatives)
        self._dB_by_dcoilcurrents = [self.fieldcache_get_or_create(f'B_{i}', [npoints, 3]) for i in range(ncoils)]
        return self._dB_by_dcoilcurrents

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'dB_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 1
            self.compute(compute_derivatives)
        self._d2B_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dB_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        return self._d2B_by_dXdcoilcurrents

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
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

        coils = self._coils
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
        from simsopt.field.coil import PSCCoil

        coils = self._coils
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
        # print(res_current)
        # print(np.shape(dB_by_dcoilcurrents), np.shape(dB_by_dcoilcurrents[0]), np.shape(res_current))
        # print('check = ', [coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])).data for i in range(len(coils))])
        # print('check2 = ', sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))]).data)
        # print(np.shape(sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])))
        # exit()
        curve_flags = [isinstance(coil, PSCCoil) for coil in coils]
        # print('here = ', curve_flags)
        # print(np.shape(dB_by_dcoilcurrents), np.shape(v), np.shape(res_gamma))

        if np.any(curve_flags):
            order = coils[0].curve.order 
            ndofs = 2 * order + 8
            # dI = np.zeros((coils[0].curve.npsc, ndofs))
            # print('res_current = ', res_current)
            # dI_deriv = []
            # for i in range(coils[0].curve.npsc):   # Not using the rotated ones directly
                # print(i, coils[i].curve._index)
                # for q in range(coils[0].curve._psc_array.symmetry):
                # for fp in range(coils[0].curve._psc_array.nfp):
                #     for stell in coils[0].curve._psc_array.stell_list:
                # dI[i, :] = coils[i].curve.dkappa_dcoef_vjp([res_current[i]], i)
            dI = np.zeros((len(coils), ndofs))
            # for i in range(len(coils)):   # Not using the rotated ones directly
                # print(i, coils[i].curve._index)
                # for q in range(coils[0].curve._psc_array.symmetry):
            q = 0
            for fp in range(coils[0].curve._psc_array.nfp):
                for stell in coils[0].curve._psc_array.stell_list:
                    for i in range(coils[0].curve.npsc):   # Not using the rotated ones directly
                        # dI[i, :] += coils[i + q * coils[0].curve.npsc].curve.dkappa_dcoef_vjp(
                        #     [res_current[i + q * coils[0].curve.npsc]], i + q * coils[0].curve.npsc)
                        dI[i, :] += coils[i + q * coils[0].curve.npsc].curve.dkappa_dcoef_vjp(
                            [res_current[i + q * coils[0].curve.npsc]], i) * stell
                    q += 1
                # dI[i % coils[0].curve.npsc, :] += coils[i].curve.dkappa_dcoef_vjp([res_current[i]], i)   
                # coils[i].curve.plot()
                # print(i, dI[i, :])
            Linv = coils[0].curve._psc_array.L_inv
            # Linv[coils[0].curve.npsc:, :] = 0.0
            # q = 0
            # dI_all = np.zeros((len(coils), ndofs))
            # for fp in range(coils[0].curve._psc_array.nfp):
            #     for stell in coils[0].curve._psc_array.stell_list:
            #         dI_all[q * coils[0].curve.npsc:(q + 1) * coils[0].curve.npsc, :] = dI[:coils[0].curve.npsc, :] * stell
            #         q += 1
            # dI = - Linv @ dI_all
            dI = - Linv @ dI
            self.dI_psc = dI[:, 2 * order + 1: 2 * order + 5]
            # print('dI_ mid = ', self.dI_psc)
            # dI = np.zeros(dI.shape)
            ### Issue is that the disrete symmetries not getting imposed right here. Have
            # dJ / dB * dB / dI * dI / ddofs and fairly sure dI / ddofs looks correct when dB/dI is just set to one. 
            # Clearly dB / dI should be diffferent for the coils at discrete symmetries. 

            # print(dI.shape)
            # for q in range(1, 4):
            #     dI[:coils[0].curve.npsc, :] += dI[coils[0].curve.npsc * q:coils[0].curve.npsc * (q + 1), :]
            # dI = np.zeros(dI.shape)
            # print(np.max(np.abs(dI)), dI.shape)
            # for i in range(len(coils)): 
            #     dI_deriv.append(np.asarray(Derivative{coils[i].curve: dI[i]}))
            # dB = 
            # print([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([Derivative({coils[i].curve: dI[i, :]})])) for i in range(len(coils))])
            # print(res_gamma, res_gammadash, res_current)
            # exit()
            return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], dI[i, :]) for i in range(len(coils))])
        else:
            return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])

    def dA_by_dcoilcurrents(self, compute_derivatives=0):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'A_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 0
            self.compute(compute_derivatives)
        self._dA_by_dcoilcurrents = [self.fieldcache_get_or_create(f'A_{i}', [npoints, 3]) for i in range(ncoils)]
        return self._dA_by_dcoilcurrents

    def d2A_by_dXdcoilcurrents(self, compute_derivatives=1):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'dA_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 1
            self.compute(compute_derivatives)
        self._d2A_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dA_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        return self._d2A_by_dXdcoilcurrents

    def d3A_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'ddA_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 2
            self.compute(compute_derivatives)
        self._d3A_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddA_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]
        return self._d3A_by_dXdXdcoilcurrents

    def A_and_dA_vjp(self, v, vgrad):
        r"""
        Same as :obj:`simsopt.geo.biotsavart.BiotSavart.A_vjp` but returns the vector Jacobian product for :math:`A` and :math:`\nabla A`, i.e. it returns

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{A}_i \}_k, \{ \sum_{i=1}^{n} {\mathbf{v}_\mathrm{grad}}_i \cdot \partial_{\mathbf{c}_k} \nabla \mathbf{A}_i \}_k.
        """

        coils = self._coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
        res_grad_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_grad_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vector_potential_vjp_graph(points, gammas, gammadashs, currents, v,
                                                    res_gamma, res_gammadash, vgrad, res_grad_gamma, res_grad_gammadash)

        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        d2A_by_dXdcoilcurrents = self.d2A_by_dXdcoilcurrents()
        res_grad_current = [np.sum(vgrad * d2A_by_dXdcoilcurrents[i]) for i in range(len(d2A_by_dXdcoilcurrents))]

        res = (
            sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))]),
            sum([coils[i].vjp(res_grad_gamma[i], res_grad_gammadash[i], np.asarray([res_grad_current[i]])) for i in range(len(coils))])
        )

        return res

    def A_vjp(self, v):
        r"""
        Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
        :math:`\{\mathbf{A}_i\}_{i=1}^n`.
        These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
        This function returns the vector Jacobian product of this dependency, i.e.

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{A}_i \}_k.

        """

        coils = self._coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vector_potential_vjp_graph(points, gammas, gammadashs, currents, v,
                                                    res_gamma, res_gammadash, [], [], [])
        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"],
                                      serial_objs_dict=serial_objs_dict,
                                      recon_objs=recon_objs)
        coils = decoder.process_decoded(d["coils"],
                                        serial_objs_dict=serial_objs_dict,
                                        recon_objs=recon_objs)
        bs = cls(coils)
        bs.set_points_cart(xyz)
        return bs

class PSC_BiotSavart(BiotSavart):
    """
    """
    def __init__(self, psc_array):
        from simsopt.field import coils_via_symmetries, Current, psc_coils_via_symmetries
        self.psc_array = psc_array
        self.npsc = psc_array.num_psc
        curves = [self.psc_array.curves[i] for i in range(self.npsc)]
        currents = [Current(self.psc_array.I[i] * 1e-5) * 1e5 for i in range(self.npsc)]
        [currents[i].fix_all() for i in range(self.npsc)]
        # self.currents = currents
        coils = psc_coils_via_symmetries(curves, currents, psc_array.nfp, psc_array.stellsym)
        BiotSavart.__init__(self, coils)

    def set_currents(self):
        from simsopt.field import coils_via_symmetries, Current, psc_coils_via_symmetries
        # print([self._coils[i].current.get_value() for i in range(self.npsc)])
        order = self._coils[0].curve.order 
        dofs = np.array([self._coils[i].curve.get_dofs() for i in range(self.npsc)])
        dofs = dofs[:, 2 * order + 1:2 * order + 5]
        normalization = np.sqrt(np.sum(dofs ** 2, axis=-1))
        dofs = dofs / normalization[:, None]
        alphas1 = np.arctan2(2 * (dofs[:, 0] * dofs[:, 1] + dofs[:, 2] * dofs[:, 3]), 
                            1 - 2.0 * (dofs[:, 1] ** 2 + dofs[:, 2] ** 2))
        deltas1 = -np.pi / 2.0 + 2.0 * np.arctan2(
            np.sqrt(1.0 + 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])), 
            np.sqrt(1.0 - 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])))
        self.psc_array.setup_orientations(alphas1, deltas1)
        self.psc_array.update_psi()
        self.psc_array.setup_currents_and_fields()
        self.psc_array.psi_deriv()
        psi = self.psc_array.psi / self.psc_array.fac
        Linv = self.psc_array.L_inv[:self.psc_array.num_psc, :self.psc_array.num_psc] # / psc_array.fac
        I = (-self.psc_array.L_inv @ self.psc_array.psi_total) / self.psc_array.fac
        for i in range(self.npsc):
            self._coils[i]._current.unfix_all()
            self._coils[i]._current.x = [I[i] * 1e-5]
            self._coils[i]._current.fix_all()
            # self._coils[i].current.set('x0', self.psc_array.I[i])
            # self.currents[i].set('x0', self.psc_array.I[i])
        # print([self._coils[i].current.get_value() for i in range(self.npsc)])
        # print([self._coils[i].current.get_value() for i in range(self.npsc)])
        # currents = [Current(self.psc_array.I[i] * 1e-5) * 1e5 for i in range(self.npsc)]
        # for i in range(self.npsc):
        #     self._coils[i].current = self.currents[i]
        # [currents[i].fix_all() for i in range(self.npsc)]
        # curves = [self.psc_array.curves[i] for i in range(self.npsc)]
        # curves = [self.psc_array.curves[i] for i in range(self.npsc)]
        # curves = [self._coils[i].curve for i in range(self.npsc)]
        # coils = psc_coils_via_symmetries(curves, currents, self.psc_array.nfp, self.psc_array.stellsym)
        # BiotSavart.__init__(self, coils)

    def B_vjp(self, v):
        r"""
        Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
        :math:`\{\mathbf{B}_i\}_{i=1}^n`.
        These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
        This function returns the vector Jacobian product of this dependency, i.e.

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k.

        """
        from simsopt.geo.curveplanarfourier import PSCCurve

        coils = self._coils
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
        
        curve_flags = [isinstance(coil.curve, PSCCurve) for coil in coils]
        if np.any(curve_flags):
            order = coils[0].curve.order 
            ndofs = 2 * order + 8
            dI = np.zeros((len(coils), ndofs))
            q = 0
            for fp in range(self.psc_array.nfp):
                for stell in self.psc_array.stell_list:
                    for i in range(self.npsc):
                        dI[i, :] += coils[i + q * self.npsc].curve.dkappa_dcoef_vjp(
                            [res_current[i + q * self.npsc]], self.psc_array.dpsi) / (self.psc_array.nfp * len(self.psc_array.stell_list))      
                        
                    q += 1
            self.dpsi_debug = dI[:, 2*order+1:2*order+5]
            Linv = self.psc_array.L_inv
            dI = (- Linv @ dI)   
            # print('dI = ', dI[:, 2*order+1:2*order+5])
            return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], dI[i, :]) for i in range(len(coils))])
        else:
            return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], res_current[i]) for i in range(len(coils))])
