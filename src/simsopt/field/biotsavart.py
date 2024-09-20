import numpy as np

import simsoptpp as sopp
from .magneticfield import MagneticField
from .._core.json import GSONDecoder
from .._core.derivative import Derivative

__all__ = ['BiotSavart']


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

# class PSC_BiotSavart(BiotSavart):
#     """
#     """
#     def __init__(self, psc_array):
#         from simsopt.field import coils_via_symmetries, Current
#         self.psc_array = psc_array
#         self.npsc = psc_array.num_psc
#         curves = [self.psc_array.curves[i] for i in range(self.npsc)]
#         currents = [Current(self.psc_array.I[i] * 1e-5) * 1e5 for i in range(self.npsc)]
#         [currents[i].fix_all() for i in range(self.npsc)]
#         coils = coils_via_symmetries(curves, currents, psc_array.nfp, psc_array.stellsym)
#         BiotSavart.__init__(self, coils)

#     def B_vjp(self, v):
#         r"""
#         Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
#         :math:`\{\mathbf{B}_i\}_{i=1}^n`.
#         These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
#         This function returns the vector Jacobian product of this dependency, i.e.

#         .. math::

#             \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k.

#         """
#         coils = self._coils
#         gammas = [coil.curve.gamma() for coil in coils]
#         gammadashs = [coil.curve.gammadash() for coil in coils]
#         currents = [coil.current.get_value() for coil in coils]
#         res_gamma = [np.zeros_like(gamma) for gamma in gammas]
#         res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
#         points = self.get_points_cart_ref()
#         sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
#                                    res_gamma, res_gammadash, [], [], [])
#         dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
#         res_current = [coil.curve.passive_current() for coil in coils]
#         return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], res_current[i]) for i in range(len(coils))])

# def JaxBiotSavart(sopp.BiotSavart, MagneticField):
#     r"""
#     """
#     def __init__(self, coils, **kwargs):
#         if "external_dof_setter" not in kwargs:
#             kwargs["external_dof_setter"] = sopp.Curve.set_dofs_impl
#         self._coils = coils
#         sopp.BiotSavart.__init__(self, coils)
#         MagneticField.__init__(self, depends_on=coils)
#         self.I_pure = I_pure
#         points = np.asarray(self.quadpoints)
#         ones = jnp.ones_like(points)

#         self.I_jax = jit(lambda dofs: self.I_pure())
#         self.I_impl_jax = jit(lambda dofs, p: self.I_pure(dofs, p))
#         self.dI_by_dcoeff_jax = jit(jacfwd(self.I_jax))
#         self.dI_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.I_jax, x)[1](v)[0])

#         self.B_jax = jit(lambda dofs: self.B_pure())
#         self.B_impl_jax = jit(lambda dofs, p: self.B_pure(dofs, p))
#         self.dB_by_dcoeff_jax = jit(jacfwd(self.B_jax))
#         self.dB_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.B_jax, x)[1](v)[0])
#         self.d2B_by_d2coeff_jax = lambda x, q: jvp(lambda p: self.gammadash_pure(x, p), (q,), (ones,))[1]

#         # self.gammadash_pure = lambda x, q: jvp(lambda p: self.gamma_pure(x, p), (q,), (ones,))[1]
#         # self.gammadash_jax = jit(lambda x: self.gammadash_pure(x, points))
#         # self.dgammadash_by_dcoeff_jax = jit(jacfwd(self.gammadash_jax))
#         # self.dgammadash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadash_jax, x)[1](v)[0])

#         # self.gammadashdash_pure = lambda x, q: jvp(lambda p: self.gammadash_pure(x, p), (q,), (ones,))[1]
#         # self.gammadashdash_jax = jit(lambda x: self.gammadashdash_pure(x, points))
#         # self.dgammadashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdash_jax))
#         # self.dgammadashdash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadashdash_jax, x)[1](v)[0])

#         # self.gammadashdashdash_pure = lambda x, q: jvp(lambda p: self.gammadashdash_pure(x, p), (q,), (ones,))[1]
#         # self.gammadashdashdash_jax = jit(lambda x: self.gammadashdashdash_pure(x, points))
#         # self.dgammadashdashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdashdash_jax))
#         # self.dgammadashdashdash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadashdashdash_jax, x)[1](v)[0])

#         # self.dkappa_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), x)[1](v)[0])

#         # self.dtorsion_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v)[0])

#     @jit
#     def I_pure(self, dofs, current_dof_indices):
#         """
#         """
#         return jnp.array(dofs[current_dof_indices])

#     @jit
#     def B_pure(self, dofs, points, current_dof_indices):
#         """
#         """
#         I = jnp.array(dofs[current_dof_indices])  # these are the Current dofs
#         # X = dofs[~current_dof_indices]  # these are the Curve dofs
#         gammas = jnp.array([coil.curve.gamma() for coil in self._coils])
#         gammadashs = jnp.array([coil.curve.gammadashs() for coil in self._coils])
#         B = jnp.zeros((len(points), 3))
#         # First two axes, over the total number of coils and the integral over the coil 
#         # points, should be summed
#         B = jnp.sum(I[:, None, None] * jnp.sum(jnp.cross(gammas[:, :, None, :] - points[None, None, :, :], 
#             gammadashs) / jnp.linalg.norm(
#                 gammas[:, :, None, :] - points[None, None, :, :], axis=-1), axis=1), axis=0)
#         return B * 1e-7

#     def dB_by_dcoilcurrents(self, compute_derivatives=0):
#         points = self.get_points_cart_ref()
#         npoints = len(points)
#         ncoils = len(self._coils)
#         if any([not self.fieldcache_get_status(f'B_{i}') for i in range(ncoils)]):
#             assert compute_derivatives >= 0
#             self.compute(compute_derivatives)
#         self._dB_by_dcoilcurrents = [self.fieldcache_get_or_create(f'B_{i}', [npoints, 3]) for i in range(ncoils)]
#         return self._dB_by_dcoilcurrents

#     def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
#         points = self.get_points_cart_ref()
#         npoints = len(points)
#         ncoils = len(self._coils)
#         if any([not self.fieldcache_get_status(f'dB_{i}') for i in range(ncoils)]):
#             assert compute_derivatives >= 1
#             self.compute(compute_derivatives)
#         self._d2B_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dB_{i}', [npoints, 3, 3]) for i in range(ncoils)]
#         return self._d2B_by_dXdcoilcurrents

#     def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
#         points = self.get_points_cart_ref()
#         npoints = len(points)
#         ncoils = len(self._coils)
#         if any([not self.fieldcache_get_status(f'ddB_{i}') for i in range(ncoils)]):
#             assert compute_derivatives >= 2
#             self.compute(compute_derivatives)
#         self._d3B_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddB_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]
#         return self._d3B_by_dXdXdcoilcurrents

#     def B_and_dB_vjp(self, v, vgrad):
#         r"""
#         Same as :obj:`simsopt.geo.biotsavart.BiotSavart.B_vjp` but returns the vector Jacobian product for :math:`B` and :math:`\nabla B`, i.e. it returns

#         .. math::

#             \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k, \{ \sum_{i=1}^{n} {\mathbf{v}_\mathrm{grad}}_i \cdot \partial_{\mathbf{c}_k} \nabla \mathbf{B}_i \}_k.
#         """

#         coils = self._coils
#         gammas = [coil.curve.gamma() for coil in coils]
#         gammadashs = [coil.curve.gammadash() for coil in coils]
#         currents = [coil.current.get_value() for coil in coils]
#         res_gamma = [np.zeros_like(gamma) for gamma in gammas]
#         res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
#         res_grad_gamma = [np.zeros_like(gamma) for gamma in gammas]
#         res_grad_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

#         points = self.get_points_cart_ref()
#         sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
#                                    res_gamma, res_gammadash, vgrad, res_grad_gamma, res_grad_gammadash)

#         dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
#         res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
#         d2B_by_dXdcoilcurrents = self.d2B_by_dXdcoilcurrents()
#         res_grad_current = [np.sum(vgrad * d2B_by_dXdcoilcurrents[i]) for i in range(len(d2B_by_dXdcoilcurrents))]

#         res = (
#             sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))]),
#             sum([coils[i].vjp(res_grad_gamma[i], res_grad_gammadash[i], np.asarray([res_grad_current[i]])) for i in range(len(coils))])
#         )

#         return res
    
#     def B_vjp(self, v):
#         r"""
#         Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
#         :math:`\{\mathbf{B}_i\}_{i=1}^n`.
#         These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
#         This function returns the vector Jacobian product of this dependency, i.e.

#         .. math::

#             \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k.

#         """
#         from simsopt.field.coil import PSCCoil

#         coils = self._coils
#         gammas = [coil.curve.gamma() for coil in coils]
#         gammadashs = [coil.curve.gammadash() for coil in coils]
#         currents = [coil.current.get_value() for coil in coils]
#         res_gamma = [np.zeros_like(gamma) for gamma in gammas]
#         res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

#         points = self.get_points_cart_ref()
#         sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
#                                    res_gamma, res_gammadash, [], [], [])
#         dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
#         res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
#         return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])

#     def dA_by_dcoilcurrents(self, compute_derivatives=0):
#         points = self.get_points_cart_ref()
#         npoints = len(points)
#         ncoils = len(self._coils)
#         if any([not self.fieldcache_get_status(f'A_{i}') for i in range(ncoils)]):
#             assert compute_derivatives >= 0
#             self.compute(compute_derivatives)
#         self._dA_by_dcoilcurrents = [self.fieldcache_get_or_create(f'A_{i}', [npoints, 3]) for i in range(ncoils)]
#         return self._dA_by_dcoilcurrents

#     def d2A_by_dXdcoilcurrents(self, compute_derivatives=1):
#         points = self.get_points_cart_ref()
#         npoints = len(points)
#         ncoils = len(self._coils)
#         if any([not self.fieldcache_get_status(f'dA_{i}') for i in range(ncoils)]):
#             assert compute_derivatives >= 1
#             self.compute(compute_derivatives)
#         self._d2A_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dA_{i}', [npoints, 3, 3]) for i in range(ncoils)]
#         return self._d2A_by_dXdcoilcurrents

#     def d3A_by_dXdXdcoilcurrents(self, compute_derivatives=2):
#         points = self.get_points_cart_ref()
#         npoints = len(points)
#         ncoils = len(self._coils)
#         if any([not self.fieldcache_get_status(f'ddA_{i}') for i in range(ncoils)]):
#             assert compute_derivatives >= 2
#             self.compute(compute_derivatives)
#         self._d3A_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddA_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]
#         return self._d3A_by_dXdXdcoilcurrents

#     def A_and_dA_vjp(self, v, vgrad):
#         r"""
#         Same as :obj:`simsopt.geo.biotsavart.BiotSavart.A_vjp` but returns the vector Jacobian product for :math:`A` and :math:`\nabla A`, i.e. it returns

#         .. math::

#             \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{A}_i \}_k, \{ \sum_{i=1}^{n} {\mathbf{v}_\mathrm{grad}}_i \cdot \partial_{\mathbf{c}_k} \nabla \mathbf{A}_i \}_k.
#         """

#         coils = self._coils
#         gammas = [coil.curve.gamma() for coil in coils]
#         gammadashs = [coil.curve.gammadash() for coil in coils]
#         currents = [coil.current.get_value() for coil in coils]
#         res_gamma = [np.zeros_like(gamma) for gamma in gammas]
#         res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
#         res_grad_gamma = [np.zeros_like(gamma) for gamma in gammas]
#         res_grad_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

#         points = self.get_points_cart_ref()
#         sopp.biot_savart_vector_potential_vjp_graph(points, gammas, gammadashs, currents, v,
#                                                     res_gamma, res_gammadash, vgrad, res_grad_gamma, res_grad_gammadash)

#         dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
#         res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
#         d2A_by_dXdcoilcurrents = self.d2A_by_dXdcoilcurrents()
#         res_grad_current = [np.sum(vgrad * d2A_by_dXdcoilcurrents[i]) for i in range(len(d2A_by_dXdcoilcurrents))]

#         res = (
#             sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))]),
#             sum([coils[i].vjp(res_grad_gamma[i], res_grad_gammadash[i], np.asarray([res_grad_current[i]])) for i in range(len(coils))])
#         )

#         return res

#     def A_vjp(self, v):
#         r"""
#         Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
#         :math:`\{\mathbf{A}_i\}_{i=1}^n`.
#         These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
#         This function returns the vector Jacobian product of this dependency, i.e.

#         .. math::

#             \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{A}_i \}_k.

#         """

#         coils = self._coils
#         gammas = [coil.curve.gamma() for coil in coils]
#         gammadashs = [coil.curve.gammadash() for coil in coils]
#         currents = [coil.current.get_value() for coil in coils]
#         res_gamma = [np.zeros_like(gamma) for gamma in gammas]
#         res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

#         points = self.get_points_cart_ref()
#         sopp.biot_savart_vector_potential_vjp_graph(points, gammas, gammadashs, currents, v,
#                                                     res_gamma, res_gammadash, [], [], [])
#         dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
#         res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
#         return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])

#     def as_dict(self, serial_objs_dict) -> dict:
#         d = super().as_dict(serial_objs_dict=serial_objs_dict)
#         d["points"] = self.get_points_cart()
#         return d

#     @classmethod
#     def from_dict(cls, d, serial_objs_dict, recon_objs):
#         decoder = GSONDecoder()
#         xyz = decoder.process_decoded(d["points"],
#                                       serial_objs_dict=serial_objs_dict,
#                                       recon_objs=recon_objs)
#         coils = decoder.process_decoded(d["coils"],
#                                         serial_objs_dict=serial_objs_dict,
#                                         recon_objs=recon_objs)
#         bs = cls(coils)
#         bs.set_points_cart(xyz)
#         return bs