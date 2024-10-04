import numpy as np
from jax import vjp, jacfwd, jvp, hessian, jacrev, grad
import jax.numpy as jnp

import simsoptpp as sopp
from .magneticfield import MagneticField
from .._core.json import GSONDecoder
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
from ..geo.jit import jit

__all__ = ['BiotSavart', 'JaxBiotSavart', 'CoilCoilNetForces', 'CoilSelfNetForces',
           'CoilCoilNetTorques', 'TotalVacuumEnergy', 'CoilCoilNetForces12',
           'CoilCoilNetTorques12']


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


class JaxBiotSavart(sopp.BiotSavart, MagneticField):
    r"""
    """
    def __init__(self, coils):
        self._coils = coils
        sopp.BiotSavart.__init__(self, coils)
        MagneticField.__init__(self, depends_on=coils)
        
        self.B_jax = jit(lambda curve_dofs, currents, p: self.B_pure(curve_dofs, currents, p))
        self.B_jax_reduced = jit(lambda curve_dofs: self.B_pure_reduced(curve_dofs))
        # self.get_dofs_jax = jit(self.get_dofs)
        # self.B_impl = jit(lambda p: self.B_pure(self.get_gammas(), self.get_currents(), p))
        self.dB_by_dcurvedofs_jax = jit(jacfwd(self.B_jax, argnums=0))
        self.dB_by_dcoilcurrents_jax = jit(jacfwd(self.B_jax, argnums=1))
        self.dB_by_dX_jax = jit(jacfwd(self.B_jax, argnums=2))  #jit(jacfwd(self.B_jax, argnums=2))
        self.dB_by_dcurvedofs_vjp_jax = jit(lambda x, v: vjp(self.B_jax_reduced, x)[1](v)[0])
        self.d2B_by_dXdX_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=2))
        self.d2B_by_dXdcoilcurrents_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=1))
        self.d2B_by_dXdcurvedofs_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=0))
        self.d3B_by_dXdXdcoilcurrents_jax = jit(jacfwd(self.d2B_by_dXdX_jax, argnums=1))

        # Seems like B_jax and A_jax should not use jit, since then
        # it does not update correctly when the dofs change.
        self.A_jax = jit(lambda curve_dofs, currents, p: self.A_pure(curve_dofs, currents, p))
        self.dA_by_dcurvedofs_jax = jit(jacfwd(self.A_jax, argnums=0))
        self.dA_by_dcoilcurrents_jax = jit(jacfwd(self.A_jax, argnums=1))
        self.dA_by_dX_jax = jit(jacfwd(self.A_jax, argnums=2))  #jit(jacfwd(self.A_jax, argnums=2))
        self.d2A_by_dXdX_jax = jit(jacfwd(self.dA_by_dX_jax, argnums=2))
        self.d2A_by_dXdcoilcurrents_jax = jit(jacfwd(self.dA_by_dX_jax, argnums=1))
        self.d2A_by_dXdcurvedofs_jax = jit(jacfwd(self.dA_by_dX_jax, argnums=0))
        self.d3A_by_dXdXdcoilcurrents_jax = jit(jacfwd(self.d2A_by_dXdX_jax, argnums=1))

        self.coil_coil_forces_jax = jit(lambda curve_dofs, currents: self.coil_coil_forces_pure(curve_dofs, currents))
        self.coil_coil_torques_jax = jit(lambda curve_dofs, currents: self.coil_coil_torques_pure(curve_dofs, currents))
        self.coil_coil_inductances_jax = jit(lambda curve_dofs, a, b: self.coil_coil_inductances_pure(curve_dofs, a, b))
        self.total_vacuum_energy_jax = jit(lambda curve_dofs, currents, a, b: self.total_vacuum_energy_pure(curve_dofs, currents, a, b))
        self.dtve_by_dX_jax = jacfwd(self.total_vacuum_energy_pure, argnums=0)
        self.net_forces_squared_jax = jit(lambda curve_dofs, currents: self.net_forces_squared_pure(curve_dofs, currents))
        self.net_torques_squared_jax = jit(lambda curve_dofs, currents: self.net_torques_squared_pure(curve_dofs, currents))
        self.coil_self_forces_jax = jit(lambda curve_dofs, currents, a, b: self.coil_self_forces_pure(curve_dofs, currents, a, b))
        self.net_self_forces_squared_jax = jit(lambda curve_dofs, currents, a, b: self.net_self_forces_squared_pure(curve_dofs, currents, a, b))

    # @jit
    def B_pure(self, curve_dofs, currents, points):
        """
        Assumes that the quadpoints are uniformly space on the curve!
        """
        # First two axes, over the total number of coils and the integral over the coil 
        # # points, should be summed
        if currents is None:
            currents = self.get_currents()
        if points is None:
            points = self.get_points_cart_ref()
        gammas = self.get_gammas(curve_dofs)
        p_minus_g = points[None, None, :, :] - gammas[:, :, None, :]
        B = jnp.sum(currents[:, None, None] * jnp.sum(jnp.cross(
            self.get_gammadashs(curve_dofs)[:, :, None, :],
            p_minus_g
        ) / jnp.linalg.norm(p_minus_g
                , axis=-1)[:, :, :, None] ** 3, axis=1), axis=0)
        return B * 1e-7 / jnp.shape(gammas)[1]  # Divide by number of quadpoints
    
    def B_pure_reduced(self, curve_dofs):
        """
        Assumes that the quadpoints are uniformly space on the curve!
        """
        # First two axes, over the total number of coils and the integral over the coil 
        # # points, should be summed
        points = self.get_points_cart_ref()
        currents = self.get_currents()
        gammas = self.get_gammas(curve_dofs)
        p_minus_g = points[None, None, :, :] - gammas[:, :, None, :]
        B = jnp.sum(currents[:, None, None] * jnp.sum(jnp.cross(
            self.get_gammadashs(curve_dofs)[:, :, None, :],
            p_minus_g
        ) / jnp.linalg.norm(p_minus_g
                , axis=-1)[:, :, :, None] ** 3, axis=1), axis=0)
        return B * 1e-7 / jnp.shape(gammas)[1]  # Divide by number of quadpoints
    
    # @jit
    def A_pure(self, curve_dofs, currents, points):
        """
        """
        gammas = self.get_gammas(curve_dofs)
        return jnp.sum(currents[:, None, None] * jnp.sum(
                self.get_gammadashs(curve_dofs)[:, :, None, :] / jnp.linalg.norm(
                gammas[:, :, None, :] - points[None, None, :, :], axis=-1
                )[:, :, :, None], axis=1), axis=0) * 1e-7 / jnp.shape(gammas)[1]  # Divide by number of quadpoints

    # @jit
    def B(self):
        return self.B_jax(self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref())
    #np.array(self.B_jax(self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()))
    
    def B_vjp(self, v):
        import time
        t1 = time.time()
        coils = self._coils
        # dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        # print(jnp.shape(v), jnp.shape(dB_by_dcoilcurrents), jnp.shape(v * dB_by_dcoilcurrents[0]))
        res_current = jnp.sum(jnp.sum(v[None, :, :] * self.dB_by_dcoilcurrents(), axis=-1), axis=-1)
        # print(jnp.shape(res_current))
        t2 = time.time()
        print('Current dJ time = ', t2 - t1)
        t1 = time.time()
        dB_by_dcurvedofs = self.dB_by_dcurvedofs()
        # t2 = time.time()
        # print('Curve dJ time = ', t2 - t1)
        # print(jnp.shape(dB_by_dcurvedofs))
        # t1 = time.time()
        # print(jnp.shape(v), jnp.shape(self.get_curve_dofs()))
        # print(jnp.shape(vjp(self.B_pure_reduced, self.get_curve_dofs())[1]))
        # print(vjp(self.B_pure_reduced, self.get_curve_dofs())[1](v),
        #       jnp.shape(vjp(self.B_jax_reduced, self.get_curve_dofs())[1](v)))
        # res_curvedofs = self.dB_by_dcurvedofs_vjp_impl(v)
        # print('res_curvedofs size = ', jnp.shape(res_curvedofs))
        # print(jnp.shape(v), jnp.shape(self.dB_by_dcurvedofs()))
        res_curvedofs = [jnp.sum(jnp.sum(v[None, :, :] * dB_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dB_by_dcurvedofs))]
        # print(jnp.shape(res_curvedofs), len(res_curvedofs))
        curve_derivs = [Derivative({coils[i].curve: res_curvedofs[i]}) for i in range(len(res_curvedofs))]
        current_derivs = [coils[i].current.vjp(np.array([res_current[i]])) for i in range(len(coils))]
        # t2 = time.time()
        # print('dJ construction time = ', t2 - t1)
        return sum(curve_derivs + current_derivs)
    
    def B_vjp_jax(self, v):
        import time
        t1 = time.time()
        coils = self._coils
        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        t2 = time.time()
        print('Current dJ time = ', t2 - t1)
        t1 = time.time()
        dB_by_dcurvedofs = self.dB_by_dcurvedofs()
        # t2 = time.time()
        # print('Curve dJ time = ', t2 - t1)
        # print(jnp.shape(dB_by_dcurvedofs))
        # t1 = time.time()
        # print(jnp.shape(v), jnp.shape(self.get_curve_dofs()))
        # print(jnp.shape(vjp(self.B_pure_reduced, self.get_curve_dofs())[1]))
        # print(vjp(self.B_pure_reduced, self.get_curve_dofs())[1](v),
        #       jnp.shape(vjp(self.B_jax_reduced, self.get_curve_dofs())[1](v)))
        # res_curvedofs = self.dB_by_dcurvedofs_vjp_impl(v)
        # print('res_curvedofs size = ', jnp.shape(res_curvedofs))
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dB_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dB_by_dcurvedofs))]
        curve_derivs = [Derivative({coils[i].curve: res_curvedofs[i]}) for i in range(len(res_curvedofs))]
        current_derivs = [coils[i].current.vjp(np.array([res_current[i]])) for i in range(len(coils))]
        t2 = time.time()
        print('dJ construction time = ', t2 - t1)
        return sum(curve_derivs + current_derivs)
    
    
    def dB_by_dcurvedofs_vjp_impl(self, v):
        r"""
        """
        return self.dB_by_dcurvedofs_vjp_jax(
            self.get_curve_dofs(), v
        )
    
    # @jit
    def A(self):
        return self.A_jax(self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref())

    def A_vjp(self, v):
        coils = self._coils
        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        dA_by_dcurvedofs = self.dA_by_dcurvedofs()
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dA_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dA_by_dcurvedofs))]
        curve_derivs = [Derivative({coils[i].curve: res_curvedofs[i]}) for i in range(len(res_curvedofs))]
        current_derivs = [coils[i].current.vjp(np.array([res_current[i]])) for i in range(len(coils))]
        return sum(curve_derivs + current_derivs)
    
    # def get_dofs(self):
    #     ll = [self._coils[i].current.get_value() for i in range(len(self._coils))]
    #     lc = [self._coils[i].curve.get_dofs() for i in range(len(self._coils))]
    #     return (ll + lc)
    
    def get_curve_dofs(self):
        # get the dofs of the UNIQUE coils (the rest of the coils don't have dofs because
        # they are obtained by symmetries)
        return jnp.array([self._coils[i].curve.get_dofs() for i in range(len(self._coils)
            ) if jnp.size(self._coils[i].curve.get_dofs()) != 0])

    def get_gammas(self, dofs):
        gammas = jnp.zeros((len(self._coils), len(self._coils[0].curve.quadpoints), 3))
        for i, c in enumerate(self._coils):
            gammas = gammas.at[i, :, :].add(c.curve.gamma_impl_jax(dofs[i % len(dofs)], self._coils[i].curve.quadpoints))
        return jnp.array(gammas)

    def get_quadpoints(self):
        return jnp.array([c.curve.quadpoints for c in self._coils])
    
    def get_kappas(self):
        dofs = self.get_curve_dofs()
        return jnp.array([c.curve.kappa_impl_jax(
            c.curve.gammadash_impl_jax(dofs[i], self._coils[i].curve.quadpoints), 
            c.curve.gammadashdash_impl_jax(dofs[i], self._coils[i].curve.quadpoints)
            ) for i, c in enumerate(self._coils)])
    
    def get_tangents(self):
        return jnp.array([c.curve.frenet_frame()[0] for c in self._coils])
    
    def get_binormals(self):
        return jnp.array([c.curve.frenet_frame()[-1] for c in self._coils])

    def get_currents(self):
        return jnp.array([c.current.get_value() for c in self._coils])

    def get_gammadashs(self, dofs):
        gammadashs = jnp.zeros((len(self._coils), len(self._coils[0].curve.quadpoints), 3))
        for i, c in enumerate(self._coils):
            gammadashs = gammadashs.at[i, :, :].add(
                c.curve.gammadash_impl_jax(dofs[i % len(dofs)], self._coils[i].curve.quadpoints))
        return jnp.array(gammadashs)
    
    def get_gammadashdashs(self, dofs):
        gammadashdashs = jnp.zeros((len(self._coils), len(self._coils[0].curve.quadpoints), 3))
        for i, c in enumerate(self._coils):
            gammadashdashs = gammadashdashs.at[i, :, :].add(
                c.curve.gammadashdash_impl_jax(dofs[i % len(dofs)], 
                                           self._coils[i].curve.quadpoints))
        return jnp.array(gammadashdashs)

    def dB_by_dX(self):
        r"""
        """
        temp = jnp.transpose(jnp.diagonal(self.dB_by_dX_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()), 
            axis1=0, axis2=2), axes=[2, 1, 0])
        return temp

    def dB_by_dcoilcurrents(self):
        r"""
        """
        return jnp.transpose(self.dB_by_dcoilcurrents_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()),
            [2, 0, 1])

    def dB_by_dcurvedofs(self):
        r"""
        Returns matrix of shape (ncoils, 3, 3, npoints)
        """
        # import time
        # t1 = time.time()
        # self.dB_by_dcurvedofs_jax(
        #     self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref())
        # t2 = time.time()
        # print('Time just to call the function = ', t2 - t1)
        return jnp.transpose(self.dB_by_dcurvedofs_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()), 
            axes=[2, 3, 0, 1])

    def d2B_by_dXdcoilcurrents(self):
        r"""
        """
        return jnp.transpose(jnp.diagonal(self.d2B_by_dXdcoilcurrents_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()
        ), axis1=0, axis2=2), axes=[2, 3, 1, 0])
    
    def d2B_by_dXdcurvedofs(self):
        r"""
        """
        return jnp.transpose(jnp.diagonal(self.d2B_by_dXdcurvedofs_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()
        ), axis1=0, axis2=2), axes=[2, 3, 4, 1, 0])

    def d2B_by_dXdX(self):
        return jnp.transpose(jnp.diagonal(
            jnp.diagonal(self.d2B_by_dXdX_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()
        ), axis1=0, axis2=2),
           axis1=2, axis2=4
        ),  axes=[3, 2, 1, 0])
    
    def d3B_by_dXdXdcoilcurrents(self):
        r"""
        """
        points = self.get_points_cart_ref()
        npoints = points.shape[0]
        temp1 = self.d3B_by_dXdXdcoilcurrents_jax(
            self.get_curve_dofs(), self.get_currents(), points
        )
        temp2 = jnp.zeros((npoints, 3, 3, 3, 2))
        for i in range(npoints):
            temp2 = temp2.at[i, :, :, :, :].add(temp1[i, :, i, :, i, :, :])
        return jnp.transpose(temp2, axes=[4, 0, 3, 2, 1])

    def B_and_dB_vjp(self, v, vgrad):
        r"""
        """
        coils = self._coils
        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        d2B_by_dXdcoilcurrents = self.d2B_by_dXdcoilcurrents()
        res_grad_current = [np.sum(vgrad * d2B_by_dXdcoilcurrents[i]) for i in range(len(d2B_by_dXdcoilcurrents))]
        dB_by_dcurvedofs = self.dB_by_dcurvedofs()
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dB_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dB_by_dcurvedofs))]
        d2B_by_dXdcurvedofs = self.d2B_by_dXdcurvedofs()
        res_grad_curvedofs = [np.sum(np.sum(np.sum(vgrad[None, :, :, :] * d2B_by_dXdcurvedofs[i], axis=-1), axis=-1), axis=-1) for i in range(len(d2B_by_dXdcurvedofs))]
        res = (
            sum([Derivative({coils[i].curve: res_curvedofs[i]}) + coils[i].current.vjp(np.array([res_current[i]])) for i in range(len(coils))]),
            sum([Derivative({coils[i].curve: res_grad_curvedofs[i]})  + coils[i].current.vjp(np.array([res_grad_current[i]])) for i in range(len(coils))]),
        )
        return res
    
    def dA_by_dX(self):
        r"""
        """
        temp = jnp.transpose(jnp.diagonal(self.dA_by_dX_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()), 
            axis1=0, axis2=2), axes=[2, 1, 0])
        return temp

    def dA_by_dcoilcurrents(self):
        r"""
        """
        return jnp.transpose(self.dA_by_dcoilcurrents_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()),
            [2, 0, 1])

    def dA_by_dcurvedofs(self):
        r"""
        """
        return jnp.transpose(self.dA_by_dcurvedofs_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()), 
            axes=[2, 3, 0, 1])

    def d2A_by_dXdcoilcurrents(self):
        r"""
        """
        return jnp.transpose(jnp.diagonal(self.d2A_by_dXdcoilcurrents_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()
        ), axis1=0, axis2=2), axes=[2, 3, 1, 0])

    def d2A_by_dXdcurvedofs(self):
        r"""
        """
        return jnp.transpose(jnp.diagonal(self.d2A_by_dXdcurvedofs_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()
        ), axis1=0, axis2=2), axes=[2, 3, 4, 1, 0])

    def d2A_by_dXdX(self):
        return jnp.transpose(jnp.diagonal(
            jnp.diagonal(self.d2A_by_dXdX_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()
        ), axis1=0, axis2=2),
           axis1=2, axis2=4
        ),  axes=[3, 2, 1, 0])
    
    def d3A_by_dXdXdcoilcurrents(self):
        r"""
        """
        points = self.get_points_cart_ref()
        npoints = points.shape[0]
        temp1 = self.d3A_by_dXdXdcoilcurrents_jax(
            self.get_curve_dofs(), self.get_currents(), points
        )
        temp2 = jnp.zeros((npoints, 3, 3, 3, 2))
        for i in range(npoints):
            temp2 = temp2.at[i, :, :, :, :].add(temp1[i, :, i, :, i, :, :])
        return jnp.transpose(temp2, axes=[4, 0, 3, 2, 1])

    def A_and_dA_vjp(self, v, vgrad):
        r"""
        """
        coils = self._coils
        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        d2A_by_dXdcoilcurrents = self.d2A_by_dXdcoilcurrents()
        res_grad_current = [np.sum(vgrad * d2A_by_dXdcoilcurrents[i]) for i in range(len(d2A_by_dXdcoilcurrents))]
        dA_by_dcurvedofs = self.dA_by_dcurvedofs()
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dA_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dA_by_dcurvedofs))]
        d2A_by_dXdcurvedofs = self.d2A_by_dXdcurvedofs()
        res_grad_curvedofs = [np.sum(np.sum(np.sum(vgrad[None, :, :, :] * d2A_by_dXdcurvedofs[i], axis=-1), axis=-1), axis=-1) for i in range(len(d2A_by_dXdcurvedofs))]
        res = (
            sum([Derivative({coils[i].curve: res_curvedofs[i]}) + coils[i].current.vjp(np.array([res_current[i]])) for i in range(len(coils))]),
            sum([Derivative({coils[i].curve: res_grad_curvedofs[i]})  + coils[i].current.vjp(np.array([res_grad_current[i]])) for i in range(len(coils))]),
        )
        return res

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
    
    def coil_coil_forces(self):
        r"""
        This function implements the curvature, :math:`\kappa(\varphi)`.
        """
        return self.coil_coil_forces_jax(self.get_curve_dofs(), self.get_currents())
    
    def coil_coil_forces_pure(self, curve_dofs, currents):
        """
        """
        eps = 1e-20  # small number to avoid blow up in the denominator when i = j
        gammas = self.get_gammas(curve_dofs)
        gammadashs = self.get_gammadashs(curve_dofs)
        # gamma and gammadash are shape (ncoils, nquadpoints, 3)
        r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :]  # Note, do not use the i = j indices
        gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1) 
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3

        # Double sum over each of the closed curves
        F = (currents[None, :] * currents[:, None])[:, :, None] * jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
        net_forces = -jnp.sum(F, axis=1) * 1e-7 / jnp.shape(gammas)[1] ** 2
        return net_forces
    
    def net_forces_squared_pure(self, curve_dofs, currents):
        # Minimize the sum of the net force magnitudes ^2 of every coil
        return jnp.sum(jnp.linalg.norm(self.coil_coil_forces_jax(curve_dofs, currents), axis=-1) ** 2)
    
    def net_self_forces_squared_pure(self, curve_dofs, currents, a, b):
        # Minimize the sum of the net SELF force magnitudes ^2 of every coil
        return jnp.sum(jnp.linalg.norm(self.coil_self_forces_jax(curve_dofs, currents, a=a, b=b), axis=-1) ** 2)

    def total_vacuum_energy_pure(self, curve_dofs, currents, a, b):
        """
        """
        Ii_Ij = (currents[None, :] * currents[:, None])
        Lij = self.coil_coil_inductances_pure(curve_dofs, a=a, b=b)
        U = 0.5 * jnp.sum(Ii_Ij * Lij)
        return U
    
    def dtve_by_dX(self, curve_dofs, currents, a, b):
        temp = self.dtve_by_dX_jax(curve_dofs, currents, a=a, b=b)
        # print(temp)
        return temp
    
    def coil_coil_inductances_pure(self, curve_dofs, a, b):
        """
        """
        eps = 1e-9  # small number to avoid blow up in the denominator when i = j
        gammas = self.get_gammas(curve_dofs)
        gammadashs = self.get_gammadashs(curve_dofs)

        # gamma and gammadash are shape (ncoils, nquadpoints, 3)
        r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :]  # Note, do not use the i = j indices
        gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1) 
        rij_norm = jnp.linalg.norm(r_ij + eps, axis=-1)

        # Double sum over each of the closed curves
        Lij = jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=3), axis=2
                      ) / jnp.shape(gammas)[1] ** 2
        Lij = jnp.subtract(Lij, jnp.diagonal(Lij))
        # Diagonal elements need to be fixed below

        # Now fix the i = j case using the Eq 11 regularization from Hurwitz/Landreman 2023
        # and also used in Guinchard/Hudson/Paul 2024
        k = (4 * b) / (3 * a) * jnp.arctan2(a, b) + (4 * a) / (3 * b) * jnp.arctan2(b, a) \
            + (b ** 2) / (6 * a ** 2) * jnp.log(b / a) + (a ** 2) / (6 * b ** 2) * jnp.log(a / b) \
            - (a ** 4 - 6 * a ** 2 * b ** 2 + b ** 4) / (6 * a ** 2 * b ** 2) * jnp.log(a / b + b / a)
        delta = jnp.exp(-25.0 / 6.0 + k)
        # print(k, delta, a, b)
        # print(jnp.shape(gammadash_prod), jnp.shape(rij_norm), jnp.shape(r_ij))
        # print('rij = ', r_ij)

        # Need below line with eps != 0, otherwise some of the tangent vectors become NaN
        rij_norm = jnp.sqrt(jnp.linalg.norm(r_ij + eps, axis=-1) ** 2 + delta * a * b)
        # print('rij_norm = ', rij_norm)
        # print('sqrt_term = ', jnp.linalg.norm(r_ij, axis=-1) ** 2, jnp.shape(jnp.linalg.norm(r_ij, axis=-1) ** 2))
        Lii = jnp.diagonal(jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=3), axis=2)
                           ) / jnp.shape(gammas)[1] ** 2
        # print('Lii = ', Lii)
        for i in range(jnp.shape(Lij)[0]):
            Lij = Lij.at[i, i].add(Lii[i]) 
        # print(Lii)

        # Equation 22 in Hurwitz/Landreman 2023 is even better and implemented below
        # gd_norm = jnp.linalg.norm(gammadashs, axis=-1)
        # quadpoints = self.get_quadpoints()
        # quad_diff = quadpoints[None, :, None, :] - quadpoints[:, None, :, None]  # Note quadpoints are in [0, 1]
        # integrand2 = gd_norm[:, None, :, None] ** 2 / jnp.sqrt(
        #     (2 - 2 * jnp.cos(2 * np.pi * quad_diff)) * gd_norm[:, None, :, None] ** 2 + delta * a * b)
        # print(jnp.shape(quad_diff), jnp.shape(integrand2), jnp.shape(gammas))
        # L1 = jnp.sum(gd_norm * jnp.log(64.0 / (delta * a * b) * gd_norm ** 2), axis=-1) / jnp.shape(gammas)[1]
        # gammadash_prod = np.diagonal(gammadash_prod, axis1=0, axis2=0)
        # rij_norm = np.diagonal(rij_norm, axis1=0, axis2=0)
        # L2 = jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=3), axis=2) / jnp.shape(gammas)[1] ** 2
        # L3 = jnp.sum(jnp.sum(integrand2, axis=3), axis=2) / jnp.shape(gammas)[1] ** 2
        # print(L1, L2, L3, jnp.shape(L2))
        # Lii = L1 + jnp.diagonal(L2 - L3)
        # print(Lii)

        # zero out the original diagonal elements and replace with Lii
        # for i in range(jnp.shape(Lij)[0]):
        #     Lij = Lij.at[i, i].add(Lii[i]) 
        # Lij = jnp.add(Lij, Lii)
        return Lij * 1e-7
    
    def coil_coil_inductances(self, a, b):
        r"""
        """
        return self.coil_coil_inductances_jax(self.get_curve_dofs(), a=a, b=b)
    
    def coil_self_forces_pure(self, curve_dofs, currents, a=None, b=None):
        """
        """
        eps = 1e-20  # small number to avoid blow up in the denominator when i = j
        gammas = self.get_gammas(curve_dofs)
        gammadashs = self.get_gammadashs(curve_dofs)
        gammadashdashs = self.get_gammadashdashs(curve_dofs)

        # gamma and gammadash are shape (ncoils, nquadpoints, 3)
        r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :]  # Note, do not use the i = j indices

        # Now fix the i = j case using the Eq 11 regularization from Hurwitz/Landreman 2023
        # and also used in Guinchard/Hudson/Paul 2024
        k = (4 * b) / (3 * a) * jnp.arctan2(a, b) + (4 * a) / (3 * b) * jnp.arctan2(b, a) \
            + (b ** 2) / (6 * a ** 2) * jnp.log(b / a) + (a ** 2) / (6 * b ** 2) * jnp.log(a / b) \
            - (a ** 4 - 6 * a ** 2 * b ** 2 + b ** 4) / (6 * a ** 2 * b ** 2) * jnp.log(a / b + b / a)
        delta = jnp.exp(-25.0 / 6.0 + k)
        rij_norm = jnp.sqrt(jnp.linalg.norm(r_ij + eps, axis=-1) ** 2 + delta * a * b)

        # Equation 23 in Hurwitz/Landreman 2023 is even better and implemented below
        kappas = self.get_kappas()
        tangents = self.get_tangents()
        binormals = self.get_binormals()
        gd_norm = jnp.linalg.norm(gammadashs, axis=-1)
        quadpoints = self.get_quadpoints()
        quad_diff = quadpoints[:, None, :] - quadpoints[:, :, None]  # Note quadpoints are in [0, 1]
        r_ii = jnp.transpose(jnp.diagonal(r_ij, axis1=0, axis2=1), axes=[3, 0, 1, 2])
        Breg1 = 0.5 * kappas[:, :, None] * binormals * (-2.0 + jnp.log(64.0 / (delta * a * b) * gd_norm ** 2))[:, :, None]
        Breg_integrand1 = jnp.cross(gammadashs[:, :, None, :], r_ii) / jnp.sqrt(
            jnp.linalg.norm(r_ii + eps, axis=-1) ** 2 + delta * a * b)[:, :, :, None] ** 3
        Breg_integrand2 = - jnp.cross(gammadashs, gammadashdashs)[:, :, None, :] * ((
            1.0 - jnp.cos(2 * np.pi * quad_diff)
        ) / (jnp.sqrt((2 - 2 * jnp.cos(2 * np.pi * quad_diff)
            ) * gd_norm[:, :, None] ** 2 + delta * a * b) ** 3))[:, :, :, None]
        
        # print(jnp.shape(gammadashs), jnp.shape(r_ii), r_ii)
        Breg = jnp.sum(jnp.cross(gammadashs[:, :, None, :], r_ii) / jnp.sqrt(
            jnp.linalg.norm(r_ii + eps, axis=-1) ** 2 + delta * a * b)[:, :, :, None] ** 3, axis=1)  / jnp.shape(gammas)[1]
        Breg3 = Breg1 + jnp.sum(Breg_integrand1 + Breg_integrand2, axis=1) / jnp.shape(gammas)[1]
        # print(jnp.shape(Breg))
        # print(Breg, Breg3)
        F_self = currents[:, None] ** 2 * jnp.sum(jnp.cross(tangents, Breg), axis=1) / jnp.shape(gammas)[1]
        # print(F_self)
        return F_self * 1e-7
    
    def coil_self_forces(self, a, b):
        r"""
        """
        return self.coil_self_forces_pure(self.get_curve_dofs(), self.get_currents(), a, b)
    
    def total_vacuum_energy(self, a=0.01, b=0.01):
        r"""
        """
        return self.total_vacuum_energy_jax(self.get_curve_dofs(), self.get_currents(), a=a, b=b)

    def coil_coil_torques_pure(self, curve_dofs, currents):
        """
        T = mu0 / 4pi * \oint_{C_1} \oint_{C_2} (dl_2 x r_12)(dl1 * r_12)/|r_12|^3
        """
        eps = 1e-20  # small number to avoid blow up in the denominator when i = j
        gammas = self.get_gammas(curve_dofs)
        gammadashs = self.get_gammadashs(curve_dofs)
        Ii_Ij = (currents[None, :] * currents[:, None])[:, :, None]
        # gamma and gammadash are shape (ncoils, nquadpoints, 3)
        gammas_i = gammas[None, :, None, :, :]
        gammas_j = gammas[:, None, :, None, :]
        r_ij = gammas_i - gammas_j  # Note, do not use the i = j indices
        cross1 = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs[:, None, :, None, :], cross1)
        cross3 = jnp.cross(gammas_i, cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3

        # Double sum over each of the closed curves
        T = Ii_Ij * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques = -jnp.sum(T, axis=1) * 1e-7 / jnp.shape(gammas)[1] ** 2
        return net_torques
    
    def net_torques_squared_pure(self, curve_dofs, currents):
        # Minimize the sum of the net force magnitudes ^2 of every coil
        return jnp.sum(jnp.linalg.norm(self.coil_coil_torques_jax(curve_dofs, currents), axis=-1) ** 2)

    def coil_coil_torques(self):
        r"""
        This function implements the curvature, :math:`\kappa(\varphi)`.
        """
        return self.coil_coil_torques_jax(self.get_curve_dofs(), self.get_currents())

class CoilCoilNetForces(Optimizable):
    r"""
    """
    def __init__(self, biot_savart):
        self.biot_savart = biot_savart
        self.grad_curvedofs = jit(jacfwd(self.biot_savart.net_forces_squared_jax, argnums=0))
        self.grad_currents = jit(jacfwd(self.biot_savart.net_forces_squared_jax, argnums=1))
        super().__init__(depends_on=[biot_savart])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.biot_savart.net_forces_squared_jax(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents()
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dF_by_dcurvedofs = self.grad_curvedofs(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents())
        dF_by_dcurrents = self.grad_currents(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents())
                
        coils = self.biot_savart._coils
        curve_derivs = [Derivative({coils[i].curve: dF_by_dcurvedofs[i]}) for i in range(len(dF_by_dcurvedofs))]
        current_derivs = [coils[i].current.vjp(np.array([dF_by_dcurrents[i]])) for i in range(len(coils))]
        return sum(curve_derivs + current_derivs)

    return_fn_map = {'J': J, 'dJ': dJ}


class CoilCoilNetForces12(Optimizable):
    r"""
    """
    def __init__(self, biot_savart1, biot_savart2):
        self.biot_savart1 = biot_savart1
        self.biot_savart2 = biot_savart2
        self.coil_coil_forces12_jax = jit(lambda curve_dofs1, currents1, curve_dofs2, currents2:
            self.coil_coil_forces12_pure(curve_dofs1, currents1, curve_dofs2, currents2))
        self.net_forces_squared12_jax = jit(lambda curve_dofs1, currents1, curve_dofs2, currents2: 
            self.net_forces_squared12_pure(curve_dofs1, currents1, curve_dofs2, currents2))
        self.grad_curvedofs1 = jit(jacfwd(self.net_forces_squared12_jax, argnums=0))
        self.grad_currents1 = jit(jacfwd(self.net_forces_squared12_jax, argnums=1))
        self.grad_curvedofs2 = jit(jacfwd(self.net_forces_squared12_jax, argnums=2))
        self.grad_currents2 = jit(jacfwd(self.net_forces_squared12_jax, argnums=3))
        super().__init__(depends_on=[biot_savart1, biot_savart2])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.net_forces_squared12_jax(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dF_by_dcurvedofs1 = self.grad_curvedofs1(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        dF_by_dcurrents1 = self.grad_currents1(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        dF_by_dcurvedofs2 = self.grad_curvedofs2(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        dF_by_dcurrents2 = self.grad_currents2(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        # dF_by_dcurvedofs = dF_by_dcurvedofs1 + dF_by_dcurvedofs2
        # dF_by_dcurrents = dF_by_dcurrents1 + dF_by_dcurrents2
                
        coils = self.biot_savart1._coils
        curve_derivs1 = [Derivative({coils[i].curve: dF_by_dcurvedofs1[i]}) for i in range(len(dF_by_dcurvedofs1))]
        current_derivs1 = [coils[i].current.vjp(np.array([dF_by_dcurrents1[i]])) for i in range(len(coils))]
        coils = self.biot_savart2._coils
        curve_derivs2 = [Derivative({coils[i].curve: dF_by_dcurvedofs2[i]}) for i in range(len(dF_by_dcurvedofs2))]
        current_derivs2 = [coils[i].current.vjp(np.array([dF_by_dcurrents2[i]])) for i in range(len(coils))]
        return sum(curve_derivs1 + current_derivs1 + curve_derivs2 + current_derivs2)

    def coil_coil_forces12_pure(self, curve_dofs1, currents1, curve_dofs2, currents2):
        """
        """
        eps = 1e-20  # small number to avoid blow up in the denominator when i = j
        gammas1 = self.biot_savart1.get_gammas(curve_dofs1)
        gammadashs1 = self.biot_savart1.get_gammadashs(curve_dofs1)
        gammas2 = self.biot_savart2.get_gammas(curve_dofs2)
        gammadashs2 = self.biot_savart2.get_gammadashs(curve_dofs2)
        Ii_Ij = (self.biot_savart1.get_currents()[None, :] * self.biot_savart2.get_currents()[:, None])[:, :, None]
        # gamma and gammadash are shape (ncoils, nquadpoints, 3)
        r_ij = gammas1[None, :, None, :, :] - gammas2[:, None, :, None, :]  # Note, do not use the i = j indices
        gammadash_prod = jnp.sum(gammadashs1[None, :, None, :, :] * gammadashs2[:, None, :, None, :], axis=-1) 
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3

        # Double sum over each of the closed curves
        F = Ii_Ij * jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
        # Get net forces on the coils2
        F2 = -jnp.sum(F, axis=1) * 1e-7 / jnp.shape(gammas1)[1] / jnp.shape(gammas2)[1]
        # Get net forces on the coils1
        F1 = -jnp.sum(F, axis=0) * 1e-7 / jnp.shape(gammas1)[1] / jnp.shape(gammas2)[1]
        return jnp.vstack((F1, F2))

    def coil_coil_forces12(self):
        return self.coil_coil_forces12_pure(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(), 
            self.biot_savart2.get_curve_dofs(),
            self.biot_savart2.get_currents())

    def net_forces_squared12_pure(self, curve_dofs1, currents1, curve_dofs2, currents2):
        # Minimize the sum of the net force magnitudes ^2 of every coil
        return jnp.sum(jnp.linalg.norm(self.coil_coil_forces12_jax(
            curve_dofs1, currents1, curve_dofs2, currents2), axis=-1) ** 2)

    return_fn_map = {'J': J, 'dJ': dJ}

class CoilSelfNetForces(Optimizable):
    r"""
    """
    def __init__(self, biot_savart, a=0.01, b=0.01):
        self.biot_savart = biot_savart
        self.grad_curvedofs = jit(jacfwd(self.biot_savart.net_self_forces_squared_jax, argnums=0))
        self.grad_currents = jit(jacfwd(self.biot_savart.net_self_forces_squared_jax, argnums=1))
        super().__init__(depends_on=[biot_savart])
        if a is None:
            self.a = 0.01
        else:
            self.a = a
        if b is None:
            self.b = self.a
        else:
            self.b = b

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.biot_savart.net_self_forces_squared_jax(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents(),
            self.a,
            self.b
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dF_by_dcurvedofs = self.grad_curvedofs(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents(),
            self.a,
            self.b
        )
        dF_by_dcurrents = self.grad_currents(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents(),
            self.a,
            self.b
        )
                
        coils = self.biot_savart._coils
        curve_derivs = [Derivative({coils[i].curve: dF_by_dcurvedofs[i]}) for i in range(len(dF_by_dcurvedofs))]
        current_derivs = [coils[i].current.vjp(np.array([dF_by_dcurrents[i]])) for i in range(len(coils))]
        return sum(curve_derivs + current_derivs)

    return_fn_map = {'J': J, 'dJ': dJ}

class TotalVacuumEnergy(Optimizable):
    r"""
    """
    def __init__(self, biot_savart, a=None, b=None):
        self.biot_savart = biot_savart
        self.grad_curvedofs = jit(jacfwd(self.biot_savart.total_vacuum_energy_jax, argnums=0))
        self.grad_currents = jit(jacfwd(self.biot_savart.total_vacuum_energy_jax, argnums=1))
        if a is None:
            self.a = 0.01
        else:
            self.a = a
        if b is None:
            self.b = self.a
        else:
            self.b = b
        super().__init__(depends_on=[biot_savart])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.biot_savart.total_vacuum_energy_jax(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents(),
            self.a,
            self.b
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dU_by_dcurvedofs = self.grad_curvedofs(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents(),
            self.a,
            self.b
            )
        dU_by_dcurrents = self.grad_currents(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents(),
            self.a,
            self.b
            )
                
        coils = self.biot_savart._coils
        curve_derivs = [Derivative({coils[i].curve: dU_by_dcurvedofs[i]}) for i in range(len(dU_by_dcurvedofs))]
        current_derivs = [coils[i].current.vjp(np.array([dU_by_dcurrents[i]])) for i in range(len(coils))]
        return sum(curve_derivs + current_derivs)

    return_fn_map = {'J': J, 'dJ': dJ}

class CoilCoilNetTorques(Optimizable):
    r"""
    """
    def __init__(self, biot_savart):
        self.biot_savart = biot_savart
        self.grad_curvedofs = jit(jacfwd(self.biot_savart.net_torques_squared_jax, argnums=0))
        self.grad_currents = jit(jacfwd(self.biot_savart.net_torques_squared_jax, argnums=1))
        super().__init__(depends_on=[biot_savart])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.biot_savart.net_torques_squared_jax(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents()
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dT_by_dcurvedofs = self.grad_curvedofs(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents())
        dT_by_dcurrents = self.grad_currents(
            self.biot_savart.get_curve_dofs(), 
            self.biot_savart.get_currents())
                
        coils = self.biot_savart._coils
        curve_derivs = [Derivative({coils[i].curve: dT_by_dcurvedofs[i]}) for i in range(len(dT_by_dcurvedofs))]
        current_derivs = [coils[i].current.vjp(np.array([dT_by_dcurrents[i]])) for i in range(len(coils))]
        return sum(curve_derivs + current_derivs)

    return_fn_map = {'J': J, 'dJ': dJ}

class CoilCoilNetTorques12(Optimizable):
    r"""
    """
    def __init__(self, biot_savart1, biot_savart2):
        self.biot_savart1 = biot_savart1
        self.biot_savart2 = biot_savart2
        self.coil_coil_torques12_jax = jit(lambda curve_dofs1, currents1, curve_dofs2, currents2:
            self.coil_coil_torques12_pure(curve_dofs1, currents1, curve_dofs2, currents2))
        self.net_torques_squared12_jax = jit(lambda curve_dofs1, currents1, curve_dofs2, currents2: 
            self.net_torques_squared12_pure(curve_dofs1, currents1, curve_dofs2, currents2))
        self.grad_curvedofs1 = jit(jacfwd(self.net_torques_squared12_jax, argnums=0))
        self.grad_currents1 = jit(jacfwd(self.net_torques_squared12_jax, argnums=1))
        self.grad_curvedofs2 = jit(jacfwd(self.net_torques_squared12_jax, argnums=2))
        self.grad_currents2 = jit(jacfwd(self.net_torques_squared12_jax, argnums=3))
        super().__init__(depends_on=[biot_savart1, biot_savart2])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.net_torques_squared12_jax(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dF_by_dcurvedofs1 = self.grad_curvedofs1(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        dF_by_dcurrents1 = self.grad_currents1(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        dF_by_dcurvedofs2 = self.grad_curvedofs2(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        dF_by_dcurrents2 = self.grad_currents2(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(),
            self.biot_savart2.get_curve_dofs(), 
            self.biot_savart2.get_currents()
            )
        # dF_by_dcurvedofs = dF_by_dcurvedofs1 + dF_by_dcurvedofs2
        # dF_by_dcurrents = dF_by_dcurrents1 + dF_by_dcurrents2
                
        coils = self.biot_savart1._coils
        curve_derivs1 = [Derivative({coils[i].curve: dF_by_dcurvedofs1[i]}) for i in range(len(dF_by_dcurvedofs1))]
        current_derivs1 = [coils[i].current.vjp(np.array([dF_by_dcurrents1[i]])) for i in range(len(coils))]
        coils = self.biot_savart2._coils
        curve_derivs2 = [Derivative({coils[i].curve: dF_by_dcurvedofs2[i]}) for i in range(len(dF_by_dcurvedofs2))]
        current_derivs2 = [coils[i].current.vjp(np.array([dF_by_dcurrents2[i]])) for i in range(len(coils))]
        return sum(curve_derivs1 + current_derivs1 + curve_derivs2 + current_derivs2)

    def coil_coil_torques12_pure(self, curve_dofs1, currents1, curve_dofs2, currents2):
        eps = 1e-20  # small number to avoid blow up in the denominator when i = j
        gammas1 = self.biot_savart1.get_gammas(curve_dofs1)
        gammadashs1 = self.biot_savart1.get_gammadashs(curve_dofs1)
        gammas2 = self.biot_savart2.get_gammas(curve_dofs2)
        gammadashs2 = self.biot_savart2.get_gammadashs(curve_dofs2)
        Ii_Ij = (self.biot_savart1.get_currents()[None, :] * self.biot_savart2.get_currents()[:, None])[:, :, None]
        # gamma and gammadash are shape (ncoils, nquadpoints, 3)
        r_ij = gammas1[None, :, None, :, :] - gammas2[:, None, :, None, :]  # Note, do not use the i = j indices
        gammadash_i = gammadashs1[None, :, None, :, :]
        gammadash_j = gammadashs2[:, None, :, None, :]
        cross1 = jnp.cross(gammadash_i, r_ij)
        cross2 = jnp.cross(gammadash_j, cross1)
        cross3 = jnp.cross(
            gammas1[None, :, None, :, :] + gammas2[:, None, :, None, :], 
            cross2
        )  # gammas_i here originally
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3

        # Double sum over each of the closed curves
        T = Ii_Ij * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        # Get net torques on the coils2
        T2 = -jnp.sum(T, axis=1) * 1e-7 / jnp.shape(gammas1)[1] / jnp.shape(gammas2)[1]
        # Get net torques on the coils1
        T1 = -jnp.sum(T, axis=0) * 1e-7 / jnp.shape(gammas1)[1] / jnp.shape(gammas2)[1]
        return jnp.vstack((T1, T2))

    def coil_coil_torques12(self):
        return self.coil_coil_torques12_pure(
            self.biot_savart1.get_curve_dofs(), 
            self.biot_savart1.get_currents(), 
            self.biot_savart2.get_curve_dofs(),
            self.biot_savart2.get_currents())

    def net_torques_squared12_pure(self, curve_dofs1, currents1, curve_dofs2, currents2):
        # Minimize the sum of the net force magnitudes ^2 of every coil
        return jnp.sum(jnp.linalg.norm(self.coil_coil_torques12_jax(
            curve_dofs1, currents1, curve_dofs2, currents2), axis=-1) ** 2)

    return_fn_map = {'J': J, 'dJ': dJ}