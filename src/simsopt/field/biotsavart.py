import numpy as np
from jax import vjp, jacfwd, jvp, hessian, jacrev
import jax.numpy as jnp

import simsoptpp as sopp
from .magneticfield import MagneticField
from .._core.json import GSONDecoder
from .._core.derivative import Derivative
from ..geo.jit import jit

__all__ = ['BiotSavart', 'JaxBiotSavart']


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
        
        self.B_jax = lambda curve_dofs, currents, p: self.B_pure(curve_dofs, currents, p)
        # self.B_impl = jit(lambda p: self.B_pure(self.get_gammas(), self.get_currents(), p))
        self.dB_by_dcurvedofs_jax = jit(jacfwd(self.B_jax, argnums=0))

        # B_vjp returns v * dB/dcurvedofs
        # self.dB_vjp_jax = jit(lambda x, y, z, v: vjp(self.B_jax, x, y, z)[1](v)[0])
        self.dB_by_dcoilcurrents_jax = jit(jacfwd(self.B_jax, argnums=1))
        self.dB_by_dX_jax = jacfwd(self.B_jax, argnums=2)  #jit(jacfwd(self.B_jax, argnums=2))
        self.d2B_by_dXdX_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=2))
        self.d2B_by_dXdcoilcurrents_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=1))
        self.d3B_by_dXdXdcoilcurrents_jax = jit(jacfwd(self.d2B_by_dXdX_jax, argnums=1))

        # Seems like B_jax and A_jax should not use jit, since then
        # it does not update correctly when the dofs change.
        self.A_jax = lambda curve_dofs, currents, p: self.A_pure(curve_dofs, currents, p)
        # self.A_impl = jit(lambda p: self.A_pure(self.get_curve_dofs(), self.get_currents(), p))
        # self.A_vjp_jax = jit(lambda x, v: vjp(self.A_pure, x)[1](v)[0])
        self.dA_by_dcurvedofs_jax = jit(jacfwd(self.A_jax, argnums=0))
        self.dA_by_dcoilcurrents_jax = jit(jacfwd(self.A_jax, argnums=1))
        self.dA_by_dX_jax = jacfwd(self.A_jax, argnums=2)  #jit(jacfwd(self.A_jax, argnums=2))
        self.d2A_by_dXdX_jax = jit(hessian(self.A_jax, argnums=2))
        self.d2A_by_dXdcoilcurrents_jax = jit(jacfwd(self.dA_by_dX_jax, argnums=1))
        self.d3A_by_dXdXdcoilcurrents_jax = jit(jacfwd(self.d2A_by_dXdX_jax, argnums=1))

    # @jit
    def B_pure(self, curve_dofs, currents, points):
        """
        Assumes that the quadpoints are uniformly space on the curve!
        """
        # First two axes, over the total number of coils and the integral over the coil 
        # # points, should be summed
        gammas = self.get_gammas(curve_dofs)
        p_minus_g = points[None, None, :, :] - gammas[:, :, None, :]
        denom = 1.0 / jnp.linalg.norm(p_minus_g
                , axis=-1)[:, :, :, None] ** 3
        B = jnp.sum(currents[:, None, None] * jnp.sum(jnp.cross(
            -p_minus_g, 
            self.get_gammadashs(curve_dofs)[:, :, None, :]
        ) * denom, axis=1), axis=0)
        return B * 1e-7 / jnp.shape(gammas)[1]  # Divide by number of quadpoints
    
    # @jit
    def A_pure(self, curve_dofs, currents, points):
        """
        """
        gammas = self.get_gammas(curve_dofs)
        gammadashs = self.get_gammadashs(curve_dofs)
        A = jnp.sum(currents[:, None, None] * jnp.sum(gammadashs[:, :, None, :] / jnp.linalg.norm(
                gammas[:, :, None, :] - points[None, None, :, :], axis=-1)[:, :, :, None], axis=1), axis=0)
        return A * 1e-7 / jnp.shape(gammas)[1]  # Divide by number of quadpoints

    def B(self):
        return self.B_jax(self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref())
    
    def B_vjp(self, v):
        coils = self._coils
        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        dB_by_dcurvedofs = self.dB_by_dcurvedofs()
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dB_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dB_by_dcurvedofs))]
        return sum([Derivative({coils[i].curve: res_curvedofs[i]}) + Derivative({coils[i].current: res_current[i]}) for i in range(len(coils))])

    def A(self):
        return self.A_jax(self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref())

    def A_vjp(self, v):
        coils = self._coils
        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        dA_by_dcurvedofs = self.dA_by_dcurvedofs()
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dA_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dA_by_dcurvedofs))]
        print(jnp.shape(v), jnp.shape(dA_by_dcoilcurrents), jnp.shape(res_current), jnp.shape(dA_by_dcurvedofs), jnp.shape(res_curvedofs))
        print(Derivative({coils[0].current: res_current[0]}), coils[0].current, res_current[0])
        return sum([Derivative({coils[i].curve: res_curvedofs[i]}) + Derivative({coils[i].current: res_current[i]}) for i in range(len(coils))])
    
    def get_dofs(self):
        ll = [self._coils[i].current.get_value() for i in range(len(self._coils))]
        lc = [self._coils[i].curve.get_dofs() for i in range(len(self._coils))]
        return (ll + lc)
    
    def get_curve_dofs(self):
        return jnp.array([self._coils[i].curve.get_dofs() for i in range(len(self._coils))])

    def get_gammas(self, dofs):
        return jnp.array([c.curve.gamma_jax(dofs[i]) for i, c in enumerate(self._coils)])

    def get_currents(self):
        return jnp.array([c.current.get_value() for c in self._coils])

    def get_gammadashs(self, dofs):
        # curve_dofs = self.get_curve_dofs()
        return jnp.array([coil.curve.gammadash_impl_jax(
            dofs[i], coil.curve.quadpoints) for i, coil in enumerate(self._coils)]
        )

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
        """
        return jnp.transpose(self.dB_by_dcurvedofs_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()), 
            axes=[2, 3, 0, 1])

    def d2B_by_dXdcoilcurrents(self):
        r"""
        """
        return jnp.transpose(jnp.diagonal(self.d2B_by_dXdcoilcurrents_jax(
            self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()
        ), axis1=0, axis2=2), axes=[2, 3, 1, 0])

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

# @jit
    # def coil_coil_forces_pure(self, gammas, currents):
    #     """
    #     """
    #     currents_i = currents
    #     currents_j = currents
    #     gammas_i = gammas
    #     gammas_j = gammas
    #     gammadashs_i = self.get_gammadashs()
    #     gammadashs_j = gammadashs_i
    #     r_ij = gammas_i[None, :, :] - gammas_j[:, None, :]  # Note, do not use the i = j indices
    #     jnp.diag(r_ij[:, :, 0]) = 1e100
    #     jnp.diag(r_ij[:, :, 1]) = 1e100
    #     jnp.diag(r_ij[:, :, 2]) = 1e100
    #     F = jnp.cross(currents_i * gammadashs_i, 
    #                   jnp.cross(currents_j * gammadashs_j, r_ij)
    #     ) / jnp.linalg.norm(r_ij, axis=-1)[:, :, None] ** 3
    #     return F * 1e-7 / jnp.shape(gammas_i)[1] ** 2

    def coil_coil_inductances_pure(self, curve_dofs):
        """
            Using the Double integral Neumann formula for two infinitely thin wires.
            Requires a model for the self-inductance.
        """
        gammas = self.get_gammas(curve_dofs)
        gammadashs = self.get_gammas(curve_dofs)
        r_ij = gammas[None, :, :] - gammas[:, None, :]  # Note, do not use the i = j indices
        # jnp.diag(r_ij[:, :, 0]) = 1e100
        # jnp.diag(r_ij[:, :, 1]) = 1e100
        # jnp.diag(r_ij[:, :, 2]) = 1e100
        L = (jnp.sum(gammadashs * gammadashs, axis=-1)
            ) / jnp.linalg.norm(r_ij, axis=-1)
        return L * 1e-7 / jnp.shape(gammas)[1] ** 2
    
    def total_magnetic_energy_pure(self, curve_dofs, currents):
        return jnp.dot(currents, jnp.dot(self.coil_coil_inductances_pure(curve_dofs), currents)) * 0.5
    
    
