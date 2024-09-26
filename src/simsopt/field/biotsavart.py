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
        self.dB_by_dcoilcurrents_jax = jit(jacfwd(self.B_jax, argnums=1))
        self.dB_by_dX_jax = jacfwd(self.B_jax, argnums=2)  #jit(jacfwd(self.B_jax, argnums=2))
        self.d2B_by_dXdX_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=2))
        self.d2B_by_dXdcoilcurrents_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=1))
        self.d2B_by_dXdcurvedofs_jax = jit(jacfwd(self.dB_by_dX_jax, argnums=0))
        self.d3B_by_dXdXdcoilcurrents_jax = jit(jacfwd(self.d2B_by_dXdX_jax, argnums=1))

        # Seems like B_jax and A_jax should not use jit, since then
        # it does not update correctly when the dofs change.
        self.A_jax = lambda curve_dofs, currents, p: self.A_pure(curve_dofs, currents, p)
        self.dA_by_dcurvedofs_jax = jit(jacfwd(self.A_jax, argnums=0))
        self.dA_by_dcoilcurrents_jax = jit(jacfwd(self.A_jax, argnums=1))
        self.dA_by_dX_jax = jacfwd(self.A_jax, argnums=2)  #jit(jacfwd(self.A_jax, argnums=2))
        self.d2A_by_dXdX_jax = jit(jacfwd(self.dA_by_dX_jax, argnums=2))
        self.d2A_by_dXdcoilcurrents_jax = jit(jacfwd(self.dA_by_dX_jax, argnums=1))
        self.d2A_by_dXdcurvedofs_jax = jit(jacfwd(self.dA_by_dX_jax, argnums=0))
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
        return np.array(self.B_jax(self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref()))
    
    def B_vjp(self, v):
        coils = self._coils
        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        dB_by_dcurvedofs = self.dB_by_dcurvedofs()
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dB_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dB_by_dcurvedofs))]
        return sum([Derivative({coils[i].curve: res_curvedofs[i]}) + coils[i].current.vjp(np.array([res_current[i]])) for i in range(len(coils))])
    
    def A(self):
        return self.A_jax(self.get_curve_dofs(), self.get_currents(), self.get_points_cart_ref())

    def A_vjp(self, v):
        coils = self._coils
        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        dA_by_dcurvedofs = self.dA_by_dcurvedofs()
        res_curvedofs = [np.sum(np.sum(v[None, :, :] * dA_by_dcurvedofs[i], axis=-1), axis=-1) for i in range(len(dA_by_dcurvedofs))]
        return sum([Derivative({coils[i].curve: res_curvedofs[i]}) + coils[i].current.vjp(np.array([res_current[i]])) for i in range(len(coils))])
    
    def get_dofs(self):
        ll = [self._coils[i].current.get_value() for i in range(len(self._coils))]
        lc = [self._coils[i].curve.get_dofs() for i in range(len(self._coils))]
        return (ll + lc)
    
    def get_curve_dofs(self):
        print([self._coils[i].curve.get_dofs() for i in range(len(self._coils)
            ) if jnp.size(self._coils[i].curve.get_dofs()) != 0])
        # get the dofs of the UNIQUE coils (the rest of the coils don't have dofs because
        # they are obtained by symmetries)
        return jnp.array([self._coils[i].curve.get_dofs() for i in range(len(self._coils)
            ) if jnp.size(self._coils[i].curve.get_dofs()) != 0])

    def get_gammas(self, dofs):
        gammas = np.zeros((len(self._coils), len(self._coils[0].curve.quadpoints), 3))
        [c.curve.gamma_impl(gammas[i, :, :], self._coils[i].curve.quadpoints
                                                    ) for i, c in enumerate(self._coils)]
        # What to do with dofs parameter?
        return jnp.array(gammas)

    def get_currents(self):
        return jnp.array([c.current.get_value() for c in self._coils])

    def get_gammadashs(self, dofs):
        gammadashs = np.zeros((len(self._coils), len(self._coils[0].curve.quadpoints), 3))
        [c.curve.gammadash_impl(gammadashs[i, :, :]) for i, c in enumerate(self._coils)]
        # What to do with dofs parameter?
        return jnp.array(gammadashs)

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

# @jit
    def coil_coil_forces_pure(self, curve_dofs, currents):
        """
        """
        eps = 1e-20  # small number to avoid blow up in the denominator when i = j
        gammas = self.get_gammas(curve_dofs)
        gammadashs = self.get_gammadashs(curve_dofs)
        Ii_Ij = (currents[None, :] * currents[:, None])[:, :, None]
        # gamma and gammadash are shape (ncoils, nquadpoints, 3)
        r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :]  # Note, do not use the i = j indices
        gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1) 
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3

        # Double sum over each of the closed curves
        F = Ii_Ij * jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
        net_forces = -jnp.sum(F, axis=1) * 1e-7 / jnp.shape(gammas)[1] ** 2
        return net_forces
    
    def coil_coil_forces(self):
        r"""
        This function implements the curvature, :math:`\kappa(\varphi)`.
        """
        return self.coil_coil_forces_pure(self.get_curve_dofs(), self.get_currents())

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
    
    
