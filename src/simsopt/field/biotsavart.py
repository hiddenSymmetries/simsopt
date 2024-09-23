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
        # print(np.shape(self.dB_by_dX()))
        # exit()
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
    def __init__(self, coils, **kwargs):
        if "external_dof_setter" not in kwargs:
            kwargs["external_dof_setter"] = sopp.Curve.set_dofs_impl
        self._coils = coils
        sopp.BiotSavart.__init__(self, coils)
        MagneticField.__init__(self, depends_on=coils)

        self.B_jax = jit(lambda dofs: self.B_pure())
        self.B_impl_jax = jit(lambda dofs: self.B_pure())
        self.dB_by_dX_jax = jit(jacfwd(self.B_jax))
        self.dB_by_dX_vjp_jax = jit(lambda x, v: vjp(self.B_jax, x)[1](v)[0])
        self.dB_by_dcoilcurrents_jax = jit(jacfwd(self.B_jax))
        self.dB_by_dcoilcurrents_vjp_jax = jit(lambda x, v: vjp(self.B_jax, x)[1](v)[0])
        self.d2B_by_dXdX_jax = jit(hessian(self.B_jax))
        self.d2B_by_dXdX_vjp_jax = jit(lambda x, v: vjp(self.dB_by_dX_jax, x)[1](v)[0])
        self.d2B_by_dXdcoilcurrents_jax = jit(jacfwd(self.dB_by_dX_jax))
        self.d2B_by_dXdcoilcurrents_vjp_jax = jit(lambda x, v: vjp(self.dB_by_dX_jax, x)[1](v)[0])
        self.d3B_by_dXdXdcoilcurrents_jax = jit(jacfwd(self.d2B_by_dXdX_jax))
        self.d3B_by_dXdXdcoilcurrents_vjp_jax = jit(lambda x, v: vjp(self.d2B_by_dXdX_vjp_jax, x)[1](v)[0])

        self.A_jax = jit(lambda dofs: self.A_pure())
        self.A_impl_jax = jit(lambda dofs: self.A_pure())
        self.dA_by_dX_jax = jit(jacfwd(self.A_jax))
        self.dA_by_dX_vjp_jax = jit(lambda x, v: vjp(self.A_jax, x)[1](v)[0])
        self.dA_by_dcoilcurrents_jax = jit(jacfwd(self.A_jax))
        self.dA_by_dcoilcurrents_vjp_jax = jit(lambda x, v: vjp(self.A_jax, x)[1](v)[0])
        self.d2A_by_dXdX_jax = jit(hessian(self.A_jax))
        self.d2A_by_dXdX_vjp_jax = jit(lambda x, v: vjp(self.dA_by_dX_jax, x)[1](v)[0])
        self.d2A_by_dXdcoilcurrents_jax = jit(hessian(self.A_jax))
        self.d2A_by_dXdcoilcurrents_vjp_jax = jit(lambda x, v: vjp(self.dA_by_dcoilcurrents_jax, x)[1](v)[0])
        self.d3A_by_dXdXdcoilcurrents_jax = jit(jacrev(jacfwd(jacrev(self.A_jax))))
        self.d3A_by_dXdXdcoilcurrents_vjp_jax = jit(lambda x, v: vjp(self.d2A_by_dXdX_vjp_jax, x)[1](v)[0])

    # @jit
    def B_pure(self):
        """
        """
        points = self.get_points_cart_ref()
        I = jnp.array([coil.current.get_value() for coil in self._coils])
        gammas = jnp.array([coil.curve.gamma() for coil in self._coils])
        gammadashs = jnp.array([coil.curve.gammadash() for coil in self._coils])
        # (2, 200, 3) (17, 3), (2,)
        # exit()
        B = jnp.zeros((len(points), 3))
        # First two axes, over the total number of coils and the integral over the coil 
        # points, should be summed
        B = jnp.sum(I[:, None, None] * jnp.sum(jnp.cross(gammas[:, :, None, :] - points[None, None, :, :], 
            gammadashs[:, :, None, :]) / jnp.linalg.norm(
                gammas[:, :, None, :] - points[None, None, :, :], axis=-1)[:, :, :, None] ** 3, axis=1), axis=0)
        # print(jnp.shape(gammas[:, :, None, :]), jnp.shape(B), jnp.shape(points[None, None, :, :]), jnp.shape(I[:, None, None]))
        return B * 1e-7
    
    @jit
    def A_pure(self):
        """
        """
        points = self.get_points_cart_ref()
        I = jnp.array([c.current.get_value() for c in self._coils])
        gammas = jnp.array([coil.curve.gamma() for coil in self._coils])
        gammadashs = jnp.array([coil.curve.gammadash() for coil in self._coils])
        A = jnp.zeros((len(points), 3))
        A = jnp.sum(I[:, None, None] * jnp.sum(gammadashs / jnp.linalg.norm(
                gammas[:, :, None, :] - points[None, None, :, :], axis=-1), axis=1), axis=0)
        return A * 1e-7
    
    def dB_by_dcoilcurrents(self):
        r"""
        """
        return jnp.transpose(self.dB_by_dcoilcurrents_jax(
            jnp.array([c.current.get_value() for c in self._coils])), [2, 0, 1])
    
    def dB_by_dcoilcurrents_vjp_impl(self, v):
        return self.dB_by_dcoilcurrents_vjp_jax(jnp.array([c.current.get_value() for c in self._coils]), v)
        
    def dB_by_dX(self):
        r"""
        """
        points = self.get_points_cart_ref()
        print(jnp.shape(jnp.diagonal(self.dB_by_dX_jax(
            points), axis1=0, axis2=2)))
        # exit()
        return jnp.diagonal(self.dB_by_dX_jax(points), axis1=0, axis2=2)
        
    def dB_by_dX_vjp_impl(self, v):
        return self.dB_by_dX_vjp_jax(jnp.array([c.curve.gamma() for c in self._coils]), v)

    def d2B_by_dXdcoilcurrents(self):
        r"""
        """
        print(jnp.shape(self.d2B_by_dXdcoilcurrents_jax(
            jnp.array([c.current.get_value() for c in self._coils]))))
        exit()
        return jnp.transpose(self.d2B_by_dXdcoilcurrents_jax(
            jnp.array([c.current.get_value() for c in self._coils])
        ), [1, 2, 3, 0])

    def d2B_by_dXdcoilcurrents_vjp_impl(self, v):
        r"""
        """
        return self.d2B_by_dXdcoilcurrents_vjp_jax(
            jnp.array([c.curve.gamma() for c in self._coils]), 
            jnp.array([c.current.get_value() for c in self._coils]),
            v
        )
    
    def d2B_by_dXdX_impl(self, d2B_by_dXdX):
        r"""
        """
        d2B_by_dXdX[:, :, :, :] = self.d2B_by_dXdX_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.curve.gamma() for c in self._coils])
        )

    def d2B_by_dXdX_vjp_impl(self, v):
        r"""
        """
        return self.d2B_by_dXdX_vjp_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.curve.gamma() for c in self._coils]),
            v
        )
    
    def d3B_by_dXdXdcoilcurrents(self, d3B_by_dXdXdcoilcurrents):
        r"""
        """
        return self.d3B_by_dXdXdcoilcurrents_jax(
            jnp.array([c.current.get_value() for c in self._coils])
        )

    def d3B_by_dXdXdcoilcurrents_vjp_impl(self, v):
        r"""
        """
        return self.d3B_by_dXdXdcoilcurrents_vjp_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.current.get_value() for c in self._coils]),
            v
        )

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
        print(jnp.shape(d2B_by_dXdcoilcurrents), jnp.shape(d2B_by_dXdcoilcurrents[0]), jnp.shape(vgrad))
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

    def dA_by_dcoilcurrents_impl(self, dA_by_dcoilcurrents):
        r"""
        """
        dA_by_dcoilcurrents[:, :, :] = self.dA_by_dcoilcurrents_jax(
            jnp.array([c.current.get_value() for c in self._coils]))
    
    def dA_by_dcoilcurrents_vjp_impl(self, v):
        return self.dA_by_dcoilcurrents_vjp_jax(jnp.array([c.current.get_value() for c in self._coils]), v)
        
    def dA_by_dX_impl(self, dA_by_dX):
        r"""
        """
        dA_by_dX[:, :, :] = self.dA_by_dX_jax(
            jnp.array([c.curve.gamma() for c in self._coils]))
        
    def dA_by_dX_vjp_impl(self, v):
        return self.dA_by_dX_vjp_jax(jnp.array([c.curve.gamma() for c in self._coils]), v)

    def d2A_by_dXdcoilcurrents_impl(self, d2A_by_dXdcoilcurrents):
        r"""
        """
        d2A_by_dXdcoilcurrents[:, :, :, :] = self.d2A_by_dXdcoilcurrents_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.current.get_value() for c in self._coils])
        )

    def d2A_by_dXdcoilcurrents_vjp_impl(self, v):
        r"""
        """
        return self.d2A_by_dXdcoilcurrents_vjp_jax(
            jnp.array([c.curve.gamma() for c in self._coils]), 
            jnp.array([c.current.get_value() for c in self._coils]),
            v
        )
    
    def d2A_by_dXdX_impl(self, d2A_by_dXdX):
        r"""
        """
        d2A_by_dXdX[:, :, :, :] = self.d2A_by_dXdX_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.curve.gamma() for c in self._coils])
        )

    def d2A_by_dXdX_vjp_impl(self, v):
        r"""
        """
        return self.d2A_by_dXdX_vjp_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.curve.gamma() for c in self._coils]),
            v
        )
    
    def d3A_by_dXdXdcoilcurrents_impl(self, d3A_by_dXdXdcoilcurrents):
        r"""
        """
        d3A_by_dXdXdcoilcurrents_impl[:, :, :, :, :] = self.d3A_by_dXdXdcoilcurrents_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.current.get_value() for c in self._coils])
        )

    def d3A_by_dXdXdcoilcurrents_vjp_impl(self, v):
        r"""
        """
        return self.d3A_by_dXdXdcoilcurrents_vjp_jax(
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.curve.gamma() for c in self._coils]),
            jnp.array([c.current.get_value() for c in self._coils]),
            v
        )

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