"""
Implements strain optimization for HTS coils
"""

import numpy as np
import jax.numpy as jnp
from jax import vjp, jvp, grad
import simsoptpp as sopp
from simsopt.geo.jit import jit
from simsopt.geo import ZeroRotation, Curve
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.curveobjectives import Lp_curvature_pure

# class TorsionalStrainPenalty(Optimizable):

#     def __init__(self, curvefilament, width=3, max_strain):
#         self.curvefilament = curvefilament
#         self.width = width
#         self.max_strain = max_strain
#         super().__init__(depends_on=[curvefilament])

#     def J(self):
#         return Lp_curvature_pure(self.curvefilament.torsional_strain,
#             self.curvefilament.curve.gammadash, p, self.max_strain)

#     @derivative_dec
#     def dJ(self):


class StrainOpt(Optimizable):
    """Class for strain optimization"""

    def __init__(self, framedcurve, width=3):
        self.framedcurve = framedcurve
        self.width = width
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width: strain_opt_pure(
            gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width))
        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width: grad(
            self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width: grad(
            self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width: grad(
            self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width))
        self.thisgrad3 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width: grad(
            self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width))
        self.thisgrad4 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width: grad(
            self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width))
        self.thisgrad5 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width: grad(
            self.J_jax, argnums=5)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width))
        super().__init__(depends_on=[framedcurve])

    def torsional_strain(self):
        """Exports torsion along a coil for a StrainOpt object"""
        torsion = self.framedcurve.frame_torsion()
        return torsion**2 * self.width**2 / 12  # From 2020 Paz-Soldan

    def binormal_curvature_strain(self):
        binormal_curvature = self.framedcurve.frame_binormal_curvature()
        return (self.width/2)*binormal_curvature  # From 2020 Paz-Soldan

    # def J(self):
    #     """
    #     This returns the value of the quantity.
    #     """
    #     gamma = self.curve.curve.gamma()
    #     d1gamma = self.curve.curve.gammadash()
    #     d2gamma = self.curve.curve.gammadashdash()
    #     d3gamma = self.curve.curve.gammadashdashdash()
    #     alpha = self.curve.rotation.alpha(self.curve.quadpoints)
    #     alphadash = self.curve.rotation.alphadash(self.curve.quadpoints)
    #     width = self.width

    #     return self.J_jax(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width)

    # @derivative_dec
    # def dJ(self):
    #     """
    #     This returns the derivative of the quantity with respect to the curve dofs.
    #     """
    #     gamma = self.curve.curve.gamma()
    #     d1gamma = self.curve.curve.gammadash()
    #     d2gamma = self.curve.curve.gammadashdash()
    #     d3gamma = self.curve.curve.gammadashdashdash()
    #     alpha = self.curve.rotation.alpha(self.curve.quadpoints)
    #     alphadash = self.curve.rotation.alphadash(self.curve.quadpoints)
    #     width = self.width

    #     grad0 = self.thisgrad0(gamma, d1gamma, d2gamma,
    #                            d3gamma, alpha, alphadash, width)
    #     grad1 = self.thisgrad1(gamma, d1gamma, d2gamma,
    #                            d3gamma, alpha, alphadash, width)
    #     grad2 = self.thisgrad2(gamma, d1gamma, d2gamma,
    #                            d3gamma, alpha, alphadash, width)
    #     grad3 = self.thisgrad3(gamma, d1gamma, d2gamma,
    #                            d3gamma, alpha, alphadash, width)
    #     grad4 = self.thisgrad4(gamma, d1gamma, d2gamma,
    #                            d3gamma, alpha, alphadash, width)
    #     grad5 = self.thisgrad5(gamma, d1gamma, d2gamma,
    #                            d3gamma, alpha, alphadash, width)

    #     return self.curve.curve.dgamma_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1) \
    #         + self.curve.curve.dgammadashdash_by_dcoeff_vjp(grad2) + self.curve.curve.dgammadashdashdash_by_dcoeff_vjp(grad3) \
    #         + self.curve.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4) + \
    #         self.curve.rotation.dalphadash_by_dcoeff_vjp(
    #             self.curve.quadpoints, grad5)

    # return_fn_map = {'J': J, 'dJ': dJ}
