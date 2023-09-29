"""Implements the force on a coil in its own magnetic field and the field of other coils."""
import math
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from .biotsavart import BiotSavart
from .coil import Coil
from .selffield import B_regularized_pure, B_regularized, regularization_circ, regularization_rect
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


def coil_force_pure(B, I, t):
    """force on coil for optimization"""
    force = jnp.cross(I * t, B)
    return force


def self_force(coil, regularization):
    """
    Compute the self-force of a coil.
    """
    I = coil.current.get_value()
    tangent = coil.curve.gammadash() / np.linalg.norm(coil.curve.gammadash(), axis=1)[:, None]
    B = B_regularized(coil, regularization)
    return coil_force_pure(B, I, tangent)


def self_force_circ(coil, a):
    """Compute the Lorentz self-force of a coil with circular cross-section"""
    return self_force(coil, regularization_circ(a))


def self_force_rect(coil, a, b):
    """Compute the Lorentz self-force of a coil with rectangular cross-section"""
    return self_force(coil, regularization_rect(a, b))


@jit
def force_opt_pure(gamma, gammadash, gammadashdash,
                   current, phi, B_ext, regularization):
    """Cost function for force optimization. Optimize for peak self force on the coil (so far)"""
    t = gammadash / jnp.linalg.norm(gammadash)
    B_self = B_regularized_pure(
        gamma, gammadash, gammadashdash, phi, phi, current, regularization)
    B_tot = B_self + B_ext
    force = coil_force_pure(B_tot, current, t)
    f_norm = jnp.linalg.norm(force, axis=1)
    result = jnp.max(f_norm)
    # result = jnp.sum(f_norm)
    return result


class ForceOpt(Optimizable):
    """Optimizable class to optimize forces on a coil"""

    def __init__(self, coil, coils, a=0.05):
        self.coil = coil
        self.curve = coil.curve
        self.coils = coils
        self.a = a
        self.B_ext = BiotSavart(coils).set_points(self.curve.gamma()).B()
        self.B_self = 0
        self.B = 0
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash,
                         current, phi, B_ext: force_opt_pure(gamma, gammadash, gammadashdash,
                                                             current, phi, B_ext))

        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash, current, phi, B_ext: grad(
            self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, phi, B_ext))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash, current, phi, B_ext: grad(
            self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, phi, B_ext))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash, current, phi, B_ext: grad(
            self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, phi, B_ext))

        super().__init__(depends_on=[coil])

    def J(self):
        gamma = self.coil.curve.gamma()
        d1gamma = self.coil.curve.gammadash()
        d2gamma = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        phi = self.coil.curve.quadpoints
        phi = self.coil.curve.quadpoints
        B_ext = self.B_ext
        return self.J_jax(gamma, d1gamma, d2gamma, current, phi, B_ext)

    @derivative_dec
    def dJ(self):
        gamma = self.coil.curve.gamma()
        d1gamma = self.coil.curve.gammadash()
        d2gamma = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        phi = self.coil.curve.quadpoints
        phi = self.coil.curve.quadpoints
        B_ext = self.B_ext

        grad0 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, B_ext)
        grad1 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, B_ext)
        grad2 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, B_ext)

        return (
            self.coil.curve.dgamma_by_dcoeff_vjp(grad0) 
            + self.coil.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(grad2)
        )

    return_fn_map = {'J': J, 'dJ': dJ}
