"""Implements the force on a coil in its own magnetic field and the field of other coils."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from .biotsavart import BiotSavart
from .selffield import B_regularized_pure, B_regularized, regularization_circ, regularization_rect
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


def coil_force(coil, allcoils, regularization):
    gammadash = coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    mutual_coils = [c for c in allcoils if c is not coil]
    mutual_field = BiotSavart(mutual_coils).set_points(coil.curve.gamma()).B()
    mutualforce = np.cross(coil.current.get_value() * tangent, mutual_field)
    selfforce = self_force(coil, regularization)
    return selfforce + mutualforce


def coil_force_pure(B, I, t):
    """force on coil for optimization"""
    return jnp.cross(I * t, B)


def self_force(coil, regularization):
    """
    Compute the self-force of a coil.
    """
    I = coil.current.get_value()
    tangent = coil.curve.gammadash() / np.linalg.norm(coil.curve.gammadash(),
                                                      axis=1)[:, None]
    B = B_regularized(coil, regularization)
    return coil_force_pure(B, I, tangent)


def self_force_circ(coil, a):
    """Compute the Lorentz self-force of a coil with circular cross-section"""
    return self_force(coil, regularization_circ(a))


def self_force_rect(coil, a, b):
    """Compute the Lorentz self-force of a coil with rectangular cross-section"""
    return self_force(coil, regularization_rect(a, b))


@jit
def lp_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold):
    r"""Pure function for minimizing the Lorentz force on a coil.

    The function is

     .. math::
        J = \frac{1}{p}\left(\int \text{max}(|\vec{F}| - F_0, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    and :math:`\ell` is arclength along the coil.
    """

    B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    force = jnp.cross(current * tangent, B_self + B_mutual)
    force_norm = jnp.linalg.norm(force, axis=1)[:, None]
    return (jnp.sum(jnp.maximum(force_norm - threshold, 0)**p * gammadash_norm))*(1./p)


class LpCurveForce(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \text{max}(|\vec{F}| - F_0, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    and :math:`\ell` is arclength along the coil.
    """

    def __init__(self, coil, allcoils, regularization, p=1.0, threshold=0.0):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints

        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            lp_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold)
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dgammadashdash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dB_mutual = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        self.biotsavart.set_points(self.coil.curve.gamma())

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.curve.gammadashdash(),
            self.coil.current.get_value(),
            self.biotsavart.B()
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):
        self.biotsavart.set_points(self.coil.curve.gamma())

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.curve.gammadashdash(),
            self.coil.current.get_value(),
            self.biotsavart.B()
        ]

        dJ_dB = self.dJ_dB_mutual(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)

        return (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))
            + self.biotsavart.B_vjp(dJ_dB)
        )

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def mean_squared_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual):
    r"""Pure function for minimizing the Lorentz force on a coil.

    The function is

    .. math:
        J = \frac{\int |\vec{F}|^2 d\ell}{\int d\ell}

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    force = jnp.cross(current * tangent, B_self + B_mutual)
    force_norm = jnp.linalg.norm(force, axis=1)[:, None]
    return jnp.sum(gammadash_norm * force_norm**2) / jnp.sum(gammadash_norm)


class MeanSquaredForce(Optimizable):
    r"""Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math:
        J = \frac{\int |\vec{F}|^2 d\ell}{\int d\ell}

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    def __init__(self, coil, allcoils, regularization):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints

        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            mean_squared_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual)
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dgammadashdash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dB_mutual = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        self.biotsavart.set_points(self.coil.curve.gamma())

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.curve.gammadashdash(),
            self.coil.current.get_value(),
            self.biotsavart.B()
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):
        self.biotsavart.set_points(self.coil.curve.gamma())

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.curve.gammadashdash(),
            self.coil.current.get_value(),
            self.biotsavart.B()
        ]

        dJ_dB = self.dJ_dB_mutual(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)

        return (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))
            + self.biotsavart.B_vjp(dJ_dB)
        )

    return_fn_map = {'J': J, 'dJ': dJ}
