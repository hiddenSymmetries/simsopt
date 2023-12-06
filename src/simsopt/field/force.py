"""Implements the force on a coil in its own magnetic field and the field of other coils."""
import math
from scipy import constants
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from .biotsavart import BiotSavart
from .selffield import B_regularized_pure, B_regularized, regularization_circ, regularization_rect
# from ..geo.jit import jit
from jax import jit #replace above
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from functools import partial
from scipy.linalg import block_diag

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

def coil_force_pure(B, I, t):
    print(B)
    print("h")
    """force on coil for optimization"""
    return jnp.cross(I * t, B)

def self_force(coil, regularization):
    """
    Compute the self-force of a coil.
    """
    I = coil.current.get_value()
    tangent = coil.curve.gammadash() / np.linalg.norm(coil.curve.gammadash(),axis=1)[:, None]
    B = B_regularized(coil, regularization)
    return coil_force_pure(B, I, tangent)

def self_force_circ(coil, a):
    """Compute the Lorentz self-force of a coil with circular cross-section"""
    return self_force(coil, regularization_circ(a))

def self_force_rect(coil, a, b):
    """Compute the Lorentz self-force of a coil with rectangular cross-section"""
    return self_force(coil, regularization_rect(a, b))

@jit
def selfforce_opt_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization):
    """..."""
    B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    tangent = gammadash / jnp.linalg.norm(gammadash, axis=1)[:, None]
    return coil_force_pure(B_self, current, tangent)

class MeanSquaredForceOpt(Optimizable):
    """Optimizable class to optimize forces on coils"""
    def __init__(self, basecoils, allcoils, regularization):
        self.basecoils = np.array(basecoils)
        self.allcoils = np.array(allcoils)

        @partial(jit, static_argnums=5)
        def grad_selfforce(gamma, gammadash, gammadashdash, current, quadpoints, deriv_pos):
            gradients = jnp.empty((quadpoints.size, 3, 3))
            for j in range(quadpoints.size):
                gammaj = gamma[j].reshape((1, 3))
                gammadashj = gammadash[j].reshape((1, 3))
                gammadashdashj = gammadashdash[j].reshape((1, 3))
                quadpointsj = jnp.array([quadpoints[j]])
                if deriv_pos == 0:
                    grad = jax.jacfwd(lambda g: selfforce_opt_pure(g, gammadashj, gammadashdashj, quadpointsj, current,
                                                                   regularization).reshape(3))(gammaj)
                elif deriv_pos == 1:
                    grad = jax.jacfwd(lambda g: selfforce_opt_pure(gammaj, g, gammadashdashj, quadpointsj, current,
                                                                   regularization).reshape(3))(gammadashj)
                elif deriv_pos == 2:
                    grad = jax.jacfwd(lambda g: selfforce_opt_pure(gammaj, gammadashj, g, quadpointsj, current,
                                                                   regularization).reshape(3))(gammadashdashj)
                else:
                    raise ValueError('deriv_pos must be an integer between 0 and 2')
                gradients = gradients.at[j, :, :].set(grad[:, 0, :])
            return gradients

        self.selfforce_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, quadpoints:
            selfforce_opt_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
        )

        self.dselfforce_dgamma = jit(
            lambda gamma, gammadash, gammadashdash, current, quadpoints:
            grad_selfforce(gamma, gammadash, gammadashdash, current, quadpoints, 0)
        )

        self.dselfforce_dgammadash = jit(
            lambda gamma, gammadash, gammadashdash, current, quadpoints:
            grad_selfforce(gamma, gammadash, gammadashdash, current, quadpoints, 1)
        )

        self.dselfforce_dgammadashdash = jit(
            lambda gamma, gammadash, gammadashdash, current, quadpoints:
            grad_selfforce(gamma, gammadash, gammadashdash, current, quadpoints, 2)
        )

        super().__init__(depends_on=basecoils)

    def J(self):
        J = 0
        for i in range(self.basecoils.size):
            coil = self.basecoils[i]
            current = coil.current.get_value()
            gamma = coil.curve.gamma()
            gammadash = coil.curve.gammadash()
            gammadashdash = coil.curve.gammadashdash()
            gammadash_norm = jnp.linalg.norm(gammadash,axis=1)
            tangent = gammadash / gammadash_norm[:, None]
            quadpoints = coil.curve.quadpoints

            B_mutual = BiotSavart(np.delete(self.allcoils, np.where(self.allcoils == coil)[0][0])).set_points(gamma).B()
            forces_mutual = coil_force_pure(B_mutual, current, tangent)
            forces_self = self.selfforce_jax(gamma, gammadash, gammadashdash, current, quadpoints)
            forces_total = forces_mutual + forces_self
            forces_norm = np.einsum('ij,ij->i', forces_total, forces_total)

            J += np.sum(forces_norm * gammadash_norm) / np.sum(gammadash_norm)
        return J

    @derivative_dec
    def dJ(self):
        dJ = 0
        for i in range(self.basecoils.size):
            coil = self.basecoils[i]
            current = coil.current.get_value()
            gamma = coil.curve.gamma()
            gammadash = coil.curve.gammadash()
            gammadashdash = coil.curve.gammadashdash()
            gammadash_norm = jnp.linalg.norm(gammadash,axis=1)
            tangent = gammadash / gammadash_norm[:, None]
            quadpoints = coil.curve.quadpoints

            #Finding the forces, the mutual magnetic field, and derivatives of the self-force
            biotsavart = BiotSavart(np.delete(self.allcoils, np.where(self.allcoils == coil)[0][0]))
            B_mutual = biotsavart.set_points(gamma).B()
            forces_mutual = coil_force_pure(B_mutual, current, tangent)
            forces_self = self.selfforce_jax(gamma, gammadash, gammadashdash, current, quadpoints)
            forces_total = forces_mutual + forces_self
            
            dselfforce_dgamma = self.dselfforce_dgamma(gamma, gammadash, gammadashdash, current, quadpoints)
            dselfforce_dgammadash = self.dselfforce_dgammadash(gamma, gammadash, gammadashdash, current, quadpoints)
            dselfforce_dgammadashdash = self.dselfforce_dgammadashdash(gamma, gammadash, gammadashdash, current, quadpoints)

            #Calculating dJ with VJPs...
            prefactor = (np.sum(gammadash_norm)) ** (-1)
            vec = prefactor * 2 * np.einsum('i,ij,ijk->ik',gammadash_norm,forces_total,dselfforce_dgamma)
            dJ += coil.curve.dgamma_by_dcoeff_vjp(vec)

            vec = 0
            vec += 2 * np.einsum('i,ij,ijk->ik',gammadash_norm,forces_total,dselfforce_dgammadash)
            vec += np.einsum('ij,ij,ik->ik', forces_total, forces_total, tangent)
            vec += 2 * current * np.cross(B_mutual, forces_total)
            vec += -2 * current * np.einsum('ij,ij,ik->ik', tangent, np.cross(B_mutual, forces_total), tangent)
            vec *= prefactor
            dJ += coil.curve.dgammadash_by_dcoeff_vjp(vec)

            vec = prefactor * 2 * np.einsum('i,ij,ijk->ik', gammadash_norm, forces_total, dselfforce_dgammadashdash)
            dJ += coil.curve.dgammadashdash_by_dcoeff_vjp(vec)

            vec = prefactor * 2 * current * np.einsum('i,ij->ij', gammadash_norm, np.cross(forces_total, tangent))
            dJ += biotsavart.B_vjp(vec)

            vec = np.array([np.sum(prefactor * 2 * np.einsum('i,ij,ij->i', gammadash_norm, forces_total, forces_total) / current)])
            dJ += coil.current.vjp(vec)

            prefactor = (np.sum(np.einsum('ij,ij,i->i', forces_total, forces_total, gammadash_norm))
                         / (np.sum(gammadash_norm) ** 2))
            dJ += coil.curve.dgammadash_by_dcoeff_vjp(prefactor * tangent)
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}
