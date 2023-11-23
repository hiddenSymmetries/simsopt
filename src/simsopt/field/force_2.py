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

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

def coil_force_pure(B, I, t):
    """force on coil for optimization"""
    return jnp.cross(I * t, B)

# def mutualforce_opt():

@partial(jit, static_argnums=6)
def selfforce_opt_pure(gamma, gammadash, gammadashdash, current, phi, regularization, num_basecoils, quadpoints):
    """Returns a 3D matrix of the form [F_0, F_1, ..., F_{N-1}], where F_i is a 3xN_quad matrix specifying the vector
    force at each quad_point."""
    gamma = jnp.split(gamma, num_basecoils)
    gammadash = jnp.split(gammadash, num_basecoils)
    gammadashdash = jnp.split(gammadashdash, num_basecoils)

    forces_self = jnp.empty([num_basecoils, quadpoints.size, 3])
    for i in range(num_basecoils):
        B_self = B_regularized_pure(gamma[i], gammadash[i], gammadashdash[i], quadpoints, current[i], regularization)
        tangent = gammadash[i] / jnp.linalg.norm(gammadash[i], axis=1)[:, None]
        forces_self.at[i, :, :].set(coil_force_pure(B_self, current[i], tangent))
    return forces_self

class MeanSquaredForceOpt(Optimizable):
    """Optimizable class to optimize forces on coils"""
    def __init__(self, basecoils, allcoils, regularization):
        self.quadpoints = basecoils[0].curve.quadpoints
        self.allcoils = np.array(allcoils)
        self.basecoils = np.array(basecoils)
        self.mutualcoils = (np.array([np.delete(self.allcoils, np.where(self.allcoils == self.basecoils[i])[0][0])
            for i in range(self.basecoils.size)])) #self.mutualcoils[i] is a vector of all coils excluding the ith base coil
        self.biotsavart = np.array([BiotSavart(self.mutualcoils[i]) for i in range(self.basecoils.size)])

        self.self_force_from_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, phi:
            selfforce_opt_pure(gamma, gammadash, gammadashdash, current, phi, regularization, self.basecoils.size,
                               self.quadpoints)
        )

        self.grad0_self = jit(
            lambda gamma, gammadash, gammadashdash, current, phi:
            grad(selfforce_opt_pure, argnums=0)(gamma, gammadash, gammadashdash, current, phi, regularization, self.basecoils.size,
                               self.quadpoints)
        )
        self.grad1_self = jit(
            lambda gamma, gammadash, gammadashdash, current, phi:
            grad(selfforce_opt_pure, argnums=1)(gamma, gammadash, gammadashdash, current, phi, regularization, self.basecoils.size,
                               self.quadpoints)
        )
        self.grad2_self = jit(
            lambda gamma, gammadash, gammadashdash, current, phi:
            grad(selfforce_opt_pure, argnums=2)(gamma, gammadash, gammadashdash, current, phi, regularization, self.basecoils.size,
                               self.quadpoints)
        )
        self.grad3_self = jit(
            lambda gamma, gammadash, gammadashdash, current, phi:
            grad(selfforce_opt_pure, argnums=3)(gamma, gammadash, gammadashdash, current, phi, regularization, self.basecoils.size,
                               self.quadpoints)
        )

        super().__init__(depends_on=basecoils)

    def J(self):
        #Mutual force calculation
        forces_mutual = np.empty([self.basecoils.size, self.quadpoints.size, 3])
        for i in range(self.basecoils.size):
            B_mutual = self.biotsavart[i].set_points(self.basecoils[i].curve.gamma()).B()
            I = self.basecoils[i].current.get_value()
            tangent = self.basecoils[i].curve.gammadash() / jnp.linalg.norm(self.basecoils[i].curve.gammadash(), axis=1)[:, None]
            forces_mutual[i, :, :] = coil_force_pure(B_mutual, I, tangent)

        #Self force calculation
        gamma_conc = jnp.concatenate([coil.curve.gamma() for coil in self.basecoils])
        gammadash_conc = jnp.concatenate([coil.curve.gammadash() for coil in self.basecoils])
        gammadashdash_conc = jnp.concatenate([coil.curve.gammadashdash() for coil in self.basecoils])
        current_conc = jnp.array([coil.current.get_value() for coil in self.basecoils])
        phi_conc = jnp.concatenate([coil.curve.quadpoints for coil in self.basecoils])
        forces_self = self.self_force_from_jax(gamma_conc, gammadash_conc, gammadashdash_conc, current_conc, phi_conc)

        #Total force & mean squared calculation
        forces_total = forces_mutual + forces_self
        forces_norm = np.linalg.norm(forces_total, axis=2) #0th index is coil, 1st is quadpoint
        arc_lengths = np.array([np.linalg.norm(coil.curve.gammadash(), axis=1) for coil in self.basecoils])
        mean_squared_forces = np.array([np.mean(forces_norm[i] ** 2 * arc_lengths[i]) / np.mean(arc_lengths[i])
            for i in range(self.basecoils.size)])
        total_mean_squared_forces = np.sum(mean_squared_forces)
        return total_mean_squared_forces

    @derivative_dec
    def dJ(self):
        gamma_conc = jnp.concatenate([coil.curve.gamma() for coil in self.basecoils])
        gammadash_conc = jnp.concatenate([coil.curve.gammadash() for coil in self.basecoils])
        gammadashdash_conc = jnp.concatenate([coil.curve.gammadashdash() for coil in self.basecoils])
        current_conc = jnp.array([coil.current.get_value() for coil in self.basecoils])
        phi_conc = jnp.concatenate([coil.curve.quadpoints for coil in self.basecoils])

        grad0 = self.grad0_self(gamma_conc, gammadash_conc, gammadashdash_conc, current_conc, phi_conc)
        print(grad0)

        # for i in range(base_coils):
        #     tangent = self.basecoils[i].curve.gammadash() / jnp.linalg.norm(self.basecoils[i].curve.gammadash(),axis=1)[:, None]
        # self.basecoils[i].curve.dgammadash_by_dcoeff_vjp(grad1[i * self.quadpoints.size:(i + 1) * self.quadpoints.size])

        # # Mutual force calculation
        # forces_mutual = np.empty([self.basecoils.size, self.quadpoints.size, 3])
        # for i in range(self.basecoils.size):
        #     B_mutual = self.biotsavart[i].set_points(self.basecoils[i].curve.gamma()).B()
        #     I = self.basecoils[i].current.get_value()
        #     tangent = self.basecoils[i].curve.gammadash() / jnp.linalg.norm(self.basecoils[i].curve.gammadash(),axis=1)[:, None]
        #     forces_mutual[i, :, :] = coil_force_pure(B_mutual, I, tangent)
        #     for j in range(self.basecoils[i].x.size):
        #         print(self.basecoils[i].x[j])
        #
        #
        # # Self force calculation
        # gamma_conc = jnp.concatenate([coil.curve.gamma() for coil in self.basecoils])
        # gammadash_conc = jnp.concatenate([coil.curve.gammadash() for coil in self.basecoils])
        # gammadashdash_conc = jnp.concatenate([coil.curve.gammadashdash() for coil in self.basecoils])
        # current_conc = jnp.array([coil.current.get_value() for coil in self.basecoils])
        # phi_conc = jnp.concatenate([coil.curve.quadpoints for coil in self.basecoils])
        # forces_self = self.self_force_from_jax(gamma_conc, gammadash_conc, gammadashdash_conc, current_conc, phi_conc)
        #
        # # Total force & mean squared calculation
        # forces_total = forces_mutual + forces_self
        #
        # #################
        #
        # #self-force derivative w.r.t. coeffs
        # grad0 = self.grad0_self(gamma_conc, gammadash_conc, gammadashdash_conc, current_conc, phi_conc)
        # grad1 = self.grad1_self(gamma_conc, gammadash_conc, gammadashdash_conc, current_conc, phi_conc)
        # grad2 = self.grad2_self(gamma_conc, gammadash_conc, gammadashdash_conc, current_conc, phi_conc)
        # selfforces_derivative = np.empty([self.basecoils.size, self.quadpoints.size, 3])
        # for i in range(self.basecoils.size):
        #     selfforces_derivative[i, :, :] = (
        #             self.basecoils[i].curve.dgamma_by_dcoeff_vjp(grad0[i * self.quadpoints.size:(i+1) * self.quadpoints.size])
        #             + self.basecoils[i].curve.dgammadash_by_dcoeff_vjp(grad1[i * self.quadpoints.size:(i+1) * self.quadpoints.size])
        #             + self.basecoils[i].curve.dgammadashdash_by_dcoeff_vjp(grad2[i * self.quadpoints.size:(i+1) * self.quadpoints.size])
        #     )

        # #mutual force derivatives w.r.t. coeffs
        # for j in range(self.basecoils.size):
        #     #mutual magnetic field of ith coil with others
        #     gradB = self.biotsavart[i].set_points(self.basecoils[i].curve.gamma()).B()



    return_fn_map = {'J': J, 'dJ': dJ}
