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

        self.tt = jit(lambda gammadash: gammadash / jnp.linalg.norm(gammadash,axis=1)[:, None])

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
        # Mutual force calculation
        forces_mutual = np.empty([self.basecoils.size, self.quadpoints.size, 3])
        for i in range(self.basecoils.size):
            B_mutual = self.biotsavart[i].set_points(self.basecoils[i].curve.gamma()).B()
            I = self.basecoils[i].current.get_value()
            tangent = self.basecoils[i].curve.gammadash() / jnp.linalg.norm(self.basecoils[i].curve.gammadash(),axis=1)[:, None]
            forces_mutual[i, :, :] = coil_force_pure(B_mutual, I, tangent)

        # Self force calculation
        gamma_conc = jnp.concatenate([coil.curve.gamma() for coil in self.basecoils])
        gammadash_conc = jnp.concatenate([coil.curve.gammadash() for coil in self.basecoils])
        gammadashdash_conc = jnp.concatenate([coil.curve.gammadashdash() for coil in self.basecoils])
        current_conc = jnp.array([coil.current.get_value() for coil in self.basecoils])
        phi_conc = jnp.concatenate([coil.curve.quadpoints for coil in self.basecoils])
        forces_self = self.self_force_from_jax(gamma_conc, gammadash_conc, gammadashdash_conc, current_conc, phi_conc)

        # Total force & mean squared calculation
        forces_total = forces_mutual + forces_self

        # 1) d_lambda_d calculation
        """
        grad[0] is the derivative of a single number wrt to gamma at differnet points,
        so it's like a 3xquadpoints matrix, or an array of 3D vectors of length quadpoints.
        similarly, dgamma_by_coef is a 2D matrix of 3D vectors.
        moving on here, the right component is the same, a 2D matrix of 3D vectors. t_i is bigger, a
        matrix wrt to numbasecoils (4), quadpoints (15), and space (3). so, to take this derivative, we
        can just sum over the individual basecoils like so:

        As a note, t[quad_p,xyz]
        """

        dlambdads = []
        dphi = 2*np.pi/self.quadpoints.size
        for i in range(self.basecoils.size):
            tangent = self.basecoils[i].curve.gammadash() / jnp.linalg.norm(self.basecoils[i].curve.gammadash(),axis=1)[:, None]
            dlambdads.append(dphi * self.basecoils[i].curve.dgammadash_by_dcoeff_vjp(tangent))

        # 1) d_lambda_n calculation
        dlambdans = []
        for i in range(self.basecoils.size):
            coil = self.basecoils[i]
            current = coil.current.get_value()

            alpha = 2 * np.linalg.norm(coil.curve.gammadash(), axis=1)[:, None] * forces_total[i]
            #alpha has dimensions (15, 3)

            tangent = coil.curve.gammadash() / jnp.linalg.norm(coil.curve.gammadash(),axis=1)[:, None]
            norm_forces = np.linalg.norm(forces_total[i],axis=1)
            sum1 = self.basecoils[i].curve.dgammadash_by_dcoeff_vjp(tangent * norm_forces[:, None])

            B_mutual = self.biotsavart[i].set_points(self.basecoils[i].curve.gamma()).B()
            bcrossa = current * np.cross(B_mutual, alpha)
            sum3 = self.basecoils[i].curve.dgammadash_by_dcoeff_vjp(bcrossa)

            sum4 = self.basecoils[i].curve.dgammadash_by_dcoeff_vjp(tangent * -(tangent*bcrossa).sum(1)[:, None])

            cross = current * np.cross(alpha, tangent) * np.linalg.norm(coil.curve.gammadash(),axis=1)[:, None]
            sum5 = self.biotsavart[i].B_vjp(cross)

            one = np.asarray([1.])
            sum6 = coil.current.vjp(one * np.linalg.norm(coil.curve.gammadash(),axis=1)[:, None] * forces_total[i] / current)

            sum = sum1 + sum3 + sum4 + sum5 + sum6
            dlambdans.append(sum * dphi)
            #do sum5!!!!
        print(dlambdans)
        #1a
        #TROUBLESHOOT: the forcetotal are the same for all 4 coils
        #1b

        exit()
        #1d

        ###### so that it runs for now
        return dlambdads[0]


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
