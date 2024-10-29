"""Implements the force on a coil in its own magnetic field and the field of other coils."""
from scipy import constants
import time
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt._core.derivative import Derivative
from .biotsavart import BiotSavart
from .selffield import B_regularized_pure, B_regularized, regularization_circ, regularization_rect
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


# def coil_force(coil, allcoils, regularization):
#     """
#     """
#     eps = 1e-20  # small number to avoid blow up in the denominator when i = j
#     gammas = np.array([c.curve.gamma() for c in allcoils])
#     gammadashs = np.array([c.curve.gammadash() for c in allcoils])
#     # gamma and gammadash are shape (ncoils, nquadpoints, 3)
#     r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :]  # Note, do not use the i = j indices
#     gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1) 
#     rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3

#     # Single sum over the jth of the closed curves
#     F = (currents[None, :] * currents[:, None])[:, :, :, None] * jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3)
#     mutualforce = -F * 1e-7 / jnp.shape(gammas)[1]
#     selfforce = self_force(coil, regularization)
#     return pointwise_forces + selfforce

def coil_force(coil, allcoils, regularization, nturns=1):
    gammadash = coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    mutual_coils = [c for c in allcoils if c is not coil]

    ### Line below seems to be the issue -- all these BiotSavart objects seem to stick
    ### around and not to go out of scope after these calls!
    mutual_field = BiotSavart(mutual_coils).set_points(coil.curve.gamma()).B()
    mutualforce = np.cross(coil.current.get_value() * tangent, mutual_field)
    selfforce = self_force(coil, regularization)
    return (selfforce + mutualforce) / nturns

def coil_net_forces(coils, allcoils, regularization, nturns=None):
    net_forces = np.zeros((len(coils), 3))
    if nturns is None:
        nturns = np.ones(len(coils))
    for i, coil in enumerate(coils):
        Fi = coil_force(coil, allcoils, regularization[i], nturns[i])
        gammadash = coil.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        net_forces[i, :] += np.sum(gammadash_norm * Fi, axis=0) / gammadash.shape[0]
    return net_forces

def coil_torque(coil, allcoils, regularization, nturns=1):
    gamma = coil.curve.gamma()
    center = coil.curve.center(coil.curve.gamma(), coil.curve.gammadash())
    return np.cross(gamma - center, coil_force(coil, allcoils, regularization, nturns))

def coil_net_torques(coils, allcoils, regularization, nturns=None):
    net_torques = np.zeros((len(coils), 3))
    if nturns is None:
        nturns = np.ones(len(coils))
    for i, coil in enumerate(coils):
        Ti = coil_torque(coil, allcoils, regularization[i], nturns[i])
        gammadash = coil.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        net_torques[i, :] += np.sum(gammadash_norm * Ti, axis=0) / gammadash.shape[0]
    return net_torques

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
    return (jnp.sum(jnp.maximum(force_norm - threshold, 0)**p * gammadash_norm)) * (1. / p)


class LpCurveForce(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \text{max}(|d\vec{F}/d\ell| - dF_0/d\ell, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    and :math:`\ell` is arclength along the coil.
    """

    def __init__(self, coil, allcoils, regularization, p=2.0, threshold=0.0):
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
        J = \frac{\int |d\vec{F}/d\ell|^2 d\ell}{\int d\ell}

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

@jit
def direct_squared_mean_force_pure(gammas, gammadashs, currents):
    r"""
    """
    eps = 1e-10  # small number to avoid blow up in the denominator when i = j
    r_ij = gammas[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
    gammadash_prod = jnp.sum(gammadashs[:, None, :, None, :] * gammadashs[None, :, None, :, :], axis=-1) 
    # gammadash_norm = jnp.linalg.norm(gammadashs,axis=-1)[:, None]
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    F = jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
    Ii_Ij = currents[:, None] * currents[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    net_forces = -jnp.sum(Ii_Ij[:, :, None] * F, axis=1) / jnp.shape(gammas)[1]  #** 2
    return jnp.sum(jnp.linalg.norm(net_forces, axis=-1) ** 2) * 1e-14

class DirectSquaredMeanForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    def __init__(self, allcoils):
        self.allcoils = allcoils

        self.J_jax = jit(
            lambda gammas, gammadashs, currents:
            direct_squared_mean_force_pure(gammas, gammadashs, currents)
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammadashs, currents:
            grad(self.J_jax, argnums=0)(gammas, gammadashs, currents)
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammadashs, currents:
            grad(self.J_jax, argnums=1)(gammas, gammadashs, currents)
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammadashs, currents:
            grad(self.J_jax, argnums=2)(gammas, gammadashs, currents)
        )

        super().__init__(depends_on=allcoils)


    def J(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
        ]     

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
        ]     
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}

@jit
def mixed_squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents, currents2):
    r"""
    """
    eps = 1e-10  # small number to avoid blow up in the denominator when i = j
    r_ij = gammas[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
    gammadash_prod = jnp.sum(gammadashs[:, None, :, None, :] * gammadashs[None, :, None, :, :], axis=-1) 
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    F = jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
    Ii_Ij = currents[:, None] * currents[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    net_forces = -jnp.sum(Ii_Ij[:, :, None] * F, axis=1) / jnp.shape(gammas)[1]   #** 2

    # repeat with gamma, gamma2
    r_ij = gammas[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    gammadash_prod = jnp.sum(gammadashs[:, None, :, None, :] * gammadashs2[None, :, None, :, :], axis=-1) 
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    F = jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
    Ii_Ij = currents[:, None] * currents2[None, :]
    # Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    net_forces += -jnp.sum(Ii_Ij[:, :, None] * F, axis=1) / jnp.shape(gammas2)[1] # jnp.shape(gammas)[1] / 
    summ = jnp.sum(jnp.linalg.norm(net_forces, axis=-1) ** 2)

    # repeat with gamma2, gamma
    r_ij = gammas2[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
    gammadash_prod = jnp.sum(gammadashs2[:, None, :, None, :] * gammadashs[None, :, None, :, :], axis=-1) 
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    F = jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
    Ii_Ij = currents2[:, None] * currents[None, :]
    # Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    net_forces = -jnp.sum(Ii_Ij[:, :, None] * F, axis=1) / jnp.shape(gammas)[1]  # / jnp.shape(gammas2)[1]

    # repeat with gamma2, gamma2
    r_ij = gammas2[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    gammadash_prod = jnp.sum(gammadashs2[:, None, :, None, :] * gammadashs2[None, :, None, :, :], axis=-1) 
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    F = jnp.sum(jnp.sum((gammadash_prod / rij_norm3)[:, :, :, :, None] * r_ij, axis=3), axis=2)
    Ii_Ij = currents2[:, None] * currents2[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    net_forces += -jnp.sum(Ii_Ij[:, :, None] * F, axis=1) / jnp.shape(gammas2)[1]   # ** 2
    summ += jnp.sum(jnp.linalg.norm(net_forces, axis=-1) ** 2)
    return summ * 1e-14  # factor of (mu0 / 4pi)^2

class MixedSquaredMeanForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    def __init__(self, allcoils, allcoils2):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            mixed_squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        super().__init__(depends_on=(allcoils + allcoils2))


    def J(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def mixed_lp_force_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, 
                        quadpoints, quadpoints2,
                        currents, currents2, regularizations, regularizations2, p, threshold):
    r"""
    """
    B_self = [B_regularized_pure(gammas[i], gammadashs[i], gammadashdashs[i], quadpoints, currents[i], regularizations[i]) for i in range(jnp.shape(gammas)[0])]
    B_self2 = [B_regularized_pure(gammas2[i], gammadashs2[i], gammadashdashs2[i], quadpoints2, currents2[i], regularizations2[i]) for i in range(jnp.shape(gammas2)[0])]
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
    tangents = gammadashs / gammadash_norms
    gammadash_norms2 = jnp.linalg.norm(gammadashs2, axis=-1)[:, :, None]
    tangents2 = gammadashs2 / gammadash_norms2
    selfforce = jnp.array([jnp.cross(currents[i] * tangents[i], B_self[i]) for i in range(jnp.shape(gammas)[0])])
    selfforce2 = jnp.array([jnp.cross(currents2[i] * tangents2[i], B_self2[i]) for i in range(jnp.shape(gammas2)[0])])
     
    eps = 1e-10  # small number to avoid blow up in the denominator when i = j
    r_ij = gammas[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices

    ### Note that need to do dl1 x dl2 x r12 here instead of just (dl1 * dl2)r12
    # because these are not equivalent expressions if we are squaring the pointwise forces
    # before integration over coil i! 
    cross_prod = jnp.cross(tangents[:, None, :, None, :], jnp.cross(gammadashs[None, :, None, :, :], r_ij))
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents[:, None] * currents[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    F = jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas)[1]

    # repeat with gamma, gamma2
    r_ij = gammas[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    cross_prod = jnp.cross(tangents[:, None, :, None, :], jnp.cross(gammadashs2[None, :, None, :, :], r_ij))
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents[:, None] * currents2[None, :]
    F += jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas2)[1]
    force_norm = jnp.linalg.norm(F * 1e-7 + selfforce, axis=-1)
    summ = jnp.sum(jnp.maximum(force_norm[:, :, None] - threshold, 0) ** p * gammadash_norms)

    # repeat with gamma2, gamma
    r_ij = gammas2[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
    cross_prod = jnp.cross(tangents2[:, None, :, None, :], jnp.cross(gammadashs[None, :, None, :, :], r_ij))
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents2[:, None] * currents[None, :]
    F = jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas)[1]

    # repeat with gamma2, gamma2
    r_ij = gammas2[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    cross_prod = jnp.cross(tangents2[:, None, :, None, :], jnp.cross(gammadashs2[None, :, None, :, :], r_ij))
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents2[:, None] * currents2[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    F += jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas2)[1]
    force_norm2 = jnp.linalg.norm(F * 1e-7 + selfforce2, axis=-1)
    summ += jnp.sum(jnp.maximum(force_norm2[:, :, None] - threshold, 0) ** p * gammadash_norms2)
    return summ * (1 / p)

class MixedLpCurveForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    def __init__(self, allcoils, allcoils2, regularizations, regularizations2, p=2.0, threshold=0.0):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2
        quadpoints = self.allcoils[0].curve.quadpoints
        quadpoints2 = self.allcoils2[0].curve.quadpoints

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            mixed_lp_force_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, quadpoints, quadpoints2, 
                                currents, currents2, regularizations, regularizations2, p, threshold)
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadashdash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadashdash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=6)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=7)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        super().__init__(depends_on=(allcoils + allcoils2))


    def J(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dgammadashdash = self.dJ_dgammadashdash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dgammadashdash2 = self.dJ_dgammadashdash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class MixedLpCurveTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    # @jit
    def mixed_lp_torque_pure(self, gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, 
                            quadpoints, quadpoints2,
                            currents, currents2, regularizations, regularizations2, p, threshold):
        r"""
        """
        B_self = [B_regularized_pure(gammas[i], gammadashs[i], gammadashdashs[i], quadpoints, currents[i], regularizations[i]) for i in range(jnp.shape(gammas)[0])]
        B_self2 = [B_regularized_pure(gammas2[i], gammadashs2[i], gammadashdashs2[i], quadpoints2, currents2[i], regularizations2[i]) for i in range(jnp.shape(gammas2)[0])]
        gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
        tangents = gammadashs / gammadash_norms
        gammadash_norms2 = jnp.linalg.norm(gammadashs2, axis=-1)[:, :, None]
        tangents2 = gammadashs2 / gammadash_norms2
        centers = jnp.array([c.curve.center(gammas[i], gammadashs[i]) for i, c in enumerate(self.allcoils)])[:, None, :]
        centers2 = jnp.array([c.curve.center(gammas2[i], gammadashs2[i]) for i, c in enumerate(self.allcoils2)])[:, None, :]
        selftorque = jnp.array([jnp.cross((gammas - centers)[i], jnp.cross(currents[i] * tangents[i], B_self[i])) for i in range(jnp.shape(gammas)[0])])
        selftorque2 = jnp.array([jnp.cross((gammas2 - centers2)[i], jnp.cross(currents2[i] * tangents2[i], B_self2[i])) for i in range(jnp.shape(gammas2)[0])])
        
        eps = 1e-10  # small number to avoid blow up in the denominator when i = j
        r_ij = gammas[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices

        ### Note that need to do dl1 x dl2 x r12 here instead of just (dl1 * dl2)r12
        # because these are not equivalent expressions if we are squaring the pointwise forces
        # before integration over coil i! 
        cross_prod = jnp.cross((gammas - centers)[:, None, :, None, :], jnp.cross(tangents[:, None, :, None, :], jnp.cross(gammadashs[None, :, None, :, :], r_ij)))
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents[:, None] * currents[None, :]
        Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
        T = jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas)[1]

        # repeat with gamma, gamma2
        r_ij = gammas[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
        cross_prod = jnp.cross((gammas - centers)[:, None, :, None, :], jnp.cross(tangents[:, None, :, None, :], jnp.cross(gammadashs2[None, :, None, :, :], r_ij)))
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents[:, None] * currents2[None, :]
        T += jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas2)[1]
        torque_norm = jnp.linalg.norm(T * 1e-7 + selftorque, axis=-1)
        summ = jnp.sum(jnp.maximum(torque_norm[:, :, None] - threshold, 0) ** p * gammadash_norms)

        # repeat with gamma2, gamma
        r_ij = gammas2[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
        cross_prod = jnp.cross((gammas2 - centers2)[:, None, :, None, :], jnp.cross(tangents2[:, None, :, None, :], jnp.cross(gammadashs[None, :, None, :, :], r_ij)))
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents2[:, None] * currents[None, :]
        T = jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas)[1]

        # repeat with gamma2, gamma2
        r_ij = gammas2[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
        cross_prod = jnp.cross((gammas2 - centers2)[:, None, :, None, :], jnp.cross(tangents2[:, None, :, None, :], jnp.cross(gammadashs2[None, :, None, :, :], r_ij)))
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents2[:, None] * currents2[None, :]
        Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
        T += jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas2)[1]
        torque_norm2 = jnp.linalg.norm(T * 1e-7 + selftorque2, axis=-1)
        summ += jnp.sum(jnp.maximum(torque_norm2[:, :, None] - threshold, 0) ** p * gammadash_norms2)
        return summ * (1 / p)

    def __init__(self, allcoils, allcoils2, regularizations, regularizations2, p=2.0, threshold=0.0):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2
        quadpoints = self.allcoils[0].curve.quadpoints
        quadpoints2 = self.allcoils2[0].curve.quadpoints
        centers = self.allcoils[0].curve.center
        centers2 = self.allcoils2[0].curve.center

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            self.mixed_lp_torque_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, quadpoints, quadpoints2, 
                                currents, currents2, regularizations, regularizations2, p, threshold)
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadashdash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dgammadashdash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=6)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2:
            grad(self.J_jax, argnums=7)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2)
        )

        super().__init__(depends_on=(allcoils + allcoils2))


    def J(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils]),
            jnp.array([c.curve.gammadashdash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dgammadashdash = self.dJ_dgammadashdash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dgammadashdash2 = self.dJ_dgammadashdash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class MixedSquaredMeanTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    # @jit
    def mixed_squared_mean_torque(self, gammas, gammas2, gammadashs, gammadashs2, currents, currents2):
        r"""
        """
        eps = 1e-10  # small number to avoid blow up in the denominator when i = j
        r_ij = gammas[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
        centers = jnp.array([c.curve.center(gammas[i], gammadashs[i]) for i, c in enumerate(self.allcoils)])[:, None, :]
        centers2 = jnp.array([c.curve.center(gammas2[i], gammadashs2[i]) for i, c in enumerate(self.allcoils2)])[:, None, :]
        cross1 = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs[:, None, :, None, :], cross1)
        cross3 = jnp.cross((gammas - centers)[:, None, :, None, :], cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents[:, None] * currents[None, :]
        Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
        T = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques = -jnp.sum(T, axis=1) / jnp.shape(gammas)[1]   # ** 2

        # repeat with gamma, gamma2
        r_ij = gammas[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
        cross1 = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs[:, None, :, None, :], cross1)
        cross3 = jnp.cross((gammas - centers)[:, None, :, None, :], cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents[:, None] * currents2[None, :]
        T = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques += -jnp.sum(T, axis=1) / jnp.shape(gammas2)[1]
        summ = jnp.sum(jnp.linalg.norm(net_torques, axis=-1) ** 2)

        # repeat with gamma2, gamma
        r_ij = gammas2[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
        cross1 = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs2[:, None, :, None, :], cross1)
        cross3 = jnp.cross((gammas2 - centers2)[:, None, :, None, :], cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents2[:, None] * currents[None, :]
        T = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques = -jnp.sum(T, axis=1)/ jnp.shape(gammas)[1]  # / jnp.shape(gammas2)[1]

        # repeat with gamma2, gamma2
        r_ij = gammas2[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
        cross1 = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs2[:, None, :, None, :], cross1)
        cross3 = jnp.cross((gammas2 - centers2)[:, None, :, None, :], cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents2[:, None] * currents2[None, :]
        Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
        T = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques += -jnp.sum(T, axis=1)/ jnp.shape(gammas2)[1]  #** 2   
        summ += jnp.sum(jnp.linalg.norm(net_torques, axis=-1) ** 2)
        return summ * 1e-14


    def __init__(self, allcoils, allcoils2):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2
        centers = self.allcoils[0].curve.center
        centers2 = self.allcoils2[0].curve.center

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            self.mixed_squared_mean_torque(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2)
        )

        super().__init__(depends_on=(allcoils + allcoils2))


    def J(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
        ]     
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}

@jit
def squared_mean_force_pure(current, gammadash, B_mutual):
    r"""
    """
    # B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    # gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    # tangent = gammadash / gammadash_norm
    return (current * jnp.linalg.norm(jnp.sum(jnp.cross(gammadash, B_mutual), axis=0))) ** 2   # / jnp.sum(gammadash_norm)  # factor for the integral

class SquaredMeanForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    def __init__(self, coil, allcoils):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in self.allcoils if c is not self.coil]

        self.J_jax = jit(
            lambda current, gammadash, B_mutual:
            squared_mean_force_pure(current, gammadash, B_mutual)
        )

        self.dJ_dcurrent = jit(
            lambda current, gammadash, B_mutual:
            grad(self.J_jax, argnums=0)(current, gammadash, B_mutual)
        )

        self.dJ_dgammadash = jit(
            lambda current, gammadash, B_mutual:
            grad(self.J_jax, argnums=1)(current, gammadash, B_mutual)
        )

        self.dJ_dB = jit(
            lambda current, gammadash, B_mutual:
            grad(self.J_jax, argnums=2)(current, gammadash, B_mutual)
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        biotsavart = BiotSavart(self.othercoils)
        biotsavart.set_points(self.coil.curve.gamma())

        args = [
            self.coil.current.get_value(),
            self.coil.curve.gammadash(),
            biotsavart.B(),
        ]     
        for c in self.othercoils:
            c._children = set()

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):
        biotsavart = BiotSavart(self.othercoils)
        biotsavart.set_points(self.coil.curve.gamma())

        args = [
            self.coil.current.get_value(),
            self.coil.curve.gammadash(),
            biotsavart.B(),
        ] 

        dJ_dB = self.dJ_dB(*args)
        dB_dX = biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)

        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))
            + biotsavart.B_vjp(dJ_dB)
        )
        for c in self.othercoils:
            c._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}

class SquaredMeanTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    # @jit
    def squared_mean_torque_pure(self, current, gamma, gammadash, B_mutual):
        r"""
        """
        return (current * jnp.linalg.norm(jnp.sum(jnp.cross(gamma - self.coil.curve.center(gamma, gammadash), jnp.cross(gammadash, B_mutual)), axis=0))) ** 2  # / jnp.sum(gammadash_norm)  # factor for the integral


    def __init__(self, coil, allcoils):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in allcoils if c is not coil]

        self.J_jax = jit(
            lambda current, gamma, gammadash, B_mutual:
            self.squared_mean_torque_pure(current, gamma, gammadash, B_mutual)
        )

        self.dJ_dcurrent = jit(
            lambda current, gamma, gammadash, B_mutual:
            grad(self.J_jax, argnums=0)(current, gamma, gammadash, B_mutual)
        )

        self.dJ_dgamma = jit(
            lambda current, gamma, gammadash, B_mutual:
            grad(self.J_jax, argnums=1)(current, gamma, gammadash, B_mutual)
        )

        self.dJ_dgammadash = jit(
            lambda current, gamma, gammadash, B_mutual:
            grad(self.J_jax, argnums=2)(current, gamma, gammadash, B_mutual)
        )

        self.dJ_dB = jit(
            lambda current, gamma, gammadash, B_mutual:
            grad(self.J_jax, argnums=3)(current, gamma, gammadash, B_mutual)
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        biotsavart = BiotSavart(self.othercoils)
        biotsavart.set_points(self.coil.curve.gamma())
        args = [
            self.coil.current.get_value(),
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            biotsavart.B(),
        ]     
        for c in self.othercoils:
            c._children = set() 
        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):
        biotsavart = BiotSavart(self.othercoils)
        biotsavart.set_points(self.coil.curve.gamma())
        args = [
            self.coil.current.get_value(),
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            biotsavart.B(),
        ]       
        dJ_dB = self.dJ_dB(*args)
        dB_dX = biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)
        for c in self.othercoils:
            c._children = set()

        return (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))
            + biotsavart.B_vjp(dJ_dB)
        )

    return_fn_map = {'J': J, 'dJ': dJ}

class MeanSquaredTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

    where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
    along the coil.
    """

    # @jit
    def mean_squared_torque_pure(self, gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual):
        r"""
        """
        B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
        gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
        tangent = gammadash / gammadash_norm
        force = jnp.cross(current * tangent, B_self + B_mutual)
        torque = jnp.cross(gamma - self.coil.curve.center(gamma, gammadash), force)
        torque_norm = jnp.linalg.norm(torque, axis=1)[:, None]
        return jnp.sum(gammadash_norm * torque_norm ** 2) / jnp.sum(gammadash_norm)

    def __init__(self, coil, allcoils, regularization):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints
        center = self.coil.curve.center

        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            self.mean_squared_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual)
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
            self.biotsavart.B(),
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
            self.biotsavart.B(),
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


class LpCurveTorque(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \text{max}(|\vec{F}| - F_0, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    and :math:`\ell` is arclength along the coil.
    """

    # @jit
    def lp_torque_pure(self, gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold):
        r"""Pure function for minimizing the Lorentz force on a coil.

        The function is

        .. math::
            J = \frac{1}{p}\left(\int \text{max}(|\vec{T}| - T_0, 0)^p d\ell\right)

        where :math:`\vec{T}` is the Lorentz torque, :math:`T_0` is a threshold torque,  
        and :math:`\ell` is arclength along the coil.
        """
        B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
        gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
        tangent = gammadash / gammadash_norm
        force = jnp.cross(current * tangent, B_self + B_mutual)
        torque = jnp.cross(gamma - self.coil.curve.center(gamma, gammadash), force)
        torque_norm = jnp.linalg.norm(torque, axis=1)[:, None]
        return (jnp.sum(jnp.maximum(torque_norm - threshold, 0)**p * gammadash_norm)) * (1 / p)  #/ jnp.sum(gammadash_norm)

    def __init__(self, coil, allcoils, regularization, p=2.0, threshold=0.0):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints
        center = self.coil.curve.center

        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            self.lp_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold)
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
            self.biotsavart.B(),
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
            self.biotsavart.B(),
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


# @jit
# def tve_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold):
#     r"""Pure function for minimizing the Lorentz force on a coil.

#     The function is

#      .. math::
#         J = \frac{1}{p}\left(\int \text{max}(|\vec{T}| - T_0, 0)^p d\ell\right)

#     where :math:`\vec{T}` is the Lorentz torque, :math:`T_0` is a threshold torque,  
#     and :math:`\ell` is arclength along the coil.
#     """
#     B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
#     gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
#     tangent = gammadash / gammadash_norm
#     force = jnp.cross(current * tangent, B_self + B_mutual)
#     torque = jnp.cross(gamma, force)
#     torque_norm = jnp.linalg.norm(torque, axis=1)[:, None]
#     return (jnp.sum(jnp.maximum(torque_norm - threshold, 0)**p * gammadash_norm))*(1./p)

# class TVE(Optimizable):
#     r"""  Optimizable class to minimize the Lorentz force on a coil.

#     The objective function is

#     .. math::
#         J = 0.5 * I_i * I_j * Lij

#     where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
#     and :math:`\ell` is arclength along the coil.
#     """

#     def __init__(self, coil, allcoils, regularization, p=1.0, threshold=0.0):
#         self.coil = coil
#         self.allcoils = allcoils
#         self.othercoils = [c for c in allcoils if c is not coil]
#         self.biotsavart = BiotSavart(self.othercoils)
#         quadpoints = self.coil.curve.quadpoints

#         self.J_jax = jit(
#             lambda gamma, gammadash, gammadashdash, current, B_mutual:
#             tve_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold)
#         )

#         self.dJ_dgamma = jit(
#             lambda gamma, gammadash, gammadashdash, current, B_mutual:
#             grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, B_mutual)
#         )

#         self.dJ_dgammadash = jit(
#             lambda gamma, gammadash, gammadashdash, current, B_mutual:
#             grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual)
#         )

#         self.dJ_dgammadashdash = jit(
#             lambda gamma, gammadash, gammadashdash, current, B_mutual:
#             grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, B_mutual)
#         )

#         self.dJ_dcurrent = jit(
#             lambda gamma, gammadash, gammadashdash, current, B_mutual:
#             grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual)
#         )

#         self.dJ_dB_mutual = jit(
#             lambda gamma, gammadash, gammadashdash, current, B_mutual:
#             grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, current, B_mutual)
#         )

#         super().__init__(depends_on=allcoils)

#     def J(self):
#         self.biotsavart.set_points(self.coil.curve.gamma())

#         args = [
#             self.coil.curve.gamma(),
#             self.coil.curve.gammadash(),
#             self.coil.curve.gammadashdash(),
#             self.coil.current.get_value(),
#             self.biotsavart.B()
#         ]     

#         return self.J_jax(*args)

#     @derivative_dec
#     def dJ(self):
#         self.biotsavart.set_points(self.coil.curve.gamma())

#         args = [
#             self.coil.curve.gamma(),
#             self.coil.curve.gammadash(),
#             self.coil.curve.gammadashdash(),
#             self.coil.current.get_value(),
#             self.biotsavart.B()
#         ]

#         dJ_dB = self.dJ_dB_mutual(*args)
#         dB_dX = self.biotsavart.dB_by_dX()
#         dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)

#         return (
#             self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args) + dJ_dX)
#             + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
#             + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args))
#             + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))
#             + self.biotsavart.B_vjp(dJ_dB)
#         )

#     return_fn_map = {'J': J, 'dJ': dJ}
