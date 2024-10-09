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


def coil_force(coil, allcoils, regularization):
    gammadash = coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    mutual_coils = [c for c in allcoils if c is not coil]
    mutual_field = BiotSavart(mutual_coils).set_points(coil.curve.gamma()).B()
    mutualforce = np.cross(coil.current.get_value() * tangent, mutual_field)
    selfforce = self_force(coil, regularization)
    return selfforce + mutualforce

def coil_net_forces(coils, allcoils, regularization):
    net_forces = np.zeros((len(coils), 3))
    for i, coil in enumerate(coils):
        Fi = coil_force(coil, allcoils, regularization[i])
        gammadash = coil.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        net_forces[i, :] += np.sum(gammadash_norm * Fi, axis=0) / gammadash.shape[0]
    return net_forces

def coil_torque(coil, allcoils, regularization):
    gamma = coil.curve.gamma()
    return np.cross(gamma, coil_force(coil, allcoils, regularization))

def coil_net_torques(coils, allcoils, regularization):
    net_torques = np.zeros((len(coils), 3))
    for i, coil in enumerate(coils):
        Ti = coil_torque(coil, allcoils, regularization[i])
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
    return (jnp.sum(jnp.maximum(force_norm - threshold, 0)**p * gammadash_norm)) / jnp.sum(gammadash_norm)


class LpCurveForce(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \text{max}(|\vec{F}| - F_0, 0)^p d\ell\right)

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


class MeanSquaredForce2(Optimizable):
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
        curvedofs = self.coil.curve.get_dofs()
        self.gamma = self.coil.curve.gamma_jax(curvedofs)
        self.gammadash = self.coil.curve.gammadash_jax(curvedofs)
        self.gammadashdash = self.coil.curve.gammadashdash_jax(curvedofs)

        self.J_jax = jit(
            lambda curve_dofs, current, B_mutual:
            self.mean_squared_force2_pure(curve_dofs, quadpoints, current, regularization, B_mutual)
        )

        self.dJ_dcurvedofs = jit(
            lambda curve_dofs, current, B_mutual:
            grad(self.J_jax, argnums=0)(curve_dofs, current, B_mutual)
        )

        self.dJ_dcurrent = jit(
            lambda curve_dofs, current, B_mutual:
            grad(self.J_jax, argnums=1)(curve_dofs, current, B_mutual)
        )

        self.dJ_dB_mutual = jit(
            lambda curve_dofs, current, B_mutual:
            grad(self.J_jax, argnums=2)(curve_dofs, current, B_mutual)
        )

        super().__init__(depends_on=allcoils)

    def mean_squared_force2_pure(self, curve_dofs, quadpoints, current, regularization, B_mutual):
        r"""Pure function for minimizing the Lorentz force on a coil.

        The function is

        .. math:
            J = \frac{\int |\vec{F}|^2 d\ell}{\int d\ell}

        where :math:`\vec{F}` is the Lorentz force and :math:`\ell` is arclength
        along the coil.
        """
        # print(jnp.shape(curve_dofs), jnp.shape(self.coil.curve.x))
        # self.coil.curve.set_dofs(np.array(curve_dofs))
        # gammas = jnp.zeros((len(self._coils), len(self._coils[0].curve.quadpoints), 3))
        # for i, c in enumerate(self._coils):
        #     gammas = gammas.at[i, :, :].add(c.curve.gamma_impl_jax(dofs[i % len(dofs)], self._coils[i].curve.quadpoints))
        # return jnp.array(gammas)
        # self.gamma = self.coil.curve.gamma_jax(curve_dofs)
        # self.gammadash = self.coil.curve.gammadash_jax(curve_dofs)
        # self.gammadashdash = self.coil.curve.gammadashdash_jax(curve_dofs)
        gamma = self.coil.curve.gamma_jax(curve_dofs) 
        gammadash = self.coil.curve.gammadash_jax(curve_dofs)
        B_self = B_regularized_pure(gamma, gammadash, 
            self.coil.curve.gammadashdash_jax(curve_dofs), 
            quadpoints, current, regularization)
        gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
        tangent = gammadash / gammadash_norm
        force = jnp.cross(current * tangent, B_self + B_mutual)
        force_norm = jnp.linalg.norm(force, axis=1)[:, None]
        return jnp.sum(gammadash_norm * force_norm**2) / jnp.sum(gammadash_norm)

    def J(self):
        self.biotsavart.set_points(self.coil.curve.gamma())

        curve_dofs = self.coil.curve.get_dofs()
        args = [
            curve_dofs,
            self.coil.current.get_value(),
            self.biotsavart.B()
        ]     

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):
        self.biotsavart.set_points(self.coil.curve.gamma())

        curve_dofs = self.coil.curve.get_dofs()
        args = [
            curve_dofs,
            self.coil.current.get_value(),
            self.biotsavart.B()
        ]     
        dJ_dB = self.dJ_dB_mutual(*args)
        dJ_dcurvedofs = self.dJ_dcurvedofs(*args)
        dJ_dcurvedofs = Derivative({self.coil.curve: dJ_dcurvedofs})
        #sum([Derivative({self.coil.curve: dJ_dcurvedofs[i]}) for i in range(len(dJ_dcurvedofs))])
        # print(jnp.shape(dJ_dcurvedofs), jnp.shape(dJ_dB), jnp.shape(self.biotsavart.B_vjp(dJ_dB)))
        # print(dJ_dcurvedofs, self.biotsavart.B_vjp(dJ_dB), 
        #     self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)])))
        # dB_dX = self.biotsavart.dB_by_dX()
        # dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)

        return (
            dJ_dcurvedofs   # dJ / dck term
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))  # dJ / dI term
            # + self.biotsavart.B_vjp(dJ_dB)  # dJ / dc term from B_mutual
            # Need a dJ / dc term from B_mutual, which seems to be = dJ / dB * dB / dc
            # but this is accounted for now directly in dJ_dcurvedofs. Isn't there also
            # a term like dJ / dB_mutual?
        )

    return_fn_map = {'J': J, 'dJ': dJ}

@jit
def squared_mean_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual):
    r"""
    """
    B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    force = jnp.cross(current * tangent, B_self + B_mutual)
    # force_norm = jnp.linalg.norm(force, axis=1)[:, None]
    return jnp.linalg.norm(jnp.sum(force * gammadash_norm, axis=0)) ** 2 / jnp.sum(gammadash_norm)  # factor for the integral

class SquaredMeanForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

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
            squared_mean_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual)
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
def squared_mean_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual):
    r"""
    """
    B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    force = jnp.cross(current * tangent, B_self + B_mutual)
    torque = jnp.cross(gamma, force)
    return jnp.linalg.norm(jnp.sum(gammadash_norm * torque, axis=0)) ** 2 / jnp.sum(gammadash_norm)  # factor for the integral

class SquaredMeanTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

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
            squared_mean_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual)
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
def mean_squared_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual):
    r"""
    """
    B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    force = jnp.cross(current * tangent, B_self + B_mutual)
    torque = jnp.cross(gamma, force)
    torque_norm = jnp.linalg.norm(torque, axis=1)[:, None]
    return jnp.sum(gammadash_norm * torque_norm ** 2) / jnp.sum(gammadash_norm)

class MeanSquaredTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell)^2

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
            mean_squared_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual)
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
def lp_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold):
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
    torque = jnp.cross(gamma, force)
    torque_norm = jnp.linalg.norm(torque, axis=1)[:, None]
    return (jnp.sum(jnp.maximum(torque_norm - threshold, 0)**p * gammadash_norm)) / jnp.sum(gammadash_norm)


class LpCurveTorque(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \text{max}(|\vec{F}| - F_0, 0)^p d\ell\right)

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
            lp_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold)
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
