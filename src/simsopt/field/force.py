"""Implements the force on a coil in its own magnetic field and the field of other coils."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from .biotsavart import BiotSavart
from .selffield import B_regularized_pure, B_regularized, B_regularized_circ, B_regularized_rect
from .coil import RegularizedCoil
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


def coil_force(target_coil, source_coils):
    r"""
    Compute the force per unit length on a coil from m other coils, in Newtons/meter. Note that BiotSavart objects
    are created below, which can lead to growth of the number of optimizable graph dependencies 
    created by this function. The target_coil must be a RegularizedCoil object, while the
    source_coils can be either Coil or RegularizedCoil objects.

    .. math::
        dF_i/d\ell = I_i \vec{t_i} \times (\vec{B_{self}} + \vec{B_{mutual}})

    where :math:`\vec{t_i}` is the tangent vector to the ith coil curve,
    :math:`\vec{B_{self}}` is the self-field of the ith coil,
    :math:`\vec{B_{mutual}}` is the mutual field from the other coils.

    Args:
        target_coil (RegularizedCoil): 
            RegularizedCoil to compute the pointwise forces on.
        source_coils (list of Coil or RegularizedCoil, shape (m,)): 
            List of coils contributing forces on the primary coil. 
            Can be a mix of Coil and RegularizedCoil objects.
    Returns:
        array (shape (n,3)): Array of forces per unit length along the coil curve.
    """
    if not isinstance(target_coil, RegularizedCoil):
        raise TypeError("coil_force requires target_coil to be a RegularizedCoil object, "
                       f"but got {type(target_coil).__name__}")
    gammadash = target_coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    mutual_coils = [c for c in source_coils if c is not target_coil]
    mutual_field = BiotSavart(mutual_coils).set_points(target_coil.curve.gamma())
    B_mutual = mutual_field.B()
    mutualforce = np.cross(target_coil.current.get_value() * tangent, B_mutual)
    selfforce = self_force(target_coil)
    return (selfforce + mutualforce)

def coil_net_force(target_coil, source_coils):
    r"""
    Compute the net forces on one coil from m other coils, in Newtons. This is
    the integrated pointwise force per unit length dF_i/d\ell on the ith coil curve.

    .. math::
        F_net = \int (dF_i/d\ell) d\ell

    Args:
        target_coil (RegularizedCoil): RegularizedCoil to compute the net forces on.
        source_coils (list of Coil or RegularizedCoil, shape (m,)): 
            List of coils contributing forces on the primary coil. 
            Can be a mix of Coil and RegularizedCoil objects.
    Returns:
        np.array (shape (3,)): Array of net forces.
    """
    Fi = coil_force(target_coil, source_coils)
    gammadash = target_coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    net_force = np.sum(gammadash_norm * Fi, axis=0) / gammadash.shape[0]
    return net_force

def _coil_force_pure(B, I, t):
    r"""
    Compute the pointwise Lorentz force per unit length on a coil with n quadrature points, in Newtons/meter. 

    .. math::
        dF/d\ell = I \vec{t} \times \vec{B}

    where :math:`\vec{t}` is the tangent vector to the coil curve,
    :math:`\vec{B}` is the magnetic field at the quadrature points,
    :math:`I` is the coil current.

    Args:
        B (array, shape (n,3)): Array of magnetic field.
        I (float): Coil current.
        t (array, shape (n,3)): Array of coil tangent vectors.
    Returns:
        array (shape (n,3)): Array of force per unit length.
    """
    return jnp.cross(I * t, B)


def coil_torque(target_coil, source_coils):
    r"""
    Compute the torques per unit length on a coil from m other coils in Newtons 
    (note that the force is per unit length, so the force has units of Newtons/meter 
    and the torques per unit length have units of Newtons).

    .. math::
        dT_i/d\ell = (\gamma_i - c_i) \times (dF_i/d\ell)

    where :math:`\gamma_i` is the position vector of the ith coil curve, 
    :math:`c_i` is the centroid of the ith coil curve,
    :math:`dF_i/d\ell` is the pointwise force per unit length on the ith coil curve.

    Args:
        target_coil (RegularizedCoil): RegularizedCoil to compute the pointwise torques on.
        source_coils (list of Coil or RegularizedCoil, shape (m,)): 
            List of coils contributing torques on the primary coil. 
            Can be a mix of Coil and RegularizedCoil objects.
    Returns:
        np.array (shape (n,3)): Array of torques per unit length along the coil curve.
    """
    if not isinstance(target_coil, RegularizedCoil):
        raise TypeError("coil_torque requires target_coil to be a RegularizedCoil object, "
                       f"but got {type(target_coil).__name__}")
    gamma = target_coil.curve.gamma()
    center = target_coil.curve.centroid()
    return np.cross(gamma - center, coil_force(target_coil, source_coils))


def coil_net_torque(target_coil, source_coils):
    r"""
    Compute the net torques on a coil from m other coils, in Newton-meters. This is
    the integrated pointwise torque per unit length on a coil curve.

    .. math::
        T_net = \int dT_i/d\ell d\ell

    Args:
        target_coil (RegularizedCoil): RegularizedCoil to compute the net torques on.
        source_coils (list of Coil or RegularizedCoil, shape (m,)): 
            List of coils contributing torques on the primary coil. 
            Can be a mix of Coil and RegularizedCoil objects.
    Returns:
        np.array (shape (3,)): Array of net torques.
    """
    if not isinstance(target_coil, RegularizedCoil):
        raise TypeError("coil_net_torque requires target_coil to be a RegularizedCoil object, "
                       f"but got {type(target_coil).__name__}")
    Ti = coil_torque(target_coil, source_coils)
    gammadash = target_coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    net_torque = np.sum(gammadash_norm * Ti, axis=0) / gammadash.shape[0]
    return net_torque


def self_force(target_coil):
    """
    Compute the self-force per unit length of a coil, in Newtons/meter.
    Args:
        target_coil (RegularizedCoil): RegularizedCoil to compute the self-force per unit length on.
    Returns:
        array (shape (n,3)): Array of self-force per unit length.
    """
    if not isinstance(target_coil, RegularizedCoil):
        raise TypeError("self_force requires target_coil to be a RegularizedCoil object, "
                       f"but got {type(target_coil).__name__}")
    I = target_coil.current.get_value()
    tangent = target_coil.curve.gammadash() / np.linalg.norm(target_coil.curve.gammadash(),
                                                      axis=1)[:, None]
    B = B_regularized(target_coil)
    return _coil_force_pure(B, I, tangent)


def self_force_circ(coil, a):
    """Compute the Lorentz self-force of a coil with circular cross-section"""
    I = coil.current.get_value()
    tangent = coil.curve.gammadash() / np.linalg.norm(coil.curve.gammadash(),
                                                      axis=1)[:, None]
    B = B_regularized_circ(coil, a)
    return _coil_force_pure(B, I, tangent)


def self_force_rect(coil, a, b):
    """Compute the Lorentz self-force of a coil with rectangular cross-section"""
    I = coil.current.get_value()
    tangent = coil.curve.gammadash() / np.linalg.norm(coil.curve.gammadash(),
                                                      axis=1)[:, None]
    B = B_regularized_rect(coil, a, b)
    return _coil_force_pure(B, I, tangent)


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
