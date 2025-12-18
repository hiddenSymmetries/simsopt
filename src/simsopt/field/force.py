"""Implements the force on a coil in its own magnetic field and the field of other coils."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscp
from jax import grad, vmap
from jax.lax import cond
from .biotsavart import BiotSavart
from .selffield import B_regularized_pure, B_regularized, regularization_circ, regularization_rect
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

__all__ = [
    "coil_force",
    "self_force_circ",
    "self_force_rect",
    "coil_coil_inductances_pure",
    "coil_coil_inductances_full_pure",
    "coil_coil_inductances_inv_pure",
    "NetFluxes",
    "TVE",
    "coil_net_forces",
    "coil_torque",
    "LpCurveForce_deprecated",
    "SquaredMeanForce_deprecated",
    "MeanSquaredForce_deprecated",
    "LpCurveTorque_deprecated",
    "SquaredMeanTorque_deprecated",
    "MeanSquaredTorque_deprecated",
    "SquaredMeanForce",
    "LpCurveForce",
    "SquaredMeanTorque",
    "LpCurveTorque",
]


def coil_force(coil, allcoils, regularization, nturns=1):
    """
    Compute the force on a coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize.
        regularization (Regularization): Regularization object.
        nturns (int): Number of coil turns.

    Returns:
        array: Array of force.
    """
    gammadash = coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    mutual_coils = [c for c in allcoils if c is not coil]

    ### Line below seems to be the issue -- all these BiotSavart objects seem to stick
    ### around and not to go out of scope after these calls!
    mutual_field = BiotSavart(mutual_coils).set_points(coil.curve.gamma())
    B_mutual = mutual_field.B()
    mutualforce = np.cross(coil.current.get_value() * tangent, B_mutual)
    selfforce = self_force(coil, regularization)
    return (selfforce + mutualforce) / nturns


def coil_net_forces(coils, allcoils, regularization, nturns=None):
    """
    Compute the net forces on a list of coils.

    Args:
        coils (list): List of coils to optimize.
        allcoils (list): List of coils to optimize.
        regularization (Regularization): Regularization object.
        nturns (list): List of coil turn numbers.

    Returns:
        array: Array of net forces.
    """
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
    """
    Compute the torques on a coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize.
        regularization (Regularization): Regularization object.
        nturns (int): Number of coil turns.

    Returns:
        array: Array of torques.
    """
    gamma = coil.curve.gamma()
    center = coil.curve.center(coil.curve.gamma(), coil.curve.gammadash())
    return np.cross(gamma - center, coil_force(coil, allcoils, regularization, nturns))


def coil_net_torques(coils, allcoils, regularization, nturns=None):
    """
    Compute the net torques on a list of coils.

    Args:
        coils (list): List of coils to optimize.
        allcoils (list): List of coils to optimize.
        regularization (Regularization): Regularization object.
        nturns (list): List of coil turn numbers.

    Returns:
        array: Array of net torques.
    """
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
    """
    force on coil for optimization

    Args:
        B (array): Array of magnetic field.
        I (array): Array of coil current.
        t (array): Array of coil tangent vectors.

    Returns:
        array: Array of force.
    """
    return jnp.cross(I * t, B)


def self_force(coil, regularization):
    """
    Compute the self-force of a coil.

    Args:
        coil (Coil): Coil to optimize.
        regularization (Regularization): Regularization object.

    Returns:
        array: Array of self-force.
    """
    I = coil.current.get_value()
    tangent = coil.curve.gammadash() / np.linalg.norm(coil.curve.gammadash(),
                                                      axis=1)[:, None]
    B = B_regularized(coil, regularization)
    return coil_force_pure(B, I, tangent)


def self_force_circ(coil, a):
    """
    Compute the Lorentz self-force of a coil with circular cross-section

    Args:
        coil (Coil): Coil to optimize.
        a (array): Array of coil positions.

    Returns:
        array: Array of self-force.
    """
    return self_force(coil, regularization_circ(a))


def self_force_rect(coil, a, b):
    """
    Compute the Lorentz self-force of a coil with rectangular cross-section

    Args:
        coil (Coil): Coil to optimize.
        a (array): Array of coil positions.
        b (array): Array of coil tangent vectors.

    Returns:
        array: Array of self-force.
    """
    return self_force(coil, regularization_rect(a, b))


def pointData_forces_torques(coils, allcoils, aprimes, bprimes, nturns_list):
    """
    Create an array of pointwise forces and torques for plotting in vtk files,
    usually using curves_to_vtk.

    Args:
        coils (list): List of coils to optimize.
        allcoils (list): List of coils to optimize.
        aprimes (array): Array of coil positions.
        bprimes (array): Array of coil tangent vectors.
        nturns_list (list): List of coil turn numbers.

    Returns:
        dict: Dictionary of pointwise forces and torques.
    """
    contig = np.ascontiguousarray
    forces = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    torques = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    for i, c in enumerate(coils):
        aprime = aprimes[i]
        bprime = bprimes[i]
        forces[i, :-1, :] = coil_force(c, allcoils, regularization_rect(aprime, bprime), nturns_list[i])
        torques[i, :-1, :] = coil_torque(c, allcoils, regularization_rect(aprime, bprime), nturns_list[i])

    forces[:, -1, :] = forces[:, 0, :]
    torques[:, -1, :] = torques[:, 0, :]
    forces = forces.reshape(-1, 3)
    torques = torques.reshape(-1, 3)
    point_data = {"Pointwise_Forces": (contig(forces[:, 0]), contig(forces[:, 1]), contig(forces[:, 2])),
                  "Pointwise_Torques": (contig(torques[:, 0]), contig(torques[:, 1]), contig(torques[:, 2]))}
    return point_data


# @jit
def lp_force_pure_deprecated(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample):
    r"""Pure function for minimizing the Lorentz force on a coil.

    The function is

     .. math::
        J = \frac{1}{p}\left(\int \text{max}(|\vec{F}| - F_0, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    and :math:`\ell` is arclength along the coil.

    Args:
        gamma (array): Array of coil positions.
        gammadash (array): Array of coil tangent vectors.
        gammadashdash (array): Array of coil tangent vectors.
        quadpoints (array): Array of quadrature points.
        current (array): Array of coil currents.
        regularization (Regularization): Regularization object.

    Returns:
        float: Value of the objective function.
    """
    gamma = gamma[::downsample, :]
    gammadash = gammadash[::downsample, :]
    gammadashdash = gammadashdash[::downsample, :]
    quadpoints = quadpoints[::downsample]
    gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    return (jnp.sum(jnp.maximum(
        jnp.linalg.norm(jnp.cross(
            current * tangent, B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization) + B_mutual
        ), axis=1)[:, None] - threshold, 0)**p * gammadash_norm) / jnp.shape(gamma)[0]) * (1. / p)


class LpCurveForce_deprecated(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil in MN/m

    The objective function is

    .. math::
        J = \frac{1}{pL}\left(\int \text{max}(|d\vec{F}/d\ell| - dF_0/d\ell, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize. 
        regularization (Regularization): Regularization object.
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, coil, allcoils, regularization, p=2.0, threshold=0.0, downsample=1):
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints
        self.downsample = downsample
        args = {"static_argnums": (5,)}
        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            lp_force_pure_deprecated(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dB_mutual = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        J = self.J_jax(gamma, self.coil.curve.gammadash(), self.coil.curve.gammadashdash(),
                       self.coil.current.get_value(), self.biotsavart.B(), self.downsample)

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()
        return J

    @derivative_dec
    def dJ(self):

        # First part related to dB terms cannot be downsampled!
        gamma = self.coil.curve.gamma()
        gammadash = self.coil.curve.gammadash()
        gammadashdash = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        self.biotsavart.set_points(gamma)
        args = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            1
        ]
        dJ_dB = self.dJ_dB_mutual(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)
        B_vjp = self.biotsavart.B_vjp(dJ_dB)

        # Second part related to coil dJ can be downsampled!
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args2 = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            self.downsample
        ]

        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args2) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args2))
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args2))
            + B_vjp
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
        )

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


# @jit
def mean_squared_force_pure_deprecated(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample):
    r"""Pure function for minimizing the Lorentz force on a coil.

    The function is

    .. math:
        J = \frac{\int |d\vec{F}/d\ell|^2 d\ell}{L}

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil.

    Args:
        gamma (array): Array of coil positions.
        gammadash (array): Array of coil tangent vectors.
        gammadashdash (array): Array of coil tangent vectors.
        quadpoints (array): Array of quadrature points.
        current (array): Array of coil current.
        regularization (Regularization): Regularization object.
        B_mutual (array): Array of mutual magnetic field.
        downsample (int): Downsample factor for the objective function.

    Returns:
        float: Value of the objective function.
    """
    quadpoints = quadpoints[::downsample]
    gamma = gamma[::downsample, :]
    gammadash = gammadash[::downsample, :]
    gammadashdash = gammadashdash[::downsample, :]
    B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
    tangent = gammadash / gammadash_norm
    force = jnp.cross(current * tangent, B_self + B_mutual)
    force_norm = jnp.linalg.norm(force, axis=1)[:, None]
    return jnp.sum(gammadash_norm * force_norm**2) / jnp.sum(gammadash_norm)


class MeanSquaredForce_deprecated(Optimizable):
    r"""Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math:
        J = \frac{\int |\vec{F}|^2 d\ell}{L}

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize. 
        regularization (Regularization): Regularization object.
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, coil, allcoils, regularization, downsample=1):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints
        self.downsample = downsample
        args = {"static_argnums": (5,)}

        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            mean_squared_force_pure_deprecated(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dB_mutual = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.curve.gammadashdash(),
            self.coil.current.get_value(),
            self.biotsavart.B(),
            self.downsample
        ]
        J = self.J_jax(*args)

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return J

    @derivative_dec
    def dJ(self):
        # First part related to dB terms cannot be downsampled!
        gamma = self.coil.curve.gamma()
        gammadash = self.coil.curve.gammadash()
        gammadashdash = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        self.biotsavart.set_points(gamma)
        args = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            1
        ]
        dJ_dB = self.dJ_dB_mutual(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)
        B_vjp = self.biotsavart.B_vjp(dJ_dB)

        # Second part related to coil dJ can be downsampled!
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args2 = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            self.downsample
        ]

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args2) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args2))
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args2))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
            + B_vjp
        )
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def squared_mean_force_pure_deprecated(current, gammadash, B_mutual, downsample):
    r"""
    Pure function for computing the squared mean force on a coil.

    Args:
        current (array): Array of coil current.
        gammadash (array): Array of coil tangent vectors.
        B_mutual (array): Array of mutual magnetic field.
        downsample (int): Downsample factor for the objective function.
    """
    gammadash = gammadash[::downsample, :]
    return (current * jnp.linalg.norm(jnp.sum(jnp.cross(gammadash, B_mutual), axis=0) / gammadash.shape[0])) ** 2


class SquaredMeanForce_deprecated(Optimizable):
    r"""Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell}{L})^2

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize.
        downsample (int): Downsample factor for the objective function.

    Returns:
        float: Value of the objective function.
    """

    def __init__(self, coil, allcoils, downsample=1):
        if not isinstance(downsample, int):
            raise ValueError("downsample must be an integer")
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        self.downsample = downsample
        args = {"static_argnums": (3,)}
        self.J_jax = jit(
            lambda current, gammadash, B_mutual, downsample:
            squared_mean_force_pure_deprecated(current, gammadash, B_mutual, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda current, gammadash, B_mutual, downsample:
            grad(self.J_jax, argnums=0)(current, gammadash, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda current, gammadash, B_mutual, downsample:
            grad(self.J_jax, argnums=1)(current, gammadash, B_mutual, downsample),
            **args
        )

        self.dJ_dB = jit(
            lambda current, gammadash, B_mutual, downsample:
            grad(self.J_jax, argnums=2)(current, gammadash, B_mutual, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))

        args = [
            self.coil.current.get_value(),
            self.coil.curve.gammadash(),
            self.biotsavart.B(),
            self.downsample,
        ]

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        # dJ_dB term needs the full non-downsampled term
        gamma = self.coil.curve.gamma()
        gammadash = self.coil.curve.gammadash()
        current = self.coil.current.get_value()
        self.biotsavart.set_points(gamma)
        args = [
            current,
            gammadash,
            self.biotsavart.B(),
            1,
        ]
        dJ_dB = self.dJ_dB(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)
        B_vjp = self.biotsavart.B_vjp(dJ_dB)

        # Remaining Jacobian terms can be downsampled for calculation
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args2 = [
            current,
            gammadash,
            self.biotsavart.B(),
            self.downsample,
        ]
        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args2))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
            + B_vjp
        )

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class SquaredMeanTorque_deprecated(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{\tau}_i d\ell}{L})^2

    where :math:`\vec{\tau}` is the Lorentz torque, L is the total coil length,
    and :math:`\ell` is arclength along the coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize.
        regularization (Regularization): Regularization object.
        downsample (int): Downsample factor for the objective function.

    Returns:
        float: Value of the objective function.
    """

    def squared_mean_torque_pure_deprecated(self, current, gamma, gammadash, B_mutual, downsample):
        r"""
        """
        gamma = gamma[::downsample, :]
        gammadash = gammadash[::downsample, :]
        return (current * jnp.linalg.norm(jnp.sum(jnp.cross(gamma - self.coil.curve.center(gamma, gammadash), jnp.cross(gammadash, B_mutual)), axis=0) / gamma.shape[0])) ** 2

    def __init__(self, coil, allcoils, regularization=None, downsample=1):
        if not isinstance(downsample, int):
            raise ValueError("downsample must be an integer")
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        self.downsample = downsample
        args = {"static_argnums": (4,)}
        self.J_jax = jit(
            lambda current, gamma, gammadash, B_mutual, downsample:
            self.squared_mean_torque_pure_deprecated(current, gamma, gammadash, B_mutual, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda current, gamma, gammadash, B_mutual, downsample:
            grad(self.J_jax, argnums=0)(current, gamma, gammadash, B_mutual, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda current, gamma, gammadash, B_mutual, downsample:
            grad(self.J_jax, argnums=1)(current, gamma, gammadash, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda current, gamma, gammadash, B_mutual, downsample:
            grad(self.J_jax, argnums=2)(current, gamma, gammadash, B_mutual, downsample),
            **args
        )

        self.dJ_dB = jit(
            lambda current, gamma, gammadash, B_mutual, downsample:
            grad(self.J_jax, argnums=3)(current, gamma, gammadash, B_mutual, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args = [
            self.coil.current.get_value(),
            gamma,
            self.coil.curve.gammadash(),
            self.biotsavart.B(),
            self.downsample
        ]
        J = self.J_jax(*args)

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()
        return J

    @derivative_dec
    def dJ(self):

        # First dJ_dB term cannot be downsampled
        current = self.coil.current.get_value()
        gamma = self.coil.curve.gamma()
        gammadash = self.coil.curve.gammadash()
        self.biotsavart.set_points(gamma)
        args = [
            current,
            gamma,
            gammadash,
            self.biotsavart.B(),
            1
        ]
        dJ_dB = self.dJ_dB(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)
        B_vjp = self.biotsavart.B_vjp(dJ_dB)

        # Remaining dJ terms can be downsampled before calculation
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args2 = [
            current,
            gamma,
            gammadash,
            self.biotsavart.B(),
            self.downsample
        ]
        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args2) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args2))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
            + B_vjp
        )

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class MeanSquaredTorque_deprecated(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (1/L) \frac{\int (\vec{\tau}_i)^2 d\ell

    where :math:`\vec{\tau}` is the Lorentz torque, L is the total coil length,
    and :math:`\ell` is arclength along the coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize.
        regularization (Regularization): Regularization object.
        downsample (int): Downsample factor for the objective function.

    Returns:
        float: Value of the objective function.
    """

    # @jit
    def mean_squared_torque_pure_deprecated(self, gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample):
        r"""
        """
        quadpoints = quadpoints[::downsample]
        gamma = gamma[::downsample, :]
        gammadash = gammadash[::downsample, :]
        gammadashdash = gammadashdash[::downsample, :]
        B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
        gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
        tangent = gammadash / gammadash_norm
        force = jnp.cross(current * tangent, B_self + B_mutual)
        torque = jnp.cross(gamma - self.coil.curve.center(gamma, gammadash), force)
        torque_norm = jnp.linalg.norm(torque, axis=1)[:, None]
        return jnp.sum(gammadash_norm * torque_norm ** 2) / jnp.sum(gammadash_norm)

    def __init__(self, coil, allcoils, regularization, downsample=1):
        self.coil = coil
        self.allcoils = allcoils
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints
        self.downsample = downsample
        args = {"static_argnums": (5,)}

        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            self.mean_squared_torque_pure_deprecated(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dB_mutual = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(gamma[::self.downsample, :])

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.curve.gammadashdash(),
            self.coil.current.get_value(),
            self.biotsavart.B(),
        ]
        J = self.J_jax(*args)

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return J

    @derivative_dec
    def dJ(self):

        # First dJ_dB term cannot be downsampled
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(gamma)
        gammadash = self.coil.curve.gammadash()
        gammadashdash = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        args = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            1
        ]
        dJ_dB = self.dJ_dB_mutual(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)
        B_vjp = self.biotsavart.B_vjp(dJ_dB)

        # Remaining terms can be downsampled before calculation
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args2 = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            self.downsample
        ]
        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args2) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args2))
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args2))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
            + B_vjp
        )

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class LpCurveTorque_deprecated(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{pL}\left(\int \text{max}(|\vec{T}| - T_0, 0)^p d\ell\right)

    where :math:`\vec{T}` is the Lorentz torque, :math:`T_0` is a threshold torque,  
    L is the total length of the coil,
    and :math:`\ell` is arclength along the coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize.
        regularization (Regularization): Regularization object.
        p (float): Power of the objective function.
        threshold (float): Threshold value for the objective function.
        downsample (int): Downsample factor for the objective function.

    Returns:
        float: Value of the objective function.
    """

    def lp_torque_pure_deprecated(self, gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample):
        r"""Pure function for minimizing the Lorentz force on a coil.

        The function is

        .. math::
            J = \frac{1}{pL}\left(\int \text{max}(|\vec{T}| - T_0, 0)^p d\ell\right)

        where :math:`\vec{T}` is the Lorentz torque, :math:`T_0` is a threshold torque,  
        L is the total length of the coil,
        and :math:`\ell` is arclength along the coil.

        Args:
            gamma (array): Array of coil positions.
            gammadash (array): Array of coil tangent vectors.
            gammadashdash (array): Array of coil tangent vectors.
            quadpoints (array): Array of quadrature points.
            current (array): Array of coil current.
            regularization (Regularization): Regularization object.
            B_mutual (array): Array of mutual magnetic field.
            p (float): Power of the objective function.
            threshold (float): Threshold value for the objective function.
            downsample (int): Downsample factor for the objective function.

        Returns:
            float: Value of the objective function.
        """
        gamma = gamma[::downsample, :]
        gammadash = gammadash[::downsample, :]
        gammadashdash = gammadashdash[::downsample, :]
        quadpoints = quadpoints[::downsample]
        B_self = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)
        gammadash_norm = jnp.linalg.norm(gammadash, axis=1)[:, None]
        tangent = gammadash / gammadash_norm
        force = jnp.cross(current * tangent, B_self + B_mutual)
        torque = jnp.cross(gamma - self.coil.curve.center(gamma, gammadash), force)
        torque_norm = jnp.linalg.norm(torque, axis=1)[:, None]
        return (jnp.sum(jnp.maximum(torque_norm - threshold, 0)**p * gammadash_norm) / gamma.shape[0]) * (1 / p)

    def __init__(self, coil, allcoils, regularization, p=2.0, threshold=0.0, downsample=1):
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints
        self.downsample = downsample

        args = {"static_argnums": (5,)}
        self.J_jax = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            self.lp_torque_pure_deprecated(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        self.dJ_dB_mutual = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual, downsample:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, current, B_mutual, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))

        args = [
            gamma,
            self.coil.curve.gammadash(),
            self.coil.curve.gammadashdash(),
            self.coil.current.get_value(),
            self.biotsavart.B(),
            self.downsample
        ]
        J = self.J_jax(*args)

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return J

    @derivative_dec
    def dJ(self):

        # First dJ_dB term cannot be downsampled
        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(gamma)
        gammadash = self.coil.curve.gammadash()
        gammadashdash = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        args = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            1
        ]
        dJ_dB = self.dJ_dB_mutual(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)
        B_vjp = self.biotsavart.B_vjp(dJ_dB)

        # Remaining terms can be downsampled before calculation
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args2 = [
            gamma,
            gammadash,
            gammadashdash,
            current,
            self.biotsavart.B(),
            self.downsample
        ]
        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args2) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args2))
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args2))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
            + B_vjp
        )

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def coil_coil_inductances_pure(gamma, gammadash, gammas2, gammadashs2, a, b, downsample, cross_section):
    """
    Calculate the mutual inductance between a single coil and a list of other coils.

    This function computes both:
    1. The self-inductance of the input coil (first element of output)
    2. The mutual inductances between the input coil and each coil in gammas2 (remaining elements)

    Args:
        gamma (array): Position vectors for the single coil, shape (nquadpoints, 3).
        gammadash (array): Tangent vectors for the single coil, shape (nquadpoints, 3).
        gammas2 (array): Position vectors for the list of other coils, shape (ncoils, nquadpoints, 3).
        gammadashs2 (array): Tangent vectors for the list of other coils, shape (ncoils, nquadpoints, 3).
        a (float): First cross-sectional dimension (radius for circular, width for rectangular).
        b (float): Second cross-sectional dimension (equal to a for circular, height for rectangular).
        downsample (int): Factor by which to downsample the coil points for computation.
        cross_section (str): Shape of the coil cross-section ('circular' or 'rectangular').

    Returns:
        array: Array of inductances, where:
              - First element [0] is the self-inductance of the input coil
              - Remaining elements [1:] are mutual inductances with each coil in gammas2
              Shape is (1 + ncoils,).
    """
    # Downsample if desired
    gamma = gamma[::downsample, :]
    gammadash = gammadash[::downsample, :]
    gammas2 = gammas2[:, ::downsample, :]
    gammadashs2 = gammadashs2[:, ::downsample, :]

    # Initialize output array for inductances
    Lij = jnp.zeros(1 + gammas2.shape[0])

    # Compute mutual inductances (i != j)
    r_ij = gamma[:, None, :] - gammas2[:, None, :, :]
    rij_norm = jnp.linalg.norm(r_ij, axis=-1)
    gammadash_prod = jnp.sum(gammadash[:, None, :] * gammadashs2[:, None, :, :], axis=-1)

    # Double sum over each of the closed curves for off-diagonal elements
    Lij = Lij.at[1:].add(jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=-1), axis=-1) /
                         (jnp.shape(gamma)[0] * jnp.shape(gammas2)[1]))

    # Compute self-inductance (diagonal element)
    eps = 1e-10  # Small number to avoid blowup during dJ() calculation
    r_ij_self = gamma[:, None, :] - gamma[None, :, :] + eps
    rij_norm_self = jnp.linalg.norm(r_ij_self, axis=-1)
    gammadash_prod_self = jnp.sum(gammadash[:, None, :] * gammadash[None, :, :], axis=-1)

    # Compute regularization based on cross-section type
    is_circular = cross_section == 'circular'

    # For circular cross-section: a/sqrt(e)
    circ_reg = a ** 2 / jnp.sqrt(jnp.exp(1.0))

    # For rectangular cross-section: exp(-25/6 + k) * sqrt(a*b)
    k = (4 * b) / (3 * a) * jnp.arctan2(a, b) + \
        (4 * a) / (3 * b) * jnp.arctan2(b, a) + \
        (b ** 2) / (6 * a ** 2) * jnp.log(b / a) + \
        (a ** 2) / (6 * b ** 2) * jnp.log(a / b) - \
        (a ** 4 - 6 * a ** 2 * b ** 2 + b ** 4) / \
        (6 * a ** 2 * b ** 2) * jnp.log(a / b + b / a)
    rect_reg = jnp.exp(-25.0/6.0 + k) * a * b

    # Select regularization based on cross-section type
    reg = jnp.where(is_circular, circ_reg, rect_reg)

    # Add self-inductance term
    Lij = Lij.at[0].add(jnp.sum(jnp.sum(gammadash_prod_self /
                                        jnp.sqrt(rij_norm_self**2 + reg), axis=-1), axis=-1) /
                        jnp.shape(gamma)[0]**2)

    return 1e-7 * Lij


def coil_coil_inductances_full_pure(gammas, gammadashs, a_list, b_list, downsample, cross_section):
    r"""Compute the full inductance matrix for a set of coils.

    The function computes both the self-inductance (diagonal elements) and mutual inductance
    (off-diagonal elements) for a set of coils. The calculation includes finite-build effects
    through either circular or rectangular cross-sections.

    Args:
        gammas (array): Array of coil positions.
        gammadashs (array): Array of coil tangent vectors.
        a_list (array): Array of coil radii or widths.
        b_list (array): Array of coil thicknesses or heights.
        downsample (int): Downsample factor for the objective function.
        cross_section (str): Cross-section type ('circular' or 'rectangular').

    Returns:
        array: Full inductance matrix Lij.
    """
    # Downsample if desired
    gammas = gammas[:, ::downsample, :]
    gammadashs = gammadashs[:, ::downsample, :]
    N = gammas.shape[0]

    # Compute Lij, i != j
    eps = 1e-10
    r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :] + eps
    rij_norm = jnp.linalg.norm(r_ij, axis=-1)
    gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1)

    # Double sum over each of the closed curves for off-diagonal elements
    Lij = jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=-1), axis=-1) / jnp.shape(gammas)[1] ** 2

    # Compute diagonal elements based on cross-section type
    # For circular cross-section
    diag_circ = jnp.sum(jnp.sum(gammadash_prod / jnp.sqrt(rij_norm ** 2 + a_list[None, :, None, None] ** 2 / jnp.sqrt(jnp.exp(1.0))),
                                axis=-1), axis=-1) / jnp.shape(gammas)[1] ** 2

    # For rectangular cross-section
    k = (4 * b_list) / (3 * a_list) * jnp.arctan2(a_list, b_list) + \
        (4 * a_list) / (3 * b_list) * jnp.arctan2(b_list, a_list) + \
        (b_list ** 2) / (6 * a_list ** 2) * jnp.log(b_list / a_list) + \
        (a_list ** 2) / (6 * b_list ** 2) * jnp.log(a_list / b_list) - \
        (a_list ** 4 - 6 * a_list ** 2 * b_list ** 2 + b_list ** 4) / \
        (6 * a_list ** 2 * b_list ** 2) * jnp.log(a_list / b_list + b_list / a_list)
    delta_ab = jnp.exp(-25.0 / 6.0 + k) * a_list * b_list
    diag_rect = jnp.sum(jnp.sum(gammadash_prod / jnp.sqrt(rij_norm ** 2 + delta_ab[None, :, None, None]),
                                axis=-1), axis=-1) / jnp.shape(gammas)[1] ** 2

    # Use where to select diagonal elements based on cross-section type
    diag_mask = jnp.eye(N, dtype=bool)
    diag_values = jnp.where(cross_section == 'circular', diag_circ, diag_rect)

    # Update diagonal elements
    Lij = jnp.where(diag_mask, diag_values, Lij)
    return 1e-7 * Lij


def coil_coil_inductances_inv_pure(gammas, gammadashs, a_list, b_list, downsample, cross_section):
    """
    Pure function for computing the inverse of the coil inductance matrix.

    Args:
        gammas (array): Array of coil positions.
        gammadashs (array): Array of coil tangent vectors.
        a_list (array): Array of coil radii.
        b_list (array): Array of coil thicknesses.
        downsample (int): Downsample factor for the objective function.
        cross_section (str): Cross-section type.

    Returns:
        array: Array of inverse of the coil inductance matrix.
    """
    # Lij is symmetric positive definite so has a cholesky decomposition
    C = jnp.linalg.cholesky(coil_coil_inductances_full_pure(gammas, gammadashs, a_list, b_list, downsample, cross_section))
    inv_C = jscp.linalg.solve_triangular(C, jnp.eye(C.shape[0]), lower=True)
    inv_L = jscp.linalg.solve_triangular(C.T, inv_C, lower=False)
    return inv_L


def coil_currents_barebones(gammas, gammadashs, gammas_TF, gammadashs_TF, currents_TF, a_list, b_list, downsample, cross_section):
    """
    Pure function for computing the induced currents in a set of passive coils 
    due to a set of TF coils + the other passive coils. 

    Args:
        gammas (array): Array of passive coil positions.
        gammadashs (array): Array of passive coil tangent vectors.
        gammas_TF (array): Array of TF coil positions.
        gammadashs_TF (array): Array of TF coil tangent vectors.
        currents_TF (array): Array of TF coil current.
        a_list (array): Array of passive coil radii.
        b_list (array): Array of passive coil thicknesses.
        downsample (int): Downsample factor for the objective function.
        cross_section (str): Cross-section type.

    Returns:
        array: Array of induced currents.
    """
    return -coil_coil_inductances_inv_pure(gammas, gammadashs, a_list, b_list, downsample, cross_section) @ net_fluxes_pure(gammas, gammadashs, gammas_TF, gammadashs_TF, currents_TF, downsample)


def tve_pure(gamma, gammadash, gammas2, gammadashs2, current, currents2, a, b, downsample, cross_section):
    r"""Pure function for minimizing the total vacuum energy on a coil.

    The function is

     .. math::
        J = \frac{1}{2}I_iL_{ij}I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil.

    Args:
        gamma (array): Array of coil positions.
        gammadash (array): Array of coil tangent vectors.
        gammas2 (array): Array of coil positions.
        gammadashs2 (array): Array of coil tangent vectors.
        current (array): Array of coil current.
        currents2 (array): Array of coil current.
        a (float): Coil radius.
        b (float): Coil thickness.
        downsample (int): Downsample factor for the objective function.
        cross_section (str): Cross-section type.   

    Returns:
        float: Value of the objective function.
    """
    Ii_Ij = (current * currents2)
    Lij = coil_coil_inductances_pure(
        gamma, gammadash, gammas2, gammadashs2, a, b, downsample, cross_section
    )
    U = 0.5 * (jnp.sum(Ii_Ij * Lij[1:]) + Lij[0] * current ** 2)
    return U


class TVE(Optimizable):
    r"""Optimizable class for minimizing the total vacuum energy on a coil.

    The function is

     .. math::
        J = \frac{1}{2}I_iL_{ij}I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil.

    Args:
        coil (Coil): Coil to optimize.
        allcoils (list): List of coils to optimize.
        a (float): Coil radius.
        b (float): Coil thickness.
        downsample (int): Downsample factor for the objective function.
        cross_section (str): Cross-section type.
        psc_array (PSCArray): PSCoils to optimize.
    """

    def __init__(self, coil, allcoils, a=0.05, b=None, downsample=1, cross_section='circular', psc_array=None):
        import warnings
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not coil]
        self.downsample = downsample
        self.psc_array = psc_array
        if psc_array is not None:
            warnings.warn("PSCArray does NOT work in TVE objective unless all the coils are "
                          "in the passive coil array.")
        if b is None:
            b = a

        args = {"static_argnums": (6,)}
        self.J_jax = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample:
            tve_pure(gamma, gammadash, gammas2, gammadashs2, current, currents2, a, b, downsample, cross_section),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample),
            **args
        )

        self.dJ_dgammas2 = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample),
            **args
        )

        self.dJ_dgammadashs2 = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample),
            **args
        )

        self.dJ_dcurrents2 = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample:
            grad(self.J_jax, argnums=5)(gamma, gammadash, gammas2, gammadashs2, current, currents2, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            jnp.array([c.curve.gamma() for c in self.othercoils]),
            jnp.array([c.curve.gammadash() for c in self.othercoils]),
            self.coil.current.get_value(),
            jnp.array([c.current.get_value() for c in self.othercoils]),
            self.downsample
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            jnp.array([c.curve.gamma() for c in self.othercoils]),
            jnp.array([c.curve.gammadash() for c in self.othercoils]),
            self.coil.current.get_value(),
            jnp.array([c.current.get_value() for c in self.othercoils]),
            self.downsample
        ]
        dJ_dgammas2 = self.dJ_dgammas2(*args)
        dJ_dgammadashs2 = self.dJ_dgammadashs2(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dcurrents2 = self.dJ_dcurrents2(*args)
        # Passive coils require extra contribution to the objective gradients from the current dependence
        if self.psc_array is not None:
            vjp = self.psc_array.vjp_setup(np.array(np.hstack([dJ_dcurrent, dJ_dcurrents2])))
        else:
            vjp = self.coil.current.vjp(jnp.asarray([dJ_dcurrent]))
            vjp += sum([c.current.vjp(jnp.asarray([dJ_dcurrents2[i]])) for i, c in enumerate(self.othercoils)])

        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args))
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgammas2[i]) for i, c in enumerate(self.othercoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadashs2[i]) for i, c in enumerate(self.othercoils)])
            + vjp
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def net_fluxes_pure(gammas, gammadashs, gammas2, gammadashs2, currents2, downsample):
    """
    Calculate the net magnetic flux through a set of coils due to another set of coils.

    This function computes the total magnetic flux passing through a set of coils
    due to the magnetic field generated by another set of coils. The flux is calculated
    using the line integral of the vector potential along the coil paths.

    Args:
        gammas (array): Position vectors for the coils receiving flux.
        gammadashs (array): Tangent vectors for the coils receiving flux.
        gammas2 (array): Position vectors for the coils generating flux.
        gammadashs2 (array): Tangent vectors for the coils generating flux.
        currents2 (array): Current values for the coils generating flux.
        downsample (int): Factor by which to downsample the coil points for computation.

    Returns:
        array: Net magnetic flux through each coil in the first set.
    """
    # Downsample if desired
    gammas = gammas[:, ::downsample, :]
    gammadashs = gammadashs[:, ::downsample, :]
    gammas2 = gammas2[:, ::downsample, :]
    gammadashs2 = gammadashs2[:, ::downsample, :]
    rij_norm = jnp.linalg.norm(gammas[:, :, None, None, :] - gammas2[None, None, :, :, :], axis=-1)
    # sum over the currents, and sum over the biot savart integral
    A_ext = jnp.sum(currents2[None, None, :, None] * jnp.sum(gammadashs2[None, None, :, :, :] / rij_norm[:, :, :, :, None], axis=-2), axis=-2) / jnp.shape(gammadashs2)[1]
    # Now sum over the PSC coil loops
    return 1e-7 * jnp.sum(jnp.sum(A_ext * gammadashs, axis=-1), axis=-1) / jnp.shape(gammadashs)[1]


def net_ext_fluxes_pure(gammadash, A_ext, downsample):
    """
    Calculate the net magnetic flux through a coil due to an external vector potential.

    This function computes the total magnetic flux passing through a coil due to
    an external magnetic field represented by its vector potential. The flux is
    calculated using the line integral of the vector potential along the coil path.

    Args:
        gammadash (array): Tangent vectors along the coil.
        A_ext (array): External vector potential evaluated at coil points.
        downsample (int): Factor by which to downsample the coil points for computation.

    Returns:
        float: Net magnetic flux through the coil due to the external field.
    """
    # Downsample if desired
    gammadash = gammadash[::downsample, :]
    # Dot the vectors (sum over last axis), then sum over the quadpoints
    return jnp.sum(jnp.sum(A_ext * gammadash, axis=-1), axis=-1) / jnp.shape(gammadash)[0]


class NetFluxes(Optimizable):
    r"""Optimizable class for minimizing the total net flux (from an external field)
    through a coil. Unclear why you would want to do this.
    This is mostly a test class for the passive coil arrays.

    The function is

     .. math::
        \Psi = \int A_{ext}\cdot d\ell / L

    where :math:`A_{ext}` is the vector potential of an external magnetic field,
    evaluated along the quadpoints along the curve,
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
    """

    def __init__(self, coil, othercoils, downsample=1):
        self.coil = coil
        self.othercoils = [c for c in othercoils if c is not coil]  # just to double check coil is not in there
        self.downsample = downsample
        self.biotsavart = BiotSavart(self.othercoils)

        args = {"static_argnums": (2,)}
        self.J_jax = jit(
            lambda gammadash, A_ext, downsample:
            net_ext_fluxes_pure(gammadash, A_ext, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gammadash, A_ext, downsample:
            grad(self.J_jax, argnums=0)(gammadash, A_ext, downsample),
            **args
        )

        self.dJ_dA = jit(
            lambda gammadash, A_ext, downsample:
            grad(self.J_jax, argnums=1)(gammadash, A_ext, downsample),
            **args
        )

        super().__init__(depends_on=[coil] + othercoils)

    def J(self):

        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args = [
            self.coil.curve.gammadash(),
            self.biotsavart.A(),
            self.downsample
        ]
        J = self.J_jax(*args)
        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return J

    @derivative_dec
    def dJ(self):

        gamma = self.coil.curve.gamma()
        self.biotsavart.set_points(gamma)
        args = [
            self.coil.curve.gammadash(),
            self.biotsavart.A(),
            1
        ]

        dJ_dA = self.dJ_dA(*args)
        dA_dX = self.biotsavart.dA_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dA, dA_dX)
        A_vjp = self.biotsavart.A_vjp(dJ_dA)

        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(dJ_dX) +
            self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + A_vjp
        )

        #### LINES BELOW ARE RELATED TO OPEN SIMSOPT bug
        # Without these lines, the number of optimizable references multiply
        # like crazy as number of coils increases, slowing the optimization to a halt.
        # With these lines, the Jacobian calculations of these terms will be incorrect
        # with python 3.10 onwards (python 3.9 works regardless!)
        # self.biotsavart._children = set()
        # self.coil._children = set()
        # self.coil.curve._children = set()
        # self.coil.current._children = set()
        # for c in self.othercoils:
        #     c._children = set()
        #     c.curve._children = set()
        #     c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample):
    all_lengths = [g.shape[0] for g in gammas] + [g2.shape[0] for g2 in gammas2]
    min_npts = min(all_lengths)

    def subsample(arr, target_n):
        arr = jnp.asarray(arr)
        n = arr.shape[0]
        if n == target_n:
            idxs = jnp.arange(0, n, downsample)
        else:
            idxs = jnp.linspace(0, n-1, target_n).round().astype(int)
            idxs = idxs[::downsample]
        return arr[idxs, ...]
    gammas = jnp.stack([subsample(g, min_npts) for g in gammas])
    gammadashs = jnp.stack([subsample(g, min_npts) for g in gammadashs])
    gammas2 = jnp.stack([subsample(g, min_npts) for g in gammas2])
    gammadashs2 = jnp.stack([subsample(g, min_npts) for g in gammadashs2])
    currents = jnp.array(currents)
    currents2 = jnp.array(currents2)

    n1 = gammas.shape[0]
    n2 = gammas2.shape[0]
    npts1 = gammas.shape[1]
    npts2 = gammas2.shape[1]
    eps = 1e-10

    # Precompute tangents and norms
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
    tangents = gammadashs / gammadash_norms

    def mutual_B_field_group1(i, pt):
        def biot_savart_from_j(j):
            return cond(
                j == i,
                lambda _: jnp.zeros(3),
                lambda _: jnp.asarray(jnp.sum(
                    jnp.cross(gammadashs[j], pt - gammas[j]) /
                    (jnp.linalg.norm(pt - gammas[j] + eps, axis=1) ** 3)[:, None],
                    axis=0
                ) * currents[j]),
                operand=None
            )

        def biot_savart_from_j2(j2):
            return jnp.sum(jnp.cross(gammadashs2[j2], pt - gammas2[j2]) / (jnp.linalg.norm(pt - gammas2[j2] + eps, axis=1) ** 3)[:, None], axis=0) * currents2[j2]
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        B_mutual2 = jnp.sum(vmap(biot_savart_from_j2)(jnp.arange(n2)), axis=0)
        return (B_mutual1 / npts1) + (B_mutual2 / npts2)

    def mean_force_group1(i, gamma_i, tangent_i, gammadash_norm_i, current_i):
        # Compute force at each point
        def force_at_point(idx):
            return current_i * jnp.cross(tangent_i[idx], mutual_B_field_group1(i, gamma_i[idx])) * gammadash_norm_i[idx, 0]
        return jnp.sum(vmap(force_at_point)(jnp.arange(npts1)), axis=0) / gammadash_norm_i.shape[0]

    mean_forces = vmap(mean_force_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas, tangents, gammadash_norms, currents
    )
    summ = jnp.sum(jnp.linalg.norm(mean_forces, axis=-1) ** 2)
    return summ * 1e-14


class SquaredMeanForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell}{L})^2

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil. This class assumes 
    there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. This is also useful for optimizing passive coils,
    which require a modified Jacobian calculation.

    Args:
        allcoils (list): List of coils to optimize. If using passive coils, this should be the passive coils.
        allcoils2 (list): List of coils to optimize. If using passive coils, this should be the TF coils. 
        downsample (int): Downsample factor for the objective function.
        psc_array (PSCAarray): PSCArray object. If using passive coils, this should be the PSCArray object.
    """

    def __init__(self, allcoils, allcoils2, downsample=1, psc_array=None):
        if not isinstance(downsample, int):
            raise ValueError("downsample must be an integer")
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        self.downsample = downsample
        self.psc_array = psc_array
        args = {"static_argnums": (6,)}

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        super().__init__(depends_on=(allcoils + allcoils2))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        if self.psc_array is not None:
            vjp = self.psc_array.vjp_setup(np.array(dJ_dcurrent))
        else:
            vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])
        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + vjp
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def lp_force_pure(
    gammas, gammas2, gammadashs, gammadashs2, gammadashdashs,
    quadpoints, currents, currents2, regularizations, p, threshold, downsample=1
):
    """
    Computes the mixed Lp force objective by summing over all coils in the first set, where each coil receives force from all coils (including itself and the second set).
    This version allows each coil to have its own quadrature points array.
    """
    all_lengths = [g.shape[0] for g in gammas] + [g2.shape[0] for g2 in gammas2]
    min_npts = min(all_lengths)

    def subsample(arr, target_n):
        arr = jnp.asarray(arr)
        n = arr.shape[0]
        if n == target_n:
            idxs = jnp.arange(0, n, downsample)
        else:
            idxs = jnp.linspace(0, n-1, target_n).round().astype(int)
            idxs = idxs[::downsample]
        return arr[idxs, ...]
    gammas = jnp.stack([subsample(g, min_npts) for g in gammas])
    gammadashs = jnp.stack([subsample(g, min_npts) for g in gammadashs])
    gammadashdashs = jnp.stack([subsample(g, min_npts) for g in gammadashdashs])
    quadpoints = jnp.stack([subsample(q, min_npts) for q in quadpoints])
    quadpoints = quadpoints[0]
    gammas2 = jnp.stack([subsample(g, min_npts) for g in gammas2])
    gammadashs2 = jnp.stack([subsample(g, min_npts) for g in gammadashs2])
    currents = jnp.array(currents)
    currents2 = jnp.array(currents2)
    regularizations = jnp.atleast_1d(jnp.squeeze(regularizations))

    n1 = gammas.shape[0]
    n2 = gammas2.shape[0]
    npts1 = gammas.shape[1]
    npts2 = gammas2.shape[1]
    eps = 1e-10

    # Precompute tangents and norms
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
    tangents = gammadashs / gammadash_norms

    # Precompute B_self for each coil
    B_self = vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0))(
        gammas, gammadashs, gammadashdashs, quadpoints, currents, regularizations
    )

    # Helper to compute mutual field at each point for a coil
    def mutual_B_field_group1(i, pt):
        def biot_savart_from_j(j):
            B = cond(
                j == i,
                lambda _: jnp.zeros(3),
                lambda _: jnp.asarray(jnp.sum(
                    jnp.cross(gammadashs[j], pt - gammas[j]) /
                    (jnp.linalg.norm(pt - gammas[j] + eps, axis=1) ** 3)[:, None],
                    axis=0
                ) * currents[j]),
                operand=None
            )
            return B

        def biot_savart_from_j2(j2):
            return jnp.sum(jnp.cross(gammadashs2[j2], pt - gammas2[j2]) / (jnp.linalg.norm(pt - gammas2[j2] + eps, axis=1) ** 3)[:, None], axis=0) * currents2[j2]
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        B_mutual2 = jnp.sum(vmap(biot_savart_from_j2)(jnp.arange(n2)), axis=0)
        return (B_mutual1 / npts1) + (B_mutual2 / npts2)

    def per_coil_obj_group1(i, gamma_i, tangent_i, B_self_i, current_i):
        def force_at_point(idx):
            F = current_i * (mutual_B_field_group1(i, gamma_i[idx]) * 1e-7 + B_self_i[idx])
            return jnp.linalg.norm(jnp.cross(tangent_i[idx], F))
        return vmap(force_at_point)(jnp.arange(npts1))

    obj1 = vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas, tangents, B_self, currents
    )

    # Only sum over the first group
    return (jnp.sum(jnp.sum(jnp.maximum(obj1 - threshold, 0) ** p * gammadash_norms[:, :, 0])) / npts1) * (1. / p)


class LpCurveForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{pL}\left(\int \text{max}(|d\vec{F}/d\ell| - dF_0/d\ell, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. 

    Args:
        allcoils (list): List of coils to optimize. If using passive coils, this should be the passive coils.
        allcoils2 (list): List of coils to optimize. If using passive coils, this should be the TF coils. 
        regularizations (list): List of regularizations for the coils.
        regularizations2 (list): List of regularizations for the coils.
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, allcoils, allcoils2, regularizations, p=2.0, threshold=0.0, downsample=1, psc_array=None):
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        if not isinstance(regularizations, list):
            regularizations = [regularizations]
        regularizations = jnp.array(regularizations)
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        quadpoints = [c.curve.quadpoints for c in allcoils]
        self.downsample = downsample
        self.psc_array = psc_array
        args = {"static_argnums": (7,)}
        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            lp_force_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, quadpoints,
                          currents, currents2, regularizations, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
                grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
                grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
                grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
                grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
                grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
                grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )
        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
                grad(self.J_jax, argnums=6)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        super().__init__(depends_on=(allcoils + allcoils2))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.curve.gammadashdash() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.curve.gammadashdash() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dgammadashdash = self.dJ_dgammadashdash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        if self.psc_array is not None:
            vjp = self.psc_array.vjp_setup(np.array(dJ_dcurrent))
        else:
            vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])
        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.allcoils)])
            + vjp
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def lp_torque_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs,
                   quadpoints, currents, currents2, regularizations, p, threshold, downsample):
    r"""
    Pure function for computing the mixed lp torque on a coil.

    Args:
        gammas (array): Array of coil positions.
        gammas2 (array): Array of coil positions.
        gammadashs (array): Array of coil tangent vectors.
        gammadashs2 (array): Array of coil tangent vectors.
        quadpoints (array): Array of quadrature points.
        quadpoints2 (array): Array of quadrature points.
        currents (array): Array of coil currents.
        currents2 (array): Array of coil currents.
        regularizations (array): Array of coil regularizations.
        regularizations2 (array): Array of coil regularizations.
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): Downsample factor for the objective function.

    Returns:
        float: Value of the objective function.
    """
    all_lengths = [g.shape[0] for g in gammas] + [g2.shape[0] for g2 in gammas2]
    min_npts = min(all_lengths)

    def subsample(arr, target_n):
        arr = jnp.asarray(arr)
        n = arr.shape[0]
        if n == target_n:
            idxs = jnp.arange(0, n, downsample)
        else:
            idxs = jnp.linspace(0, n-1, target_n).round().astype(int)
            idxs = idxs[::downsample]
        return arr[idxs, ...]
    gammas = jnp.stack([subsample(g, min_npts) for g in gammas])
    gammadashs = jnp.stack([subsample(g, min_npts) for g in gammadashs])
    gammadashdashs = jnp.stack([subsample(g, min_npts) for g in gammadashdashs])
    quadpoints = jnp.stack([subsample(q, min_npts) for q in quadpoints])
    quadpoints = quadpoints[0]
    gammas2 = jnp.stack([subsample(g, min_npts) for g in gammas2])
    gammadashs2 = jnp.stack([subsample(g, min_npts) for g in gammadashs2])
    currents = jnp.array(currents)
    currents2 = jnp.array(currents2)
    regularizations = jnp.atleast_1d(jnp.squeeze(regularizations))

    def center(gamma, gammadash):
        # Compute the centroid of the curve
        arclength = jnp.linalg.norm(gammadash, axis=-1)
        barycenter = jnp.sum(gamma * arclength[:, None], axis=0) / jnp.sum(arclength)
        return barycenter

    centers = vmap(center, in_axes=(0, 0))(gammas, gammadashs)

    # Precompute B_self for each coil
    B_self = vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0))(
        gammas, gammadashs, gammadashdashs, quadpoints, currents, regularizations
    )
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
    tangents = gammadashs / gammadash_norms

    n1 = gammas.shape[0]
    n2 = gammas2.shape[0]
    npts1 = gammas.shape[1]
    npts2 = gammas2.shape[1]
    eps = 1e-10

    # Helper to compute mutual field at each point for a coil
    def mutual_B_field_group1(i, pt):
        def biot_savart_from_j(j):
            return cond(
                j == i,
                lambda _: jnp.zeros(3),
                lambda _: jnp.asarray(jnp.sum(
                    jnp.cross(gammadashs[j], pt - gammas[j]) /
                    (jnp.linalg.norm(pt - gammas[j] + eps, axis=1) ** 3)[:, None],
                    axis=0
                ) * currents[j]),
                operand=None
            )

        def biot_savart_from_j2(j2):
            return jnp.sum(jnp.cross(gammadashs2[j2], pt - gammas2[j2]) / (jnp.linalg.norm(pt - gammas2[j2] + eps, axis=1) ** 3)[:, None], axis=0) * currents2[j2]
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        B_mutual2 = jnp.sum(vmap(biot_savart_from_j2)(jnp.arange(n2)), axis=0)
        return ((B_mutual1 / npts1) + (B_mutual2 / npts2)) * 1e-7

    def per_coil_obj_group1(i, gamma_i, center_i, tangent_i, B_self_i, current_i):
        def torque_at_point(idx):
            B_mutual = mutual_B_field_group1(i, gamma_i[idx])
            F = current_i * jnp.cross(tangent_i[idx], B_mutual + B_self_i[idx])
            tau = jnp.cross(gamma_i[idx] - center_i, F)
            return jnp.linalg.norm(tau)
        return vmap(torque_at_point)(jnp.arange(npts1))

    obj1 = vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas, centers, tangents, B_self, currents
    )

    return jnp.sum(jnp.sum(jnp.maximum(obj1 - threshold, 0) ** p * gammadash_norms[:, :, 0])) / npts1 * (1. / p)


class LpCurveTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{pL}\left(\int \text{max}(|d\vec{\tau}/d\ell| - d\tau_0/d\ell, 0)^p d\ell\right)

    where :math:`\vec{\tau}` is the Lorentz torque, :math:`\tau_0` is a threshold torque,  
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. 

    Args:
        allcoils (list): List of coils to optimize.
        allcoils2 (list): List of coils to optimize.
        regularizations (list): List of regularizations for the coils.
        regularizations2 (list): List of regularizations for the coils.
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): Downsample factor for the objective function.
        psc_array (PSCArray): PSC coil array to use for the objective function.
    """

    def __init__(self, allcoils, allcoils2, regularizations, p=2.0, threshold=0.0, downsample=1, psc_array=None):
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        if not isinstance(regularizations, list):
            regularizations = [regularizations]
        regularizations = jnp.array(regularizations)
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        quadpoints = [c.curve.quadpoints for c in allcoils]
        self.downsample = downsample
        self.psc_array = psc_array
        args = {"static_argnums": (7,)}

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            lp_torque_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, quadpoints,
                           currents, currents2, regularizations, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample:
            grad(self.J_jax, argnums=6)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, currents, currents2, downsample),
            **args
        )

        super().__init__(depends_on=(allcoils + allcoils2))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.curve.gammadashdash() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.curve.gammadashdash() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dgammadashdash = self.dJ_dgammadashdash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        if self.psc_array is not None:
            vjp = self.psc_array.vjp_setup(np.array(dJ_dcurrent))
        else:
            vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.allcoils)])
            + vjp
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def squared_mean_torque(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample):
    all_lengths = [g.shape[0] for g in gammas] + [g2.shape[0] for g2 in gammas2]
    min_npts = min(all_lengths)

    def subsample(arr, target_n):
        arr = jnp.asarray(arr)
        n = arr.shape[0]
        if n == target_n:
            idxs = jnp.arange(0, n, downsample)
        else:
            idxs = jnp.linspace(0, n-1, target_n).round().astype(int)
            idxs = idxs[::downsample]
        return arr[idxs, ...]
    gammas = jnp.stack([subsample(g, min_npts) for g in gammas])
    gammadashs = jnp.stack([subsample(g, min_npts) for g in gammadashs])
    gammas2 = jnp.stack([subsample(g, min_npts) for g in gammas2])
    gammadashs2 = jnp.stack([subsample(g, min_npts) for g in gammadashs2])
    currents = jnp.array(currents)
    currents2 = jnp.array(currents2)

    n1 = gammas.shape[0]
    n2 = gammas2.shape[0]
    npts1 = gammas.shape[1]
    npts2 = gammas2.shape[1]
    eps = 1e-10

    def center(gamma, gammadash):
        arclength = jnp.linalg.norm(gammadash, axis=-1)
        barycenter = jnp.sum(gamma * arclength[:, None], axis=0) / jnp.sum(arclength)
        return barycenter

    centers = jnp.stack([center(g, gd) for g, gd in zip(gammas, gammadashs)])

    # Helper to compute mutual field at each point for a coil
    def mutual_B_field_group1(i, pt):
        def biot_savart_from_j(j):
            return cond(
                j == i,
                lambda _: jnp.zeros(3),
                lambda _: jnp.asarray(jnp.sum(
                    jnp.cross(gammadashs[j], pt - gammas[j]) /
                    (jnp.linalg.norm(pt - gammas[j] + eps, axis=1) ** 3)[:, None],
                    axis=0
                ) * currents[j]),
                operand=None
            )

        def biot_savart_from_j2(j2):
            return jnp.sum(jnp.cross(gammadashs2[j2], pt - gammas2[j2]) / (jnp.linalg.norm(pt - gammas2[j2] + eps, axis=1) ** 3)[:, None], axis=0) * currents2[j2]
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        B_mutual2 = jnp.sum(vmap(biot_savart_from_j2)(jnp.arange(n2)), axis=0)
        return (B_mutual1 / npts1) + (B_mutual2 / npts2)

    def mean_torque_group1(i, gamma_i, gammadash_i, center_i, current_i):
        arclength = jnp.linalg.norm(gammadash_i, axis=-1)
        tangent = gammadash_i / arclength[:, None]

        def torque_at_point(idx):
            B_mutual = mutual_B_field_group1(i, gamma_i[idx])
            F = current_i * jnp.cross(tangent[idx], B_mutual)
            tau = jnp.cross(gamma_i[idx] - center_i, F)
            return tau * arclength[idx]
        torques = vmap(torque_at_point)(jnp.arange(npts1))
        mean_torque = jnp.sum(torques, axis=0) / npts1
        return mean_torque

    mean_torques = vmap(mean_torque_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas, gammadashs, centers, currents
    )
    summ = jnp.sum(jnp.linalg.norm(mean_torques, axis=-1) ** 2)
    return summ * 1e-14


class SquaredMeanTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{\tau}_i d\ell}{L})^2

    where :math:`\vec{\tau}` is the Lorentz torque, L is the total coil length,
    and :math:`\ell` is arclength along the coil. This class assumes 
    there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. 

    Args:
        allcoils (list): List of coils to optimize.
        allcoils2 (list): List of coils to optimize.
        downsample (int): Downsample factor for the objective function.
        psc_array (PSCArray): PSC coil array to use for the objective function.

    Returns:
        float: Value of the objective function.
    """

    def __init__(self, allcoils, allcoils2, downsample=1, psc_array=None):
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        self.downsample = downsample
        self.psc_array = psc_array
        args = {"static_argnums": (6,)}

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            squared_mean_torque(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
            **args
        )

        super().__init__(depends_on=(allcoils + allcoils2))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.allcoils],
            [c.curve.gamma() for c in self.allcoils2],
            [c.curve.gammadash() for c in self.allcoils],
            [c.curve.gammadash() for c in self.allcoils2],
            [c.current.get_value() for c in self.allcoils],
            [c.current.get_value() for c in self.allcoils2],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        if self.psc_array is not None:
            vjp = self.psc_array.vjp_setup(np.array(dJ_dcurrent))
        else:
            vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.allcoils)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.allcoils2)])
            + vjp
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}
