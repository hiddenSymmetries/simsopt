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


def coil_force(coil, allcoils, regularization, nturns=1):
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
    mutual_field._children = set()
    for c in mutual_coils:
        c._children = set()
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


# @jit
def lp_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample):
    r"""Pure function for minimizing the Lorentz force on a coil.

    The function is

     .. math::
        J = \frac{1}{p}\left(\int \text{max}(|\vec{F}| - F_0, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    and :math:`\ell` is arclength along the coil.
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


class LpCurveForce(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{pL}\left(\int \text{max}(|d\vec{F}/d\ell| - dF_0/d\ell, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
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
            lp_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample),
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
        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()
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
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
            + B_vjp
        )

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


# @jit
def mean_squared_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample):
    r"""Pure function for minimizing the Lorentz force on a coil.

    The function is

    .. math:
        J = \frac{\int |d\vec{F}/d\ell|^2 d\ell}{L}

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil.
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


class MeanSquaredForce(Optimizable):
    r"""Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math:
        J = \frac{\int |\vec{F}|^2 d\ell}{L}

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil.
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
            mean_squared_force_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample),
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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return J

    @derivative_dec
    def dJ(self):
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
        dJ_dB = self.dJ_dB_mutual(*args)
        dB_dX = self.biotsavart.dB_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dB, dB_dX)

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(self.dJ_dgammadashdash(*args))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))
            + self.biotsavart.B_vjp(dJ_dB)
        )
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def mixed_squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample):
    r"""
    Computes the objective function

    .. math:
        J = (\frac{\int \vec{F}_i d\ell}{L})^2

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil. This class assumes 
    there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. 
    """
    # Downsample if need be
    gammas = gammas[:, ::downsample, :]
    gammadashs = gammadashs[:, ::downsample, :]
    gammas2 = gammas2[:, ::downsample, :]
    gammadashs2 = gammadashs2[:, ::downsample, :]

    eps = 1e-10  # small number to avoid blow up in the denominator when i = j
    r_ij = gammas[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
    cross1 = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
    cross2 = jnp.cross(gammadashs[:, None, :, None, :], cross1)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents[:, None] * currents[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    F = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross2 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
    net_forces = -jnp.sum(F, axis=1) / jnp.shape(gammas)[1] ** 2

    # repeat with gamma, gamma2
    r_ij = gammas[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    cross1 = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
    cross2 = jnp.cross(gammadashs[:, None, :, None, :], cross1)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents[:, None] * currents2[None, :]
    F = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross2 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
    net_forces += -jnp.sum(F, axis=1) / jnp.shape(gammas2)[1] / jnp.shape(gammas2)[1]
    summ = jnp.sum(jnp.linalg.norm(net_forces, axis=-1) ** 2)

    # repeat with gamma2, gamma
    r_ij = gammas2[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
    cross1 = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
    cross2 = jnp.cross(gammadashs2[:, None, :, None, :], cross1)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents2[:, None] * currents[None, :]
    F = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross2 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
    net_forces = -jnp.sum(F, axis=1) / jnp.shape(gammas)[1] / jnp.shape(gammas2)[1]

    # repeat with gamma2, gamma2
    r_ij = gammas2[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    cross1 = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
    cross2 = jnp.cross(gammadashs2[:, None, :, None, :], cross1)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents2[:, None] * currents2[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    F = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross2 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
    net_forces += -jnp.sum(F, axis=1) / jnp.shape(gammas2)[1] ** 2
    summ += jnp.sum(jnp.linalg.norm(net_forces, axis=-1) ** 2)
    return summ * 1e-14


class MixedSquaredMeanForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell}{L})^2

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil. This class assumes 
    there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. 
    """

    def __init__(self, allcoils, allcoils2, downsample=1):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2
        self.downsample = downsample
        args = {"static_argnums": (6,)}

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            mixed_squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
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
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
            self.downsample
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
            self.downsample
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

# @jit


def mixed_lp_force_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2,
                        quadpoints, quadpoints2,
                        currents, currents2, regularizations, regularizations2, p, threshold,
                        downsample=1,
                        ):
    r"""
    The objective function is

    .. math::
        J = \frac{1}{pL}\left(\int \text{max}(|d\vec{F}/d\ell| - dF_0/d\ell, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. 
    """
    quadpoints = quadpoints[::downsample]
    gammas = gammas[:, ::downsample, :]
    gammadashs = gammadashs[:, ::downsample, :]
    gammadashdashs = gammadashdashs[:, ::downsample, :]

    quadpoints2 = quadpoints2[::downsample]
    gammas2 = gammas2[:, ::downsample, :]
    gammadashs2 = gammadashs2[:, ::downsample, :]
    gammadashdashs2 = gammadashdashs2[:, ::downsample, :]

    B_self = jnp.array([B_regularized_pure(gammas[i], gammadashs[i], gammadashdashs[i], quadpoints,
                                           currents[i], regularizations[i]) for i in range(jnp.shape(gammas)[0])])
    B_self2 = jnp.array([B_regularized_pure(gammas2[i], gammadashs2[i], gammadashdashs2[i], quadpoints2,
                                            currents2[i], regularizations2[i]) for i in range(jnp.shape(gammas2)[0])])
    gammadash_norms = jnp.linalg.norm(gammadashs, axis=-1)[:, :, None]
    tangents = gammadashs / gammadash_norms
    gammadash_norms2 = jnp.linalg.norm(gammadashs2, axis=-1)[:, :, None]
    tangents2 = gammadashs2 / gammadash_norms2
    eps = 1e-10  # small number to avoid blow up in the denominator when i = j
    r_ij = gammas[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices

    ### Note that need to do dl1 x dl2 x r12 here instead of just (dl1 * dl2)r12
    # because these are not equivalent expressions if we are squaring the pointwise forces
    # before integration over coil i!
    # cross_prod = jnp.cross(tangents[:, None, :, None, :], jnp.cross(gammadashs[None, :, None, :, :], r_ij))
    cross_prod = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents[:, None] * currents[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    F = jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas)[1]

    # repeat with gamma, gamma2
    r_ij = gammas[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    cross_prod = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents[:, None] * currents2[None, :]
    F += jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas2)[1]
    force_norm = jnp.linalg.norm(jnp.cross(tangents, F * 1e-7 + currents[:, None, None] * B_self), axis=-1)
    summ = jnp.sum(jnp.maximum(force_norm[:, :, None] - threshold, 0) ** p * gammadash_norms) / jnp.shape(gammas)[1]

    # repeat with gamma2, gamma
    r_ij = gammas2[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
    cross_prod = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents2[:, None] * currents[None, :]
    F = jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas)[1]

    # repeat with gamma2, gamma2
    r_ij = gammas2[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
    cross_prod = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
    rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
    Ii_Ij = currents2[:, None] * currents2[None, :]
    Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
    F += jnp.sum(Ii_Ij[:, :, None, None] * jnp.sum(cross_prod / rij_norm3[:, :, :, :, None], axis=3), axis=1) / jnp.shape(gammas2)[1]
    force_norm2 = jnp.linalg.norm(jnp.cross(tangents2, F * 1e-7 + currents2[:, None, None] * B_self2), axis=-1)
    summ += jnp.sum(jnp.maximum(force_norm2[:, :, None] - threshold, 0) ** p * gammadash_norms2) / jnp.shape(gammas2)[1]
    return summ * (1 / p)


class MixedLpCurveForce(Optimizable):
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
    """

    def __init__(self, allcoils, allcoils2, regularizations, regularizations2, p=2.0, threshold=0.0, downsample=1):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2
        quadpoints = self.allcoils[0].curve.quadpoints
        quadpoints2 = self.allcoils2[0].curve.quadpoints
        self.downsample = downsample

        args = {"static_argnums": (8,)}
        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            mixed_lp_force_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, quadpoints, quadpoints2,
                                currents, currents2, regularizations, regularizations2, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadashdash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=6)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )
        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=7)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
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
            self.downsample
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
            self.downsample
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
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash2[i]) for i, c in enumerate(self.allcoils2)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.allcoils2)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class MixedLpCurveTorque(Optimizable):
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
    """

    # @jit
    def mixed_lp_torque_pure(self, gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2,
                             quadpoints, quadpoints2,
                             currents, currents2, regularizations, regularizations2, p, threshold, downsample):
        r"""
        """
        # Downsample everything if necessary
        quadpoints = quadpoints[::downsample]
        gammas = gammas[:, ::downsample, :]
        gammadashs = gammadashs[:, ::downsample, :]
        gammadashdashs = gammadashdashs[:, ::downsample, :]
        quadpoints2 = quadpoints2[::downsample]
        gammas2 = gammas2[:, ::downsample, :]
        gammadashs2 = gammadashs2[:, ::downsample, :]
        gammadashdashs2 = gammadashdashs2[:, ::downsample, :]

        # Compute the self torques for each list of coils
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
        summ = jnp.sum(jnp.maximum(torque_norm[:, :, None] - threshold, 0) ** p * gammadash_norms) / jnp.shape(gammas)[1]

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
        summ += jnp.sum(jnp.maximum(torque_norm2[:, :, None] - threshold, 0) ** p * gammadash_norms2) / jnp.shape(gammas2)[1]
        return summ * (1 / p)

    def __init__(self, allcoils, allcoils2, regularizations, regularizations2, p=2.0, threshold=0.0, downsample=1):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2
        quadpoints = self.allcoils[0].curve.quadpoints
        quadpoints2 = self.allcoils2[0].curve.quadpoints
        self.downsample = downsample
        args = {"static_argnums": (8,)}

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            self.mixed_lp_torque_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, quadpoints, quadpoints2,
                                      currents, currents2, regularizations, regularizations2, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=0)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgamma2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=1)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=2)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=3)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadashdash = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=4)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dgammadashdash2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=5)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=6)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
        )

        self.dJ_dcurrent2 = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample:
            grad(self.J_jax, argnums=7)(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs, gammadashdashs2, currents, currents2, downsample),
            **args
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
            self.downsample
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
            self.downsample
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
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash2[i]) for i, c in enumerate(self.allcoils2)])
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
    along the coil. This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. 
    """

    # @jit
    def mixed_squared_mean_torque(self, gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample):
        r"""
        """
        # Downsample if needed
        gammas = gammas[:, ::downsample, :]
        gammadashs = gammadashs[:, ::downsample, :]
        gammas2 = gammas2[:, ::downsample, :]
        gammadashs2 = gammadashs2[:, ::downsample, :]

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
        net_torques = -jnp.sum(T, axis=1) / jnp.shape(gammas)[1] ** 2

        # repeat with gamma, gamma2
        r_ij = gammas[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
        cross1 = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs[:, None, :, None, :], cross1)
        cross3 = jnp.cross((gammas - centers)[:, None, :, None, :], cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents[:, None] * currents2[None, :]
        T = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques += -jnp.sum(T, axis=1) / jnp.shape(gammas2)[1] / jnp.shape(gammas2)[1]
        summ = jnp.sum(jnp.linalg.norm(net_torques, axis=-1) ** 2)

        # repeat with gamma2, gamma
        r_ij = gammas2[:, None, :, None, :] - gammas[None, :, None, :, :]  # Note, do not use the i = j indices
        cross1 = jnp.cross(gammadashs[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs2[:, None, :, None, :], cross1)
        cross3 = jnp.cross((gammas2 - centers2)[:, None, :, None, :], cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents2[:, None] * currents[None, :]
        T = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques = -jnp.sum(T, axis=1) / jnp.shape(gammas)[1] / jnp.shape(gammas2)[1]

        # repeat with gamma2, gamma2
        r_ij = gammas2[:, None, :, None, :] - gammas2[None, :, None, :, :]  # Note, do not use the i = j indices
        cross1 = jnp.cross(gammadashs2[None, :, None, :, :], r_ij)
        cross2 = jnp.cross(gammadashs2[:, None, :, None, :], cross1)
        cross3 = jnp.cross((gammas2 - centers2)[:, None, :, None, :], cross2)
        rij_norm3 = jnp.linalg.norm(r_ij + eps, axis=-1) ** 3
        Ii_Ij = currents2[:, None] * currents2[None, :]
        Ii_Ij = Ii_Ij.at[:, :].add(-jnp.diag(jnp.diag(Ii_Ij)))
        T = Ii_Ij[:, :, None] * jnp.sum(jnp.sum(cross3 / rij_norm3[:, :, :, :, None], axis=3), axis=2)
        net_torques += -jnp.sum(T, axis=1) / jnp.shape(gammas2)[1] ** 2
        summ += jnp.sum(jnp.linalg.norm(net_torques, axis=-1) ** 2)
        return summ * 1e-14

    def __init__(self, allcoils, allcoils2, downsample=1):
        self.allcoils = allcoils
        self.allcoils2 = allcoils2
        self.downsample = downsample
        args = {"static_argnums": (6,)}

        self.J_jax = jit(
            lambda gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample:
            self.mixed_squared_mean_torque(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample),
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
            jnp.array([c.curve.gamma() for c in self.allcoils]),
            jnp.array([c.curve.gamma() for c in self.allcoils2]),
            jnp.array([c.curve.gammadash() for c in self.allcoils]),
            jnp.array([c.curve.gammadash() for c in self.allcoils2]),
            jnp.array([c.current.get_value() for c in self.allcoils]),
            jnp.array([c.current.get_value() for c in self.allcoils2]),
            self.downsample
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
            self.downsample
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

# @jit


def squared_mean_force_pure(current, gammadash, B_mutual, downsample):
    r"""
    """
    gammadash = gammadash[::downsample, :]
    return (current * jnp.linalg.norm(jnp.sum(jnp.cross(gammadash, B_mutual), axis=0) / gammadash.shape[0])) ** 2


class SquaredMeanForce(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (\frac{\int \vec{F}_i d\ell}{L})^2

    where :math:`\vec{F}` is the Lorentz force, L is the total coil length,
    and :math:`\ell` is arclength along the coil.
    """

    def __init__(self, coil, allcoils, downsample=1):
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not self.coil]
        self.downsample = downsample
        self.biotsavart = BiotSavart(self.othercoils)

        args = {"static_argnums": (3,)}
        self.J_jax = jit(
            lambda current, gammadash, B_mutual, downsample:
            squared_mean_force_pure(current, gammadash, B_mutual, downsample),
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
        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class SquaredMeanTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = ((1/L) * \frac{\int \vec{\tau}_i d\ell)^2

    where :math:`\vec{\tau}` is the Lorentz torque, L is the total coil length,
    and :math:`\ell` is arclengthalong the coil.
    """

    # @jit
    def squared_mean_torque_pure(self, current, gamma, gammadash, B_mutual, downsample):
        r"""
        """
        gamma = gamma[::downsample, :]
        gammadash = gammadash[::downsample, :]
        return (current * jnp.linalg.norm(jnp.sum(jnp.cross(gamma - self.coil.curve.center(gamma, gammadash), jnp.cross(gammadash, B_mutual)), axis=0) / gamma.shape[0])) ** 2

    def __init__(self, coil, allcoils, downsample=1):
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        self.downsample = downsample

        args = {"static_argnums": (4,)}
        self.J_jax = jit(
            lambda current, gamma, gammadash, B_mutual, downsample:
            self.squared_mean_torque_pure(current, gamma, gammadash, B_mutual, downsample),
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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()
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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class MeanSquaredTorque(Optimizable):
    r"""Optimizable class to minimize the net Lorentz force on a coil.

    The objective function is

    .. math:
        J = (1/L) \frac{\int (\vec{\tau}_i)^2 d\ell

    where :math:`\vec{\tau}` is the Lorentz torque, L is the total coil length,
    and :math:`\ell` is arclength along the coil.
    """

    # @jit
    def mean_squared_torque_pure(self, gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample):
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
            self.mean_squared_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, downsample),
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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


class LpCurveTorque(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \text{max}(|d\vec{\tau}/d\ell| - d\tau_0/d\ell, 0)^p d\ell\right)

    where :math:`\vec{F}` is the Lorentz force, :math:`F_0` is a threshold force,  
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
    """

    # @jit
    def lp_torque_pure(self, gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample):
        r"""Pure function for minimizing the Lorentz force on a coil.

        The function is

        .. math::
            J = \frac{1}{pL}\left(\int \text{max}(|\vec{T}| - T_0, 0)^p d\ell\right)

        where :math:`\vec{T}` is the Lorentz torque, :math:`T_0` is a threshold torque,  
        L is the total length of the coil,
        and :math:`\ell` is arclength along the coil.
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
            self.lp_torque_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization, B_mutual, p, threshold, downsample),
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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

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

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.biotsavart._children = set()
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def coil_coil_inductances_pure(self, gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \frac{d\ell_i\cdot d\ell_j}{|r_i - r_j} \right)

    where :math:`\vec{r}_i` is the position vector of a point on the ith coil,
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
    """
    # Downsample if desired
    quadpoints = quadpoints[::downsample]
    gamma = gamma[::downsample, :]
    gammadash = gammadash[::downsample, :]
    quadpoints2 = quadpoints2[::downsample]
    gammas2 = gammas2[:, ::downsample, :]
    gammadashs2 = gammadashs2[:, ::downsample, :]

    eps = 1e-10  # small number to avoid blow up in the denominator when i = j

    # gamma and gammadash are shape (ncoils, nquadpoints, 3)
    r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :]  # Note, do not use the i = j indices
    gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1)
    rij_norm = jnp.linalg.norm(r_ij + eps, axis=-1)

    # Double sum over each of the closed curves
    Lij = jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=3), axis=2
                    ) / jnp.shape(gammas)[1] ** 2
    Lij = jnp.subtract(Lij, jnp.diagonal(Lij))
    # Diagonal elements need to be fixed below

    # Now fix the i = j case using the Eq 11 regularization from Hurwitz/Landreman 2023
    # and also used in Guinchard/Hudson/Paul 2024
    k = (4 * b) / (3 * a) * jnp.arctan2(a, b) + (4 * a) / (3 * b) * jnp.arctan2(b, a) \
        + (b ** 2) / (6 * a ** 2) * jnp.log(b / a) + (a ** 2) / (6 * b ** 2) * jnp.log(a / b) \
        - (a ** 4 - 6 * a ** 2 * b ** 2 + b ** 4) / (6 * a ** 2 * b ** 2) * jnp.log(a / b + b / a)
    delta = jnp.exp(-25.0 / 6.0 + k)
    # print(k, delta, a, b)
    # print(jnp.shape(gammadash_prod), jnp.shape(rij_norm), jnp.shape(r_ij))
    # print('rij = ', r_ij)

    # Need below line with eps != 0, otherwise some of the tangent vectors become NaN
    rij_norm = jnp.sqrt(jnp.linalg.norm(r_ij + eps, axis=-1) ** 2 + delta * a * b)
    # print('rij_norm = ', rij_norm)
    # print('sqrt_term = ', jnp.linalg.norm(r_ij, axis=-1) ** 2, jnp.shape(jnp.linalg.norm(r_ij, axis=-1) ** 2))
    Lii = jnp.diagonal(jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=3), axis=2)
                        ) / jnp.shape(gammas)[1] ** 2
    # print('Lii = ', Lii)
    for i in range(jnp.shape(Lij)[0]):
        Lij = Lij.at[i, i].add(Lii[i])
    # print(Lii)

    # Equation 22 in Hurwitz/Landreman 2023 is even better and implemented below
    # gd_norm = jnp.linalg.norm(gammadashs, axis=-1)
    # quadpoints = self.get_quadpoints()
    # quad_diff = quadpoints[None, :, None, :] - quadpoints[:, None, :, None]  # Note quadpoints are in [0, 1]
    # integrand2 = gd_norm[:, None, :, None] ** 2 / jnp.sqrt(
    #     (2 - 2 * jnp.cos(2 * np.pi * quad_diff)) * gd_norm[:, None, :, None] ** 2 + delta * a * b)
    # print(jnp.shape(quad_diff), jnp.shape(integrand2), jnp.shape(gammas))
    # L1 = jnp.sum(gd_norm * jnp.log(64.0 / (delta * a * b) * gd_norm ** 2), axis=-1) / jnp.shape(gammas)[1]
    # gammadash_prod = np.diagonal(gammadash_prod, axis1=0, axis2=0)
    # rij_norm = np.diagonal(rij_norm, axis1=0, axis2=0)
    # L2 = jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=3), axis=2) / jnp.shape(gammas)[1] ** 2
    # L3 = jnp.sum(jnp.sum(integrand2, axis=3), axis=2) / jnp.shape(gammas)[1] ** 2
    # print(L1, L2, L3, jnp.shape(L2))
    # Lii = L1 + jnp.diagonal(L2 - L3)
    # print(Lii)

    # zero out the original diagonal elements and replace with Lii
    # for i in range(jnp.shape(Lij)[0]):
    #     Lij = Lij.at[i, i].add(Lii[i])
    # Lij = jnp.add(Lij, Lii)
    return Lij * 1e-7


class CoilInductances(Optimizable):
    r"""  Optimizable class to minimize the Lorentz force on a coil.

    The objective function is

    .. math::
        J = \frac{1}{p}\left(\int \frac{d\ell_i\cdot d\ell_j}{|r_i - r_j} \right)

    where :math:`\vec{r}_i` is the position vector of a point on the ith coil,
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.
    """

    def __init__(self, coil, allcoils, regularization, p=2.0, threshold=0.0, downsample=1):
        self.coil = coil
        self.othercoils = [c for c in allcoils if c is not coil]
        self.biotsavart = BiotSavart(self.othercoils)
        quadpoints = self.coil.curve.quadpoints
        quadpoints2 = jnp.array([c.curve.quadpoints for c in self.othercoils])
        self.downsample = downsample

        args = {"static_argnums": (6,)}
        self.J_jax = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, current2, downsample:
            self.coil_coil_inductances_pure(gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, current2, downsample:
            grad(self.J_jax, argnums=0)(gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, current2, downsample:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample),
            **args
        )

        self.dJ_dgammas2 = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, current2, downsample:
            grad(self.J_jax, argnums=2)(gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample),
            **args
        )

        self.dJ_dgammadashs2 = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, current2, downsample:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample),
            **args
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, current2, downsample:
            grad(self.J_jax, argnums=4)(gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample),
            **args
        )

        self.dJ_dcurrents2 = jit(
            lambda gamma, gammadash, gammas2, gammadashs2, current, current2, downsample:
            grad(self.J_jax, argnums=5)(gamma, gammadash, gammas2, gammadashs2, current, current2, quadpoints, quadpoints2, downsample),
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
        J = self.J_jax(*args)

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return J

    @derivative_dec
    def dJ(self):

        # Remaining terms can be downsampled before calculation
        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            jnp.array([c.curve.gamma() for c in self.othercoils]),
            jnp.array([c.curve.gammadash() for c in self.othercoils]),
            self.coil.current.get_value(),
            jnp.array([c.current.get_value() for c in self.othercoils]),
            self.downsample
        ]
        
        dJ = (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args))
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args2))
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgammas2[i]) for i, c in enumerate(self.othercoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadashs2[i]) for i, c in enumerate(self.othercoils)])
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args2)]))
            + sum([c.current.vjp(dJ_dcurrents2[i]) for i, c in enumerate(self.othercoils)])
        )

        #### ABSOLUTELY ESSENTIAL LINES BELOW
        # Otherwise optimizable references multiply
        # like crazy as number of coils increases
        self.coil._children = set()
        self.coil.curve._children = set()
        self.coil.current._children = set()
        for c in self.othercoils:
            c._children = set()
            c.curve._children = set()
            c.current._children = set()

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}

def tve_pure(self, curve_dofs, currents, a, b):
    r"""Pure function for minimizing the total vacuum energy on a coil.

    The function is

     .. math::
        J = \frac{1}{2}I_iL_{ij}I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil.
    """
    Ii_Ij = (currents[None, :] * currents[:, None])
    Lij = self.coil_coil_inductances_pure(curve_dofs, a=a, b=b)
    U = 0.5 * jnp.sum(Ii_Ij * Lij)
    return U

class TVE(Optimizable):
    r"""Optimizable class for minimizing the total vacuum energy on a coil.

    The function is

     .. math::
        J = \frac{1}{2}I_iL_{ij}I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil.
    """

    def __init__(self, coil, allcoils, regularization, p=1.0, threshold=0.0, downsample=1):
        self.coil = coil
        self.allcoils = allcoils
        quadpoints = self.coil.curve.quadpoints
        self.downsample = downsample

        args = {"static_argnums": (3,)}
        self.J_jax = jit(
            lambda gamma, gammadash, current, downsample:
            tve_pure(gamma, gammadash, quadpoints, current, regularization, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma = jit(
            lambda gamma, gammadash, current, downsample:
            grad(self.J_jax, argnums=0)(gamma, gammadash, quadpoints, current, regularization, p, threshold, downsample),
            **args
        )

        self.dJ_dgammadash = jit(
            lambda gamma, gammadash, current, downsample:
            grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        self.dJ_dcurrent = jit(
            lambda gamma, gammadash, gammadashdash, current, B_mutual:
            grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, current, B_mutual)
        )

        super().__init__(depends_on=allcoils)

    def J(self):

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.current.get_value(),
            self.downsample
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            self.coil.curve.gamma(),
            self.coil.curve.gammadash(),
            self.coil.current.get_value(),
            self.downsample
        ]

        return (
            self.coil.curve.dgamma_by_dcoeff_vjp(self.dJ_dgamma(*args) + dJ_dX)
            + self.coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + self.coil.current.vjp(jnp.asarray([self.dJ_dcurrent(*args)]))
        )

    return_fn_map = {'J': J, 'dJ': dJ}
