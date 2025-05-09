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
    "coil_coil_inductances_inv_pure",
    "induced_currents_pure",
    "NetFluxes",
    "B2_Energy",
    "coil_net_force",
    "coil_net_torque",
    "coil_torque",
    "LpCurveForce_deprecated",
    "MeanSquaredForce_deprecated",
    "SquaredMeanForce",
    "LpCurveForce",
    "SquaredMeanTorque",
    "LpCurveTorque",
]


def coil_force(coil, allcoils):
    """
    Compute the force on a coil.

    Args:
        coil (Coil): Coil to compute the pointwise forces on.
        allcoils (list): List of coils contributing forces on the primary coil.
        regularization (Regularization): Regularization object.

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
    selfforce = self_force(coil)
    return (selfforce + mutualforce)


def coil_net_force(coil, allcoils):
    """
    Compute the net forces on a list of coils.

    Args:
        coils (list): Coil to compute the net forces on.
        allcoils (list): List of coils contributing forces on the primary coil.
        regularization (Regularization): Regularization object.

    Returns:
        array: Array of net forces.
    """
    Fi = coil_force(coil, allcoils)
    gammadash = coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    net_force = np.sum(gammadash_norm * Fi, axis=0) / gammadash.shape[0]
    return net_force


def coil_torque(coil, allcoils):
    """
    Compute the torques on a coil.

    Args:
        coil (Coil): Coil to compute the pointwise torques on.
        allcoils (list): List of coils contributing torques on the primary coil.
        regularization (Regularization): Regularization object.

    Returns:
        array: Array of torques.
    """
    gamma = coil.curve.gamma()
    center = coil.curve.centroid()
    return np.cross(gamma - center, coil_force(coil, allcoils))


def coil_net_torque(coil, allcoils):
    """
    Compute the net torques on a list of coils.

    Args:
        coils (list): Coil to compute the net torques on.
        allcoils (list): List of coils contributing torques on the primary coil.
        regularization (Regularization): Regularization object.

    Returns:
        array: Array of net torques.
    """
    Ti = coil_torque(coil, allcoils)
    gammadash = coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    net_torque = np.sum(gammadash_norm * Ti, axis=0) / gammadash.shape[0]
    return net_torque


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


def self_force(coil):
    """
    Compute the self-force of a coil.

    Args:
        coil (Coil): Coil to compute the self-force on.

    Returns:
        array: Array of self-force.
    """
    I = coil.current.get_value()
    tangent = coil.curve.gammadash() / np.linalg.norm(coil.curve.gammadash(),
                                                      axis=1)[:, None]
    B = B_regularized(coil)
    return coil_force_pure(B, I, tangent)


def self_force_circ(coil, a):
    """
    Compute the Lorentz self-force of a coil with circular cross-section

    Args:
        coil (Coil): Coil to compute the self-force on.
        a (array): Array of coil positions.

    Returns:
        array: Array of self-force.
    """
    coil.regularization = regularization_circ(a)
    return self_force(coil)


def self_force_rect(coil, a, b):
    """
    Compute the Lorentz self-force of a coil with rectangular cross-section

    Args:
        coil (Coil): Coil to compute the self-force on.
        a (array): Array of coil positions.
        b (array): Array of coil tangent vectors.

    Returns:
        array: Array of self-force.
    """
    coil.regularization = regularization_rect(a, b)
    return self_force(coil)

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
        coil (Coil): Coil whose force is being computed.
        allcoils (list): List of coils to use for computing LpCurveForce of the primary coil. 
        regularization (Regularization): Regularization object.
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, coil, allcoils, regularization, p=2.0, threshold=0.0, downsample=1):
        coil.regularization = regularization  # for backwards compatibility
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
        coil (Coil): Coil whose force is being computed.
        allcoils (list): List of coils to use for computing MeanSquaredForce of the primary coil. 
        regularization (Regularization): Regularization object.
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, coil, allcoils, regularization, downsample=1):
        coil.regularization = regularization  # for backwards compatibility
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


def coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations):
    r"""
    Compute the full inductance matrix for a set of coils, including both mutual and 
    self-inductances. The coils are allowed to have different numbers of quadrature points,
    but for the purposes of retaining the jit speed, the coils are downsampled here 
    to have the same number of quadrature points.

    The mutual inductance between two coils is computed as:

    .. math::

        M = \frac{\mu_0}{4\pi} \iint \frac{d\vec{r}_A \cdot d\vec{r}_B}{|\vec{r}_A - \vec{r}_B|}

    and the self-inductance (with regularization) for each coil is computed as:

    .. math::

        L = \frac{\mu_0}{4\pi} \int_0^{2\pi} d\phi \int_0^{2\pi} d\tilde{\phi} 
            \frac{\vec{r}_c' \cdot \tilde{\vec{r}}_c'}{\sqrt{|\vec{r}_c - \tilde{\vec{r}}_c|^2 + \delta a b}}

    where $\delta a b$ is a regularization parameter depending on the cross-section.

    Args:
        gammas (array): Array of coil positions.
        gammadashs (array): Array of coil tangent vectors.
        downsample (int): Downsample factor for the objective function.
        regularizations (array): Array of regularizations coming from finite cross-section.

    Returns:
        array: Full inductance matrix Lij.
    """
    all_lengths = [g.shape[0] for g in gammas]
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
    N = gammas.shape[0]

    # Compute Lij, i != j
    eps = 1e-10
    r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :] + eps
    rij_norm = jnp.linalg.norm(r_ij, axis=-1)
    gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1)

    # Double sum over each of the closed curves for off-diagonal elements
    Lij = jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=-1), axis=-1) / jnp.shape(gammas)[1] ** 2

    diag_values = jnp.sum(jnp.sum(gammadash_prod / jnp.sqrt(rij_norm ** 2 + regularizations[None, :, None, None]),
                                axis=-1), axis=-1) / jnp.shape(gammas)[1] ** 2

    # Use where to select diagonal elements based on cross-section type
    diag_mask = jnp.eye(N, dtype=bool)

    # Update diagonal elements
    Lij = jnp.where(diag_mask, diag_values, Lij)
    return 1e-7 * Lij


def coil_coil_inductances_inv_pure(gammas, gammadashs, downsample, regularizations):
    """
    Pure function for computing the inverse of the coil inductance matrix.

    Args:
        gammas (array): Array of coil positions.
        gammadashs (array): Array of coil tangent vectors.
        downsample (int): Downsample factor for the objective function.
        regularizations (array): Array of regularizations coming from finite cross-section.

    Returns:
        array: Array of inverse of the coil inductance matrix.
    """
    # Lij is symmetric positive definite so has a cholesky decomposition
    C = jnp.linalg.cholesky(coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations))
    inv_C = jscp.linalg.solve_triangular(C, jnp.eye(C.shape[0]), lower=True)
    inv_L = jscp.linalg.solve_triangular(C.T, inv_C, lower=False)
    return inv_L


def induced_currents_pure(gammas, gammadashs, gammas_TF, gammadashs_TF, currents_TF, downsample, regularizations):
    """
    Pure function for computing the induced currents in a set of passive coils 
    due to a set of TF coils + the other passive coils. 

    Args:
        gammas (array): Array of passive coil positions.
        gammadashs (array): Array of passive coil tangent vectors.
        gammas_TF (array): Array of TF coil positions.
        gammadashs_TF (array): Array of TF coil tangent vectors.
        currents_TF (array): Array of TF coil current.
        downsample (int): Downsample factor for the objective function.
        regularizations (array): Array of regularizations coming from finite cross-section.

    Returns:
        array: Array of induced currents.
    """
    return -coil_coil_inductances_inv_pure(gammas, gammadashs, downsample, regularizations) @ net_fluxes_pure(gammas, gammadashs, gammas_TF, gammadashs_TF, currents_TF, downsample)


def b2energy_pure(gammas, gammadashs, currents, downsample, regularizations):
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
        downsample (int): Downsample factor for the objective function.
        regularizations (array): Array of regularizations coming from finite cross-section.

    Returns:
        float: Value of the objective function.
    """
    Ii_Ij = (currents[:, None] * currents[None, :])
    Lij = coil_coil_inductances_pure(
        gammas, gammadashs, downsample, regularizations
    )
    U = 0.5 * (jnp.sum(Ii_Ij * Lij))
    return U


class B2_Energy(Optimizable):
    r"""Optimizable class for minimizing the total vacuum energy on a coil.

    The function is

     .. math::
        J = \frac{1}{2}I_i L_{ij} I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil.

    Args:
        allcoils (Coil): List of coils contributing to the total energy.
        downsample (int): Downsample factor for the objective function.
        cross_section (str): Cross-section type.
    """

    def __init__(self, allcoils, downsample=1):
        self.allcoils = allcoils
        self.downsample = downsample
        regularizations = jnp.asarray([c.regularization for c in self.allcoils])

        args = {"static_argnums": (3,)}
        self.J_jax = jit(
            lambda gammas, gammadashs, currents, downsample:
            b2energy_pure(gammas, gammadashs, currents, downsample, regularizations),
            **args
        )

        self.dJ_dgammas = jit(
            lambda gammas, gammadashs, currents, downsample:
            grad(self.J_jax, argnums=0)(gammas, gammadashs, currents, downsample),
            **args
        )

        self.dJ_dgammadashs = jit(
            lambda gammas, gammadashs, currents, downsample:
            grad(self.J_jax, argnums=1)(gammas, gammadashs, currents, downsample),
            **args
        )

        self.dJ_dcurrents = jit(
            lambda gammas, gammadashs, currents, downsample:
            grad(self.J_jax, argnums=2)(gammas, gammadashs, currents, downsample),
            **args
        )

        super().__init__(depends_on=allcoils)

    def J(self):

        args = [
            jnp.asarray([c.curve.gamma() for c in self.allcoils]),
            jnp.asarray([c.curve.gammadash() for c in self.allcoils]),
            jnp.asarray([c.current.get_value() for c in self.allcoils]),
            self.downsample
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            jnp.asarray([c.curve.gamma() for c in self.allcoils]),
            jnp.asarray([c.curve.gammadash() for c in self.allcoils]),
            jnp.asarray([c.current.get_value() for c in self.allcoils]),
            self.downsample
        ]
        dJ_dgammas = self.dJ_dgammas(*args)
        dJ_dgammadashs = self.dJ_dgammadashs(*args)
        dJ_dcurrents = self.dJ_dcurrents(*args)
        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrents[i]])) for i, c in enumerate(self.allcoils)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgammas[i]) for i, c in enumerate(self.allcoils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadashs[i]) for i, c in enumerate(self.allcoils)])
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

    Note that the first set of coils is assumed to all have the same number of quadrature 
    points for the purposes of jit speed. Same with the second set of coils, although
    the number of points does not have to be the same between the two sets.

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

    Args:
        coil (Coil): Coil whose net flux is being computed.
        othercoils (list): List of coils to use for computing the net flux.
        downsample (int): Factor by which to downsample the coil points for computation.
    """

    def __init__(self, coil, othercoils, downsample=1):
        if not isinstance(othercoils, list):
            othercoils = [othercoils]
        self.coil = coil
        self.othercoils = [c for c in othercoils if c not in [coil]]
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


def squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents,
                            currents2, downsample):
    """
    Compute the squared mean force on coil set 1 due to coil set 1 and another coil set 2.

    The coils are allowed to have different numbers of quadrature points, but for the purposes of
    jax speed, the coils are downsampled here to have the same number of quadrature points.
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
       
        # Compute the mutual field from coil set 1 to coil set 1, masking j == i
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        # Compute the mutual field from coil set 1 to coil set 2
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
    which require a modified Jacobian calculation. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        allcoils (list): List of coils to use for computing MeanSquaredForce. 
        allcoils2 (list): List of coils that provide forces on the first set of coils but that
            we do not care about optimizing their forces. 
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, allcoils, allcoils2, downsample=1):
        if not isinstance(downsample, int):
            raise ValueError("downsample must be an integer")
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        self.downsample = downsample
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
    Computes the mixed Lp force objective by summing over all coils in the first set, 
    where each coil receives force from all coils (including itself and the second set).
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
    regularizations = jnp.array(regularizations)

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
        
        # Compute the mutual field from coil set 1 to coil set 1, masking j == i
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        # Compute the mutual field from coil set 1 to coil set 2
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
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types ofcross-sections.

    Args:
        allcoils (list): List of coils to use for computing LpCurveForce. 
        allcoils2 (list): List of coils that provide forces on the first set of coils but that
            we do not care about optimizing their forces. 
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, allcoils, allcoils2, p=2.0, threshold=0.0, downsample=1):
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        regularizations = jnp.array([c.regularization for c in allcoils])
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        quadpoints = [c.curve.quadpoints for c in allcoils]
        self.downsample = downsample
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
        gammadashdashs (array): Array of second derivatives of coil positions.
        quadpoints (array): Array of quadrature points.
        currents (array): Array of coil currents.
        currents2 (array): Array of coil currents.
        regularizations (array): Array of coil regularizations.
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
    regularizations = jnp.array(regularizations)

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
        
        # Compute the mutual field from coil set 1 to coil set 1, masking j == i
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        # Compute the mutual field from coil set 1 to coil set 2
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
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        allcoils (list): List of coils to use for computing LpCurveTorque. 
        allcoils2 (list): List of coils that provide torques on the first set of coils but that
            we do not care about optimizing their torques. 
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): Downsample factor for the objective function.
    """

    def __init__(self, allcoils, allcoils2, p=2.0, threshold=0.0, downsample=1):
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        regularizations = jnp.array([c.regularization for c in allcoils])
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        quadpoints = [c.curve.quadpoints for c in allcoils]
        self.downsample = downsample
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
    """
    Compute the squared mean torque in coil set 1 due to themselves and coil set 2.

    Args:
        gammas (array): Array of coil positions in coil set 1.
        gammas2 (array): Array of coil positions in coil set 2.
        gammadashs (array): Array of coil tangent vectors in coil set 1.
        gammadashs2 (array): Array of coil tangent vectors in coil set 2.
        currents (array): Array of coil currents in coil set 1.
        currents2 (array): Array of coil currents in coil set 2.
        downsample (int): Downsample factor for the curve quadrature points.

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
        
        # Compute the mutual field from coil set 1 to coil set 1, masking j == i
        B_mutual1 = jnp.sum(vmap(biot_savart_from_j)(jnp.arange(n1)), axis=0)
        # Compute the mutual field from coil set 1 to coil set 2
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
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        allcoils (list): List of coils to use for computing SquaredMeanTorque. 
        allcoils2 (list): List of coils that provide torques on the first set of coils but that
            we do not care about optimizing their torques. 
        downsample (int): Downsample factor for the objective function.

    Returns:
        float: Value of the objective function.
    """

    def __init__(self, allcoils, allcoils2, downsample=1):
        if not isinstance(allcoils, list):
            allcoils = [allcoils]
        if not isinstance(allcoils2, list):
            allcoils2 = [allcoils2]
        self.allcoils = allcoils
        self.allcoils2 = [c for c in allcoils2 if c not in allcoils]
        self.downsample = downsample
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
