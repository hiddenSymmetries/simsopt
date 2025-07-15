"""Implements the force on a coil in its own magnetic field and the field of other coils."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscp
from jax import grad, vmap
from jax.lax import cond
from .biotsavart import BiotSavart
from .coil import RegularizedCoil
from .selffield import B_regularized_pure, B_regularized
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

__all__ = [
    "coil_force",
    "self_force",
    "_coil_coil_inductances_pure",
    "_coil_coil_inductances_inv_pure",
    "_induced_currents_pure",
    "NetFluxes",
    "B2Energy",
    "coil_net_force",
    "coil_net_torque",
    "coil_torque",
    "SquaredMeanForce",
    "LpCurveForce",
    "SquaredMeanTorque",
    "LpCurveTorque",
]


def coil_force(target_coil, source_coils):
    """
    Compute the force per unit length on a coil from m other coils, in Newtons/meter. Note that BiotSavart objects
    are created below, which can lead to growth of the number of optimizable graph dependencies.

    Args:
        target_coil (Coil): Coil to compute the pointwise forces on.
        source_coils (list of Coil, shape (m,)): List of coils contributing forces on the primary coil.

    Returns:
        array: Array of forces per unit length.
    """
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
    """
    Compute the net forces on one coil from m other coils, in Newtons. This is
    the integrated pointwise force per unit length on a coil curve.

    Args:
        target_coil (Coil): Coil to compute the net forces on.
        source_coils (list of Coil, shape (m,)): List of coils contributing forces on the primary coil.

    Returns:
        array: Array of net forces.
    """
    Fi = coil_force(target_coil, source_coils)
    gammadash = target_coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    net_force = np.sum(gammadash_norm * Fi, axis=0) / gammadash.shape[0]
    return net_force


def coil_torque(target_coil, source_coils):
    """
    Compute the torques per unit length on a coil from m other coils in Newtons 
    (note that the force is per unit length, so the force has units of Newtons/meter 
    and the torques per unit length have units of Newtons).

    Args:
        target_coil (Coil): Coil to compute the pointwise torques on.
        source_coils (list of Coil, shape (m,)): List of coils contributing torques on the primary coil.

    Returns:
        array: Array of torques.
    """
    gamma = target_coil.curve.gamma()
    center = target_coil.curve.centroid()
    return np.cross(gamma - center, coil_force(target_coil, source_coils))


def coil_net_torque(target_coil, source_coils):
    """
    Compute the net torques on a coil from m other coils, in Newton-meters. This is
    the integrated pointwise torque per unit length on a coil curve.

    Args:
        target_coil (Coil): Coil to compute the net torques on.
        source_coils (list of Coil, shape (m,)): List of coils contributing torques on the primary coil.

    Returns:
        array: Array of net torques.
    """
    Ti = coil_torque(target_coil, source_coils)
    gammadash = target_coil.curve.gammadash()
    gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
    net_torque = np.sum(gammadash_norm * Ti, axis=0) / gammadash.shape[0]
    return net_torque


def _coil_force_pure(B, I, t):
    """
    Compute the pointwise Lorentz force per unit length on a coil with n quadrature points, in Newtons/meter. 

    Args:
        B (array, shape (n,3)): Array of magnetic field.
        I (float): Coil current.
        t (array, shape (n,3)): Array of coil tangent vectors.

    Returns:
        array (shape (n,3)): Array of force per unit length.
    """
    return jnp.cross(I * t, B)


def self_force(target_coil):
    """
    Compute the self-force per unit length of a coil, in Newtons/meter.

    Args:
        target_coil (Coil): Coil to compute the self-force per unit length on.

    Returns:
        array (shape (n,3)): Array of self-force per unit length.
    """
    I = target_coil.current.get_value()
    tangent = target_coil.curve.gammadash() / np.linalg.norm(target_coil.curve.gammadash(),
                                                      axis=1)[:, None]
    B = B_regularized(target_coil)
    return _coil_force_pure(B, I, tangent)


def _coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations):
    r"""
    Compute the full inductance matrix for a set of coils, including both mutual and 
    self-inductances. The coils are allowed to have different numbers of quadrature points,
    but for the purposes of retaining the jit speed, the coils are downsampled here 
    to have the same number of quadrature points, denoted n.

    The mutual inductance between two coils is computed as:

    .. math::

        M = \frac{\mu_0}{4\pi} \iint \frac{d\vec{r}_A \cdot d\vec{r}_B}{|\vec{r}_A - \vec{r}_B|}

    and the self-inductance (with regularization) for each coil is computed as:

    .. math::

        L = \frac{\mu_0}{4\pi} \int_0^{2\pi} d\phi \int_0^{2\pi} d\tilde{\phi} 
            \frac{\vec{r}_c' \cdot \tilde{\vec{r}}_c'}{\sqrt{|\vec{r}_c - \tilde{\vec{r}}_c|^2 + \delta a b}}

    where $\delta a b$ is a regularization parameter depending on the cross-section. The units
    of the inductance matrices are henries.

    Args:
        gammas (array, shape (m,n,3)): 
            Array of coil positions for all m coils (which are each downsampled to shape (n,3)).
        gammadashs (array, shape (m,n,3)): 
            Array of coil tangent vectors for all m coils (which are each downsampled to shape (n,3)).
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
        regularizations (array, shape (m,)): 
            Array of regularizations coming from finite cross-section for all m coils.

    Returns:
        array (shape (m,m)): Full inductance matrix Lij.
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


def _coil_coil_inductances_inv_pure(gammas, gammadashs, downsample, regularizations):
    """
    Pure function for computing the inverse of the coil inductance matrix L. This matrix
    is symmetric positive definite by definition.
    
    Performs a Cholesky decomposition of the coil inductance matrix L and then solves for the
    inverse. Note that inverse of a (nonsingular)lower triangular matrix C is upper triangular 
    and vice versa.

    .. math::

        L = (C C^T)

    where :math:`C` is a lower triangular matrix from the Cholesky decomposition of :math:`L`. 
    Then, we solve two triangular systems of equations:

    .. math::

        C^{-1}C = I

        L^{-1} = (C C^T)^{-1} = (C^T)^{-1} C^{-1}

    so that we can solve for :math:`L^{-1}` by multiplying both sides by :math:`C^T` and 
    solving it as an upper triangular system,

    .. math::

        C^TL^{-1} = C^{-1}

    Args:
        gammas (array, shape (m,n,3)): 
            Array of coil positions for all m coils (which are each downsampled to shape (n,3)).
        gammadashs (array, shape (m,n,3)): 
            Array of coil tangent vectors for all m coils (which are each downsampled to shape (n,3)).
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
        regularizations (array, shape (m,)): 
            Array of regularizations coming from finite cross-section for all m coils.

    Returns:
        array (shape (m,m)): Array of inverse of the coil inductance matrix.
    """
    # Lij is symmetric positive definite so has a cholesky decomposition
    C = jnp.linalg.cholesky(_coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations))
    inv_C = jscp.linalg.solve_triangular(C, jnp.eye(C.shape[0]), lower=True)
    inv_L = jscp.linalg.solve_triangular(C.T, inv_C, lower=False)
    return inv_L


def _induced_currents_pure(gammas, gammadashs, gammas_TF, gammadashs_TF, currents_TF, downsample, regularizations):
    """
    Pure function for computing the induced currents in a set of m passive coils with n quadrature points
    due to a set of m' TF coils with n' quadrature points (and themselves). 

    Args:
        gammas (array, shape (m,n,3)): 
            Array of passive coil positions for all m coils.
        gammadashs (array, shape (m,n,3)): 
            Array of passive coil tangent vectors for all m coils.
        gammas_TF (array, shape (m',n',3)): 
            Array of TF coil positions for all m' coils.
        gammadashs_TF (array, shape (m',n',3)): 
            Array of TF coil tangent vectors for all m' coils.
        currents_TF (array, shape (m',)): 
            Array of TF coil current.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
        regularizations (array, shape (m,)): 
            Array of regularizations coming from finite cross-section for all m coils.

    Returns:
        array (shape (m,)): Array of induced currents.
    """
    return -_coil_coil_inductances_inv_pure(gammas, gammadashs, downsample, regularizations) @ net_fluxes_pure(gammas, gammadashs, gammas_TF, gammadashs_TF, currents_TF, downsample)


def b2energy_pure(gammas, gammadashs, currents, downsample, regularizations):
    r"""
    Pure function for minimizing the total vacuum magnetic field energy from a set of m coils
    which may have different numbers of quadrature points (but are downsampled to have 
    the same number, denoted n, of quadrature points).

    The function is

     .. math::
        J = \frac{1}{2}\sum_{i,j}I_iL_{ij}I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil. The units of the objective function are Joules.

    Args:
        gammas (array, shape (m,n,3)): 
            Array of coil positions for all m coils.
        gammadashs (array, shape (m,n,3)): 
            Array of coil tangent vectors for all m coils.
        currents (array, shape (m,)): 
            Array of coil current for all m coils.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
        regularizations (array, shape (m,)): 
            Array of regularizations coming from finite cross-section for all m coils.

    Returns:
        float: Value of the objective function.
    """
    Ii_Ij = (currents[:, None] * currents[None, :])
    Lij = _coil_coil_inductances_pure(
        gammas, gammadashs, downsample, regularizations
    )
    U = 0.5 * (jnp.sum(Ii_Ij * Lij))
    return U


class B2Energy(Optimizable):
    r"""
    Optimizable class for minimizing the total vacuum magnetic field energy from a set of m coils.

    The function is

     .. math::
        J = \frac{1}{2}\sum_{i,j}I_i L_{ij} I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil. The units of the objective function are Joules.

    Args:
        coils_to_target (list of Coil, shape (m,)): 
            List of coils contributing to the total energy.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
    """

    def __init__(self, coils_to_target, downsample=1):
        self.coils_to_target = coils_to_target
        self.downsample = downsample
        if not isinstance(self.coils_to_target[0], RegularizedCoil):
            raise ValueError("B2Energy can only be used with RegularizedCoil objects")
        regularizations = jnp.asarray([c.regularization for c in self.coils_to_target])

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

        super().__init__(depends_on=coils_to_target)

    def J(self):

        args = [
            jnp.asarray([c.curve.gamma() for c in self.coils_to_target]),
            jnp.asarray([c.curve.gammadash() for c in self.coils_to_target]),
            jnp.asarray([c.current.get_value() for c in self.coils_to_target]),
            self.downsample
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            jnp.asarray([c.curve.gamma() for c in self.coils_to_target]),
            jnp.asarray([c.curve.gammadash() for c in self.coils_to_target]),
            jnp.asarray([c.current.get_value() for c in self.coils_to_target]),
            self.downsample
        ]
        dJ_dgammas = self.dJ_dgammas(*args)
        dJ_dgammadashs = self.dJ_dgammadashs(*args)
        dJ_dcurrents = self.dJ_dcurrents(*args)
        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrents[i]])) for i, c in enumerate(self.coils_to_target)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgammas[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadashs[i]) for i, c in enumerate(self.coils_to_target)])
            + vjp
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def net_fluxes_pure(gammas, gammadashs, gammas2, gammadashs2, currents2, downsample):
    """
    Calculate the net magnetic flux through a set of m coils with n quadrature points
    due to another set of m' coils with n' quadrature points.

    This function computes the total magnetic flux passing through a set of coils
    due to the magnetic field generated by another set of coils. The flux is calculated
    using the line integral of the vector potential along the coil paths.

    math:: 
        \Psi = \sum_i \int_{C_i} A_{ext}\cdot d\ell_i / L_i

    where :math:`A_{ext}` is the vector potential of an external magnetic field,
    evaluated along the quadpoints along the curve,
    :math:`L_i` is the total length of the ith coil, and :math:`\ell_i` is arclength
    along the ith coil.

    Note that the first set of coils is assumed to all have the same number of quadrature 
    points for the purposes of jit speed. Same with the second set of coils, although
    the number of points does not have to be the same between the two sets.

    The units of the objective function are Weber.

    Args:
        gammas (array, shape (m,n,3)): 
            Position vectors for the coils receiving flux.
        gammadashs (array, shape (m,n,3)): 
            Tangent vectors for the coils receiving flux.
        gammas2 (array, shape (m',n',3)): 
            Position vectors for the coils generating flux.
        gammadashs2 (array, shape (m',n',3)): 
            Tangent vectors for the coils generating flux.
        currents2 (array, shape (m',)): 
            Current values for the coils generating flux.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

    Returns:
        array (shape (m,)): 
            Net magnetic flux through each coil in the first set.
    """
    # Downsample if desired
    gammas = gammas[:, ::downsample, :]
    gammadashs = gammadashs[:, ::downsample, :]
    gammas2 = gammas2[:, ::downsample, :]
    gammadashs2 = gammadashs2[:, ::downsample, :]
    rij_norm = jnp.linalg.norm(gammas[:, :, None, None, :] - gammas2[None, None, :, :, :], axis=-1)
    # sum over the currents, and sum over the biot savart integral
    A_ext = jnp.sum(currents2[None, None, :, None] * jnp.sum(gammadashs2[None, None, :, :, :] / rij_norm[:, :, :, :, None], axis=-2), axis=-2) / jnp.shape(gammadashs2)[1]
    # Now sum over all the coil loops
    return 1e-7 * jnp.sum(jnp.sum(A_ext * gammadashs, axis=-1), axis=-1) / jnp.shape(gammadashs)[1]


def net_ext_fluxes_pure(gammadash, A_ext, downsample):
    """
    Calculate the net magnetic flux through a coil with n quadrature points
    due to an external vector potential evaluated at those points.

    math:: 
        \Psi = \int A_{ext}\cdot d\ell / L

    where :math:`A_{ext}` is the vector potential of an external magnetic field,
    evaluated along the quadpoints along the curve,
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.

    This function computes the total magnetic flux passing through a coil due to
    an external magnetic field represented by its vector potential. The flux is
    calculated using the line integral of the vector potential along the coil path.

    The units of the objective function are Weber.

    Args:
        gammadash (array, shape (n,3)): 
            Tangent vectors along the coil.
        A_ext (array, shape (n,3)): 
            External vector potential evaluated at coil points.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

    Returns:
        float: Net magnetic flux through the coil due to the external field.
    """
    # Downsample if desired
    gammadash = gammadash[::downsample, :]
    # Dot the vectors (sum over last axis), then sum over the quadpoints
    return jnp.sum(jnp.sum(A_ext * gammadash, axis=-1), axis=-1) / jnp.shape(gammadash)[0]


class NetFluxes(Optimizable):
    r"""
    Optimizable class for minimizing the total net flux from m coils
    through a single coil with n quadrature points. 
    This is mostly a test class for the passive coil arrays.

    The function is

     .. math::
        \Psi = \int A_{ext}\cdot d\ell / L

    where :math:`A_{ext}` is the vector potential of an external magnetic field,
    evaluated along the quadpoints along the curve,
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.

    The units of the objective function are Weber.

    Args:
        target_coil (Coil): Coil whose net flux is being computed.
        source_coils (list of Coil, shape (m,)): 
            List of coils to use for computing the net flux.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
    """

    def __init__(self, target_coil, source_coils, downsample=1):
        if not isinstance(source_coils, list):
            source_coils = [source_coils]
        self.target_coil = target_coil
        self.source_coils = [c for c in source_coils if c not in [target_coil]]
        self.downsample = downsample
        self.biotsavart = BiotSavart(self.source_coils)

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

        super().__init__(depends_on=[target_coil] + source_coils)

    def J(self):

        gamma = self.target_coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[::self.downsample, :]))
        args = [
            self.target_coil.curve.gammadash(),
            self.biotsavart.A(),
            self.downsample
        ]
        J = self.J_jax(*args)
        return J

    @derivative_dec
    def dJ(self):

        gamma = self.target_coil.curve.gamma()
        self.biotsavart.set_points(gamma)
        args = [
            self.target_coil.curve.gammadash(),
            self.biotsavart.A(),
            1
        ]

        dJ_dA = self.dJ_dA(*args)
        dA_dX = self.biotsavart.dA_by_dX()
        dJ_dX = np.einsum('ij,ikj->ik', dJ_dA, dA_dX)
        A_vjp = self.biotsavart.A_vjp(dJ_dA)

        dJ = (
            self.target_coil.curve.dgamma_by_dcoeff_vjp(dJ_dX) +
            self.target_coil.curve.dgammadash_by_dcoeff_vjp(self.dJ_dgammadash(*args))
            + A_vjp
        )
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def squared_mean_force_pure(gammas, gammas2, gammadashs, gammadashs2, currents,
                            currents2, downsample):
    """
    Compute the squared mean force on a set of m coils with varying quadrature points 
    (downsampled to have the smallest number, denoted n, of quadrature points),
    due to themselves and another set of m' coils with varying quadrature points
    (downsampled to n quadrature points).

    The objective function is

    .. math:
        J = \sum_i \left(\frac{\int \frac{d\vec{F}_i}{d\ell_i} d\ell_i}{L_i}\right)^2

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    :math:`L_i` is the total coil length,
    and :math:`\ell_i` is arclength along the ith coil. The units of the objective function are (N/m)^2.

    The coils are allowed to have different numbers of quadrature points, but for the purposes of
    jax speed, the coils are downsampled here to have the same number of quadrature points.

    Args:
        gammas (array, shape (m,n,3)): 
            Position vectors for the coils receiving force.
        gammas2 (array, shape (m',n,3)): 
            Position vectors for the coils generating force.
        gammadashs (array, shape (m,n,3)): 
            Tangent vectors for the coils receiving force.
        gammadashs2 (array, shape (m',n,3)): 
            Tangent vectors for the coils generating force.
        currents (array, shape (m,)): 
            Currents for the coils receiving force.
        currents2 (array, shape (m',)): 
            Currents for the coils generating force.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
    Returns:
        float: The squared mean force.
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
    r"""
    Optimizable class to minimize the (net (integrated) Lorentz force per unit length)^2 on a set of m coils
    from themselves and another set of m' coils.

    The objective function is

    .. math:
        J = \sum_i \left(\frac{\int \frac{d\vec{F}_i}{d\ell_i} d\ell_i}{L_i}\right)^2

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    :math:`L_i` is the total coil length,
    and :math:`\ell_i` is arclength along the ith coil. The units of the objective function are (N/m)^2.
    
    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. This is also useful for optimizing passive coils,
    which require a modified Jacobian calculation. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        coils_to_target (list of Coil, shape (m,)): 
            List of coils to use for computing MeanSquaredForce. 
        source_coils (list of Coil, shape (m',)): 
            List of coils that provide forces on the first set of coils but that
            we do not care about optimizing their forces. 
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
    """

    def __init__(self, coils_to_target, source_coils, downsample: int = 1):
        if not isinstance(coils_to_target, list):
            coils_to_target = [coils_to_target]
        if not isinstance(source_coils, list):
            source_coils = [source_coils]
        self.coils_to_target = coils_to_target
        self.source_coils = [c for c in source_coils if c not in coils_to_target]
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

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.coils_to_target)])
        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.source_coils)])
            + vjp
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.source_coils)])
        )
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}

def lp_force_pure(
    gammas, gammas2, gammadashs, gammadashs2, gammadashdashs,
    quadpoints, currents, currents2, regularizations, p, threshold, downsample=1
):
    """
    Computes the Lp force objective by summing over a set of m coils, 
    where each coil receives force from all coils (including itself and m' coils in a separate set).
    This version allows each coil to have its own quadrature points array.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{F}/d\ell_i| - dF_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dF_0/d\ell_i` is a threshold force at the ith coil.

    The units of the objective function are (N/m)^p.

    Args:
        gammas (array, shape (m,n,3)): 
            Position vectors for the coils receiving force.
        gammas2 (array, shape (m',n,3)): 
            Position vectors for the coils generating force.
        gammadashs (array, shape (m,n,3)): 
            Tangent vectors for the coils receiving force.
        gammadashdashs (array, shape (m,n,3)): 
            Second derivative of tangent vectors for the coils receiving force.
        quadpoints (array, shape (m,n,3)): 
            Quadrature points for the coils receiving force.
        currents (array, shape (m,)): 
            Currents for the coils receiving force.
        currents2 (array, shape (m',)):
            Currents for the coils generating force.
        regularizations (array, shape (m,)):
            Regularizations for the coils receiving force.
        p (float):
            Exponent for the Lp force objective.
        threshold (float):
            Threshold force for the coils receiving force.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

    Returns:
        float: The Lp force objective.
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
    r"""
    Optimizable class to minimize the squared Lorentz force per unit length integrated and summed over a
    set of m coils, from themselves and another set of m' coils.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{F}/d\ell_i| - dF_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dF_0/d\ell_i` is a threshold force at the ith coil.

    The units of the objective function are (N/m)^p.

    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types ofcross-sections.

    Args:
        coils_to_target (list of Coil, shape (m,)): 
            List of coils to use for computing LpCurveForce. 
        source_coils (list of Coil, shape (m',)): 
            List of coils that provide forces on the first set of coils but that
            we do not care about optimizing their forces. 
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
    """

    def __init__(self, coils_to_target, source_coils, p: float = 2.0, threshold: float = 0.0, downsample: int = 1):
        if not isinstance(coils_to_target, list):
            coils_to_target = [coils_to_target]
        if not isinstance(source_coils, list):
            source_coils = [source_coils]
        if not isinstance(coils_to_target[0], RegularizedCoil):
            raise ValueError("LpCurveForce can only be used with RegularizedCoil objects")
        regularizations = jnp.array([c.regularization for c in coils_to_target])
        self.coils_to_target = coils_to_target
        self.source_coils = [c for c in source_coils if c not in coils_to_target]
        quadpoints = [c.curve.quadpoints for c in coils_to_target]
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

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.curve.gammadashdash() for c in self.coils_to_target],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.curve.gammadashdash() for c in self.coils_to_target],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dgammadashdash = self.dJ_dgammadashdash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.coils_to_target)])
        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.coils_to_target)])
            + vjp
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.source_coils)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def lp_torque_pure(gammas, gammas2, gammadashs, gammadashs2, gammadashdashs,
                   quadpoints, currents, currents2, regularizations, p, threshold, downsample):
    r"""
    Pure function for computing the lp torque on a set of m coils with varying quadrature points,
    (each downsampled to n quadrature points) from themselves and another set of m' coils 
    with varying quadrature points (each downsampled to n quadrature points).

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{T}/d\ell_i| - dT_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dT_0/d\ell_i` is a threshold torque at the ith coil.

    The units of the objective function are (N)^p.

    Args:
        gammas (array, shape (m,n,3)): Array of coil positions.
        gammas2 (array, shape (m',n,3)): Array of coil positions.
        gammadashs (array, shape (m,n,3)): Array of coil tangent vectors.
        gammadashs2 (array, shape (m',n,3)): Array of coil tangent vectors.
        gammadashdashs (array, shape (m,n,3)): Array of second derivatives of coil positions.
        quadpoints (array, shape (m,n,3)): Array of quadrature points.
        currents (array, shape (m,)): Array of coil currents.
        currents2 (array, shape (m',)): Array of coil currents.
        regularizations (array, shape (m,)): Array of coil regularizations.
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

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
    r"""
    Optimizable class to minimize the Lorentz torque per unit length integrated and summed over a
    set of m coils, from themselves and another set of m' coils.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{T}/d\ell_i| - dT_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dT_0/d\ell_i` is a threshold torque at the ith coil.

    The units of the objective function are (N)^p.

    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        coils_to_target (list of Coil, shape (m,)): List of coils to use for computing LpCurveTorque. 
        source_coils (list of Coil, shape (m',)): List of coils that provide torques on the first set of coils but that
            we do not care about optimizing their torques. 
        p (float): Power of the objective function.
        threshold (float): Threshold for the objective function.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 
    """

    def __init__(self, coils_to_target, source_coils, p: float = 2.0, threshold: float = 0.0, downsample: int = 1):
        if not isinstance(coils_to_target, list):
            coils_to_target = [coils_to_target]
        if not isinstance(source_coils, list):
            source_coils = [source_coils]
        if not isinstance(coils_to_target[0], RegularizedCoil):
            raise ValueError("LpCurveTorque can only be used with RegularizedCoil objects")
        regularizations = jnp.array([c.regularization for c in coils_to_target])
        self.coils_to_target = coils_to_target
        self.source_coils = [c for c in source_coils if c not in coils_to_target]
        quadpoints = [c.curve.quadpoints for c in coils_to_target]
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

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.curve.gammadashdash() for c in self.coils_to_target],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.curve.gammadashdash() for c in self.coils_to_target],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dgammadashdash = self.dJ_dgammadashdash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.coils_to_target)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash[i]) for i, c in enumerate(self.coils_to_target)])
            + vjp
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.source_coils)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def squared_mean_torque(gammas, gammas2, gammadashs, gammadashs2, currents, currents2, downsample):
    """
    Compute the squared mean torque on a set of m coils with varying quadrature points 
    (downsampled to n quadrature points) due to themselves and another set of m' coils 
    with varying quadrature points (downsampled to n quadrature points).

    The objective function is

    .. math:
        J = \sum_i(\frac{\int \frac{d\vec{T}_i}{d\ell_i} d\ell_i}{L_i})^2
        
    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length.

    The units of the objective function are (N)^2.

    Args:
        gammas (array, shape (m,n,3)): Array of coil positions in coil set 1.
        gammas2 (array, shape (m',n,3)): Array of coil positions in coil set 2.
        gammadashs (array, shape (m,n,3)): Array of coil tangent vectors in coil set 1.
        gammadashs2 (array, shape (m',n,3)): Array of coil tangent vectors in coil set 2.
        currents (array, shape (m,)): Array of coil currents in coil set 1.
        currents2 (array, shape (m',)): Array of coil currents in coil set 2.
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

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
    r"""
    Optimizable class to minimize the (net (integrated) Lorentz torque per unit length)^2 summed 
    over a set of m coils due to themselves and another set of m' coils.

    The objective function is

    .. math:
        J = \sum_i(\frac{\int \frac{d\vec{T}_i}{d\ell_i} d\ell_i}{L_i})^2
        
    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length.

    The units of the objective function are (N)^2.
    
    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        coils_to_target (list of Coil, shape (m,)): List of coils to use for computing SquaredMeanTorque. 
        source_coils (list of Coil, shape (m',)): List of coils that provide torques on the first set of coils but that
            we do not care about optimizing their torques. 
        downsample (int): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

    Returns:
        float: Value of the objective function.
    """

    def __init__(self, coils_to_target, source_coils, downsample: int = 1):
        if not isinstance(coils_to_target, list):
            coils_to_target = [coils_to_target]
        if not isinstance(source_coils, list):
            source_coils = [source_coils]
        self.coils_to_target = coils_to_target
        self.source_coils = [c for c in source_coils if c not in coils_to_target]
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

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):

        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]
        dJ_dgamma = self.dJ_dgamma(*args)
        dJ_dgammadash = self.dJ_dgammadash(*args)
        dJ_dcurrent = self.dJ_dcurrent(*args)
        dJ_dgamma2 = self.dJ_dgamma2(*args)
        dJ_dgammadash2 = self.dJ_dgammadash2(*args)
        dJ_dcurrent2 = self.dJ_dcurrent2(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent[i]])) for i, c in enumerate(self.coils_to_target)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma2[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash2[i]) for i, c in enumerate(self.source_coils)])
            + vjp
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent2[i]])) for i, c in enumerate(self.source_coils)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}
