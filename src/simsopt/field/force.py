"""Implements the force on a coil in its own magnetic field and the field of other coils."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscp
from jax import grad, vmap
from jax.lax import cond
from .biotsavart import BiotSavart
from .coil import RegularizedCoil
from .selffield import B_regularized_pure
from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

__all__ = [
    "_coil_coil_inductances_pure",
    "_coil_coil_inductances_inv_pure",
    "_induced_currents_pure",
    "NetFluxes",
    "B2Energy",
    "SquaredMeanForce",
    "LpCurveForce",
    "SquaredMeanTorque",
    "LpCurveTorque",
]


def _check_quadpoints_consistency(coils, label="coils"):
    """Check that all coils in a list have the same number of quadrature points.
    
    Args:
        coils: list of coils to check.
        label: descriptive label for the coil group (used in error message).
    
    Raises:
        ValueError: if not all coils have the same number of quadrature points.
    """
    nquadpoints = [len(c.curve.quadpoints) for c in coils]
    if len(set(nquadpoints)) > 1:
        raise ValueError(
            f"All coils in {label} must have the same number of quadrature points, "
            f"but got {nquadpoints}."
        )


def _B_at_point_from_coil_set_pure(pt, gammas, gammadashs, currents, exclude_index, eps):
    r"""
    Compute the magnetic field at a single point due to a set of coils via the Biot-Savart law,
    optionally excluding one coil (e.g. to avoid self-contribution).

    This is a pure JAX implementation of the Biot-Savart integral used by force and torque
    objectives in this module. We do not use the :class:`BiotSavart` class here because
    constructing one :class:`BiotSavart` per objective (e.g. per coil in :class:`LpCurveForce`)
    leads to a large number of optimizable dependencies and weak references. That makes
    operations like ``Jf.x = dofs`` scale poorly with the number of coils (tens of millions of
    function calls and tens of seconds for ~64 coils). See `GitHub issue #487
    <https://github.com/hiddenSymmetries/simsopt/issues/487>`_.

    .. math::
        B = \frac{\mu_0}{4\pi} \frac{1}{n_{pts}} \sum_{j \neq \mathrm{exclude}} I_j \int \frac{d\vec{\ell}_j \times (\vec{r} - \vec{r}_j)}{|\vec{r} - \vec{r}_j|^3}

    Args:
        pt: Array of shape (3,); evaluation point.
        gammas: Array of shape (m, n, 3); positions for m coils with n quadrature points.
        gammadashs: Array of shape (m, n, 3); tangent vectors.
        currents: Array of shape (m,); coil currents.
        exclude_index: Index of coil to exclude from the sum (use -1 to include all).
        eps: Small constant added to distances to avoid division by zero.

    Returns:
        Array of shape (3,); magnetic field contribution (without mu_0/(4*pi)).
    """
    n = gammas.shape[0]
    npts = gammas.shape[1]

    def from_j(j):
        return cond(
            (exclude_index >= 0) & (j == exclude_index),
            lambda _: jnp.zeros(3),
            lambda _: jnp.asarray(
                jnp.sum(
                    jnp.cross(gammadashs[j], pt - gammas[j])
                    / (jnp.linalg.norm(pt - gammas[j] + eps, axis=1) ** 3)[:, None],
                    axis=0
                ) * currents[j]
            ),
            operand=None
        )

    B = jnp.sum(vmap(from_j)(jnp.arange(n)), axis=0)
    return B / npts * 1e-7


def _mutual_B_field_at_point_pure(
    i, pt,
    gammas_targets, gammadashs_targets, currents_targets,
    gammas_sources, gammadashs_sources, currents_sources,
    eps
):
    r"""
    Compute the mutual magnetic field at a point on target coil i from all target coils
    (excluding coil i) and all source coils in Tesla.

    Used by :func:`squared_mean_force_pure`, :func:`lp_force_pure`, :func:`lp_torque_pure`,
    and :func:`squared_mean_torque`. See :func:`_B_at_point_from_coil_set_pure`
    for why Biot-Savart is reimplemented here instead of using :class:`BiotSavart`.

    Args:
        i: Index of target coil.
        pt: Array of shape (3,); evaluation point.
        gammas_targets: Array of shape (m, n, 3); positions for m target coils with n quadrature points.
        gammadashs_targets: Array of shape (m, n, 3); tangent vectors for m target coils with n quadrature points.
        currents_targets: Array of shape (m,); currents for m target coils.
        gammas_sources: Array of shape (m', n, 3); positions for m' source coils with n quadrature points.
        gammadashs_sources: Array of shape (m', n, 3); tangent vectors for m' source coils with n quadrature points.
        currents_sources: Array of shape (m',); currents for m' source coils.
        eps: Small constant added to distances to avoid division by zero.

    Returns:
        Array of shape (3,); mutual magnetic field at point pt in Tesla.
    """
    B_targets = _B_at_point_from_coil_set_pure(
        pt, gammas_targets, gammadashs_targets, currents_targets, exclude_index=i, eps=eps
    )
    B_sources = _B_at_point_from_coil_set_pure(
        pt, gammas_sources, gammadashs_sources, currents_sources, exclude_index=-1, eps=eps
    )
    return B_targets + B_sources


def _lorentz_force_density_pure(tangents, current, magnetic_field):
    """Compute Lorentz force density I * (t x B)."""
    return current * jnp.cross(tangents, magnetic_field)


def _prepare_target_source_inputs_pure(
    gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources,
    currents_targets, currents_sources, downsample
):
    """
    Downsample and convert shared target/source inputs used by force/torque objectives.
    
    Args:
        gammas_targets: Array of shape (m, n, 3); positions for m target coils with n quadrature points.
        gammadashs_targets: Array of shape (m, n, 3); tangent vectors for m target coils with n quadrature points.
        gammas_sources: Array of shape (m', n, 3); positions for m' source coils with n quadrature points.
        gammadashs_sources: Array of shape (m', n, 3); tangent vectors for m' source coils with n quadrature points.
        currents_targets: Array of shape (m,); currents for m target coils.
        currents_sources: Array of shape (m',); currents for m' source coils.
        downsample: Factor by which to downsample the quadrature points.

    Returns:
        Tuple of arrays: (gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_targets, currents_sources).
    """
    return (
        jnp.stack(gammas_targets)[:, ::downsample, :],
        jnp.stack(gammadashs_targets)[:, ::downsample, :],
        jnp.stack(gammas_sources)[:, ::downsample, :],
        jnp.stack(gammadashs_sources)[:, ::downsample, :],
        jnp.array(currents_targets),
        jnp.array(currents_sources),
    )


def _prepare_regularized_target_source_inputs_pure(
    gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints,
    gammas_sources, gammadashs_sources, currents_targets, currents_sources,
    regularizations, downsample
):
    """
    Downsample/convert inputs for regularized Lp force/torque objectives. Just a wrapper around 
    _prepare_target_source_inputs_pure that also prepares additional inputs for regularized coils.
    
    Args:
        gammas_targets: Array of shape (m, n, 3); positions for m target coils with n quadrature points.
        gammadashs_targets: Array of shape (m, n, 3); tangent vectors for m target coils with n quadrature points.
        gammadashdashs_targets: Array of shape (m, n, 3); second derivatives of tangent vectors for m target coils with n quadrature points.
        quadpoints: Array of shape (m, n); quadrature points for m target coils with n quadrature points.
        gammas_sources: Array of shape (m', n, 3); positions for m' source coils with n quadrature points.
        gammadashs_sources: Array of shape (m', n, 3); tangent vectors for m' source coils with n quadrature points.
        currents_targets: Array of shape (m,); currents for m target coils.
        currents_sources: Array of shape (m',); currents for m' source coils.
        regularizations: Array of shape (m,); regularizations for m target coils.
        downsample: Factor by which to downsample the quadrature points.

    Returns:
        Tuple of arrays: (gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints, gammas_sources, gammadashs_sources, currents_targets, currents_sources, regularizations).
    """
    gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_targets, currents_sources = (
        _prepare_target_source_inputs_pure(
            gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources,
            currents_targets, currents_sources, downsample
        )
    )
    return (
        gammas_targets,
        gammadashs_targets,
        jnp.stack(gammadashdashs_targets)[:, ::downsample, :],
        jnp.asarray(quadpoints[0])[::downsample],
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        jnp.array(regularizations),
    )


def _coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations, eps=1e-10):
    r"""
    Compute the full inductance matrix for a set of coils, including both mutual and 
    self-inductances. All coils are assumed to have the same number of quadrature points, 
    denoted n. The units of the inductance matrix are H, where H = henries.

    The mutual inductance between two coils is computed as:

    .. math::

        M = \frac{\mu_0}{4\pi} \iint \frac{d\vec{r}_A \cdot d\vec{r}_B}{|\vec{r}_A - \vec{r}_B|}

    and self-inductance of a regularized coil is computed as:

    .. math::

        L = \frac{\mu_0}{4\pi} \int_0^{2\pi} d\phi \int_0^{2\pi} d\tilde{\phi} 
            \frac{\vec{r}_c' \cdot \tilde{\vec{r}}_c'}{\sqrt{|\vec{r}_c - \tilde{\vec{r}}_c|^2 + \delta a b}}

    where $\delta a b$ is a regularization parameter depending on the cross-section. The units
    of the inductance matrices are henries.

    Args:
        gammas (array, shape (m,n,3)): 
            Array of coil positions for all m coils.
        gammadashs (array, shape (m,n,3)): 
            Array of coil tangent vectors for all m coils.
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
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can 
            have different size and shape cross-sections in this list of regularization terms.
        eps (float): Small constant to avoid division by zero for mutual inductance between coil_i and itself.
    Returns:
        array (shape (m,m)): Full inductance matrix Lij.
    """
    gammas = jnp.asarray(gammas)[:, ::downsample, :]
    gammadashs = jnp.asarray(gammadashs)[:, ::downsample, :]
    N = gammas.shape[0]

    # Compute Lij, i != j
    r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :] + eps
    rij_norm = jnp.linalg.norm(r_ij, axis=-1)
    gammadash_prod = jnp.sum(gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1)

    # Double sum over each of the closed curves for off-diagonal elements
    Lij = jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=-1), axis=-1) / jnp.shape(gammas)[1] ** 2

    # Compute diagonal elements for each coil
    diag_values = jnp.sum(jnp.sum(gammadash_prod / jnp.sqrt(rij_norm ** 2 + regularizations[None, :, None, None]),
                                axis=-1), axis=-1) / jnp.shape(gammas)[1] ** 2

    # Now use a mask to replace the wrong diagonal with the correct numbers in diag_values
    diag_mask = jnp.eye(N, dtype=bool)
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

    The units of the inverse of the coil inductance matrix are 1/H, where H = henries.

    Args:
        gammas (array, shape (m,n,3)): 
            Array of coil positions for all m coils.
        gammadashs (array, shape (m,n,3)): 
            Array of coil tangent vectors for all m coils.
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
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can 
            have different size and shape cross-sections in this list of regularization terms.

    Returns:
        array (shape (m,m)): Array of inverse of the coil inductance matrix.
    """
    # Lij is symmetric positive definite so has a cholesky decomposition
    C = jnp.linalg.cholesky(_coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations))
    inv_C = jscp.linalg.solve_triangular(C, jnp.eye(C.shape[0]), lower=True)
    inv_L = jscp.linalg.solve_triangular(C.T, inv_C, lower=False)
    return inv_L


def _induced_currents_pure(gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_sources, downsample, regularizations):
    """
    Pure function for computing the induced currents in a set of m passive coils with n quadrature points
    due to a set of m' source coils with n' quadrature points (and themselves). 

    .. math::
        I = -L^{-1} \Psi

    where :math:`L` is the coil inductance matrix, :math:`\Psi` is the net flux through 
    the passive coils due to the source coils,
    and :math:`I` is the induced currents in the passive coils. 
    The units of the induced currents are Amperes.

    Args:
        gammas_targets (array, shape (m,n,3)): 
            Array of passive coil positions for all m coils.
        gammadashs_targets (array, shape (m,n,3)): 
            Array of passive coil tangent vectors for all m coils.
        gammas_sources (array, shape (m',n',3)): 
            Array of source coil positions for all m' coils.
        gammadashs_sources (array, shape (m',n',3)): 
            Array of source coil tangent vectors for all m' coils.
        currents_sources (array, shape (m',)): 
            Array of source coil currents.
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
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can 
            have different size and shape cross-sections in this list of regularization terms.

    Returns:
        array (shape (m,)): Array of induced currents.
    """
    return -_coil_coil_inductances_inv_pure(gammas_targets, gammadashs_targets, downsample, regularizations) @ _net_fluxes_pure(gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_sources, downsample)


def b2energy_pure(gammas, gammadashs, currents, downsample, regularizations):
    r"""
    Pure function for evaluating the total vacuum magnetic field energy from a set of m coils
    with n quadrature points each.
    The function is

     .. math::
        J = \frac{1}{2}\sum_{i,j}I_iL_{ij}I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil. 
    The units of the objective function are MJ (megajoules).

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
        float: Value of the objective function in MJ (megajoules).
    """
    Ii_Ij = (currents[:, None] * currents[None, :])
    Lij = _coil_coil_inductances_pure(
        gammas, gammadashs, downsample, regularizations
    )
    U = 0.5 * (jnp.sum(Ii_Ij * Lij))
    return U / 1e6  # Convert from Joules to MJ


class B2Energy(Optimizable):
    r"""
    Optimizable class for minimizing the total vacuum magnetic field energy from a set of m coils.

    The function is

     .. math::
        J = \frac{1}{2}\sum_{i,j}I_i L_{ij} I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil. 
    The units of the objective function are MJ (megajoules).

    Args:
        coils_to_target (list of RegularizedCoil, shape (m,)): 
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
        _check_quadpoints_consistency(self.coils_to_target, "coils_to_target")
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
        r"""Evaluate the B^2 energy objective.

        Returns:
            float: The total vacuum magnetic field energy
                :math:`J = \frac{1}{2}\sum_{i,j} I_i L_{ij} I_j` in MJ.
        """
        args = [
            jnp.asarray([c.curve.gamma() for c in self.coils_to_target]),
            jnp.asarray([c.curve.gammadash() for c in self.coils_to_target]),
            jnp.asarray([c.current.get_value() for c in self.coils_to_target]),
            self.downsample
        ]

        return self.J_jax(*args)

    @derivative_dec
    def dJ(self):
        r"""Compute the derivative of the B^2 energy objective with respect to
        all optimizable degrees of freedom (coil geometry and currents).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
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


def _net_fluxes_pure(gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_sources, downsample):
    r"""
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
        gammas_targets (array, shape (m,n,3)): 
            Position vectors for the coils receiving flux.
        gammadashs_targets (array, shape (m,n,3)): 
            Tangent vectors for the coils receiving flux.
        gammas_sources (array, shape (m',n',3)): 
            Position vectors for the coils generating flux.
        gammadashs_sources (array, shape (m',n',3)): 
            Tangent vectors for the coils generating flux.
        currents_sources (array, shape (m',)): 
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
    gammas_targets = gammas_targets[:, ::downsample, :]
    gammadashs_targets = gammadashs_targets[:, ::downsample, :]
    gammas_sources = gammas_sources[:, ::downsample, :]
    gammadashs_sources = gammadashs_sources[:, ::downsample, :]
    rij_norm = jnp.linalg.norm(gammas_targets[:, :, None, None, :] - gammas_sources[None, None, :, :, :], axis=-1)
    # sum over the currents, and sum over the biot savart integral
    A_ext = jnp.sum(currents_sources[None, None, :, None] * jnp.sum(gammadashs_sources[None, None, :, :, :] / rij_norm[:, :, :, :, None], axis=-2), axis=-2) / jnp.shape(gammadashs_sources)[1]
    # Now sum over all the coil loops
    return 1e-7 * jnp.sum(jnp.sum(A_ext * gammadashs_targets, axis=-1), axis=-1) / jnp.shape(gammadashs_targets)[1]


def net_ext_fluxes_pure(gammadash, A_ext, downsample):
    r"""
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
        r"""Evaluate the net flux objective.

        Computes :math:`\Psi = \int A_{ext} \cdot d\ell / L` using the BiotSavart
        vector potential from the source coils evaluated at the target coil quadrature points.

        Returns:
            float: Net magnetic flux through the target coil in Weber.
        """
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
        r"""Compute the derivative of the net flux objective with respect to
        all optimizable degrees of freedom (target coil geometry and source coil
        geometry/currents).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
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


def squared_mean_force_pure(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets,
                            currents_sources, downsample, eps=1e-10):
    r"""
    Compute the squared mean force on a set of m coils with n quadrature points,
    due to themselves and another set of m' coils with n' quadrature points.

    The objective function is

    .. math:
        J = \sum_i \left(\frac{\int \frac{d\vec{F}_i}{d\ell_i} d\ell_i}{L_i}\right)^2

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    in units of MN/m. The units of the squared mean force are therefore (MN/m)^2.
    :math:`L_i` is the total coil length,
    and :math:`\ell_i` is arclength along the ith coil. The units of the objective function are (MN/m)^2, where MN = meganewtons.

    All coils within a set are assumed to have the same number of quadrature points,
    though the two sets may differ.

    Args:
        gammas_targets (array, shape (m,n,3)): 
            Position vectors for the coils receiving force.
        gammas_sources (array, shape (m',n,3)): 
            Position vectors for the coils generating force.
        gammadashs_targets (array, shape (m,n,3)): 
            Tangent vectors for the coils receiving force.
        gammadashs_sources (array, shape (m',n,3)): 
            Tangent vectors for the coils generating force.
        currents_targets (array, shape (m,)): 
            Currents for the coils receiving force.
        currents_sources (array, shape (m',)): 
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
        eps (float): Small constant to avoid division by zero for force between coil_i and itself.
    Returns:
        float: The squared mean force.
    """
    gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_targets, currents_sources = (
        _prepare_target_source_inputs_pure(
            gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources,
            currents_targets, currents_sources, downsample
        )
    )

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    # Precompute tangents and norms
    gammadash_norms = jnp.linalg.norm(gammadashs_targets, axis=-1)[:, :, None]
    tangents = gammadashs_targets / gammadash_norms

    def mean_force_group1(i, gamma_i, tangent_i, gammadash_norm_i, current_i):
        B_mutual = vmap(
            lambda pt: _mutual_B_field_at_point_pure(
                i, pt,
                gammas_targets, gammadashs_targets, currents_targets,
                gammas_sources, gammadashs_sources, currents_sources,
                eps
            )
        )(gamma_i)
        force_density = _lorentz_force_density_pure(tangent_i, current_i, B_mutual)
        return jnp.sum(force_density * gammadash_norm_i, axis=0) / npts1

    mean_forces = vmap(mean_force_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, tangents, gammadash_norms, currents_targets
    )
    # already multiplied by (mu_0/(4*pi)) in _mutual_B_field_at_point_pure, 
    # which gives a factor of (mu_0/(4*pi))^2 = 1e-14
    # Then convert from (N/m)^2 to (MN/m)^2 by dividing by (1e6)^2 = 1e12
    mean_forces_squared = jnp.sum(jnp.linalg.norm(mean_forces, axis=-1) ** 2)
    return mean_forces_squared * 1e-12


class SquaredMeanForce(Optimizable):
    r"""
    Optimizable class to minimize the (net (integrated) Lorentz force per unit length)^2 on a set of m coils
    from themselves and another set of m' coils.

    The objective function is

    .. math:
        J = \sum_i \left(\frac{\int \frac{d\vec{F}_i}{d\ell_i} d\ell_i}{L_i}\right)^2

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    in units of MN/m. The units of the squared mean force are therefore (MN/m)^2.
    :math:`L_i` is the total coil length,
    and :math:`\ell_i` is arclength along the ith coil. The units of the objective function are (MN/m)^2, where MN = meganewtons.
    
    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. This is also useful for optimizing passive coils,
    which require a modified Jacobian calculation. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        coils_to_target (list of Coil or RegularizedCoil, shape (m,)): 
            List of coils to use for computing SquaredMeanForce. 
        source_coils (list of Coil or RegularizedCoil, shape (m',)): 
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
        _check_quadpoints_consistency(self.coils_to_target, "coils_to_target")
        if self.source_coils:
            _check_quadpoints_consistency(self.source_coils, "source_coils")
        self.downsample = downsample
        args = {"static_argnums": (6,)}

        self.J_jax = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            squared_mean_force_pure(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgamma_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=0)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgamma_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=1)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=2)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=3)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dcurrent_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=4)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dcurrent_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=5)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):
        r"""Evaluate the squared mean force objective.

        Returns:
            float: The sum of squared mean Lorentz forces
                :math:`\sum_i \left(\frac{\int (d\vec{F}_i/d\ell_i)\,d\ell_i}{L_i}\right)^2`
                in (MN/m)^2.
        """
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
        r"""Compute the derivative of the squared mean force objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        coils_to_target and source_coils).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]
        dJ_dgamma_targets = self.dJ_dgamma_targets(*args)
        dJ_dgammadash_targets = self.dJ_dgammadash_targets(*args)
        dJ_dcurrent_targets = self.dJ_dcurrent_targets(*args)
        dJ_dgamma_sources = self.dJ_dgamma_sources(*args)
        dJ_dgammadash_sources = self.dJ_dgammadash_sources(*args)
        dJ_dcurrent_sources = self.dJ_dcurrent_sources(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent_targets[i]])) for i, c in enumerate(self.coils_to_target)])
        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_sources[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_sources[i]) for i, c in enumerate(self.source_coils)])
            + vjp
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent_sources[i]])) for i, c in enumerate(self.source_coils)])
        )
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}

def lp_force_pure(
    gammas_targets, 
    gammas_sources, 
    gammadashs_targets, 
    gammadashs_sources, 
    gammadashdashs_targets, 
    quadpoints, 
    currents_targets, 
    currents_sources, 
    regularizations, 
    p, 
    threshold, 
    downsample,
    eps=1e-10
):
    r"""
    Computes the Lp force objective by summing over a set of m coils, 
    where each coil receives force from all coils (including itself and m' coils in a separate set).
    All coils within a set are assumed to have the same number of quadrature points,
    though the two sets may differ.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{F}/d\ell_i| - dF_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    in units of MN/m, where MN = meganewtons. 
    The units of the objective function are therefore (MN/m)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dF_0/d\ell_i` is a threshold force at the ith coil.

    Args:
        gammas_targets (array, shape (m,n,3)): 
            Position vectors for the coils receiving force.
        gammas_sources (array, shape (m',n,3)): 
            Position vectors for the coils generating force.
        gammadashs_targets (array, shape (m,n,3)): 
            Tangent vectors for the coils receiving force.
        gammadashs_sources (array, shape (m',n,3)): 
            Tangent vectors for the coils generating force.
        gammadashdashs_targets (array, shape (m,n,3)): 
            Second derivative of tangent vectors for the coils receiving force.
        quadpoints (array, shape (m,n)): 
            Quadrature points for target coils. Since target coils are required to have
            matching quadrature, the first entry is used.
        currents_targets (array, shape (m,)): 
            Currents for the coils receiving force.
        currents_sources (array, shape (m',)):
            Currents for the coils generating force.
        regularizations (array, shape (m,)):
            Array of regularizations coming from finite cross-section for all coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can 
            have different size and shape cross-sections in this list of regularization terms.
        p (float):
            Exponent for the Lp force objective.
        threshold (float):
            Threshold force per unit length in units of MN/m (meganewtons per meter).
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
        eps (float): Small constant to avoid division by zero for force between coil_i and itself.
    Returns:
        float: The Lp force objective.
    """
    gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints, gammas_sources, gammadashs_sources, currents_targets, currents_sources, regularizations = (
        _prepare_regularized_target_source_inputs_pure(
            gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints,
            gammas_sources, gammadashs_sources, currents_targets, currents_sources,
            regularizations, downsample
        )
    )

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    # Precompute tangents and norms
    gammadash_norms = jnp.linalg.norm(gammadashs_targets, axis=-1)[:, :, None]
    tangents = gammadashs_targets / gammadash_norms

    # Precompute B_self for each coil
    B_self = vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0))(
        gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints, currents_targets, regularizations
    )

    def per_coil_obj_group1(i, gamma_i, tangent_i, B_self_i, current_i):
        B_mutual = vmap(
            lambda pt: _mutual_B_field_at_point_pure(
                i, pt,
                gammas_targets, gammadashs_targets, currents_targets,
                gammas_sources, gammadashs_sources, currents_sources,
                eps
            )
        )(gamma_i)
        F = _lorentz_force_density_pure(tangent_i, current_i, B_mutual + B_self_i)
        # Force per unit length is in N/m, convert to MN/m
        return jnp.linalg.norm(F, axis=-1) / 1e6

    obj1 = vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, tangents, B_self, currents_targets
    )

    # obj1 is now in MN/m, threshold is in MN/m
    return (jnp.sum(jnp.sum(jnp.maximum(obj1 - threshold, 0) ** p * gammadash_norms[:, :, 0])) / npts1) * (1. / p)


class LpCurveForce(Optimizable):
    r"""
    Optimizable class to minimize the Lp Lorentz force per unit length integrated and summed over a
    set of m coils, from themselves and another set of m' coils.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{F}/d\ell_i| - dF_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length, 
    in units of MN/m, where MN = meganewtons. The units of the objective function are therefore (MN/m)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dF_0/d\ell_i` is a threshold force at the ith coil.

    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        coils_to_target (list of RegularizedCoil, shape (m,)): 
            List of coils to use for computing LpCurveForce. 
        source_coils (list of RegularizedCoil, shape (m',)): 
            List of coils that provide forces on the first set of coils but that
            we do not care about optimizing their forces. 
        p (float): Power of the objective function.
        threshold (float): Threshold force per unit length in units of MN/m (meganewtons per meter).
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
        _check_quadpoints_consistency(self.coils_to_target, "coils_to_target")
        if self.source_coils:
            _check_quadpoints_consistency(self.source_coils, "source_coils")
        quadpoints = [c.curve.quadpoints for c in coils_to_target]
        self.downsample = downsample
        args = {"static_argnums": (7,)}
        self.J_jax = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            lp_force_pure(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, quadpoints,
                          currents_targets, currents_sources, regularizations, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
                grad(self.J_jax, argnums=0)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgamma_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
                grad(self.J_jax, argnums=1)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
                grad(self.J_jax, argnums=2)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
                grad(self.J_jax, argnums=3)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadashdash_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
                grad(self.J_jax, argnums=4)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dcurrent_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
                grad(self.J_jax, argnums=5)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )
        self.dJ_dcurrent_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
                grad(self.J_jax, argnums=6)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):
        r"""Evaluate the Lp curve force objective.

        Returns:
            float: The Lp force objective
                :math:`\frac{1}{p}\sum_i \frac{1}{L_i}\int \max(|d\vec{F}/d\ell_i| - dF_0/d\ell_i, 0)^p\,d\ell_i`
                in (MN/m)^p.
        """
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
        r"""Compute the derivative of the Lp curve force objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        coils_to_target and source_coils).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
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
        dJ_dgamma_targets = self.dJ_dgamma_targets(*args)
        dJ_dgammadash_targets = self.dJ_dgammadash_targets(*args)
        dJ_dgammadashdash_targets = self.dJ_dgammadashdash_targets(*args)
        dJ_dcurrent_targets = self.dJ_dcurrent_targets(*args)
        dJ_dgamma_sources = self.dJ_dgamma_sources(*args)
        dJ_dgammadash_sources = self.dJ_dgammadash_sources(*args)
        dJ_dcurrent_sources = self.dJ_dcurrent_sources(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent_targets[i]])) for i, c in enumerate(self.coils_to_target)])
        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + vjp
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_sources[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_sources[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent_sources[i]])) for i, c in enumerate(self.source_coils)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def lp_torque_pure(
    gammas_targets, 
    gammas_sources, 
    gammadashs_targets, 
    gammadashs_sources, 
    gammadashdashs_targets,
    quadpoints, 
    currents_targets, 
    currents_sources, 
    regularizations, 
    p, 
    threshold, 
    downsample,
    eps=1e-10
):
    r"""
    Pure function for computing the Lp torque on a set of m coils with n quadrature points
    from themselves and another set of m' coils with n' quadrature points.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{T}/d\ell_i| - dT_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    in units of MN, where MN = meganewtons. 
    The units of the objective function are therefore (MN)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dT_0/d\ell_i` is a threshold torque per unit length at the ith coil.

    Args:
        gammas_targets (array, shape (m,n,3)): Array of target coil positions.
        gammas_sources (array, shape (m',n,3)): Array of source coil positions.
        gammadashs_targets (array, shape (m,n,3)): Array of target coil tangent vectors.
        gammadashs_sources (array, shape (m',n,3)): Array of source coil tangent vectors.
        gammadashdashs_targets (array, shape (m,n,3)): Array of second derivatives of target coil positions.
        quadpoints (array, shape (m,n)): 
            Quadrature points for target coils. Since target coils are required to have
            matching quadrature, the first entry is used.
        currents_targets (array, shape (m,)): Array of target coil currents.
        currents_sources (array, shape (m',)): Array of source coil currents.
        regularizations (array, shape (m,)): 
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can 
            have different size and shape cross-sections in this list of regularization terms.
        p (float): Power of the objective function.
        threshold (float): Threshold torque per unit length in units of MN (meganewtons).
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
        eps (float): Small constant to avoid division by zero for torque between coil_i and itself.
    Returns:
        float: Value of the objective function.
    """
    from simsopt.geo.curve import centroid_pure
    gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints, gammas_sources, gammadashs_sources, currents_targets, currents_sources, regularizations = (
        _prepare_regularized_target_source_inputs_pure(
            gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints,
            gammas_sources, gammadashs_sources, currents_targets, currents_sources,
            regularizations, downsample
        )
    )

    centers = vmap(centroid_pure, in_axes=(0, 0))(gammas_targets, gammadashs_targets)

    # Precompute B_self for each coil
    B_self = vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0))(
        gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints, currents_targets, regularizations
    )
    gammadash_norms = jnp.linalg.norm(gammadashs_targets, axis=-1)[:, :, None]
    tangents = gammadashs_targets / gammadash_norms

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    def per_coil_obj_group1(i, gamma_i, center_i, tangent_i, B_self_i, current_i):
        def torque_at_point(idx):
            B_mutual = _mutual_B_field_at_point_pure(
                i, gamma_i[idx],
                gammas_targets, gammadashs_targets, currents_targets,
                gammas_sources, gammadashs_sources, currents_sources,
                eps
            )
            F = current_i * jnp.cross(tangent_i[idx], B_mutual + B_self_i[idx])
            tau = jnp.cross(gamma_i[idx] - center_i, F)
            # Torque per unit length is in N, convert to MN
            torque_per_unit_length_N = jnp.linalg.norm(tau)
            return torque_per_unit_length_N / 1e6  # Convert to MN
        return vmap(torque_at_point)(jnp.arange(npts1))

    obj1 = vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, centers, tangents, B_self, currents_targets
    )

    # obj1 is now in MN, threshold is in MN
    return jnp.sum(jnp.sum(jnp.maximum(obj1 - threshold, 0) ** p * gammadash_norms[:, :, 0])) / npts1 * (1. / p)


class LpCurveTorque(Optimizable):
    r"""
    Optimizable class to minimize the Lorentz torque per unit length integrated and summed over a
    set of m coils, from themselves and another set of m' coils.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{T}/d\ell_i| - dT_0/d\ell_i, 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    in units of MN, where MN = meganewtons. 
    The units of the objective function are therefore (MN)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`dT_0/d\ell_i` is a threshold torque per unit length at the ith coil.

    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        coils_to_target (list of RegularizedCoil, shape (m,)): List of coils to use for computing LpCurveTorque. 
        source_coils (list of RegularizedCoil, shape (m',)): List of coils that provide torques on the first set of coils but that
            we do not care about optimizing their torques. 
        p (float): Power of the objective function.
        threshold (float): Threshold torque per unit length in units of MN (meganewtons).
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
        _check_quadpoints_consistency(self.coils_to_target, "coils_to_target")
        if self.source_coils:
            _check_quadpoints_consistency(self.source_coils, "source_coils")
        quadpoints = [c.curve.quadpoints for c in coils_to_target]
        self.downsample = downsample
        args = {"static_argnums": (7,)}

        self.J_jax = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            lp_torque_pure(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, quadpoints,
                           currents_targets, currents_sources, regularizations, p, threshold, downsample),
            **args
        )

        self.dJ_dgamma_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=0)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgamma_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=1)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=2)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=3)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadashdash_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=4)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dcurrent_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=5)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dcurrent_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=6)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, gammadashdashs_targets, currents_targets, currents_sources, downsample),
            **args
        )

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):
        r"""Evaluate the Lp curve torque objective.

        Returns:
            float: The Lp torque objective
                :math:`\frac{1}{p}\sum_i \frac{1}{L_i}\int \max(|d\vec{T}/d\ell_i| - dT_0/d\ell_i, 0)^p\,d\ell_i`
                in (MN)^p.
        """
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
        r"""Compute the derivative of the Lp curve torque objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        coils_to_target and source_coils).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
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
        dJ_dgamma_targets = self.dJ_dgamma_targets(*args)
        dJ_dgammadash_targets = self.dJ_dgammadash_targets(*args)
        dJ_dgammadashdash_targets = self.dJ_dgammadashdash_targets(*args)
        dJ_dcurrent_targets = self.dJ_dcurrent_targets(*args)
        dJ_dgamma_sources = self.dJ_dgamma_sources(*args)
        dJ_dgammadash_sources = self.dJ_dgammadash_sources(*args)
        dJ_dcurrent_sources = self.dJ_dcurrent_sources(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent_targets[i]])) for i, c in enumerate(self.coils_to_target)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadashdash_by_dcoeff_vjp(dJ_dgammadashdash_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + vjp
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_sources[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_sources[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent_sources[i]])) for i, c in enumerate(self.source_coils)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}


def squared_mean_torque(
    gammas_targets, 
    gammas_sources, 
    gammadashs_targets, 
    gammadashs_sources, 
    currents_targets, 
    currents_sources, 
    downsample,
    eps=1e-10
):
    r"""
    Compute the squared mean torque on a set of m coils with n quadrature points 
    due to themselves and another set of m' coils with n' quadrature points.

    The objective function is

    .. math:
        J = \sum_i(\frac{\int \frac{d\vec{T}_i}{d\ell_i} d\ell_i}{L_i})^2
        
    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    in units of MN. The units of the squared mean torque are therefore (MN)^2.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length.

    Args:
        gammas_targets (array, shape (m,n,3)): Array of target coil positions.
        gammas_sources (array, shape (m',n,3)): Array of source coil positions.
        gammadashs_targets (array, shape (m,n,3)): Array of target coil tangent vectors.
        gammadashs_sources (array, shape (m',n,3)): Array of source coil tangent vectors.
        currents_targets (array, shape (m,)): Array of target coil currents.
        currents_sources (array, shape (m',)): Array of source coil currents.
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
        eps (float): Small constant to avoid division by zero for torque between coil_i and itself.
    Returns:
        float: Value of the objective function.
    """
    from simsopt.geo.curve import centroid_pure
    gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_targets, currents_sources = (
        _prepare_target_source_inputs_pure(
            gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources,
            currents_targets, currents_sources, downsample
        )
    )

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    centers = vmap(centroid_pure, in_axes=(0, 0))(gammas_targets, gammadashs_targets)

    def mean_torque_group1(i, gamma_i, gammadash_i, center_i, current_i):
        arclength = jnp.linalg.norm(gammadash_i, axis=-1)
        tangent = gammadash_i / arclength[:, None]
        B_mutual = vmap(
            lambda pt: _mutual_B_field_at_point_pure(
                i, pt,
                gammas_targets, gammadashs_targets, currents_targets,
                gammas_sources, gammadashs_sources, currents_sources,
                eps
            )
        )(gamma_i)
        F = _lorentz_force_density_pure(tangent, current_i, B_mutual)
        torques = jnp.cross(gamma_i - center_i[None, :], F) * arclength[:, None]
        return jnp.sum(torques, axis=0) / npts1

    mean_torques = vmap(mean_torque_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, gammadashs_targets, centers, currents_targets
    )
    # already multiplied by (mu_0/(4*pi)) in _mutual_B_field_at_point_pure, 
    # which gives a factor of (mu_0/(4*pi))^2 = 1e-14
    # Then convert from (N)^2 to (MN)^2 by dividing by (1e6)^2 = 1e12
    mean_torques_squared = jnp.sum(jnp.linalg.norm(mean_torques, axis=-1) ** 2)
    return mean_torques_squared * 1e-12


class SquaredMeanTorque(Optimizable):
    r"""
    Optimizable class to minimize the (net (integrated) Lorentz torque per unit length)^2 summed 
    over a set of m coils due to themselves and another set of m' coils.

    The objective function is

    .. math:
        J = \sum_i(\frac{\int \frac{d\vec{T}_i}{d\ell_i} d\ell_i}{L_i})^2
        
    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,  
    in units of MN. The units of the squared mean torque are therefore (MN)^2.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length.

    The units of the objective function are (MN)^2, where MN = meganewtons.
    
    This class assumes there are two distinct lists of coils,
    which may have different finite-build parameters. In order to avoid buildup of optimizable 
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. The two sets of coils may contain 
    coils with all different number of quadrature points and different types of cross-sections.

    Args:
        coils_to_target (list of Coil or RegularizedCoil, shape (m,)): List of coils to use for computing SquaredMeanTorque. 
        source_coils (list of Coil or RegularizedCoil, shape (m',)): List of coils that provide torques on the first set of coils but that
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
        _check_quadpoints_consistency(self.coils_to_target, "coils_to_target")
        if self.source_coils:
            _check_quadpoints_consistency(self.source_coils, "source_coils")
        self.downsample = downsample
        args = {"static_argnums": (6,)}

        self.J_jax = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            squared_mean_torque(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgamma_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=0)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgamma_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=1)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=2)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dgammadash_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=3)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dcurrent_targets = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=4)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        self.dJ_dcurrent_sources = jit(
            lambda gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample:
            grad(self.J_jax, argnums=5)(gammas_targets, gammas_sources, gammadashs_targets, gammadashs_sources, currents_targets, currents_sources, downsample),
            **args
        )

        super().__init__(depends_on=(coils_to_target + source_coils))

    def J(self):
        r"""Evaluate the squared mean torque objective.

        Returns:
            float: The sum of squared mean Lorentz torques
                :math:`\sum_i \left(\frac{\int (d\vec{T}_i/d\ell_i)\,d\ell_i}{L_i}\right)^2`
                in (MN)^2.
        """
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
        r"""Compute the derivative of the squared mean torque objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        coils_to_target and source_coils).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        args = [
            [c.curve.gamma() for c in self.coils_to_target],
            [c.curve.gamma() for c in self.source_coils],
            [c.curve.gammadash() for c in self.coils_to_target],
            [c.curve.gammadash() for c in self.source_coils],
            [c.current.get_value() for c in self.coils_to_target],
            [c.current.get_value() for c in self.source_coils],
            self.downsample,
        ]
        dJ_dgamma_targets = self.dJ_dgamma_targets(*args)
        dJ_dgammadash_targets = self.dJ_dgammadash_targets(*args)
        dJ_dcurrent_targets = self.dJ_dcurrent_targets(*args)
        dJ_dgamma_sources = self.dJ_dgamma_sources(*args)
        dJ_dgammadash_sources = self.dJ_dgammadash_sources(*args)
        dJ_dcurrent_sources = self.dJ_dcurrent_sources(*args)

        vjp = sum([c.current.vjp(jnp.asarray([dJ_dcurrent_targets[i]])) for i, c in enumerate(self.coils_to_target)])

        dJ = (
            sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_targets[i]) for i, c in enumerate(self.coils_to_target)])
            + sum([c.curve.dgamma_by_dcoeff_vjp(dJ_dgamma_sources[i]) for i, c in enumerate(self.source_coils)])
            + sum([c.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash_sources[i]) for i, c in enumerate(self.source_coils)])
            + vjp
            + sum([c.current.vjp(jnp.asarray([dJ_dcurrent_sources[i]])) for i, c in enumerate(self.source_coils)])
        )

        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}
