"""JAX Boozer surface residuals, penalties, and exact constraints.

Inputs use full surface DOFs and float64 on the intended device.
Zero or underflowing fields can yield non-finite weighted residuals.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.jit import jit
from simsopt_jax.pytree import pytree_dataclass

from ._math_utils import as_jax_float64
from .boozer_residual import (
    _require_derivative_order,
    boozer_point_residual,
    objective_derivatives,
    residual_derivatives,
    taylor_field,
)
from .field import (
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_d2B_by_dXdX_from_spec,
)
from .specs import GroupedCoilSetSpec, SurfaceSpec, SurfaceXYZTensorFourierSpec
from .surface_fourier_series import surface_get_dofs, surface_spec_with_dofs
from .surface_geometry import (
    surface_area,
    surface_gamma,
    surface_gammadash1,
    surface_gammadash2,
    surface_volume,
)

LabelKind = Literal["volume", "area", "aspect_ratio", "toroidal_flux"]


@pytree_dataclass(data=("surface",), meta=("kind", "idx"))
class BoozerLabelSpec:
    """Describe a volume, area, aspect-ratio, or toroidal-flux label.

    The label may use a different grid but must share the Boozer surface's DOFs.
    Aspect ratio remains finite for singular cylindrical maps.

    Args:
        surface (SurfaceSpec): Immutable surface coefficients and label quadrature grid;
            includes all native DOFs.
        kind (str): One of volume, area, aspect_ratio, or toroidal_flux.
        idx (int): ToroidalFlux quadrature row index; zero for the other labels.
    """

    surface: SurfaceSpec
    kind: LabelKind
    idx: int


@pytree_dataclass(
    data=(
        "surface",
        "coils",
        "label",
        "targetlabel",
        "constraint_weight",
    )
)
class BoozerProblem:
    """Hold the immutable surface, coil, and label state for Boozer formulations.

    Args:
        surface (SurfaceSpec): Immutable surface coefficients and quadrature grid;
            coefficients are replaced by the decision vector's full native DOFs.
        coils (GroupedCoilSetSpec): Grouped coil geometry in meters and currents in
            amperes.
        label (BoozerLabelSpec): Label surface sharing the Boozer surface's DOFs, with
            its own quadrature.
        targetlabel (jax.Array): Float64 scalar shape () target in label units
            (m^3, m^2, Wb, or dimensionless).
        constraint_weight (jax.Array | None): Float64 scalar shape () penalty
            coefficient, or None for an exact problem.
    """

    surface: SurfaceSpec
    coils: GroupedCoilSetSpec
    label: BoozerLabelSpec
    targetlabel: jax.Array
    constraint_weight: jax.Array | None


def make_boozer_problem(
    *,
    surface: SurfaceSpec,
    coils: GroupedCoilSetSpec,
    label: BoozerLabelSpec,
    targetlabel: float,
    constraint_weight: float | None,
) -> BoozerProblem:
    """Create a Boozer problem with float64 target and weight on the active device.

    Args:
        surface (SurfaceSpec): Immutable surface coefficients and quadrature grid; all
            native DOFs are included.
        coils (GroupedCoilSetSpec): Grouped coil geometry in meters and currents in
            amperes.
        label (BoozerLabelSpec): Label convention and quadrature snapshot.
        targetlabel (float): Target label in cubic meters (volume), square meters
            (area), webers (toroidal flux), or dimensionless (aspect ratio).
        constraint_weight (float | None): Native penalty coefficient for the squared
            label and z constraints; None for an exact problem.

    Returns:
        BoozerProblem: Immutable snapshot with target and optional weight as float64
            device scalars.
    """
    return BoozerProblem(
        surface=surface,
        coils=coils,
        label=label,
        targetlabel=as_jax_float64(np.float64(targetlabel)),
        constraint_weight=(
            None if constraint_weight is None else as_jax_float64(np.float64(constraint_weight))
        ),
    )


def _G_from_coil_currents(coils: GroupedCoilSetSpec) -> jax.Array:
    """Return mu0 times the sum of absolute coil currents, or zero without coils."""
    no_currents = jnp.zeros(0, jnp.float64)
    currents = jnp.concatenate([no_currents, *(group.currents for group in coils.groups)])
    return 2.0 * np.pi * jnp.sum(jnp.abs(currents)) * (4 * np.pi * 10 ** (-7) / (2 * np.pi))


def _split(problem: BoozerProblem, x: jax.Array, *, optimize_G: bool, multipliers: int):
    nsurface = surface_get_dofs(problem.surface).shape[0]
    expected = nsurface + 1 + int(optimize_G) + multipliers
    if x.shape != (expected,):
        raise ValueError(
            f"expected a decision vector of shape ({expected},) for {nsurface} surface "
            f"DOFs, iota{', G' if optimize_G else ''}"
            f"{' and the two multipliers' if multipliers else ''}; got {x.shape}."
        )
    G = x[nsurface + 1] if optimize_G else _G_from_coil_currents(problem.coils)
    return x[:nsurface], x[nsurface], G, x[expected - multipliers :]


def _field(coils: GroupedCoilSetSpec, order: int) -> Callable[[jax.Array], jax.Array]:
    """Return a field evaluator using a Taylor model for positive order."""
    if order == 0:
        return lambda points: cast(jax.Array, grouped_biot_savart_B_from_spec(points, coils))

    def coefficients_at(expansion):
        coefficients = tuple(grouped_biot_savart_B_and_dB_from_spec(expansion, coils))
        if order == 2:
            coefficients += (
                cast(jax.Array, grouped_biot_savart_d2B_by_dXdX_from_spec(expansion, coils)),
            )
        return coefficients

    return partial(taylor_field, coefficients_at)


def _boozer_terms(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    optimize_G: bool,
    multipliers: int,
    order: int,
    weight_inv_modB: bool,
) -> tuple[jax.Array, jax.Array]:
    """Return the Boozer residual, shape (3 * npoints,), and z(0, 0) at x."""
    surface_dofs, iota, G, _ = _split(problem, x, optimize_G=optimize_G, multipliers=multipliers)
    spec = surface_spec_with_dofs(problem.surface, surface_dofs)
    gamma, xphi, xtheta = surface_gamma(spec), surface_gammadash1(spec), surface_gammadash2(spec)
    B = _field(problem.coils, order)(gamma.reshape(-1, 3))
    residual = boozer_point_residual(
        G, iota, B, xphi.reshape(-1, 3), xtheta.reshape(-1, 3), weight_inv_modB=weight_inv_modB
    )
    return residual, gamma[0, 0, 2]


def _aspect_ratio(spec: SurfaceSpec) -> jax.Array:
    """Return the major-to-minor radius ratio from volume and mean cross-section."""
    gamma, xphi, xtheta = surface_gamma(spec), surface_gammadash1(spec), surface_gammadash2(spec)
    x, y = gamma[..., 0], gamma[..., 1]
    radius = jnp.sqrt(x * x + y * y)
    section = (
        xtheta[..., 2] * (x * xphi[..., 1] - y * xphi[..., 0])
        - xphi[..., 2] * (x * xtheta[..., 1] - y * xtheta[..., 0])
    ) / radius
    mean_area = jnp.abs(jnp.mean(section)) / (2.0 * np.pi)
    minor_radius = jnp.sqrt(mean_area / np.pi)
    major_radius = jnp.abs(surface_volume(spec)) / (2.0 * np.pi**2 * minor_radius**2)
    return major_radius / minor_radius


def _label_value(label: BoozerLabelSpec, coils: GroupedCoilSetSpec, surface_dofs) -> jax.Array:
    spec = surface_spec_with_dofs(label.surface, surface_dofs)
    if label.kind == "volume":
        return surface_volume(spec)
    if label.kind == "area":
        return surface_area(spec)
    if label.kind == "aspect_ratio":
        return _aspect_ratio(spec)
    # ToroidalFlux: the line integral of A along gamma(idx, :).
    gamma = surface_gamma(spec)
    potential = grouped_biot_savart_A_from_spec(gamma[label.idx], coils)
    return jnp.sum(potential * surface_gammadash2(spec)[label.idx]) / gamma.shape[1]


def _label_residual(problem: BoozerProblem, x: jax.Array) -> jax.Array:
    """Return the label error at the full surface DOFs in x."""
    nsurface = surface_get_dofs(problem.surface).shape[0]
    return _label_value(problem.label, problem.coils, x[:nsurface]) - problem.targetlabel


def _penalty_terms(
    problem: BoozerProblem, x: jax.Array, *, optimize_G: bool, weight_inv_modB: bool, order: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the Boozer residual and square-root-weighted label and z constraints."""
    if problem.constraint_weight is None:
        raise ValueError("the penalty formulation needs the problem's constraint_weight.")
    residual, z = _boozer_terms(
        problem,
        x,
        optimize_G=optimize_G,
        multipliers=0,
        order=order,
        weight_inv_modB=weight_inv_modB,
    )
    sqrt_weight = jnp.sqrt(problem.constraint_weight)
    return residual, sqrt_weight * _label_residual(problem, x), sqrt_weight * z


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _not_differentiable_in_x(x: jax.Array, refusal: str) -> jax.Array:
    """Return x unchanged, raising TypeError if any transform differentiates it."""
    return x


@_not_differentiable_in_x.defjvp
def _refuse_derivatives_in_x(refusal, primals, tangents):
    raise TypeError(refusal)


def _decision_vector(x: jax.Array, formulation: str, derivatives: int) -> jax.Array:
    """Cast x to float64 and reject differentiation for positive derivative orders."""
    x = as_jax_float64(x)
    if not derivatives:
        return x
    return _not_differentiable_in_x(
        x,
        f"{formulation}(derivatives={derivatives}) is not differentiable in x: its outputs "
        f"use the field's Taylor model of order {derivatives}. Differentiate the "
        "derivatives=0 value instead; coil derivatives at a fixed x are exact.",
    )


_STATIC_OPTIONS = ("derivatives", "optimize_G", "weight_inv_modB")


@partial(jit, static_argnames=_STATIC_OPTIONS)
def boozer_surface_residual(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool,
    weight_inv_modB: bool = False,
) -> tuple[jax.Array, ...]:
    """Evaluate Boozer residuals and optional first and second derivatives.

    Derivative outputs support coil derivatives at fixed ``x`` only. For large
    grids, configure backend point chunks before requesting second derivatives.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and optional
            G (T m), cast to float64; nx = nsurface + 1 + int(optimize_G).
        derivatives (int): Derivative order, 0 for values, 1 to add first derivatives,
            or 2 to add second derivatives.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        tuple[jax.Array, ...]: Residual shape (3 * npoints,), optionally Jacobian (3 *
            npoints, nx) and residual Hessians (3 * npoints, nx, nx). Residual units are
            tesla squared meters, or tesla meters when weighted; derivative units follow
            x.

    Raises:
        ValueError: If ``derivatives`` is not 0, 1 or 2.
        TypeError: If a transform traces ``x`` for ``derivatives >= 1``, even with
            a zero tangent.
    """

    x = _decision_vector(x, "boozer_surface_residual", derivatives)

    def residual(point):
        return _boozer_terms(
            problem,
            point,
            optimize_G=optimize_G,
            multipliers=0,
            order=derivatives,
            weight_inv_modB=weight_inv_modB,
        )[0]

    return residual_derivatives(residual, x, derivatives)


@partial(jit, static_argnames=_STATIC_OPTIONS)
def boozer_penalty_constraints(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool = False,
    weight_inv_modB: bool = True,
) -> jax.Array | tuple[jax.Array, ...]:
    """Evaluate the Boozer penalty with the problem's constraint weight.

    For large grids, configure backend point chunks before second derivatives.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and optional
            G (T m), cast to float64; nx = nsurface + 1 + int(optimize_G).
        derivatives (int): Derivative order, 0 for values, 1 to add first derivatives,
            or 2 to add second derivatives.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        jax.Array | tuple[jax.Array, ...]: Scalar shape () least-squares objective plus
            weighted constraints, or (value, gradient) or (value, gradient, Hessian),
            with shapes (), (nx,), (nx, nx). The Boozer term is divided by 3 * npoints;
            units follow the residual and native constraint coefficient.

    Raises:
        ValueError: If ``derivatives`` is not 0, 1 or 2, or the problem has no
            ``constraint_weight``.
        TypeError: If a transform traces ``x`` for ``derivatives >= 1``, even with
            a zero tangent.
    """

    x = _decision_vector(x, "boozer_penalty_constraints", derivatives)

    def penalty(point):
        residual, label_residual, z_residual = _penalty_terms(
            problem,
            point,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            order=derivatives,
        )
        boozer = 0.5 * jnp.sum(residual * residual) / residual.shape[0]
        return boozer + 0.5 * label_residual**2 + 0.5 * z_residual**2

    result = objective_derivatives(penalty, x, derivatives)
    return result[0] if derivatives == 0 else result


@partial(jit, static_argnames=_STATIC_OPTIONS)
def boozer_penalty_residual(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool = False,
    weight_inv_modB: bool = True,
) -> tuple[jax.Array, ...]:
    """Evaluate normalized penalty residuals and optional derivatives.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and optional
            G (T m), cast to float64; nx = nsurface + 1 + int(optimize_G).
        derivatives (int): Derivative order, 0 for values, 1 to add first derivatives,
            or 2 to add second derivatives.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        tuple[jax.Array, ...]: Residual shape (nr,), optionally Jacobian (nr, nx) and
            Hessians (nr, nx, nx), where nr = 3 * npoints + 2. Boozer entries are
            divided by sqrt(3 * npoints); the label and z entries are multiplied by
            sqrt(constraint_weight). Half the squared norm equals
            :func:`boozer_penalty_constraints`.

    Raises:
        ValueError: If ``derivatives`` is not 0, 1 or 2, or the problem has no
            ``constraint_weight``.
        TypeError: If a transform traces ``x`` for ``derivatives >= 1``, even with
            a zero tangent.
    """

    x = _decision_vector(x, "boozer_penalty_residual", derivatives)

    def residuals(point):
        residual, label_residual, z_residual = _penalty_terms(
            problem,
            point,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            order=derivatives,
        )
        scale = np.sqrt(residual.shape[0])
        return jnp.concatenate((residual / scale, jnp.stack((label_residual, z_residual))))

    return residual_derivatives(residuals, x, derivatives)


@partial(jit, static_argnames=("derivatives", "optimize_G"))
def boozer_exact_constraints(
    problem: BoozerProblem,
    xl: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Evaluate unweighted Lagrangian stationarity and label/z constraints.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        xl (jax.Array): Shape (nx + 2,) full native surface DOFs, iota, optional G,
            then label and z multipliers; cast to float64.
        derivatives (int): 0 for the Lagrangian conditions or 1 to add their Jacobian.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.

    Returns:
        jax.Array | tuple[jax.Array, jax.Array]: Lagrangian stationarity and label/z
            constraints, shape (nx + 2,), optionally with Jacobian (nx + 2, nx + 2);
            rows have their native stationarity and constraint units.

    Raises:
        ValueError: If ``derivatives`` is not 0 or 1.
        TypeError: If a transform traces ``xl`` for either derivative order, even
            with a zero tangent.
    """
    _require_derivative_order(derivatives, 1)
    xl = as_jax_float64(xl)
    nx = xl.shape[0] - 2

    def lagrangian(x, multipliers):
        residual, z = _boozer_terms(
            problem,
            jnp.concatenate((x, multipliers)),
            optimize_G=optimize_G,
            multipliers=2,
            order=derivatives + 1,
            weight_inv_modB=False,
        )
        return (
            0.5 * jnp.sum(residual * residual)
            - multipliers[0] * _label_residual(problem, x)
            - multipliers[1] * z
        )

    def conditions(point):
        stationarity, constraints = jax.grad(lagrangian, argnums=(0, 1))(point[:nx], point[nx:])
        return jnp.concatenate((stationarity, -constraints))

    jacobian_hint = "; request their Jacobian with derivatives=1" if derivatives == 0 else ""
    xl = _not_differentiable_in_x(
        xl,
        f"boozer_exact_constraints(derivatives={derivatives}) is not differentiable in xl: "
        "its conditions are a gradient evaluated with the field's Taylor model of order "
        f"{derivatives + 1}{jacobian_hint}. Coil derivatives at a fixed xl are exact; total "
        "derivatives through a surface that depends on the coils are not supported.",
    )
    if derivatives == 0:
        return conditions(xl)
    value, jacobian = residual_derivatives(conditions, xl, derivatives)
    return value, jacobian


@partial(jit, static_argnames=("derivatives",))
def boozer_exact_residual(
    problem: BoozerProblem, x: jax.Array, residual_rows: jax.Array, *, derivatives: int = 0
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Evaluate the masked, unweighted BoozerExact system and optional Jacobian.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and G (T m),
            cast to float64; nx = nsurface + 2.
        residual_rows (jax.Array): Shape (nrows,) int32 indices of the flattened
            Boozer residual, in native mask order.
        derivatives (int): 0 for the system or 1 to add its Jacobian.

    Returns:
        jax.Array | tuple[jax.Array, jax.Array]: Unweighted selected residuals and label
            constraint, plus z without stellarator symmetry, shape (nb,), optionally
            Jacobian (nb, nx). Boozer rows have tesla squared meter units; constraint
            rows use label units and meters.

    Raises:
        ValueError: If ``derivatives`` is not 0 or 1.
        RuntimeError: If the problem's surface is not a ``SurfaceXYZTensorFourier``.
        TypeError: If a transform traces ``x`` for ``derivatives=1``, even with
            a zero tangent.
    """
    _require_derivative_order(derivatives, 1)
    if not isinstance(problem.surface, SurfaceXYZTensorFourierSpec):
        raise RuntimeError(
            "Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier"
        )
    x = _decision_vector(x, "boozer_exact_residual", derivatives)
    axis = not problem.surface.stellsym

    def system(point):
        residual, z = _boozer_terms(
            problem,
            point,
            optimize_G=True,
            multipliers=0,
            order=derivatives,
            weight_inv_modB=False,
        )
        constraints = jnp.stack((_label_residual(problem, point), z))[: 1 + axis]
        return jnp.concatenate((residual[residual_rows], constraints))

    if derivatives == 0:
        return system(x)
    value, jacobian = residual_derivatives(system, x, derivatives)
    return value, jacobian
