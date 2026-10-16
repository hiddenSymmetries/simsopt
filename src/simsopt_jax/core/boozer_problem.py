"""Native ``BoozerSurface`` formulations of the Boozer residual, in JAX.

For the state a :class:`BoozerProblem` holds and a decision vector ``x``:

- :func:`boozer_surface_residual` returns native ``boozer_surface_residual``;
- :func:`boozer_penalty_constraints` returns native
  ``BoozerSurface.boozer_penalty_constraints_vectorized`` (least squares plus
  the label and ``z(0, 0) = 0`` penalties);
- :func:`boozer_exact_constraints` returns native
  ``BoozerSurface.boozer_exact_constraints`` (the Lagrangian conditions);
- :func:`boozer_exact_residual` returns the BoozerExact system that native
  ``BoozerSurface.solve_residual_equation_exactly_newton`` solves, at the
  residual rows that
  :func:`simsopt_jax_adapters.geo.boozer_problem.boozer_exact_residual_rows`
  takes from the surface's native ``get_stellsym_mask()``;
- :func:`boozer_penalty_residual` returns the residuals and Jacobian of
  native ``BoozerSurface._get_residual_vector_and_jacobian``, whose half
  squared norm is the penalty (the least-squares solvers' formulation).

``x`` is ``[surface DOFs, iota, G]``, or ``[surface DOFs, iota]`` when
``optimize_G`` is false and ``G`` is native's constant from the coil currents;
the exact constraints append the two multipliers. Surface DOFs are the native
``get_dofs()`` vector, fixed DOFs included, and derivatives are taken with
respect to all of them (native's ``dgamma_by_dcoeff`` convention). Arguments
named as in native select the same outputs, with native's defaults; the two
penalty formulations take ``constraint_weight`` from the problem.

Every function is jitted with the problem and ``x`` as traced operands: new
DOF, ``iota``, ``G``, coil, target and weight values reuse the compiled
program, while ``derivatives``, ``optimize_G``, ``weight_inv_modB`` and the
problem's shapes, surface class and label kind select one. Build problems with
:func:`simsopt_jax_adapters.geo.boozer_problem.boozer_problem` and pass ``x``
already on the device. Second derivatives evaluate ``d2B/dXdX`` at every
quadrature point: call ``simsopt_jax.backend.set_backend("jax", device=...)``
first, whose point chunks bound that memory on large grids. Zero fields give native's non-finite weighted residuals. Where
``|B|^2`` underflows float64, weighted residuals can differ from native in
finiteness (XLA on CPU flushes subnormal results to zero), and the limits of
:mod:`simsopt_jax.core.surface_geometry` apply to the labels.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.pytree import pytree_dataclass

from ._math_utils import as_jax_float64
from .boozer_residual import BoozerPoints, boozer_least_squares, boozer_residual
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

__all__ = [
    "BoozerLabelSpec",
    "BoozerProblem",
    "boozer_exact_constraints",
    "boozer_exact_residual",
    "boozer_penalty_constraints",
    "boozer_penalty_residual",
    "boozer_surface_residual",
    "make_boozer_problem",
]

LabelKind = Literal["volume", "area", "aspect_ratio", "toroidal_flux"]


@pytree_dataclass(data=("surface",), meta=("kind", "idx"))
class BoozerLabelSpec:
    """A native label (``Volume``, ``Area``, ``AspectRatio``, ``ToroidalFlux``).

    ``surface`` is the spec of the label's own surface, whose quadrature grid
    may differ from the Boozer surface's but whose DOFs are the Boozer
    surface's. ``idx`` is ``ToroidalFlux.idx`` (``0`` otherwise).
    ``aspect_ratio`` evaluates the mean cross-sectional area in the closed form
    native uses for its derivatives, so it agrees with native wherever native's
    value (``det``/``inv`` of the cylindrical map) is defined and stays finite
    where that map is singular and native raises ``LinAlgError``.

    Args:
        surface (SurfaceSpec): Immutable surface coefficients and quadrature grid; all
            native DOFs are included.
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
    """The state of a native ``BoozerSurface`` that its formulations read.

    ``surface`` and ``coils`` are the Boozer surface's spec (its DOF values are
    replaced by those of ``x``) and the field's grouped coils, which also give
    a ``ToroidalFlux`` label its field. ``targetlabel`` is a float64 scalar.
    ``constraint_weight`` is the float64 penalty weight, which only the
    penalty formulation reads (``None`` if the problem has none, as a native
    BoozerExact ``BoozerSurface``).

    Args:
        surface (SurfaceSpec): Immutable surface coefficients and quadrature grid; all
            native DOFs are included.
        coils (GroupedCoilSetSpec): Grouped coil geometry in meters and currents in
            amperes.
        label (BoozerLabelSpec): Label surface sharing the Boozer surface's DOFs, with
            its own quadrature.
        targetlabel (jax.Array): Scalar shape () target in the label's units (m^3, m^2,
            Wb, or dimensionless).
        constraint_weight (jax.Array | None): Scalar shape () native penalty
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
    """A :class:`BoozerProblem` with its numbers placed as device operands.

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
        BoozerProblem: Immutable snapshot object with target and optional penalty weight placed
            as float64 device scalars.
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
    """Native's ``G`` when it is not a variable: ``mu0`` times the sum of ``|I|``
    (``0`` without coils).
    """
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


def _boozer_points(
    problem: BoozerProblem, surface_dofs: jax.Array, derivatives: int
) -> tuple[BoozerPoints, tuple[jax.Array, ...]]:
    """The surface and field at the quadrature points, and ``z(0, 0)`` with,
    for ``derivatives > 0``, its coefficient derivative.
    """

    def positions(dofs):
        spec = surface_spec_with_dofs(problem.surface, dofs)
        return surface_gamma(spec), surface_gammadash1(spec), surface_gammadash2(spec)

    gamma, xphi, xtheta = positions(surface_dofs)
    flat = gamma.reshape(-1, 3)
    if derivatives == 0:
        points = BoozerPoints(
            B=cast(jax.Array, grouped_biot_savart_B_from_spec(flat, problem.coils)),
            xphi=xphi.reshape(-1, 3),
            xtheta=xtheta.reshape(-1, 3),
        )
        return points, (gamma[0, 0, 2],)
    nsurface = surface_dofs.shape[0]
    dgamma, dxphi, dxtheta = jax.jacfwd(positions)(surface_dofs)
    B, dB = grouped_biot_savart_B_and_dB_from_spec(flat, problem.coils)
    points = BoozerPoints(
        B=B,
        xphi=xphi.reshape(-1, 3),
        xtheta=xtheta.reshape(-1, 3),
        dB_by_dX=dB,
        d2B_by_dXdX=(
            cast(jax.Array, grouped_biot_savart_d2B_by_dXdX_from_spec(flat, problem.coils))
            if derivatives == 2
            else None
        ),
        dgamma_by_dcoeff=dgamma.reshape(-1, 3, nsurface),
        dgammadash1_by_dcoeff=dxphi.reshape(-1, 3, nsurface),
        dgammadash2_by_dcoeff=dxtheta.reshape(-1, 3, nsurface),
    )
    return points, (gamma[0, 0, 2], dgamma[0, 0, 2])


def _aspect_ratio(spec: SurfaceSpec) -> jax.Array:
    """Native ``Surface.aspect_ratio()``: major over minor radius, from the
    volume and the mean cross-sectional area.
    """
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


def _label_derivatives(
    problem: BoozerProblem, surface_dofs: jax.Array, order: int
) -> tuple[jax.Array, ...]:
    """The label and, up to ``order``, its gradient and Hessian.

    """
    label = partial(_label_value, problem.label, problem.coils)
    if order == 0:
        return (label(surface_dofs),)
    value, gradient = jax.value_and_grad(label)(surface_dofs)
    if order == 1:
        return value, gradient
    return value, gradient, jax.hessian(label)(surface_dofs)


def _pad(vector: jax.Array, size: int) -> jax.Array:
    return jnp.pad(vector, (0, size - vector.shape[0]))


def _pad_square(matrix: jax.Array, size: int) -> jax.Array:
    return jnp.pad(matrix, ((0, size - matrix.shape[0]), (0, size - matrix.shape[1])))


def _weighted_constraints(
    problem: BoozerProblem, x: jax.Array, *, derivatives: int, optimize_G: bool
) -> tuple[jax.Array, jax.Array, BoozerPoints, tuple[tuple[jax.Array, ...], ...]]:
    """What the two penalty formulations share at ``x``: ``iota``, ``G``, the
    Boozer points and, per derivative order up to ``derivatives``, the terms
    of the weighted constraint residuals ``sqrt(w) (label - target)`` and
    ``sqrt(w) z(0, 0)``: their values, their gradients (length ``nx``) and the
    label's Hessian ``(nx, nx)`` alone (``z(0, 0)`` is linear in the DOFs)."""
    if problem.constraint_weight is None:
        raise ValueError("the penalty formulation needs the problem's constraint_weight.")
    surface_dofs, iota, G, _ = _split(problem, x, optimize_G=optimize_G, multipliers=0)
    points, z = _boozer_points(problem, surface_dofs, derivatives)
    label = _label_derivatives(problem, surface_dofs, derivatives)
    sqrt_weight = jnp.sqrt(problem.constraint_weight)
    nx = x.shape[0]
    constraints = ((sqrt_weight * (label[0] - problem.targetlabel), sqrt_weight * z[0]),)
    if derivatives > 0:
        constraints += ((sqrt_weight * _pad(label[1], nx), sqrt_weight * _pad(z[1], nx)),)
    if derivatives > 1:
        constraints += ((_pad_square(sqrt_weight * label[2], nx),),)
    return iota, G, points, constraints


_STATIC_OPTIONS = ("derivatives", "optimize_G", "weight_inv_modB")


@partial(jax.jit, static_argnames=_STATIC_OPTIONS)
def boozer_surface_residual(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool,
    weight_inv_modB: bool = False,
) -> tuple[jax.Array, ...]:
    """Native ``boozer_surface_residual``: ``(r,)``, ``(r, J)`` or ``(r, J, H)``.

    ``H`` holds every residual's second derivative, ``(nresiduals, nx, nx)``.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) vector [all surface DOFs in meters, dimensionless
            iota, optional G in tesla meters]; nx = nsurface + 1 + int(optimize_G).
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
    """
    surface_dofs, iota, G, _ = _split(problem, x, optimize_G=optimize_G, multipliers=0)
    points, _ = _boozer_points(problem, surface_dofs, derivatives)
    return boozer_residual(
        G,
        iota,
        points,
        derivatives=derivatives,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )


@partial(jax.jit, static_argnames=_STATIC_OPTIONS)
def boozer_penalty_constraints(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool = False,
    weight_inv_modB: bool = True,
) -> jax.Array | tuple[jax.Array, ...]:
    """Native ``boozer_penalty_constraints_vectorized`` with
    ``constraint_weight = problem.constraint_weight``: the value, ``(value,
    gradient)`` or ``(value, gradient, hessian)``.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) vector [all surface DOFs in meters, dimensionless
            iota, optional G in tesla meters]; nx = nsurface + 1 + int(optimize_G).
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
    """
    iota, G, points, constraints = _weighted_constraints(
        problem, x, derivatives=derivatives, optimize_G=optimize_G
    )
    nresiduals = 3 * points.B.shape[0]
    boozer = boozer_least_squares(
        G,
        iota,
        points,
        derivatives=derivatives,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )
    boozer = tuple(term / nresiduals for term in boozer)
    label_residual, z_residual = constraints[0]
    value = boozer[0] + 0.5 * label_residual**2 + 0.5 * z_residual**2
    if derivatives == 0:
        return value
    dlabel_residual, dz_residual = constraints[1]
    gradient = boozer[1] + label_residual * dlabel_residual + z_residual * dz_residual
    if derivatives == 1:
        return value, gradient
    hessian = (
        boozer[2]
        + jnp.outer(dlabel_residual, dlabel_residual)
        + jnp.outer(dz_residual, dz_residual)
        + label_residual * constraints[2][0]
    )
    return value, gradient, hessian


@partial(jax.jit, static_argnames=_STATIC_OPTIONS)
def boozer_penalty_residual(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool = False,
    weight_inv_modB: bool = True,
) -> tuple[jax.Array, ...]:
    """Native ``BoozerSurface._get_residual_vector_and_jacobian`` with
    ``constraint_weight = problem.constraint_weight``: ``(r,)`` or, for
    ``derivatives=1``, ``(r, J)``.

    ``r`` is the Boozer residual over the square root of its length, then
    ``sqrt(w) (label - target)`` and ``sqrt(w) z(0, 0)``; ``0.5 |r|^2`` is the
    penalty of :func:`boozer_penalty_constraints`.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) vector [all surface DOFs in meters, dimensionless
            iota, optional G in tesla meters]; nx = nsurface + 1 + int(optimize_G).
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
            sqrt(constraint_weight). Squaring and halving its norm gives
            boozer_penalty_constraints.
    """
    iota, G, points, constraints = _weighted_constraints(
        problem, x, derivatives=derivatives, optimize_G=optimize_G
    )
    boozer = boozer_residual(
        G,
        iota,
        points,
        derivatives=derivatives,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )
    scale = np.sqrt(boozer[0].shape[0])
    residual = jnp.concatenate((boozer[0] / scale, jnp.stack(constraints[0])))
    if derivatives == 0:
        return (residual,)
    return residual, jnp.concatenate((boozer[1] / scale, jnp.stack(constraints[1])))


@partial(jax.jit, static_argnames=("derivatives", "optimize_G"))
def boozer_exact_constraints(
    problem: BoozerProblem,
    xl: jax.Array,
    *,
    derivatives: int = 0,
    optimize_G: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Native ``boozer_exact_constraints``: ``res``, or ``(res, dres)``.

    ``xl`` is ``x`` followed by the multipliers of the label constraint and of
    ``z(0, 0) = 0``; the residual is not weighted.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        xl (jax.Array): Shape (nx + 2,) decision vector followed by the label and z
            Lagrange multipliers; x uses the full native surface DOFs, iota, and
            optional G.
        derivatives (int): 0 for the Lagrangian conditions or 1 to add their Jacobian.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.

    Returns:
        jax.Array | tuple[jax.Array, jax.Array]: Lagrangian stationarity and label/z
            constraints, shape (nx + 2,), optionally with Jacobian (nx + 2, nx + 2);
            rows have their native stationarity and constraint units.
    """
    surface_dofs, iota, G, multipliers = _split(
        problem, xl, optimize_G=optimize_G, multipliers=2
    )
    points, z = _boozer_points(problem, surface_dofs, derivatives + 1)
    boozer = boozer_least_squares(
        G,
        iota,
        points,
        derivatives=derivatives + 1,
        optimize_G=optimize_G,
        weight_inv_modB=False,
    )
    label = _label_derivatives(problem, surface_dofs, derivatives + 1)
    nx = xl.shape[0] - 2
    dlabel, dz = _pad(label[1], nx), _pad(z[1], nx)
    stationarity = boozer[1] - multipliers[0] * dlabel - multipliers[1] * dz
    res = jnp.concatenate((stationarity, jnp.stack((label[0] - problem.targetlabel, z[0]))))
    if derivatives == 0:
        return res
    hessian = boozer[2] - multipliers[0] * _pad_square(label[2], nx)
    constraints = jnp.stack((dlabel, dz))
    dres = jnp.block(
        [[hessian, -constraints.T], [constraints, jnp.zeros((2, 2), constraints.dtype)]]
    )
    return res, dres


@partial(jax.jit, static_argnames=("derivatives",))
def boozer_exact_residual(
    problem: BoozerProblem, x: jax.Array, residual_rows: jax.Array, *, derivatives: int = 0
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """The BoozerExact system ``b`` (or ``(b, J)``) of native
    ``solve_residual_equation_exactly_newton`` at ``x = [surface DOFs, iota, G]``.

    ``b`` is the unweighted residual at ``residual_rows`` (int32 indices,
    from :func:`simsopt_jax_adapters.geo.boozer_problem.boozer_exact_residual_rows`
    of the problem's surface), then ``label - target`` and, without stellarator
    symmetry, ``z(0, 0)``. As natively, only ``SurfaceXYZTensorFourier`` has it.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) vector [all surface DOFs in meters, dimensionless
            iota, G in tesla meters], nx = nsurface + 2.
        residual_rows (jax.Array): Shape (nrows,) integer indices of the flattened
            Boozer residual, in native mask order.
        derivatives (int): 0 for the system or 1 to add its Jacobian.

    Returns:
        jax.Array | tuple[jax.Array, jax.Array]: Unweighted selected residuals and label
            constraint, plus z without stellarator symmetry, shape (nb,), optionally
            Jacobian (nb, nx). Boozer rows have tesla squared meter units; constraint
            rows use label units and meters.
    """
    if not isinstance(problem.surface, SurfaceXYZTensorFourierSpec):
        raise RuntimeError(
            "Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier"
        )
    surface_dofs, iota, G, _ = _split(problem, x, optimize_G=True, multipliers=0)
    points, z = _boozer_points(problem, surface_dofs, derivatives)
    boozer = boozer_residual(
        G, iota, points, derivatives=derivatives, optimize_G=True, weight_inv_modB=False
    )
    label = _label_derivatives(problem, surface_dofs, derivatives)
    axis = not problem.surface.stellsym
    b = jnp.concatenate(
        (boozer[0][residual_rows], jnp.stack((label[0] - problem.targetlabel, z[0]))[: 1 + axis])
    )
    if derivatives == 0:
        return b
    tail = jnp.stack((_pad(label[1], x.shape[0]), _pad(z[1], x.shape[0])))[: 1 + axis]
    return b, jnp.concatenate((boozer[1][residual_rows], tail))
