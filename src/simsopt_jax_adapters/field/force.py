"""JAX coil force, torque, energy and flux objectives with native Optimizable APIs.

Geometry is placed on the active JAX device; values and gradients return to
the host. Native JAX curves may transfer implicitly during geometry evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.field.coil import RegularizedCoil
from simsopt.field.force import _check_downsample, _check_quadpoints_consistency
from simsopt_jax.backend.jit import jit
from simsopt_jax.core import coil_forces
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.backend.runtime import host_array, host_tree

from simsopt_jax_adapters.geo.curve_objectives import _host_float


def _stacked(coils, geometry) -> jax.Array:
    """Stack coil geometry on the host and place it on the active JAX device."""
    return _as_jax_float64(np.stack([geometry(coil.curve) for coil in coils]))


def _coil_group(coils) -> coil_forces.CoilGroup:
    return (
        _stacked(coils, lambda curve: curve.gamma()),
        _stacked(coils, lambda curve: curve.gammadash()),
        _as_jax_float64(np.asarray([coil.current.get_value() for coil in coils], dtype=np.float64)),
    )


def _coil_derivative(coils, dgammas, dgammadashs, dcurrents, dgammadashdashs=None) -> Derivative:
    """Project geometry and current cotangents of ``coils`` onto their DOFs."""
    total = Derivative()
    for index, coil in enumerate(coils):
        total += coil.curve.dgamma_by_dcoeff_vjp(dgammas[index])
        total += coil.curve.dgammadash_by_dcoeff_vjp(dgammadashs[index])
        if dgammadashdashs is not None:
            total += coil.curve.dgammadashdash_by_dcoeff_vjp(dgammadashdashs[index])
        total += coil.current.vjp(dcurrents[index:index + 1])
    return total


def _as_coil_list(coils) -> list:
    return coils if isinstance(coils, list) else [coils]


def _target_and_source_coils(target_coils, source_coils_coarse, source_coils_fine, downsample):
    """Return validated target and source lists with overlapping coils removed."""
    target_coils = _as_coil_list(target_coils)
    source_coils_coarse = _as_coil_list(source_coils_coarse)
    source_coils_fine = [] if source_coils_fine is None else _as_coil_list(source_coils_fine)
    coarse = [c for c in source_coils_coarse if c not in target_coils]
    fine = [c for c in source_coils_fine if c not in target_coils]
    if len(coarse) == 0 and len(fine) == 0:
        raise ValueError(
            "source_coils_coarse and source_coils_fine must together contain "
            "at least one coil not in target_coils."
        )
    fine = [c for c in fine if c not in coarse]
    for coils, label in ((target_coils, "target_coils"), (coarse, "source_coils_coarse"), (fine, "source_coils_fine")):
        if len(coils) > 0:
            _check_quadpoints_consistency(coils, label)
    for coils, label in ((target_coils, "target_coils"), (coarse, "source_coils_coarse"), (fine, "source_coils_fine")):
        if len(coils) > 0:
            _check_downsample(coils, downsample, label)
    return target_coils, coarse, fine


def _check_regularized(target_coils, objective_name) -> None:
    if not isinstance(target_coils[0], RegularizedCoil):
        raise ValueError(f"{objective_name} can only be used with RegularizedCoil objects")


def _regularizations(coils) -> np.ndarray:
    """Capture the coils' cross-section regularizations at construction."""
    return host_array([coil.regularization for coil in coils], dtype=np.float64)


class _CoilSetObjective(Optimizable):
    """Force and torque objective base with target and source lists read at evaluation.

    ``source_coils`` retains the combined source lists from construction.
    """

    _operands: Callable[[], tuple]
    _derivative: Callable[..., Derivative]
    _kernel: Callable[..., jax.Array]
    _kernel_grad: Callable[..., tuple]

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None, downsample: int = 1):
        self.target_coils, self.source_coils_coarse, self.source_coils_fine = (
            _target_and_source_coils(target_coils, source_coils_coarse, source_coils_fine, downsample)
        )
        self.source_coils = self.source_coils_coarse + self.source_coils_fine
        self.downsample = downsample
        super().__init__(depends_on=(self.target_coils + self.source_coils))

    def _source_lists(self) -> tuple[list, ...]:
        return tuple(coils for coils in (self.source_coils_coarse, self.source_coils_fine) if coils)

    def _source_groups(self) -> tuple[coil_forces.CoilGroup, ...]:
        return tuple(_coil_group(coils) for coils in self._source_lists())

    def _source_derivative(self, dsources) -> Derivative:
        total = Derivative()
        for coils, cotangents in zip(self._source_lists(), dsources, strict=True):
            total += _coil_derivative(coils, *cotangents)
        return total

    def J(self):
        """Return the objective value.

        Returns:
            float: objective value with units given by the class contract.
        """
        return _host_float(self._kernel(*self._operands(), downsample=self.downsample))

    @derivative_dec
    def dJ(self):
        """Return the coil DOF gradient, accumulating shared DOFs.

        ``partials=True`` includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return self._derivative(self._kernel_grad)

    return_fn_map = {"J": J, "dJ": dJ}


# One compiled program per objective, shapes, downsample and source-group count.
_lp_force = jit(coil_forces.lp_force, static_argnames=("downsample",))
_lp_force_grad = jit(
    jax.grad(coil_forces.lp_force, argnums=(0, 1, 4)), static_argnames=("downsample",)
)
_lp_torque = jit(coil_forces.lp_torque, static_argnames=("downsample",))
_lp_torque_grad = jit(
    jax.grad(coil_forces.lp_torque, argnums=(0, 1, 4)), static_argnames=("downsample",)
)
_squared_mean_force = jit(coil_forces.squared_mean_force, static_argnames=("downsample",))
_squared_mean_force_grad = jit(
    jax.grad(coil_forces.squared_mean_force, argnums=(0, 1)), static_argnames=("downsample",)
)
_squared_mean_torque = jit(coil_forces.squared_mean_torque, static_argnames=("downsample",))
_squared_mean_torque_grad = jit(
    jax.grad(coil_forces.squared_mean_torque, argnums=(0, 1)), static_argnames=("downsample",)
)
_b2energy = jit(coil_forces.b2energy, static_argnames=("downsample",))
_b2energy_grad = jit(
    jax.grad(coil_forces.b2energy, argnums=(0, 1, 2)), static_argnames=("downsample",)
)
_net_flux = jit(coil_forces.net_flux, static_argnames=("downsample",))
_net_flux_grad = jit(
    jax.grad(coil_forces.net_flux, argnums=(0, 1, 2)), static_argnames=("downsample",)
)


class _LpObjective(_CoilSetObjective):
    """Shared evaluation of :class:`JaxLpCurveForce` and :class:`JaxLpCurveTorque`."""

    _native_name: str

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None,
                 p: float = 2.0, threshold: float = 0.0, downsample: int = 1):
        target_coils = _as_coil_list(target_coils)
        _check_regularized(target_coils, self._native_name)
        self._regularizations = _regularizations(target_coils)
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, downsample)
        self._quadpoints = np.asarray(self.target_coils[0].curve.quadpoints, dtype=np.float64)
        self._p = np.float64(p)
        self._threshold = np.float64(threshold)

    @property
    def p(self) -> float:
        """The exponent fixed at construction.

        Returns:
            float: dimensionless exponent.
        """
        return float(self._p)

    @property
    def threshold(self) -> float:
        """The threshold fixed at construction.

        Returns:
            float: threshold in MN/m (force) or MN (torque).
        """
        return float(self._threshold)

    @property
    def regularizations(self) -> np.ndarray:
        """The target coils' cross-section regularizations captured at construction.

        Returns:
            Array of shape (ntargets,): a copy, in m^2.
        """
        return self._regularizations.copy()

    def _operands(self):
        return (
            _coil_group(self.target_coils),
            _stacked(self.target_coils, lambda curve: curve.gammadashdash()),
            _as_jax_float64(self._quadpoints),
            _as_jax_float64(self._regularizations),
            self._source_groups(),
            _as_jax_float64(self._p),
            _as_jax_float64(self._threshold),
        )

    def _derivative(self, gradient) -> Derivative:
        (dgammas, dgammadashs, dcurrents), dgammadashdashs, dsources = host_tree(
            gradient(*self._operands(), downsample=self.downsample), dtype=np.float64
        )
        return _coil_derivative(
            self.target_coils, dgammas, dgammadashs, dcurrents, dgammadashdashs
        ) + self._source_derivative(dsources)


class JaxLpCurveForce(_LpObjective):
    r"""Penalize coil force per unit length above a threshold.

    The objective is 1/p times the summed arclength integral of force-density
    excess to the p-th power, in (MN/m)^p m, without a p-th root. Fields include
    regularized self fields, other target coils and sources. Quadrature uses
    the unit-period curve parameter. Target regularizations are fixed at
    construction; coil lists and ``downsample`` are read at each evaluation.

    Args:
        target_coils (Coil or list of Coil): RegularizedCoil objects on which the force is computed. Coils that
            also appear in a source list are removed from that list.
        source_coils_coarse (Coil or list of Coil): external source coils with one shared quadrature count.
        source_coils_fine (Coil, list of Coil or None): optional second source list with its own quadrature
            count, e.g. finely resolved TF coils next to coarse dipole coils. Default: None.
        p (float): dimensionless exponent, fixed at construction. Default: 2.0.
        threshold (float): threshold force per unit length in MN/m, fixed at construction. Default: 0.0.
        downsample (int): stride over the quadrature points of every coil list; it must divide every list's
            quadrature count. Default: 1.
    """

    _native_name = "LpCurveForce"
    _kernel = staticmethod(_lp_force)
    _kernel_grad = staticmethod(_lp_force_grad)


class JaxLpCurveTorque(_LpObjective):
    r"""Penalize coil torque per unit length above a threshold.

    The objective is 1/p times the summed arclength integral of torque-density
    excess to the p-th power, in MN^p m, without a p-th root. Torque is about
    each target's arclength centroid, using the fields of :class:`JaxLpCurveForce`.
    Quadrature uses the unit-period curve parameter. Target regularizations
    are fixed at construction; coil lists and ``downsample`` are read at each
    evaluation.

    Args:
        target_coils (Coil or list of Coil): RegularizedCoil objects on which the torque is computed. Coils that
            also appear in a source list are removed from that list.
        source_coils_coarse (Coil or list of Coil): external source coils with one shared quadrature count.
        source_coils_fine (Coil, list of Coil or None): optional second source list with its own quadrature
            count, e.g. finely resolved TF coils next to coarse dipole coils. Default: None.
        p (float): dimensionless exponent, fixed at construction. Default: 2.0.
        threshold (float): threshold torque per unit length in MN, fixed at construction. Default: 0.0.
        downsample (int): stride over the quadrature points of every coil list; it must divide every list's
            quadrature count. Default: 1.
    """

    _native_name = "LpCurveTorque"
    _kernel = staticmethod(_lp_torque)
    _kernel_grad = staticmethod(_lp_torque_grad)


class _SquaredMeanObjective(_CoilSetObjective):
    """Shared evaluation of the squared mean force and torque objectives."""

    def _operands(self):
        return _coil_group(self.target_coils), self._source_groups()

    def _derivative(self, gradient) -> Derivative:
        dtargets, dsources = host_tree(
            gradient(*self._operands(), downsample=self.downsample), dtype=np.float64
        )
        return _coil_derivative(self.target_coils, *dtargets) + self._source_derivative(dsources)


class JaxSquaredMeanForce(_SquaredMeanObjective):
    r"""Sum squared net Lorentz forces on target coils, in MN^2.

    Integrate force density over arclength using unit-period quadrature,
    without coil-length normalization. Fields include other targets and all
    sources, excluding self fields. Coil lists and ``downsample`` are read at
    each evaluation.

    Args:
        target_coils (Coil or list of Coil): coils on which the net force is computed. Coils
            that also appear in a source list are removed from that list.
        source_coils_coarse (Coil or list of Coil): external source coils with one shared quadrature count.
        source_coils_fine (Coil, list of Coil or None): optional second source list with its own quadrature
            count, e.g. finely resolved TF coils next to coarse dipole coils. Default: None.
        downsample (int): stride over the quadrature points of every coil list; it must divide every list's
            quadrature count. Default: 1.
    """

    _kernel = staticmethod(_squared_mean_force)
    _kernel_grad = staticmethod(_squared_mean_force_grad)


class JaxSquaredMeanTorque(_SquaredMeanObjective):
    r"""Sum squared net torques about target arclength centroids, in (MN m)^2.

    Integrate torque density over arclength using unit-period quadrature,
    without coil-length normalization. Fields include other targets and all
    sources, excluding self fields. Coil lists and ``downsample`` are read at
    each evaluation.

    Args:
        target_coils (Coil or list of Coil): coils on which the net torque is computed. Coils
            that also appear in a source list are removed from that list.
        source_coils_coarse (Coil or list of Coil): external source coils with one shared quadrature count.
        source_coils_fine (Coil, list of Coil or None): optional second source list with its own quadrature
            count, e.g. finely resolved TF coils next to coarse dipole coils. Default: None.
        downsample (int): stride over the quadrature points of every coil list; it must divide every list's
            quadrature count. Default: 1.
    """

    _kernel = staticmethod(_squared_mean_torque)
    _kernel_grad = staticmethod(_squared_mean_torque_grad)


class JaxB2Energy(Optimizable):
    r"""Compute the vacuum magnetic field energy of a coil set in MJ.

    Include regularized self-inductance for each coil's finite cross section.
    Regularizations are fixed at construction; ``target_coils`` and
    ``downsample`` are read at each evaluation.

    Args:
        target_coils (list of RegularizedCoil, shape (m,)): coils contributing to the energy, with a common
            quadrature count.
        downsample (int): stride over the quadrature points; it must divide the quadrature count. Default: 1.
    """

    def __init__(self, target_coils, downsample=1):
        self.target_coils = target_coils
        self.downsample = downsample
        _check_regularized(target_coils, "B2Energy")
        _check_quadpoints_consistency(self.target_coils, "target_coils")
        _check_downsample(self.target_coils, downsample, "target_coils")
        self._regularizations = _regularizations(target_coils)
        super().__init__(depends_on=target_coils)

    @property
    def regularizations(self) -> np.ndarray:
        """The coils' cross-section regularizations captured at construction.

        Returns:
            Array of shape (ncoils,): a copy, in m^2.
        """
        return self._regularizations.copy()

    def _operands(self):
        return (*_coil_group(self.target_coils), _as_jax_float64(self._regularizations))

    def J(self):
        """Return the objective value.

        Returns:
            float: objective value with units given by the class contract.
        """
        return _host_float(_b2energy(*self._operands(), downsample=self.downsample))

    @derivative_dec
    def dJ(self):
        """Return the coil DOF gradient, accumulating shared DOFs.

        ``partials=True`` includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        cotangents = host_tree(_b2energy_grad(*self._operands(), downsample=self.downsample), dtype=np.float64)
        return _coil_derivative(self.target_coils, *cotangents)

    return_fn_map = {"J": J, "dJ": dJ}


class JaxNetFluxes(Optimizable):
    r"""Compute source-coil flux through a target coil in Wb.

    Approximate the vector-potential line integral using unit-period target
    quadrature sampled with ``downsample``; sources retain full quadrature.
    Sources are fixed at construction, so editing or reassigning
    ``source_coils`` has no effect. ``target_coil`` and ``downsample`` are read
    at each evaluation.

    Args:
        target_coil (Coil): coil whose net flux is computed; its own current does not enter.
        source_coils (Coil or list of Coil): source coils with one shared quadrature count; the target is removed.
        downsample (int): stride over the target's quadrature points; it must divide the quadrature count of
            the target and of the sources. Default: 1.
    """

    def __init__(self, target_coil, source_coils, downsample=1):
        source_coils = _as_coil_list(source_coils)
        self.target_coil = target_coil
        self.source_coils = [c for c in source_coils if c not in [target_coil]]
        if len(self.source_coils) == 0:
            raise ValueError("source_coils must contain at least one coil not in target_coil.")
        self.downsample = downsample
        _check_downsample([self.target_coil], downsample, "target_coil")
        _check_quadpoints_consistency(self.source_coils, "source_coils")
        _check_downsample(self.source_coils, downsample, "source_coils")
        self._sources = tuple(self.source_coils)
        super().__init__(depends_on=[target_coil] + source_coils)

    def _operands(self):
        curve = self.target_coil.curve
        return (
            _as_jax_float64(curve.gamma()),
            _as_jax_float64(curve.gammadash()),
            _coil_group(self._sources),
        )

    def J(self):
        """Return the objective value.

        Returns:
            float: objective value with units given by the class contract.
        """
        return _host_float(_net_flux(*self._operands(), downsample=self.downsample))

    @derivative_dec
    def dJ(self):
        """Return the coil DOF gradient, accumulating shared DOFs.

        ``partials=True`` includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        dgamma, dgammadash, dsources = host_tree(
            _net_flux_grad(*self._operands(), downsample=self.downsample), dtype=np.float64
        )
        curve = self.target_coil.curve
        return (
            curve.dgamma_by_dcoeff_vjp(dgamma)
            + curve.dgammadash_by_dcoeff_vjp(dgammadash)
            + _coil_derivative(self._sources, *dsources)
        )

    return_fn_map = {"J": J, "dJ": dJ}
