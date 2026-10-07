"""JAX coil force, torque and energy objectives as drop-in native Optimizables.

Each class mirrors the objective of the same name without ``JAX`` in
:mod:`simsopt.field.force`: same constructor arguments and validation, value,
dependencies and ``Derivative`` (fixed and free partials of the coils' curves
and currents). The objective and its gradient with respect to the coils'
geometry and currents run as one jitted JAX program, where native jits one
gradient program per argument; the gradient is projected through the curves'
own coefficient VJPs and the currents' ``vjp``, as native does. Coil geometry
moves to the active JAX device and results back through explicit transfers, so
for C++ curves (and their rotated copies) J and dJ make no implicit transfer;
JAX-backed native curves (``JaxCurve`` subclasses, filaments) still transfer
implicitly inside their own geometry. As in native, ``p``, ``threshold``, the
target coils' regularizations and the sources of :class:`NetFluxesJAX` are
fixed at construction, while ``downsample``, ``target_coil`` and the force and
torque objectives' coil lists are attributes read at every evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.field.coil import RegularizedCoil
from simsopt.field.force import _check_downsample, _check_quadpoints_consistency
from simsopt_jax.core import coil_forces
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.runtime.host_boundary import host_array, host_tree

__all__ = [
    "B2EnergyJAX",
    "LpCurveForceJAX",
    "LpCurveTorqueJAX",
    "NetFluxesJAX",
    "SquaredMeanForceJAX",
    "SquaredMeanTorqueJAX",
]


def _host_float(value) -> float:
    return float(host_array(value, dtype=np.float64))


def _stacked(coils, geometry) -> jax.Array:
    """``geometry(curve)`` of every coil, stacked on the host and explicitly placed."""
    return _as_jax_float64(np.stack([geometry(coil.curve) for coil in coils]))


def _coil_group(coils) -> coil_forces.CoilGroup:
    return (
        _stacked(coils, lambda curve: curve.gamma()),
        _stacked(coils, lambda curve: curve.gammadash()),
        _as_jax_float64(np.asarray([coil.current.get_value() for coil in coils], dtype=np.float64)),
    )


def _second_derivatives(coils) -> jax.Array:
    return _stacked(coils, lambda curve: curve.gammadashdash())


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
    """Native's target and source lists, after its removal of duplicates and checks."""
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
    """The coils' cross-section regularizations, captured at construction as native does."""
    return host_array([coil.regularization for coil in coils], dtype=np.float64)


class _CoilSetObjective(Optimizable):
    """Target and source coils of a force or torque objective, with native's list rules.

    ``target_coils``, ``source_coils_coarse`` and ``source_coils_fine`` are the
    native attributes, read at every evaluation; ``source_coils``, as in
    native, is the two source lists at construction.
    """

    _operands: Callable[[], tuple]

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine, downsample):
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

    def _value(self, kernel):
        return _host_float(kernel(*self._operands(), downsample=self.downsample))

    def _source_derivative(self, dsources) -> Derivative:
        total = Derivative()
        for coils, cotangents in zip(self._source_lists(), dsources, strict=True):
            total += _coil_derivative(coils, *cotangents)
        return total


# One compiled program per objective, shapes, downsample and source-group count.
_lp_force = jax.jit(coil_forces.lp_force, static_argnames=("downsample",))
_lp_force_grad = jax.jit(
    jax.grad(coil_forces.lp_force, argnums=(0, 1, 4)), static_argnames=("downsample",)
)
_lp_torque = jax.jit(coil_forces.lp_torque, static_argnames=("downsample",))
_lp_torque_grad = jax.jit(
    jax.grad(coil_forces.lp_torque, argnums=(0, 1, 4)), static_argnames=("downsample",)
)
_squared_mean_force = jax.jit(coil_forces.squared_mean_force, static_argnames=("downsample",))
_squared_mean_force_grad = jax.jit(
    jax.grad(coil_forces.squared_mean_force, argnums=(0, 1)), static_argnames=("downsample",)
)
_squared_mean_torque = jax.jit(coil_forces.squared_mean_torque, static_argnames=("downsample",))
_squared_mean_torque_grad = jax.jit(
    jax.grad(coil_forces.squared_mean_torque, argnums=(0, 1)), static_argnames=("downsample",)
)
_b2energy = jax.jit(coil_forces.b2energy, static_argnames=("downsample",))
_b2energy_grad = jax.jit(
    jax.grad(coil_forces.b2energy, argnums=(0, 1, 2)), static_argnames=("downsample",)
)
_net_flux = jax.jit(coil_forces.net_flux, static_argnames=("downsample",))
_net_flux_grad = jax.jit(
    jax.grad(coil_forces.net_flux, argnums=(0, 1, 2)), static_argnames=("downsample",)
)


class _LpObjective(_CoilSetObjective):
    """Shared evaluation of :class:`LpCurveForceJAX` and :class:`LpCurveTorqueJAX`."""

    _native_name: str

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine, p, threshold, downsample):
        target_coils = _as_coil_list(target_coils)
        _check_regularized(target_coils, self._native_name)
        self._regularizations = _regularizations(target_coils)
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, downsample)
        self._quadpoints = np.asarray(self.target_coils[0].curve.quadpoints, dtype=np.float64)
        self._p = np.float64(p)
        self._threshold = np.float64(threshold)

    def _operands(self):
        return (
            _coil_group(self.target_coils),
            _second_derivatives(self.target_coils),
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


class LpCurveForceJAX(_LpObjective):
    r"""JAX-backed mirror of :class:`~simsopt.field.force.LpCurveForce`.

    ``J = (1/p) sum_i (1/n) sum_k max(|dF/dl| - threshold, 0)^p |gammadash|``
    in (MN/m)^p, with the force per unit length on each regularized target
    coil from its self field, the other targets and the sources.
    """

    _native_name = "LpCurveForce"

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None,
                 p: float = 2.0, threshold: float = 0.0, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, p, threshold, downsample)

    def J(self):
        return self._value(_lp_force)

    @derivative_dec
    def dJ(self):
        return self._derivative(_lp_force_grad)

    return_fn_map = {"J": J, "dJ": dJ}


class LpCurveTorqueJAX(_LpObjective):
    r"""JAX-backed mirror of :class:`~simsopt.field.force.LpCurveTorque`.

    As :class:`LpCurveForceJAX` for the torque per unit length (MN) about
    each target coil's arclength centroid.
    """

    _native_name = "LpCurveTorque"

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None,
                 p: float = 2.0, threshold: float = 0.0, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, p, threshold, downsample)

    def J(self):
        return self._value(_lp_torque)

    @derivative_dec
    def dJ(self):
        return self._derivative(_lp_torque_grad)

    return_fn_map = {"J": J, "dJ": dJ}


class _SquaredMeanObjective(_CoilSetObjective):
    """Shared evaluation of the squared mean force and torque objectives."""

    def _operands(self):
        return _coil_group(self.target_coils), self._source_groups()

    def _derivative(self, gradient) -> Derivative:
        dtargets, dsources = host_tree(
            gradient(*self._operands(), downsample=self.downsample), dtype=np.float64
        )
        return _coil_derivative(self.target_coils, *dtargets) + self._source_derivative(dsources)


class SquaredMeanForceJAX(_SquaredMeanObjective):
    r"""JAX-backed mirror of :class:`~simsopt.field.force.SquaredMeanForce`.

    ``J = sum_i |(1/L_i) int dF_i/dl dl|^2`` in (MN/m)^2 over the target
    coils, from the other targets and the sources.
    """

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, downsample)

    def J(self):
        return self._value(_squared_mean_force)

    @derivative_dec
    def dJ(self):
        return self._derivative(_squared_mean_force_grad)

    return_fn_map = {"J": J, "dJ": dJ}


class SquaredMeanTorqueJAX(_SquaredMeanObjective):
    r"""JAX-backed mirror of :class:`~simsopt.field.force.SquaredMeanTorque`.

    As :class:`SquaredMeanForceJAX` for the torque per unit length (MN) about
    each target coil's arclength centroid.
    """

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, downsample)

    def J(self):
        return self._value(_squared_mean_torque)

    @derivative_dec
    def dJ(self):
        return self._derivative(_squared_mean_torque_grad)

    return_fn_map = {"J": J, "dJ": dJ}


class B2EnergyJAX(Optimizable):
    r"""JAX-backed mirror of :class:`~simsopt.field.force.B2Energy`.

    ``J = (1/2) sum_ij I_i L_ij I_j`` in MJ, with the regularized
    self-inductances of the coils' cross sections on the diagonal of ``L``.
    """

    def __init__(self, target_coils, downsample=1):
        self.target_coils = target_coils
        self.downsample = downsample
        _check_regularized(target_coils, "B2Energy")
        _check_quadpoints_consistency(self.target_coils, "target_coils")
        _check_downsample(self.target_coils, downsample, "target_coils")
        self._regularizations = _regularizations(target_coils)
        super().__init__(depends_on=target_coils)

    def _operands(self):
        return (*_coil_group(self.target_coils), _as_jax_float64(self._regularizations))

    def J(self):
        return _host_float(_b2energy(*self._operands(), downsample=self.downsample))

    @derivative_dec
    def dJ(self):
        cotangents = host_tree(_b2energy_grad(*self._operands(), downsample=self.downsample), dtype=np.float64)
        return _coil_derivative(self.target_coils, *cotangents)

    return_fn_map = {"J": J, "dJ": dJ}


class NetFluxesJAX(Optimizable):
    r"""JAX-backed mirror of :class:`~simsopt.field.force.NetFluxes`.

    ``J = (1/n) sum_k A(gamma_k) . gammadash_k`` in Wb: the flux through the
    target coil of the sources' vector potential ``A``, at the target's
    ``downsample``-strided points. As native, ``dJ`` is the gradient of the
    flux at full target resolution (``downsample=1``), and the sources are the
    ``source_coils`` at construction (native builds a ``BiotSavart`` from
    them): reassigning the attribute changes neither value nor derivative.
    The sources are captured at construction for both the value and the
    gradient; native's gradient reads the list live after in-place edits,
    which makes its value and gradient inconsistent, and is not reproduced.
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
        return _host_float(_net_flux(*self._operands(), downsample=self.downsample))

    @derivative_dec
    def dJ(self):
        dgamma, dgammadash, dsources = host_tree(
            _net_flux_grad(*self._operands(), downsample=1), dtype=np.float64
        )
        curve = self.target_coil.curve
        return (
            curve.dgamma_by_dcoeff_vjp(dgamma)
            + curve.dgammadash_by_dcoeff_vjp(dgammadash)
            + _coil_derivative(self._sources, *dsources)
        )

    return_fn_map = {"J": J, "dJ": dJ}
