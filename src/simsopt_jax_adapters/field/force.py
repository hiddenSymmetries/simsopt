"""JAX coil force, torque and energy objectives as drop-in native Optimizables.

Each class mirrors the native objective named by removing the ``Jax`` prefix in
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
target coils' regularizations and the sources of :class:`JaxNetFluxes` are
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
    "JaxB2Energy",
    "JaxLpCurveForce",
    "JaxLpCurveTorque",
    "JaxNetFluxes",
    "JaxSquaredMeanForce",
    "JaxSquaredMeanTorque",
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
    """Shared evaluation of :class:`JaxLpCurveForce` and :class:`JaxLpCurveTorque`."""

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


class JaxLpCurveForce(_LpObjective):
    r"""JAX version of :class:`~simsopt.field.force.LpCurveForce`: :math:`L^p` penalty on coil force per unit length.

    .. math::
        J = \frac{1}{p}\sum_i \frac{1}{N}\sum_{k}
            \max\left(\left|\frac{d\vec{F}_i}{d\ell}(t_k)\right| - F_0, 0\right)^p |\gamma_i'(t_k)|
          \approx \frac{1}{p}\sum_i \int \max\left(\left|\frac{d\vec{F}_i}{d\ell}\right| - F_0, 0\right)^p d\ell_i,

    where :math:`d` = ``downsample``, the sum over :math:`k` runs over the :math:`N = n/d` points
    :math:`t_k` taken with stride :math:`d` from the :math:`n` target quadrature points, and :math:`\gamma_i'` is
    the derivative with respect to the curve parameter. On the default uniform grid over the full period
    :math:`[0, 1)`, :math:`t_k = kd/n` and the quadrature approximates the arclength integral on the right. The force per unit length

    .. math::
        \frac{d\vec{F}_i}{d\ell} = I_i\, \hat{t}_i \times (\vec{B}_{i,\text{self}} + \vec{B}_{i,\text{mutual}})

    is computed in N/m and converted to MN/m before thresholding, with :math:`\hat{t}_i = \gamma_i'/|\gamma_i'|`,
    the regularized self field of coil :math:`i`'s finite cross section, and the field of the other target coils
    and all source coils (also sampled with stride :math:`d`). Target coils are removed from the source lists.

    :math:`F_0` is ``threshold``. The units of :math:`J` are :math:`(\mathrm{MN/m})^p\,\mathrm{m}`.

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

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None,
                 p: float = 2.0, threshold: float = 0.0, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, p, threshold, downsample)

    def J(self):
        """Return the objective value given by the formula in the class docstring.

        Returns:
            float: objective value with units given by the class contract.
        """
        return self._value(_lp_force)

    @derivative_dec
    def dJ(self):
        """Project combined geometry/current cotangents onto native coil DOFs.

        Shared DOFs accumulate; partials=True includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return self._derivative(_lp_force_grad)

    return_fn_map = {"J": J, "dJ": dJ}


class JaxLpCurveTorque(_LpObjective):
    r"""JAX version of :class:`~simsopt.field.force.LpCurveTorque`: :math:`L^p` penalty on coil torque per unit length.

    .. math::
        J = \frac{1}{p}\sum_i \frac{1}{N}\sum_{k}
            \max\left(\left|\frac{d\vec{T}_i}{d\ell}(t_k)\right| - T_0, 0\right)^p |\gamma_i'(t_k)|
          \approx \frac{1}{p}\sum_i \int \max\left(\left|\frac{d\vec{T}_i}{d\ell}\right| - T_0, 0\right)^p d\ell_i,

    with the torque per unit length about the arclength centroid :math:`\vec{c}_i` of coil :math:`i`,

    .. math::
        \frac{d\vec{T}_i}{d\ell} = (\gamma_i - \vec{c}_i) \times \frac{d\vec{F}_i}{d\ell}, \qquad
        \vec{c}_i = \frac{\sum_k \gamma_i(t_k)\, |\gamma_i'(t_k)|}{\sum_k |\gamma_i'(t_k)|},

    where :math:`d` = ``downsample``, the sum over :math:`k` runs over the :math:`N = n/d` points
    :math:`t_k` taken with stride :math:`d` from the :math:`n` target quadrature points, and :math:`\gamma_i'` is
    the derivative with respect to the curve parameter. On the default uniform grid over the full period
    :math:`[0, 1)`, :math:`t_k = kd/n` and the quadrature approximates the arclength integral on the right. The force per unit length

    .. math::
        \frac{d\vec{F}_i}{d\ell} = I_i\, \hat{t}_i \times (\vec{B}_{i,\text{self}} + \vec{B}_{i,\text{mutual}})

    is computed in N/m and converted to MN/m before thresholding, with :math:`\hat{t}_i = \gamma_i'/|\gamma_i'|`,
    the regularized self field of coil :math:`i`'s finite cross section, and the field of the other target coils
    and all source coils (also sampled with stride :math:`d`). Target coils are removed from the source lists.

    :math:`T_0` is ``threshold``, the torque per unit length is in MN, and :math:`J` is in :math:`\mathrm{MN}^p\,\mathrm{m}`.

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

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None,
                 p: float = 2.0, threshold: float = 0.0, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, p, threshold, downsample)

    def J(self):
        """Return the objective value given by the formula in the class docstring.

        Returns:
            float: objective value with units given by the class contract.
        """
        return self._value(_lp_torque)

    @derivative_dec
    def dJ(self):
        """Project combined geometry/current cotangents onto native coil DOFs.

        Shared DOFs accumulate; partials=True includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
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


class JaxSquaredMeanForce(_SquaredMeanObjective):
    r"""JAX version of :class:`~simsopt.field.force.SquaredMeanForce`: squared net Lorentz force on each coil.

    .. math::
        J = \sum_i \left|\frac{1}{N}\sum_k \frac{d\vec{F}_i}{d\ell}(t_k)\, |\gamma_i'(t_k)|\right|^2
          \approx \sum_i \left|\int \frac{d\vec{F}_i}{d\ell}\, d\ell_i\right|^2,

    where :math:`d` = ``downsample``, the sum over :math:`k` runs over the :math:`N = n/d` points
    :math:`t_k` taken with stride :math:`d` from the :math:`n` target quadrature points, and :math:`\gamma_i'` is
    the derivative with respect to the curve parameter. On the default uniform grid over the full period
    :math:`[0, 1)`, :math:`t_k = kd/n` and the quadrature approximates the integral on the right. The force per
    unit length
    :math:`d\vec{F}_i/d\ell = I_i\, \hat{t}_i \times \vec{B}_{i,\text{mutual}}`, in N/m, uses only the field of
    the other target coils and all source coils (also sampled with stride :math:`d`); there is no self force.
    :math:`J` is converted from N^2 to MN^2.

    Args:
        target_coils (Coil or list of Coil): coils on which the net force is computed. Coils
            that also appear in a source list are removed from that list.
        source_coils_coarse (Coil or list of Coil): external source coils with one shared quadrature count.
        source_coils_fine (Coil, list of Coil or None): optional second source list with its own quadrature
            count, e.g. finely resolved TF coils next to coarse dipole coils. Default: None.
        downsample (int): stride over the quadrature points of every coil list; it must divide every list's
            quadrature count. Default: 1.
    """

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, downsample)

    def J(self):
        """Return the objective value given by the formula in the class docstring.

        Returns:
            float: objective value with units given by the class contract.
        """
        return self._value(_squared_mean_force)

    @derivative_dec
    def dJ(self):
        """Project combined geometry/current cotangents onto native coil DOFs.

        Shared DOFs accumulate; partials=True includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return self._derivative(_squared_mean_force_grad)

    return_fn_map = {"J": J, "dJ": dJ}


class JaxSquaredMeanTorque(_SquaredMeanObjective):
    r"""JAX version of :class:`~simsopt.field.force.SquaredMeanTorque`: squared net Lorentz torque on each coil.

    .. math::
        J = \sum_i \left|\frac{1}{N}\sum_k
            (\gamma_i(t_k) - \vec{c}_i) \times \frac{d\vec{F}_i}{d\ell}(t_k)\, |\gamma_i'(t_k)|\right|^2
          \approx \sum_i \left|\int (\gamma_i - \vec{c}_i) \times \frac{d\vec{F}_i}{d\ell}\, d\ell_i\right|^2,

    with :math:`\vec{c}_i = \sum_k \gamma_i(t_k) |\gamma_i'(t_k)| / \sum_k |\gamma_i'(t_k)|` the arclength
    centroid of coil :math:`i`, where :math:`d` = ``downsample``, the sum over :math:`k` runs over the :math:`N = n/d` points
    :math:`t_k` taken with stride :math:`d` from the :math:`n` target quadrature points, and :math:`\gamma_i'` is
    the derivative with respect to the curve parameter. On the default uniform grid over the full period
    :math:`[0, 1)`, :math:`t_k = kd/n` and the quadrature approximates the integral on the right. The force per
    unit length
    :math:`d\vec{F}_i/d\ell = I_i\, \hat{t}_i \times \vec{B}_{i,\text{mutual}}`, in N/m, uses only the field of
    the other target coils and all source coils (also sampled with stride :math:`d`); there is no self force.
    :math:`J` is converted from (N m)^2 to (MN m)^2.

    Args:
        target_coils (Coil or list of Coil): coils on which the net torque is computed. Coils
            that also appear in a source list are removed from that list.
        source_coils_coarse (Coil or list of Coil): external source coils with one shared quadrature count.
        source_coils_fine (Coil, list of Coil or None): optional second source list with its own quadrature
            count, e.g. finely resolved TF coils next to coarse dipole coils. Default: None.
        downsample (int): stride over the quadrature points of every coil list; it must divide every list's
            quadrature count. Default: 1.
    """

    def __init__(self, target_coils, source_coils_coarse, source_coils_fine=None, downsample: int = 1):
        super().__init__(target_coils, source_coils_coarse, source_coils_fine, downsample)

    def J(self):
        """Return the objective value given by the formula in the class docstring.

        Returns:
            float: objective value with units given by the class contract.
        """
        return self._value(_squared_mean_torque)

    @derivative_dec
    def dJ(self):
        """Project combined geometry/current cotangents onto native coil DOFs.

        Shared DOFs accumulate; partials=True includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return self._derivative(_squared_mean_torque_grad)

    return_fn_map = {"J": J, "dJ": dJ}


class JaxB2Energy(Optimizable):
    r"""JAX version of :class:`~simsopt.field.force.B2Energy`: vacuum magnetic field energy of a set of coils.

    .. math::
        J = \frac{1}{2}\sum_{i,j} I_i L_{ij} I_j,

    where :math:`I_i` is the current in coil :math:`i` and :math:`L_{ij}` is the inductance matrix, computed on
    every ``downsample``-th quadrature point, with the regularized self-inductance of each coil's finite cross
    section on the diagonal. :math:`J` is in MJ (joules times :math:`10^{-6}`).

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

    def _operands(self):
        return (*_coil_group(self.target_coils), _as_jax_float64(self._regularizations))

    def J(self):
        """Return the objective value given by the formula in the class docstring.

        Returns:
            float: objective value with units given by the class contract.
        """
        return _host_float(_b2energy(*self._operands(), downsample=self.downsample))

    @derivative_dec
    def dJ(self):
        """Project combined geometry/current cotangents onto native coil DOFs.

        Shared DOFs accumulate; partials=True includes fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        cotangents = host_tree(_b2energy_grad(*self._operands(), downsample=self.downsample), dtype=np.float64)
        return _coil_derivative(self.target_coils, *cotangents)

    return_fn_map = {"J": J, "dJ": dJ}


class JaxNetFluxes(Optimizable):
    r"""JAX version of :class:`~simsopt.field.force.NetFluxes`: flux of the source coils through a coil.

    .. math::
        \Psi = \frac{1}{N}\sum_{k=0}^{N-1} \vec{A}(\gamma(t_k)) \cdot \gamma'(t_k)
            \approx \oint \vec{A} \cdot d\vec{\ell},
        \qquad t_k = \frac{kd}{n}, \quad N = \frac{n}{d},

    where :math:`\gamma` is the target curve with :math:`n` quadrature points, :math:`\gamma'` its derivative
    with respect to the unit-period curve parameter, :math:`d` = ``downsample``, and :math:`\vec{A}` is the
    Biot-Savart vector potential of the source coils at their full quadrature. Both the value and the gradient
    use only these :math:`N` strided target points. :math:`\Psi` is in Wb.

    The sources are the ``source_coils`` given at construction; reassigning or editing that list afterwards
    changes neither the value nor the gradient.

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
        """Return the objective value given by the formula in the class docstring.

        Returns:
            float: objective value with units given by the class contract.
        """
        return _host_float(_net_flux(*self._operands(), downsample=self.downsample))

    @derivative_dec
    def dJ(self):
        """Project combined geometry/current cotangents onto native coil DOFs.

        Shared DOFs accumulate; partials=True includes fixed DOF partials.

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
