"""JAX coil force, torque and energy objectives against the native ones."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import logging
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.field import Coil, Current, RegularizedCoil, coils_via_symmetries
from simsopt.field.force import (
    B2Energy,
    LpCurveForce,
    LpCurveTorque,
    NetFluxes,
    SquaredMeanForce,
    SquaredMeanTorque,
)
from simsopt.field.force import lp_force_pure as native_lp_force
from simsopt.field.force import squared_mean_force_pure as native_squared_mean_force
from simsopt.field.selffield import regularization_circ, regularization_rect
from simsopt.geo import CurveXYZFourier, create_equally_spaced_curves
from simsopt.geo.curveperturbed import CurvePerturbed, GaussianSampler, PerturbationSample
from simsopt_jax.core import coil_forces
from simsopt_jax_adapters.field import (
    B2EnergyJAX,
    LpCurveForceJAX,
    LpCurveTorqueJAX,
    NetFluxesJAX,
    SquaredMeanForceJAX,
    SquaredMeanTorqueJAX,
)

_NCOILS = 2


def _coils(*, shared_dofs: bool = False) -> tuple[list, list]:
    """Return regularized base coils and all their symmetric copies.

    Some curve DOFs and one current are fixed. With ``shared_dofs``, a twin
    coil on its own quadrature grid shares the DOFs of the second base curve
    and of the second base current (scaled); it is returned last.
    """
    base = create_equally_spaced_curves(
        _NCOILS, 2, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24
    )
    rng = np.random.default_rng(11)
    for curve in base:
        curve.x = curve.x + 0.03 * rng.standard_normal(curve.x.shape)
    base[0].fix("xc(0)")
    base[1].fix("zs(1)")
    currents = [Current(1e5), Current(1.2e5)]
    currents[0].fix_all()
    regularizations = [regularization_circ(0.05), regularization_rect(0.04, 0.06)]
    coils = coils_via_symmetries(base, currents, 2, True, regularizations)
    if shared_dofs:
        twin = CurveXYZFourier(np.linspace(0, 1, 30, endpoint=False) + 0.013, 3, dofs=base[1].dofs)
        coils.append(RegularizedCoil(twin, -0.7 * Current(0.0, dofs=currents[1].dofs), regularizations[0]))
    return coils[:_NCOILS], coils


def _objective_pairs(*, shared_dofs: bool = False) -> list[tuple[str, Optimizable, Optimizable]]:
    """Native and JAX objectives over the same coils, each active at the test state."""
    targets, coils = _coils(shared_dofs=shared_dofs)
    coarse = coils[: 4 * _NCOILS]
    fine = coils[4 * _NCOILS:]
    cases = [
        ("LpCurveForce", lambda cls: cls(targets, coarse, fine, p=4)),
        ("LpCurveForce(threshold, downsample=2)",
         lambda cls: cls(targets, coarse, fine, p=2.5, threshold=0.002, downsample=2)),
        ("LpCurveTorque", lambda cls: cls(targets, coarse, fine, p=3, threshold=1e-4)),
        ("SquaredMeanForce", lambda cls: cls(targets, coarse, fine)),
        ("SquaredMeanTorque(downsample=3)",
         lambda cls: cls(targets, coarse[_NCOILS:5], coarse[5:], downsample=3)),
        ("B2Energy", lambda cls: cls(coarse)),
        ("B2Energy(downsample=2)", lambda cls: cls(coarse, downsample=2)),
        ("NetFluxes", lambda cls: cls(coarse[1], coarse)),
        ("NetFluxes(downsample=4)", lambda cls: cls(coarse[1], coarse, downsample=4)),
    ]
    classes = {
        "LpCurveForce": (LpCurveForce, LpCurveForceJAX),
        "LpCurveTorque": (LpCurveTorque, LpCurveTorqueJAX),
        "SquaredMeanForce": (SquaredMeanForce, SquaredMeanForceJAX),
        "SquaredMeanTorque": (SquaredMeanTorque, SquaredMeanTorqueJAX),
        "B2Energy": (B2Energy, B2EnergyJAX),
        "NetFluxes": (NetFluxes, NetFluxesJAX),
    }
    pairs = []
    for name, build in cases:
        native_class, jax_class = classes[name.split("(")[0]]
        pairs.append((name, build(native_class), build(jax_class)))
    return pairs


def _partials(objective: Optimizable) -> Derivative:
    return cast(Callable[..., Derivative], objective.dJ)(partials=True)


def _assert_matches_native(name: str, native: Optimizable, adapter: Optimizable) -> None:
    native_partials, adapter_partials = _partials(native), _partials(adapter)
    native_value = float(native.J())
    assert native_value != 0.0, f"{name} must be active at the test state"
    np.testing.assert_allclose(adapter.J(), native_value, rtol=1e-12, atol=0.0, err_msg=name)
    assert set(adapter_partials.data) == set(native_partials.data), name
    for owner, expected in native_partials.data.items():
        expected = np.asarray(expected)
        # Fixed DOFs keep their partials, as in the native Derivative.
        np.testing.assert_allclose(
            adapter_partials.data[owner], expected, rtol=1e-11, atol=1e-12 * np.max(np.abs(expected)),
            err_msg=f"{name} partials of {owner.name}",
        )
    native_gradient = np.asarray(native.dJ())
    np.testing.assert_allclose(
        adapter.dJ(), native_gradient, rtol=1e-11, atol=1e-12 * np.max(np.abs(native_gradient)),
        err_msg=f"{name} free gradient",
    )


@pytest.mark.parametrize("shared_dofs", [False, True], ids=["symmetric_copies", "shared_dofs_twin"])
def test_force_objectives_match_native_values_gradients_and_partials(shared_dofs):
    for name, native, adapter in _objective_pairs(shared_dofs=shared_dofs):
        assert adapter.dof_names == native.dof_names, name
        _assert_matches_native(name, native, adapter)


def test_force_gradients_match_central_differences():
    rng = np.random.default_rng(5)
    for name, _, adapter in _objective_pairs(shared_dofs=True):
        if name.startswith("NetFluxes(downsample"):
            continue  # native differentiates the full-resolution flux, see below
        x0 = np.array(adapter.x, dtype=float)
        direction = rng.standard_normal(x0.shape) * np.maximum(np.abs(x0), 1.0)
        step = 1e-7
        adapter.x = x0 + step * direction
        plus = adapter.J()
        adapter.x = x0 - step * direction
        minus = adapter.J()
        adapter.x = x0
        np.testing.assert_allclose(
            adapter.dJ() @ direction, (plus - minus) / (2 * step), rtol=1e-6, err_msg=name
        )


def test_force_objectives_follow_dof_and_downsample_changes_like_native():
    """DOFs and ``downsample`` are read at every evaluation, as native reads them."""
    for name, native, adapter in _objective_pairs(shared_dofs=True):
        adapter.x = np.asarray(adapter.x) * 1.01 + 0.005
        _assert_matches_native(name, native, adapter)
    for name, native, adapter in _objective_pairs():
        native.downsample = adapter.downsample = 2 * adapter.downsample
        _assert_matches_native(f"{name}, downsample changed", native, adapter)


def test_net_fluxes_gradient_is_the_full_resolution_gradient_like_native():
    """Native NetFluxes evaluates J at the downsampled points but dJ at all points."""
    _, coils = _coils()
    target, sources = coils[1], coils[:4 * _NCOILS]
    native = NetFluxes(target, sources, downsample=4)
    downsampled = NetFluxesJAX(target, sources, downsample=4)
    full = NetFluxesJAX(target, sources)
    assert abs(downsampled.J() - full.J()) > 1e-6 * abs(full.J())
    np.testing.assert_allclose(downsampled.dJ(), full.dJ(), rtol=1e-14, atol=0.0)
    _assert_matches_native("NetFluxes(downsample=4)", native, downsampled)


def test_coil_lists_follow_native_after_reassignment():
    """NetFluxes keeps the sources of construction (native's BiotSavart); force
    objectives read their target and source lists at every evaluation."""
    targets, coils = _coils()
    for native, adapter in (
        (NetFluxes(coils[1], coils[:6]), NetFluxesJAX(coils[1], coils[:6])),
        (SquaredMeanForce(targets, coils[:6]), SquaredMeanForceJAX(targets, coils[:6])),
    ):
        native.source_coils = adapter.source_coils = coils[2:4]
        _assert_matches_native(f"{type(native).__name__}, source_coils reassigned", native, adapter)
    # An in-place edit of the list changes neither the value nor the gradient: both
    # use the sources of construction (native's gradient would follow the edit).
    adapter = NetFluxesJAX(coils[1], coils[:6])
    value, gradient = adapter.J(), adapter.dJ()
    adapter.source_coils.reverse()
    np.testing.assert_allclose(adapter.J(), value, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(adapter.dJ(), gradient, rtol=1e-15, atol=0.0)
    native = LpCurveForce(targets, coils[:6], p=4)
    adapter = LpCurveForceJAX(targets, coils[:6], p=4)
    native.source_coils_coarse = adapter.source_coils_coarse = coils[3:5]
    _assert_matches_native("LpCurveForce, source_coils_coarse reassigned", native, adapter)
    assert native.source_coils == adapter.source_coils


def test_self_exclusion_keeps_the_singular_self_terms_out_of_the_gradient():
    """Two samples of one target lie 1e-10 apart per component, so the native distance
    offset cancels their difference exactly; native evaluates the target's own field in
    an untaken branch, and its gradient stays finite."""
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ring = np.stack((np.cos(angles), np.sin(angles), 0.2 * np.sin(2 * angles)), axis=1)
    ring_dash = 2 * np.pi * np.stack((-np.sin(angles), np.cos(angles), 0.4 * np.cos(2 * angles)), axis=1)
    target_gamma = ring - ring[0]
    target_gamma[1] = 1e-10
    targets = (target_gamma[None], ring_dash[None], np.asarray([1e5]))
    sources = (
        jnp.asarray(np.stack((1.3 * ring, 1.6 * ring))),
        jnp.asarray(np.stack((1.3 * ring_dash, 1.6 * ring_dash))),
        jnp.asarray([2e5, -1e5]),
    )
    gammadashdashs = -4 * np.pi**2 * ring[None]
    quadpoints = np.linspace(0.0, 1.0, 12, endpoint=False)
    regularizations = np.asarray([regularization_circ(0.05)])
    cases = (
        ("SquaredMeanForce",
         lambda t: native_squared_mean_force(t[0], sources[0], t[1], sources[1], t[2], sources[2], 1),
         lambda t: coil_forces.squared_mean_force(t, (sources,), 1)),
        ("LpCurveForce",
         lambda t: native_lp_force(t[0], sources[0], t[1], sources[1], gammadashdashs, [quadpoints], t[2],
                                   sources[2], regularizations, 2.0, 0.0, 1),
         lambda t: coil_forces.lp_force(t, gammadashdashs, quadpoints, regularizations, (sources,), 2.0, 0.0, 1)),
    )
    for name, native_kernel, jax_kernel in cases:
        native_value, native_gradient = jax.value_and_grad(native_kernel)(targets)
        value, gradient = jax.value_and_grad(jax_kernel)(targets)
        assert all(np.all(np.isfinite(g)) for g in native_gradient), name
        np.testing.assert_allclose(float(value), float(native_value), rtol=1e-12, atol=0.0, err_msg=name)
        for actual, expected in zip(gradient, native_gradient, strict=True):
            _assert_close_arrays(actual, expected, name)

    # The same target through the public objectives: a perturbed curve on a zero
    # base curve has exactly the sampled geometry, and partials of the base DOFs.
    base = CurveXYZFourier(quadpoints, 1)
    sampler = GaussianSampler(quadpoints, sigma=1.0, length_scale=0.5, n_derivs=2)
    curve = CurvePerturbed(base, PerturbationSample(
        sampler, sample=[target_gamma, ring_dash, gammadashdashs[0]]
    ))
    target = RegularizedCoil(curve, Current(1e5), regularization_circ(0.05))
    source_coils = [
        Coil(_scaled_ring(quadpoints, scale), Current(current))
        for scale, current in ((1.3, 2e5), (1.6, -1e5))
    ]
    np.testing.assert_array_equal(curve.gamma()[1] - curve.gamma()[0], 1e-10)
    for name, native, adapter in (
        ("SquaredMeanForce", SquaredMeanForce([target], source_coils), SquaredMeanForceJAX([target], source_coils)),
        ("LpCurveForce", LpCurveForce([target], source_coils, p=2), LpCurveForceJAX([target], source_coils, p=2)),
    ):
        assert all(np.all(np.isfinite(partial)) for partial in _partials(native).data.values()), name
        _assert_matches_native(f"public {name}", native, adapter)


def _scaled_ring(quadpoints, scale):
    """``scale`` times the source rings of the self-exclusion case, as a Fourier curve."""
    curve = CurveXYZFourier(quadpoints, 2)
    curve.set("xc(1)", scale)
    curve.set("ys(1)", scale)
    curve.set("zs(2)", 0.2 * scale)
    return curve


def _assert_close_arrays(actual, expected, name: str) -> None:
    expected = np.asarray(expected)
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=1e-11, atol=1e-12 * np.max(np.abs(expected)), err_msg=name
    )


def test_settings_fixed_at_construction_stay_fixed_like_native():
    """As native, ``p``, ``threshold`` and the targets' regularizations are construction settings."""
    targets, coils = _coils()
    native = LpCurveForce(targets, coils, p=4, threshold=0.001)
    adapter = LpCurveForceJAX(targets, coils, p=4, threshold=0.001)
    before = adapter.J()
    for coil in targets:
        coil.regularization = 2.0 * coil.regularization
    np.testing.assert_allclose(adapter.J(), before, rtol=1e-15, atol=0.0)
    _assert_matches_native("LpCurveForce after a regularization change", native, adapter)
    native_energy, adapter_energy = B2Energy(coils), B2EnergyJAX(coils)
    _assert_matches_native("B2Energy with the changed regularizations", native_energy, adapter_energy)


def _segment(x_center, x_amplitude, numquadpoints=24):
    """``x = x_center + x_amplitude cos(2 pi t)``, ``y = z = 0``: zero tangent at ``t = 0``."""
    curve = CurveXYZFourier(numquadpoints, 1)
    curve.set("xc(0)", x_center)
    curve.set("xc(1)", x_amplitude)
    return curve


def test_degenerate_coils_give_the_native_nans():
    """A target with a zero tangent gives native's NaN value and NaN gradient pattern."""
    _, coils = _coils()
    degenerate = RegularizedCoil(_segment(1.0, 0.3), Current(1e5), regularization_circ(0.05))
    for native, adapter in (
        (LpCurveForce([degenerate], coils, p=2), LpCurveForceJAX([degenerate], coils, p=2)),
        (SquaredMeanTorque([degenerate], coils), SquaredMeanTorqueJAX([degenerate], coils)),
    ):
        assert np.isnan(native.J()) and np.isnan(adapter.J())
        native_partials, adapter_partials = _partials(native), _partials(adapter)
        for owner, expected in native_partials.data.items():
            expected = np.asarray(expected)
            np.testing.assert_array_equal(np.isnan(adapter_partials.data[owner]), np.isnan(expected))
            finite = ~np.isnan(expected)
            np.testing.assert_allclose(
                np.asarray(adapter_partials.data[owner])[finite], expected[finite], rtol=1e-11,
                atol=1e-12 * np.max(np.abs(expected[finite]), initial=0.0),
            )


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda cls, targets, coils: cls(targets, targets), "must together contain at least one coil"),
        (lambda cls, targets, coils: cls(targets, coils, downsample=5), "must evenly divide"),
        (lambda cls, targets, coils: cls(targets, coils, downsample=0), "downsample must be >= 1"),
        (lambda cls, targets, coils: cls([Coil(c.curve, c.current) for c in targets], coils),
         "can only be used with RegularizedCoil objects"),
        (lambda cls, targets, coils: cls(targets, [*coils, Coil(CurveXYZFourier(30, 2), Current(1.0))]),
         "same number of quadrature points"),
    ],
    ids=["no_sources", "downsample_not_dividing", "downsample_zero", "unregularized", "quadrature_mismatch"],
)
def test_construction_rejects_what_native_rejects(build, message):
    targets, coils = _coils()
    for cls in (LpCurveForce, LpCurveForceJAX):
        with pytest.raises(ValueError, match=message):
            build(cls, targets, coils)


def test_force_objectives_make_no_implicit_transfers():
    """J and dJ move data only through explicit transfers, also after a DOF change."""
    pairs = _objective_pairs(shared_dofs=True)
    for _, _, adapter in pairs:
        adapter.J()
        adapter.dJ()
    with jax.transfer_guard("disallow"):
        for _, _, adapter in pairs:
            adapter.J()
            adapter.dJ()
        for _, _, adapter in pairs:
            adapter.x = np.asarray(adapter.x) + 0.01
            adapter.J()
            adapter.dJ()
    for name, native, adapter in pairs:
        _assert_matches_native(name, native, adapter)


def test_force_programs_compile_once_per_coil_shape(caplog):
    """New DOFs, currents and new objectives on equal shapes reuse the compiled programs."""
    targets, coils = _coils()
    first = LpCurveForceJAX(targets, coils, p=4)
    first.J()
    first.dJ()
    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        first.x = np.asarray(first.x) + 0.01
        first.J()
        first.dJ()
        other_targets, other_coils = _coils()
        second = LpCurveForceJAX(other_targets, other_coils, p=2, threshold=0.001)
        second.J()
        second.dJ()
    compiles = [record.getMessage() for record in caplog.records if "Compiling" in record.getMessage()]
    assert compiles == []
    caplog.clear()
    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        first.downsample = 2
        first.J()
    # Control: a new downsample changes the shapes and does compile.
    assert any("Compiling" in record.getMessage() for record in caplog.records)
