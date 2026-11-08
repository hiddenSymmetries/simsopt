"""Downsampled flux derivatives against independently sampled native loops."""

from __future__ import annotations

import numpy as np
from unittest_jax_support import JaxTestCase, optional_jax_imports

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.field import Coil, Current
from simsopt.field.force import NetFluxes
from simsopt.geo import CurveXYZFourier

with optional_jax_imports():
    from simsopt_jax_adapters.field import JaxNetFluxes
    from geo.test_curve_objectives_jax import _partials


def _objectives(stride: int, shared: bool):
    target_curve = CurveXYZFourier(12, 3)
    target_coefficients = {"xc(1)": 0.6, "ys(1)": 0.6, "zs(2)": 0.11, "xc(3)": 0.08}
    target_curve.local_full_x = np.asarray(
        [
            target_coefficients.get(name, 0.0)
            for name in target_curve.local_full_dof_names
        ]
    )
    source_curve = CurveXYZFourier(24, 3)
    source_coefficients = {
        "xc(0)": 0.17,
        "zc(0)": 0.45,
        "xc(1)": 0.95,
        "ys(1)": 0.95,
        "zs(3)": 0.09,
    }
    source_curve.local_full_x = np.asarray(
        [
            source_coefficients.get(name, 0.0)
            for name in source_curve.local_full_dof_names
        ]
    )
    target_curve.fix("xc(1)")
    source_curve.fix("zc(0)")
    source_current = Current(1.2e5)
    if shared:
        source_current.fix_all()
    sources = [Coil(source_curve, source_current)]
    if shared:
        twin = CurveXYZFourier(
            np.linspace(0, 1, 24, endpoint=False) + 0.017, 3, dofs=source_curve.dofs
        )
        sources.append(Coil(twin, -0.4 * Current(0.0, dofs=source_current.dofs)))
    sampled = CurveXYZFourier(
        np.asarray(target_curve.quadpoints)[::stride], 3, dofs=target_curve.dofs
    )
    target = Coil(target_curve, Current(9e4))
    reference = NetFluxes(Coil(sampled, target.current), sources, downsample=1)
    adapter = JaxNetFluxes(target, sources, downsample=stride)
    return adapter, reference, (target_curve, source_curve, source_current)


def _full_partial(partials: Derivative, owner: Optimizable) -> np.ndarray:
    return sum(
        (np.asarray(partials.data[item]) for item in owner.dofs.dep_opts()),
        start=np.zeros(owner.local_full_dof_size),
    )


class TestNetFluxesDownsamplingJax(JaxTestCase):
    def test_value_and_free_gradient_match_explicitly_sampled_native_loop(self):
        """Target stride samples the flux loop while source quadrature remains full."""
        for stride in (1, 2, 3):
            for shared in (False, True):
                with self.subTest(stride=stride, shared=shared), self.case():
                    self._case_sampled_reference(stride, shared)

    def _case_sampled_reference(self, stride: int, shared: bool):
        adapter, reference, _ = _objectives(stride, shared)
        self.assertNotEqual(reference.J(), 0.0)
        np.testing.assert_allclose(adapter.J(), reference.J(), rtol=1e-12, atol=0)
        expected = np.asarray(_partials(reference)(adapter))
        np.testing.assert_allclose(
            adapter.dJ(), expected, rtol=1e-11, atol=1e-12 * np.max(np.abs(expected))
        )

    def test_fixed_and_shared_partials_match_explicitly_sampled_native_loop(self):
        """Full partials retain fixed coefficients/current and accumulate shared DOFs."""
        for stride in (1, 2, 3):
            for shared in (False, True):
                with self.subTest(stride=stride, shared=shared), self.case():
                    self._case_full_partials(stride, shared)

    def _case_full_partials(self, stride: int, shared: bool):
        adapter, reference, owners = _objectives(stride, shared)
        actual, expected = _partials(adapter), _partials(reference)
        for owner in owners:
            full_expected = _full_partial(expected, owner)
            self.assertGreater(np.linalg.norm(full_expected), 0)
            np.testing.assert_allclose(
                _full_partial(actual, owner),
                full_expected,
                rtol=1e-11,
                atol=1e-12 * np.max(np.abs(full_expected)),
                err_msg=f"stride={stride}, shared={shared}, owner={owner.name}",
            )

    def test_public_gradient_differentiates_public_downsampled_value(self):
        """The free-DOF directional derivative agrees with a central difference of J."""
        for stride in (1, 2, 3):
            for shared in (False, True):
                with self.subTest(stride=stride, shared=shared), self.case():
                    self._case_finite_difference(stride, shared)

    def _case_finite_difference(self, stride: int, shared: bool):
        adapter, _, _ = _objectives(stride, shared)
        state = np.asarray(adapter.x).copy()
        direction = np.random.default_rng(713).standard_normal(state.shape)
        direction /= np.linalg.norm(direction)
        analytic = np.asarray(adapter.dJ()) @ direction
        step = 1e-6
        adapter.x = state + step * direction
        plus = adapter.J()
        adapter.x = state - step * direction
        minus = adapter.J()
        adapter.x = state
        np.testing.assert_allclose(
            analytic, (plus - minus) / (2 * step), rtol=2e-7, atol=1e-10
        )
