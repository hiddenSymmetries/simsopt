"""Direct parity coverage for the JAX curve-subclass kernels.

``CurveHelical``, ``CurvePlanarFourier``, and ``CurveRZFourier`` are otherwise
reached only incidentally (building a curve as a side effect of a Biot-Savart
scenario). This module gives
each of the three pure JAX kernels
(``src/simsopt_jax/core/curve_helical.py``, ``curve_planar_fourier.py``,
``curve_rz_fourier.py``) direct, subclass-specific coverage:

* the raw ``*_pure(dofs, quadpoints, ...)`` kernel is called directly and
  compared to the native (oracle) curve's ``gamma()``;
* the generic spec composition in ``simsopt_jax.core.curve_geometry``
  (``curve_geometry_from_dofs``, built on the same three kernels) is compared
  to the native ``gamma()`` / ``gammadash()`` / ``gammadashdash()`` for
  position, tangent, and second-derivative parity;
* the production adapter boundary
  (``simsopt_jax_adapters.geo.curve_specs.curve_spec_from_adapter_curve``,
  the same conversion the JAX Biot-Savart path uses) is exercised for a
  DOF round trip: dofs set on the native curve must reappear unchanged in the
  derived spec and reproduce the same curve on the JAX side.

The native curve is the oracle in every comparison; both lanes run in the
same test process (native side effects never cross a process boundary).
Bitwise equality is not required -- the two implementations reassociate the
underlying trigonometric sums differently, so values are compared with a
tolerance several orders of magnitude above the observed noise floor
(~1e-13 for second derivatives, ~1e-15 for positions).
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    from simsopt_jax.core.curve_helical import curve_helical_pure
    from simsopt_jax.core.curve_planar_fourier import curveplanarfourier_pure
    from simsopt_jax.core.curve_rz_fourier import curverzfourier_pure
    from simsopt_jax.core import (
        curve_geometry_from_dofs,
        make_curve_helical_spec,
        make_curve_planarfourier_spec,
        make_curve_rzfourier_spec,
    )
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

from unittest_jax_support import host_array, parity_default_device, parity_rng

import numpy as np

from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier


_GAMMA_ATOL = 1e-10
_GAMMA_RTOL = 1e-10
_DERIV_ATOL = 1e-9
_DERIV_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Curve construction (native is the oracle; dofs are randomized but seeded)
# ---------------------------------------------------------------------------


def _make_helical_curve(seed: int) -> CurveHelical:
    order, m, ell, R0, r = 4, 5, 2, 1.0, 0.3
    curve = CurveHelical(24, order, m, ell, R0, r)
    dofs = np.zeros(curve.dof_size)
    dofs[0] = np.pi / 2
    dofs += 0.01 * parity_rng(seed).normal(size=dofs.shape)
    curve.x = dofs
    return curve


def _make_planar_curve(seed: int) -> CurvePlanarFourier:
    order = 3
    curve = CurvePlanarFourier(20, order)
    dofs = np.zeros(curve.dof_size)
    dofs[0] = 1.0
    dofs[1] = 0.1
    dofs[order + 1] = 0.1
    # The C++ CurvePlanarFourier normalizes every nonzero quaternion by its
    # exact norm, without epsilon regularization; the JAX kernel does likewise.
    q_start = 2 * order + 1
    dofs[q_start : q_start + 4] = np.array([0.9, 0.2, -0.15, 0.05])
    dofs[-3:] = np.array([2.0, -0.5, 0.3])
    dofs += 0.01 * parity_rng(seed).normal(size=dofs.shape)
    curve.x = dofs
    return curve


def _make_rzfourier_curve(seed: int, *, stellsym: bool) -> CurveRZFourier:
    order, nfp = 3, 3
    curve = CurveRZFourier(20, order, nfp, stellsym)
    dofs = np.zeros(curve.dof_size)
    dofs[0] = 1.3
    dofs[1] = 0.15
    dofs += 0.01 * parity_rng(seed).normal(size=dofs.shape)
    curve.x = dofs
    return curve


def _helical_spec(curve: CurveHelical):
    return make_curve_helical_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        order=curve.order,
        m=curve.m,
        ell=curve.ell,
        R0=curve.R0,
        r=curve.r,
    )


def _planar_spec(curve: CurvePlanarFourier):
    return make_curve_planarfourier_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        order=curve.order,
    )


def _rzfourier_spec(curve: CurveRZFourier):
    return make_curve_rzfourier_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
    )


# ---------------------------------------------------------------------------
# Assertion helpers (shared numeric comparison; construction stays DAMP above)
# ---------------------------------------------------------------------------


def _assert_pure_kernel_gamma_matches_native(curve, pure_fn, *static_args) -> None:
    """Compare the raw ``*_pure`` kernel directly against native ``gamma()``."""
    quadpoints = np.asarray(curve.quadpoints)
    gamma_jax = host_array(
        pure_fn(np.asarray(curve.get_dofs()), quadpoints, *static_args)
    )
    np.testing.assert_allclose(
        gamma_jax, curve.gamma(), rtol=_GAMMA_RTOL, atol=_GAMMA_ATOL
    )


def _assert_spec_geometry_matches_native(curve, spec) -> None:
    """Compare position, tangent, and second derivative to the native oracle."""
    gamma, gammadash, gammadashdash = curve_geometry_from_dofs(spec, spec.dofs)
    np.testing.assert_allclose(
        host_array(gamma), curve.gamma(), rtol=_GAMMA_RTOL, atol=_GAMMA_ATOL
    )
    np.testing.assert_allclose(
        host_array(gammadash),
        curve.gammadash(),
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
    )
    np.testing.assert_allclose(
        host_array(gammadashdash),
        curve.gammadashdash(),
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
    )


def _assert_dof_round_trip(curve, *, seed: int) -> None:
    """Adapter DOF pass-through, then geometry re-verified at two DOF configurations.

    The ``assert_array_equal`` checks below confirm that
    ``curve_spec_from_adapter_curve`` (the production adapter boundary, not
    the hand-built factories) neither reorders, truncates, nor rounds the
    DOF vector it is constructed from: each ``spec.dofs`` is checked against
    the exact ``curve.get_dofs()`` array the constructor was just given, so
    these are pass-through checks on the constructor's input handling, not a
    round trip through independent state. The substantive verification is
    ``_assert_spec_geometry_matches_native``, run once at the curve's
    starting dofs and again after mutating the native curve's dofs in place
    to a second, perturbed configuration, confirming the adapter's derived
    geometry tracks native at more than one DOF value.
    """
    original_dofs = np.array(curve.get_dofs(), copy=True)
    spec = curve_spec_from_adapter_curve(curve)
    np.testing.assert_array_equal(host_array(spec.dofs), original_dofs)
    _assert_spec_geometry_matches_native(curve, spec)

    perturbed_dofs = original_dofs + 0.02 * parity_rng(seed).normal(
        size=original_dofs.shape
    )
    curve.x = perturbed_dofs
    updated_spec = curve_spec_from_adapter_curve(curve)
    np.testing.assert_array_equal(host_array(updated_spec.dofs), curve.get_dofs())
    _assert_spec_geometry_matches_native(curve, updated_spec)


# ---------------------------------------------------------------------------
# CurveHelical
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CurvePlanarFourier
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CurveRZFourier
# ---------------------------------------------------------------------------


class TestCurveSubclassesParity(JaxTestCase):
    def test_curve_helical_pure_kernel_matches_native_position(self):
        """Helical pure JAX positions match native curve evaluation."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_helical_pure_kernel_matches_native_position(
                    parity_lane, patches
                )

    def _case_curve_helical_pure_kernel_matches_native_position(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_helical_curve(seed=101)
        _assert_pure_kernel_gamma_matches_native(
            curve,
            curve_helical_pure,
            curve.order,
            curve.m,
            curve.ell,
            curve.R0,
            curve.r,
        )

    def test_curve_helical_position_and_derivatives_match_native(self):
        """Helical spec positions and parameter derivatives match native geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_helical_position_and_derivatives_match_native(
                    parity_lane, patches
                )

    def _case_curve_helical_position_and_derivatives_match_native(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_helical_curve(seed=102)
        _assert_spec_geometry_matches_native(curve, _helical_spec(curve))

    def test_curve_helical_dof_round_trip_matches_native(self):
        """Helical native DOF setters round-trip to matching immutable spec geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_helical_dof_round_trip_matches_native(
                    parity_lane, patches
                )

    def _case_curve_helical_dof_round_trip_matches_native(self, parity_lane, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_helical_curve(seed=103)
        _assert_dof_round_trip(curve, seed=203)

    def test_curve_planarfourier_pure_kernel_matches_native_position(self):
        """Planar Fourier pure JAX positions match native curve evaluation."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_planarfourier_pure_kernel_matches_native_position(
                    parity_lane, patches
                )

    def _case_curve_planarfourier_pure_kernel_matches_native_position(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_planar_curve(seed=111)
        _assert_pure_kernel_gamma_matches_native(
            curve, curveplanarfourier_pure, curve.order
        )

    def test_curve_planarfourier_position_and_derivatives_match_native(self):
        """Planar Fourier spec positions and parameter derivatives match native geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_planarfourier_position_and_derivatives_match_native(
                    parity_lane, patches
                )

    def _case_curve_planarfourier_position_and_derivatives_match_native(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_planar_curve(seed=112)
        _assert_spec_geometry_matches_native(curve, _planar_spec(curve))

    def test_curve_planarfourier_dof_round_trip_matches_native(self):
        """Planar Fourier native DOF setters round-trip to matching immutable spec
        geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_planarfourier_dof_round_trip_matches_native(
                    parity_lane, patches
                )

    def _case_curve_planarfourier_dof_round_trip_matches_native(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_planar_curve(seed=113)
        _assert_dof_round_trip(curve, seed=213)

    def test_curve_planarfourier_small_quaternion_matches_native(self):
        """Planar Fourier geometry with a very small quaternion matches native normalized
        rotation."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_planarfourier_small_quaternion_matches_native(
                    parity_lane, patches
                )

    def _case_curve_planarfourier_small_quaternion_matches_native(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_planar_curve(seed=114)
        dofs = np.array(curve.get_dofs(), copy=True)
        q_start = 2 * curve.order + 1
        dofs[q_start : q_start + 4] = np.array([1.0, 2.0, 3.0, 4.0]) * 1e-12
        curve.x = dofs
        _assert_pure_kernel_gamma_matches_native(
            curve, curveplanarfourier_pure, curve.order
        )
        _assert_spec_geometry_matches_native(
            curve, curve_spec_from_adapter_curve(curve)
        )

    def test_curve_rzfourier_pure_kernel_matches_native_position(self):
        """RZ Fourier pure JAX positions match native curve evaluation."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for case_id_1, stellsym in zip(
                ["stellsym", "non_stellsym"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    parity_lane=parity_lane,
                    case_id_1=case_id_1,
                    stellsym=stellsym,
                ), self.case() as patches:
                    self._case_curve_rzfourier_pure_kernel_matches_native_position(
                        parity_lane, stellsym, patches
                    )

    def _case_curve_rzfourier_pure_kernel_matches_native_position(
        self, parity_lane, stellsym, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_rzfourier_curve(seed=121 if stellsym else 122, stellsym=stellsym)
        _assert_pure_kernel_gamma_matches_native(
            curve,
            curverzfourier_pure,
            curve.order,
            curve.nfp,
            curve.stellsym,
        )

    def test_curve_rzfourier_position_and_derivatives_match_native(self):
        """RZ Fourier spec positions and parameter derivatives match native geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for case_id_1, stellsym in zip(
                ["stellsym", "non_stellsym"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    parity_lane=parity_lane,
                    case_id_1=case_id_1,
                    stellsym=stellsym,
                ), self.case() as patches:
                    self._case_curve_rzfourier_position_and_derivatives_match_native(
                        parity_lane, stellsym, patches
                    )

    def _case_curve_rzfourier_position_and_derivatives_match_native(
        self, parity_lane, stellsym, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_rzfourier_curve(seed=123 if stellsym else 124, stellsym=stellsym)
        _assert_spec_geometry_matches_native(curve, _rzfourier_spec(curve))

    def test_curve_rzfourier_dof_round_trip_matches_native(self):
        """RZ Fourier native DOF setters round-trip to matching immutable spec geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for case_id_1, stellsym in zip(
                ["stellsym", "non_stellsym"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    parity_lane=parity_lane,
                    case_id_1=case_id_1,
                    stellsym=stellsym,
                ), self.case() as patches:
                    self._case_curve_rzfourier_dof_round_trip_matches_native(
                        parity_lane, stellsym, patches
                    )

    def _case_curve_rzfourier_dof_round_trip_matches_native(
        self, parity_lane, stellsym, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_rzfourier_curve(seed=125 if stellsym else 126, stellsym=stellsym)
        _assert_dof_round_trip(curve, seed=225 if stellsym else 226)
