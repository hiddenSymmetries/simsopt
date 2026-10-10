"""Symmetry-curve geometry and derivative parity against native curves.

The pure kernels consume immutable specs; the native mutable curve supplies
an independent geometry and derivative oracle at identical DOFs.
"""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax.core import (
        CurveXYZFourierSymmetriesSpec,
        curve_gamma_vjp_from_dofs,
        curve_gammadash_vjp_from_dofs,
        curve_gammadashdash_vjp_from_dofs,
        curve_gammadashdashdash_vjp_from_dofs,
        curve_spec_from_curve,
        make_curve_xyzfouriersymmetries_spec,
    )
    from simsopt_jax.backend import invalidate_backend_cache
    from simsopt_jax.core.curve_geometry import (
        _slice_1d_static,
        _update_1d_static,
        curve_gamma_and_gammadash_from_dofs,
        curve_geometry_from_dofs,
        curve_pullback_from_dofs,
    )
    from simsopt_jax.core.specs import curve_spec_kind
from unittest import mock
import os


import numpy as np


from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries


_DIRECT_KERNEL = {
    "rtol": 1e-10,
    "atol": 1e-12,
    "requires_same_state": True,
    "requires_direct_cpp_oracle": True,
    "vector_parity_required": True,
}
_DIRECT_RTOL = _DIRECT_KERNEL["rtol"]
_DIRECT_ATOL = _DIRECT_KERNEL["atol"]

_DERIV_HEAVY = {
    "scalar_value_rtol": 1e-10,
    "scalar_value_atol": 1e-12,
    "first_derivative_rtol": 1e-08,
    "first_derivative_atol": 1e-10,
    "second_derivative_rtol": 1e-06,
    "second_derivative_atol": 1e-08,
    "requires_same_input": True,
    "requires_direct_cpp_oracle": True,
    "fd_validation_secondary": True,
}
_DERIV_RTOL = _DERIV_HEAVY["first_derivative_rtol"]
_DERIV_ATOL = _DERIV_HEAVY["first_derivative_atol"]
_SECOND_DERIV_RTOL = _DERIV_HEAVY["second_derivative_rtol"]
_SECOND_DERIV_ATOL = _DERIV_HEAVY["second_derivative_atol"]


_NQUADPOINTS = 64
_ORDER = 3
_RAND_SCALE = 1e-2
_RNG_SEED = 1729

if JAX_IMPORT_ERROR is None:
    _CURVE_VJP_CASES = (
        ("gamma", curve_gamma_vjp_from_dofs, "dgamma_by_dcoeff"),
        ("gammadash", curve_gammadash_vjp_from_dofs, "dgammadash_by_dcoeff"),
        (
            "gammadashdash",
            curve_gammadashdash_vjp_from_dofs,
            "dgammadashdash_by_dcoeff",
        ),
        (
            "gammadashdashdash",
            curve_gammadashdashdash_vjp_from_dofs,
            "dgammadashdashdash_by_dcoeff",
        ),
    )


def _stellsym_num_dofs(order: int) -> int:
    return (order + 1) + order + order


def _non_stellsym_num_dofs(order: int) -> int:
    return 3 * (2 * order + 1)


def _seed_stellsym_dofs(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = _stellsym_num_dofs(order)
    dofs = np.zeros(ndofs, dtype=np.float64)
    # xc(0) sets the loop radius scale; xc(1), ys(1), zs(1) are the
    # dominant first-harmonic amplitudes for a helical coil.
    dofs[0] = 1.0
    dofs[1] = 0.25
    dofs[order + 1] = 0.25
    dofs[2 * order + 1] = 0.25
    return dofs + _RAND_SCALE * rng.random(ndofs)


def _seed_non_stellsym_dofs(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = _non_stellsym_num_dofs(order)
    dofs = np.zeros(ndofs, dtype=np.float64)
    # xc(0), yc(0) place the loop centre; xc(1) and ys(1) provide the
    # first-harmonic shape.
    dofs[0] = 1.0
    dofs[1] = 0.25
    dofs[2 * order + 1] = 0.0
    dofs[3 * order + 2] = 0.25
    return dofs + _RAND_SCALE * rng.random(ndofs)


def _build_curve(
    *,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
    rng: np.random.Generator,
    nquadpoints: int = _NQUADPOINTS,
) -> CurveXYZFourierSymmetries:
    curve = CurveXYZFourierSymmetries(
        nquadpoints,
        order,
        nfp=nfp,
        stellsym=stellsym,
        ntor=ntor,
    )
    if stellsym:
        dofs = _seed_stellsym_dofs(order, rng)
    else:
        dofs = _seed_non_stellsym_dofs(order, rng)
    curve.x = dofs
    return curve


def _curve_spec(curve: CurveXYZFourierSymmetries) -> CurveXYZFourierSymmetriesSpec:
    return make_curve_xyzfouriersymmetries_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
        ntor=curve.ntor,
    )


_FORWARD_PARITY_CASES = [
    ("3-2-True", (3, 2, True)),
    ("3-2-False", (3, 2, False)),
    ("5-3-True", (5, 3, True)),
    ("5-3-False", (5, 3, False)),
    ("4-1-True", (4, 1, True)),
    ("7-4-True", (7, 4, True)),
]


def _numpy_gamma_oracle(
    dofs: np.ndarray,
    quadpoints: np.ndarray,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
) -> np.ndarray:
    """Pure-NumPy ``gamma`` reference for ``CurveXYZFourierSymmetries``.

    Implements the closed-form Fourier series documented at
    ``src/simsopt/geo/curvexyzfouriersymmetries.py:66-90`` without
    sharing any code with ``jaxXYZFourierSymmetriescurve_pure``. Acts as
    the independent oracle for the JAX kernel parity tests.
    """
    theta = np.asarray(quadpoints, dtype=np.float64)
    angles_full = 2.0 * np.pi * nfp * np.outer(theta, np.arange(order + 1))
    cos_full = np.cos(angles_full)  # (N, order+1)
    sin_tail = np.sin(angles_full[:, 1:])  # (N, order)

    if stellsym:
        xc = dofs[: order + 1]
        ys = dofs[order + 1 : 2 * order + 1]
        zs = dofs[2 * order + 1 :]
        hat_x = cos_full @ xc
        hat_y = sin_tail @ ys
        z = sin_tail @ zs
    else:
        xc = dofs[0 : order + 1]
        xs = dofs[order + 1 : 2 * order + 1]
        yc = dofs[2 * order + 1 : 3 * order + 2]
        ys = dofs[3 * order + 2 : 4 * order + 2]
        zc = dofs[4 * order + 2 : 5 * order + 3]
        zs = dofs[5 * order + 3 :]
        hat_x = cos_full @ xc + sin_tail @ xs
        hat_y = cos_full @ yc + sin_tail @ ys
        z = cos_full @ zc + sin_tail @ zs

    tor = 2.0 * np.pi * theta * ntor
    cos_tor = np.cos(tor)
    sin_tor = np.sin(tor)
    x = hat_x * cos_tor - hat_y * sin_tor
    y = hat_x * sin_tor + hat_y * cos_tor
    return np.stack((x, y, z), axis=1).astype(np.float64)


class TestCurvexyzfouriersymmetriesSpecJax(JaxTestCase):
    def test_spec_gamma_byte_identical_to_curve_gamma(self):
        """Spec-driven gamma matches ``curve.gamma()`` at the direct-kernel lane."""
        for case_id_0, (nfp, ntor, stellsym) in _FORWARD_PARITY_CASES:
            with self.subTest(
                case_id_0=case_id_0, nfp=nfp, ntor=ntor, stellsym=stellsym
            ), self.case():
                self._case_spec_gamma_byte_identical_to_curve_gamma(nfp, ntor, stellsym)

    def _case_spec_gamma_byte_identical_to_curve_gamma(self, nfp, ntor, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED)
        curve = _build_curve(
            order=_ORDER,
            nfp=nfp,
            stellsym=stellsym,
            ntor=ntor,
            rng=rng,
        )
        spec = _curve_spec(curve)
        self.assertTrue(
            isinstance(spec, CurveXYZFourierSymmetriesSpec),
            "isinstance(spec, CurveXYZFourierSymmetriesSpec)",
        )

        gamma_curve = np.asarray(curve.gamma(), dtype=np.float64)
        gamma_jax = np.asarray(
            curve_geometry_from_dofs(spec, spec.dofs)[0], dtype=np.float64
        )
        # Independent NumPy oracle (closed-form series from the docstring;
        # shares no code with `jaxXYZFourierSymmetriescurve_pure`).
        gamma_numpy = _numpy_gamma_oracle(
            np.asarray(spec.dofs, dtype=np.float64),
            np.asarray(spec.quadpoints, dtype=np.float64),
            order=int(spec.order),
            nfp=int(spec.nfp),
            stellsym=bool(spec.stellsym),
            ntor=int(spec.ntor),
        )

        self.assertTrue(
            gamma_curve.shape == (_NQUADPOINTS, 3),
            "gamma_curve.shape == (_NQUADPOINTS, 3)",
        )
        self.assertTrue(
            gamma_jax.shape == (_NQUADPOINTS, 3),
            "gamma_jax.shape == (_NQUADPOINTS, 3)",
        )
        self.assertTrue(
            gamma_numpy.shape == (_NQUADPOINTS, 3),
            "gamma_numpy.shape == (_NQUADPOINTS, 3)",
        )
        # Both JaxCurve route and the new spec route must agree with the
        # independent oracle (these two paths share the JAX kernel, so an
        # algebraic regression in `jaxXYZFourierSymmetriescurve_pure` would
        # fail the oracle comparison even though they would still agree with
        # each other).
        np.testing.assert_allclose(
            gamma_jax,
            gamma_numpy,
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )
        np.testing.assert_allclose(
            gamma_curve,
            gamma_numpy,
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )

    def test_explicit_spec_round_trips_fields(self):
        """Explicit spec construction mirrors the host class state."""
        for case_id_0, stellsym in zip(
            ["stellsym=True", "stellsym=False"], [True, False], strict=True
        ):
            with self.subTest(case_id_0=case_id_0, stellsym=stellsym), self.case():
                self._case_explicit_spec_round_trips_fields(stellsym)

    def _case_explicit_spec_round_trips_fields(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 1)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)

        self.assertTrue(
            isinstance(spec, CurveXYZFourierSymmetriesSpec),
            "isinstance(spec, CurveXYZFourierSymmetriesSpec)",
        )
        self.assertTrue(spec.order == curve.order, "spec.order == curve.order")
        self.assertTrue(spec.nfp == curve.nfp, "spec.nfp == curve.nfp")
        self.assertTrue(
            spec.stellsym == curve.stellsym, "spec.stellsym == curve.stellsym"
        )
        self.assertTrue(spec.ntor == curve.ntor, "spec.ntor == curve.ntor")

        np.testing.assert_array_equal(
            np.asarray(spec.dofs),
            np.asarray(curve.get_dofs(), dtype=np.float64),
        )
        np.testing.assert_array_equal(
            np.asarray(spec.quadpoints),
            np.asarray(curve.quadpoints, dtype=np.float64),
        )

        # ``curve_spec_kind`` discriminator must recognize the new variant.
        self.assertTrue(
            curve_spec_kind(spec) == "xyz_fourier_symmetries",
            'curve_spec_kind(spec) == "xyz_fourier_symmetries"',
        )

    def test_curve_spec_from_curve_rejects_without_adapter_hook(self):
        """Legacy host curves without adapter hooks stay outside the implicit boundary."""
        rng = np.random.default_rng(_RNG_SEED + 2)
        curve = _build_curve(
            order=_ORDER,
            nfp=5,
            stellsym=True,
            ntor=3,
            rng=rng,
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "CurveXYZFourierSymmetries does not expose an immutable JAX spec",
        ):
            curve_spec_from_curve(curve)

    def test_spec_higher_derivatives_match_curve(self):
        """``curve_geometry_from_dofs`` ladder matches CPU oracle for gamma/dash/dashdash."""
        for case_id_0, stellsym in zip(
            ["stellsym=True", "stellsym=False"], [True, False], strict=True
        ):
            with self.subTest(case_id_0=case_id_0, stellsym=stellsym), self.case():
                self._case_spec_higher_derivatives_match_curve(stellsym)

    def _case_spec_higher_derivatives_match_curve(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 3)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)

        gamma_cpu = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash_cpu = np.asarray(curve.gammadash(), dtype=np.float64)
        gammadashdash_cpu = np.asarray(curve.gammadashdash(), dtype=np.float64)

        gamma_jax, gammadash_jax, gammadashdash_jax = (
            np.asarray(term, dtype=np.float64)
            for term in curve_geometry_from_dofs(spec, spec.dofs)
        )

        np.testing.assert_allclose(
            gamma_jax,
            gamma_cpu,
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )
        np.testing.assert_allclose(
            gammadash_jax,
            gammadash_cpu,
            rtol=_DERIV_RTOL,
            atol=_DERIV_ATOL,
        )
        np.testing.assert_allclose(
            gammadashdash_jax,
            gammadashdash_cpu,
            rtol=_SECOND_DERIV_RTOL,
            atol=_SECOND_DERIV_ATOL,
        )

    def test_curve_pullback_shape(self):
        """VJP cotangent has the shape of ``spec.dofs``."""
        for case_id_0, stellsym in zip(
            ["stellsym=True", "stellsym=False"], [True, False], strict=True
        ):
            with self.subTest(case_id_0=case_id_0, stellsym=stellsym), self.case():
                self._case_curve_pullback_shape(stellsym)

    def _case_curve_pullback_shape(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 4)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)

        gamma_jax, gammadash_jax = curve_gamma_and_gammadash_from_dofs(spec, spec.dofs)
        cotangent_g = jnp.ones_like(gamma_jax)
        cotangent_gd = jnp.ones_like(gammadash_jax)

        coeff_cotangent = curve_pullback_from_dofs(
            spec, spec.dofs, cotangent_g, cotangent_gd
        )
        self.assertTrue(
            coeff_cotangent.shape == spec.dofs.shape,
            "coeff_cotangent.shape == spec.dofs.shape",
        )
        # The all-ones cotangent must produce a finite, non-zero result for a
        # non-degenerate curve. Zero would mean the pullback misrouted.
        coeff_np = np.asarray(coeff_cotangent, dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(coeff_np)), "np.all(np.isfinite(coeff_np))")
        self.assertTrue(
            np.linalg.norm(coeff_np) > 0.0, "np.linalg.norm(coeff_np) > 0.0"
        )

    def test_named_geometry_vjp_wrappers_match_cpu_derivatives(self):
        """Named geometry VJP wrappers match CPU derivative tensor contractions."""
        for case_id_0, (term_name, vjp_fn, derivative_method) in zip(
            [name for name, _fn, _method in _CURVE_VJP_CASES],
            _CURVE_VJP_CASES,
            strict=True,
        ):
            for case_id_1, stellsym in zip(
                ["stellsym=True", "stellsym=False"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    term_name=term_name,
                    vjp_fn=vjp_fn,
                    derivative_method=derivative_method,
                    case_id_1=case_id_1,
                    stellsym=stellsym,
                ), self.case():
                    self._case_named_geometry_vjp_wrappers_match_cpu_derivatives(
                        term_name, vjp_fn, derivative_method, stellsym
                    )

    def _case_named_geometry_vjp_wrappers_match_cpu_derivatives(
        self, term_name, vjp_fn, derivative_method, stellsym
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 5 + int(stellsym))
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)
        derivative_cpu = np.asarray(
            getattr(curve, derivative_method)(),
            dtype=np.float64,
        )
        cotangent = rng.normal(size=derivative_cpu.shape[:2])

        vjp_jax = np.asarray(
            vjp_fn(spec, spec.dofs, jnp.asarray(cotangent, dtype=jnp.float64)),
            dtype=np.float64,
        )
        vjp_cpu = np.einsum("ij,ijk->k", cotangent, derivative_cpu)

        np.testing.assert_allclose(
            vjp_jax,
            vjp_cpu,
            rtol=_DERIV_RTOL,
            atol=_DERIV_ATOL,
            err_msg=f"CurveXYZFourierSymmetries {term_name} named VJP mismatch.",
        )

    def test_spec_geometry_runs_under_strict_transfer_guard(self):
        """Spec-driven geometry executes under ``jax.transfer_guard('disallow')``.

        The spec is placed on-device via ``jax.device_put`` and a compiled
        callable evaluates ``curve_geometry_from_dofs`` without implicit
        host-to-device transfers.
        """
        rng = np.random.default_rng(_RNG_SEED + 5)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=True,
            ntor=2,
            rng=rng,
        )
        spec = jax.device_put(_curve_spec(curve))

        @jax.jit
        def evaluate(spec_arg):
            gamma, gammadash, gammadashdash = curve_geometry_from_dofs(
                spec_arg, spec_arg.dofs
            )
            return gamma, gammadash, gammadashdash

        compiled = evaluate.lower(spec).compile()
        with jax.transfer_guard("disallow"):
            gamma, gammadash, gammadashdash = compiled(spec)

        self.assertTrue(
            gamma.shape == (_NQUADPOINTS, 3), "gamma.shape == (_NQUADPOINTS, 3)"
        )
        self.assertTrue(
            gammadash.shape == (_NQUADPOINTS, 3), "gammadash.shape == (_NQUADPOINTS, 3)"
        )
        self.assertTrue(
            gammadashdash.shape == (_NQUADPOINTS, 3),
            "gammadashdash.shape == (_NQUADPOINTS, 3)",
        )
        self.assertTrue(
            np.all(np.isfinite(np.asarray(gamma))),
            "np.all(np.isfinite(np.asarray(gamma)))",
        )
        self.assertTrue(
            np.all(np.isfinite(np.asarray(gammadash))),
            "np.all(np.isfinite(np.asarray(gammadash)))",
        )
        self.assertTrue(
            np.all(np.isfinite(np.asarray(gammadashdash))),
            "np.all(np.isfinite(np.asarray(gammadashdash)))",
        )

    def test_static_slice_and_update_grad_run_under_strict_transfer_guard(self):
        """Compiled static slices and updates have exact values and gradients under a
        strict transfer guard."""
        values_host = np.linspace(-2.0, 3.0, num=6, dtype=np.float64)
        replacement_host = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        values = jax.device_put(jnp.asarray(values_host))
        replacement = jax.device_put(jnp.asarray(replacement_host))

        def objective(values_arg, replacement_arg):
            segment = _slice_1d_static(values_arg, 2, 5)
            updated = _update_1d_static(values_arg, 1, replacement_arg)
            return jnp.sum(segment * segment) + jnp.sum(updated)

        value_fn = jax.jit(objective).lower(values, replacement).compile()
        grad_fn = (
            jax.jit(jax.grad(objective, argnums=(0, 1)))
            .lower(values, replacement)
            .compile()
        )

        with jax.transfer_guard("disallow"):
            actual_value = value_fn(values, replacement)
            grad_values, grad_replacement = grad_fn(values, replacement)

        expected_updated = np.concatenate(
            [values_host[:1], replacement_host, values_host[4:]]
        )
        expected_value = np.sum(values_host[2:5] ** 2) + np.sum(expected_updated)
        expected_grad_values = np.zeros_like(values_host)
        expected_grad_values[2:5] += 2.0 * values_host[2:5]
        expected_grad_values[:1] += 1.0
        expected_grad_values[4:] += 1.0

        np.testing.assert_allclose(
            np.asarray(actual_value), expected_value, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(grad_values), expected_grad_values, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(grad_replacement),
            np.ones_like(replacement_host),
            rtol=0.0,
            atol=0.0,
        )

    def test_static_slice_and_update_preserve_input_dtype_under_fp64_runtime(self):
        """Regression: every staged operand must follow the operand dtype.

        A runtime-dtype (fp64) scalar inside ``_update_1d_static`` promoted the
        whole masked update — and via ``_mapped_full_dofs`` the mapped coil dofs
        and the downstream Biot-Savart chain — to fp64 for fp32 operands.
        """
        patches = self.patches
        patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_BACKEND_MODE": "jax_cpu_fast"})
        )
        invalidate_backend_cache()
        try:
            array_host = np.linspace(-2.0, 3.0, num=6, dtype=np.float32)
            replacement_host = np.array([10.0, 20.0, 30.0], dtype=np.float32)
            array = jax.device_put(jnp.asarray(array_host))
            replacement = jax.device_put(jnp.asarray(replacement_host))

            segment = _slice_1d_static(array, 2, 5)
            updated = _update_1d_static(array, 1, replacement)

            self.assertTrue(
                segment.dtype == jnp.float32, "segment.dtype == jnp.float32"
            )
            self.assertTrue(
                updated.dtype == jnp.float32, "updated.dtype == jnp.float32"
            )
            np.testing.assert_allclose(
                np.asarray(updated),
                np.concatenate([array_host[:1], replacement_host, array_host[4:]]),
                rtol=0.0,
                atol=0.0,
            )
        finally:
            invalidate_backend_cache()

    def test_coprime_invariant_unchanged(self):
        """The host-class constructor still enforces ``gcd(ntor, nfp) == 1``."""
        with self.assertRaisesRegex(Exception, "nfp and ntor must be coprime"):
            CurveXYZFourierSymmetries(
                _NQUADPOINTS, _ORDER, nfp=4, stellsym=True, ntor=2
            )

    def test_spec_constructor_enforces_coprime_invariant(self):
        """The explicit spec constructor preserves the same coprime contract."""
        with self.assertRaisesRegex(ValueError, "requires nfp and ntor coprime"):
            make_curve_xyzfouriersymmetries_spec(
                dofs=np.zeros(_stellsym_num_dofs(_ORDER), dtype=np.float64),
                quadpoints=np.linspace(0.0, 1.0, _NQUADPOINTS, endpoint=False),
                order=_ORDER,
                nfp=4,
                stellsym=True,
                ntor=2,
            )
