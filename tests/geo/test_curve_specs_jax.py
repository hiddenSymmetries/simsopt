"""JAX curve specs against native curves and independent NumPy references."""

from __future__ import annotations

import functools

from unittest_jax_support import JaxTestCase, host_array, optional_jax_imports, parity_default_device
from unittest_jax_support import parity_rng, require_cuda

with optional_jax_imports():
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend import set_backend
    from simsopt_jax.backend.jit import jit
    from simsopt_jax import core
    from simsopt_jax.core.curve_geometry import _curve_geometry_terms_from_dofs, _slice_1d_static, _update_1d_static
    from simsopt_jax.core.curve_planar_fourier import curveplanarfourier_pure
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

import numpy as np
from numpy.random import PCG64DXSM, Generator
from numpy.testing import assert_allclose, assert_array_equal
from simsopt.geo import CurveFilament, CurvePerturbed, CurveXYZFourier, FrameRotation, GaussianSampler
from simsopt.geo import PerturbationSample, RotatedCurve
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.framedcurve import FramedCurve, FramedCurveCentroid, FramedCurveFrenet

_close_gamma = functools.partial(assert_allclose, rtol=1e-10, atol=1e-10)
_close_deriv = functools.partial(assert_allclose, rtol=1e-9, atol=1e-9)
_LANES = [dict(case_id_0=f"{lane}_parity", parity_lane=lane) for lane in ("cpu", "gpu")]


def _for_cases(cases):
    """Run the test per case dict in its own subTest, runtime isolation and parity lane."""

    def decorate(body):
        @functools.wraps(body)
        def test(self):
            for case in cases:
                with self.subTest(**case), self.case() as patches:
                    if "parity_lane" in case:
                        patches.enter_context(parity_default_device(case["parity_lane"]))
                    body(self, case)

        return test

    return decorate


def _noisy(curve, dofs, seed):
    """Set ``curve``'s DOFs to ``dofs`` plus seeded noise of scale 0.01; return it."""
    curve.x = dofs + 0.01 * parity_rng(seed).normal(size=dofs.shape)
    return curve


def _make_helical_curve(seed: int) -> CurveHelical:
    curve = CurveHelical(24, 4, 5, 2, 1.0, 0.3)
    dofs = np.zeros(curve.dof_size)
    dofs[0] = np.pi / 2
    return _noisy(curve, dofs, seed)


def _make_planar_curve(seed: int) -> CurvePlanarFourier:
    curve = CurvePlanarFourier(20, 3)
    dofs = np.zeros(curve.dof_size)
    dofs[[0, 1, 4]] = 1.0, 0.1, 0.1
    # Native and JAX normalize the quaternion by its exact norm, without epsilon.
    dofs[7:11] = 0.9, 0.2, -0.15, 0.05
    dofs[-3:] = 2.0, -0.5, 0.3
    return _noisy(curve, dofs, seed)


def _make_native_jax_curves(seed: int):
    """Native JaxCurve twins of CurveXYZFourier and CurvePlanarFourier."""
    xyz = JaxCurveXYZFourier(24, 3)
    dofs = 0.05 * parity_rng(seed).normal(size=xyz.dof_size)
    dofs[[xyz.local_dof_names.index(name) for name in ("xc(1)", "ys(1)", "zs(1)")]] += (1.0, 1.0, 0.3)
    xyz.x = dofs
    planar = JaxCurvePlanarFourier(20, 3)
    planar.x = _make_planar_curve(seed).get_dofs()
    return xyz, planar


def _make_rzfourier_curve(seed: int, *, stellsym: bool) -> CurveRZFourier:
    curve = CurveRZFourier(20, 3, 3, stellsym)
    dofs = np.zeros(curve.dof_size)
    dofs[:2] = 1.3, 0.15
    return _noisy(curve, dofs, seed)


def _assert_spec_geometry_matches_native(curve, spec) -> None:
    """Compare position, tangent and second derivative with the native oracle."""
    gamma, dash, dashdash = core.curve_geometry_from_dofs(spec, spec.dofs)
    _close_gamma(host_array(gamma), curve.gamma())
    _close_deriv(host_array(dash), curve.gammadash())
    _close_deriv(host_array(dashdash), curve.gammadashdash())


def _assert_dof_round_trip(curve, *, seed: int) -> None:
    """The adapter passes DOFs through, and geometry matches native at two DOF sets."""
    original_dofs = np.array(curve.get_dofs(), copy=True)
    spec = curve_spec_from_adapter_curve(curve)
    assert_array_equal(host_array(spec.dofs), original_dofs)
    _assert_spec_geometry_matches_native(curve, spec)
    curve.x = original_dofs + 0.02 * parity_rng(seed).normal(size=original_dofs.shape)
    updated_spec = curve_spec_from_adapter_curve(curve)
    assert_array_equal(host_array(updated_spec.dofs), curve.get_dofs())
    _assert_spec_geometry_matches_native(curve, updated_spec)


class TestCurveSubclassesParity(JaxTestCase):
    """Helical, planar and RZ Fourier specs match native on the CPU and GPU lanes."""

    @_for_cases(_LANES)
    def test_curve_helical_dof_round_trip_matches_native(self, case):
        """Helical native DOF setters round-trip to matching immutable spec geometry."""
        _assert_dof_round_trip(_make_helical_curve(seed=103), seed=203)

    @_for_cases(_LANES)
    def test_curve_planarfourier_dof_round_trip_matches_native(self, case):
        """Planar Fourier native DOF setters round-trip to matching spec geometry."""
        _assert_dof_round_trip(_make_planar_curve(seed=113), seed=213)

    @_for_cases(_LANES)
    def test_curve_planarfourier_small_quaternion_matches_native(self, case):
        """Planar geometry with a very small quaternion matches native's rotation."""
        curve = _make_planar_curve(seed=114)
        dofs = np.array(curve.get_dofs(), copy=True)
        q_start = 2 * curve.order + 1
        dofs[q_start : q_start + 4] = np.array([1.0, 2.0, 3.0, 4.0]) * 1e-12
        curve.x = dofs
        gamma = curveplanarfourier_pure(np.asarray(curve.get_dofs()), np.asarray(curve.quadpoints), curve.order)
        _close_gamma(host_array(gamma), curve.gamma())
        _assert_spec_geometry_matches_native(curve, curve_spec_from_adapter_curve(curve))

    @_for_cases([{**lane, "case_id_1": name, "stellsym": stellsym}
                 for lane in _LANES for name, stellsym in (("stellsym", True), ("non_stellsym", False))])
    def test_curve_rzfourier_dof_round_trip_matches_native(self, case):
        """RZ Fourier native DOF setters round-trip to matching spec geometry."""
        stellsym = case["stellsym"]
        curve = _make_rzfourier_curve(125 if stellsym else 126, stellsym=stellsym)
        _assert_dof_round_trip(curve, seed=225 if stellsym else 226)

    @_for_cases(_LANES)
    def test_native_jax_curve_twins_round_trip_to_matching_specs(self, case):
        """JaxCurveXYZFourier and JaxCurvePlanarFourier map to their curve specs."""
        for curve in _make_native_jax_curves(seed=130):
            _assert_dof_round_trip(curve, seed=230)


_NQUAD = 128
# Spectral-derivative references resolve the Frenet frame to round-off only
# from 256 points (at 128 the spectral error of gammadashdash is 2e-8).
_SPECTRAL_NQUAD = 256
_FRAMES = (FramedCurveCentroid, FramedCurveFrenet)
_BASE = {"xc(1)": 1.0, "ys(1)": 1.0, "zs(2)": 0.2, "xs(3)": 0.05}
_PERTURBATION = {"xs(2)": 0.03, "yc(3)": -0.02, "zs(1)": 0.04}


def _xyz_curve(coefficients: dict[str, float], nquad: int = _NQUAD):
    curve = CurveXYZFourier(nquad, 3)
    dofs = np.zeros(curve.dof_size)
    for name, value in coefficients.items():
        dofs[curve.local_dof_names.index(name)] = value
    curve.x = dofs
    return curve


def _gaussian_sample(curve, n_derivs: int) -> PerturbationSample:
    """Seeded Gaussian-process sample, so value comparisons see one fixed draw."""
    sampler = GaussianSampler(curve.quadpoints, 0.05, 0.3, n_derivs=n_derivs)
    return PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(670)))


def _perturbed_and_equivalent() -> tuple[CurvePerturbed, CurveXYZFourier]:
    """Base plus a Fourier sample with three exact derivatives, and the summed curve."""
    p = _xyz_curve(_PERTURBATION)
    sample = [p.gamma(), p.gammadash(), p.gammadashdash(), p.gammadashdashdash()]
    perturbed = CurvePerturbed(_xyz_curve(_BASE), PerturbationSample(None, sample=sample))
    # The two coefficient sets share no mode, so their union is their sum.
    return perturbed, _xyz_curve({**_BASE, **_PERTURBATION})


def _filament(curve, frame) -> CurveFilament:
    # A non-unit scale multiplies every rotation-angle derivative the frame uses.
    rotation = FrameRotation(curve.quadpoints, 2, scale=0.7)
    rotation.x = np.array([0.1, 0.2, 0.3, -0.1, 0.05])
    return CurveFilament(frame(curve, rotation), 0.02, 0.03)


def _spectral_derivative(values: np.ndarray) -> np.ndarray:
    """d/dphi of periodic samples on the uniform grid phi in [0, 1)."""
    wavenumbers = np.fft.fftfreq(values.shape[0], d=1.0 / values.shape[0])
    spectrum = 2j * np.pi * wavenumbers[:, None] * np.fft.fft(values, axis=0)
    return np.real(np.fft.ifft(spectrum, axis=0))


def _terms(spec, order: int) -> list[np.ndarray]:
    if order == 1:
        return [host_array(t) for t in core.curve_gamma_and_gammadash_from_dofs(spec, spec.dofs)]
    return [host_array(t) for t in core.curve_geometry_from_dofs(spec, spec.dofs)]


def _assert_spectral_ladder(gamma, gammadash, gammadashdash) -> None:
    """Each derivative is the spectral derivative of the term below it."""
    assert_allclose(_spectral_derivative(gamma), gammadash, rtol=0, atol=1e-11)
    assert_allclose(_spectral_derivative(gammadash), gammadashdash, rtol=0, atol=1e-11)


class TestCurveWrapperSpecs(JaxTestCase):
    """Perturbed and filament specs follow native geometry and native errors."""

    def test_filament_over_perturbed_curve_matches_summed_curve(self):
        """Filament terms over a perturbed curve include the sample derivatives."""
        perturbed, summed = _perturbed_and_equivalent()
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                filament = _filament(perturbed, frame)
                spec = curve_spec_from_adapter_curve(filament)
                reference = curve_spec_from_adapter_curve(_filament(summed, frame))
                order1 = _terms(spec, 1)
                assert_allclose(order1[1], filament.gammadash(), rtol=0, atol=1e-13)
                for term, expected in zip(order1, _terms(reference, 1), strict=True):
                    assert_allclose(term, expected, rtol=0, atol=1e-13)
                if frame is FramedCurveFrenet:
                    # The Frenet second derivative needs a fourth sample derivative.
                    with self.assertRaisesRegex(ValueError, "derivative 4 is needed"):
                        core.curve_geometry_from_dofs(spec, spec.dofs)
                    continue
                order2 = _terms(spec, 2)
                for term, expected in zip(order2, _terms(reference, 2), strict=True):
                    assert_allclose(term, expected, rtol=0, atol=1e-11)
                assert_allclose(order2[1], filament.gammadash(), rtol=0, atol=1e-13)

    def test_filament_second_derivative_is_derivative_of_tangent(self):
        """gammadashdash equals the spectral derivative of gammadash for both frames."""
        curve = _xyz_curve({**_BASE, **_PERTURBATION}, nquad=_SPECTRAL_NQUAD)
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                spec = curve_spec_from_adapter_curve(_filament(curve, frame))
                _assert_spectral_ladder(*_terms(spec, 2))

    def test_filament_over_filament_derivatives_are_derivatives_of_lower_terms(self):
        """Nested filaments: each derivative is the spectral derivative of the term below; native evaluates only the
        position of a centroid over a filament."""
        inner = _filament(_xyz_curve(_BASE, nquad=_SPECTRAL_NQUAD), FramedCurveCentroid)
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                outer = _filament(inner, frame)
                terms = _terms(curve_spec_from_adapter_curve(outer), 2)
                if frame is FramedCurveCentroid:
                    assert_allclose(terms[0], outer.gamma(), rtol=0, atol=1e-13)
                _assert_spectral_ladder(*terms)

    def test_sample_without_frame_derivatives_raises_like_native(self):
        """A sample lacking a derivative the frame needs is rejected, as native raises."""
        base = _xyz_curve(_BASE)
        # (n_derivs, frame) -> native rejects: centroid needs 2, Frenet 3.
        for n_derivs, frame, native_raises in ((1, FramedCurveCentroid, True), (1, FramedCurveFrenet, True),
                                               (2, FramedCurveFrenet, True), (2, FramedCurveCentroid, False)):
            with self.subTest(n_derivs=n_derivs, frame=frame.__name__):
                filament = _filament(CurvePerturbed(base, _gaussian_sample(base, n_derivs)), frame)
                if not native_raises:
                    tangent = _terms(curve_spec_from_adapter_curve(filament), 1)[1]
                    assert_allclose(tangent, filament.gammadash(), rtol=0, atol=1e-13)
                    continue
                with self.assertRaises(ValueError):
                    filament.gammadash()
                with self.assertRaisesRegex(ValueError, f"has {n_derivs} derivatives"):
                    curve_spec_from_adapter_curve(filament)

    def test_perturbed_curve_without_second_sample_derivative_raises_like_native(self):
        """Order-2 geometry of a perturbed curve needs the sample's second derivative."""
        base = _xyz_curve(_BASE)
        perturbed = CurvePerturbed(base, _gaussian_sample(base, 1))
        spec = curve_spec_from_adapter_curve(perturbed)
        assert_allclose(_terms(spec, 1)[1], perturbed.gammadash(), rtol=0, atol=1e-13)
        with self.assertRaises(ValueError):
            perturbed.gammadashdash()
        with self.assertRaisesRegex(ValueError, "has 1 derivatives, but derivative 2"):
            core.curve_geometry_from_dofs(spec, spec.dofs)

    def test_unsupported_frames_and_wrapper_orders_raise(self):
        """Custom frames, unknown frame kinds and inner RotatedCurves are rejected."""
        base = _xyz_curve(_BASE)
        custom_frame = CurveFilament(FramedCurve(base), 0.02, 0.03)
        with self.assertRaisesRegex(NotImplementedError, "FramedCurve"):
            curve_spec_from_adapter_curve(custom_frame)
        valid = curve_spec_from_adapter_curve(_filament(base, FramedCurveCentroid))
        names = "dofs quadpoints base_curve base_curve_map rotation rotation_map dn db"
        fields = {name: getattr(valid, name) for name in names.split()}
        with self.assertRaisesRegex(ValueError, "frame_kind"):
            core.make_curve_filament_spec(frame_kind="custom", **fields)
        rotated = RotatedCurve(base, np.pi / 3, flip=False)
        for inner_rotation in (CurvePerturbed(rotated, _gaussian_sample(base, 3)),
                               _filament(rotated, FramedCurveCentroid)):
            with self.subTest(wrapper=type(inner_rotation).__name__), \
                    self.assertRaisesRegex(NotImplementedError, "RotatedCurve"):
                curve_spec_from_adapter_curve(inner_rotation)

    def test_dof_map_segments_must_have_equal_lengths(self):
        """Owner and target ranges of a DOF-map segment must match and fit the template."""
        for segment in ((0, 3, 1, 3), (0, 2, 3, 5)):
            with self.subTest(segment=segment), self.assertRaisesRegex(ValueError, "equal-length target range"):
                core.make_optimizable_dof_map_spec(template_full_dofs=np.zeros(4), owner_segments=(segment,),
                                                   input_mode="full", input_start=0, input_end=4)

    def test_wrapper_dof_map_templates_follow_the_runtime_device(self):
        """Under the JAX CPU backend in a CUDA process, wrapper DOF-map templates join the other spec leaves on the
        runtime CPU device, not on the GPU."""
        require_cuda("CUDA device required for a runtime device other than the GPU")
        set_backend("jax", device="cpu")
        cpu = jax.devices("cpu")[0]
        base = _xyz_curve(_BASE)
        for wrapper in (CurvePerturbed(base, _gaussian_sample(base, 1)), _filament(base, FramedCurveCentroid)):
            with self.subTest(wrapper=type(wrapper).__name__):
                spec = curve_spec_from_adapter_curve(wrapper)
                devices = {leaf.devices().pop() for leaf in jax.tree.leaves(spec)}
                self.assertEqual(devices, {cpu})


_close_direct = functools.partial(assert_allclose, rtol=1e-10, atol=1e-12)
_close_first_deriv = functools.partial(assert_allclose, rtol=1e-08, atol=1e-10)
_close_second_deriv = functools.partial(assert_allclose, rtol=1e-06, atol=1e-08)
_NQUADPOINTS, _ORDER, _RNG_SEED = 64, 3, 1729
_SPEC_FIELDS = ("order", "nfp", "stellsym", "ntor")
_rng = lambda offset: np.random.default_rng(_RNG_SEED + offset)
_STELLSYM_CASES = [dict(case_id_0=f"stellsym={s}", stellsym=s) for s in (True, False)]
_FORWARD_PARITY_CASES = [dict(case_id_0=f"{nfp}-{ntor}-{stellsym}", nfp=nfp, ntor=ntor, stellsym=stellsym)
                         for nfp, ntor, stellsym in [(3, 2, True), (3, 2, False), (5, 3, True), (5, 3, False),
                                                     (4, 1, True), (7, 4, True)]]
_VJP_CASES = [dict(case_id_0=name, term_name=name, term_index=index, derivative_method=f"d{name}_by_dcoeff",
                   case_id_1=f"stellsym={s}", stellsym=s)
              for index, name in enumerate(("gamma", "gammadash", "gammadashdash", "gammadashdashdash"))
              for s in (True, False)]


def _build_curve(nfp, ntor, stellsym, rng):
    """Order-3 symmetric curve with first-harmonic DOFs plus 0.01 uniform noise."""
    curve = CurveXYZFourierSymmetries(_NQUADPOINTS, _ORDER, nfp, stellsym, ntor)
    o = _ORDER
    dofs = np.zeros(3 * o + 1 if stellsym else 3 * (2 * o + 1), dtype=np.float64)
    # xc(0) = 1; first harmonics xc(1), ys(1) and, if stellsym, zs(1) = 0.25.
    dofs[0] = 1.0
    dofs[[1, o + 1, 2 * o + 1] if stellsym else [1, 3 * o + 2]] = 0.25
    curve.x = dofs + 1e-2 * rng.random(dofs.size)
    return curve


def _curve_spec(curve: CurveXYZFourierSymmetries) -> core.CurveXYZFourierSymmetriesSpec:
    fields = {name: getattr(curve, name) for name in _SPEC_FIELDS}
    return core.make_curve_xyzfouriersymmetries_spec(dofs=curve.get_dofs(), quadpoints=curve.quadpoints, **fields)


def _numpy_gamma_oracle(dofs, quadpoints, order, nfp, stellsym, ntor) -> np.ndarray:
    """Closed-form ``CurveXYZFourierSymmetries`` gamma that shares no JAX kernel code."""
    theta = np.asarray(quadpoints, dtype=np.float64)
    angles_full = 2.0 * np.pi * nfp * np.outer(theta, np.arange(order + 1))
    cos_full, sin_tail = np.cos(angles_full), np.sin(angles_full[:, 1:])
    if stellsym:
        xc, ys, zs = np.split(dofs, [order + 1, 2 * order + 1])
        hat_x, hat_y, z = cos_full @ xc, sin_tail @ ys, sin_tail @ zs
    else:
        bounds = [order + 1, 2 * order + 1, 3 * order + 2, 4 * order + 2, 5 * order + 3]
        xc, xs, yc, ys, zc, zs = np.split(dofs, bounds)
        hat_x = cos_full @ xc + sin_tail @ xs
        hat_y = cos_full @ yc + sin_tail @ ys
        z = cos_full @ zc + sin_tail @ zs
    tor = 2.0 * np.pi * theta * ntor
    cos_tor, sin_tor = np.cos(tor), np.sin(tor)
    x = hat_x * cos_tor - hat_y * sin_tor
    y = hat_x * sin_tor + hat_y * cos_tor
    return np.stack((x, y, z), axis=1).astype(np.float64)


class TestCurvexyzfouriersymmetriesSpecJax(JaxTestCase):
    """Symmetry-curve spec geometry and derivatives against native curves."""

    @_for_cases(_FORWARD_PARITY_CASES)
    def test_spec_gamma_byte_identical_to_curve_gamma(self, case):
        """Spec gamma and ``curve.gamma()`` both match an independent NumPy oracle."""
        curve = _build_curve(case["nfp"], case["ntor"], case["stellsym"], _rng(0))
        spec = _curve_spec(curve)
        self.assertIsInstance(spec, core.CurveXYZFourierSymmetriesSpec)
        gamma_curve = np.asarray(curve.gamma(), dtype=np.float64)
        gamma_jax = np.asarray(core.curve_geometry_from_dofs(spec, spec.dofs)[0], dtype=np.float64)
        gamma_numpy = _numpy_gamma_oracle(
            np.asarray(spec.dofs, dtype=np.float64), np.asarray(spec.quadpoints, dtype=np.float64),
            order=int(spec.order), nfp=int(spec.nfp), stellsym=bool(spec.stellsym), ntor=int(spec.ntor))
        for gamma in (gamma_curve, gamma_jax, gamma_numpy):
            self.assertEqual(gamma.shape, (_NQUADPOINTS, 3))
        # The spec and the native JaxCurve share the JAX kernel; the oracle does not.
        _close_direct(gamma_jax, gamma_numpy)
        _close_direct(gamma_curve, gamma_numpy)

    @_for_cases(_STELLSYM_CASES)
    def test_explicit_spec_round_trips_fields(self, case):
        """Explicit spec construction mirrors the host class state."""
        curve = _build_curve(3, 2, case["stellsym"], _rng(1))
        spec = _curve_spec(curve)
        self.assertIsInstance(spec, core.CurveXYZFourierSymmetriesSpec)
        for name in _SPEC_FIELDS:
            self.assertEqual(getattr(spec, name), getattr(curve, name))
        assert_array_equal(np.asarray(spec.dofs), np.asarray(curve.get_dofs(), dtype=np.float64))
        assert_array_equal(np.asarray(spec.quadpoints), np.asarray(curve.quadpoints, dtype=np.float64))

    def test_curve_spec_from_curve_rejects_without_adapter_hook(self):
        """Legacy host curves without adapter hooks stay outside the implicit boundary."""
        curve = _build_curve(5, 3, True, _rng(2))
        message = "CurveXYZFourierSymmetries does not expose an immutable JAX spec"
        with self.assertRaisesRegex(NotImplementedError, message):
            core.curve_spec_from_curve(curve)

    @_for_cases(_STELLSYM_CASES)
    def test_spec_higher_derivatives_match_curve(self, case):
        """``curve_geometry_from_dofs`` gamma, gammadash and gammadashdash match native."""
        curve = _build_curve(3, 2, case["stellsym"], _rng(3))
        spec = _curve_spec(curve)
        native = (curve.gamma(), curve.gammadash(), curve.gammadashdash())
        checks = (_close_direct, _close_first_deriv, _close_second_deriv)
        for term, expected, close in zip(core.curve_geometry_from_dofs(spec, spec.dofs), native, checks, strict=True):
            close(np.asarray(term, dtype=np.float64), np.asarray(expected, dtype=np.float64))

    @_for_cases(_VJP_CASES)
    def test_geometry_vjps_match_cpu_derivatives(self, case):
        """Autodiff pullbacks of the geometry terms match CPU derivative tensor contractions."""
        rng = _rng(5 + int(case["stellsym"]))
        curve = _build_curve(3, 2, case["stellsym"], rng)
        spec = _curve_spec(curve)
        derivative_cpu = np.asarray(getattr(curve, case["derivative_method"])(), dtype=np.float64)
        cotangent = rng.normal(size=derivative_cpu.shape[:2])
        index = case["term_index"]
        _, pullback = jax.vjp(lambda dofs: _curve_geometry_terms_from_dofs(spec, dofs, max(index, 1))[index], spec.dofs)
        (vjp_jax,) = pullback(jnp.asarray(cotangent, dtype=jnp.float64))
        _close_first_deriv(np.asarray(vjp_jax, dtype=np.float64), np.einsum("ij,ijk->k", cotangent, derivative_cpu),
                           err_msg=f"CurveXYZFourierSymmetries {case['term_name']} VJP mismatch.")

    def test_spec_geometry_runs_under_strict_transfer_guard(self):
        """Compiled spec geometry runs under ``jax.transfer_guard('disallow')``."""
        spec = _curve_spec(_build_curve(3, 2, True, _rng(5)))
        compiled = jit(lambda s: core.curve_geometry_from_dofs(s, s.dofs)).lower(spec).compile()
        with jax.transfer_guard("disallow"):
            gamma, gammadash, gammadashdash = compiled(spec)
        for term in (gamma, gammadash, gammadashdash):
            self.assertEqual(term.shape, (_NQUADPOINTS, 3))
        for term in (gamma, gammadash, gammadashdash):
            self.assertTrue(np.all(np.isfinite(np.asarray(term))))

    def test_static_slice_and_update_grad_run_under_strict_transfer_guard(self):
        """Compiled static slices and updates have exact values and gradients under a strict transfer guard."""
        values_host = np.linspace(-2.0, 3.0, num=6, dtype=np.float64)
        replacement_host = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        values, replacement = jnp.asarray(values_host), jnp.asarray(replacement_host)

        def objective(values_arg, replacement_arg):
            segment = _slice_1d_static(values_arg, 2, 5)
            updated = _update_1d_static(values_arg, 1, replacement_arg)
            return jnp.sum(segment * segment) + jnp.sum(updated)

        value_fn = jit(objective).lower(values, replacement).compile()
        grad_fn = jit(jax.grad(objective, argnums=(0, 1))).lower(values, replacement).compile()
        with jax.transfer_guard("disallow"):
            actual_value = value_fn(values, replacement)
            grad_values, grad_replacement = grad_fn(values, replacement)

        updated = np.concatenate([values_host[:1], replacement_host, values_host[4:]])
        expected_value = np.sum(values_host[2:5] ** 2) + np.sum(updated)
        expected_grad_values = np.zeros_like(values_host)
        expected_grad_values[2:5] += 2.0 * values_host[2:5]
        expected_grad_values[:1] += 1.0
        expected_grad_values[4:] += 1.0
        exact = functools.partial(assert_allclose, rtol=0.0, atol=0.0)
        exact(np.asarray(actual_value), expected_value)
        exact(np.asarray(grad_values), expected_grad_values)
        exact(np.asarray(grad_replacement), np.ones_like(replacement_host))

    def test_coprime_invariant_unchanged(self):
        """The host-class constructor still enforces ``gcd(ntor, nfp) == 1``."""
        with self.assertRaisesRegex(Exception, "nfp and ntor must be coprime"):
            CurveXYZFourierSymmetries(_NQUADPOINTS, _ORDER, 4, True, 2)

    def test_spec_constructor_enforces_coprime_invariant(self):
        """The explicit spec constructor preserves the same coprime contract."""
        with self.assertRaisesRegex(ValueError, "requires nfp and ntor coprime"):
            core.make_curve_xyzfouriersymmetries_spec(dofs=np.zeros(3 * _ORDER + 1, dtype=np.float64),
                                                      quadpoints=np.linspace(0.0, 1.0, _NQUADPOINTS, endpoint=False),
                                                      order=_ORDER, nfp=4, stellsym=True, ntor=2)
