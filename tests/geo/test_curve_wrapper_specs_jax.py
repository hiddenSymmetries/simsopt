"""JAX specs of perturbed and filament curves follow native geometry and native errors.

A perturbation sample that is an exact Fourier series makes ``CurvePerturbed``
the same curve as one ``CurveXYZFourier`` with summed coefficients, so filament
geometry over the perturbed curve has an independent analytic reference. The
second parameter derivative, which native filaments do not provide, is also
checked against the spectral derivative of the first.
"""

from __future__ import annotations

from unittest_jax_support import (
    JaxTestCase,
    host_array,
    optional_jax_imports,
    require_cuda,
)

with optional_jax_imports():
    import jax
    from simsopt_jax.backend import invalidate_backend_cache, set_backend
    from simsopt_jax.core import (
        curve_gamma_and_gammadash_from_dofs,
        curve_geometry_from_dofs,
        make_curve_filament_spec,
        make_optimizable_dof_map_spec,
    )
    from simsopt_jax.core.specs import CurveFilamentSpec
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

from typing import cast

import numpy as np
from numpy.random import PCG64DXSM, Generator
from simsopt.geo import (
    CurveFilament,
    CurvePerturbed,
    CurveXYZFourier,
    FrameRotation,
    GaussianSampler,
    PerturbationSample,
    RotatedCurve,
)
from simsopt.geo.framedcurve import FramedCurve, FramedCurveCentroid, FramedCurveFrenet

_NQUAD = 128
# Spectral-derivative references resolve the Frenet frame to round-off only
# from 256 points (at 128 the spectral error of gammadashdash is 2e-8).
_SPECTRAL_NQUAD = 256
_ORDER = 3
_FRAMES = (FramedCurveCentroid, FramedCurveFrenet)


def _xyz_curve(coefficients: dict[str, float], nquad: int = _NQUAD) -> CurveXYZFourier:
    curve = CurveXYZFourier(nquad, _ORDER)
    dofs = np.zeros(curve.dof_size)
    for name, value in coefficients.items():
        dofs[curve.local_dof_names.index(name)] = value
    curve.x = dofs
    return curve


_BASE = {"xc(1)": 1.0, "ys(1)": 1.0, "zs(2)": 0.2, "xs(3)": 0.05}
_PERTURBATION = {"xs(2)": 0.03, "yc(3)": -0.02, "zs(1)": 0.04}


def _gaussian_sample(curve, n_derivs: int) -> PerturbationSample:
    """Seeded Gaussian-process sample, so value comparisons see one fixed draw."""
    sampler = GaussianSampler(curve.quadpoints, 0.05, 0.3, n_derivs=n_derivs)
    return PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(670)))


def _perturbed_and_equivalent() -> tuple[CurvePerturbed, CurveXYZFourier]:
    """Return base + Fourier sample (with three exact derivatives) and the summed curve."""
    perturbation = _xyz_curve(_PERTURBATION)
    sample = [
        perturbation.gamma(),
        perturbation.gammadash(),
        perturbation.gammadashdash(),
        perturbation.gammadashdashdash(),
    ]
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
    evaluate = curve_gamma_and_gammadash_from_dofs if order == 1 else curve_geometry_from_dofs
    return [host_array(term) for term in evaluate(spec, spec.dofs)]


class TestCurveWrapperSpecs(JaxTestCase):
    def test_filament_over_perturbed_curve_matches_summed_curve(self):
        """Order-1 and order-2 filament terms over a perturbed curve include the sample derivatives."""
        perturbed, summed = _perturbed_and_equivalent()
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                filament = _filament(perturbed, frame)
                spec = curve_spec_from_adapter_curve(filament)
                reference = curve_spec_from_adapter_curve(_filament(summed, frame))
                order1 = _terms(spec, 1)
                np.testing.assert_allclose(order1[1], filament.gammadash(), rtol=0, atol=1e-13)
                for term, expected in zip(order1, _terms(reference, 1), strict=True):
                    np.testing.assert_allclose(term, expected, rtol=0, atol=1e-13)
                if frame is FramedCurveFrenet:
                    # The Frenet second derivative needs a fourth sample derivative.
                    with self.assertRaisesRegex(ValueError, "derivative 4 is needed"):
                        curve_geometry_from_dofs(spec, spec.dofs)
                    continue
                order2 = _terms(spec, 2)
                for term, expected in zip(order2, _terms(reference, 2), strict=True):
                    np.testing.assert_allclose(term, expected, rtol=0, atol=1e-11)
                np.testing.assert_allclose(order2[1], filament.gammadash(), rtol=0, atol=1e-13)

    def test_filament_second_derivative_is_derivative_of_tangent(self):
        """gammadashdash equals the spectral derivative of gammadash for both frames."""
        curve = _xyz_curve({**_BASE, **_PERTURBATION}, nquad=_SPECTRAL_NQUAD)
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                gamma, gammadash, gammadashdash = _terms(
                    curve_spec_from_adapter_curve(_filament(curve, frame)), 2
                )
                np.testing.assert_allclose(_spectral_derivative(gamma), gammadash, rtol=0, atol=1e-11)
                np.testing.assert_allclose(_spectral_derivative(gammadash), gammadashdash, rtol=0, atol=1e-11)

    def test_filament_over_filament_derivatives_are_derivatives_of_lower_terms(self):
        """Nested filaments: each derivative is the spectral derivative of the term below.

        Native filaments provide no second derivative, so native evaluates only
        the position of a centroid filament over another filament.
        """
        inner = _filament(_xyz_curve(_BASE, nquad=_SPECTRAL_NQUAD), FramedCurveCentroid)
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                outer = _filament(inner, frame)
                gamma, gammadash, gammadashdash = _terms(curve_spec_from_adapter_curve(outer), 2)
                if frame is FramedCurveCentroid:
                    np.testing.assert_allclose(gamma, outer.gamma(), rtol=0, atol=1e-13)
                np.testing.assert_allclose(_spectral_derivative(gamma), gammadash, rtol=0, atol=1e-11)
                np.testing.assert_allclose(_spectral_derivative(gammadash), gammadashdash, rtol=0, atol=1e-11)

    def test_sample_without_frame_derivatives_raises_like_native(self):
        """A sample lacking the derivative a frame needs is rejected at conversion, as native raises."""
        base = _xyz_curve(_BASE)
        # (n_derivs, frame) -> native rejects: centroid needs 2, Frenet 3.
        for n_derivs, frame, native_raises in (
            (1, FramedCurveCentroid, True),
            (1, FramedCurveFrenet, True),
            (2, FramedCurveFrenet, True),
            (2, FramedCurveCentroid, False),
        ):
            with self.subTest(n_derivs=n_derivs, frame=frame.__name__):
                filament = _filament(CurvePerturbed(base, _gaussian_sample(base, n_derivs)), frame)
                if not native_raises:
                    spec = curve_spec_from_adapter_curve(filament)
                    np.testing.assert_allclose(_terms(spec, 1)[1], filament.gammadash(), rtol=0, atol=1e-13)
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
        np.testing.assert_allclose(_terms(spec, 1)[1], perturbed.gammadash(), rtol=0, atol=1e-13)
        with self.assertRaises(ValueError):
            perturbed.gammadashdash()
        with self.assertRaisesRegex(ValueError, "has 1 derivatives, but derivative 2"):
            curve_geometry_from_dofs(spec, spec.dofs)

    def test_unsupported_frames_and_wrapper_orders_raise(self):
        """Custom frames, unknown frame kinds and inner RotatedCurve placements are rejected."""
        base = _xyz_curve(_BASE)
        custom_frame = CurveFilament(FramedCurve(base), 0.02, 0.03)
        with self.assertRaisesRegex(NotImplementedError, "FramedCurve"):
            curve_spec_from_adapter_curve(custom_frame)
        valid = cast(
            CurveFilamentSpec,
            curve_spec_from_adapter_curve(_filament(base, FramedCurveCentroid)),
        )
        with self.assertRaisesRegex(ValueError, "frame_kind"):
            make_curve_filament_spec(
                dofs=valid.dofs,
                quadpoints=valid.quadpoints,
                base_curve=valid.base_curve,
                base_curve_map=valid.base_curve_map,
                rotation=valid.rotation,
                rotation_map=valid.rotation_map,
                frame_kind="custom",
                dn=valid.dn,
                db=valid.db,
            )
        rotated = RotatedCurve(base, np.pi / 3, flip=False)
        for inner_rotation in (
            CurvePerturbed(rotated, _gaussian_sample(base, 3)),
            _filament(rotated, FramedCurveCentroid),
        ):
            with self.subTest(wrapper=type(inner_rotation).__name__), self.assertRaisesRegex(
                NotImplementedError, "RotatedCurve"
            ):
                curve_spec_from_adapter_curve(inner_rotation)

    def test_dof_map_segments_must_have_equal_lengths(self):
        """Owner and target ranges of a DOF-map segment must match and fit the template."""
        for segment in ((0, 3, 1, 3), (0, 2, 3, 5)):
            with self.subTest(segment=segment), self.assertRaisesRegex(
                ValueError, "equal-length target range"
            ):
                make_optimizable_dof_map_spec(
                    template_full_dofs=np.zeros(4),
                    owner_segments=(segment,),
                    input_mode="full",
                    input_start=0,
                    input_end=4,
                )

    def test_wrapper_dof_map_templates_follow_the_runtime_device(self):
        """Under a jax-cpu policy in a CUDA process, wrapper DOF-map templates join the
        other spec leaves on the runtime CPU device instead of JAX's default GPU."""
        require_cuda("CUDA device required for a runtime device JAX would not choose")
        invalidate_backend_cache()
        set_backend("jax_cpu_parity", configure_runtime=False)
        cpu = jax.devices("cpu")[0]
        base = _xyz_curve(_BASE)
        for wrapper in (CurvePerturbed(base, _gaussian_sample(base, 1)), _filament(base, FramedCurveCentroid)):
            with self.subTest(wrapper=type(wrapper).__name__):
                spec = curve_spec_from_adapter_curve(wrapper)
                devices = {leaf.devices().pop() for leaf in jax.tree.leaves(spec)}
                self.assertEqual(devices, {cpu})
