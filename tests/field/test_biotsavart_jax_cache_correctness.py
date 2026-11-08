"""Cache and host-boundary regressions for the BiotSavart correctness follow-up."""

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    from simsopt_jax_adapters.field import biotsavart_backend as backend
from unittest import mock


from typing import cast

import numpy as np

from simsopt._core.derivative import Derivative
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import RotatedCurve, create_equally_spaced_curves
from simsopt.geo.curveperturbed import (
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import FramedCurveCentroid, FrameRotation


_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]])


def _coils(*, perturbed=False):
    curves = create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=32
    )
    if perturbed:
        sampler = GaussianSampler(curves[0].quadpoints, 1e-3, 0.2, n_derivs=1)
        curves = [
            CurvePerturbed(
                curves[0],
                PerturbationSample(sampler, randomgen=np.random.default_rng(713)),
            )
        ]
    currents = [Current(1e5) for _curve in curves]
    return curves, coils_via_symmetries(curves, currents, 2, True)


class _CountedSample(np.ndarray):
    copies = []

    def tobytes(self, order="C"):
        self.copies.append(self.nbytes)
        return super().tobytes(order)

    def copy(self, order="C"):
        self.copies.append(self.nbytes)
        return super().copy(order)


class TestBiotsavartJaxCacheCorrectness(JaxTestCase):
    def test_warmed_full_owner_projection_host_boundaries(self):
        """Warmed owner projection matches native gradients with one free-vector
        placement and one packed host read."""
        patches = self.patches
        _, coils = _coils()
        field = backend.JaxBiotSavart(coils)
        field.set_points(_POINTS)
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        native = BiotSavart(coils)
        native.set_points(_POINTS)
        native.B()
        expected = cast(Derivative, native.B_vjp(cotangent))
        pullback = field.B_pullback_native(cotangent)
        arrays, indices = pullback.d_coil_arrays, pullback.coil_indices
        field.coil_cotangents_to_derivative(arrays, indices)
        placements = []
        put = jax.device_put  # noqa: TID251 wraps the raw primitive to count placements

        def counted_put(value, *args, **kwargs):
            placements.extend(
                leaf.nbytes
                for leaf in jax.tree.leaves(value)
                if isinstance(leaf, np.ndarray) and leaf.ndim > 0
            )
            return put(value, *args, **kwargs)

        patches.enter_context(mock.patch.object(jax, "device_put", counted_put))
        materializations = patches.enter_context(
            mock.patch.object(backend, "host_array", wraps=backend.host_array)
        )
        metadata_placements = patches.enter_context(
            mock.patch.object(
                backend,
                "_place_array_tree_on_device",
                wraps=backend._place_array_tree_on_device,
            )
        )
        actual = field.coil_cotangents_to_derivative(arrays, indices)
        np.testing.assert_allclose(
            cast(np.ndarray, actual(field)),
            cast(np.ndarray, expected(native)),
            rtol=1e-11,
            atol=1e-13,
        )
        # One free-vector H2D boundary (32 doubles), one packed D2H.
        self.assertTrue(
            (len(placements), sum(placements), materializations.call_count)
            == (1, 256, 1),
            "(len(placements), sum(placements), materializations.call_count) == (1, 256, 1)",
        )
        self.assertTrue(
            not metadata_placements.called, "not metadata_placements.called"
        )

    def test_packed_projection_preserves_distinct_owners_with_shared_dofs(self):
        """Packed projection retains independent writable native partials for distinct
        owners sharing DOFs."""
        curves, _ = _coils()
        curve = curves[0]
        shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
        curve.fix(curve.local_dof_names[0])
        first, second = Current(1e5), Current(2e4)
        second.fix_all()
        coils = [Coil(curve, first + second), Coil(shared, -2.5 * (first - second))]
        field, native = backend.JaxBiotSavart(coils), BiotSavart(coils)
        for evaluator in (field, native):
            evaluator.set_points(_POINTS)
            evaluator.B()
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        expected = cast(Derivative, native.B_vjp(cotangent))
        actual = field.B_vjp(cotangent)
        self.assertTrue(
            set(actual.data) == set(expected.data),
            "set(actual.data) == set(expected.data)",
        )
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner], expected.data[owner], rtol=1e-11, atol=1e-13
            )
        self.assertTrue(
            not np.array_equal(actual.data[curve], actual.data[shared]),
            "not np.array_equal(actual.data[curve], actual.data[shared])",
        )
        self.assertTrue(
            cast(np.ndarray, actual.data[curve]).flags.writeable,
            "cast(np.ndarray, actual.data[curve]).flags.writeable",
        )
        self.assertTrue(
            not np.shares_memory(actual.data[curve], actual.data[shared]),
            "not np.shares_memory(actual.data[curve], actual.data[shared])",
        )

    def test_packed_projection_preserves_fixed_nested_curve_partials(self):
        """Packed projection preserves native partials for partially or fully fixed
        nested curve owners."""
        for fixed in ["partial", "full"]:
            with self.subTest(fixed=fixed), self.case():
                self._case_packed_projection_preserves_fixed_nested_curve_partials(
                    fixed
                )

    def _case_packed_projection_preserves_fixed_nested_curve_partials(self, fixed):
        """Run one row with case-local objects released before runtime cleanup."""
        curves, coils = _coils(perturbed=True)
        if fixed == "partial":
            curves[0].curve.fix(2)
        else:
            curves[0].curve.fix_all()
        field, native = backend.JaxBiotSavart(coils), BiotSavart(coils)
        for evaluator in (field, native):
            evaluator.set_points(_POINTS)
        np.testing.assert_allclose(
            np.asarray(field.B()), native.B(), rtol=1e-12, atol=1e-14
        )
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        expected, actual = (
            cast(Derivative, native.B_vjp(cotangent)),
            field.B_vjp(cotangent),
        )
        self.assertTrue(
            set(actual.data) == set(expected.data),
            "set(actual.data) == set(expected.data)",
        )
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner], expected.data[owner], rtol=1e-11, atol=1e-13
            )

    def test_unchanged_perturbation_evaluations_do_not_copy_samples(self):
        """Repeated unchanged perturbed-field evaluations reuse results without copying
        samples or reading fingerprints."""
        patches = self.patches
        curves, coils = _coils(perturbed=True)
        curve = curves[0]
        curve.sample._sample = [
            sample.view(_CountedSample) for sample in curve.sample._sample
        ]
        field = backend.JaxBiotSavart(coils)
        field.set_points(_POINTS)
        expected = np.asarray(field.B()).copy()
        native = BiotSavart(coils)
        native.set_points(_POINTS)
        np.testing.assert_allclose(expected, native.B(), rtol=1e-12, atol=1e-14)
        spec = field.coil_dof_extraction_spec()
        fingerprint_reads = patches.enter_context(
            mock.patch.object(
                field,
                "_current_captured_coil_state_fingerprint",
                wraps=field._current_captured_coil_state_fingerprint,
            )
        )
        _CountedSample.copies.clear()
        for _ in range(10):
            np.testing.assert_allclose(np.asarray(field.B()), expected, rtol=0, atol=0)
        self.assertTrue(not _CountedSample.copies, "not _CountedSample.copies")
        self.assertTrue(not fingerprint_reads.called, "not fingerprint_reads.called")
        self.assertTrue(
            field.coil_dof_extraction_spec() is spec,
            "field.coil_dof_extraction_spec() is spec",
        )

    def test_notified_non_dof_parameter_changes_match_native(self):
        """A changed current scale, filament offset, frame-rotation scale or
        RotatedCurve rotation matrix reaches the adapter's B and B pullback as it
        reaches native, whether the changed object, the field, or the field's DOF
        setter sends the notification."""
        for parameter in (
            "current_scale", "filament_offset", "rotation_scale", "rotation_matrix",
        ):
            for notifier in ("changed_object", "field", "field_x_setter"):
                with self.subTest(parameter=parameter, notifier=notifier), self.case():
                    self._case_notified_non_dof_parameter_changes_match_native(
                        parameter, notifier
                    )

    def _case_notified_non_dof_parameter_changes_match_native(self, parameter, notifier):
        """Run one row with case-local objects released before runtime cleanup."""
        curves, _ = _coils()
        rotation = FrameRotation(curves[0].quadpoints, 1)
        rotation.x = np.array([0.2, -0.1, 0.05])
        filament = CurveFilament(FramedCurveCentroid(curves[0], rotation), 0.01, 0.02)
        current = ScaledCurrent(Current(1e5), 2.0)
        rotated = RotatedCurve(curves[1], 0.3, False)
        coils = [Coil(filament, current), Coil(rotated, Current(-4e4))]
        field = backend.JaxBiotSavart(coils).set_points(_POINTS)
        field.B()
        if parameter == "current_scale":
            current.scale = 3.0
            changed = current
        elif parameter == "filament_offset":
            filament.dn, filament.db = 0.03, -0.01
            changed = filament
        elif parameter == "rotation_scale":
            rotation.scale = 0.5
            changed = rotation
        else:
            # Native evaluation reads rotmat and its transpose.
            rotated.rotmat = RotatedCurve(curves[1], 0.7, False).rotmat
            rotated.rotmatT = rotated.rotmat.T.copy()
            changed = rotated
        if notifier == "changed_object":
            changed.set_recompute_flag()
        elif notifier == "field":
            field.set_recompute_flag()
        else:
            field.x = field.x
        native = BiotSavart(coils).set_points(_POINTS)
        np.testing.assert_allclose(
            np.asarray(field.B()), native.B(), rtol=1e-12, atol=1e-14
        )
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        expected = cast(Derivative, native.B_vjp(cotangent))
        actual = field.B_vjp(cotangent)
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner], expected.data[owner], rtol=1e-11, atol=1e-13
            )
