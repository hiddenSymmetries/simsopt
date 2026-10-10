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
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo.curveperturbed import (
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier


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
        materializations = []
        metadata_placements = []
        put = jax.device_put
        materialize = backend.host_array
        place_metadata = backend._place_array_tree_on_device

        def counted_put(value, *args, **kwargs):
            placements.extend(
                leaf.nbytes
                for leaf in jax.tree.leaves(value)
                if isinstance(leaf, np.ndarray) and leaf.ndim > 0
            )
            return put(value, *args, **kwargs)

        def counted_materialize(value, **kwargs):
            materializations.append(1)
            return materialize(value, **kwargs)

        def counted_metadata(tree, device):
            metadata_placements.append(1)
            return place_metadata(tree, device)

        patches.enter_context(mock.patch.object(jax, "device_put", counted_put))
        patches.enter_context(
            mock.patch.object(backend, "host_array", counted_materialize)
        )
        patches.enter_context(
            mock.patch.object(backend, "_place_array_tree_on_device", counted_metadata)
        )
        actual = field.coil_cotangents_to_derivative(arrays, indices)
        print(
            f"BF2: placements={len(placements)} bytes={sum(placements)} "
            f"materializations={len(materializations)}"
        )
        np.testing.assert_allclose(
            cast(np.ndarray, actual(field)),
            cast(np.ndarray, expected(native)),
            rtol=1e-11,
            atol=1e-13,
        )
        # One free-vector H2D boundary (32 doubles), one packed D2H.
        self.assertTrue(
            (len(placements), sum(placements), len(materializations)) == (1, 256, 1),
            "(len(placements), sum(placements), len(materializations)) == (1, 256, 1)",
        )
        self.assertTrue(not metadata_placements, "not metadata_placements")

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
        fingerprint_reads = []
        fingerprint = field._current_captured_coil_state_fingerprint

        def counted_fingerprint():
            fingerprint_reads.append(1)
            return fingerprint()

        patches.enter_context(
            mock.patch.object(
                field, "_current_captured_coil_state_fingerprint", counted_fingerprint
            )
        )
        _CountedSample.copies.clear()
        for _ in range(10):
            np.testing.assert_allclose(np.asarray(field.B()), expected, rtol=0, atol=0)
        print(
            f"BF3: sample_copies={len(_CountedSample.copies)} "
            f"bytes={sum(_CountedSample.copies)}"
        )
        self.assertTrue(not _CountedSample.copies, "not _CountedSample.copies")
        self.assertTrue(not fingerprint_reads, "not fingerprint_reads")
        self.assertTrue(
            field.coil_dof_extraction_spec() is spec,
            "field.coil_dof_extraction_spec() is spec",
        )
