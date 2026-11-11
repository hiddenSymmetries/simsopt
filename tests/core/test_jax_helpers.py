"""Exact values and derivatives of the JAX helpers, and the pytree records they use."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from typing import cast

import numpy as np
from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax import pytree
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core import field as core_field
    from simsopt_jax.core._device_scalars import device_one, placement_zero
    from simsopt_jax.core._math_utils import pairwise_sum_axis
    from simsopt_jax.core.biotsavart import _radius_squared, _separation
    from simsopt_jax.core.biotsavart import biot_savart_A, biot_savart_B
    from simsopt_jax.core.biotsavart import biot_savart_dB_by_dX
    from simsopt_jax.core.curve_xyz_fourier import _constant_row
    from simsopt_jax.core.specs import make_grouped_coil_set_spec
    from simsopt_jax.pytree import pytree_dataclass
    from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

    # Native simsopt.geo imports jax too.
    from simsopt.field import Coil, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves


class TestKernelNumericHelpers(JaxTestCase):
    """Kernel helpers keep exact zeros, point singularities and IEEE sums."""

    def test_radius_squared_preserves_zero_singularity(self):
        """Squared radius remains exactly zero at a coincident point."""
        radius_squared = _radius_squared(jnp.zeros((1, 3), dtype=jnp.float64))
        np.testing.assert_allclose(np.asarray(radius_squared), np.zeros((1,)))

    def test_separation_preserves_zero_singularity(self):
        """A coincident point gives zero components and a zero squared radius."""
        x = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
        components, radius_squared = _separation(x, jnp.reshape(x, (1, 1, 3)))
        for value in (*components, radius_squared):
            np.testing.assert_array_equal(np.asarray(value), np.zeros((1, 1)))

    def test_biot_savart_public_kernels_preserve_point_singularity(self):
        """The public A, B and dB/dX kernels keep the filament singularity."""
        arguments = [
            jnp.asarray(value, dtype=jnp.float64) for value in ([[0.0] * 3], [[[0.0] * 3]], [[[1.0, 0.0, 0.0]]], [1.0])
        ]
        for kernel in (biot_savart_A, biot_savart_B, biot_savart_dB_by_dX):
            self.assertFalse(np.all(np.isfinite(np.asarray(kernel(*arguments)))))

    def test_constant_row_builds_its_row_from_a_traced_reference(self):
        """The row is selected statically but sourced from the reference's device."""

        @jit
        def rows_for(reference):
            return tuple(_constant_row(3, is_one=is_one, reference=reference) for is_one in (True, False))

        ones_row, zeros_row = rows_for(jnp.asarray(1.0, dtype=jnp.float64))
        np.testing.assert_allclose(np.asarray(ones_row), np.ones((1, 3)))
        np.testing.assert_allclose(np.asarray(zeros_row), np.zeros((1, 3)))

    def test_pairwise_sum_padding_reads_no_values(self):
        """Zero padding keeps IEEE sums: inf stays inf, an overflow never turns NaN."""
        rows = [((np.inf, 1.0, 1.0), np.inf), ((1.0e308, 1.0e308, -1.0e308), np.inf), ((1.0e308, -1.0e308, 1.0), 1.0)]
        for values, expected in rows:
            with self.subTest(values=values):
                total = pairwise_sum_axis(jnp.asarray(values), axis=0)
                self.assertEqual(float(total), expected)


def _gamma(k: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` in float64."""
    unit_roundoff = float(np.finfo(np.float64).eps) / 2.0
    return k * unit_roundoff / (1.0 - k * unit_roundoff)


def _coils(quadrature: int):
    """Two circular coils (radius, height) = (0.5, 0) and (0.4, 0.3) around x = 1."""
    phi = 2.0 * np.pi * np.arange(quadrature) / quadrature
    zero = 0.0 * phi
    gammas, gammadashs = [], []
    for radius, height in ((0.5, 0.0), (0.4, 0.3)):
        cos, sin = radius * np.cos(phi), radius * np.sin(phi)
        gammas.append(np.stack([1.0 + cos, sin, height + zero], axis=1))
        gammadashs.append(2.0 * np.pi * np.stack([-sin, cos, zero], axis=1))
    return np.stack(gammas), np.stack(gammadashs)


def _B_current_cotangent(points, gammas, gammadashs, currents, cotangent):
    """``d(cotangent . B)/dI`` in T/A per unit cotangent, shape (C,)."""
    x, gamma, dash = (jnp.asarray(value) for value in (points, gammas, gammadashs))
    _, pullback = jax.vjp(lambda i: biot_savart_B(x, gamma, dash, i), jnp.asarray(currents))
    return np.asarray(pullback(jnp.asarray(cotangent))[0])


class TestDeviceOneGradientPath(JaxTestCase):
    """``device_one`` adds no derivative path (one cost a Stage-II gradient 1e-11)."""

    def test_device_one_has_a_zero_cotangent(self):
        """Multiplying by device_one keeps the reference gradient of the objective."""
        reference = jnp.asarray([1.0e5, -3.0, 2.5e-7])
        cotangent = jax.grad(lambda r: device_one(r) * jnp.sum(r * r))(reference)
        np.testing.assert_array_equal(np.asarray(cotangent), 2.0 * np.asarray(reference))

    def test_biot_savart_current_cotangent_is_the_unit_field_contraction(self):
        """``d(ct . B)/dI_c = ct . B_c(I_c = 1)`` within ``2 gamma_K S`` at I = 1e5.

        Both sides sum ``1e-7 / Q * ct[p, j] * k[c, q, p, j]`` in different orders:
        ``S`` sums the absolute products, ``K`` counts ``3 P Q - 1`` additions plus
        at most 16 roundings of the integrand.
        """
        quadrature = 32
        gammas, gammadashs = _coils(quadrature)
        points = np.array([[1.0, 0.0, 0.1], [1.2, 0.3, -0.2], [0.7, -0.4, 0.25], [1.1, 0.05, 0.6]])
        currents = np.array([1.0e5, -1.0e5])
        cotangent = np.random.default_rng(20260929).standard_normal(points.shape)

        current_cotangent = _B_current_cotangent(points, gammas, gammadashs, currents, cotangent)

        difference = points[None, None, :, :] - gammas[:, :, None, :]  # [c, q, p, 3]
        radius = np.linalg.norm(difference, axis=-1)
        integrand = np.cross(gammadashs[:, :, None, :], difference) / radius[..., None] ** 3
        scale = 1.0e-7 / quadrature
        expected = scale * np.einsum("pj,cqpj->c", cotangent, integrand)
        magnitude = scale * np.einsum("pj,cqpj->c", np.abs(cotangent), np.abs(integrand))
        bound = 2.0 * _gamma(3 * points.shape[0] * quadrature - 1 + 16) * magnitude
        error = np.abs(current_cotangent - expected)
        self.assertTrue(np.all(error <= bound), (error, bound, error / np.abs(expected)))

    def test_an_infinite_current_leaves_the_other_coils_derivative_unchanged(self):
        """B is linear in each current: coil 1's derivative ignores coil 0's inf."""
        gammas, gammadashs = _coils(16)
        points = np.array([[1.0, 0.0, 0.1], [1.2, 0.3, -0.2]])
        cotangent = np.random.default_rng(20261029).standard_normal(points.shape)
        finite, infinite = (
            _B_current_cotangent(points, gammas, gammadashs, np.array(i), cotangent)
            for i in ([1.0e5, -1.0e5], [np.inf, -1.0e5])
        )
        self.assertTrue(np.isfinite(finite[1]), finite)
        np.testing.assert_array_equal(infinite[1], finite[1])


REFERENCES = ((1.0e308, 1.0e308), (-1.0e308, -1.0e308), (np.nan, 1.0), (np.inf, -np.inf), (1.0, 2.0))
SEEDS = (np.nan, np.inf, -np.inf, 1.0)
SEEDED_REFERENCES = [dict(seed=s, reference=r) for s in SEEDS for r in REFERENCES]


def _placed(values) -> jax.Array:
    return jnp.array(np.asarray(values, dtype=np.float64))


def _bits(value) -> np.ndarray:
    return np.asarray(value).view(np.uint64)


class TestPlacementZero(JaxTestCase):
    """``placement_zero`` is a placed +0.0 with exact zero derivatives."""

    def test_placement_zero_is_an_exact_zero_on_the_reference_device(self):
        """It is a bitwise +0.0 with the reference's dtype and device."""

        def check(reference):
            placed = _placed(reference)
            with jax.transfer_guard("disallow"):
                zero = jax.block_until_ready(placement_zero(placed))
            self.assertEqual(zero.dtype, jnp.float64)
            self.assertEqual(zero.devices(), placed.devices())
            self.assertEqual(_bits(zero), _bits(np.float64(0.0)))

        self.run_cases(check, [dict(reference=r) for r in REFERENCES])

    def test_placement_zero_tangent_is_exactly_zero(self):
        """Its tangent is a bitwise +0.0 on the reference device for every seed."""

        def check(seed, reference):
            placed, tangent = _placed(reference), _placed((seed, 1.0))
            with jax.transfer_guard("disallow"):
                _, zero_tangent = jax.jvp(placement_zero, (placed,), (tangent,))
                jax.block_until_ready(zero_tangent)
            self.assertEqual(zero_tangent.devices(), placed.devices())
            self.assertEqual(_bits(zero_tangent), _bits(np.float64(0.0)))

        self.run_cases(check, SEEDED_REFERENCES)

    def test_placement_zero_cotangent_is_exactly_zero(self):
        """It sends a bitwise +0.0 cotangent to every reference entry."""

        def check(seed, reference):
            placed, cotangent = _placed(reference), _placed(seed)
            with jax.transfer_guard("disallow"):
                _, pullback = jax.vjp(placement_zero, placed)
                (reference_cotangent,) = pullback(cotangent)
                jax.block_until_ready(reference_cotangent)
            np.testing.assert_array_equal(_bits(reference_cotangent), _bits(np.zeros(2)))

        self.run_cases(check, SEEDED_REFERENCES)

    def test_placement_zero_linearizes_a_fixed_value_on_device(self):
        """A fixed value plus the placed zero linearizes under the strict guard."""
        placed, fixed = _placed((1.0e308, 1.0e308)), _placed(3.0)
        tangent = _placed((np.inf, np.nan))
        with jax.transfer_guard("disallow"):
            value, linear = jax.linearize(lambda r: fixed + placement_zero(r), placed)
            fixed_tangent = linear(tangent)
            jax.block_until_ready((value, fixed_tangent))
        self.assertEqual(float(value), 3.0)
        self.assertEqual(_bits(fixed_tangent), _bits(np.float64(0.0)))


class TestPlacementZeroEager(JaxTestCase):
    """Placed exact zeros survive eager AD, batching and disabled JIT."""

    def assert_placed_zeros(self, *values, like=None) -> None:
        """Each value is +0.0 everywhere, on ``like``'s device and dtype if given."""
        for actual in values:
            if like is not None:
                self.assertEqual(actual.devices(), like.devices())
                self.assertEqual(actual.dtype, like.dtype)
            np.testing.assert_array_equal(actual, np.zeros(actual.shape))
            self.assertFalse(np.any(np.signbit(actual)))

    def test_primal_tangent_and_transpose_stay_placed_without_jit(self) -> None:
        """Eager value, JVP and VJP are placed +0.0 for non-finite inputs and seeds."""
        references = (
            np.asarray(np.nan),
            np.asarray((1.0e308, 1.0e308)),
            np.asarray((np.nan, np.inf, -np.inf)),
            np.asarray(((np.inf, np.nan), (-np.inf, 1.0))),
            np.empty((0,)),
        )
        for reference in references:
            placed = jnp.array(reference)
            for seed in (np.nan, np.inf, -np.inf, 1.0):
                tangent = jnp.array(np.full(reference.shape, seed))
                cotangent = jnp.array(np.asarray(seed))
                with self.subTest(shape=reference.shape, seed=seed):
                    with jax.disable_jit(), jax.transfer_guard("disallow"):
                        zero, derivative = jax.jvp(placement_zero, (placed,), (tangent,))
                        _, pullback = jax.vjp(placement_zero, placed)
                        (transpose,) = pullback(cotangent)
                        jax.block_until_ready((zero, derivative, transpose))
                    self.assert_placed_zeros(zero, derivative, transpose, like=placed)

    def test_nested_jvp_and_hvp_are_exact_zeros_without_jit(self) -> None:
        """Second-order JVPs and Hessian-vector products are +0.0, with or without JIT."""
        reference = jnp.array(np.asarray((np.nan, np.inf, 1.0e308)))
        direction = jnp.array(np.asarray((np.inf, np.nan, -np.inf)))
        seed, seed_tangent = jnp.array(np.inf), jnp.array(np.nan)

        def first_derivative(values: jax.Array) -> jax.Array:
            return jax.jvp(placement_zero, (values,), (values,))[1]

        def gradient(values: jax.Array, weight: jax.Array) -> jax.Array:
            return jax.vjp(placement_zero, values)[1](weight)[0]

        for disable_jit in (False, True):
            with self.subTest(disable_jit=disable_jit):
                with jax.disable_jit(disable_jit), jax.transfer_guard("disallow"):
                    first, second = jax.jvp(first_derivative, (reference,), (direction,))
                    derivative, hvp = jax.jvp(gradient, (reference, seed), (direction, seed_tangent))
                    jax.block_until_ready((first, second, derivative, hvp))
                self.assert_placed_zeros(first, second, derivative, hvp)

    def test_batched_zero_has_exact_nonfinite_jvp_and_transpose(self) -> None:
        """Under vmap, the value, JVP and VJP stay +0.0 for non-finite batches."""
        references = jnp.array(np.asarray(((np.nan, np.inf), (1.0e308, 1.0e308))))
        direction = jnp.array(np.full(references.shape, np.nan))
        seed = jnp.array(np.full((references.shape[0],), np.inf))
        batched = jax.vmap(placement_zero)
        with jax.disable_jit(), jax.transfer_guard("disallow"):
            values, tangent = jax.jvp(batched, (references,), (direction,))
            (transpose,) = jax.vjp(batched, references)[1](seed)
            jax.block_until_ready((values, tangent, transpose))
        self.assert_placed_zeros(values, tangent, transpose)


def _flatten(value):
    return jax.tree_util.tree_flatten(value)


def _unflatten(treedef, leaves):
    return jax.tree_util.tree_unflatten(treedef, leaves)


class TestPytreeRecords(JaxTestCase):
    """The pytree helpers register frozen records with ordered leaves, or nothing."""

    def assert_rejected(self, error, message, register, make=None) -> None:
        """``register()`` raises; then ``make()`` is one leaf, so nothing was registered."""
        with self.assertRaisesRegex(error, message):
            register()
        if make is not None:
            payload = make()
            self.assertEqual(jax.tree_util.tree_leaves(payload), [payload])

    def test_plain_class_is_frozen_and_round_trips_in_data_order(self):
        """A plain class is frozen and rebuilt from leaves in the declared order."""

        @pytree.pytree_dataclass(data=("second", "first"), meta=("mode",))
        class Payload:
            first: int
            second: int
            mode: str

        payload = Payload(first=1, second=2, mode="test")
        with self.assertRaises(FrozenInstanceError):
            setattr(payload, "first", 3)
        leaves, treedef = _flatten(payload)
        self.assertEqual(leaves, [2, 1])
        self.assertEqual(_unflatten(treedef, leaves), payload)
        self.assertEqual(_unflatten(treedef, [20, 10]), Payload(10, 20, "test"))

    def test_mutable_dataclass_is_rejected(self):
        """The dataclass helper rejects an existing mutable dataclass."""

        @dataclass
        class Mutable:
            value: int

        with self.assertRaisesRegex(TypeError, "Mutable must be a frozen dataclass"):
            pytree_dataclass(data=("value",))(Mutable)

    def test_invalid_partition_is_rejected(self):
        """Invalid data/meta partitions fail without changing the pytree registry."""

        def check(data, meta, message):
            class Payload:
                value: int
                mode: str

            self.assert_rejected(ValueError, message, lambda: pytree_dataclass(data=data, meta=meta)(Payload))

        self.run_cases(
            check,
            [
                dict(data=("value",), meta=("value", "mode"), message="data_fields and meta_fields must not overlap"),
                dict(data=("value",), meta=(), message="Missing fields: \\{'mode'\\}"),
                dict(data=("value", "extra"), meta=("mode",), message="Unexpected fields: \\{'extra'\\}"),
            ],
        )

    def test_existing_dataclass_options_and_constructor_are_preserved(self):
        """Registration keeps the dataclass options and rebuilds via its constructor."""

        @dataclass(frozen=True, init=False, eq=False)
        class Payload:
            value: int
            derived: int = field(init=False)

            def __init__(self, value):
                object.__setattr__(self, "value", value)
                object.__setattr__(self, "derived", 2 * value)

        self.assertIs(pytree_dataclass(data=("value",))(Payload), Payload)
        self.assertEqual(Payload(3).derived, 6)
        self.assertNotEqual(Payload(3), Payload(3))
        leaves, treedef = _flatten(Payload(3))
        self.assertEqual(_unflatten(treedef, leaves).derived, 6)

    def test_duplicate_partition_names_are_rejected_without_registering(self):
        """Duplicate partition names fail before registration; JAX sees one leaf."""

        def check(data, meta, partition):
            class Payload:
                value: int
                mode: str

            register = lambda: pytree_dataclass(data=data, meta=meta)(Payload)
            make = lambda: cast(Callable[..., Payload], Payload)(value=1, mode="test")
            self.assert_rejected(ValueError, f"duplicate names in {partition}", register, make)

        self.run_cases(
            check,
            [
                dict(data=("value", "value"), meta=("mode",), partition="data"),
                dict(data=("value",), meta=("mode", "mode"), partition="meta"),
            ],
        )

    def test_undecorated_subclass_is_frozen_and_all_fields_round_trip(self):
        """Inherited and new fields are frozen and rebuilt in declared data order."""

        @dataclass(frozen=True)
        class Base:
            value: int

        @pytree.pytree_dataclass(data=("extra", "value"))
        class Child(Base):
            extra: int = 2

        child = Child(value=1, extra=3)
        for name in ("extra", "unregistered"):
            with self.assertRaises(FrozenInstanceError):
                setattr(child, name, 9)
        leaves, treedef = _flatten(child)
        self.assertEqual(leaves, [3, 1])
        self.assertEqual(_unflatten(treedef, [5, 4]), Child(value=4, extra=5))

    def test_undecorated_subclass_cannot_omit_its_new_fields(self):
        """Subclass registration rejects a partition that omits new fields."""

        @dataclass(frozen=True)
        class Base:
            value: int

        class Child(Base):
            extra: int = 2

        self.assert_rejected(ValueError, "Missing fields: \\{'extra'\\}", lambda: pytree_dataclass(data=("value",))(Child))

    def test_dataclass_reconstruction_calls_post_init(self):
        """Reconstruction runs ``__post_init__`` on the rebuilt field values."""
        constructed = []

        @pytree.pytree_dataclass(data=("value",))
        class Payload:
            value: int

            def __post_init__(self):
                constructed.append(self.value)

        leaves, treedef = _flatten(Payload(1))
        self.assertEqual(leaves, [1])
        self.assertEqual(_unflatten(treedef, [2]).value, 2)
        self.assertEqual(constructed, [1, 2])


class TestGroupedFieldWithoutCoils(JaxTestCase):
    """A coil set without coils has no field, whatever form the points take."""

    def test_empty_coil_set_gives_zero_field_for_list_points(self):
        """B, A and their derivatives vanish, shaped (P, 3, ...), for list points."""
        points = [[1.0, 0.0, 0.1], [0.9, 0.2, -0.1]]
        spec = make_grouped_coil_set_spec(())
        rows = [
            ("B", core_field.grouped_biot_savart_B_from_spec, (2, 3)),
            ("A", core_field.grouped_biot_savart_A_from_spec, (2, 3)),
            ("dB", core_field.grouped_biot_savart_dB_by_dX_from_spec, (2, 3, 3)),
            ("d2B", core_field.grouped_biot_savart_d2B_by_dXdX_from_spec, (2, 3, 3, 3)),
        ]
        for name, kernel, shape in rows:
            with self.subTest(name):
                np.testing.assert_array_equal(kernel(points, spec), np.zeros(shape))
        B, dB = core_field.grouped_biot_savart_B_and_dB_from_spec(points, spec)
        np.testing.assert_array_equal(B, np.zeros((2, 3)))
        np.testing.assert_array_equal(dB, np.zeros((2, 3, 3)))


def _curves(count, nfp, stellsym):
    return create_equally_spaced_curves(count, nfp, stellsym=stellsym, R0=1.0, R1=0.25, order=2, numquadpoints=12)


class TestCurrentExpressionExtraction(JaxTestCase):
    """Coil specs from DOFs keep current expressions and shared geometry."""

    def test_coil_specs_preserve_fixed_total_current_expression(self):
        """A fixed total current minus a free one keeps opposite derivatives."""
        curves = _curves(2, 1, False)
        for curve in curves:
            curve.fix_all()
        first, total = Current(3.0), Current(10.0)
        total.fix_all()
        field = JaxBiotSavart([Coil(curves[0], first), Coil(curves[1], total - first)])

        def currents(owner_dofs: jax.Array) -> jax.Array:
            specs = field.coil_specs_from_dofs(owner_dofs)
            return jnp.stack(tuple(coil.current.value[0] for coil in specs))

        owner_dofs = jnp.asarray([4.0], dtype=jnp.float64)
        np.testing.assert_allclose(currents(owner_dofs), np.asarray([4.0, 6.0]))
        np.testing.assert_allclose(jax.jacrev(currents)(owner_dofs), np.asarray([[1.0], [-1.0]]))

    def test_coil_set_spec_preserves_scaled_current_optimizer_coordinates(self):
        """A scaled current keeps its optimizer coordinate and Jacobian scale."""
        curve = _curves(1, 1, False)[0]
        curve.fix_all()
        field = JaxBiotSavart([Coil(curve, Current(1.0) * 1.0e5)])

        def current(owner_dofs: jax.Array) -> jax.Array:
            return field.coil_set_spec_from_dofs(owner_dofs).groups[0].currents

        owner_dofs = jnp.asarray(field.x, dtype=jnp.float64)
        np.testing.assert_array_equal(owner_dofs, np.asarray([1.0]))
        np.testing.assert_allclose(current(owner_dofs), np.asarray([1.0e5]))
        np.testing.assert_allclose(jax.jacrev(current)(owner_dofs), np.asarray([[1.0e5]]))

    def test_coil_specs_reuse_geometry_for_symmetry_replicas(self):
        """Symmetry replicas share one reconstructed immutable curve payload."""
        coils = coils_via_symmetries([_curves(1, 2, True)[0]], [Current(3.0)], 2, True)
        field = JaxBiotSavart(coils)
        coil_specs = field.coil_specs_from_dofs(jnp.asarray(field.x))
        self.assertEqual(len(coil_specs), 4)
        self.assertEqual(len({id(coil_spec.curve) for coil_spec in coil_specs}), 1)
