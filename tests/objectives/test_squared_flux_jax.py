"""JaxSquaredFlux and the JAX integral_BdotN against native SquaredFlux."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    from core.test_buffer_ownership import _host_array, make_execution_gate
    import jax
    import jax.numpy as jnp
    import simsoptpp as sopp
    from simsopt._core.derivative import Derivative
    from simsopt._core.optimizable import Optimizable
    from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
    from simsopt.geo import CurveXYZFourier, SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.objectives import SquaredFlux
    from simsopt_jax.core.integral_bdotn import integral_BdotN
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.objectives import JaxSquaredFlux


from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np


_DEFINITIONS = ("quadratic flux", "normalized", "local")
_QA_INPUT = Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"


def _surface() -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(str(_QA_INPUT), range="half period", nphi=8, ntheta=9)


def _coils(surface, *, shared_dofs: bool = False) -> list:
    """Symmetric coils with a fixed current and fixed curve DOFs at a perturbed state.

    With ``shared_dofs``, one more coil carries a curve that shares the DOFs of
    the second base curve and a current that shares the DOFs of the third.
    """
    base_curves = create_equally_spaced_curves(
        3, surface.nfp, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24
    )
    rng = np.random.default_rng(11)
    for curve in base_curves:
        curve.x = curve.x + 0.02 * rng.standard_normal(curve.x.shape)
    base_curves[0].fix("xc(1)")
    base_currents = [Current(1e5), Current(1.2e5), Current(0.9e5)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    if shared_dofs:
        twin_curve = CurveXYZFourier(
            np.linspace(0, 1, 20, endpoint=False) + 0.01, 3, dofs=base_curves[1].dofs
        )
        twin_current = Current(0.0, dofs=base_currents[2].dofs)
        coils.append(Coil(twin_curve, 0.5 * twin_current))
    return coils


def _target(surface, kind: str):
    if kind == "none":
        return None
    nphi, ntheta = surface.normal().shape[:2]
    return 0.05 * np.sin(np.linspace(0.0, 3.0, nphi * ntheta)).reshape((nphi, ntheta))


def _partials(objective: Optimizable) -> Derivative:
    return cast(Callable[..., Derivative], objective.dJ)(partials=True)


def _gradient(objective: Optimizable) -> np.ndarray:
    """The free-DOF gradient ``dJ()``."""
    return cast(Callable[[], np.ndarray], objective.dJ)()


def _objectives(definition, target_kind="none", *, shared_dofs=False):
    surface = _surface()
    coils = _coils(surface, shared_dofs=shared_dofs)
    target = _target(surface, target_kind)
    native = SquaredFlux(surface, BiotSavart(coils), target=target, definition=definition)
    adapter = JaxSquaredFlux(surface, JaxBiotSavart(coils), target=target, definition=definition)
    return surface, native, adapter




















class TestSquaredFluxJax(JaxTestCase):
    def test_integral_bdotn_snapshots_numpy_before_pending_evaluation(self):
        """All flux definitions retain caller field, target and normal buffers."""
        for definition in _DEFINITIONS:
            for operand_index in range(3):
                for misaligned in (False, True):
                    with self.subTest(definition=definition, operand=operand_index, misaligned=misaligned), self.case():
                        self._case_integral_snapshot(definition, operand_index, misaligned)

    def _case_integral_snapshot(self, definition, operand_index, misaligned):
        """Mutate one host operand while a warmed flux integral waits for device work."""
        host_inputs = [_host_array(shape, misaligned=misaligned) for shape in ((32, 32, 3), (32, 32), (32, 32, 3))]
        expected = np.asarray(integral_BdotN(host_inputs[0], host_inputs[1], host_inputs[2], definition)).copy()
        inputs: list[np.ndarray | jax.Array] = [jnp.asarray(value) for value in host_inputs]
        inputs[operand_index] = host_inputs[operand_index]
        gate_index = 2 if operand_index == 0 else 0
        gate = make_execution_gate()
        gate(inputs[gate_index]).block_until_ready()
        inputs[gate_index] = gate(inputs[gate_index])
        result = integral_BdotN(inputs[0], inputs[1], inputs[2], definition)
        self.assertFalse(result.is_ready(), "flux integral finished before caller mutation")
        host_inputs[operand_index][...] = -7.0
        np.testing.assert_array_equal(result.block_until_ready(), expected)

    def test_integral_bdotn_supports_nested_jit_and_grad(self):
        """Static definition specialization and field derivatives work inside outer jit."""
        B = jnp.asarray(_host_array((2, 3, 3)))
        target = jnp.zeros((2, 3))
        normal = jnp.asarray(_host_array((2, 3, 3)))
        for definition in _DEFINITIONS:
            with self.subTest(definition=definition):
                evaluate = lambda field: integral_BdotN(field, target, normal, definition)
                np.testing.assert_array_equal(jax.jit(evaluate)(B), evaluate(B))
                np.testing.assert_allclose(jax.jit(jax.grad(evaluate))(B), jax.grad(evaluate)(B), rtol=1e-14, atol=1e-14)

    def test_integral_bdotn_matches_cpp(self):
        """All flux definitions and target forms match the native C++ integral."""
        for case_id_0, (target_kind, empty) in zip(["zero_target", "array_target", "empty_target"], [("none", False), ("array", False), ("none", True)], strict=True):
            for definition in _DEFINITIONS:
                with self.subTest(case_id_0=case_id_0, target_kind=target_kind, empty=empty, definition=definition), self.case() as patches:
                    self._case_integral_bdotn_matches_cpp(target_kind, empty, definition, patches)

    def _case_integral_bdotn_matches_cpp(self, target_kind, empty, definition, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(5)
        B = rng.standard_normal((6, 7, 3))
        normal = rng.standard_normal((6, 7, 3))
        target = rng.standard_normal((6, 7)) if target_kind == "array" else np.zeros((6, 7))
        if empty:
            target = np.zeros((0,))
        expected = sopp.integral_BdotN(B, target, normal, definition)
        actual = integral_BdotN(jnp.asarray(B), jnp.asarray(target), jnp.asarray(normal), definition)
        np.testing.assert_allclose(float(actual), expected, rtol=1e-12, atol=1e-14)

    def test_integral_bdotn_singular_inputs_match_cpp(self):
        """Zero normals and zero fields give the C++ nan/inf (or value), not a masked result."""
        for case in ["zero_normal", "zero_field"]:
            for definition in _DEFINITIONS:
                with self.subTest(case=case, definition=definition), self.case() as patches:
                    self._case_integral_bdotn_singular_inputs_match_cpp(case, definition, patches)

    def _case_integral_bdotn_singular_inputs_match_cpp(self, case, definition, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(6)
        B = rng.standard_normal((6, 7, 3))
        normal = rng.standard_normal((6, 7, 3))
        target = np.zeros((6, 7))
        if case == "zero_normal":
            normal[2, 3] = 0.0
        else:
            B[:] = 0.0
        expected = sopp.integral_BdotN(B, target, normal, definition)
        actual = float(integral_BdotN(jnp.asarray(B), jnp.asarray(target), jnp.asarray(normal), definition))
        self.assertTrue(case == "zero_field" and definition == "quadratic flux" or not np.isfinite(expected), 'case == "zero_field" and definition == "quadratic flux" or not np.isfinite(expected)')
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_squared_flux_matches_native_value_gradient_and_partials(self):
        """Squared flux matches native values, free gradients and fixed partials."""
        for case_id_0, (target_kind, shared_dofs) in zip(["symmetric_coils", "target_and_shared_dofs"], [("none", False), ("array", True)], strict=True):
            for definition in _DEFINITIONS:
                with self.subTest(case_id_0=case_id_0, target_kind=target_kind, shared_dofs=shared_dofs, definition=definition), self.case() as patches:
                    self._case_squared_flux_matches_native_value_gradient_and_partials(target_kind, shared_dofs, definition, patches)

    def _case_squared_flux_matches_native_value_gradient_and_partials(self, target_kind, shared_dofs, definition, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        _, native, adapter = _objectives(definition, target_kind, shared_dofs=shared_dofs)
        self.assertTrue(adapter.dof_names == native.dof_names, 'adapter.dof_names == native.dof_names')
        np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(_gradient(adapter), _gradient(native), rtol=1e-11, atol=1e-13)
        native_partials, adapter_partials = _partials(native), _partials(adapter)
        self.assertTrue(set(adapter_partials.data) == set(native_partials.data), 'set(adapter_partials.data) == set(native_partials.data)')
        for owner, expected in native_partials.data.items():
            # Fixed DOFs keep their partials, as in the native Derivative.
            np.testing.assert_allclose(
                adapter_partials.data[owner], expected, rtol=1e-11, atol=1e-13,
                err_msg=f"partials of {owner.name}",
            )

    def test_squared_flux_gradient_matches_central_differences(self):
        """Squared-flux gradients agree with independent central differences."""
        for definition in _DEFINITIONS:
            with self.subTest(definition=definition), self.case() as patches:
                self._case_squared_flux_gradient_matches_central_differences(definition, patches)

    def _case_squared_flux_gradient_matches_central_differences(self, definition, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        _, _, adapter = _objectives(definition, "array", shared_dofs=True)
        x0 = np.array(adapter.x, dtype=float)
        direction = np.random.default_rng(2).standard_normal(x0.shape) * np.maximum(np.abs(x0), 1.0)
        step = 1e-7
        adapter.x = x0 + step * direction
        plus = adapter.J()
        adapter.x = x0 - step * direction
        minus = adapter.J()
        adapter.x = x0
        np.testing.assert_allclose(_gradient(adapter) @ direction, (plus - minus) / (2 * step), rtol=1e-6)

    def test_squared_flux_follows_dof_changes_and_sets_field_points(self):
        """Construction sets surface evaluation points and values follow coil DOF changes."""
        surface, native, adapter = _objectives("quadratic flux")
        np.testing.assert_array_equal(
            adapter.field.get_points_cart(), surface.gamma().reshape((-1, 3))
        )
        adapter.x = np.asarray(adapter.x) * 1.01
        np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(_gradient(adapter), _gradient(native), rtol=1e-11, atol=1e-13)

    def test_target_and_definition_are_read_at_every_evaluation(self):
        """Target and definition mutations are reflected in values, gradients and specs."""
        surface, native, adapter = _objectives("quadratic flux", "none", shared_dofs=True)
        adapter.J()
        adapter.dJ()
        target = np.ascontiguousarray(_target(surface, "array"))
        for definition in ("local", "normalized", "quadratic flux"):
            native.target, native.definition = target.copy(), definition
            adapter.target, adapter.definition = target.copy(), definition
            np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14, err_msg=definition)
            np.testing.assert_allclose(_gradient(adapter), _gradient(native), rtol=1e-11, atol=1e-13, err_msg=definition)
            self.assertTrue(adapter.fixed_surface_flux_spec().definition == definition, 'adapter.fixed_surface_flux_spec().definition == definition')
        native.target *= 3.0
        adapter.target *= 3.0
        np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(np.asarray(adapter.fixed_surface_flux_spec().target), native.target)

    def test_squared_flux_makes_no_implicit_transfers(self):
        """J and dJ move data only through explicit transfers, also after a DOF change."""
        _, native, adapter = _objectives("quadratic flux", "array", shared_dofs=True)
        adapter.J()
        adapter.dJ()
        with jax.transfer_guard("disallow"):
            adapter.J()
            adapter.dJ()
            adapter.x = np.asarray(adapter.x) * 1.01
            value = adapter.J()
            gradient = _gradient(adapter)
        np.testing.assert_allclose(value, native.J(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(gradient, _gradient(native), rtol=1e-11, atol=1e-13)

    def test_squared_flux_rejects_surface_changes_after_construction(self):
        """Changing captured surface DOFs rejects value, gradient and spec evaluation."""
        surface, _, adapter = _objectives("quadratic flux")
        surface.set_rc(1, 0, surface.get_rc(1, 0) * 1.01)
        for evaluate in (adapter.J, adapter.dJ, adapter.fixed_surface_flux_spec):
            with self.assertRaisesRegex(RuntimeError, "surface DOFs have changed"):
                evaluate()

    def test_squared_flux_rejects_unknown_definition(self):
        """An unsupported flux definition raises ValueError at construction."""
        surface = _surface()
        with self.assertRaisesRegex(ValueError, "Unrecognized option"):
            JaxSquaredFlux(surface, JaxBiotSavart(_coils(surface)), definition="flux")

    def test_squared_flux_gradient_taylor(self):
        """Every flux definition attains native centered Taylor convergence."""
        for definition in _DEFINITIONS:
            with self.subTest(definition=definition), self.case():
                self._case_squared_flux_gradient_taylor(definition)

    def _case_squared_flux_gradient_taylor(self, definition):
        """Use the native flux stencil and six halved steps with shared coil DOFs."""
        _, _, adapter = _objectives(definition, "array", shared_dofs=True)
        x0 = np.array(adapter.x, dtype=float)
        direction = np.random.RandomState(1).uniform(size=x0.shape)
        derivative = _gradient(adapter) @ direction
        previous_error = 1e10
        for exponent in range(11, 17):
            step = 0.5 ** exponent
            adapter.x = x0 + step * direction
            plus = adapter.J()
            adapter.x = x0 - step * direction
            error = abs((plus - adapter.J()) / (2 * step) - derivative)
            self.assertLess(error, 0.6 ** 2 * previous_error, f"{definition}, step={step}")
            previous_error = error
        adapter.x = x0
