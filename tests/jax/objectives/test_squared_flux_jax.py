"""SquaredFluxJAX and the JAX integral_BdotN against native SquaredFlux."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsoptpp as sopp
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import SquaredFlux
from simsopt_jax.core.integral_bdotn import integral_BdotN
from simsopt_jax_adapters.field import BiotSavartJAX
from simsopt_jax_adapters.objectives import SquaredFluxJAX

_DEFINITIONS = ("quadratic flux", "normalized", "local")
_QA_INPUT = Path(__file__).resolve().parents[2] / "test_files" / "input.LandremanPaul2021_QA"


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
    adapter = SquaredFluxJAX(surface, BiotSavartJAX(coils), target=target, definition=definition)
    return surface, native, adapter


@pytest.mark.parametrize("definition", _DEFINITIONS)
@pytest.mark.parametrize(
    "target_kind, empty", [("none", False), ("array", False), ("none", True)],
    ids=["zero_target", "array_target", "empty_target"],
)
def test_integral_bdotn_matches_cpp(definition, target_kind, empty):
    rng = np.random.default_rng(5)
    B = rng.standard_normal((6, 7, 3))
    normal = rng.standard_normal((6, 7, 3))
    target = rng.standard_normal((6, 7)) if target_kind == "array" else np.zeros((6, 7))
    if empty:
        target = np.zeros((0,))
    expected = sopp.integral_BdotN(B, target, normal, definition)
    actual = integral_BdotN(jnp.asarray(B), jnp.asarray(target), jnp.asarray(normal), definition)
    np.testing.assert_allclose(float(actual), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("definition", _DEFINITIONS)
@pytest.mark.parametrize("case", ["zero_normal", "zero_field"])
def test_integral_bdotn_singular_inputs_match_cpp(definition, case):
    """Zero normals and zero fields give the C++ nan/inf (or value), not a masked result."""
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
    assert case == "zero_field" and definition == "quadratic flux" or not np.isfinite(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("definition", _DEFINITIONS)
@pytest.mark.parametrize(
    "target_kind, shared_dofs",
    [("none", False), ("array", True)],
    ids=["symmetric_coils", "target_and_shared_dofs"],
)
def test_squared_flux_matches_native_value_gradient_and_partials(definition, target_kind, shared_dofs):
    _, native, adapter = _objectives(definition, target_kind, shared_dofs=shared_dofs)
    assert adapter.dof_names == native.dof_names
    np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(_gradient(adapter), _gradient(native), rtol=1e-11, atol=1e-13)
    native_partials, adapter_partials = _partials(native), _partials(adapter)
    assert set(adapter_partials.data) == set(native_partials.data)
    for owner, expected in native_partials.data.items():
        # Fixed DOFs keep their partials, as in the native Derivative.
        np.testing.assert_allclose(
            adapter_partials.data[owner], expected, rtol=1e-11, atol=1e-13,
            err_msg=f"partials of {owner.name}",
        )


@pytest.mark.parametrize("definition", _DEFINITIONS)
def test_squared_flux_gradient_matches_central_differences(definition):
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


def test_squared_flux_follows_dof_changes_and_sets_field_points():
    surface, native, adapter = _objectives("quadratic flux")
    np.testing.assert_array_equal(
        adapter.field.get_points_cart(), surface.gamma().reshape((-1, 3))
    )
    adapter.x = np.asarray(adapter.x) * 1.01
    np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(_gradient(adapter), _gradient(native), rtol=1e-11, atol=1e-13)


def test_target_and_definition_are_read_at_every_evaluation():
    surface, native, adapter = _objectives("quadratic flux", "none", shared_dofs=True)
    adapter.J()
    adapter.dJ()
    target = np.ascontiguousarray(_target(surface, "array"))
    for definition in ("local", "normalized", "quadratic flux"):
        native.target, native.definition = target.copy(), definition
        adapter.target, adapter.definition = target.copy(), definition
        np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14, err_msg=definition)
        np.testing.assert_allclose(_gradient(adapter), _gradient(native), rtol=1e-11, atol=1e-13, err_msg=definition)
        assert adapter.fixed_surface_flux_spec().definition == definition
    native.target *= 3.0
    adapter.target *= 3.0
    np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(adapter.fixed_surface_flux_spec().target), native.target)


def test_squared_flux_makes_no_implicit_transfers():
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


def test_squared_flux_rejects_surface_changes_after_construction():
    surface, _, adapter = _objectives("quadratic flux")
    surface.set_rc(1, 0, surface.get_rc(1, 0) * 1.01)
    for evaluate in (adapter.J, adapter.dJ, adapter.fixed_surface_flux_spec):
        with pytest.raises(RuntimeError, match="surface DOFs have changed"):
            evaluate()


def test_squared_flux_rejects_unknown_definition():
    surface = _surface()
    with pytest.raises(ValueError, match="Unrecognized option"):
        SquaredFluxJAX(surface, BiotSavartJAX(_coils(surface)), definition="flux")
