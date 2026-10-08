"""JAX surface geometry against the native surface classes.

Every native quantity, coefficient derivative and VJP is compared with its JAX
counterpart (a value kernel of ``simsopt_jax.core.surface_geometry``, or a JAX
transform of ``surface_quantity_of_dofs``) for ``SurfaceRZFourier``,
``SurfaceXYZFourier`` and ``SurfaceXYZTensorFourier`` (clamped and not):
stellarator-symmetric and not, nfp 1 to 3 and the three phi ranges.
Derivatives are also checked against central differences of native values,
free-DOF projections against native objectives (fixed and shared DOFs), and
the spec boundary for native DOF layouts and coefficient entries, DOF
changes, recompilation, implicit transfers, singular normals and unsupported
surfaces.
"""

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    fixture_parity_lane,  # noqa: F401
    host_array,
    parity_default_device,
    parity_rng,
)

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import numpy as np
import pytest

from simsopt._core.derivative import Derivative, OptimizableDefaultDict
from simsopt._core.optimizable import Optimizable
from simsopt.geo import CurveXYZFourier, SurfaceGarabedian
from simsopt.geo.surface import Surface
from simsopt.geo.surfaceobjectives import Area, Volume
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core import surface_geometry
from simsopt_jax.core.specs import SurfaceSpec
from simsopt_jax.core.surface_fourier_series import surface_get_dofs, surface_spec_with_dofs
from simsopt_jax.core.surface_geometry import surface_gamma, surface_quantity_of_dofs
from simsopt_jax.runtime.host_boundary import disallow_host_transfers
from simsopt_jax_adapters.geo import surface_spec_from_surface

_NativeSurface = SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier
_Kernel = Callable[[SurfaceSpec], jax.Array]
_Transform = Callable[[Callable[[jax.Array], jax.Array]], Callable[[jax.Array], jax.Array]]


@dataclass(frozen=True)
class _Case:
    surface_class: type[_NativeSurface]
    stellsym: bool
    nfp: int
    range: str
    mpol: int = 2
    ntor: int = 2
    clamped_dims: tuple[bool, bool, bool] = (False, False, False)


_CASES = {
    "rz-stellsym-nfp1-full-torus": _Case(SurfaceRZFourier, True, 1, "full torus"),
    "rz-stellsym-nfp3-field-period": _Case(SurfaceRZFourier, True, 3, "field period"),
    "rz-nonsym-nfp2-half-period": _Case(SurfaceRZFourier, False, 2, "half period"),
    "rz-axisymmetric-ntor0": _Case(SurfaceRZFourier, False, 1, "full torus", 3, 0),
    "xyz-stellsym-nfp1-full-torus": _Case(SurfaceXYZFourier, True, 1, "full torus"),
    "xyz-stellsym-nfp3-field-period": _Case(SurfaceXYZFourier, True, 3, "field period"),
    "xyz-nonsym-nfp2-half-period": _Case(SurfaceXYZFourier, False, 2, "half period"),
    "tensor-stellsym-nfp1-full-torus": _Case(SurfaceXYZTensorFourier, True, 1, "full torus"),
    "tensor-stellsym-nfp3-field-period": _Case(
        SurfaceXYZTensorFourier, True, 3, "field period", 2, 1
    ),
    "tensor-nonsym-nfp2-half-period": _Case(SurfaceXYZTensorFourier, False, 2, "half period"),
    "tensor-clamped-stellsym-nfp3-field-period": _Case(
        SurfaceXYZTensorFourier, True, 3, "field period", clamped_dims=(True, False, True)
    ),
    "tensor-clamped-nonsym-nfp2-half-period": _Case(
        SurfaceXYZTensorFourier, False, 2, "half period", clamped_dims=(False, True, True)
    ),
}
_CLAMPED_CASES = [name for name, case in _CASES.items() if any(case.clamped_dims)]

# Native method name -> JAX kernel, by the naming rule surface_<native name>.
_VALUES: dict[str, _Kernel] = {
    name: getattr(surface_geometry, f"surface_{name}")
    for name in (
        "gamma",
        "gammadash1",
        "gammadash2",
        "gammadash1dash1",
        "gammadash1dash2",
        "gammadash2dash2",
        "normal",
        "unitnormal",
        "area",
        "volume",
    )
}


def _second_jacobian(function: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    return jax.jacfwd(jax.jacfwd(function))


# Native coefficient derivative -> (the native quantity it differentiates, JAX transform).
_DERIVATIVES: dict[str, tuple[str, _Transform]] = {
    **{
        f"d{name}_by_dcoeff": (name, jax.jacfwd)
        for name in _VALUES
        if name not in ("area", "volume")
    },
    "d2normal_by_dcoeffdcoeff": ("normal", _second_jacobian),
    "darea_by_dcoeff": ("area", jax.grad),
    "d2area_by_dcoeffdcoeff": ("area", jax.hessian),
    "dvolume_by_dcoeff": ("volume", jax.grad),
    "d2volume_by_dcoeffdcoeff": ("volume", jax.hessian),
}
# Native coefficient VJP -> the native quantity it pulls back through.
_VJPS = {f"d{name}_by_dcoeff_vjp": name for name in ("gamma", "gammadash1", "gammadash2", "normal")}
# Native clamped second derivatives truncate nfp / 2 and 1 / 2 to integers
# (simsoptpp/surfacexyztensorfourier.h); the clamped tests check these against
# differences of native first derivatives instead.
_NATIVE_CLAMPED_SECOND_DERIVATIVES = frozenset(
    {
        "gammadash1dash1",
        "gammadash2dash2",
        "dgammadash1dash1_by_dcoeff",
        "dgammadash2dash2_by_dcoeff",
    }
)
_RTOL = 1e-12
_ATOL = 1e-12


@partial(jax.jit, static_argnums=(0, 1))
def _coefficient_derivative(quantity: str, transform: _Transform, spec: SurfaceSpec) -> jax.Array:
    """A coefficient derivative the documented way: a jitted transform with the spec as argument."""
    return transform(surface_quantity_of_dofs(_VALUES[quantity], spec))(surface_get_dofs(spec))


@partial(jax.jit, static_argnums=0)
def _coefficient_vjp(quantity: str, spec: SurfaceSpec, cotangent: jax.Array) -> jax.Array:
    _, pullback = jax.vjp(surface_quantity_of_dofs(_VALUES[quantity], spec), surface_get_dofs(spec))
    return pullback(cotangent)[0]


def _evaluate(name: str, spec: SurfaceSpec) -> jax.Array:
    """The JAX counterpart of the native value or coefficient derivative ``name``."""
    if name in _DERIVATIVES:
        quantity, transform = _DERIVATIVES[name]
        return _coefficient_derivative(quantity, transform, spec)
    return _VALUES[name](spec)


def _differentiated(name: str) -> str:
    """The native quantity whose central differences approximate derivative ``name``."""
    quantity = _DERIVATIVES[name][0]
    return f"d{quantity}_by_dcoeff" if name.startswith("d2") else quantity


def _jitter(values: np.ndarray, seed: int) -> np.ndarray:
    return values + 0.01 * parity_rng(seed).standard_normal(values.size)


def _surface(
    case: str,
    seed: int = 0,
    quadpoints_phi: np.ndarray | None = None,
    quadpoints_theta: np.ndarray | None = None,
) -> _NativeSurface:
    """The native default torus (major radius 1, minor radius 0.1) with random DOF offsets.

    Explicit quadrature points replace the case's grid; the DOFs stay the same.
    """
    params = _CASES[case]
    extra = (
        {"clamped_dims": list(params.clamped_dims)}
        if params.surface_class is SurfaceXYZTensorFourier
        else {}
    )
    grid_phi, grid_theta = Surface.get_quadpoints(nphi=7, ntheta=8, range=params.range, nfp=params.nfp)
    surface = params.surface_class(
        quadpoints_phi=grid_phi if quadpoints_phi is None else quadpoints_phi,
        quadpoints_theta=grid_theta if quadpoints_theta is None else quadpoints_theta,
        nfp=params.nfp,
        stellsym=params.stellsym,
        mpol=params.mpol,
        ntor=params.ntor,
        **extra,
    )
    surface.x = _jitter(surface.get_dofs(), seed)
    return surface


def _native(surface: _NativeSurface, method: str, *args: np.ndarray) -> np.ndarray:
    """A copy: native methods return views of caches that later calls overwrite."""
    return np.array(getattr(surface, method)(*args))


def _assert_native(actual: jax.Array, expected: np.ndarray, quantity: str) -> None:
    np.testing.assert_allclose(
        host_array(actual),
        expected,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"JAX {quantity} differs from the native {quantity}",
    )


def _assert_central(
    actual: np.ndarray, plus: np.ndarray, minus: np.ndarray, step: float, message: str
) -> None:
    """``actual`` against the central difference ``(plus - minus) / (2 step)``."""
    central = (plus - minus) / (2 * step)
    np.testing.assert_allclose(
        actual, central, rtol=0.0, atol=1e-8 * np.max(np.abs(central)), err_msg=message
    )


def _cotangent(surface: _NativeSurface, seed: int) -> np.ndarray:
    return parity_rng(seed).standard_normal(surface.gamma().shape)


def _place(values: np.ndarray, spec: SurfaceSpec) -> jax.Array:
    return explicit_device_array(values, dtype=np.float64, reference=spec.quadpoints_phi)


def _free_gradient(
    owner: _NativeSurface, all_dofs_gradient: np.ndarray, objective: Optimizable
) -> np.ndarray:
    """A gradient with respect to all DOFs of ``owner``, projected onto ``objective``'s free DOFs."""
    return cast(np.ndarray, Derivative(OptimizableDefaultDict({owner: all_dofs_gradient}))(objective))


@contextmanager
def _compilations() -> Iterator[list[str]]:
    """Names of the JAX trace, lowering and compile events inside the block."""
    events: list[str] = []

    def record(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event.startswith("/jax/core/compile/"):
            events.append(event)

    jax.monitoring.register_event_duration_secs_listener(record)
    try:
        yield events
    finally:
        jax.monitoring.unregister_event_duration_listener(record)


def _evaluate_every_kernel(spec: SurfaceSpec, cotangent: jax.Array) -> dict[str, jax.Array]:
    results = {name: _evaluate(name, spec) for name in (*_VALUES, *_DERIVATIVES)}
    results.update(
        (name, _coefficient_vjp(quantity, spec, cotangent)) for name, quantity in _VJPS.items()
    )
    return jax.block_until_ready(results)


def _assert_every_kernel_native(
    results: dict[str, jax.Array], surface: _NativeSurface, cotangent: np.ndarray
) -> None:
    clamped = isinstance(surface, SurfaceXYZTensorFourier) and any(surface.clamped_dims)
    for name in (*_VALUES, *_DERIVATIVES):
        if not (clamped and name in _NATIVE_CLAMPED_SECOND_DERIVATIVES):
            _assert_native(results[name], _native(surface, name), name)
    for name in _VJPS:
        _assert_native(results[name], _native(surface, name, cotangent), name)


@pytest.mark.parametrize("case", _CASES)
def test_every_kernel_matches_native(case, parity_lane):
    surface = _surface(case)
    cotangent = _cotangent(surface, seed=1)
    with parity_default_device(parity_lane):
        spec = surface_spec_from_surface(surface)
        results = _evaluate_every_kernel(spec, _place(cotangent, spec))
    _assert_every_kernel_native(results, surface, cotangent)


@pytest.mark.parametrize(
    "case",
    ["rz-nonsym-nfp2-half-period", "xyz-stellsym-nfp3-field-period", "tensor-nonsym-nfp2-half-period"],
)
def test_coefficient_derivatives_match_central_differences_of_native_values(case):
    surface = _surface(case)
    spec = surface_spec_from_surface(surface)
    dofs = surface.get_dofs()
    direction = parity_rng(2).standard_normal(dofs.size)
    direction /= np.linalg.norm(direction)
    # Truncation error ~ step**2 and rounding error ~ 1e-16 / step balance here.
    step = 1e-6

    def native_at(offset: float, quantity: str) -> np.ndarray:
        surface.set_dofs(dofs + offset * direction)
        return _native(surface, quantity)

    for name in _DERIVATIVES:
        quantity = _differentiated(name)
        _assert_central(
            host_array(_evaluate(name, spec)) @ direction,
            native_at(step, quantity),
            native_at(-step, quantity),
            step,
            f"JAX {name} disagrees with central differences of native {quantity}",
        )


@pytest.mark.parametrize("case", _CLAMPED_CASES)
def test_clamped_second_derivatives_are_derivatives_of_native_first_derivatives(case):
    # Each second derivative, and its coefficient Jacobian, against central
    # differences in the quadrature points of the native first derivative and of
    # its native coefficient Jacobian, evaluated on shifted grids.
    surface = _surface(case)
    spec = surface_spec_from_surface(surface)
    quadpoints_phi = np.asarray(surface.quadpoints_phi)
    quadpoints_theta = np.asarray(surface.quadpoints_theta)
    step = 1e-6

    def native_on_shifted_grid(quantity: str, phi_offset: float, theta_offset: float) -> np.ndarray:
        shifted = _surface(case, quadpoints_phi=quadpoints_phi + phi_offset,
                           quadpoints_theta=quadpoints_theta + theta_offset)
        return _native(shifted, quantity)

    for name, first_derivative, phi_step, theta_step in (
        ("gammadash1dash1", "gammadash1", step, 0.0),
        ("gammadash1dash2", "gammadash1", 0.0, step),
        ("gammadash2dash2", "gammadash2", 0.0, step),
    ):
        for jax_name, native_name in (
            (name, first_derivative),
            (f"d{name}_by_dcoeff", f"d{first_derivative}_by_dcoeff"),
        ):
            _assert_central(
                host_array(_evaluate(jax_name, spec)),
                native_on_shifted_grid(native_name, phi_step, theta_step),
                native_on_shifted_grid(native_name, -phi_step, -theta_step),
                step,
                f"JAX {jax_name} is not the derivative of native {native_name}",
            )


@pytest.mark.parametrize("case", ["rz-stellsym-nfp3-field-period", "tensor-stellsym-nfp3-field-period"])
def test_free_dof_gradients_match_native_objectives_with_fixed_and_shared_dofs(case):
    surface = _surface(case)
    names = list(surface.local_full_dof_names)
    fixed = (str(names[0]), str(names[-1]))
    for name in fixed:
        surface.fix(name)
    spec = surface_spec_from_surface(surface)
    area, volume = Area(surface), Volume(surface)
    # A different quadrature grid on a surface that shares the DOFs.
    shared_area = Area(surface, range="half period", nphi=5, ntheta=6)
    shared_spec = surface_spec_from_surface(shared_area.surface)

    for objective, objective_spec, gradient in (
        (area, spec, "darea_by_dcoeff"),
        (volume, spec, "dvolume_by_dcoeff"),
        (shared_area, shared_spec, "darea_by_dcoeff"),
    ):
        free_gradient = _free_gradient(
            objective.surface, host_array(_evaluate(gradient, objective_spec)), objective
        )
        assert free_gradient.size == surface.get_dofs().size - len(fixed)
        np.testing.assert_allclose(
            free_gradient,
            cast(np.ndarray, objective.dJ()),
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"free-DOF gradient of {type(objective).__name__} differs from native",
        )

    # The VJP of a least-squares fit of gamma, projected like a native objective's.
    cotangent = _cotangent(surface, seed=3)
    jax_vjp = host_array(_coefficient_vjp("gamma", spec, _place(cotangent, spec)))
    np.testing.assert_allclose(
        _free_gradient(surface, jax_vjp, surface),
        _free_gradient(surface, surface.dgamma_by_dcoeff_vjp(cotangent), surface),
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize(
    "case",
    ["rz-nonsym-nfp2-half-period", "xyz-nonsym-nfp2-half-period", "tensor-clamped-stellsym-nfp3-field-period"],
)
def test_new_dofs_reuse_the_programs_without_implicit_transfers(case, parity_lane):
    surface = _surface(case)
    surface.fix(str(list(surface.local_full_dof_names)[3]))
    cotangent = _cotangent(surface, seed=4)
    with parity_default_device(parity_lane), disallow_host_transfers():
        first_spec = surface_spec_from_surface(surface)
        placed_cotangent = _place(cotangent, first_spec)
        first_results = _evaluate_every_kernel(first_spec, placed_cotangent)
    _assert_every_kernel_native(first_results, surface, cotangent)
    first_gamma = _native(surface, "gamma")

    surface.x = _jitter(np.asarray(surface.x), seed=5)
    with parity_default_device(parity_lane), disallow_host_transfers(), _compilations() as compilations:
        second_spec = surface_spec_from_surface(surface)
        second_results = _evaluate_every_kernel(second_spec, placed_cotangent)
    assert compilations == [], "new DOF values retraced or recompiled a program"
    _assert_every_kernel_native(second_results, surface, cotangent)
    devices = {device for result in second_results.values() for device in result.devices()}
    assert devices == second_spec.quadpoints_phi.devices()
    # A spec is a snapshot: the first one still evaluates the first surface.
    _assert_native(surface_gamma(first_spec), first_gamma, "gamma")


@pytest.mark.parametrize(
    "case",
    ["rz-axisymmetric-ntor0", "xyz-nonsym-nfp2-half-period", "tensor-clamped-stellsym-nfp3-field-period"],
)
def test_spec_dofs_follow_the_native_get_and_set_dofs(case):
    surface = _surface(case)
    spec = surface_spec_from_surface(surface)
    np.testing.assert_array_equal(host_array(surface_get_dofs(spec)), surface.get_dofs())

    new_dofs = _jitter(surface.get_dofs(), seed=8)
    moved = surface_spec_with_dofs(spec, _place(new_dofs, spec))
    surface.set_dofs(new_dofs)
    # Every coefficient array and quadrature grid equals the native one after set_dofs.
    for moved_leaf, native_leaf in zip(
        jax.tree.leaves(moved), jax.tree.leaves(surface_spec_from_surface(surface)), strict=True
    ):
        np.testing.assert_array_equal(host_array(moved_leaf), host_array(native_leaf))
    np.testing.assert_array_equal(host_array(surface_get_dofs(moved)), new_dofs)


def _write_entries_outside_the_dofs(surface: _NativeSurface) -> None:
    """Native coefficient entries that no DOF holds: summed or skipped as natively."""
    if isinstance(surface, SurfaceRZFourier):
        surface.rc[0, surface.ntor - 1] = 0.05  # m = 0, n < 0: summed
        surface.zs[0, 0] = -0.03  # m = 0, n = -ntor: summed
        surface.rs[1, surface.ntor] = 0.07  # stellarator symmetric: skipped
    elif isinstance(surface, SurfaceXYZFourier):
        surface.xc[0, surface.ntor - 1] = 0.05  # m = 0, n < 0: summed
        surface.xs[1, surface.ntor] = 0.02  # stellarator symmetric: summed anyway
    else:
        surface.xcs[surface.mpol + 1, 0] = 0.05  # stellarator symmetric: skipped
    surface.local_full_x = surface.get_dofs()


@pytest.mark.parametrize(
    "case",
    ["rz-stellsym-nfp3-field-period", "xyz-stellsym-nfp3-field-period", "tensor-stellsym-nfp3-field-period"],
)
def test_coefficient_entries_outside_the_dofs_enter_the_geometry_as_natively(case):
    surface = _surface(case)
    unmodified_gamma = _native(surface, "gamma")
    _write_entries_outside_the_dofs(surface)
    if not isinstance(surface, SurfaceXYZTensorFourier):
        assert not np.allclose(_native(surface, "gamma"), unmodified_gamma)
    cotangent = _cotangent(surface, seed=9)
    spec = surface_spec_from_surface(surface)
    placed_cotangent = _place(cotangent, spec)
    _assert_every_kernel_native(_evaluate_every_kernel(spec, placed_cotangent), surface, cotangent)

    new_dofs = _jitter(surface.get_dofs(), seed=10)
    moved = surface_spec_with_dofs(spec, _place(new_dofs, spec))
    surface.set_dofs(new_dofs)
    _assert_every_kernel_native(_evaluate_every_kernel(moved, placed_cotangent), surface, cotangent)


def _singular_surface(singularity: str) -> SurfaceRZFourier:
    """``cusp``: dr/dtheta = dz/dtheta = 0 exactly at theta = 0 (normal 0 there).
    ``underflow``: a surface of size 1e-85, whose normals' squares underflow past
    the subnormal range to 0."""
    ntor, scale = (0, 1.0) if singularity == "cusp" else (1, 1e-85)
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nphi=5, ntheta=6, nfp=2, stellsym=True, mpol=2, ntor=ntor, range="field period"
    )
    surface.set_rc(0, 0, 1.0 * scale)
    surface.set_rc(1, 0, 0.3 * scale)
    surface.set_zs(1, 0, 0.3 * scale)
    if singularity == "cusp":
        surface.set_rc(2, 0, 0.05)
        surface.set_zs(2, 0, -0.15)
    else:
        surface.set_rc(1, 1, 0.02 * scale)
    return surface


@pytest.mark.parametrize("singularity", ["cusp", "underflow"])
def test_singular_normals_give_the_native_non_finite_values(singularity):
    surface = _singular_surface(singularity)
    spec = surface_spec_from_surface(surface)
    assert not np.isfinite(_native(surface, "unitnormal")).all()

    for name in ("normal", "unitnormal", "area", "volume",
                 "darea_by_dcoeff", "d2area_by_dcoeffdcoeff", "dvolume_by_dcoeff"):
        # Equal NaN and signed infinity positions, and equal finite values.
        np.testing.assert_allclose(
            host_array(_evaluate(name, spec)),
            _native(surface, name),
            rtol=_RTOL,
            atol=_ATOL,
            equal_nan=True,
            err_msg=f"JAX {name} differs from native at singular normals",
        )
    # Native's closed form and autodiff combine a singular point's infinite and
    # NaN terms in different orders, so only the finite entries must agree.
    actual = host_array(_evaluate("dunitnormal_by_dcoeff", spec))
    expected = _native(surface, "dunitnormal_by_dcoeff")
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=_RTOL, atol=_ATOL)


class _SurfaceRZFourierSubclass(SurfaceRZFourier):
    pass


class _SurfaceXYZTensorFourierSubclass(SurfaceXYZTensorFourier):
    pass


@pytest.mark.parametrize(
    "make_object",
    [
        lambda: SurfaceGarabedian(),
        lambda: _SurfaceRZFourierSubclass(),
        lambda: _SurfaceXYZTensorFourierSubclass(),
        lambda: CurveXYZFourier(8, 1),
    ],
    ids=["SurfaceGarabedian", "SurfaceRZFourier-subclass", "SurfaceXYZTensorFourier-subclass", "curve"],
)
def test_specs_refuse_other_classes(make_object):
    with pytest.raises(TypeError, match="supports SurfaceRZFourier, SurfaceXYZFourier"):
        surface_spec_from_surface(make_object())
