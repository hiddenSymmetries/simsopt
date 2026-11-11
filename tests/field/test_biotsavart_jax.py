"""JaxBiotSavart and its kernels against native BiotSavart, analytic fields and Taylor tests, and its state cache."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, aligned_host_copy, disallow_host_transfers, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax.extend.core import jaxprs_in_params
    from simsopt_jax.backend import set_backend
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core import biotsavart as core_biotsavart
    from simsopt_jax.core import coil_specs_from_dof_extraction_spec, curve_gamma_and_gammadash_from_spec
    from simsopt_jax.core.field import group_biot_savart_B_vjp, grouped_biot_savart_B_from_spec
    from simsopt_jax.core.field import grouped_coil_set_spec_from_inputs
    from simsopt_jax.core.specs import CoilGroupSpec, GroupedCoilSetSpec, make_field_eval_spec
    from simsopt_jax_adapters.field import biotsavart_backend as backend
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartFieldPullback, JaxBiotSavart
    from simsopt_jax_adapters.field.biotsavart_backend import _owner_segments_from_free_positions
    from simsopt_jax_adapters.field.biotsavart_backend import _per_coil_unit_field, _per_coil_unit_field_with_batch_size
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

import copy
import itertools
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

import numpy as np
from scipy.optimize import minimize

from simsopt import load
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.configs import get_data
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.field.magneticfield import MagneticFieldMultiply, MagneticFieldSum
from simsopt.geo import CurveLength, CurveXYZFourier, RotatedCurve, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.geo.curveperturbed import CurvePerturbed, GaussianSampler, PerturbationSample
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import FrameRotation, FramedCurveCentroid, FramedCurveFrenet
from simsopt.objectives import SquaredFlux

from .test_biotsavart import get_curve

# (rtol, atol) pairs.
_DIRECT = (1e-10, 1e-12)
_FIRST = (1e-08, 1e-10)
_SECOND = (1e-10, 1e-12)
_TIGHT = (1e-12, 1e-14)
_VJP = (1e-11, 1e-13)


def _close(actual, expected, tol, err_msg=""):
    """``assert_allclose`` with ``tol`` = (rtol, atol)."""
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=tol[0], atol=tol[1], err_msg=err_msg)


def _grid(**axes):
    """Every combination of ``axes`` as keyword rows, the first axis outermost."""
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def _ncsx():
    """Native BiotSavart on the NCSX coils at 50 points near the torus, and its arrays."""
    curves, currents_objs, _, nfp, _ = get_data("ncsx")
    coils = coils_via_symmetries(curves, currents_objs, nfp, stellsym=True)
    bs = BiotSavart(coils)
    points_np = np.random.default_rng(42).standard_normal((50, 3)) * 0.3
    points_np[:, 0] += 1.0
    bs.set_points(points_np)
    gammas_np = np.array([coil.curve.gamma() for coil in coils])
    gds_np = np.array([coil.curve.gammadash() for coil in coils])
    currents_np = np.array([coil.current.get_value() for coil in coils])
    return bs, points_np, gammas_np, gds_np, currents_np


def _ncsx_fields():
    """The NCSX native field and a JaxBiotSavart over its coils at the same points."""
    bs, points_np, *_ = _ncsx()
    bs_jax = JaxBiotSavart(list(bs._coils))
    bs_jax.set_points(points_np)
    return bs, bs_jax, points_np


def _native_taylor_fixture():
    """Coil and points of the native ``tests/field/test_biotsavart.py`` Taylor tests."""
    curve = get_curve()
    rng = np.random.default_rng(1)
    points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
    points += 0.001 * (rng.random(points.shape) - 0.5)
    return Coil(curve, Current(1e4)), points


def _cart_points_to_cyl(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.ascontiguousarray(np.stack((np.sqrt(x * x + y * y), np.arctan2(y, x), z), axis=1))


def _assert_cylindrical_points_match_cpu(jax_field, cpu_field):
    _close(jax_field.get_points_cyl(), np.asarray(cpu_field.get_points_cyl()), (0, 1e-15))


def _assert_cylindrical_accessors_match_cpu(jax_field, cpu_field):
    _assert_cylindrical_points_match_cpu(jax_field, cpu_field)
    _close(jax_field.AbsB(), np.asarray(cpu_field.AbsB()), _DIRECT)
    assert np.asarray(jax_field.AbsB()).shape == np.asarray(cpu_field.AbsB()).shape
    for method_name in ("B_cyl", "A_cyl"):
        _close(getattr(jax_field, method_name)(), np.asarray(getattr(cpu_field, method_name)()), _DIRECT)
    _close(jax_field.GradAbsB_cyl(), np.asarray(cpu_field.GradAbsB_cyl()), _FIRST)


# Curves and currents of every supported type, at two points near the coils.

_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]])
_FIELDS = ("B", "A", "dB_by_dX", "dA_by_dX")
_COMPOSED_WRAPPERS = (
    "shared_filament",
    "centroid_filament",
    "nested_perturbed",
    "perturbed_filament",
    "perturbed_shared_filament",
)
_FRAMES = {frame.__name__: frame for frame in (FramedCurveFrenet, FramedCurveCentroid)}


def _curves(count=2):
    return create_equally_spaced_curves(count, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=32)


def _vjp(field, quantity, cotangent) -> Derivative:
    # Warm native caches, including the unit-current A cache.
    getattr(field, quantity)()
    if quantity == "dB_by_dX":
        return field.B_and_dB_vjp(np.zeros_like(_POINTS), cotangent)[1]
    if quantity == "dA_by_dX":
        return field.A_and_dA_vjp(np.zeros_like(_POINTS), cotangent)[1]
    return getattr(field, quantity + "_vjp")(cotangent)


def _gradient(derivative: Derivative, owner: Optimizable) -> np.ndarray:
    return cast(np.ndarray, derivative(owner))


def _partials(derivative: Derivative, owner: Optimizable) -> Derivative:
    return cast(Derivative, derivative(owner, as_derivative=True))


def _assert_partials_match(actual: Derivative, expected: Derivative):
    """Every owner partial of ``expected`` matches ``actual``'s at ``_VJP``."""
    for owner in expected.data:
        _close(actual.data[owner], expected.data[owner], _VJP)


def _cotangent(quantity):
    shape = (2, 3, 3) if "by_dX" in quantity else (2, 3)
    return np.linspace(-0.7, 0.9, np.prod(shape)).reshape(shape)


def _affine_coils(scale):
    """Two fixed curves; the second current is ``scale * (fixed total - first)``."""
    curves = _curves()
    for curve in curves:
        curve.fix_all()
    first, total = Current(3.0), Current(10.0)
    total.fix_all()
    return [Coil(curves[0], first), Coil(curves[1], scale * (total - first))], first


def _composed_native_field(wrapper, fixed):
    """Native oracle on one composed curve, with its base curve and frame rotation."""
    curve = create_equally_spaced_curves(1, 1, stellsym=False, R0=1.0, R1=0.25, order=1, numquadpoints=24)[0]
    current = Current(1e5)
    shared = "shared" in wrapper
    rotation = FrameRotation(curve.quadpoints, order=4 if shared else 1, dofs=curve.dofs if shared else None)
    if not shared:
        rotation.local_full_x = np.array([0.2, -0.1, 0.05])
    if fixed:
        curve.fix(0)
        rotation.fix(rotation.local_dof_names[1])
        current.fix_all()
    sampler = GaussianSampler(curve.quadpoints, 1e-3, 0.2, n_derivs=1)
    sample = PerturbationSample(sampler, randomgen=np.random.default_rng(20261005))
    if wrapper == "nested_perturbed":
        wrapped = CurvePerturbed(CurvePerturbed(curve, sample), sample)
    else:
        frame = FramedCurveCentroid if "centroid" in wrapper else FramedCurveFrenet
        wrapped = CurveFilament(frame(curve, rotation), 0.01, 0.02)
        if wrapper.startswith("perturbed"):
            wrapped = CurvePerturbed(wrapped, sample)
    native = BiotSavart([Coil(wrapped, current)]).set_points(_POINTS)
    return native, curve, rotation


def _filament(frame, quadpoints):
    """A non-circular rotated filament on ``frame`` sampled at ``quadpoints``."""
    reference = create_equally_spaced_curves(1, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=24)[0]
    curve = CurveXYZFourier(quadpoints, 2)
    curve.x = reference.x + np.random.default_rng(20261011).normal(0.0, 0.03, reference.x.size)
    rotation = FrameRotation(quadpoints, 1)
    rotation.x = np.array([0.2, -0.1, 0.05])
    return CurveFilament(frame(curve, rotation), 0.05, 0.08)


def _jax_filament_geometry(frame, quadpoints):
    spec = curve_spec_from_adapter_curve(_filament(frame, quadpoints))
    gamma, gammadash = curve_gamma_and_gammadash_from_spec(spec)
    return np.asarray(gamma), np.asarray(gammadash)


# Fused B and dB/dX against autodiff: about 45 ulp (measured worst 2e-15 integrand).
_ROUND_OFF = 1e-14

# (name, coil_chunk_size, quadrature_block_size, loop trip counts) for 5 coils of
# 24 nodes. One or two blocks or chunks have no loop; three or more loop over the
# full ones and add the tail (coil_chunk_tail: two 2-coil chunks, each traced in
# the loop body and the 1-coil tail, over three 7-node blocks).
_TILINGS = (
    ("two_quadrature_blocks", 0, 16, ()),
    ("three_quadrature_blocks_and_tail", 0, 7, (3,)),
    ("two_coil_chunks", 3, 16, ()),
    ("coil_chunk_tail", 2, 7, (2, 3, 3)),
)

_FORWARD_KERNELS = (
    ("B", "VALUE"),
    ("B", "JACOBIAN"),
    ("A", "VALUE"),
    ("A", "JACOBIAN"),
    ("B", "HESSIAN"),
    ("A", "HESSIAN"),
    ("B", "VALUE_AND_JACOBIAN"),
)

# Measured |tiled - untiled| <= 4.5e-16 (CPU), 5.5e-16 (RTX 5090) times |untiled| +
# max|untiled| (JAX 0.12.0.1); without that floor, 3.6e-12 on entries that cancel.
# 1e-14 (45 eps) allows other summation orders and FMA; one node more or less
# moves each forward kernel by at least 1.7% of its largest entry.
_BLOCK_PARITY_RTOL = 1e-14

# 5 coils of 24 random nodes, every coordinate 0.5 to 1.5 in magnitude, so at
# least 0.2 per coordinate from the 11 points in [-0.3, 0.3]^3.
_RNG = np.random.default_rng(20261105)
_COIL_SHAPE = (5, 24, 3)
_GAMMAS = _RNG.uniform(0.5, 1.5, _COIL_SHAPE) * _RNG.choice((-1.0, 1.0), _COIL_SHAPE)
_GAMMADASHS = _RNG.standard_normal(_COIL_SHAPE)
_CURRENTS = _RNG.uniform(0.5, 1.5, 5) * 1e4
_BLOCK_POINTS = _RNG.uniform(-0.3, 0.3, (11, 3))
_BLOCK_COTANGENT = _RNG.standard_normal((11, 3))
_FORWARD_INPUTS = (_BLOCK_POINTS, _GAMMAS, _GAMMADASHS, _CURRENTS)
_VJP_INPUTS = (_BLOCK_POINTS, _BLOCK_COTANGENT, _GAMMAS, _GAMMADASHS, _CURRENTS)


def _forward_kernel(integrand, diff_mode, coil_chunk, quadrature_block, point_chunk=0):
    return core_biotsavart._make_kernel(
        core_biotsavart._Integrand[integrand],
        core_biotsavart._DiffMode[diff_mode],
        coil_chunk,
        quadrature_block,
        point_chunk,
    )


def _loop_trip_counts(jaxpr):
    """Sorted trip counts of the scans (static ``fori_loop``s) in ``jaxpr`` and its sub-jaxprs."""
    return tuple(
        sorted(
            [eqn.params["length"] for eqn in jaxpr.eqns if eqn.primitive.name == "scan"]
            + [
                count
                for eqn in jaxpr.eqns
                for sub_jaxpr in jaxprs_in_params(eqn.params)
                for count in _loop_trip_counts(sub_jaxpr)
            ]
        )
    )


# These return NumPy values, so a test row holds no JAX object when its case cleans up.
def _loops_and_result(kernel, inputs):
    """Trace ``kernel`` once; return its loop trip counts and its compiled result."""
    traced = kernel.trace(*inputs)
    result = traced.lower().compile()(*inputs)
    return _loop_trip_counts(traced.jaxpr.jaxpr), jax.tree.map(np.asarray, result)


def _B_vjp(coil_chunk_size, quadrature_block_size):
    kernel = core_biotsavart._make_B_vjp_kernel(coil_chunk_size, quadrature_block_size, 0)
    return jax.tree.map(np.asarray, kernel(*_VJP_INPUTS))


def _assert_results_match(tiled, untiled):
    """Elementwise rtol with an absolute floor at rtol times the largest value."""
    jax.tree.map(
        lambda tiled_leaf, untiled_leaf: np.testing.assert_allclose(
            tiled_leaf, untiled_leaf, rtol=_BLOCK_PARITY_RTOL, atol=_BLOCK_PARITY_RTOL * np.max(np.abs(untiled_leaf))
        ),
        tiled,
        untiled,
    )


def _coil_set(rng):
    """Two unit circles of 12 nodes, the second shifted, with random currents."""
    phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    circle = np.stack((np.cos(phi), np.sin(phi), np.zeros(12)), axis=-1)
    tangent = np.stack((-np.sin(phi), np.cos(phi), np.zeros(12)), axis=-1)
    gammas = np.stack([circle + center for center in ((0.0, 0.0, 0.0), (0.3, 0.1, -0.2))])
    currents = rng.uniform(0.5, 1.5, size=2)
    return (jnp.asarray(gammas), jnp.asarray(np.stack([tangent] * 2)), jnp.asarray(currents))


# Compiled form of the per-coil reductions. (coils, nodes, points): 5 coils is the
# only length-5 axis (24 nodes pad to 32, 12-node blocks to 16, 20 points in 7-point
# chunks). Tunings (coil chunk, quadrature block, point chunk): unblocked, and two
# quadrature blocks as in the fast lane. The fast-lane tiling (31 coils, 128 nodes
# in 64-node blocks, one 1024-point chunk) formed the A100 concatenate fusions.
_LAYOUT_COIL_COUNT = 5
_SIZES = (_LAYOUT_COIL_COUNT, 24, 20)
_TUNINGS = ((0, 0, 7), (0, 12, 7))
_FAST_LANE_SIZES = (31, 128, 1024)
_FAST_LANE_TUNING = (0, 64, 1024)

_FORWARD_DIFF_MODES = {"B": "VALUE", "dB_by_dX": "JACOBIAN"}


def _kernel_and_arguments(kernel_name, tuning, sizes):
    """Return one kernel and float64 shape structs of ``sizes`` for its inputs."""
    coils, nodes, point_count = sizes

    def spec(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float64)

    points, coil_data, currents = (spec(point_count, 3), spec(coils, nodes, 3), spec(coils))
    if kernel_name == "B_vjp":
        kernel = core_biotsavart._make_B_vjp_kernel(*tuning)
        return kernel, (points, points, coil_data, coil_data, currents)
    kernel = _forward_kernel("B", _FORWARD_DIFF_MODES[kernel_name], *tuning)
    return kernel, (points, coil_data, coil_data, currents)


def _per_coil_quadrature_sums(jaxpr, coil_count):
    """Yield (squeezed reduce_sum keeping the coil axis, its readers or None if returned)."""
    producers = {var: eqn for eqn in jaxpr.eqns for var in eqn.outvars}
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "squeeze" and coil_count in eqn.outvars[0].aval.shape:
            source = producers.get(eqn.invars[0])
            if source is not None and source.primitive.name == "reduce_sum":
                (total,) = eqn.outvars
                reads = [reader for reader in jaxpr.eqns if any(operand is total for operand in reader.invars)]
                returned = any(result is total for result in jaxpr.outvars)
                yield total, reads + [None] * returned
        for sub_jaxpr in jaxprs_in_params(eqn.params):
            yield from _per_coil_quadrature_sums(sub_jaxpr, coil_count)


def _is_contraction(eqn):
    """Whether ``eqn`` contracts a per-coil sum with a vector (currents, or its cotangent)."""
    if eqn is None or eqn.primitive.name != "dot_general":
        return False
    return sorted(len(var.aval.shape) for var in eqn.invars)[0] == 1


# Coil-DOF extraction maps.

_NQUAD = 8
_ORDER = 1
_SPLIT_CURVE_FIXED = ("yc(0)", "zc(1)")


def _bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def coil_graph():
    """Two coils: the first curve has two interior dofs fixed, the second current is fixed."""
    curves = [CurveXYZFourier(_NQUAD, _ORDER) for _ in range(2)]
    for index, curve in enumerate(curves):
        curve.x = 0.1 * (index + 1) * np.arange(1, int(curve.dof_size) + 1, dtype=np.float64)
    for name in _SPLIT_CURVE_FIXED:
        curves[0].fix(name)
    currents = [Current(1.0e4), Current(2.0e4)]
    currents[1].fix_all()
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    bs_jax = JaxBiotSavart(coils)
    oracle = BiotSavart(coils)
    assert list(oracle.dof_names) == list(bs_jax.dof_names)
    return curves, currents, bs_jax, oracle


def _owner_start(bs_jax, opt) -> int:
    return list(bs_jax.dof_names).index(f"{opt.name}:{opt.local_dof_names[0]}")


def _lowered_extraction_text(bs_jax) -> str:
    spec = bs_jax.coil_dof_extraction_spec()
    free_dofs = jnp.asarray(np.linspace(-1.0, 1.0, bs_jax.dof_size))
    lowered = jit(lambda dofs: coil_specs_from_dof_extraction_spec(spec, dofs))
    return lowered.lower(free_dofs).as_text()


_THREE_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1], [0.9, 0.3, -0.05]])
_OTHER_POINTS = np.array([[1.2, 0.05, 0.1], [0.7, -0.3, 0.0]])
_COTANGENT = np.arange(9, dtype=float).reshape(3, 3) / 11


def _spy(patches, owner, name):
    """Wrap ``owner.name`` in a call-recording mock for the rest of the case."""
    wrapped = getattr(owner, name)
    return patches.enter_context(mock.patch.object(owner, name, wraps=wrapped))


def _fields(coils, points=_THREE_POINTS):
    field, native = backend.JaxBiotSavart(coils), BiotSavart(coils)
    field.set_points(points)
    native.set_points(points)
    return field, native


def _assert_field_matches_native(field, native):
    """B, dB/dX and the B VJP equal native BiotSavart's at the live state."""
    _close(field.B(), native.B(), _TIGHT)
    _close(field.dB_by_dX(), native.dB_by_dX(), _TIGHT)
    cotangent = _COTANGENT[: native.B().shape[0]]
    expected = cast(Derivative, native.B_vjp(cotangent))
    _close(field.B_vjp(cotangent)(field), np.asarray(expected(native)), _VJP)


def _coils(*, perturbed=False):
    curves = [_perturbed(713, 2)[0]] if perturbed else _curves()
    currents = [Current(1e5) for _curve in curves]
    return curves, coils_via_symmetries(curves, currents, 2, True)


def _two_coils():
    """Two curves with currents 1e5 and -3e4, and their coils."""
    curves = _curves()
    currents = [Current(1e5), Current(-3e4)]
    return curves, currents, [Coil(c, i) for c, i in zip(curves, currents)]


def _perturbed(seed, count=1):
    """The first of ``count`` base curves, perturbed by a sample drawn with ``seed``, and the sample."""
    base = _curves(count)[0]
    sampler = GaussianSampler(base.quadpoints, 1e-3, 0.2, n_derivs=1)
    sample = PerturbationSample(sampler, randomgen=np.random.default_rng(seed))
    return CurvePerturbed(base, sample), sample


class _CountedSample(np.ndarray):
    copies = []

    def tobytes(self, order="C"):
        self.copies.append(self.nbytes)
        return super().tobytes(order)

    def copy(self, order="C"):
        self.copies.append(self.nbytes)
        return super().copy(order)


class TestBiotSavartJaxPhysics(JaxTestCase):
    """Field properties that hold independently of the native implementation."""

    def test_circular_loop_on_axis_field_is_analytic(self):
        """On a loop's axis B = (0, 0, mu0 I R^2 / (2 (R^2 + z^2)^1.5)), exact to round-off."""
        radius, current = 1.3, 2.0e5
        curve = CurveXYZFourier(64, 1)
        curve.x = np.array([radius if n in ("xc(1)", "ys(1)") else 0.0 for n in curve.local_dof_names])
        z = np.array([-2.0, -0.4, 0.0, 0.25, 3.0])
        points = np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=1)
        field = JaxBiotSavart([Coil(curve, Current(current))]).set_points(points)
        mu0 = 4.0e-7 * np.pi
        expected = np.zeros_like(points)
        expected[:, 2] = mu0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
        _close(field.B(), expected, (1e-14, 1e-14 * expected[:, 2].max()))

    def test_field_is_divergence_free(self):
        """The trace of dB/dX vanishes to round-off at each point around the NCSX axis."""
        bs, points_np, *_ = _ncsx()
        field = JaxBiotSavart(list(bs._coils)).set_points(points_np)
        dB = np.asarray(field.dB_by_dX())
        divergence = np.trace(dB, axis1=1, axis2=2)
        self.assertLess(np.max(np.abs(divergence) / np.max(np.abs(dB), axis=(1, 2))), 1e-13)


class TestBiotSavartJaxNativeParity(JaxTestCase):
    """Kernels and public accessors against native BiotSavart on the NCSX coils."""

    def test_kernel_parity_ncsx(self):
        """Raw JAX kernels match native BiotSavart on the NCSX parity fixture."""
        tols = {"B": _DIRECT, "A": _DIRECT, "dB_by_dX": _FIRST, "dA_by_dX": _FIRST}

        def check(quantity):
            bs, *arrays = _ncsx()
            expected = getattr(bs, quantity)()
            kernel = getattr(core_biotsavart, "biot_savart_" + quantity)
            _close(kernel(*(jnp.array(array) for array in arrays)), expected, tols[quantity])

        self.run_cases(check, _grid(quantity=tols))

    def test_cylindrical_public_accessors_parity_ncsx(self):
        """JaxBiotSavart's point setters chain, and its cylindrical accessors (B_cyl, A_cyl and
        GradAbsB_cyl on the cached cylindrical phi) match native."""
        bs, points_np, *_ = _ncsx()
        points_cyl = _cart_points_to_cyl(points_np)
        points_cyl[:, 1] += 2.0 * np.pi
        cached_phi = np.array([[0.0, 1.234, 0.1], [0.0, -1.5, -0.2], [0.7, 2.0 * np.pi + 0.4, 0.3]])
        bs_jax = JaxBiotSavart(list(bs._coils))
        self.assertIs(bs_jax.set_points_cyl(points_cyl), bs_jax)
        self.assertIs(bs_jax.set_points_cart(points_np), bs_jax)
        for name, points in (("ncsx", points_cyl), ("cached_phi", cached_phi)):
            with self.subTest(points=name):
                bs.set_points_cyl(points)
                self.assertIs(bs_jax.set_points_cyl(points), bs_jax)
                _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cartesian_public_accessors_normalize_cylindrical_phi_ncsx(self):
        """Cartesian point conversion follows C++ ``get_points_cyl_impl``."""
        bs, *_ = _ncsx()
        points_cart = np.array([[0.5, -0.5, 0.1], [-0.5, -0.25, -0.2], [0.0, 0.0, 0.3]])
        bs.set_points_cart(points_cart)
        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points_cart(points_cart)
        _assert_cylindrical_points_match_cpu(bs_jax, bs)

    def test_nonfinite_point_leaves_other_cylindrical_angles_native(self):
        """A nonfinite point leaves finite points' cylindrical angles native (wrap and fmod)."""
        bs, *_ = _ncsx()
        bs_jax = JaxBiotSavart(list(bs._coils))
        for setter, points in (
            ("set_points_cart", [[1.0, -1.0, 0.0], [np.inf, 0.0, 0.0]]),
            ("set_points_cyl", [[1.0, 7.0, 0.0], [1.0, np.nan, 0.0]]),
        ):
            with self.subTest(setter=setter):
                points = np.ascontiguousarray(points, dtype=np.float64)
                getattr(bs, setter)(points)
                getattr(bs_jax, setter)(points)
                expected = bs.get_points_cyl()[0]
                np.testing.assert_array_equal(bs_jax.get_points_cyl()[0], expected)

    def test_default_evaluation_point_is_origin_like_native(self):
        """Without set_points the field is evaluated at one point at the origin."""
        bs, *_ = _ncsx()
        native = BiotSavart(list(bs._coils))
        bs_jax = JaxBiotSavart(list(bs._coils))
        np.testing.assert_array_equal(bs_jax.get_points_cart(), np.zeros((1, 3)))
        _close(bs_jax.B(), native.B(), _DIRECT)

    def test_empty_coil_set_GradAbsB_is_nan_like_native(self):
        """With no coils |B| is identically zero and its gradient is native's 0/0."""
        _, points_np, *_ = _ncsx()
        native = BiotSavart([]).set_points(points_np)
        bs_jax = JaxBiotSavart([]).set_points(points_np)
        np.testing.assert_array_equal(np.asarray(bs_jax.B()), native.B())
        self.assertTrue(np.all(np.isnan(native.GradAbsB())))
        np.testing.assert_array_equal(np.asarray(bs_jax.GradAbsB()), native.GradAbsB())
        # The NaN is built on device, so it also holds under the strict guard.
        bs_jax.set_points(points_np)
        with disallow_host_transfers():
            grad_abs_B = bs_jax.GradAbsB()
        np.testing.assert_array_equal(np.asarray(grad_abs_B), native.GradAbsB())

    def test_per_current_derivatives_run_under_strict_transfer_guard(self):
        """Per-coil unit-current accessors and their (coil-chunked) builder run under the strict guard."""
        rows = (
            ("B", "dB_by_dcoilcurrents", _DIRECT),
            ("dB_by_dX", "d2B_by_dXdcoilcurrents", _FIRST),
            ("d2B_by_dXdX", "d3B_by_dXdXdcoilcurrents", _SECOND),
            ("A", "dA_by_dcoilcurrents", _DIRECT),
            ("dA_by_dX", "d2A_by_dXdcoilcurrents", _FIRST),
            ("d2A_by_dXdX", "d3A_by_dXdXdcoilcurrents", _SECOND),
        )

        def check(chunk_size):
            """None calls the public accessors; an integer, their builder at that batch size."""
            bs, bs_jax, _ = _ncsx_fields()
            points, coil_set_spec = bs_jax.get_points_cart_ref(), bs_jax.coil_set_spec()
            for cache_method, list_method, tol in rows:
                kernel = getattr(core_biotsavart, "biot_savart_" + cache_method)
                getattr(bs, cache_method)()
                with disallow_host_transfers():
                    if chunk_size is None:
                        jax_list = getattr(bs_jax, list_method)()
                    else:
                        jax_list = _per_coil_unit_field_with_batch_size(
                            points, coil_set_spec, kernel, batch_size=chunk_size
                        )
                cpu_list = getattr(bs, list_method)()
                self.assertEqual(len(jax_list), len(cpu_list))
                for k, (j_entry, c_entry) in enumerate(zip(jax_list, cpu_list)):
                    _close(j_entry, c_entry, tol, f"{list_method} coil {k}")

        self.run_cases(check, _grid(chunk_size=(None, 0, 7)))

    def test_B_vjp_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        bs_cpu, bs_jax, _ = _ncsx_fields()
        v = np.asarray(bs_cpu.B(), dtype=np.float64).copy()
        deriv_cpu = cast(Derivative, bs_cpu.B_vjp(v))
        deriv_jax = bs_jax.B_vjp(v)
        for coil in bs_cpu._coils:
            _close(deriv_jax(coil), np.asarray(deriv_cpu(coil)), _FIRST, "B_vjp differs from native")

    def test_coil_chunked_kernels_and_pullback_match_native_ncsx(self):
        """Coil chunks of 10+8 (two-chunk path) and 7+7+4 (padded loop) keep B, dB/dX, B pullback native."""

        def check(chunk_size):
            """Kernels from the factories; the chunked pullback against the public one."""
            bs, points_np, gammas_np, gds_np, currents_np = _ncsx()
            self.assertEqual(len(currents_np), 18)
            B_kernel, dB_kernel = (
                _forward_kernel("B", diff_mode, chunk_size, 0) for diff_mode in ("VALUE", "JACOBIAN")
            )
            _close(B_kernel(points_np, gammas_np, gds_np, currents_np), bs.B(), _DIRECT)
            _close(dB_kernel(points_np, gammas_np, gds_np, currents_np), bs.dB_by_dX(), _FIRST)
            v = np.asarray(bs.B(), dtype=np.float64).copy()
            arrays = (points_np, v, gammas_np, gds_np, currents_np)
            chunked = core_biotsavart._make_B_vjp_kernel(chunk_size, 0, 0)(*arrays)
            public = core_biotsavart.biot_savart_B_vjp(*arrays)
            for actual, expected in zip(chunked, public, strict=True):
                _close(actual, np.asarray(expected), _FIRST)

        self.run_cases(check, _grid(chunk_size=(10, 7)))

    def test_adapter_derivative_parity_ncsx(self):
        """JaxBiotSavart spatial derivatives match native BiotSavart on the NCSX fixture."""
        tols = {"dA_by_dX": _FIRST, "d2B_by_dXdX": _SECOND, "d2A_by_dXdX": _SECOND}

        def check(quantity):
            bs, bs_jax, _ = _ncsx_fields()
            expected = getattr(bs, quantity)()
            _close(getattr(bs_jax, quantity)(), expected, tols[quantity])

        self.run_cases(check, _grid(quantity=tols))

    def test_GradAbsB_taylor_ncsx(self):
        """GradAbsB is the second-order limit of central differences of AbsB."""
        _, bs_jax, points_np = _ncsx_fields()
        grad = np.asarray(bs_jax.GradAbsB())
        for direction in np.eye(3):
            with self.subTest(direction=direction):
                deriv = grad @ direction
                err = 1e6
                for i in range(5):
                    eps = 1e-3 * 0.5**i
                    bs_jax.set_points(points_np + eps * direction)
                    plus = np.asarray(bs_jax.AbsB())[:, 0]
                    bs_jax.set_points(points_np - eps * direction)
                    minus = np.asarray(bs_jax.AbsB())[:, 0]
                    new_err = np.linalg.norm((plus - minus) / (2 * eps) - deriv)
                    self.assertLess(new_err, 0.3 * err)
                    err = new_err

    def test_per_coil_unit_field_vectorizes_within_quadrature_group(self):
        """Unit-current fields vectorize per quadrature group and keep public coil order."""
        points = jnp.asarray([[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]], dtype=jnp.float64)
        gammas0 = jnp.arange(18, dtype=jnp.float64).reshape(2, 3, 3)
        gammas1 = jnp.arange(9, dtype=jnp.float64).reshape(1, 3, 3) - 2.0
        gammadashs0, gammadashs1 = gammas0 + 0.5, gammas1 - 0.25
        currents0 = jnp.asarray([3.0, 4.0], dtype=jnp.float64)
        currents1 = jnp.asarray([5.0], dtype=jnp.float64)
        coil_set_spec = GroupedCoilSetSpec(
            groups=(
                CoilGroupSpec(gammas0, gammadashs0, currents0, coil_indices=(2, 0)),
                CoilGroupSpec(gammas1, gammadashs1, currents1, coil_indices=(1,)),
            )
        )
        calls = []

        def kernel(kernel_points, gammas, gammadashs, currents):
            calls.append(gammas.shape)
            value = jnp.sum(gammas) + jnp.sum(gammadashs) + jnp.sum(currents)
            return jnp.broadcast_to(value, kernel_points.shape)

        results = _per_coil_unit_field(points, coil_set_spec, kernel)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls, [(1, 3, 3), (1, 3, 3)])
        self.assertEqual(len(results), 3)
        for result, gammas, gammadashs in zip(
            results, (gammas0[1], gammas1[0], gammas0[0]), (gammadashs0[1], gammadashs1[0], gammadashs0[0]), strict=True
        ):
            expected = np.broadcast_to(np.sum(np.asarray(gammas)) + np.sum(np.asarray(gammadashs)) + 1.0, points.shape)
            np.testing.assert_allclose(np.asarray(result), expected)


class TestBiotSavartJaxSpatialTaylor(JaxTestCase):
    """Shrinking-step (0.5**5..9) Taylor tests of spatial derivatives, as native test_biotsavart.py."""

    def _assert_spatial_taylor(self, value_name, derivative_name, *, central, ratio):
        """Assert ``derivative[:, k]`` is the shrinking-step limit along ``e_k``."""
        coil, points_np = _native_taylor_fixture()
        bs_jax = JaxBiotSavart([coil])
        bs_jax.set_points(points_np)
        value0 = np.asarray(getattr(bs_jax, value_name)())
        derivative = np.asarray(getattr(bs_jax, derivative_name)())
        for k, direction in enumerate(np.eye(3)):
            with self.subTest(direction=k):
                expected = derivative[:, k]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i
                    bs_jax.set_points(points_np + eps * direction)
                    plus = np.asarray(getattr(bs_jax, value_name)())
                    if central:
                        bs_jax.set_points(points_np - eps * direction)
                        minus = np.asarray(getattr(bs_jax, value_name)())
                        estimate = (plus - minus) / (2.0 * eps)
                    else:
                        estimate = (plus - value0) / eps
                    new_err = np.linalg.norm(estimate - expected)
                    self.assertLess(new_err, ratio * err)
                    err = new_err

    def test_dB_by_dX_taylor(self):
        """dB_by_dX is the first-order shrinking-step limit of B along each axis."""
        self._assert_spatial_taylor("B", "dB_by_dX", central=False, ratio=0.55)

    def test_d2B_by_dXdX_taylor(self):
        """d2B_by_dXdX is the central shrinking-step limit of dB_by_dX along each axis."""
        self._assert_spatial_taylor("dB_by_dX", "d2B_by_dXdX", central=True, ratio=0.30)

    def test_dA_by_dX_taylor(self):
        """dA_by_dX is the first-order shrinking-step limit of A along each axis."""
        self._assert_spatial_taylor("A", "dA_by_dX", central=False, ratio=0.55)

    def test_d2A_by_dXdX_taylor(self):
        """d2A_by_dXdX is the central shrinking-step limit of dA_by_dX along each axis."""
        self._assert_spatial_taylor("dA_by_dX", "d2A_by_dXdX", central=True, ratio=0.30)


class TestBiotSavartJaxCurveAndCurrentTypes(JaxTestCase):
    """Framed, filament, perturbed, symmetric curves, affine currents, fixed owners, float32."""

    def test_filament_gammadash_is_quadpoint_derivative(self):
        """JAX filament gammadash is d(gamma)/d(quadpoint) and matches native for both frames."""

        def check(frame):
            frame = _FRAMES[frame]
            quadpoints = np.linspace(0.0, 1.0, 24, endpoint=False)
            gamma, gammadash = _jax_filament_geometry(frame, quadpoints)
            error = 1e6
            for exponent in range(5):
                step = 1e-2 * 0.5**exponent
                plus, _ = _jax_filament_geometry(frame, quadpoints + step)
                minus, _ = _jax_filament_geometry(frame, quadpoints - step)
                next_error = np.linalg.norm((plus - minus) / (2 * step) - gammadash)
                self.assertLess(next_error, 0.3 * error)
                error = next_error
            native = _filament(frame, quadpoints)
            _close(gamma, native.gamma(), _TIGHT)
            _close(gammadash, native.gammadash(), _TIGHT)

        self.run_cases(check, _grid(frame=_FRAMES))

    def test_affine_current_vjps_match_native_and_finite_difference(self):
        """Affine-current fields, owner partials and flat gradients match native and FD."""

        def check(scale, quantity):
            coils, first = _affine_coils(scale)
            adapter, native = _fields(coils, _POINTS)
            cotangent = _cotangent(quantity)
            _close(getattr(adapter, quantity)(), getattr(native, quantity)(), _TIGHT)
            expected = _gradient(_vjp(native, quantity, cotangent), native)
            actual = _gradient(_vjp(adapter, quantity, cotangent), adapter)
            value = cast(np.ndarray, first.local_full_x).copy()
            step = 1e-3
            first.local_full_x = value + step
            plus = np.sum(getattr(native, quantity)() * cotangent)
            first.local_full_x = value - step
            minus = np.sum(getattr(native, quantity)() * cotangent)
            first.local_full_x = value
            finite_difference = (plus - minus) / (2 * step)
            pullback = getattr(adapter, quantity + "_pullback_native")(cotangent)
            flat = adapter.coil_cotangents_to_dofs_gradient(pullback.d_coil_arrays, pullback.coil_indices)
            _close(actual, expected, _TIGHT)
            _close(flat, expected, _TIGHT)
            _close(actual, finite_difference, (1e-9, 1e-14))
            native_partials = _partials(_vjp(native, quantity, cotangent), native)
            adapter_partials = _partials(_vjp(adapter, quantity, cotangent), adapter)
            _assert_partials_match(adapter_partials, native_partials)

        self.run_cases(check, _grid(scale=(1.0, -2.5), quantity=_FIELDS))

    def test_affine_current_vjps_shrinking_steps(self):
        """Affine-current VJPs retain linearity through five halved central steps."""
        for row in _grid(scale=(1.0, -2.5), quantity=_FIELDS):
            with self.subTest(**row), self.case() as patches:
                self._case_affine_current_vjps_shrinking_steps(patches, **row)

    def _case_affine_current_vjps_shrinking_steps(self, patches, scale, quantity):
        adapter = JaxBiotSavart(_affine_coils(scale)[0]).set_points(_POINTS)
        cotangent = _cotangent(quantity)
        dofs = np.array(adapter.x, dtype=np.float64)
        patches.callback(setattr, adapter, "x", dofs)
        analytic = _gradient(_vjp(adapter, quantity, cotangent), adapter)
        for exponent in range(5):
            step = 1e-3 * 0.5**exponent
            adapter.x = dofs + step
            plus = np.sum(getattr(adapter, quantity)() * cotangent)
            adapter.x = dofs - step
            minus = np.sum(getattr(adapter, quantity)() * cotangent)
            _close(analytic, (plus - minus) / (2 * step), (1e-9, 1e-14))

    def test_derivative_retains_fixed_partials(self):
        """Field VJPs keep native partials of fixed owners and the native free gradient."""

        def check(fixed, quantity):
            curve = _curves()[0]
            current = Current(1e5)
            curve.fix(0)
            current.fix_all()
            if fixed != "partial":
                curve.fix_all()
            other = _curves()[1]
            if fixed == "all":
                other.fix_all()
            coils = [Coil(curve, current), Coil(other, Current(2e4))]
            if fixed == "all":
                coils[1].current.fix_all()
            adapter, native = _fields(coils, _POINTS)
            cotangent = _cotangent(quantity)
            expected = _partials(_vjp(native, quantity, cotangent), native)
            derivative = _vjp(adapter, quantity, cotangent)
            actual = _partials(derivative, adapter)
            _assert_partials_match(actual, expected)
            gradient = _gradient(derivative, adapter)
            expected_gradient = _gradient(_vjp(native, quantity, cotangent), native)
            _close(gradient, expected_gradient, _VJP)

        self.run_cases(check, _grid(fixed=("partial", "curve", "all"), quantity=_FIELDS))

    def test_native_symmetry_curve_converts_for_fields_and_vjps(self):
        """Symmetric Fourier curves give native fields and VJPs with and without stellsym."""

        def check(stellsym):
            curve = CurveXYZFourierSymmetries(64, 2, nfp=3, stellsym=stellsym, ntor=2)
            rng = np.random.default_rng(1729)
            dofs = rng.normal(0.0, 0.03, curve.local_full_dof_size)
            dofs[0] = 1.0
            curve.local_full_x = dofs
            coils = [Coil(curve, Current(1e5))]
            adapter, native = _fields(coils, _POINTS)
            for quantity in _FIELDS:
                actual = np.asarray(getattr(adapter, quantity)())
                _close(actual, getattr(native, quantity)(), _VJP)
                cotangent = _cotangent(quantity)
                gradient = _gradient(_vjp(adapter, quantity, cotangent), adapter)
                expected = _gradient(_vjp(native, quantity, cotangent), native)
                _close(gradient, expected, _VJP)

        self.run_cases(check, _grid(stellsym=(False, True)))

    def test_composed_curves_match_native_fields_owner_vjps_and_finite_difference(self):
        """Nested perturbed and filament curves match native fields, owner partials and FD."""

        def check(wrapper, fixed, quantity):
            native, curve, rotation = _composed_native_field(wrapper, fixed)
            expected_field = getattr(native, quantity)().copy()
            cotangent = _cotangent(quantity)
            expected_vjp = _vjp(native, quantity, cotangent)
            expected_gradient = _gradient(expected_vjp, native)
            expected_partials = _partials(expected_vjp, native)
            adapter = JaxBiotSavart(native.coils).set_points(_POINTS)
            _close(getattr(adapter, quantity)(), expected_field, _TIGHT)
            actual_vjp = _vjp(adapter, quantity, cotangent)
            actual_gradient = _gradient(actual_vjp, adapter)
            actual_partials = _partials(actual_vjp, adapter)
            _close(actual_gradient, expected_gradient, _VJP)
            self.assertIn(curve, actual_partials.data)
            if wrapper != "nested_perturbed":
                self.assertIn(rotation, actual_partials.data)
            _assert_partials_match(actual_partials, expected_partials)
            pullback = getattr(adapter, quantity + "_pullback_native")(cotangent)
            flat = adapter.coil_cotangents_to_dofs_gradient(pullback.d_coil_arrays, pullback.coil_indices)
            _close(flat, expected_gradient, _VJP)
            if quantity in ("A", "B"):
                # Both halves of the combined VJP, with both cotangents nonzero, match
                # native; the value half ignores the gradient cotangent.
                combined_vjp = f"{quantity}_and_d{quantity}_vjp"
                vgrad = _cotangent(f"d{quantity}_by_dX")
                getattr(native, f"d{quantity}_by_dX")()
                native_halves = getattr(native, combined_vjp)(cotangent, vgrad)
                adapter_halves = getattr(adapter, combined_vjp)(cotangent, vgrad)
                for half, (native_half, adapter_half) in enumerate(zip(native_halves, adapter_halves, strict=True)):
                    combined_native = _partials(native_half, native)
                    combined_adapter = _partials(adapter_half, adapter)
                    for owner in combined_native.data:
                        actual = combined_adapter.data[owner]
                        _close(actual, combined_native.data[owner], _VJP, f"{combined_vjp} half {half}")
                        if half == 0:
                            _close(actual, actual_partials.data[owner], _VJP)

            dofs = native.x.copy()
            direction = np.linspace(-0.3, 0.4, dofs.size)
            step = 1e-4
            objectives = []
            for multiple in (2, 1, -1, -2):
                native.x = dofs + multiple * step * direction
                perturbed_field = getattr(native, quantity)()
                _close(getattr(adapter, quantity)(), perturbed_field, _TIGHT)
                objectives.append(np.sum(perturbed_field * cotangent))
            native.x = dofs
            finite_difference = (-objectives[0] + 8 * objectives[1] - 8 * objectives[2] + objectives[3]) / (12 * step)
            _close(actual_gradient @ direction, finite_difference, (1e-9, 1e-14))

        rows = _grid(wrapper=_COMPOSED_WRAPPERS, fixed=(False, True), quantity=_FIELDS)
        self.run_cases(check, rows)

    def test_raw_kernels_evaluate_single_precision_inputs_in_float64(self):
        """Raw kernels and the B pullback upcast exact float32 inputs, bitwise as float64."""
        coils = [Coil(curve, Current(1e5)) for curve in _curves()]
        single = (
            _POINTS.astype(np.float32),
            _cotangent("B").astype(np.float32),
            np.stack([coil.curve.gamma() for coil in coils]).astype(np.float32),
            np.stack([coil.curve.gammadash() for coil in coils]).astype(np.float32),
            np.array([coil.current.get_value() for coil in coils], dtype=np.float32),
        )
        double = tuple(value.astype(np.float64) for value in single)

        def evaluate(name, points, cotangent, *coil_data):
            if name == "biot_savart_B_vjp":
                return core_biotsavart.biot_savart_B_vjp(points, cotangent, *coil_data)
            return getattr(core_biotsavart, name)(points, *coil_data)

        for name in (
            "biot_savart_B",
            "biot_savart_A",
            "biot_savart_dB_by_dX",
            "biot_savart_B_and_dB",
            "biot_savart_B_vjp",
        ):
            with self.subTest(kernel=name):
                for actual, expected in zip(
                    jax.tree.leaves(evaluate(name, *single)), jax.tree.leaves(evaluate(name, *double)), strict=True
                ):
                    self.assertEqual(actual.dtype, np.float64)
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_grouped_field_evaluates_single_precision_points_in_float64(self):
        """Grouped B and its pullback evaluate float32 points and cotangents in float64."""
        coils = [Coil(curve, Current(1e5)) for curve in _curves()]
        group = (
            jnp.asarray(np.stack([coil.curve.gamma() for coil in coils])),
            jnp.asarray(np.stack([coil.curve.gammadash() for coil in coils])),
            jnp.asarray([coil.current.get_value() for coil in coils]),
        )
        points = _POINTS.astype(np.float32)
        cotangent = _cotangent("B").astype(np.float32)
        native = BiotSavart(coils).set_points(points.astype(np.float64))
        field = cast(jax.Array, grouped_biot_savart_B_from_spec(points, grouped_coil_set_spec_from_inputs([group])))
        self.assertEqual(field.dtype, np.float64, "B must be evaluated in float64")
        _close(field, native.B(), _TIGHT, "float32 points must not lower the coil geometry to float32")
        pullbacks = group_biot_savart_B_vjp(points, cotangent, *group)
        double_pullbacks = group_biot_savart_B_vjp(points.astype(np.float64), cotangent.astype(np.float64), *group)
        for pullback, double_pullback in zip(pullbacks, double_pullbacks, strict=True):
            self.assertEqual(pullback.dtype, np.float64, "pullbacks must be float64")
            np.testing.assert_array_equal(np.asarray(pullback), np.asarray(double_pullback))


class TestBiotSavartJaxKernelTiling(JaxTestCase):
    """Tiled and fused kernels against untiled ones and autodiff, and their compiled form."""

    def test_fused_B_and_dB_matches_autodiff_of_B(self):
        """The fused B + dB/dX integrand and tiled kernel equal B and its jacfwd to round-off."""
        # Integrand errors relative to |gammadash| / r**2 (B) and / r**3 (dB); kernel
        # errors relative to the largest entry at each point.
        _bs, axis_points, gammas_np, gds_np, currents_np = _ncsx()
        gammas = jnp.asarray(gammas_np)
        gammadashs = jnp.asarray(gds_np)
        currents = jnp.asarray(currents_np)
        tangent = gds_np[0, 0] / np.linalg.norm(gds_np[0, 0])
        offset = np.cross(tangent, [0.0, 0.0, 1.0])
        offset /= np.linalg.norm(offset)
        configurations = {
            "around the axis": axis_points,
            "near a coil": gammas_np[0, 0] + np.outer([1e-3, 3e-3, 1e-2], offset),
            "far away": np.array([[10.0, 3.0, -2.0], [-7.0, 8.0, 5.0]]),
        }

        def B_integrand(x):
            integrand = core_biotsavart._biot_savart_B_integrand(x, gammas, gammadashs)
            return jnp.stack(integrand, axis=-1)

        def autodiff_integrand(x):
            # jacfwd gives [..., component, direction]; the fused integrand puts
            # the direction first.
            return B_integrand(x), jnp.swapaxes(jax.jacfwd(B_integrand)(x), -1, -2)

        def fused_integrand(x):
            return core_biotsavart._biot_savart_B_and_dB_integrand(x, gammas, gammadashs)

        def worst(actual, reference, axis):
            error = np.max(np.abs(np.asarray(actual) - reference), axis=axis)
            return np.max(error / np.max(np.abs(reference), axis=axis))

        for name, points_np in configurations.items():
            with self.subTest(configuration=name):
                points = jnp.asarray(points_np)
                B_fused, dB_fused = jax.vmap(fused_integrand)(points)
                B_auto, dB_auto = jax.vmap(autodiff_integrand)(points)
                distance = np.linalg.norm(points_np[:, None, None, :] - gammas_np[None], axis=-1)
                B_scale = np.linalg.norm(gds_np, axis=-1)[None] / distance**2
                B_error = np.max(np.abs(np.asarray(B_fused - B_auto)), axis=-1)
                dB_error = np.max(np.abs(np.asarray(dB_fused - dB_auto)), axis=(-2, -1))
                self.assertLess(np.max(B_error / B_scale), _ROUND_OFF)
                self.assertLess(np.max(dB_error * distance / B_scale), _ROUND_OFF)

                inputs = (points, gammas, gammadashs, currents)
                B_kernel, dB_kernel = core_biotsavart.biot_savart_B_and_dB(*inputs)
                B_reference = np.asarray(core_biotsavart.biot_savart_B(*inputs))
                dB_reference = np.asarray(core_biotsavart.biot_savart_dB_by_dX(*inputs))
                self.assertLess(worst(B_kernel, B_reference, -1), _ROUND_OFF)
                self.assertLess(worst(dB_kernel, dB_reference, (-2, -1)), _ROUND_OFF)

    def test_forward_kernels_take_their_paths_and_match_untiled(self):
        """B, A, their Jacobians and Hessians and fused B/dB run each tiling's loops, match untiled."""
        for integrand, diff_mode in _FORWARD_KERNELS:
            kernel = {"integrand": integrand, "diff_mode": diff_mode}
            # The reference is computed outside a subTest, so no row compares
            # against another kernel's reference if computing it fails.
            with self.case():
                loops, untiled = _loops_and_result(_forward_kernel(integrand, diff_mode, 0, 0), _FORWARD_INPUTS)
            with self.subTest(**kernel, tiling="untiled"):
                self.assertEqual(loops, ())
            for name, coil_chunk_size, quadrature_block_size, trip_counts in _TILINGS:
                with self.subTest(**kernel, tiling=name), self.case():
                    loops, tiled = _loops_and_result(
                        _forward_kernel(integrand, diff_mode, coil_chunk_size, quadrature_block_size), _FORWARD_INPUTS
                    )
                    self.assertEqual(loops, trip_counts)
                    _assert_results_match(tiled, untiled)

    def test_B_vjp_matches_untiled_across_block_boundaries(self):
        """B cotangents for positions, tangents and currents match untiled for every tiling."""
        with self.case():
            untiled = _B_vjp(0, 0)
        for name, coil_chunk_size, quadrature_block_size, _ in _TILINGS:
            with self.subTest(tiling=name), self.case():
                tiled = _B_vjp(coil_chunk_size, quadrature_block_size)
                _assert_results_match(tiled, untiled)

    def test_hessian_point_chunk_size_is_one_eighth_and_keeps_disabled(self):
        """Hessian tiles are one eighth of a positive point tuning, at least one; zero disables."""

        def check(tuned, expected):
            self.assertEqual(core_biotsavart.hessian_point_chunk_size(tuned), expected)

        pairs = ((0, 0), (1, 1), (7, 1), (8, 1), (128, 16), (2048, 256), (4096, 512))
        self.run_cases(check, [dict(tuned=tuned, expected=size) for tuned, size in pairs])

    def test_d2B_kernel_tiled_over_points_matches_untiled(self):
        """Point-tiled Hessians match dense ones; value/Jacobian kernels keep their tile size."""
        rng = np.random.default_rng(7)
        coil_set = _coil_set(rng)
        points = jnp.asarray(rng.uniform(-0.4, 0.4, size=(37, 3)))
        untiled = _forward_kernel("B", "HESSIAN", 0, 0, 0)(points, *coil_set)
        # A tuned size of 64 becomes an 8-point Hessian tile: five tiles of 8 plus a
        # 5-point tail, so both loop bodies and the tail are exercised.
        tiled = _forward_kernel("B", "HESSIAN", 0, 0, 64)(points, *coil_set)
        self.assertEqual(tiled.shape, (37, 3, 3, 3))
        _close(tiled, np.asarray(untiled), (1e-13, 1e-16))
        b_and_db_tiled = _forward_kernel("B", "VALUE_AND_JACOBIAN", 0, 0, 64)(points, *coil_set)
        # The non-Hessian kernels keep the tuned size: 37 points fit one 64-point
        # tile, so their result must equal the untiled evaluation exactly.
        b_and_db_untiled = _forward_kernel("B", "VALUE_AND_JACOBIAN", 0, 0, 0)(points, *coil_set)
        for tiled_leaf, untiled_leaf in zip(b_and_db_tiled, b_and_db_untiled, strict=True):
            np.testing.assert_array_equal(np.asarray(tiled_leaf), np.asarray(untiled_leaf))

    def test_per_coil_quadrature_sums_feed_only_their_contraction(self):
        """Each per-coil quadrature sum of traced B, dB/dX and B-VJP feeds only its contraction."""

        def check(kernel, sizes, tuning):
            coils, nodes, _ = sizes
            traced_kernel, arguments = _kernel_and_arguments(kernel, tuning, sizes)
            # disable_jit inlines jit calls and unrolls fori_loops, so each sum and
            # every equation that reads it, in the VJP too, are in one jaxpr.
            with jax.disable_jit():
                traced = jax.make_jaxpr(traced_kernel)(*arguments)
            sums = list(_per_coil_quadrature_sums(traced.jaxpr, coils))
            # One sum per component and quadrature block for each traced point
            # chunk and derivative pass; none found would make the guard vacuous.
            block_size = tuning[1]
            blocks = 1 if block_size <= 0 else (nodes + block_size - 1) // block_size
            self.assertTrue(
                sums and len(sums) % (3 * blocks) == 0,
                f"{len(sums)} per-coil quadrature sums for 3 components x {blocks} blocks",
            )
            offending = [
                f"{total.aval.shape} -> " + ", ".join("result" if eqn is None else eqn.primitive.name for eqn in reads)
                for total, reads in sums
                if not reads or not all(map(_is_contraction, reads))
            ]
            self.assertEqual(offending, [], "per-coil quadrature sums used before their contraction")

        cases = ((_SIZES, _TUNINGS[0]), (_SIZES, _TUNINGS[1]), (_FAST_LANE_SIZES, _FAST_LANE_TUNING))
        rows = [
            dict(kernel=kernel, sizes=sizes, tuning=tuning)
            for kernel in ("B", "dB_by_dX", "B_vjp")
            for sizes, tuning in cases
        ]
        self.run_cases(check, rows)


class TestBiotSavartJaxCoilDofMaps(JaxTestCase):
    """Coil-DOF maps keep segments and input order; pullbacks keep leaf order and indices."""

    def setUp(self):
        super().setUp()
        self.coil_graph = coil_graph()
        self.addCleanup(delattr, self, "coil_graph")

    def test_segments_from_free_positions_merge_runs_and_split_at_fixed_dofs(self):
        """Segments merge contiguous free positions, split at fixed DOFs, and allow none."""
        segments = _owner_segments_from_free_positions
        self.assertEqual(segments(5, [0, 1, 2]), ((5, 8, 0, 3),))
        self.assertEqual(segments(2, [0, 1, 2, 4, 5, 6, 7]), ((2, 5, 0, 3), (5, 9, 4, 8)))
        self.assertEqual(segments(0, [1, 3, 5]), ((0, 1, 1, 2), (1, 2, 3, 4), (2, 3, 5, 6)))
        self.assertEqual(segments(7, []), ())

    def test_extraction_spec_carries_one_segment_per_run(self):
        """Curve and current maps hold one copy segment per contiguous free-DOF run."""
        curves, currents, bs_jax, _oracle = self.coil_graph
        spec = bs_jax.coil_dof_extraction_spec()
        split = _owner_start(bs_jax, curves[0])
        self.assertEqual(
            spec.coils[0].curve_map.owner_segments, ((split, split + 3, 0, 3), (split + 3, split + 7, 4, 8))
        )
        intact = _owner_start(bs_jax, curves[1])
        self.assertEqual(spec.coils[1].curve_map.owner_segments, ((intact, intact + 9, 0, 9),))
        current = _owner_start(bs_jax, currents[0])
        self.assertEqual(spec.coils[0].current_map.owner_segments, ((current, current + 1, 0, 1),))
        self.assertEqual(spec.coils[1].current_map.owner_segments, ())

    def test_coil_specs_from_dofs_match_the_optimizable_setter_bitwise(self):
        """Compiled reconstruction matches native DOF setters bitwise under the strict guard."""
        curves, currents, bs_jax, oracle = self.coil_graph
        spec = bs_jax.coil_dof_extraction_spec()
        free_dofs = np.random.default_rng(20260914).normal(size=bs_jax.dof_size)
        oracle.x = free_dofs
        expected_curve_dofs = [np.asarray(curve.local_full_x).copy() for curve in curves]
        expected_currents = [float(current.get_value()) for current in currents]
        # The guard refuses implicit transfers, so the DOFs are placed before it.
        device_dofs = jnp.asarray(free_dofs)
        compiled = jit(lambda dofs: coil_specs_from_dof_extraction_spec(spec, dofs)).lower(device_dofs).compile()
        with jax.transfer_guard("disallow"):
            coil_specs = compiled(device_dofs)
        for coil_spec, expected_dofs, expected_current in zip(coil_specs, expected_curve_dofs, expected_currents):
            self.assertTrue(np.array_equal(_bits(coil_spec.curve.dofs), _bits(expected_dofs)))
            self.assertTrue(np.array_equal(_bits(jnp.atleast_1d(coil_spec.current.value)), _bits([expected_current])))

    def test_fully_free_extraction_lowers_without_selector_matmuls(self):
        """Fully free DOF reconstruction lowers without selector matrix multiplications."""
        curves = [CurveXYZFourier(_NQUAD, _ORDER) for _ in range(2)]
        currents = [Current(1.0e4), Current(2.0e4)]
        bs_jax = JaxBiotSavart([Coil(c, i) for c, i in zip(curves, currents)])
        self.assertNotIn("stablehlo.dot_general", _lowered_extraction_text(bs_jax))

    def test_partially_fixed_extraction_lowers_one_placement_matmul_per_run(self):
        """Only the split curve's two runs need a placement; free maps are slices."""
        text = _lowered_extraction_text(self.coil_graph[2])
        self.assertEqual(text.count("stablehlo.dot_general"), 2)

    def test_gradient_through_the_mapping_matches_the_native_selection(self):
        """Autodiff extraction-map cotangents equal the native setter selection bitwise."""
        curves, _currents, bs_jax, oracle = self.coil_graph
        spec = bs_jax.coil_dof_extraction_spec()
        rng = np.random.default_rng(7)
        free_dofs = rng.normal(size=bs_jax.dof_size)
        weights = [rng.normal(size=curve.local_full_dof_size) for curve in curves]

        def weighted_curve_dofs(dofs):
            coil_specs = coil_specs_from_dof_extraction_spec(spec, dofs)
            return sum(
                jnp.dot(jnp.asarray(weight), coil_spec.curve.dofs) for weight, coil_spec in zip(weights, coil_specs)
            )

        autodiff_gradient = np.asarray(jax.grad(weighted_curve_dofs)(jnp.asarray(free_dofs)))

        def full_dofs():
            return np.concatenate([np.asarray(curve.local_full_x) for curve in curves])

        selection = np.zeros((bs_jax.dof_size, sum(w.size for w in weights)))
        oracle.x = np.zeros(bs_jax.dof_size)
        baseline = full_dofs()
        for column in range(bs_jax.dof_size):
            probe = np.zeros(bs_jax.dof_size)
            probe[column] = 1.0
            oracle.x = probe
            selection[column] = full_dofs() - baseline
        expected_gradient = selection @ np.concatenate(weights)
        self.assertTrue(np.array_equal(_bits(autodiff_gradient), _bits(expected_gradient)))

    def test_native_pullback_preserves_leaf_order_and_static_coil_indices(self):
        """Grouped pullbacks keep leaf order and static coil indices through flattening and JIT."""
        arrays = tuple(jnp.asarray([value], dtype=jnp.float64) for value in range(1, 7))
        pullback = BiotSavartFieldPullback(
            d_coil_arrays=((arrays[0], arrays[1], arrays[2]), (arrays[3], arrays[4], arrays[5])),
            coil_indices=((2, 0), (1,)),
        )
        leaves, treedef = jax.tree_util.tree_flatten(pullback)
        self.assertTrue(all(a is b for a, b in zip(leaves, arrays, strict=True)))
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(restored.coil_indices, ((2, 0), (1,)))
        compiled = jit(lambda value: jax.tree.map(lambda leaf: 2 * leaf, value))(restored)
        self.assertEqual(compiled.coil_indices, pullback.coil_indices)
        for actual, expected in zip(jax.tree_util.tree_leaves(compiled), arrays, strict=True):
            np.testing.assert_array_equal(actual, 2 * expected)


class TestStageTwoBiotSavartJax(JaxTestCase):
    """Reduced stage-two optimization with native objectives on the JAX field."""

    def test_native_objective_optimization(self):
        """The JAX gradient matches native and two L-BFGS-B iterations decrease J finitely."""
        set_backend("jax", device="cpu")
        s = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.3)
        s.set_zs(1, 0, 0.3)
        base_curves = create_equally_spaced_curves(2, s.nfp, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32)
        base_currents = [Current(1e5) for _ in base_curves]
        base_currents[0].fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, stellsym=True)
        points = s.gamma().reshape((-1, 3))

        def objective(field):
            field.set_points(points)
            return SquaredFlux(s, field) + 1e-5 * sum(CurveLength(c) for c in base_curves)

        native_JF = objective(BiotSavart(coils))
        JF = objective(JaxBiotSavart(coils))

        def fun(dofs):
            JF.x = dofs
            return JF.J(), JF.dJ()

        initial_value, initial_gradient = fun(JF.x)
        np.testing.assert_allclose(native_JF.J(), initial_value, rtol=1e-12)
        _close(initial_gradient, native_JF.dJ(), (1e-10, 1e-12 * np.max(np.abs(initial_gradient))))
        res = minimize(fun, JF.x, jac=True, method="L-BFGS-B", options={"maxiter": 2, "maxcor": 10})
        self.assertGreaterEqual(res.nit, 1)
        self.assertLess(res.fun, initial_value)
        self.assertTrue(np.all(np.isfinite(res.jac)))


class TestBiotSavartJaxStateCache(JaxTestCase):
    """Coil-state and field-output reuse, and invalidation on every notified change."""

    def test_evaluations_at_one_dof_state_build_coil_state_once(self):
        """Unchanged evaluations reuse coil state and outputs; a free-DOF change rebuilds once."""
        field, native = _fields(_coils()[1])
        builds = _spy(self.patches, backend, "_jitted_coil_set_spec_from_extraction_spec")
        B_evaluations = _spy(self.patches, backend, "grouped_biot_savart_B_from_spec")
        first = field.B()
        for _ in range(3):
            self.assertIs(field.B(), first)
        field.dB_by_dX()
        field.B_vjp(_COTANGENT)(field)
        field.dB_by_dcoilcurrents()
        field.coil_set_spec()
        counts = (builds.call_count, B_evaluations.call_count)
        self.assertEqual(counts, (1, 1), "coil spec and B builds at one coil-DOF state")
        field.x = np.asarray(field.x) + 1e-3
        _assert_field_matches_native(field, native)
        self.assertEqual(builds.call_count, 2)

    def test_per_current_outputs_reuse_arrays_and_follow_every_state_change(self):
        """Per-current outputs are new lists of reused arrays, refreshed by every state change."""

        def check(method_name, kernel):
            curves, currents, coils = _two_coils()
            field, _native = _fields(coils)
            evaluate = getattr(field, method_name)
            previous = evaluate()
            repeated = evaluate()
            self.assertIsNot(repeated, previous)
            self.assertTrue(all(a is b for a, b in zip(previous, repeated, strict=True)))
            repeated.clear()
            self.assertEqual(len(evaluate()), len(field.coils), "caller list edits changed outputs")
            mutations = {
                "free": lambda: setattr(field, "x", np.asarray(field.x) * 1.01),
                "full": lambda: setattr(field, "full_x", np.asarray(field.full_x) * 0.99),
                "parent": lambda: setattr(curves[1], "x", curves[1].x + 2e-3),
                "current": lambda: setattr(currents[0], "x", np.array([2.5e5])),
                "layout": lambda: curves[0].fix(0),
                "fixed": lambda: curves[0].set(0, curves[0].get(0) + 5e-3),
                "points": lambda: field.set_points(_OTHER_POINTS),
                "backend": lambda: set_backend("native"),
            }
            for mutation, mutate in mutations.items():
                mutate()
                actual = evaluate()
                expected = backend._per_coil_unit_field(field.get_points_cart_ref(), field.coil_set_spec(), kernel)
                self.assertTrue(all(a is not b for a, b in zip(actual, previous, strict=True)), mutation)
                for a, b in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=mutation)
                self.assertTrue(all(a is b for a, b in zip(actual, evaluate(), strict=True)), mutation)
                previous = actual

        methods = (
            ("dB_by_dcoilcurrents", backend.biot_savart_B),
            ("d2B_by_dXdcoilcurrents", backend.biot_savart_dB_by_dX),
            ("d3B_by_dXdXdcoilcurrents", backend.biot_savart_d2B_by_dXdX),
            ("dA_by_dcoilcurrents", backend.biot_savart_A),
            ("d2A_by_dXdcoilcurrents", backend.biot_savart_dA_by_dX),
            ("d3A_by_dXdXdcoilcurrents", backend.biot_savart_d2A_by_dXdX),
        )
        self.run_cases(check, [dict(method_name=m, kernel=k) for m, k in methods])

    def test_point_changes_release_cached_outputs_and_preserve_coil_geometry(self):
        """Every point setter releases obsolete field arrays and keeps the cached coil geometry."""

        def check(setter):
            field, _native = _fields([Coil(c, Current(1e5)) for c in _curves()])
            spec = field.coil_set_spec()
            field.B()
            field.dB_by_dX()
            field.B_and_dB()
            field.dB_by_dcoilcurrents()
            outputs = jax.tree_util.tree_leaves(tuple(field._field_outputs.values()))
            references = [weakref.ref(array) for array in outputs]
            del outputs
            setters = {
                "cart": lambda: field.set_points(_OTHER_POINTS),
                "jax": lambda: field.set_points(jnp.asarray(_OTHER_POINTS)),
                "cyl": lambda: field.set_points_cyl(np.array([[1.0, 0.3, 0.1], [0.9, -0.2, -0.1]])),
                "spec": lambda: field.set_points_from_spec(make_field_eval_spec(jnp.asarray(_OTHER_POINTS))),
                "clear": field.clear_points,
            }
            setters[setter]()
            self.assertTrue(all(reference() is None for reference in references), "obsolete outputs remain retained")
            self.assertIs(field.coil_set_spec(), spec)

        self.run_cases(check, _grid(setter=("cart", "jax", "cyl", "spec", "clear")))

    def test_field_and_parent_dof_changes_refresh_the_field(self):
        """Field and parent DOF setters refresh the JAX field to native values."""
        curves = _curves()
        currents = [Current(1e5) for _ in curves]
        field, native = _fields(coils_via_symmetries(curves, currents, 2, True))
        _assert_field_matches_native(field, native)
        field.x = np.asarray(field.x) * 1.01
        _assert_field_matches_native(field, native)
        curves[1].x = curves[1].x + 2e-3
        _assert_field_matches_native(field, native)
        field.full_x = np.asarray(field.full_x) * 0.995
        _assert_field_matches_native(field, native)

    def test_current_change_refreshes_the_field(self):
        """Changing a source current refreshes the JAX field to native values."""
        _curves, currents, coils = _two_coils()
        field, native = _fields(coils)
        _assert_field_matches_native(field, native)
        currents[0].x = np.array([2.5e5])
        _assert_field_matches_native(field, native)

    def test_fixed_dof_change_refreshes_the_field(self):
        """Fixing and then changing a captured DOF refreshes the field to native values."""
        curves = _curves()
        field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        _assert_field_matches_native(field, native)
        curves[0].fix(0)
        _assert_field_matches_native(field, native)
        curves[0].set(0, curves[0].get(0) + 5e-3)
        _assert_field_matches_native(field, native)

    def test_set_points_refreshes_the_field(self):
        """Replacing evaluation points refreshes the JAX field to native values."""
        field, native = _fields([Coil(c, Current(1e5)) for c in _curves()])
        _assert_field_matches_native(field, native)
        for points in (_OTHER_POINTS, _THREE_POINTS):
            field.set_points(points)
            native.set_points(points)
            _assert_field_matches_native(field, native)

    def test_shared_dofs_refresh_every_coil_that_reads_them(self):
        """Changing shared curve DOFs refreshes every coil that reads them to native values."""
        curve = _curves(1)[0]
        shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
        field, native = _fields([Coil(curve, Current(1e5)), Coil(shared, Current(-4e4))])
        _assert_field_matches_native(field, native)
        shared.x = np.asarray(shared.x) + 3e-3
        np.testing.assert_array_equal(curve.x, shared.x)
        _assert_field_matches_native(field, native)

    def test_current_sum_term_change_refreshes_the_field(self):
        """Changing a term of a composed current refreshes the field to native values."""
        curves = _curves()
        first, second = Current(1e5), Current(2e4)
        field, native = _fields([Coil(curves[0], first + second), Coil(curves[1], -2.5 * (first - second))])
        _assert_field_matches_native(field, native)
        second.x = np.array([-7e4])
        _assert_field_matches_native(field, native)

    def test_resample_refreshes_the_field(self):
        """Notified and direct sample replacements refresh perturbed geometry and the field."""
        perturbed, sample = _perturbed(2024)
        coils = [Coil(perturbed, Current(1e5))]
        field, native = _fields(coils)
        _assert_field_matches_native(field, native)
        before = np.asarray(field.B()).copy()
        perturbed.resample()
        _assert_field_matches_native(field, native)
        self.assertFalse(np.array_equal(np.asarray(field.B()), before))
        # A sample redrawn without notification: native caches stay stale, the
        # adapter detects the replaced sample on its next evaluation.
        before = np.asarray(field.B()).copy()
        sample.resample()
        perturbed.invalidate_cache()
        _assert_field_matches_native(field, _fields(coils)[1])
        self.assertFalse(np.array_equal(np.asarray(field.B()), before))

    def test_notified_in_place_sample_edit_refreshes_the_field(self):
        """An in-place sample edit reaches B, dB/dX and the B VJP at the next notification."""
        perturbed, sample = _perturbed(2025)
        field, native = _fields([Coil(perturbed, Current(1e5))])
        _assert_field_matches_native(field, native)
        before = np.asarray(field.B()).copy()
        sample[0][:, 0] += 0.05
        perturbed.set_recompute_flag()
        _assert_field_matches_native(field, native)
        self.assertFalse(np.array_equal(np.asarray(field.B()), before))

    def test_backend_reconfiguration_rebuilds_the_coil_state(self):
        """Backend invalidation rebuilds the coil state once and keeps native agreement."""
        field, native = _fields([Coil(c, Current(1e5)) for c in _curves()])
        builds = _spy(self.patches, backend, "_jitted_coil_set_spec_from_extraction_spec")
        field.B()
        field.B()
        self.assertEqual(builds.call_count, 1)
        set_backend("native")
        _assert_field_matches_native(field, native)
        self.assertEqual(builds.call_count, 2)

    def test_owner_updates_fingerprint_fixed_dofs_once(self):
        """Each owner update notifies the field; fixed DOFs are compared once, at the next read."""
        curves = _curves(3)
        field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        curves[0].fix(0)
        _assert_field_matches_native(field, native)
        reads = _spy(self.patches, field, "_current_captured_coil_state_fingerprint")
        for curve in curves:
            curve.x = np.asarray(curve.x) + 1e-3
        field.B()
        field.B_vjp(_COTANGENT)(field)
        self.assertEqual(reads.call_count, 1, f"{reads.call_count} fixed-DOF fingerprints for one evaluation")
        _assert_field_matches_native(field, native)

    def test_set_points_owns_a_jax_point_set_that_aliases_a_numpy_buffer(self):
        """Editing the caller's buffer after set_points moves neither the points nor B."""
        field, native = _fields([Coil(c, Current(1e5)) for c in _curves()])
        buffer = aligned_host_copy(_THREE_POINTS)
        points = jax.device_put(buffer)  # noqa: TID251 test needs a JAX array aliasing a host buffer
        if next(iter(points.devices())).platform == "cpu":
            self.assertTrue(np.shares_memory(np.asarray(points), buffer), "precondition: CPU placement aliases")
        field.set_points(points)
        field.B()
        buffer += 0.3
        np.testing.assert_array_equal(field.get_points_cart(), _THREE_POINTS)
        _assert_field_matches_native(field, native)

    def test_copies_follow_later_dof_changes(self):
        """A copy is a registered field over the coils: later DOF and layout changes reach it."""

        def check(make_copy):
            curves, currents, coils = _two_coils()
            field, native = _fields(coils)
            field.B()
            field.B_vjp(_COTANGENT)(field)
            if make_copy == "copy":
                copied = copy.copy(field)
            else:
                copied = copy.deepcopy(field, memo={id(field.coils): field.coils})
            self.assertIsNot(copied, field)
            self.assertTrue(all(mine is theirs for mine, theirs in zip(copied.coils, field.coils, strict=True)))
            np.testing.assert_array_equal(copied.get_points_cart(), _THREE_POINTS)
            currents[0].x = np.array([2e5])
            curves[1].x = np.asarray(curves[1].x) + 2e-3
            _assert_field_matches_native(copied, native)
            _assert_field_matches_native(field, native)
            curves[0].fix(0)
            curves[0].set(0, curves[0].get(0) + 5e-3)
            self.assertEqual(copied.dof_size, field.dof_size)
            _assert_field_matches_native(copied, native)

        self.run_cases(check, _grid(make_copy=("copy", "deepcopy_shared_coils")))

    def test_deepcopy_needs_deep_copyable_coils(self):
        """simsopt coils are not deep-copyable, so neither is a field over them (as natively)."""
        field, _native = _fields([Coil(c, Current(1e5)) for c in _curves()])
        with self.assertRaises(TypeError):
            copy.deepcopy(field)

    def test_resampling_updates_captured_geometry_and_pullbacks(self):
        """Resampling perturbed curves refreshes field values and VJPs, symmetry replicas too."""

        def check(replicas):
            perturbed = _perturbed(713, 2)[0]
            current = Current(1e5)
            if replicas:
                coils = coils_via_symmetries([perturbed], [current], 2, True)
            else:
                coils = [Coil(perturbed, current)]
            native = BiotSavart(coils).set_points(_POINTS)
            adapter = backend.JaxBiotSavart(coils).set_points(_POINTS)
            before = np.asarray(adapter.B()).copy()
            perturbed.resample()
            native.set_points(_POINTS)
            adapter.set_points(_POINTS)
            expected = native.B()
            actual = np.asarray(adapter.B())
            self.assertGreater(np.max(np.abs(expected - before)), 1e-6)
            _close(actual, expected, _TIGHT)
            adapter_vjp = adapter.B_vjp(_cotangent("B"))(adapter)
            native_vjp = cast(Derivative, native.B_vjp(_cotangent("B")))
            _close(adapter_vjp, np.asarray(native_vjp(native)), _VJP)

        self.run_cases(check, _grid(replicas=(False, True)))

    def test_unchanged_perturbation_evaluations_do_not_copy_samples(self):
        """Unchanged perturbed-field evaluations copy no sample and read no fingerprint."""
        curves, coils = _coils(perturbed=True)
        curve = curves[0]
        curve.sample._sample = [sample.view(_CountedSample) for sample in curve.sample._sample]
        field, native = _fields(coils, _POINTS)
        expected = np.asarray(field.B()).copy()
        _close(expected, native.B(), _TIGHT)
        spec = field.coil_dof_extraction_spec()
        fingerprint_reads = _spy(self.patches, field, "_current_captured_coil_state_fingerprint")
        _CountedSample.copies.clear()
        for _ in range(10):
            np.testing.assert_allclose(np.asarray(field.B()), expected, rtol=0, atol=0)
        self.assertFalse(_CountedSample.copies)
        self.assertFalse(fingerprint_reads.called)
        self.assertIs(field.coil_dof_extraction_spec(), spec)

    def test_notified_non_dof_parameter_changes_match_native(self):
        """A changed current scale, filament offset, rotation scale or rotation matrix reaches B
        and its pullback as natively, notified by the object, the field or its DOF setter."""

        def check(parameter, notifier):
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
            notify = {
                "changed_object": changed.set_recompute_flag,
                "field": field.set_recompute_flag,
                "field_x_setter": lambda: setattr(field, "x", field.x),
            }
            notify[notifier]()
            native = BiotSavart(coils).set_points(_POINTS)
            _close(field.B(), native.B(), _TIGHT)
            cotangent = _cotangent("B")
            expected = cast(Derivative, native.B_vjp(cotangent))
            _assert_partials_match(field.B_vjp(cotangent), expected)

        parameters = ("current_scale", "filament_offset", "rotation_scale", "rotation_matrix")
        notifiers = ("changed_object", "field", "field_x_setter")
        self.run_cases(check, _grid(parameter=parameters, notifier=notifiers))

    def test_fixed_dof_change_of_every_owner_reaches_the_field(self):
        """Every DOF owner of a graph with replicas, a current sum, a scaled current and a framed filament holds a
        fixed DOF; changing each with ``set`` reaches B, dB/dX and the B VJP as natively."""
        curves = _curves()
        rotation = FrameRotation(curves[0].quadpoints, 1)
        rotation.x = np.array([0.2, -0.1, 0.05])
        filament = CurveFilament(FramedCurveCentroid(curves[0], rotation), 0.01, 0.02)
        first, second = Current(1e5), Current(2e4)
        coils = [*coils_via_symmetries([curves[1]], [first + second], 2, True),
                 Coil(filament, ScaledCurrent(first, 2.0))]
        field, native = _fields(coils)
        owners = [opt for opt in field.unique_dof_lineage if opt.local_full_dof_size]
        self.assertEqual({type(owner) for owner in owners}, {CurveXYZFourier, FrameRotation, Current})
        for owner in owners:
            owner.fix(owner.local_full_dof_size - 1)
        _assert_field_matches_native(field, native)
        for owner in owners:
            last = owner.local_full_dof_size - 1
            owner.set(last, 1.01 * owner.get(last) + 1e-3)
            _assert_field_matches_native(field, native)

    def test_dofs_unshared_by_replace_dofs_are_fingerprinted(self):
        """A curve that stops sharing DOFs through ``replace_dofs`` becomes a DOF owner of its own, so a later
        change of its fixed DOF reaches the field."""
        curve = _curves(1)[0]
        shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
        field, native = _fields([Coil(curve, Current(1e5)), Coil(shared, Current(-4e4))])
        _assert_field_matches_native(field, native)
        own = CurveXYZFourier(curve.quadpoints, curve.order)
        own.x = np.asarray(curve.x) + 2e-3
        own.fix(0)
        shared.replace_dofs(own.dofs)
        # The native curve reads replaced DOFs at their next change.
        shared.set(0, shared.get(0) + 5e-3)
        _assert_field_matches_native(field, native)
        shared.set(0, shared.get(0) - 9e-3)
        _assert_field_matches_native(field, native)

    def test_every_rotation_and_current_scale_is_read_again(self):
        """Each rotation matrix and current scale of stellarator-symmetric replicas, changed in place and notified
        through the field's DOF setter, reaches B and its pullback as natively."""
        curves, coils = _coils()
        field = backend.JaxBiotSavart(coils).set_points(_POINTS)
        carriers = [opt for opt in field.ancestors if isinstance(opt, (RotatedCurve, ScaledCurrent))]
        # Per base curve: three rotated replicas and two flipped (scaled) currents.
        self.assertEqual(len(carriers), 5 * len(curves))
        for carrier in carriers:
            field.B()
            if isinstance(carrier, ScaledCurrent):
                carrier.scale = 1.5 * carrier.scale
            else:
                # Native evaluation reads rotmat and its transpose.
                carrier.rotmat = carrier.rotmat @ RotatedCurve(curves[0], 0.05, False).rotmat
                carrier.rotmatT = carrier.rotmat.T.copy()
            field.x = field.x
            native = BiotSavart(coils).set_points(_POINTS)
            _close(field.B(), native.B(), _TIGHT)
            cotangent = _cotangent("B")
            _assert_partials_match(field.B_vjp(cotangent), cast(Derivative, native.B_vjp(cotangent)))

    def test_x_equals_the_native_free_dof_vector(self):
        """``x`` equals ``Optimizable.x`` bit for bit through DOF changes, fixes, frees and replaced DOFs."""
        curves, currents, coils = _two_coils()
        field = backend.JaxBiotSavart(coils)

        def check():
            expected = Optimizable.x.fget(field)
            np.testing.assert_array_equal(field.x, expected)
            self.assertEqual(field.x.dtype, expected.dtype)

        check()
        curves[1].x = np.asarray(curves[1].x) + 1e-3
        check()
        curves[0].fix_all()
        check()
        currents[1].fix_all()
        check()
        curves[0].unfix(2)
        check()
        currents[0].replace_dofs(Current(-2e4).dofs)
        check()
        currents[0].fix_all()
        curves[0].fix_all()
        curves[1].fix_all()
        check()
        self.assertEqual(field.x.shape, (0,))


class TestBiotSavartJaxCoilStateToken(JaxTestCase):
    """Traceable runtime cache invalidation is keyed to coil DOF state."""

    @staticmethod
    def _make_two_basic_coils():
        coils = []
        for current_amp in (1.0e6, -1.0e6):
            curve = CurveXYZFourier(quadpoints=16, order=1)
            curve.x = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            coils.append(Coil(curve, Current(current_amp)))
        return coils

    @staticmethod
    def _coil_arrays_in_original_order(coil_set_spec):
        coil_count = sum(len(group.coil_indices) for group in coil_set_spec.groups)
        gammas = [None] * coil_count
        gammadashs = [None] * coil_count
        currents = [None] * coil_count
        for group in coil_set_spec.groups:
            for position, coil_index in enumerate(group.coil_indices):
                gammas[coil_index] = group.gammas[position]
                gammadashs[coil_index] = group.gammadashs[position]
                currents[coil_index] = group.currents[position]
        return gammas, gammadashs, currents

    def test_live_coil_set_spec_matches_explicit_dofs_and_native_coils(self):
        """The cached live spec is the explicit-DOF reconstruction of the native coils."""
        coils = self._make_two_basic_coils()
        bs_jax = backend.JaxBiotSavart(coils)
        live = self._coil_arrays_in_original_order(bs_jax.coil_set_spec())
        explicit = self._coil_arrays_in_original_order(bs_jax.coil_set_spec_from_dofs(bs_jax.x))
        native = (
            [coil.curve.gamma() for coil in coils],
            [coil.curve.gammadash() for coil in coils],
            [coil.current.get_value() for coil in coils],
        )
        for entries in zip(live, explicit, native, strict=True):
            for actual, from_dofs, expected in zip(*entries, strict=True):
                _close(actual, np.asarray(from_dofs), (1e-14, 1e-15))
                _close(actual, expected, (1e-13, 1e-14))

    def test_biotsavart_extraction_spec_changes_only_for_captured_dof_contract(self):
        """Free-DOF changes reuse the extraction spec; layout or fixed-DOF changes replace it."""
        coils = self._make_two_basic_coils()
        field = backend.JaxBiotSavart(list(coils))
        curve = coils[0].curve
        initial_spec = field.coil_dof_extraction_spec()
        initial_value = float(curve.local_full_x[0])
        curve.set(0, initial_value + 1.0e-3)
        self.assertIs(field.coil_dof_extraction_spec(), initial_spec)
        curve.fix(0)
        fixed_spec = field.coil_dof_extraction_spec()
        self.assertIsNot(fixed_spec, initial_spec)
        curve.set(0, initial_value + 2.0e-3)
        self.assertIsNot(field.coil_dof_extraction_spec(), fixed_spec)


class TestBiotSavartJaxOwnerProjection(JaxTestCase):
    """The packed owner projection: host transfers, shared-DOF owners, fixed nested owners."""

    def test_warmed_full_owner_projection_host_boundaries(self):
        """A warmed owner projection is native with one free-vector placement and one host read."""
        _, coils = _coils()
        field, native = _fields(coils, _POINTS)
        cotangent = _cotangent("B")
        native.B()
        expected = cast(Derivative, native.B_vjp(cotangent))
        pullback = field.B_pullback_native(cotangent)
        arrays, indices = pullback.d_coil_arrays, pullback.coil_indices
        field.coil_cotangents_to_derivative(arrays, indices)
        placements = []
        put = jax.device_put  # noqa: TID251 wraps the raw primitive to count placements

        def counted_put(value, *args, **kwargs):
            placements.extend(
                leaf.nbytes for leaf in jax.tree.leaves(value) if isinstance(leaf, np.ndarray) and leaf.ndim > 0
            )
            return put(value, *args, **kwargs)

        self.patches.enter_context(mock.patch.object(jax, "device_put", counted_put))
        materializations = _spy(self.patches, backend, "host_array")
        metadata_placements = _spy(self.patches, backend, "_place_array_tree_on_device")
        actual = field.coil_cotangents_to_derivative(arrays, indices)
        _close(actual(field), np.asarray(expected(native)), _VJP)
        # One free-vector H2D boundary (32 doubles), one packed D2H.
        self.assertEqual((len(placements), sum(placements), materializations.call_count), (1, 256, 1))
        self.assertFalse(metadata_placements.called)

    def test_packed_projection_preserves_distinct_owners_with_shared_dofs(self):
        """Distinct owners sharing DOFs keep independent writable native partials."""
        curves, _ = _coils()
        curve = curves[0]
        shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
        curve.fix(curve.local_dof_names[0])
        first, second = Current(1e5), Current(2e4)
        second.fix_all()
        coils = [Coil(curve, first + second), Coil(shared, -2.5 * (first - second))]
        field, native = _fields(coils, _POINTS)
        for evaluator in (field, native):
            evaluator.B()
        cotangent = _cotangent("B")
        expected = cast(Derivative, native.B_vjp(cotangent))
        actual = field.B_vjp(cotangent)
        self.assertEqual(set(actual.data), set(expected.data))
        _assert_partials_match(actual, expected)
        self.assertFalse(np.array_equal(actual.data[curve], actual.data[shared]))
        self.assertTrue(cast(np.ndarray, actual.data[curve]).flags.writeable)
        self.assertFalse(np.shares_memory(actual.data[curve], actual.data[shared]))

    def test_packed_projection_preserves_fixed_nested_curve_partials(self):
        """Packed projection keeps native partials of partly or fully fixed nested curves."""

        def check(fixed):
            curves, coils = _coils(perturbed=True)
            if fixed == "partial":
                curves[0].curve.fix(2)
            else:
                curves[0].curve.fix_all()
            field, native = _fields(coils, _POINTS)
            _close(field.B(), native.B(), _TIGHT)
            cotangent = _cotangent("B")
            expected = cast(Derivative, native.B_vjp(cotangent))
            actual = field.B_vjp(cotangent)
            self.assertEqual(set(actual.data), set(expected.data))
            _assert_partials_match(actual, expected)

        self.run_cases(check, _grid(fixed=("partial", "full")))


class TestBiotSavartJaxPointsAndSerialization(JaxTestCase):
    """Points are owned snapshots, must be set before evaluation, and survive serialization."""

    def test_cartesian_points_are_owned_snapshots(self):
        """Editing caller-owned Cartesian points leaves the adapter's points and B native."""
        points = aligned_host_copy(_POINTS)
        coils = [Coil(_curves()[0], Current(1e5))]
        native = BiotSavart(coils).set_points(points)
        adapter = backend.JaxBiotSavart(coils).set_points(points)
        adapter.B()
        points[:] += 0.3
        np.testing.assert_array_equal(adapter.get_points_cart(), native.get_points_cart())
        _close(adapter.B(), native.B(), _TIGHT)

    def test_cleared_points_must_be_set_before_evaluation(self):
        """After clear_points, evaluation raises until points are set again."""
        coils = [Coil(_curves()[0], Current(1e5))]
        adapter = backend.JaxBiotSavart(coils)
        adapter.clear_points()
        for quantity in ("B", "GradAbsB", "B_cyl", "get_points_cart"):
            with self.subTest(quantity=quantity):
                with self.assertRaisesRegex(RuntimeError, "call set_points"):
                    getattr(adapter, quantity)()
        with self.assertRaisesRegex(RuntimeError, "call set_points"):
            adapter.B_vjp(np.zeros((1, 3)))
        native = BiotSavart(coils).set_points(_POINTS)
        _close(adapter.set_points(_POINTS).B(), native.B(), _TIGHT)

    def test_serialization_preserves_points_and_field(self):
        """GSON and JSON round trips keep points and field values in either coordinate system."""

        def check(coordinates):
            coils = [Coil(_curves()[0], Current(1e5))]
            native, adapter = BiotSavart(coils), backend.JaxBiotSavart(coils)
            for field in (native, adapter):
                getattr(field, "set_points_" + coordinates)(_POINTS)
            expected_points = native.get_points_cart()
            with TemporaryDirectory() as tmp:
                for name, field in (("native", native), ("adapter", adapter)):
                    serialized = {}
                    payload = field.as_dict(serialized)
                    restored = type(field).from_dict(payload, serialized, {})
                    filename = Path(tmp) / (name + ".json")
                    field.save(filename, fmt="json")
                    for result in (restored, load(filename)):
                        points = result.get_points_cart()
                        np.testing.assert_array_equal(points, expected_points)
                        _close(result.B(), native.B(), _TIGHT)

        self.run_cases(check, _grid(coordinates=("cart", "cyl")))

    def test_payload_without_points_is_rejected_like_native(self):
        """A payload without points raises KeyError in both fields; None restores cleared points."""
        coils = [Coil(_curves()[0], Current(1e5))]
        for field in (BiotSavart(coils), backend.JaxBiotSavart(coils)):
            with self.subTest(field=type(field).__name__):
                serialized = {}
                payload = field.as_dict(serialized)
                del payload["points"]
                with self.assertRaises(KeyError):
                    type(field).from_dict(payload, serialized, {})
        adapter = backend.JaxBiotSavart(coils)
        adapter.clear_points()
        serialized = {}
        payload = adapter.as_dict(serialized)
        restored = backend.JaxBiotSavart.from_dict(payload, serialized, {})
        self.assertIsNone(restored.get_points_cart_ref())


class TestBiotSavartJaxContract(JaxTestCase):
    """Native field arithmetic rejects the JAX adapter."""

    def test_native_field_arithmetic_is_rejected(self):
        """Adapter sums and scalar multiples, and native sum and multiple wrappers over it, raise."""
        operations = {
            "left_scale": lambda field, native: 2 * field,
            "right_scale": lambda field, native: field * 2,
            "add": lambda field, native: field + field,
            "reverse_add": lambda field, native: 0 + field,
            "native_left": lambda field, native: native + field,
            "native_right": lambda field, native: field + native,
            "sum_wrapper": lambda field, native: MagneticFieldSum([field, native]),
            "scale_wrapper": lambda field, native: MagneticFieldMultiply(2, field),
        }

        def check(operation):
            field, native = backend.JaxBiotSavart([]), BiotSavart([])
            with self.assertRaisesRegex(TypeError, "simsopt.field.BiotSavart"):
                operations[operation](field, native)

        self.run_cases(check, _grid(operation=operations))
