"""Immutable pytree specs for the pure JAX kernel layer.

These dataclasses are the stable JAX-facing state boundary for geometry,
and coil geometry kernels. The public ``Optimizable`` wrappers still
own mutable compatibility state and flat-DOF orchestration, but compiled JAX
paths should consume these explicit specs rather than live object graphs.
They carry JAX arrays as pytree data leaves, so treat them as immutable payloads
for tracing, not as dictionary keys.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import gcd
from typing import Literal, TypeVar, Union

import jax
import numpy as np

from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.runtime.host_boundary import host_value

from ._math_utils import (
    as_jax_float64 as _as_float64_array,
    runtime_device_put,
)

__all__ = [
    "CoilSpec",
    "CoilGroupSpec",
    "CoilDofExtractionSpec",
    "CoilSetDofExtractionSpec",
    "CoilSymmetrySpec",
    "apply_coil_symmetry",
    "CurveFilamentSpec",
    "CurveHelicalSpec",
    "OrientedCurveXYZFourierSpec",
    "make_oriented_curve_xyzfourier_spec",
    "CurvePlanarFourierSpec",
    "CurveSpec",
    "CurveSpecKind",
    "CurvePerturbedSpec",
    "CurrentValueSpec",
    "CurveRZFourierSpec",
    "CurveXYZFourierSpec",
    "CurveXYZFourierSymmetriesSpec",
    "FieldEvalSpec",
    "FrameRotationSpec",
    "GroupedCoilSetSpec",
    "OptimizableDofMapSpec",
    "RotationSpec",
    "ZeroRotationSpec",
    "curve_spec_kind",
    "make_coil_dof_extraction_spec",
    "make_coil_symmetry_spec",
    "make_coil_group_spec",
    "make_coil_set_dof_extraction_spec",
    "make_curve_filament_spec",
    "make_curve_helical_spec",
    "make_curve_planarfourier_spec",
    "make_curve_perturbed_spec",
    "make_curve_rzfourier_spec",
    "make_curve_xyzfourier_spec",
    "make_curve_xyzfouriersymmetries_spec",
    "make_field_eval_spec",
    "make_frame_rotation_spec",
    "make_grouped_coil_set_spec",
    "make_optimizable_dof_map_spec",
    "make_zero_rotation_spec",
    "host_resident_spec",
]


_SpecT = TypeVar("_SpecT")


def host_resident_spec(spec: _SpecT) -> _SpecT:
    """Return ``spec`` with every array leaf materialized on the host.

    Call this on any spec a compiled program captures in a closure rather than
    receives as an argument. XLA turns a captured concrete array into an MLIR
    literal by copying it back to the host, once per lowering, which
    ``jax.transfer_guard("disallow")`` refuses on a real device; host leaves
    lower to the same literals with no copy. Specs passed as program arguments
    must NOT be host-resident -- that placement is the argument's own implicit
    host-to-device transfer. The read-back goes through the host-boundary
    owner so it is audited like every other device-to-host crossing.

    Args:
        spec (pytree): Immutable spec whose array leaves have arbitrary shape.

    Returns:
        pytree object: Same spec structure with array leaves materialized on
            the host, preserving each leaf shape and dtype.
    """
    return host_value(spec)


@pytree_dataclass(data=("dofs", "quadpoints"), meta=("order",))
class CurveXYZFourierSpec:
    """Immutable payload for pure JAX CurveXYZFourier geometry.

    Args:
        dofs (jax.Array): Shape (3 * (2 * order + 1),), in meters; x, y, z blocks each
            use constant, sin(1), cos(1), ..., sin(order), cos(order).
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


@pytree_dataclass(data=("dofs", "quadpoints"), meta=("order",))
class OrientedCurveXYZFourierSpec:
    """Immutable payload for pure JAX OrientedCurveXYZFourier geometry.

    Args:
        dofs (jax.Array): Shape (6 + 6 * order,); translation xyz in meters,
            yaw/pitch/roll in radians, then x/y/z sine/cosine blocks in meters with no
            constant modes.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


@pytree_dataclass(
    data=("dofs", "quadpoints"),
    meta=("order", "nfp", "stellsym"),
)
class CurveRZFourierSpec:
    """Immutable payload for pure JAX CurveRZFourier geometry.

    Args:
        dofs (jax.Array): In meters; shape (2 * order + 1,) for symmetry with [rc, zs],
            otherwise (4 * order + 2,) with [rc, rs, zc, zs]. Cosine blocks include mode
            zero.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        nfp (int): Number of field periods; positive and static during tracing.
        stellsym (bool): Use stellarator-symmetric Fourier coefficient restrictions.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    nfp: int
    stellsym: bool


@pytree_dataclass(data=("dofs", "quadpoints"), meta=("order",))
class CurvePlanarFourierSpec:
    """Immutable payload for pure JAX CurvePlanarFourier geometry.

    Args:
        dofs (jax.Array): Shape (2 * order + 8,); radial cosine and sine coefficients in
            meters, four dimensionless quaternion components (scalar first), then xyz
            center in meters.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


@pytree_dataclass(
    data=("dofs", "quadpoints"),
    meta=("order", "m", "ell", "R0", "r"),
)
class CurveHelicalSpec:
    """Immutable payload for pure JAX CurveHelical geometry.

    Args:
        dofs (jax.Array): Shape (2 * order + 1,), angular coefficients in radians; A
            cosine modes including zero, then B sine modes starting at one.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        m (int): Helical poloidal winding count.
        ell (int): Nonzero helical toroidal winding count.
        R0 (float): Major radius in meters.
        r (float): Minor radius in meters.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    m: int
    ell: int
    R0: float
    r: float


@pytree_dataclass(
    data=("dofs", "quadpoints"),
    meta=("order", "nfp", "stellsym", "ntor"),
)
class CurveXYZFourierSymmetriesSpec:
    """Immutable payload for pure JAX CurveXYZFourierSymmetries geometry.

    Mirrors ``simsopt.geo.curvexyzfouriersymmetries.CurveXYZFourierSymmetries``
    constructor parameters needed by ``jaxXYZFourierSymmetriescurve_pure``.
    ``nfp`` and ``ntor`` must be coprime (enforced at host-side construction;
    the spec is the frozen runtime payload).

    Args:
        dofs (jax.Array): In meters; shape (3 * order + 1,) with [xc, ys, zs] under
            symmetry, otherwise (6 * order + 3,) with [xc, xs, yc, ys, zc, zs]. Cosine
            blocks include mode zero.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        nfp (int): Number of field periods; positive and static during tracing.
        stellsym (bool): Use stellarator-symmetric Fourier coefficient restrictions.
        ntor (int): Toroidal winding count, coprime to nfp.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    nfp: int
    stellsym: bool
    ntor: int


@pytree_dataclass(
    data=("template_full_dofs",),
    meta=("owner_segments", "input_mode", "input_start", "input_end"),
)
class OptimizableDofMapSpec:
    """Immutable map from owner DOFs to a full local template or its requested slice.

    Args:
        template_full_dofs (jax.Array): Baseline full local DOFs, shape (D_full,),
            including fixed entries.
        owner_segments (tuple[tuple[int, int, int, int], ...]): Half-open (owner_start,
            owner_end, target_start, target_end) copy ranges from owner DOFs into the
            full template.
        input_mode (str): full selects all reconstructed DOFs; any other value selects
            the local slice input_start:input_end.
        input_start (int): Inclusive start index of the requested local slice.
        input_end (int): Exclusive end of that slice.
    """

    template_full_dofs: jax.Array
    owner_segments: tuple[tuple[int, int, int, int], ...]
    input_mode: str
    input_start: int
    input_end: int


@pytree_dataclass(
    data=("dofs", "quadpoints"),
    meta=("order", "scale"),
)
class FrameRotationSpec:
    """Immutable payload for pure JAX FrameRotation evaluation.

    Args:
        dofs (jax.Array): Shape (2 * order + 1,), coefficients ordered constant, sin(1),
            cos(1), ...; scale converts their values into radians.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        scale (float): Dimensionless multiplier applied to the Fourier rotation angle.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    scale: float


@pytree_dataclass(data=("quadpoints",), meta=())
class ZeroRotationSpec:
    """Immutable zero-rotation payload.

    Args:
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
    """

    quadpoints: jax.Array


@pytree_dataclass(data=("value",), meta=())
class CurrentValueSpec:
    """Immutable scalar-current payload.

    Args:
        value (jax.Array): Current value in amperes, shape (1,), as consumed by coil
            reconstruction.
    """

    value: jax.Array


@pytree_dataclass(
    data=("rotmat",),
    meta=("scale", "has_rotation"),
)
class CoilSymmetrySpec:
    """Immutable rotation/scale payload for symmetric coil replicas.

    Args:
        rotmat (jax.Array): Shape (3, 3) row-vector spatial rotation/reflection matrix.
        scale (float): Dimensionless multiplier applied to the replica current,
            including sign reversal for reflection.
        has_rotation (bool): Whether to apply rotmat to curve positions and tangents.
    """

    rotmat: jax.Array
    scale: float
    has_rotation: bool


@pytree_dataclass(data=("curve", "current", "symmetry"), meta=())
class CoilSpec:
    """Immutable coil payload: curve identity, current, and spatial placement.

    Args:
        curve (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        current (CurrentValueSpec): Scalar coil-current payload in amperes.
        symmetry (CoilSymmetrySpec): Spatial transform and current scaling.
    """

    curve: CurveSpec
    current: CurrentValueSpec
    symmetry: CoilSymmetrySpec


@pytree_dataclass(
    data=(
        "curve",
        "curve_map",
        "current_map",
        "symmetry",
        "current_term_maps",
    ),
    meta=(
        "current_term_scales",
        "curve_source_index",
    ),
)
class CoilDofExtractionSpec:
    """Immutable owner-DOF -> coil-spec reconstruction payload.

    Frozen: only the owner DOF vector varies per call. A program that takes
    this payload as an *argument* wants it device-resident, which is how the
    maker returns it; a program that *captures* it in a closure must first
    call ``host_resident_spec`` on it -- see that function for why.

    Args:
        curve (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        curve_map (OptimizableDofMapSpec): Map from owner DOFs into curve coefficients.
        current_map (OptimizableDofMapSpec): Map from owner DOFs into the current value;
            used when no term maps are supplied.
        symmetry (CoilSymmetrySpec): Spatial transform and current scaling.
        current_term_maps (tuple[OptimizableDofMapSpec, ...]): Independent current-term
            maps; empty uses current_map.
        current_term_scales (tuple[float, ...]): Dimensionless weights paired with
            current_term_maps for a linear current expression.
        curve_source_index (int or None): Shared-curve reconstruction key; None
            reconstructs this coil independently.
    """

    curve: CurveSpec
    curve_map: OptimizableDofMapSpec
    current_map: OptimizableDofMapSpec
    symmetry: CoilSymmetrySpec
    current_term_maps: tuple[OptimizableDofMapSpec, ...] = ()
    current_term_scales: tuple[float, ...] = ()
    curve_source_index: int | None = None


@pytree_dataclass(data=("coils",), meta=())
class CoilSetDofExtractionSpec:
    """Immutable owner-DOF -> grouped-coil reconstruction payload.

    Args:
        coils (tuple[CoilDofExtractionSpec, ...]): Per-coil reconstruction contracts in
            public coil order.
    """

    coils: tuple[CoilDofExtractionSpec, ...]


@pytree_dataclass(data=("points",), meta=())
class FieldEvalSpec:
    """Immutable field-evaluation point cloud.

    Args:
        points (jax.Array): Cartesian evaluation points, shape (P, 3), in meters.
    """

    points: jax.Array


@pytree_dataclass(
    data=("gammas", "gammadashs", "currents"),
    meta=("coil_indices",),
)
class CoilGroupSpec:
    """One rectangular coil batch with a shared quadrature count.

    Args:
        gammas (jax.Array): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (jax.Array): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (jax.Array): Coil currents, shape (C,), in amperes.
        coil_indices (tuple[int, ...]): Original coil indices, one per batch row, shape
            (C,) when represented as an index array.
    """

    gammas: jax.Array
    gammadashs: jax.Array
    currents: jax.Array
    coil_indices: tuple[int, ...]

    def field_inputs(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the grouped geometry and current kernel inputs.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array]: Geometry, tangents and
                currents, shapes (C, Q, 3), (C, Q, 3), (C,), retaining the stored
                arrays.
        """
        return self.gammas, self.gammadashs, self.currents

    def as_grouped_data(self) -> tuple[jax.Array, jax.Array, jax.Array, list[int]]:
        """Return kernel inputs together with original coil indices.

        Returns:
            tuple: Stored arrays of shapes (C, Q, 3), (C, Q, 3), (C,), followed by
                a new list[int] of original coil indices.
        """
        return self.gammas, self.gammadashs, self.currents, list(self.coil_indices)


@pytree_dataclass(data=("groups",), meta=())
class GroupedCoilSetSpec:
    """Immutable grouped coil geometry/current payload.

    Args:
        groups (tuple[CoilGroupSpec, ...]): Rectangular batches, each with a shared
            quadrature count.
    """

    groups: tuple[CoilGroupSpec, ...]

    def field_inputs(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array], ...]:
        """Return kernel input triples in group order.

        Returns:
            tuple[tuple]: One (gammas, gammadashs, currents) tuple per group with
                shapes (C, Q, 3), (C, Q, 3), (C,); group sizes may vary.
        """
        return tuple(group.field_inputs() for group in self.groups)

    def coil_index_lists(self) -> tuple[tuple[int, ...], ...]:
        """Return the original coil ordering for each quadrature group.

        Returns:
            tuple[tuple[int, ...], ...]: Original coil indices for each group in
                matching row order.
        """
        return tuple(group.coil_indices for group in self.groups)

    def as_grouped_data(
        self,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, list[int]], ...]:
        """Return grouped kernel inputs and original coil indices.

        Returns:
            tuple[tuple]: Array triples of shapes (C, Q, 3), (C, Q, 3), (C,) and
                new index lists, one tuple per group.
        """
        return tuple(group.as_grouped_data() for group in self.groups)


RotationSpec = Union[FrameRotationSpec, ZeroRotationSpec]


@pytree_dataclass(
    data=(
        "dofs",
        "quadpoints",
        "base_curve",
        "base_curve_map",
        "sample_gamma",
        "sample_gammadash",
        "sample_gammadashdash",
        "sample_gammadashdashdash",
    ),
    meta=(),
)
class CurvePerturbedSpec:
    """Immutable wrapper payload for a perturbed base curve.

    Args:
        dofs (jax.Array): Wrapper DOFs, shape (D,), including dependencies in the layout
            described by base_curve_map.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        base_curve (CurveSpec): Immutable geometry of the unperturbed or centerline
            curve.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve
            inputs.
        sample_gamma (jax.Array): Additive sampled position perturbation, shape (Q, 3),
            in meters.
        sample_gammadash (jax.Array): Additive first parameter-derivative perturbation,
            shape (Q, 3), in meters.
        sample_gammadashdash (jax.Array): Additive second parameter-derivative
            perturbation, shape (Q, 3), in meters.
        sample_gammadashdashdash (jax.Array): Additive third parameter-derivative
            perturbation, shape (Q, 3), in meters.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    base_curve: CurveSpec
    base_curve_map: OptimizableDofMapSpec
    sample_gamma: jax.Array
    sample_gammadash: jax.Array
    sample_gammadashdash: jax.Array
    sample_gammadashdashdash: jax.Array


@pytree_dataclass(
    data=(
        "dofs",
        "quadpoints",
        "base_curve",
        "base_curve_map",
        "rotation",
        "rotation_map",
    ),
    meta=("frame_kind", "dn", "db"),
)
class CurveFilamentSpec:
    """Immutable wrapper payload for a finite-build filament curve.

    Args:
        dofs (jax.Array): Wrapper DOFs, shape (D,), including curve and rotation
            dependencies in their mapped layouts.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        base_curve (CurveSpec): Immutable geometry of the unperturbed or centerline
            curve.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve
            inputs.
        rotation (RotationSpec): Zero or Fourier rotation of the normal/binormal frame.
        rotation_map (OptimizableDofMapSpec): Map from wrapper DOFs to rotation
            coefficients.
        frame_kind (str): centroid or frenet frame for the filament offset.
        dn (float): Normal-frame filament offset in meters.
        db (float): Binormal-frame filament offset in meters.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    base_curve: CurveSpec
    base_curve_map: OptimizableDofMapSpec
    rotation: RotationSpec
    rotation_map: OptimizableDofMapSpec
    frame_kind: str
    dn: float
    db: float


CurveSpec = Union[
    CurveXYZFourierSpec,
    OrientedCurveXYZFourierSpec,
    CurveRZFourierSpec,
    CurvePlanarFourierSpec,
    CurveHelicalSpec,
    CurveXYZFourierSymmetriesSpec,
    CurvePerturbedSpec,
    CurveFilamentSpec,
]

CurveSpecKind = Literal[
    "xyz_fourier",
    "oriented_xyz_fourier",
    "rz_fourier",
    "planar_fourier",
    "helical",
    "xyz_fourier_symmetries",
    "perturbed",
    "filament",
]


def curve_spec_kind(spec: CurveSpec) -> CurveSpecKind:
    """Return the closed discriminant for a curve spec variant.

    Args:
        spec (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.

    Returns:
        str: Closed discriminant identifying the supported curve-spec variant.
    """
    if isinstance(spec, CurveXYZFourierSpec):
        return "xyz_fourier"
    if isinstance(spec, OrientedCurveXYZFourierSpec):
        return "oriented_xyz_fourier"
    if isinstance(spec, CurveRZFourierSpec):
        return "rz_fourier"
    if isinstance(spec, CurvePlanarFourierSpec):
        return "planar_fourier"
    if isinstance(spec, CurveHelicalSpec):
        return "helical"
    if isinstance(spec, CurveXYZFourierSymmetriesSpec):
        return "xyz_fourier_symmetries"
    if isinstance(spec, CurvePerturbedSpec):
        return "perturbed"
    if isinstance(spec, CurveFilamentSpec):
        return "filament"
    raise TypeError(f"Unsupported curve spec type: {type(spec).__name__}")


def make_coil_group_spec(
    gammas: object,
    gammadashs: object,
    currents: object,
    coil_indices: Iterable[int],
) -> CoilGroupSpec:
    """Build an immutable CoilGroupSpec payload for JAX kernels.

    Args:
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.
        coil_indices (Iterable[int]): Original coil indices, one per batch row, shape
            (C,) when represented as an index array.

    Returns:
        CoilGroupSpec object: Frozen pytree payload; newly converted array
            leaves retain their argument shape, use float64 and
            snapshot host NumPy inputs.
    """
    return CoilGroupSpec(
        gammas=_as_float64_array(gammas),
        gammadashs=_as_float64_array(gammadashs),
        currents=_as_float64_array(currents),
        coil_indices=tuple(int(index) for index in coil_indices),
    )


def make_curve_xyzfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
) -> CurveXYZFourierSpec:
    """Build an immutable CurveXYZFourierSpec payload for JAX kernels.

    Args:
        dofs (array-like): Shape (3 * (2 * order + 1),), in meters; x, y, z blocks each
            use constant, sin(1), cos(1), ..., sin(order), cos(order).
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.

    Returns:
        CurveXYZFourierSpec object: Frozen pytree payload; newly converted
            array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    return CurveXYZFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
    )


def make_oriented_curve_xyzfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
) -> OrientedCurveXYZFourierSpec:
    """Build an immutable OrientedCurveXYZFourierSpec payload for JAX kernels.

    Args:
        dofs (array-like): Shape (6 + 6 * order,); translation xyz in meters,
            yaw/pitch/roll in radians, then x/y/z sine/cosine blocks in meters with no
            constant modes.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.

    Returns:
        OrientedCurveXYZFourierSpec object: Frozen pytree payload; newly
            converted array leaves retain their argument shape, use runtime
            floating precision and snapshot host NumPy inputs.
    """
    return OrientedCurveXYZFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
    )


def make_curve_rzfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    nfp: int,
    stellsym: bool,
) -> CurveRZFourierSpec:
    """Build an immutable CurveRZFourierSpec payload for JAX kernels.

    Args:
        dofs (array-like): In meters; shape (2 * order + 1,) for symmetry with [rc, zs],
            otherwise (4 * order + 2,) with [rc, rs, zc, zs]. Cosine blocks include mode
            zero.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        nfp (int): Number of field periods; positive and static during tracing.
        stellsym (bool): Use stellarator-symmetric Fourier coefficient restrictions.

    Returns:
        CurveRZFourierSpec object: Frozen pytree payload; newly converted
            array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    return CurveRZFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        nfp=int(nfp),
        stellsym=bool(stellsym),
    )


def make_curve_xyzfouriersymmetries_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
) -> CurveXYZFourierSymmetriesSpec:
    """Build an immutable CurveXYZFourierSymmetriesSpec payload for JAX kernels.

    Args:
        dofs (array-like): In meters; shape (3 * order + 1,) with [xc, ys, zs] under
            symmetry, otherwise (6 * order + 3,) with [xc, xs, yc, ys, zc, zs]. Cosine
            blocks include mode zero.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        nfp (int): Number of field periods; positive and static during tracing.
        stellsym (bool): Use stellarator-symmetric Fourier coefficient restrictions.
        ntor (int): Toroidal winding count, coprime to nfp.

    Returns:
        CurveXYZFourierSymmetriesSpec object: Frozen pytree payload; newly
            converted array leaves retain their argument shape, use runtime
            floating precision and snapshot host NumPy inputs.
    """
    nfp_int = int(nfp)
    ntor_int = int(ntor)
    if gcd(ntor_int, nfp_int) != 1:
        raise ValueError(
            "CurveXYZFourierSymmetriesSpec requires nfp and ntor coprime; "
            f"got nfp={nfp_int}, ntor={ntor_int}"
        )
    return CurveXYZFourierSymmetriesSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        nfp=nfp_int,
        stellsym=bool(stellsym),
        ntor=ntor_int,
    )


def make_curve_planarfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
) -> CurvePlanarFourierSpec:
    """Build an immutable CurvePlanarFourierSpec payload for JAX kernels.

    Args:
        dofs (array-like): Shape (2 * order + 8,); radial cosine and sine coefficients
            in meters, four dimensionless quaternion components (scalar first), then xyz
            center in meters.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.

    Returns:
        CurvePlanarFourierSpec object: Frozen pytree payload; newly converted
            array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    return CurvePlanarFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
    )


def make_curve_helical_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    m: int,
    ell: int,
    R0: float,
    r: float,
) -> CurveHelicalSpec:
    """Build an immutable CurveHelicalSpec payload for JAX kernels.

    Args:
        dofs (array-like): Shape (2 * order + 1,), angular coefficients in radians; A
            cosine modes including zero, then B sine modes starting at one.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        m (int): Helical poloidal winding count.
        ell (int): Nonzero helical toroidal winding count.
        R0 (float): Major radius in meters.
        r (float): Minor radius in meters.

    Returns:
        CurveHelicalSpec object: Frozen pytree payload; newly converted array
            leaves retain their argument shape, use float64 and
            snapshot host NumPy inputs.
    """
    return CurveHelicalSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        m=int(m),
        ell=int(ell),
        R0=float(R0),
        r=float(r),
    )


def make_optimizable_dof_map_spec(
    *,
    template_full_dofs: object,
    owner_segments: Iterable[tuple[int, int, int, int]],
    input_mode: str,
    input_start: int,
    input_end: int,
) -> OptimizableDofMapSpec:
    """Build an immutable OptimizableDofMapSpec payload for JAX kernels.

    Args:
        template_full_dofs (array-like): Baseline full local DOFs, shape (D_full,),
            including fixed entries.
        owner_segments (Iterable[tuple[int, int, int, int]]): Half-open (owner_start,
            owner_end, target_start, target_end) copy ranges from owner DOFs into the
            full template.
        input_mode (str): full selects all reconstructed DOFs; any other value selects
            the local slice input_start:input_end.
        input_start (int): Inclusive start index of the requested local slice.
        input_end (int): Exclusive end of that slice.

    Returns:
        OptimizableDofMapSpec object: Frozen pytree payload; newly converted
            array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    return OptimizableDofMapSpec(
        template_full_dofs=_as_float64_array(template_full_dofs),
        owner_segments=tuple(
            (
                int(owner_start),
                int(owner_end),
                int(target_start),
                int(target_end),
            )
            for owner_start, owner_end, target_start, target_end in owner_segments
        ),
        input_mode=str(input_mode),
        input_start=int(input_start),
        input_end=int(input_end),
    )


def make_frame_rotation_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    scale: float,
) -> FrameRotationSpec:
    """Build an immutable FrameRotationSpec payload for JAX kernels.

    Args:
        dofs (array-like): Shape (2 * order + 1,), coefficients ordered constant,
            sin(1), cos(1), ...; scale converts their values into radians.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        order (int): Maximum Fourier mode, nonnegative and static during tracing.
        scale (float): Dimensionless multiplier applied to the Fourier rotation angle.

    Returns:
        FrameRotationSpec object: Frozen pytree payload; newly converted array
            leaves retain their argument shape, use float64 and
            snapshot host NumPy inputs.
    """
    return FrameRotationSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        scale=float(scale),
    )


def make_zero_rotation_spec(*, quadpoints: object) -> ZeroRotationSpec:
    """Build an immutable ZeroRotationSpec payload for JAX kernels.

    Args:
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).

    Returns:
        ZeroRotationSpec object: Frozen pytree payload; newly converted array
            leaves retain their argument shape, use float64 and
            snapshot host NumPy inputs.
    """
    return ZeroRotationSpec(quadpoints=_as_float64_array(quadpoints))


def make_curve_perturbed_spec(
    *,
    dofs: object,
    quadpoints: object,
    base_curve: CurveSpec,
    base_curve_map: OptimizableDofMapSpec,
    sample_gamma: object,
    sample_gammadash: object,
    sample_gammadashdash: object,
    sample_gammadashdashdash: object,
) -> CurvePerturbedSpec:
    """Build an immutable CurvePerturbedSpec payload for JAX kernels.

    Args:
        dofs (array-like): Wrapper DOFs, shape (D,), including dependencies in the
            layout described by base_curve_map.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        base_curve (CurveSpec): Immutable geometry of the unperturbed or centerline
            curve.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve
            inputs.
        sample_gamma (array-like): Additive sampled position perturbation, shape (Q, 3),
            in meters.
        sample_gammadash (array-like): Additive first parameter-derivative perturbation,
            shape (Q, 3), in meters.
        sample_gammadashdash (array-like): Additive second parameter-derivative
            perturbation, shape (Q, 3), in meters.
        sample_gammadashdashdash (array-like): Additive third parameter-derivative
            perturbation, shape (Q, 3), in meters.

    Returns:
        CurvePerturbedSpec object: Frozen pytree payload; newly converted
            array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    return CurvePerturbedSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        base_curve=base_curve,
        base_curve_map=base_curve_map,
        sample_gamma=_as_float64_array(sample_gamma),
        sample_gammadash=_as_float64_array(sample_gammadash),
        sample_gammadashdash=_as_float64_array(sample_gammadashdash),
        sample_gammadashdashdash=_as_float64_array(sample_gammadashdashdash),
    )


def make_curve_filament_spec(
    *,
    dofs: object,
    quadpoints: object,
    base_curve: CurveSpec,
    base_curve_map: OptimizableDofMapSpec,
    rotation: RotationSpec,
    rotation_map: OptimizableDofMapSpec,
    frame_kind: str,
    dn: float,
    db: float,
) -> CurveFilamentSpec:
    """Build an immutable CurveFilamentSpec payload for JAX kernels.

    Args:
        dofs (array-like): Wrapper DOFs, shape (D,), including curve and rotation
            dependencies in their mapped layouts.
        quadpoints (array-like): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        base_curve (CurveSpec): Immutable geometry of the unperturbed or centerline
            curve.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve
            inputs.
        rotation (RotationSpec): Zero or Fourier rotation of the normal/binormal frame.
        rotation_map (OptimizableDofMapSpec): Map from wrapper DOFs to rotation
            coefficients.
        frame_kind (str): centroid or frenet frame for the filament offset.
        dn (float): Normal-frame filament offset in meters.
        db (float): Binormal-frame filament offset in meters.

    Returns:
        CurveFilamentSpec object: Frozen pytree payload; newly converted array
            leaves retain their argument shape, use float64 and
            snapshot host NumPy inputs.
    """
    return CurveFilamentSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        base_curve=base_curve,
        base_curve_map=base_curve_map,
        rotation=rotation,
        rotation_map=rotation_map,
        frame_kind=str(frame_kind),
        dn=float(dn),
        db=float(db),
    )


def _normalize_rotmat(rotmat: object | None) -> tuple[jax.Array, bool]:
    if rotmat is None:
        return runtime_device_put(np.eye(3, dtype=np.float64), dtype=np.float64), False
    return _as_float64_array(rotmat), True


def make_coil_symmetry_spec(
    *,
    rotmat: object | None = None,
    scale: float = 1.0,
) -> CoilSymmetrySpec:
    """Build an immutable CoilSymmetrySpec payload for JAX kernels.

    Args:
        rotmat (array-like or None): Shape (3, 3) row-vector spatial rotation/reflection
            matrix; None requests identity.
        scale (float): Dimensionless multiplier applied to the replica current,
            including sign reversal for reflection.

    Returns:
        CoilSymmetrySpec object: Frozen pytree payload; newly converted array
            leaves retain their argument shape, use float64 and
            snapshot host NumPy inputs.
    """
    rotmat_jax, has_rotation = _normalize_rotmat(rotmat)
    return CoilSymmetrySpec(
        rotmat=rotmat_jax,
        scale=float(scale),
        has_rotation=has_rotation,
    )


def make_coil_dof_extraction_spec(
    *,
    curve: CurveSpec,
    curve_map: OptimizableDofMapSpec,
    current_map: OptimizableDofMapSpec,
    current_term_maps: tuple[OptimizableDofMapSpec, ...] = (),
    current_term_scales: tuple[float, ...] = (),
    curve_source_index: int | None = None,
    rotmat: object | None = None,
    scale: float = 1.0,
) -> CoilDofExtractionSpec:
    """Build an immutable CoilDofExtractionSpec payload for JAX kernels.

    Args:
        curve (CurveSpec): Immutable curve geometry, quadrature nodes and static
            parameters.
        curve_map (OptimizableDofMapSpec): Map from owner DOFs into curve coefficients.
        current_map (OptimizableDofMapSpec): Map from owner DOFs into the current value;
            used when no term maps are supplied.
        current_term_maps (tuple[OptimizableDofMapSpec, ...]): Independent current-term
            maps; empty uses current_map.
        current_term_scales (tuple[float, ...]): Dimensionless weights paired with
            current_term_maps for a linear current expression.
        curve_source_index (int or None): Shared-curve reconstruction key; None
            reconstructs this coil independently.
        rotmat (array-like or None): Shape (3, 3) row-vector spatial rotation/reflection
            matrix; None requests identity.
        scale (float): Dimensionless multiplier applied to the replica current,
            including sign reversal for reflection.

    Returns:
        CoilDofExtractionSpec object: Frozen pytree payload; newly converted
            array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    if len(current_term_maps) != len(current_term_scales):
        raise ValueError("current term maps and scales must have equal length")
    return CoilDofExtractionSpec(
        curve=curve,
        curve_map=curve_map,
        current_map=current_map,
        symmetry=make_coil_symmetry_spec(rotmat=rotmat, scale=scale),
        current_term_maps=current_term_maps,
        current_term_scales=tuple(float(value) for value in current_term_scales),
        curve_source_index=(
            None if curve_source_index is None else int(curve_source_index)
        ),
    )


def make_coil_set_dof_extraction_spec(
    coils: Iterable[CoilDofExtractionSpec],
) -> CoilSetDofExtractionSpec:
    """Build an immutable CoilSetDofExtractionSpec payload for JAX kernels.

    Args:
        coils (Iterable[CoilDofExtractionSpec]): Per-coil extraction contracts in public
            coil order.

    Returns:
        CoilSetDofExtractionSpec object: Frozen pytree payload; newly
            converted array leaves retain their argument shape, use runtime
            floating precision and snapshot host NumPy inputs.
    """
    return CoilSetDofExtractionSpec(coils=tuple(coils))


def apply_coil_symmetry(
    gamma: jax.Array,
    gammadash: jax.Array,
    current: jax.Array,
    symmetry: CoilSymmetrySpec,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply rotation/scale transform to curve geometry and current.

    Args:
        gamma (jax.Array): Cartesian curve positions, shape (Q, 3), in meters.
        gammadash (jax.Array): First derivative with respect to the normalized curve
            parameter, shape (Q, 3), in meters.
        current (jax.Array): Scalar current, shape (), in amperes.
        symmetry (CoilSymmetrySpec): Spatial transform and current scaling.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Positions and tangents of
            shape (Q, 3), transformed as row vectors by rotmat when enabled, and
            scalar current multiplied by symmetry.scale.
    """
    if symmetry.has_rotation:
        rotmat = _as_float64_array(symmetry.rotmat)
        gamma = gamma @ rotmat
        gammadash = gammadash @ rotmat
    return (
        gamma,
        gammadash,
        current * _as_float64_array(symmetry.scale),
    )


def make_field_eval_spec(points: object) -> FieldEvalSpec:
    """Build an immutable FieldEvalSpec payload for JAX kernels.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.

    Returns:
        FieldEvalSpec object: Frozen pytree payload; newly converted array
            leaves retain their argument shape, use float64 and
            snapshot host NumPy inputs.
    """
    return FieldEvalSpec(points=_as_float64_array(points))


def make_grouped_coil_set_spec(groups: Iterable[CoilGroupSpec | tuple[jax.Array, jax.Array, jax.Array, tuple[int, ...]]]) -> GroupedCoilSetSpec:
    """Build an immutable GroupedCoilSetSpec payload for JAX kernels.

    Args:
        groups (Iterable[CoilGroupSpec or tuple]): Groups of geometry/current arrays
            with shapes (C, Q, 3), (C, Q, 3), (C,) and original coil-index tuples;
            quadrature count may vary between groups.

    Returns:
        GroupedCoilSetSpec object: Frozen pytree payload; newly converted
            array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    group_specs = []
    for group in groups:
        if isinstance(group, CoilGroupSpec):
            group_specs.append(group)
            continue
        gammas, gammadashs, currents, coil_indices = group
        group_specs.append(
            make_coil_group_spec(
                gammas,
                gammadashs,
                currents,
                coil_indices,
            )
        )
    return GroupedCoilSetSpec(groups=tuple(group_specs))
