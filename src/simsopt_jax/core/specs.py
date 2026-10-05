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
    as_runtime_float64 as _as_runtime_float64,
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
    """
    return host_value(spec)


@pytree_dataclass(data=("dofs", "quadpoints"), meta=("order",))
class CurveXYZFourierSpec:
    """Immutable payload for pure JAX CurveXYZFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


@pytree_dataclass(data=("dofs", "quadpoints"), meta=("order",))
class OrientedCurveXYZFourierSpec:
    """Immutable payload for pure JAX OrientedCurveXYZFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


@pytree_dataclass(
    data=("dofs", "quadpoints"),
    meta=("order", "nfp", "stellsym"),
)
class CurveRZFourierSpec:
    """Immutable payload for pure JAX CurveRZFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    nfp: int
    stellsym: bool


@pytree_dataclass(data=("dofs", "quadpoints"), meta=("order",))
class CurvePlanarFourierSpec:
    """Immutable payload for pure JAX CurvePlanarFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


@pytree_dataclass(
    data=("dofs", "quadpoints"),
    meta=("order", "m", "ell", "R0", "r"),
)
class CurveHelicalSpec:
    """Immutable payload for pure JAX CurveHelical geometry."""

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
    """Immutable mapping from an owner's full DOF vector into one nested Optimizable."""

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
    """Immutable payload for pure JAX FrameRotation evaluation."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    scale: float


@pytree_dataclass(data=("quadpoints",), meta=())
class ZeroRotationSpec:
    """Immutable zero-rotation payload."""

    quadpoints: jax.Array


@pytree_dataclass(data=("value",), meta=())
class CurrentValueSpec:
    """Immutable scalar-current payload."""

    value: jax.Array


@pytree_dataclass(
    data=("rotmat",),
    meta=("scale", "has_rotation"),
)
class CoilSymmetrySpec:
    """Immutable rotation/scale payload for symmetric coil replicas."""

    rotmat: jax.Array
    scale: float
    has_rotation: bool


@pytree_dataclass(data=("curve", "current", "symmetry"), meta=())
class CoilSpec:
    """Immutable coil payload: curve identity, current, and spatial placement."""

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
    """Immutable owner-DOF -> grouped-coil reconstruction payload."""

    coils: tuple[CoilDofExtractionSpec, ...]


@pytree_dataclass(data=("points",), meta=())
class FieldEvalSpec:
    """Immutable field-evaluation point cloud."""

    points: jax.Array


@pytree_dataclass(
    data=("gammas", "gammadashs", "currents"),
    meta=("coil_indices",),
)
class CoilGroupSpec:
    """One rectangular coil batch with a shared quadrature count."""

    gammas: jax.Array
    gammadashs: jax.Array
    currents: jax.Array
    coil_indices: tuple[int, ...]

    def field_inputs(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.gammas, self.gammadashs, self.currents

    def as_grouped_data(self) -> tuple[jax.Array, jax.Array, jax.Array, list[int]]:
        return self.gammas, self.gammadashs, self.currents, list(self.coil_indices)


@pytree_dataclass(data=("groups",), meta=())
class GroupedCoilSetSpec:
    """Immutable grouped coil geometry/current payload."""

    groups: tuple[CoilGroupSpec, ...]

    def field_inputs(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array], ...]:
        return tuple(group.field_inputs() for group in self.groups)

    def coil_index_lists(self) -> tuple[tuple[int, ...], ...]:
        return tuple(group.coil_indices for group in self.groups)

    def as_grouped_data(
        self,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, list[int]], ...]:
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
    """Immutable wrapper payload for a perturbed base curve."""

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
    """Immutable wrapper payload for a finite-build filament curve."""

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
    """Return the closed discriminant for a curve spec variant."""
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
    return FrameRotationSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        scale=float(scale),
    )


def make_zero_rotation_spec(*, quadpoints: object) -> ZeroRotationSpec:
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
    return CoilSetDofExtractionSpec(coils=tuple(coils))


def apply_coil_symmetry(
    gamma: jax.Array,
    gammadash: jax.Array,
    current: jax.Array,
    symmetry: CoilSymmetrySpec,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply rotation/scale transform to curve geometry and current."""
    if symmetry.has_rotation:
        rotmat = _as_runtime_float64(symmetry.rotmat, reference=gamma)
        gamma = gamma @ rotmat
        gammadash = gammadash @ rotmat
    return (
        gamma,
        gammadash,
        current * _as_runtime_float64(symmetry.scale, reference=current),
    )


def make_field_eval_spec(points: object) -> FieldEvalSpec:
    return FieldEvalSpec(points=_as_float64_array(points))


def make_grouped_coil_set_spec(groups: Iterable[CoilGroupSpec | tuple[jax.Array, jax.Array, jax.Array, tuple[int, ...]]]) -> GroupedCoilSetSpec:
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
