"""Immutable geometry and coil-state pytrees for pure JAX kernels.

Array fields are dynamic leaves; metadata is static during tracing.
Use these payloads as kernel inputs, not as dictionary keys.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import gcd
from typing import Union

import jax

from simsopt_jax.pytree import pytree_dataclass

from ._math_utils import (
    as_jax_float64 as _as_float64_array,
    eye,
)


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
        curve (CurveSpec): Immutable geometry, quadrature nodes and static parameters.
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
    """Immutable map from owner DOFs to coil geometry and current.

    Pass device-resident payloads as arguments; copy them to the host with
    ``simsopt_jax.backend.runtime.host_value`` before capturing them in a
    compiled closure under ``jax.transfer_guard("disallow")``.

    Args:
        curve (CurveSpec): Immutable geometry, quadrature nodes and static parameters.
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
    """Immutable map from owner DOFs to grouped coils.

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
    meta=("sample_derivatives",),
)
class CurvePerturbedSpec:
    """Immutable wrapper payload for a perturbed base curve.

    Args:
        dofs (jax.Array): Wrapper DOFs, shape (D,), including dependencies in the layout
            described by base_curve_map.
        quadpoints (jax.Array): Normalized, dimensionless curve parameters, shape (Q,),
            conventionally in [0, 1).
        base_curve (CurveSpec): Immutable unperturbed or centerline geometry.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve inputs.
        sample_gamma (jax.Array): Additive sampled position perturbation, shape (Q, 3),
            in meters.
        sample_gammadash (jax.Array): Additive first parameter-derivative perturbation,
            shape (Q, 3), in meters.
        sample_gammadashdash (jax.Array): Additive second parameter-derivative
            perturbation, shape (Q, 3), in meters.
        sample_gammadashdashdash (jax.Array): Additive third parameter-derivative
            perturbation, shape (Q, 3), in meters.
        sample_derivatives (int): Number of parameter derivatives the sample provides,
            1 to 3; higher sample terms are zero placeholders, which filament frames
            refuse to differentiate through.
    """

    dofs: jax.Array
    quadpoints: jax.Array
    base_curve: CurveSpec
    base_curve_map: OptimizableDofMapSpec
    sample_gamma: jax.Array
    sample_gammadash: jax.Array
    sample_gammadashdash: jax.Array
    sample_gammadashdashdash: jax.Array
    sample_derivatives: int


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
        base_curve (CurveSpec): Immutable unperturbed or centerline geometry.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve inputs.
        rotation (RotationSpec): Zero or Fourier rotation of the normal/binormal frame.
        rotation_map (OptimizableDofMapSpec): Map of wrapper DOFs to rotation coefficients.
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
    CurveRZFourierSpec,
    CurvePlanarFourierSpec,
    CurveHelicalSpec,
    CurveXYZFourierSymmetriesSpec,
    CurvePerturbedSpec,
    CurveFilamentSpec,
]


# Base-curve derivatives a filament frame consumes beyond the filament's own:
# the centroid frame is built from gammadash, the Frenet frame from gammadashdash.
_FILAMENT_FRAME_EXTRA_DERIVATIVES = {"centroid": 1, "frenet": 2}


def _filament_base_derivative_order(frame_kind: str, order: int) -> int:
    """Base-curve derivatives a ``frame_kind`` filament needs for ``order`` of its own."""
    return order + _FILAMENT_FRAME_EXTRA_DERIVATIVES[frame_kind]


def _require_sample_derivatives(spec: CurvePerturbedSpec, order: int) -> None:
    """Raise ``ValueError`` if the sample lacks derivative ``order``."""
    if order > spec.sample_derivatives:
        raise ValueError(
            f"The perturbation sample has {spec.sample_derivatives} derivatives, "
            f"but derivative {order} is needed. Adjust the `n_derivs` parameter "
            "of the sampler to access higher derivatives (the JAX spec carries "
            "at most 3)."
        )


def _require_curve_derivatives(spec: CurveSpec, order: int) -> None:
    """Require sufficient sample derivatives throughout nested curve and filament specs."""
    if isinstance(spec, CurvePerturbedSpec):
        _require_sample_derivatives(spec, order)
        _require_curve_derivatives(spec.base_curve, order)
    elif isinstance(spec, CurveFilamentSpec):
        _require_curve_derivatives(
            spec.base_curve, _filament_base_derivative_order(spec.frame_kind, order)
        )


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
        CoilGroupSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
        CurveXYZFourierSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
    """
    return CurveXYZFourierSpec(
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
        CurveRZFourierSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
        CurveXYZFourierSymmetriesSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
        CurvePlanarFourierSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
        CurveHelicalSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
            full template; each owner range and its target range have equal length.
        input_mode (str): full selects all reconstructed DOFs; any other value selects
            the local slice input_start:input_end.
        input_start (int): Inclusive start index of the requested local slice.
        input_end (int): Exclusive end of that slice.

    Returns:
        OptimizableDofMapSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.

    Raises:
        ValueError: A segment has unequal owner/target lengths or a target range
            outside the template.
    """
    template = _as_float64_array(template_full_dofs)
    segments = tuple(
        (int(owner_start), int(owner_end), int(target_start), int(target_end))
        for owner_start, owner_end, target_start, target_end in owner_segments
    )
    for owner_start, owner_end, target_start, target_end in segments:
        if not (
            0 <= owner_start <= owner_end
            and 0 <= target_start <= target_end <= template.shape[0]
            and owner_end - owner_start == target_end - target_start
        ):
            raise ValueError(
                f"Owner segment {(owner_start, owner_end, target_start, target_end)} "
                "must copy an owner range into an equal-length target range within "
                f"the {template.shape[0]} template DOFs."
            )
    return OptimizableDofMapSpec(
        template_full_dofs=template,
        owner_segments=segments,
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
        FrameRotationSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
        ZeroRotationSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
    sample_derivatives: int,
) -> CurvePerturbedSpec:
    """Build an immutable CurvePerturbedSpec payload for JAX kernels.

    Args:
        dofs (array-like): Wrapper DOFs, shape (D,), including dependencies in the
            layout described by base_curve_map.
        quadpoints (array-like): Normalized dimensionless parameters, shape (Q,),
            on the base curve's sampling grid; does not resample geometry.
        base_curve (CurveSpec): Immutable unperturbed or centerline geometry.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve inputs.
        sample_gamma (array-like): Additive sampled position perturbation, shape (Q, 3),
            in meters.
        sample_gammadash (array-like): Additive first parameter-derivative perturbation,
            shape (Q, 3), in meters.
        sample_gammadashdash (array-like): Additive second parameter-derivative
            perturbation, shape (Q, 3), in meters.
        sample_gammadashdashdash (array-like): Additive third parameter-derivative
            perturbation, shape (Q, 3), in meters.
        sample_derivatives (int): Number of available sample derivatives, 1 to 3;
            required. Pass zero placeholders for higher sample terms.

    Returns:
        CurvePerturbedSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.

    Raises:
        ValueError: sample_derivatives is outside 1 to 3.
    """
    if not 1 <= sample_derivatives <= 3:
        raise ValueError(
            f"sample_derivatives must be 1, 2 or 3, got {sample_derivatives}."
        )
    return CurvePerturbedSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        base_curve=base_curve,
        base_curve_map=base_curve_map,
        sample_gamma=_as_float64_array(sample_gamma),
        sample_gammadash=_as_float64_array(sample_gammadash),
        sample_gammadashdash=_as_float64_array(sample_gammadashdash),
        sample_gammadashdashdash=_as_float64_array(sample_gammadashdashdash),
        sample_derivatives=int(sample_derivatives),
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
        quadpoints (array-like): Normalized dimensionless parameters, shape (Q,),
            on the base curve's and rotation's grid; does not resample geometry.
        base_curve (CurveSpec): Immutable unperturbed or centerline geometry.
        base_curve_map (OptimizableDofMapSpec): Map from wrapper DOFs to base-curve inputs.
        rotation (RotationSpec): Zero or Fourier rotation of the normal/binormal frame.
        rotation_map (OptimizableDofMapSpec): Map of wrapper DOFs to rotation coefficients.
        frame_kind (str): centroid or frenet frame for the filament offset.
        dn (float): Normal-frame filament offset in meters.
        db (float): Binormal-frame filament offset in meters.

    Returns:
        CurveFilamentSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.

    Raises:
        ValueError: frame_kind is not centroid or frenet, or a perturbed base
            curve's sample lacks derivatives required for the filament tangent
            (2 for centroid, 3 for frenet).
    """
    if frame_kind not in _FILAMENT_FRAME_EXTRA_DERIVATIVES:
        raise ValueError(
            f"frame_kind must be 'centroid' or 'frenet', got {frame_kind!r}."
        )
    spec = CurveFilamentSpec(
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
    _require_curve_derivatives(spec, 1)
    return spec


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
        CoilSymmetrySpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
    """
    return CoilSymmetrySpec(
        rotmat=eye(3) if rotmat is None else _as_float64_array(rotmat),
        scale=float(scale),
        has_rotation=rotmat is not None,
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
        curve (CurveSpec): Immutable geometry, quadrature nodes and static parameters.
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
        CoilDofExtractionSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
        CoilSetDofExtractionSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
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
        FieldEvalSpec object: Frozen pytree; converted arrays
            keep shape and use float64; host NumPy inputs are copied.
    """
    return FieldEvalSpec(points=_as_float64_array(points))


def make_grouped_coil_set_spec(
    groups: Iterable[CoilGroupSpec | tuple[object, object, object, tuple[int, ...]]],
) -> GroupedCoilSetSpec:
    """Build an immutable GroupedCoilSetSpec payload for JAX kernels.

    Args:
        groups (Iterable[CoilGroupSpec or tuple]): Existing specs or geometry/current arrays
            with shapes (C, Q, 3), (C, Q, 3), (C,) and original coil-index tuples;
            quadrature count may vary between groups.

    Returns:
        GroupedCoilSetSpec object: Frozen pytree payload retaining supplied specs;
            newly converted array leaves retain their argument shape, use float64
            and snapshot host NumPy inputs.
    """
    group_specs: list[CoilGroupSpec] = []
    for group in groups:
        if isinstance(group, CoilGroupSpec):
            group_specs.append(group)
            continue
        gammas, gammadashs, currents, coil_indices = group
        group_specs.append(
            make_coil_group_spec(gammas, gammadashs, currents, coil_indices)
        )
    return GroupedCoilSetSpec(groups=tuple(group_specs))
