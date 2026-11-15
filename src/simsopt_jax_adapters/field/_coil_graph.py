from collections.abc import Callable

import numpy as np
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import FrameRotation


def _is_scaled_current_like(current):
    return hasattr(current, "current_to_scale") and hasattr(current, "scale")


def _is_rotated_curve_like(curve):
    return hasattr(curve, "curve") and hasattr(curve, "rotmat")


def _as_float64_scalar(value):
    return np.asarray(value, dtype=np.float64).item()


def _unwrap_coil_curve_and_current_objects(curve, current):
    scale = 1.0
    while _is_scaled_current_like(current):
        scale *= _as_float64_scalar(current.scale)
        current = current.current_to_scale

    rotmat = None
    while _is_rotated_curve_like(curve):
        next_rotmat = np.asarray(curve.rotmat, dtype=np.float64)
        rotmat = next_rotmat if rotmat is None else next_rotmat @ rotmat
        curve = curve.curve

    return curve, rotmat, current, scale


def _scale_parameters(opt) -> np.ndarray:
    return np.array((_as_float64_scalar(opt.scale),))


def _filament_parameters(opt) -> np.ndarray:
    return np.array((_as_float64_scalar(opt.dn), _as_float64_scalar(opt.db)))


def _rotation_parameters(opt) -> np.ndarray:
    return np.asarray(opt.rotmat, dtype=np.float64).ravel()


def _parameter_reader(opt) -> Callable[[object], np.ndarray] | None:
    if _is_scaled_current_like(opt) or isinstance(opt, FrameRotation):
        return _scale_parameters
    if isinstance(opt, CurveFilament):
        return _filament_parameters
    if _is_rotated_curve_like(opt):
        return _rotation_parameters
    return None


def coil_graph_parameter_readers(optimizables) -> tuple[tuple[object, Callable[[object], np.ndarray]], ...]:
    """Select the coil-graph nodes that carry non-DOF parameters, each with the function that reads them.

    Which nodes carry parameters follows from their types and from attributes that coil-graph classes
    set at construction, so a selection stays valid while the graph is unchanged.

    Args:
        optimizables (Iterable[Optimizable]): Coil-graph nodes, in a fixed order.

    Returns:
        tuple: ``(node, reader)`` pairs in node order, for :func:`read_coil_graph_parameters`.
    """
    return tuple((opt, reader) for opt in optimizables if (reader := _parameter_reader(opt)) is not None)


def read_coil_graph_parameters(readers) -> np.ndarray:
    """Return the current non-DOF parameters captured by JAX coil reconstruction.

    Includes current and frame-rotation scales, filament offsets and captured
    rotation matrices; a change invalidates the captured reconstruction.

    Args:
        readers (tuple): ``(node, reader)`` pairs from :func:`coil_graph_parameter_readers`.

    Returns:
        numpy.ndarray: Float64 parameters, shape (K,), in node order; scales
            and rotation-matrix entries are dimensionless and filament offsets
            are in meters.
    """
    return np.concatenate([reader(opt) for opt, reader in readers]) if readers else np.zeros(0)
