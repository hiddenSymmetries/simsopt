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


def coil_graph_parameters(optimizables) -> np.ndarray:
    """Non-DOF parameters a JAX coil reconstruction captures from a coil graph.

    These are the attributes native evaluation reads on every call but which
    are not DOFs: ``ScaledCurrent.scale``, ``CurveFilament.dn`` and ``db``,
    ``FrameRotation.scale`` and the ``RotatedCurve.rotmat`` the
    reconstruction captures. A changed entry means a captured reconstruction
    is stale.

    Args:
        optimizables (Iterable[Optimizable]): Coil-graph nodes, in a fixed order.

    Returns:
        numpy.ndarray: Float64 parameters, shape (K,), in node order; scales
            and rotation-matrix entries are dimensionless and filament offsets
            are in meters.
    """
    parameters = []
    for opt in optimizables:
        if _is_scaled_current_like(opt) or isinstance(opt, FrameRotation):
            parameters.append(_as_float64_scalar(opt.scale))
        elif isinstance(opt, CurveFilament):
            parameters.extend((_as_float64_scalar(opt.dn), _as_float64_scalar(opt.db)))
        elif _is_rotated_curve_like(opt):
            parameters.extend(np.asarray(opt.rotmat, dtype=np.float64).ravel())
    return np.asarray(parameters, dtype=np.float64)
