import numpy as np


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
