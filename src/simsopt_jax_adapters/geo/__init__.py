"""Legacy geometry-object adapters for ``simsopt_jax``."""

from .curve_objectives import (
    JaxCurveCurveDistance,
    JaxCurveLength,
    JaxCurveSurfaceDistance,
    JaxLpCurveCurvature,
    JaxMeanSquaredCurvature,
)

__all__ = (
    "JaxCurveCurveDistance",
    "JaxCurveLength",
    "JaxCurveSurfaceDistance",
    "JaxLpCurveCurvature",
    "JaxMeanSquaredCurvature",
)
