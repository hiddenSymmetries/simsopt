"""Legacy geometry-object adapters for ``simsopt_jax``."""

from .curve_objectives import (
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureJAX,
    MeanSquaredCurvatureJAX,
)

__all__ = (
    "CurveCurveDistanceJAX",
    "CurveLengthJAX",
    "CurveSurfaceDistanceJAX",
    "LpCurveCurvatureJAX",
    "MeanSquaredCurvatureJAX",
)
