"""Legacy field-object adapters for ``simsopt_jax``."""

from .biotsavart_backend import BiotSavartJAX
from .force import (
    B2EnergyJAX,
    LpCurveForceJAX,
    LpCurveTorqueJAX,
    NetFluxesJAX,
    SquaredMeanForceJAX,
    SquaredMeanTorqueJAX,
)

__all__ = (
    "B2EnergyJAX",
    "BiotSavartJAX",
    "LpCurveForceJAX",
    "LpCurveTorqueJAX",
    "NetFluxesJAX",
    "SquaredMeanForceJAX",
    "SquaredMeanTorqueJAX",
)
