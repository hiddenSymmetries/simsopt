"""Legacy field-object adapters for ``simsopt_jax``."""

from .biotsavart_backend import JaxBiotSavart
from .force import (
    JaxB2Energy,
    JaxLpCurveForce,
    JaxLpCurveTorque,
    JaxNetFluxes,
    JaxSquaredMeanForce,
    JaxSquaredMeanTorque,
)

__all__ = (
    "JaxB2Energy",
    "JaxBiotSavart",
    "JaxLpCurveForce",
    "JaxLpCurveTorque",
    "JaxNetFluxes",
    "JaxSquaredMeanForce",
    "JaxSquaredMeanTorque",
)
