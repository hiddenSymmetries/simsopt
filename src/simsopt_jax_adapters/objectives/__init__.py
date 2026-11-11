"""Native objective adapters for ``simsopt_jax``."""

from .flux import JaxSquaredFlux
from .fused import JaxFusedStageTwoObjective

__all__ = ("JaxFusedStageTwoObjective", "JaxSquaredFlux")
