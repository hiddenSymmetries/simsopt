"""
Solve package: serial/MPI optimizers, permanent-magnet and wireframe optimization.

Macromagnetics and permanent_magnet_optimization are lazy-loaded to avoid pulling
in heavy JAX dependencies when only serial/MPI solvers are needed (e.g. in
tests/field). This prevents OOM during force Taylor tests on CI.
"""
from .serial import *
from .mpi import *
from .wireframe_optimization import *

# permanent_magnet_optimization and macromagnetics are lazy-loaded via __getattr__
# to avoid pulling in JAX when only serial/MPI is needed.

__all__ = (
    serial.__all__
    + mpi.__all__
    + wireframe_optimization.__all__
    + ["relax_and_split", "GPMO"]  # permanent_magnet_optimization
    + ["MacroMag", "Tiles", "SolverParams", "assemble_blocks_subset", "muse2tiles", "build_prism", "rotation_angle", "get_rotmat"]  # macromagnetics
)


def __getattr__(name: str):
    """Lazy-load permanent_magnet_optimization or macromagnetics on first access."""
    if name in ("relax_and_split", "GPMO"):
        from . import permanent_magnet_optimization
        return getattr(permanent_magnet_optimization, name)
    if name in ("MacroMag", "Tiles", "SolverParams", "assemble_blocks_subset", "muse2tiles", "build_prism", "rotation_angle", "get_rotmat"):
        from . import macromagnetics
        return getattr(macromagnetics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
