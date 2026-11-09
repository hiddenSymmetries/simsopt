"""Canonical namespace for Simsopt JAX implementations.

Requires Python 3.12 or newer, like the JAX 0.12 it uses (``pip install simsopt[jax]``).
Importing it enables JAX float64 (``jax_enable_x64``) for the process, as
importing ``simsopt.geo`` does.
"""

# _requirements (the Python and JAX version gate) must load first: tests treat
# an ImportError raised after it has loaded as a broken package, not an
# unsupported runtime.
from . import _requirements, core, runtime

__all__ = ("core", "runtime")
