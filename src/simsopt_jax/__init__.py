"""Canonical namespace for Simsopt JAX implementations.

Requires Python 3.11 or newer, like the JAX 0.10 it uses (``pip install simsopt[jax]``).
"""

from . import _requirements, core, runtime

__all__ = ("core", "runtime")
