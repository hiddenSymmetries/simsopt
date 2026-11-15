"""Canonical namespace for Simsopt JAX implementations.

Requires Python 3.12+ and JAX 0.12+ (``pip install simsopt[jax]``).
Importing enables process-wide JAX float64 (``jax_enable_x64``).
"""

# _requirements (the Python and JAX version gate) must load first: tests treat
# an ImportError raised after it has loaded as a broken package, not an
# unsupported runtime.
from . import _requirements, backend, core

__all__ = ("backend", "core")
