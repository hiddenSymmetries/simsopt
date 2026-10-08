"""Interpreter requirement of the JAX backend, checked before any other module loads.

``simsopt_jax`` needs JAX 0.10 or newer (the ``jax`` extra), which needs Python 3.11, and the
package uses Python 3.10+ features such as ``zip(..., strict=True)`` and dataclass slots. The
native ``simsopt`` package does not import it and keeps its own minimum.
"""

import sys

MINIMUM_PYTHON = (3, 11)

if sys.version_info < MINIMUM_PYTHON:
    raise ImportError(
        "simsopt_jax requires Python %d.%d or newer (JAX 0.10 does); found %d.%d. "
        "Native simsopt works without it." % (MINIMUM_PYTHON + sys.version_info[:2])
    )
