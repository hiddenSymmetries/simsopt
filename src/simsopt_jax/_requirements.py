"""Interpreter and JAX requirements checked before any backend kernels load.

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


from jaxlib.version import __version__ as jaxlib_version
from numpy.lib import NumpyVersion

MINIMUM_JAX = "0.10.0"


def _require_jax_version(name: str, version: str) -> None:
    """Reject unsupported optional dependencies before their kernels can load."""
    release_version = version.partition("+")[0]
    if NumpyVersion(release_version) < NumpyVersion(MINIMUM_JAX):
        raise ImportError(
            f"simsopt_jax requires {name} >= {MINIMUM_JAX}; found {version}. "
            "Native simsopt works without it."
        )


# JAX itself reports an incompatible jaxlib as RuntimeError. Check its static
# version first so unsupported optional installations follow the ImportError gate.
_require_jax_version("jaxlib", jaxlib_version)

import jax

_require_jax_version("jax", jax.__version__)
