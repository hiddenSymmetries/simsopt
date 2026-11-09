"""Interpreter and JAX requirements checked, and float64 enabled, before any kernels load.

``simsopt_jax`` needs JAX 0.12 or newer (the ``jax`` extra), which needs Python 3.12. Every
0.12 release qualifies, including four-part ones such as ``0.12.0.1``. The native ``simsopt``
package does not import it and keeps its own minimum.
"""

import re
import sys

MINIMUM_PYTHON = (3, 12)

if sys.version_info < MINIMUM_PYTHON:
    raise ImportError(
        "simsopt_jax requires Python %d.%d or newer (JAX 0.12 does); found %d.%d. "
        "Native simsopt works without it." % (MINIMUM_PYTHON + sys.version_info[:2])
    )


from jaxlib.version import __version__ as jaxlib_version

# The floor is a release series, (major, minor): any patch, four-part or local
# release of it passes. NumpyVersion orders "0.12.0.1" before "0.12.0", so it
# cannot express this floor.
MINIMUM_JAX = (0, 12)
_MINIMUM_JAX_TEXT = ".".join(map(str, MINIMUM_JAX))
_RELEASE_SERIES = re.compile(r"(\d+)\.(\d+)")


def _require_jax_version(name: str, version: str) -> None:
    """Reject an unsupported jax or jaxlib before its kernels can load.

    Args:
        name (str): Distribution name for the message, ``"jax"`` or ``"jaxlib"``.
        version (str): Its ``__version__``, for example ``"0.12.0.1"`` or
            ``"0.12.0+cuda12"``; only the leading ``major.minor`` is compared.

    Raises:
        ImportError: The release series is older than ``MINIMUM_JAX`` or the
            version does not start with ``major.minor``.
    """
    series = _RELEASE_SERIES.match(version)
    if series is None or (int(series[1]), int(series[2])) < MINIMUM_JAX:
        raise ImportError(
            f"simsopt_jax requires {name} >= {_MINIMUM_JAX_TEXT}; found {version}. "
            "Native simsopt works without it."
        )


# JAX itself reports an incompatible jaxlib as RuntimeError. Check its static
# version first so unsupported optional installations follow the ImportError gate.
_require_jax_version("jaxlib", jaxlib_version)

import jax

_require_jax_version("jax", jax.__version__)


def enable_float64() -> None:
    """Run JAX in float64 (``jax_enable_x64``), the precision every kernel computes in.

    The one place ``simsopt_jax`` sets it: this module calls it at import, and
    ``apply_jax_runtime_config`` calls it again, so ``set_backend`` restores
    float64 when something disabled it after the import.
    """
    jax.config.update("jax_enable_x64", True)


# Importing simsopt_jax enables float64 for the process, as importing
# simsopt.geo does, here so that no array a kernel module builds at import is
# float32.
enable_float64()
