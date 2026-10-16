"""Fresh-process child: ``set_backend`` overrides the platform ``simsopt.geo`` chose.

Usage: ``set_backend_after_native_import.py DEVICE PLATFORMS_AFTER_IMPORT``.
The parent exports no JAX platform variable, so importing ``simsopt.geo`` sets
the legacy ``jax_platform_name="cpu"``. A non-empty ``PLATFORMS_AFTER_IMPORT``
is exported to ``JAX_PLATFORMS`` only after that import, as a driver would.
"""

import os
import sys

import jax
from simsopt import geo  # noqa: F401
from simsopt_jax.backend import set_backend
from simsopt_jax.backend.runtime import JaxDevice

_DEVICES: dict[str, JaxDevice] = {"cpu": "cpu", "gpu": "gpu"}
_JAX_PLATFORMS = {"cpu": "cpu", "gpu": "cuda"}


def main(device: str, platforms_after_import: str) -> None:
    """Select ``device`` and check that it becomes JAX's default backend."""
    assert jax.config.values["jax_platform_name"] == "cpu"
    if platforms_after_import:
        os.environ["JAX_PLATFORMS"] = platforms_after_import
    set_backend("jax", device=_DEVICES[device], intent="parity")
    assert jax.config.values["jax_platforms"] == (
        platforms_after_import or _JAX_PLATFORMS[device]
    )
    assert jax.config.values["jax_platform_name"] == ""
    assert jax.default_backend() == device


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
