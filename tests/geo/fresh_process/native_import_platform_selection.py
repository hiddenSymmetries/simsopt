"""Fresh-process child: importing ``simsopt.geo`` keeps an explicit CPU platform.

The parent sets ``JAX_PLATFORMS=cpu``, ``JAX_PLATFORM_NAME=cpu`` or neither
before this process starts; the ``simsopt.geo`` import below is the event
under test.
"""

import os

import jax
from simsopt import geo  # noqa: F401


def main() -> None:
    """Check the platform name, default backend and x64 after the import."""
    explicit_platform = (
        "JAX_PLATFORMS" in os.environ or "JAX_PLATFORM_NAME" in os.environ
    )
    assert jax.config.values["jax_platform_name"] == (
        "cpu" if not explicit_platform or "JAX_PLATFORM_NAME" in os.environ else ""
    )
    assert jax.default_backend() == "cpu"
    assert jax.config.values["jax_enable_x64"]


if __name__ == "__main__":
    main()
