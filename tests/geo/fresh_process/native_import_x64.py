"""Fresh-process child: importing ``simsopt.geo`` enables x64 despite the environment.

The parent sets ``JAX_PLATFORMS=cpu`` and ``JAX_ENABLE_X64`` to a disabling
value before this process starts. The precondition is checked between the two
imports because it is the state the ``simsopt.geo`` import has to change, so
that import deliberately follows it at module level.
"""

import jax

assert not jax.config.values["jax_enable_x64"], "JAX_ENABLE_X64 did not disable x64"

from simsopt import geo  # noqa: F401


def main() -> None:
    """Check the default backend and x64 after the native import."""
    assert jax.default_backend() == "cpu"
    assert jax.config.values["jax_enable_x64"]


if __name__ == "__main__":
    main()
