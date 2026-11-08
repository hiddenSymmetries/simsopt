"""Fresh-process child: ``set_backend`` rejected by JAX's default-backend lookup
changes no JAX setting.

The parent starts this process with ``JAX_PLATFORMS=cpu``. JAX initializes the
CPU backend through a platform-specific lookup, so no default-backend lookup
has run, and the deprecated public ``jax_xla_backend`` then names a platform
JAX never initialized. The CPU request passes the platform check, and JAX
raises when the default backend is looked up. x64 is off, NaN debugging is off
and transfers are allowed before the call, which asks to change all three;
afterwards every JAX setting must be what it was.
"""

import jax
from rejected_set_backend_in_config_context import changed_settings
from simsopt_jax.backend.runtime import set_backend


def main() -> None:
    """Reject one CPU request at the default-backend lookup and compare every setting."""
    jax.devices("cpu")
    jax.config.update("jax_xla_backend", "tpu")
    jax.config.update("jax_enable_x64", False)
    settings = dict(jax.config.values)
    try:
        set_backend("jax", device="cpu", debug_nans=True, transfer_guard="disallow")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the default-backend lookup did not reject the request")
    changed = changed_settings(settings)
    assert not changed, changed


if __name__ == "__main__":
    main()
