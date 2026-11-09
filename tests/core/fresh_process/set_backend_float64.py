"""Fresh-process child: ``set_backend`` runs JAX in float64 again after
something disabled ``jax_enable_x64`` once ``simsopt_jax`` was imported.

The parent sets ``JAX_PLATFORMS=cpu``. Before each CPU mode is selected, x64
is turned off, as another library in the process might do; afterwards the raw
Biot-Savart kernel must match the analytic on-axis loop field to float64
precision.
"""

import jax
from raw_kernel_float64 import check_on_axis_loop_field
from simsopt_jax.backend.runtime import set_backend


def main() -> None:
    """Disable x64, select each CPU mode, and check the field precision."""
    for intent in ("parity", "fast"):
        jax.config.update("jax_enable_x64", False)
        set_backend("jax", device="cpu", intent=intent)
        assert jax.config.values["jax_enable_x64"] is True, intent
        check_on_axis_loop_field()


if __name__ == "__main__":
    main()
