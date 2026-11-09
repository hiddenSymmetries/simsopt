"""Fresh-process child: explicit ``set_backend`` applies the environment's settings.

The parent exports ``JAX_PLATFORMS=cpu``, ``SIMSOPT_DEBUG``, the JAX
diagnostics, compilation-cache and field tiling variables before this process
starts.
"""

import os

import jax
import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves
from simsopt_jax.backend import get_field_kernel_tuning, set_backend
from simsopt_jax.runtime.host_boundary import allow_host_transfers, host_array
from simsopt_jax_adapters.field import JaxBiotSavart


def main() -> None:
    """Set the backend, check the applied settings, then evaluate a CPU field."""
    config = set_backend("jax", device="cpu", intent="parity")
    debug = os.environ["SIMSOPT_DEBUG"] == "true"
    assert config.debug_nans and config.disable_jit
    assert config.strict == debug
    assert jax.config.values["jax_debug_nans"] and jax.config.values["jax_disable_jit"]
    assert jax.config.values["jax_transfer_guard"] == ("disallow" if debug else "allow")
    assert jax.config.values["jax_enable_x64"]
    assert (
        jax.config.values["jax_compilation_cache_dir"]
        == os.environ["SIMSOPT_JAX_COMPILATION_CACHE_DIR"]
    )
    tuning = get_field_kernel_tuning()
    assert (
        tuning.coil_chunk_size,
        tuning.quadrature_block_size,
        tuning.point_chunk_size,
    ) == ((0, 0, 0) if debug else (2, 8, 1))
    curve = create_equally_spaced_curves(1, 1, False, order=1, numquadpoints=16)[0]
    field = JaxBiotSavart([Coil(curve, Current(1e5))])
    field.set_points(np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]]))
    # Eager JAX indexing stages scalar indices; explicitly permit debug evaluation.
    if debug:
        with allow_host_transfers():
            value = field.B()
    else:
        value = field.B()
    assert jax.config.values["jax_transfer_guard"] == ("disallow" if debug else "allow")
    assert value.dtype == np.float64
    assert all(device.platform == "cpu" for device in value.devices())
    with allow_host_transfers():
        assert np.all(np.isfinite(host_array(value)))


if __name__ == "__main__":
    main()
