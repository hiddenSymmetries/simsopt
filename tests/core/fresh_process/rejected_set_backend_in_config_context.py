"""Fresh-process child: a rejected ``set_backend`` inside a JAX configuration
context changes no JAX setting.

The parent starts this process with ``JAX_PLATFORMS=cpu`` and the CUDA
determinism flag in ``XLA_FLAGS``, so a strict GPU request passes the flag
check and is rejected by the platform check: JAX never initialized CUDA. Each
context hides the process-wide value of one setting the request asks to
change. Once the context exits, every JAX setting must be what it was before
the call, and an implicit host-to-device transfer must still work although the
rejected request asked to disallow it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.backend.runtime import get_backend_mode, set_backend

_CONTEXTS = (
    ("transfer guard", lambda: jax.transfer_guard("allow")),
    ("NaN debugging", lambda: jax.debug_nans(False)),
    ("JIT", lambda: jax.disable_jit(False)),
)


def changed_settings(settings: dict[str, object]) -> dict[str, tuple[object, object]]:
    """Every JAX setting that differs from ``settings``, as (before, now) pairs."""
    return {
        setting: (settings[setting], value)
        for setting, value in jax.config.values.items()
        if settings[setting] != value
    }


def main() -> None:
    """Reject one strict GPU request inside each context and compare every setting."""
    set_backend("jax", device="cpu", intent="parity")
    settings = dict(jax.config.values)
    for name, context in _CONTEXTS:
        with context():
            try:
                set_backend(
                    "jax",
                    device="gpu",
                    strict=True,
                    debug_nans=True,
                    disable_jit=True,
                    transfer_guard="disallow",
                )
            except RuntimeError as error:
                assert "Requested JAX platform 'cuda'" in str(error), (name, error)
            else:
                raise AssertionError(f"{name}: the GPU request was not rejected")
        changed = changed_settings(settings)
        assert not changed, (name, changed)
        assert get_backend_mode() == "jax_cpu_parity", (name, get_backend_mode())
    np.testing.assert_array_equal(jnp.ones(2) + np.ones(2), np.full(2, 2.0))


if __name__ == "__main__":
    main()
