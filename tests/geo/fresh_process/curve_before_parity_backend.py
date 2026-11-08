"""Fresh-process child: a native JAX curve built before backend setup stays float64.

The parent exports ``SIMSOPT_BACKEND_MODE=jax_cpu_parity`` and a compilation
cache directory before this process starts.
"""

import jax
import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import CurveHelical, CurveLength
from simsopt_jax.backend import set_backend
from simsopt_jax_adapters.field import JaxBiotSavart


def main() -> None:
    """Construct the curve, then set the backend and evaluate length and field."""
    curve = CurveHelical(32, 2)
    set_backend("jax", device="cpu", intent="parity")
    field = JaxBiotSavart([Coil(curve, Current(1e5))])
    length_value = CurveLength(curve).J()
    assert jax.config.values["jax_enable_x64"]
    assert np.isfinite(length_value)
    np.testing.assert_allclose(
        length_value,
        np.mean(np.linalg.norm(curve.gammadash(), axis=1)),
        rtol=1e-14,
        atol=1e-14,
    )
    field.set_points(np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]]))
    value = field.B()
    assert value.dtype == np.float64
    assert np.all(np.isfinite(jax.device_get(value)))


if __name__ == "__main__":
    main()
