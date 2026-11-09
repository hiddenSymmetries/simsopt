"""Fresh-process child: native jitted kernels run on the CPU under a CUDA default.

The parent exports ``JAX_PLATFORMS=cuda,cpu`` before this process starts.
"""

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import CurveLength, create_equally_spaced_curves


def main() -> None:
    """Check CurveLength's jitted value and gradient land on the CPU device."""
    assert jax.default_backend() == "gpu"
    cpu = {jax.devices("cpu")[0]}
    curve = create_equally_spaced_curves(
        1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32
    )[0]
    length = CurveLength(curve)
    arclength = curve.incremental_arclength()

    value = length.J()
    arclength_gradient = length.dJ_dl(arclength)
    assert value.devices() == cpu
    assert arclength_gradient.devices() == cpu
    np.testing.assert_allclose(value, np.mean(arclength), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(
        arclength_gradient, np.full_like(arclength, 1.0 / arclength.size)
    )
    assert {d.platform for d in jnp.ones(2).devices()} == {"gpu"}


if __name__ == "__main__":
    main()
