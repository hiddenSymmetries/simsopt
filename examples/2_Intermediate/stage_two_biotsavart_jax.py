"""Native stage-II objectives and scipy with a JAX Biot-Savart field.

Install with ``pip install '.[jax]'``. Run with ``CI=true`` for two iterations.
For GPU, use a CUDA-enabled JAX installation and launch from the repository root::

    XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_exclude_nondeterministic_ops=true" \
        python examples/2_Intermediate/stage_two_biotsavart_jax.py --device gpu

This preserves existing XLA flags and enables deterministic operations before
JAX backend initialization; ``set_backend`` below makes CUDA the default
backend. C++ curve length mean and norm VJP kernels use explicit transfers: to
the CPU device when ``JAX_PLATFORMS=cuda,cpu`` is also exported, otherwise to
the GPU. ``--device cpu`` is the default.
This adapter supports Python objectives and derivatives. It is not a
simsoptpp.MagneticField: tracing,
InterpolatedField and native field arithmetic require native BiotSavart.
"""

import argparse
import os

import numpy as np
from scipy.optimize import minimize

from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import SquaredFlux
from simsopt_jax_adapters.field import BiotSavartJAX
from simsopt_jax.backend import set_backend


parser = argparse.ArgumentParser(description="Native stage-II objectives with JAX")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
args = parser.parse_args()
set_backend("jax", device="gpu" if args.device == "gpu" else "cpu", intent="parity")

surface = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
surface.set_rc(0, 0, 1.0)
surface.set_rc(1, 0, 0.3)
surface.set_zs(1, 0, 0.3)
curves = create_equally_spaced_curves(
    2, surface.nfp, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32
)
currents = [Current(1e5) for _ in curves]
currents[0].fix_all()
coils = coils_via_symmetries(curves, currents, surface.nfp, stellsym=True)

# The one-line swap from the native script: bs = BiotSavart(coils).
bs = BiotSavartJAX(coils)
bs.set_points(surface.gamma().reshape((-1, 3)))
flux = SquaredFlux(surface, bs)
objective = flux + 1e-5 * sum(CurveLength(c) for c in curves)


def value_and_gradient(dofs):
    objective.x = dofs
    return objective.J(), objective.dJ()


initial_value, initial_gradient = value_and_gradient(objective.x)
assert np.isfinite(initial_value)
assert np.all(np.isfinite(initial_gradient))
result = minimize(
    value_and_gradient,
    objective.x,
    jac=True,
    method="L-BFGS-B",
    options={"maxiter": 2 if os.environ.get("CI") else 100, "maxcor": 10},
)
assert np.isfinite(result.fun)
assert result.fun <= initial_value
print(f"SquaredFlux + length: {initial_value:.6e} -> {result.fun:.6e}")
