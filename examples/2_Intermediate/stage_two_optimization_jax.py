#!/usr/bin/env python
r"""Stage-II coil optimization of ``stage_two_optimization.py`` on JAX.

The objective is the one of ``stage_two_optimization.py``: quadratic flux on the
Landreman-Paul QA surface plus coil length, coil-coil distance, coil-surface
distance, curvature and mean-squared-curvature penalties. Here every L-BFGS-B
evaluation is one jitted JAX program, ``value_and_grad`` of
``fused_stage_two_objective``: it maps the free coil DOFs to coil geometry, the
Biot-Savart field, the flux integral and the penalties on the active JAX device.
The same terms are drop-in Optimizables (``SquaredFluxJAX``, ``CurveLengthJAX``,
...); the script checks them against the fused objective at the initial point.

Install with ``pip install '.[jax]'``. ``CI=true`` runs 10 iterations per stage.
For GPU, use a CUDA-enabled JAX installation and launch from the repository root::

    XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_exclude_nondeterministic_ops=true" \
        python examples/2_Intermediate/stage_two_optimization_jax.py --device gpu

This preserves existing XLA flags and enables deterministic operations before
JAX backend initialization; ``set_backend`` below makes CUDA the default backend.
"""

import argparse
import os
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import minimize

from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import QuadraticPenalty
from simsopt_jax.backend import set_backend
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    fused_stage_two_objective,
    fused_stage_two_values,
    make_stage_two_problem,
)
from simsopt_jax_adapters.field import BiotSavartJAX
from simsopt_jax_adapters.geo import (
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureJAX,
    MeanSquaredCurvatureJAX,
)
from simsopt_jax_adapters.objectives import SquaredFluxJAX

parser = argparse.ArgumentParser(description="Stage-II optimization with a fused JAX objective")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
args = parser.parse_args()
set_backend("jax", device=args.device, intent="parity")

# Problem parameters of stage_two_optimization.py.
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 5
LENGTH_WEIGHT = 1e-6
CC_THRESHOLD = 0.1
CC_WEIGHT = 1000
CS_THRESHOLD = 0.3
CS_WEIGHT = 10
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6
MAXITER = 10 if os.environ.get("CI") else 400

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(
    str(TEST_DIR / "input.LandremanPaul2021_QA"), range="half period", nphi=nphi, ntheta=ntheta
)
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavartJAX(coils)
Jf = SquaredFluxJAX(s, bs)

# The native script's objective, assembled from the drop-in JAX Optimizables.
JF = Jf \
    + LENGTH_WEIGHT * sum(CurveLengthJAX(c) for c in base_curves) \
    + CC_WEIGHT * CurveCurveDistanceJAX(curves, CC_THRESHOLD, num_basecurves=ncoils) \
    + CS_WEIGHT * CurveSurfaceDistanceJAX(curves, s, CS_THRESHOLD) \
    + CURVATURE_WEIGHT * sum(LpCurveCurvatureJAX(c, 2, CURVATURE_THRESHOLD) for c in base_curves) \
    + MSC_WEIGHT * sum(QuadraticPenalty(MeanSquaredCurvatureJAX(c), MSC_THRESHOLD, "max") for c in base_curves)


def stage_two_problem(length_weight):
    """The fused objective's operands; rebuilt problems reuse the compiled programs."""
    return make_stage_two_problem(bs, Jf.fixed_surface_flux_spec(), StageTwoObjectiveConfig(
        num_base_curves=ncoils,
        length_weight=length_weight,
        curve_curve_minimum_distance=CC_THRESHOLD,
        curve_curve_weight=CC_WEIGHT,
        curve_surface_minimum_distance=CS_THRESHOLD,
        curve_surface_weight=CS_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        curvature_weight=CURVATURE_WEIGHT,
        mean_squared_curvature_threshold=MSC_THRESHOLD,
        mean_squared_curvature_weight=MSC_WEIGHT,
    ))


value_and_grad = jax.jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
diagnostics = jax.jit(fused_stage_two_values)


def optimize(dofs, length_weight):
    problem = stage_two_problem(length_weight)

    def fun(x):
        value, gradient = jax.device_get(value_and_grad(problem, jax.device_put(x)))
        return float(value), gradient

    res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                   options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    J, jf, _, max_BdotN, length = jax.device_get(diagnostics(problem, jax.device_put(res.x)))
    print(f"{res.nit} iterations: J={J:.6e}, Jf={jf:.6e}, max|B·n|={max_BdotN:.3e}, "
          f"sum of base coil lengths={length:.3f}")
    return res.x, float(J)


initial_dofs = bs.x
initial_value = float(jax.device_get(value_and_grad(stage_two_problem(LENGTH_WEIGHT), jax.device_put(initial_dofs))[0]))
print(f"Initial objective: fused {initial_value:.15e}, drop-in Optimizables {JF.J():.15e}")
dofs, J_short = optimize(initial_dofs, LENGTH_WEIGHT)
# As in the native script, rerun with a ten times smaller length weight.
dofs, J_long = optimize(dofs, 0.1 * LENGTH_WEIGHT)
bs.x = dofs
assert np.isfinite(J_long) and J_short < initial_value
