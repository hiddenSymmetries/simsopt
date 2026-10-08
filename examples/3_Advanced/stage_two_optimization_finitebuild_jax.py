#!/usr/bin/env python
r"""Finite-build Stage-II coil optimization of ``stage_two_optimization_finitebuild.py`` on JAX.

The problem is the one of ``stage_two_optimization_finitebuild.py``: each coil
is a 2x3 multifilament pack (Singh et al. 2020) whose filaments follow the
coil centroid frame, rotated by a Fourier angle that is optimized along with
the coil shapes and currents. The objective is the quadratic flux of all
filaments on the Landreman-Paul QA surface plus a quadratic penalty on each
base coil length above its initial value and a coil-coil distance penalty on
the pack centerlines. Here every L-BFGS-B evaluation is one jitted JAX program,
``value_and_grad`` of ``fused_stage_two_objective``: it maps the free DOFs to
the centerlines, the frames and filaments, the Biot-Savart field, the flux
integral and the penalties on the active JAX device. The same terms are
drop-in Optimizables (``SquaredFluxJAX``, ``CurveLengthJAX``,
``CurveCurveDistanceJAX``); the script checks them against the fused objective
at the initial point.

Install with ``pip install '.[jax]'``. ``CI=true`` runs 10 iterations. For GPU,
use a CUDA-enabled JAX installation and launch from the repository root::

    XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_exclude_nondeterministic_ops=true" \
        python examples/3_Advanced/stage_two_optimization_finitebuild_jax.py --device gpu

This preserves existing XLA flags and enables deterministic operations before
JAX backend initialization; ``set_backend`` below makes CUDA the default backend.
"""

import argparse
import os
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import minimize

from simsopt.field import Coil, Current, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, create_multifilament_grid
from simsopt.objectives import QuadraticPenalty
from simsopt_jax.backend import set_backend
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    fused_stage_two_objective,
    fused_stage_two_values,
    make_stage_two_problem,
)
from simsopt_jax_adapters.field import BiotSavartJAX
from simsopt_jax_adapters.geo import CurveCurveDistanceJAX, CurveLengthJAX
from simsopt_jax_adapters.objectives import SquaredFluxJAX

parser = argparse.ArgumentParser(description="Finite-build Stage-II optimization on JAX")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
args = parser.parse_args()
set_backend("jax", device=args.device, intent="parity")

# Problem parameters of stage_two_optimization_finitebuild.py.
ncoils = 4
R0 = 1.00
R1 = 0.70
order = 5
LENGTH_PEN = 1e-2
DIST_MIN = 0.1
DIST_PEN = 10
numfilaments_n = 2
numfilaments_b = 3
gapsize_n = 0.02
gapsize_b = 0.04
rot_order = 1
MAXITER = 10 if os.environ.get("CI") else 400

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(
    str(TEST_DIR / "input.LandremanPaul2021_QA"), range="half period", nphi=nphi, ntheta=ntheta
)
nfil = numfilaments_n * numfilaments_b
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = []
for i in range(ncoils):
    curr = Current(1.)
    if i == 0:
        curr.fix_all()
    base_currents.append(curr * (1e5/nfil))
base_curves_finite_build = sum([
    create_multifilament_grid(c, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b, rotation_order=rot_order)
    for c in base_curves], [])
base_currents_finite_build = sum([[c]*nfil for c in base_currents], [])
curves_fb = apply_symmetries_to_curves(base_curves_finite_build, s.nfp, True)
currents_fb = apply_symmetries_to_currents(base_currents_finite_build, s.nfp, True)
# The pack centerlines, which the curve-curve distance penalty uses.
curves = apply_symmetries_to_curves(base_curves, s.nfp, True)
coils_fb = [Coil(c, curr) for (c, curr) in zip(curves_fb, currents_fb)]
bs = BiotSavartJAX(coils_fb)
Jf = SquaredFluxJAX(s, bs)
Jls = [CurveLengthJAX(c) for c in base_curves]
initial_lengths = [J.J() for J in Jls]

# The native script's objective, assembled from the drop-in JAX Optimizables.
JF = Jf \
    + LENGTH_PEN * sum(QuadraticPenalty(Jls[i], initial_lengths[i], "max") for i in range(ncoils)) \
    + DIST_PEN * CurveCurveDistanceJAX(curves, DIST_MIN)

problem = make_stage_two_problem(bs, Jf.fixed_surface_flux_spec(), StageTwoObjectiveConfig(
    num_base_curves=ncoils,
    filaments_per_pack=nfil,
    individual_length_weight=LENGTH_PEN,
    individual_length_targets=tuple(initial_lengths),
    curve_curve_minimum_distance=DIST_MIN,
    curve_curve_weight=DIST_PEN,
    curve_curve_pairs="all",
))
value_and_grad = jax.jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))


def fun(x):
    value, gradient = jax.device_get(value_and_grad(problem, jax.device_put(x)))
    # The native script scales its objective by 1e-4.
    return 1e-4 * float(value), 1e-4 * gradient


initial_dofs = bs.x
initial_value = float(jax.device_get(value_and_grad(problem, jax.device_put(initial_dofs))[0]))
print(f"Initial objective: fused {initial_value:.15e}, drop-in Optimizables {JF.J():.15e}")
res = minimize(fun, initial_dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 400, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)
J, jf, _, max_BdotN, length = jax.device_get(jax.jit(fused_stage_two_values)(problem, jax.device_put(res.x)))
bs.x = res.x
print(f"{res.nit} iterations: J={J:.6e}, Jf={jf:.6e}, max|B·n|={max_BdotN:.3e}, "
      f"coil lengths=[{', '.join(f'{J.J():.3f}' for J in Jls)}]")
assert np.isfinite(J) and J < initial_value
