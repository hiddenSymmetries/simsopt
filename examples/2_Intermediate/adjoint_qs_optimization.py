"""
Adjoint-based quasisymmetry optimization using VacuumAdjointQS.

Optimizes the plasma-boundary Fourier coefficients of a single-volume
SPEC equilibrium to reduce QS violation and drive the rotational transform
toward a target value, using analytical adjoint gradients (O(1) SPEC solves
per gradient evaluation rather than O(N_dof)).

Usage::

    python adjoint_qs_optimization.py [vacuum.sp]

The first argument is a SPEC input file; if omitted the script looks for
``vacuum.sp`` in the current directory.  Set ``SPEC_ADJOINT`` in your
environment (or edit the variable below) to point to the SPEC binary built
with Lconstraint=-2 support.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize

from simsopt.mhd import Spec, VacuumAdjointQS

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SPEC_INPUT = sys.argv[1] if len(sys.argv) > 1 else "vacuum.sp"
SPEC_ADJOINT = os.environ.get("SPEC_ADJOINT", None)   # path to adjoint SPEC binary
NFP = 3          # field periods (must match the input file)
HELICITY_M = 1   # QS helicity integers: M=1, N=0 → quasi-axisymmetry
HELICITY_N = 0
IOTA_TARGET = -0.42   # target rotational transform (set to None to disable)
IOTA_WEIGHT = 5.0
QS_WEIGHT = 1.0
MPOL_ADJ = 6     # resolution of the adjoint PDE solvers
NTOR_ADJ = 6
NDISCRETE = 10   # grid refinement factor (ntheta = 4*mpol*ndiscrete etc.)

# ------------------------------------------------------------------
# Set up SPEC instance
# ------------------------------------------------------------------
spec = Spec(SPEC_INPUT)

# Free the m=1 boundary modes only (a small representative DOF set)
spec.boundary.fix_all()
for m in range(1, 4):
    for n in range(-2, 3):
        spec.boundary.unfix(f"rc({m},{n})")
        if not (m == 0 and n == 0):
            spec.boundary.unfix(f"zs({m},{n})")

print(f"Optimising {spec.boundary.dof_size} boundary DOFs")

# ------------------------------------------------------------------
# Construct objective
# ------------------------------------------------------------------
qs = VacuumAdjointQS(
    spec,
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    iota_target=IOTA_TARGET,
    iota_weight=IOTA_WEIGHT,
    qs_weight=QS_WEIGHT,
    spec_adjoint_executable=SPEC_ADJOINT,
    mpol_adj=MPOL_ADJ,
    ntor_adj=NTOR_ADJ,
    ndiscrete=NDISCRETE,
)

# ------------------------------------------------------------------
# Run a few gradient steps with L-BFGS-B
# ------------------------------------------------------------------
history = []

def objective(x):
    spec.boundary.x = x
    J = qs.J()
    g = qs.dJ()
    history.append(J)
    print(f"  iter {len(history):4d}  J = {J:.6e}  iota = {qs.iota:.6f}")
    return float(J), g.astype(float)


x0 = spec.boundary.x.copy()
result = minimize(
    objective,
    x0,
    jac=True,
    method='L-BFGS-B',
    options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8},
)

print(f"\nOptimisation finished: success={result.success}, "
      f"J_final={result.fun:.6e}")
print(f"Initial J = {history[0]:.6e}, final J = {history[-1]:.6e}, "
      f"reduction = {history[0] / history[-1]:.1f}x")
