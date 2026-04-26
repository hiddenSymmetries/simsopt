#!/usr/bin/env python

import os
import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64

"""
Optimize a VMEC-JAX equilibrium for quasi-helical symmetry (M=1, N=-1)
throughout the volume.

This example follows 2_Intermediate/QH_fixed_resolution.py, but uses
vmec_jax's exact discrete-adjoint Jacobian instead of finite differences.
"""

max_nfev = 10  # Maximum number of exact objective/Jacobian evaluations
max_mode = 1  # Maximum poloidal and toroidal mode numbers to vary

target_aspect = 7.0
surfaces = np.arange(0, 1.01, 0.1)
helicity_m = 1
helicity_n = -1

print("Running 2_Intermediate/QH_fixed_resolution_jax.py")
print("=================================================")

enable_x64(True)

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start')
cfg, indata = vj.load_config(filename)
static = vj.build_static(cfg)
boundary = vj.boundary_from_indata(indata, static.modes)

# Define parameter space:
specs = vj.boundary_param_specs(
    boundary,
    static.modes,
    max_mode=max_mode,
    min_coeff=0.0,
    include=("rc", "zs"),
    fix=("rc00",),
)
params0 = np.zeros(len(specs))

print("Parameter space:", vj.boundary_param_names(specs))

# Configure quasisymmetry objective:
residuals_fn = vj.make_qh_residuals_fn(
    static,
    indata,
    helicity_m=helicity_m,
    helicity_n=helicity_n,
    target_aspect=target_aspect,
    surfaces=surfaces,
)

opt = vj.FixedBoundaryExactOptimizer(
    static,
    indata,
    boundary,
    specs,
    residuals_fn,
)

print("Quasisymmetry objective before optimization:", opt.quasisymmetry_objective(params0))
print("Aspect ratio before optimization:", opt.aspect_ratio(params0))

# To keep this example fast, we stop after max_nfev evaluations. For a
# production optimization, increase max_nfev and tighten the tolerances.
result = opt.run(
    params0,
    method="scipy",
    max_nfev=max_nfev,
    ftol=1e-3,
    gtol=1e-3,
    xtol=1e-3,
    target_aspect=target_aspect,
)

print("Final aspect ratio:", opt.aspect_ratio(result["x"]))
print("Quasisymmetry objective after optimization:", opt.quasisymmetry_objective(result["x"]))

opt.save_input("input.QH_fixed_resolution_jax_final", result["x"])

print("End of 2_Intermediate/QH_fixed_resolution_jax.py")
print("================================================")
