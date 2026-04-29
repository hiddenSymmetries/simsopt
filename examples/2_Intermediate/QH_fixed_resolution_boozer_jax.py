#!/usr/bin/env python

import os
import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64

from simsopt.mhd import (
    AspectRatioJax,
    BoozerQuasisymmetryResidualJax,
    VmecJaxLeastSquaresProblem,
)

"""
Optimize for quasi-helical symmetry (M=1, N=1) at a given radius.

This example follows 2_Intermediate/QH_fixed_resolution_boozer.py, but
uses vmec_jax and booz_xform_jax. The VMEC Jacobian is vmec_jax's exact
discrete-adjoint Jacobian, and the Boozer residual is differentiated by JAX.
"""

max_nfev = 1  # Maximum number of exact objective/Jacobian evaluations
max_mode = 2  # Maximum poloidal and toroidal mode numbers to vary

target_aspect = 7.0
target_surface = 0.5
helicity_m = 1
helicity_n = 1
mboz = 8
nboz = 8

print("Running 2_Intermediate/QH_fixed_resolution_boozer_jax.py")
print("========================================================")

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

# Configure objective terms in the same tuple style as LeastSquaresProblem.
# Each entry is (objective function, target, weight), so users can add or
# replace terms without changing the optimizer.
aspect = AspectRatioJax()
quasisymmetry = BoozerQuasisymmetryResidualJax(
    surfaces=[target_surface],
    helicity_m=helicity_m,
    helicity_n=helicity_n,
    mboz=mboz,
    nboz=nboz,
)
objective_tuple = [
    (aspect.value_from_state(static), target_aspect, 1.0),
    (quasisymmetry.residuals_from_state(static, indata), 0.0, 1.0),
]
stage1_objective = VmecJaxLeastSquaresProblem.from_tuples(objective_tuple)

opt = vj.FixedBoundaryExactOptimizer(
    static,
    indata,
    boundary,
    specs,
    stage1_objective.residuals_from_state,
)

print("Parameter space:", vj.boundary_param_names(specs))
_surface_indices, surfaces_used = quasisymmetry.surface_indices(static)
print("Boozer target surface:", float(surfaces_used[0]))
print("Quasisymmetry objective before optimization:", opt.quasisymmetry_objective(params0))

# To keep this example fast, we stop after the first function evaluation. For
# a production optimization, increase max_nfev and tighten the tolerances.
result = opt.run(
    params0,
    method="scipy",
    max_nfev=max_nfev,
    ftol=1e-3,
    gtol=1e-3,
    xtol=1e-3,
    target_aspect=target_aspect,
)

print("Final aspect ratio is", opt.aspect_ratio(result["x"]))
print("Quasisymmetry objective after optimization:", opt.quasisymmetry_objective(result["x"]))

opt.save_input("input.QH_fixed_resolution_boozer_jax_final", result["x"])

print("End of 2_Intermediate/QH_fixed_resolution_boozer_jax.py")
print("=======================================================")
