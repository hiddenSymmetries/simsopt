#!/usr/bin/env python

import os
import numpy as np

import vmec_jax as vj
from booz_xform_jax.jax_api import prepare_booz_xform_constants_from_inputs
from booz_xform_jax.jax_api import booz_xform_jax_impl
from vmec_jax._compat import enable_x64, jax, jnp
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.wout import equilibrium_aspect_ratio_from_state

from simsopt.mhd import make_vmec_jax_residuals_from_terms

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

# Configure quasisymmetry objective:
state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
initial_booz_inputs = vj.booz_xform_inputs_from_state(
    state=state_guess,
    static=static,
    indata=indata,
    signgs=signgs,
    flux=flux,
)
constants, grids = prepare_booz_xform_constants_from_inputs(
    inputs=initial_booz_inputs,
    mboz=mboz,
    nboz=nboz,
    asym=bool(cfg.lasym),
)
surface_indices, surfaces_used = vj.surface_indices_from_static(static, [target_surface])
surface_indices = jnp.asarray(surface_indices, dtype=jnp.int32)

xm_b = np.asarray(grids.xm_b, dtype=int)
xn_b = np.asarray(grids.xn_b, dtype=int) / int(cfg.nfp)
nonsymmetric = xm_b * helicity_n + xn_b * helicity_m != 0
nonsymmetric_indices = jnp.asarray(np.nonzero(nonsymmetric)[0], dtype=jnp.int32)

booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))


def _qs_residuals_from_state(state):
    inputs = vj.booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        flux=flux,
    )
    out = booz_fn(
        rmnc=inputs.rmnc,
        zmns=inputs.zmns,
        lmns=inputs.lmns,
        bmnc=inputs.bmnc,
        bsubumnc=inputs.bsubumnc,
        bsubvmnc=inputs.bsubvmnc,
        iota=inputs.iota,
        xm=inputs.xm,
        xn=inputs.xn,
        xm_nyq=inputs.xm_nyq,
        xn_nyq=inputs.xn_nyq,
        constants=constants,
        grids=grids,
        bmns=inputs.bmns,
        bsubumns=inputs.bsubumns,
        bsubvmns=inputs.bsubvmns,
        surface_indices=surface_indices,
    )
    bmnc_b = out["bmnc_b"][0]
    b00 = jnp.where(jnp.abs(bmnc_b[0]) > 0.0, bmnc_b[0], 1.0)
    return jnp.take(bmnc_b / b00, nonsymmetric_indices)


def _aspect_residuals_from_state(state):
    aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
    return jnp.asarray([aspect - target_aspect], dtype=jnp.float64)


def make_stage1_residual_terms():
    """
    Construct the VMEC/Boozer residual terms for the exact objective.

    Users can add or replace entries here with any JAX-compatible callable
    ``term(state) -> residual_vector``. The default includes aspect ratio and
    nonsymmetric Boozer-spectrum terms.
    """
    return [_aspect_residuals_from_state, _qs_residuals_from_state]


def make_stage1_residuals_fn():
    """
    Return the callable consumed by ``FixedBoundaryExactOptimizer``.
    """
    terms = make_stage1_residual_terms()
    return make_vmec_jax_residuals_from_terms(
        terms,
        n_non_qs=1,
        qs_total_from_state=lambda state: jnp.sum(
            _qs_residuals_from_state(state) ** 2
        ),
    )


residuals_from_state = make_stage1_residuals_fn()

opt = vj.FixedBoundaryExactOptimizer(
    static,
    indata,
    boundary,
    specs,
    residuals_from_state,
)

print("Parameter space:", vj.boundary_param_names(specs))
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
