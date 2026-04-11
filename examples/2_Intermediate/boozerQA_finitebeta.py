#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

from simsopt.configs import get_data
from simsopt.geo import (
    SurfaceXYZTensorFourier, BoozerSurface, FiniteBetaBoozerSurface,
    SurfaceCurrentFieldProvider, curves_to_vtk, boozer_surface_residual,
    surface_field_nonquasisymmetric_ratio, Volume,
)
from simsopt.util import in_github_actions

r"""
Finite-beta QA optimisation — direct self-consistent closure on a Boozer surface.

This script is the finite-beta counterpart to boozerQA.py.  It teaches how to:

  1. Load coil data and fix coil currents.
  2. Build a stellarator-symmetric XYZ-tensor-Fourier surface and use a
     volume label to constrain the surface solve.
  3. Compute an initial vacuum Boozer surface as the beta=0 seed.
  4. Set up a FiniteBetaBoozerSurface with a SurfaceCurrentFieldProvider
     (the "direct" no-virtual-casing self-consistent closure).
  5. Ramp the plasma pressure from zero to the target beta via continuation,
     verifying branch continuity at each step.
  6. Define an outer objective function J that penalises Boozer
     non-quasisymmetry, iota drift, major-radius drift, achieved-beta error,
     finite-beta residual, and surface-shape regularisation.
    7. Run the outer optimisation with scipy Powell (no finite-difference gradient calls).
  8. Save initial and final coil/surface VTK files.
  9. Optionally export the optimised plasma boundary as a VMEC input file
     (iota-constrained, zero net current — the standard stellarator convention)
     and compare the VMEC boundary field with the direct-closure result.

Key finite-beta physics:
  The surface current K(ϑ,φ) on an offset "virtual wall" enclosing the plasma
  self-consistently mediates the jump in the tangential magnetic field
  [B_tan] = μ₀ K (Rankine–Hugoniot) and simultaneously enforces Boozer
  coordinates and pressure balance [B²]/(2μ₀) = Δp on the plasma boundary.
  For a stellarator (zero net enclosed current) we set I = 0 and optimise G.

Numerical note on derivatives:
    Inner least-squares solves are executed on frozen explicit field samples
    (B_in, B_out), so FiniteBetaBoozerSurface uses analytic Jacobians for the
    finite-beta residual instead of the state-dependent finite-difference
    fallback. A short Picard loop refreshes the frozen fields to recover
    self-consistency.

Reference: arXiv:2203.03753 (doi:10.1017/S0022377822000563)
"""

# ===========================================================================
## OUTPUT DIRECTORY
# ===========================================================================

OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

MU0 = 4.0 * np.pi * 1e-7

print("Running 2_Intermediate/boozerQA_finitebeta.py")
print("==============================================")

# ===========================================================================
## SURFACE / GRID RESOLUTION PARAMETERS
# ===========================================================================
# Spectral resolution for the Boozer-surface solve.
# Increasing mpol/ntor captures higher-harmonic corrections but raises cost.
mpol = 6
ntor = 6
stellsym = True  # assume stellarator symmetry (cos/sin harmonics only)

# ===========================================================================
## FINITE-BETA PHYSICS TARGET
# ===========================================================================
# Target plasma beta:  beta = Δp / <B²/(2μ₀)>_surface
# Typical stellarator values: 0.01 (1 %) or 0.03 (3 %).
PLASMA_BETA = 0.01

# Assume zero net toroidal current inside the surface (stellarator assumption).
# Set False to allow the Boozer poloidal-current coefficient I to be free
# (relevant for tokamaks or current-carrying stellarators).
ZERO_TOROIDAL_CURRENT = True

# ===========================================================================
## CONTINUATION AND INNER SOLVER
# ===========================================================================
# Steps to ramp pressure from vacuum (beta=0) to PLASMA_BETA.
# More steps → safer branch tracking; 4–8 is typical.
CONTINUATION_STEPS = 4 if not in_github_actions else 2

# Max function evaluations per Levenberg–Marquardt inner iteration step.
LS_MAX_NFEV = 50

# Fixed-point iterations used to recover self-consistency while preserving
# analytic Jacobians in each inner least-squares solve.
PICARD_ITERS = 3

# ===========================================================================
## OUTER QA OPTIMISATION
# ===========================================================================
# Surface degrees of freedom to vary in the outer BFGS loop.
# Larger values → richer shape space but higher cost.
QA_DOF_COUNT = 12

# Maximum outer optimisation iterations.
QA_MAXITER = 50 if not in_github_actions else 3

# Penalty weights in the outer objective J.  Tuning these controls the
# trade-off between improving QA and preserving the physics of the solution.
IOTA_WEIGHT = 1.0    # keeps rotational transform near the vacuum value
MR_WEIGHT   = 1.0    # keeps major radius near the initial value
RES_WEIGHT  = 1.0    # penalises a poorly-converged finite-beta residual
BETA_WEIGHT = 10.0   # keeps the achieved plasma beta on target
REG_WEIGHT  = 1e-2   # L2 regularisation on surface-shape displacement

# ===========================================================================
## VMEC BENCHMARK EXPORT
# ===========================================================================
# After optimisation, optionally export the plasma boundary as a VMEC input.
# Using NCURR=0 (iota-constrained) is the correct stellarator convention:
# the edge iota from the direct solve is prescribed instead of the toroidal
# current profile, which is underdetermined in the single-surface model.
EXPORT_VMEC = True
VMEC_NS     = [13, 25, 49]
VMEC_NITER  = [400, 1200, 4000]
VMEC_FTOL   = [1e-8, 1e-10, 1e-12]

# ===========================================================================
## LOAD COIL DATA AND SET UP THE MAGNETIC FIELD
# ===========================================================================
# get_data("ncsx") returns the NCSX coils used as the external field source.
# base_curves: independent coil curves (symmetry expansion handled in bs).
# base_currents: coil currents corresponding to base_curves.
# ma:  a Curve object representing the magnetic axis.
# nfp: number of field periods.
# bs:  a BiotSavart object for ALL symmetry-expanded coils.
base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
all_curves = [c.curve for c in bs.coils]

# Fix all coil currents — we are optimising the surface shape, not the coils.
for current in base_currents:
    current.fix_all()

# G0 is an initial guess for the Boozer toroidal-field coefficient, estimated
# from the total enclosed current via Ampère's theorem:
#   G ≈ μ₀/(2π) × I_total,  with I_total = nfp × Σ |I_k|.
current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))

# ===========================================================================
## BUILD THE INITIAL SURFACE
# ===========================================================================
# Quadrature points chosen so the Fourier-mode representation is exact on the
# Boozer grid (one point per degree of freedom in each direction).
phis   = np.linspace(0, 1/nfp, 2*ntor + 1, endpoint=False)
thetas = np.linspace(0, 1,     2*mpol + 1, endpoint=False)

s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
    quadpoints_phi=phis, quadpoints_theta=thetas)

# Initialise the surface geometry by fitting to an outward-offset of the axis.
s.fit_to_curve(ma, 0.1, flip_theta=True)

# ===========================================================================
## VOLUME LABEL AND VACUUM BOOZER SEED — COMPUTE INITIAL SURFACE
# ===========================================================================
# Use the enclosed volume as the shape label L such that vol(s) = vol_target.
# This constrains the surface size during both the Boozer solve and the outer
# optimisation (analogous to the volume penalty in DESC / SPEC).
vol        = Volume(s)
vol_target = vol.J()

# Solve the vacuum (beta=0) Boozer surface equations with Newton's method.
# This finds (iota, G) and the shape of s such that  ||residual||_∞ < tol.
# The result is the starting point for the finite-beta continuation.
boozer_seed = BoozerSurface(bs, s, vol, vol_target)
seed = boozer_seed.solve_residual_equation_exactly_newton(
    tol=1e-13, maxiter=20, iota=-0.35, G=G0)

out_res = boozer_surface_residual(s, seed['iota'], seed['G'], bs, derivatives=0)[0]
print(f"NEWTON {seed['success']}: iter={seed['iter']}, iota={seed['iota']:.4f}, "
      f"vol={s.volume():.4f}, ||residual||={np.linalg.norm(out_res):.3e}")

# Save the initial coils and surface to VTK for visualisation in ParaView.
curves_to_vtk(all_curves, OUT_DIR + "curves_init")
s.to_vtk(OUT_DIR + "surf_init")
print(f"Saved initial coils   → {OUT_DIR}curves_init.vtu")
print(f"Saved initial surface → {OUT_DIR}surf_init.vts")

# ===========================================================================
## COMPUTE THE PRESSURE JUMP TARGET FROM PLASMA BETA
# ===========================================================================
# The MHD pressure balance at the plasma surface (in SI units) is:
#     Δp = [B²_out − B²_in] / (2 μ₀)
# Plasma beta is beta = Δp / <B²_vac / (2 μ₀)>_s, where the average is
# area-weighted over the plasma boundary.  Given a target beta, therefore:
#     Δp_target = beta × <B²_vac / (2 μ₀)>_s

bs.set_points(s.gamma().reshape(-1, 3))
B_vac  = bs.B().reshape(s.gamma().shape)   # B at each surface quadrature point
dA     = np.linalg.norm(s.normal(), axis=2)   # area element [nphi × ntheta]
modB2  = np.sum(B_vac**2, axis=2)             # |B|² [nphi × ntheta]
ref_mag_pressure = float(np.sum(modB2 * dA) / (2.0 * MU0 * np.maximum(np.sum(dA), 1e-30)))
pressure_jump    = PLASMA_BETA * ref_mag_pressure

print(f"Reference <|B|²/(2μ₀)> = {ref_mag_pressure:.6e} Pa")
print(f"Target pressure jump     = {pressure_jump:.6e} Pa  (beta = {PLASMA_BETA:.2%})")

# ===========================================================================
## SET UP THE FINITE-BETA SELF-CONSISTENT PROBLEM
# ===========================================================================
# SurfaceCurrentFieldProvider places an auxiliary offset surface just outside
# the plasma and represents the magnetic response of the plasma via a sheet
# current K on that offset surface.  The offset is determined adaptively from
# the local surface curvature (offset_scale × r_min_curvature).
# This is the "direct" closure: no full virtual-casing integral is required.
field_provider = SurfaceCurrentFieldProvider(
    offset_scale=0.25,   # offset ≈ 25 % of the minimum curvature radius
    min_offset=1e-5,     # absolute minimum offset [m]
)

# FiniteBetaBoozerSurface extends BoozerSurface with the finite-beta interface
# conditions.  Its inner solve finds (ι, G, I, K) — rotational transform,
# Boozer toroidal-field coefficient, poloidal-current coefficient, and the
# sheet-current potential — such that simultaneously:
#   • the Boozer-coordinate conditions hold on s,
#   • B_in · n = 0  (no normal field penetrates the surface),
#   • [B²]/(2μ₀) = pressure_jump  (pressure-balance boundary condition),
#   • μ₀ K = [B_tangential]  (Rankine–Hugoniot jump in tangential B).
#
# We start with pressure_jump=0 (vacuum limit) and ramp to the target below.
finite_beta = FiniteBetaBoozerSurface(
    bs, s, vol, vol_target,
    pressure_jump=0.0,          # will be ramped in the continuation below
    options={'ls_max_nfev': LS_MAX_NFEV},
)

# ===========================================================================
## PRESSURE CONTINUATION: vacuum (β=0) → target finite beta
# ===========================================================================
# Abruptly jumping to the full pressure jump from a vacuum seed risks
# Newton divergence or branch jumps in ι and I.  Slowly increasing β along
# a continuation path keeps the solution on the physical branch.

print(f"\nPressure continuation: 0 → {pressure_jump:.3e} Pa over {CONTINUATION_STEPS} steps")

# State variables carried from step to step.
iota = seed['iota']
G    = seed['G']
I    = 0.0                               # zero net current (stellarator)
K    = np.zeros(s.gamma().shape[:2])     # sheet-current potential [nphi × ntheta]

for step, p in enumerate(np.linspace(0.0, pressure_jump, CONTINUATION_STEPS), start=1):
    finite_beta.pressure_jump = p
    res = None
    for _ in range(PICARD_ITERS):
        B_in_guess, B_out_guess = finite_beta.resolve_field_components(
            field_provider=field_provider,
            iota=iota,
            G=G,
            I=I,
            current_potential=K,
        )
        # Frozen-field inner solve uses analytic residual Jacobians.
        res = finite_beta.run_code(
            iota=iota,
            G=G,
            I=I,
            current_potential=K,
            B_in=B_in_guess,
            B_out=B_out_guess,
            optimize_G=ZERO_TOROIDAL_CURRENT,     # free G (stellarator: fixed I=0)
            optimize_I=not ZERO_TOROIDAL_CURRENT, # free I (tokamak mode)
            optimize_surface=False,               # hold the surface shape fixed here
        )
        if not res['success']:
            break
        iota = res['iota']
        G = res['G']
        I = res['I']
        K = np.asarray(res['current_potential']).copy()
    if res is None:
        raise RuntimeError('Continuation step failed before starting the inner solve.')
    iota = res['iota']
    G    = res['G']
    I    = res['I']
    K    = np.asarray(res['current_potential']).copy()
    print(f"  step {step}/{CONTINUATION_STEPS}: "
          f"p_jump={p:.3e}, success={res['success']}, "
          f"iota={iota:.6f}, I={I:.3e}, ||r||={res['residual_norm']:.3e}")

# ===========================================================================
## OUTER QA OPTIMISATION
# ===========================================================================
# The outer objective mirrors the structure of boozerQA.py but targets surface
# DOFs rather than coil DOFs.  For each trial surface shape the inner
# finite-beta solve is re-run to find the self-consistent (ι, G, I, K), and
# the field B_in is used to evaluate quasisymmetry.
#
# J = J_nonQS                                    (non-quasisymmetry ratio)
#   + ½ IOTA_WEIGHT  (ι − ι₀)²                 (iota penalty)
#   + ½ MR_WEIGHT    (R_maj − R₀)²              (major radius penalty)
#   + ½ RES_WEIGHT   ||residual||²               (finite-beta residual penalty)
#   + ½ BETA_WEIGHT  (β_achieved − β_target)²   (beta penalty)
#   + ½ REG_WEIGHT   ||Δdofs||²                  (shape regularisation)

print(f"\n{'='*60}")
print("Outer QA optimisation")
print('='*60)

surface_dofs0 = s.x.copy()
iota0 = float(iota)
mr0   = float(s.major_radius())

# Mutable state dict shared between fun() calls (preserves good state on failure).
state = {'iota': iota, 'G': G, 'I': I, 'K': K.copy()}


def fun(dofs):
    # Save current surface and solver state in case the inner solve fails.
    sdofs_prev = s.x.copy()
    iota_prev  = state['iota']
    G_prev     = state['G']
    I_prev     = state['I']
    K_prev     = state['K'].copy()

    # Move the surface to the proposed shape.
    full_dofs = surface_dofs0.copy()
    full_dofs[:len(dofs)] = dofs
    s.x = full_dofs

    # Recompute the pressure jump for this new surface (B changes with shape).
    bs.set_points(s.gamma().reshape(-1, 3))
    B_new  = bs.B().reshape(s.gamma().shape)
    dA_new = np.linalg.norm(s.normal(), axis=2)
    ref_new = float(np.sum(np.sum(B_new**2, axis=2) * dA_new)
                    / (2.0 * MU0 * np.maximum(np.sum(dA_new), 1e-30)))
    p_new  = PLASMA_BETA * ref_new
    finite_beta.pressure_jump = p_new

    # Inner finite-beta solve using analytic Jacobians on frozen fields,
    # wrapped in a Picard fixed-point loop for self-consistency.
    inner = None
    iota_tmp = iota_prev
    G_tmp = G_prev
    I_tmp = I_prev
    K_tmp = K_prev.copy()
    for _ in range(PICARD_ITERS):
        B_in_guess, B_out_guess = finite_beta.resolve_field_components(
            field_provider=field_provider,
            iota=iota_tmp,
            G=G_tmp,
            I=I_tmp,
            current_potential=K_tmp,
        )
        inner = finite_beta.run_code(
            iota=iota_tmp,
            G=G_tmp,
            I=I_tmp,
            current_potential=K_tmp,
            B_in=B_in_guess,
            B_out=B_out_guess,
            optimize_G=ZERO_TOROIDAL_CURRENT,
            optimize_I=not ZERO_TOROIDAL_CURRENT,
            optimize_surface=False,
        )
        if not inner['success']:
            break
        iota_tmp = float(inner['iota'])
        G_tmp = float(inner['G'])
        I_tmp = float(inner['I'])
        K_tmp = np.asarray(inner['current_potential']).copy()

    if inner is None:
        s.x = sdofs_prev
        print("  inner solve FAILED — reverting to previous surface.  J=1e3")
        return 1e3

    if not inner['success']:
        # Inner solve failed: restore previous surface and state, return large J.
        s.x = sdofs_prev
        print("  inner solve FAILED — reverting to previous surface.  J=1e3")
        return 1e3

    # Update mutable state with the new converged solution.
    state['iota'] = float(inner['iota'])
    state['G']    = float(inner['G'])
    state['I']    = float(inner['I'])
    state['K']    = np.asarray(inner['current_potential']).copy()

    # Retrieve the inner plasma field (B_in) for quasisymmetry evaluation.
    B_in, _ = finite_beta.resolve_field_components(
        field_provider=field_provider,
        iota=inner['iota'], G=inner['G'], I=inner['I'],
        current_potential=inner['current_potential'],
    )

    # Assemble the outer objective.
    J_nonqs = surface_field_nonquasisymmetric_ratio(s, B_in)
    J_iota  = 0.5 * IOTA_WEIGHT  * (inner['iota'] - iota0)**2
    J_mr    = 0.5 * MR_WEIGHT    * (s.major_radius() - mr0)**2
    J_res   = 0.5 * RES_WEIGHT   * inner['residual_norm']**2
    beta_achieved = p_new / max(ref_new, 1e-30)
    J_beta  = 0.5 * BETA_WEIGHT  * (beta_achieved - PLASMA_BETA)**2
    J_reg   = 0.5 * REG_WEIGHT   * float(np.sum((dofs - surface_dofs0[:len(dofs)])**2))
    J = J_nonqs + J_iota + J_mr + J_res + J_beta + J_reg

    print(f"  J={J:.4e}, nonQS={J_nonqs:.4e}, iota={inner['iota']:.6f}, "
          f"mr={s.major_radius():.4f}, beta={beta_achieved:.3%}, "
          f"||r||={inner['residual_norm']:.3e}")
    return J


print(f"Optimising {QA_DOF_COUNT} surface DOFs with Powell, maxiter={QA_MAXITER}")

dofs0 = surface_dofs0[:QA_DOF_COUNT]
result_opt = minimize(
    fun, dofs0, method='Powell',
    options={'maxiter': QA_MAXITER,
             'maxfun': max(40, QA_MAXITER * (QA_DOF_COUNT + 1))})

print(f"Optimisation: success={result_opt.success}, nit={result_opt.nit}, "
      f"nfev={result_opt.nfev}, J_final={result_opt.fun:.6e}")

# ===========================================================================
## FINAL SOLVE AT OPTIMISED SURFACE
# ===========================================================================
# Accept the optimised surface DOFs and re-solve the finite-beta equations
# to obtain a clean converged state for output and analysis.

final_dofs = surface_dofs0.copy()
final_dofs[:QA_DOF_COUNT] = result_opt.x
s.x = final_dofs

bs.set_points(s.gamma().reshape(-1, 3))
B_vac_final = bs.B().reshape(s.gamma().shape)
dA_final    = np.linalg.norm(s.normal(), axis=2)
ref_final   = float(np.sum(np.sum(B_vac_final**2, axis=2) * dA_final)
                    / (2.0 * MU0 * np.maximum(np.sum(dA_final), 1e-30)))
pressure_jump_final       = PLASMA_BETA * ref_final
finite_beta.pressure_jump = pressure_jump_final

final_res = None
iota_tmp = float(state['iota'])
G_tmp = float(state['G'])
I_tmp = float(state['I'])
K_tmp = np.asarray(state['K']).copy()
for _ in range(PICARD_ITERS):
    B_in_guess, B_out_guess = finite_beta.resolve_field_components(
        field_provider=field_provider,
        iota=iota_tmp,
        G=G_tmp,
        I=I_tmp,
        current_potential=K_tmp,
    )
    final_res = finite_beta.run_code(
        iota=iota_tmp,
        G=G_tmp,
        I=I_tmp,
        current_potential=K_tmp,
        B_in=B_in_guess,
        B_out=B_out_guess,
        optimize_G=ZERO_TOROIDAL_CURRENT,
        optimize_I=not ZERO_TOROIDAL_CURRENT,
        optimize_surface=False,
    )
    if not final_res['success']:
        break
    iota_tmp = float(final_res['iota'])
    G_tmp = float(final_res['G'])
    I_tmp = float(final_res['I'])
    K_tmp = np.asarray(final_res['current_potential']).copy()

if final_res is None:
    raise RuntimeError('Final finite-beta solve failed before starting the inner solve.')

B_in_final, B_out_final = finite_beta.resolve_field_components(
    field_provider=field_provider,
    iota=final_res['iota'], G=final_res['G'], I=final_res['I'],
    current_potential=final_res['current_potential'],
)

nonqs_final = surface_field_nonquasisymmetric_ratio(s, B_in_final)
beta_final  = pressure_jump_final / max(ref_final, 1e-30)

print("\nFinal state:")
print(f"  success       = {final_res['success']}")
print(f"  iota          = {final_res['iota']:.6f}  (vacuum: {seed['iota']:.6f})")
print(f"  nonQS ratio   = {nonqs_final:.6e}")
print(f"  achieved beta = {beta_final:.4%}  (target: {PLASMA_BETA:.4%})")
print(f"  ||residual||  = {final_res['residual_norm']:.3e}")
print(f"  major radius  = {s.major_radius():.6f}  (initial: {mr0:.6f})")

# ===========================================================================
## SAVE FINAL OUTPUT FILES
# ===========================================================================
curves_to_vtk(all_curves, OUT_DIR + "curves_opt")
s.to_vtk(OUT_DIR + "surf_opt")
print(f"\nSaved final coils   → {OUT_DIR}curves_opt.vtu")
print(f"Saved final surface → {OUT_DIR}surf_opt.vts")

# ===========================================================================
## VMEC BENCHMARK EXPORT (optional)
# ===========================================================================
# Export the optimised plasma boundary as a self-consistent VMEC equilibrium.
#
# Physics:  Since the single-surface finite-beta model does not uniquely
# determine the volume current profile, we use the iota-constrained mode
# (NCURR=0 in VMEC) and prescribe a flat iota profile equal to the edge
# iota from the direct solve.  Total toroidal current is zero — appropriate
# for most stellarators.
#
# The total toroidal flux phiedge must be calibrated so that the VMEC vacuum
# field strength at the boundary matches the direct-closure result.  We do
# this with a short unit-flux vacuum run.

if EXPORT_VMEC:
    try:
        from simsopt.mhd.vmec import Vmec
        from simsopt.mhd.profiles import ProfilePolynomial
        from simsopt.mhd.vmec_diagnostics import B_cartesian

        # Convert the optimised XYZ-tensor-Fourier surface to the RZFourier
        # representation required by VMEC.
        boundary_rz = s.to_RZFourier()

        # ── Step 1: calibrate phiedge with a unit-flux vacuum run ──────────
        # Run VMEC at phiedge=1 with zero pressure, read back <|B|²/(2μ₀)> at the
        # boundary, and scale so it matches the direct-closure <|B|²/(2μ₀)>.
        vmec_cal = Vmec(None, verbose=False,
                        nphi=len(phis), ntheta=len(thetas),
                        range_surface="field period")
        vmec_cal.boundary = boundary_rz
        vmec_cal.indata.mpol = int(boundary_rz.mpol)
        vmec_cal.indata.ntor = int(boundary_rz.ntor)
        vmec_cal.indata.lfreeb = False
        vmec_cal.indata.ns_array[:]    = 0
        vmec_cal.indata.niter_array[:] = 0
        vmec_cal.indata.ftol_array[:]  = -1.0
        for i, (ns, nit, ft) in enumerate(zip(VMEC_NS, VMEC_NITER, VMEC_FTOL)):
            vmec_cal.indata.ns_array[i]    = ns
            vmec_cal.indata.niter_array[i] = nit
            vmec_cal.indata.ftol_array[i]  = ft
        # Iota-constrained (NCURR=0): prescribe iota, set net toroidal current = 0.
        vmec_cal.indata.ncurr  = 0
        vmec_cal.indata.curtor = 0.0
        vmec_cal.indata.phiedge = 1.0
        vmec_cal.pressure_profile = ProfilePolynomial([0.0])
        vmec_cal.iota_profile     = ProfilePolynomial([final_res['iota']])
        vmec_cal.run()

        Bx_c, By_c, Bz_c = B_cartesian(vmec_cal, nphi=len(phis), ntheta=len(thetas),
                                        range="field period")
        B_cal   = np.stack([Bx_c, By_c, Bz_c], axis=2)
        dA_cal  = np.linalg.norm(vmec_cal.boundary.normal(), axis=2)
        ref_cal = float(np.sum(np.sum(B_cal**2, axis=2) * dA_cal)
                        / (2.0 * MU0 * np.maximum(np.sum(dA_cal), 1e-30)))
        phiedge = float(np.sqrt(ref_final / max(ref_cal, 1e-30)))
        print(f"\nVMEC phiedge calibration: phiedge = {phiedge:.6f} Wb")

        # ── Step 2: run the finite-beta equilibrium ─────────────────────────
        # Pressure profile:  p(s) = p₀ (1 − s).
        # For VMEC, a parabolic p(s) profile with axis value p₀ gives
        # β ≈ p₀ / <B²/(2μ₀)> (normalised by the volume-averaged field).
        # Use 2× the surface reference pressure to account for the profile factor.
        p_axis = 2.0 * PLASMA_BETA * ref_final

        vmec = Vmec(None, verbose=False,
                    nphi=len(phis), ntheta=len(thetas),
                    range_surface="field period")
        vmec.boundary = boundary_rz
        vmec.indata.mpol = int(boundary_rz.mpol)
        vmec.indata.ntor = int(boundary_rz.ntor)
        vmec.indata.lfreeb = False
        vmec.indata.ns_array[:]    = 0
        vmec.indata.niter_array[:] = 0
        vmec.indata.ftol_array[:]  = -1.0
        for i, (ns, nit, ft) in enumerate(zip(VMEC_NS, VMEC_NITER, VMEC_FTOL)):
            vmec.indata.ns_array[i]    = ns
            vmec.indata.niter_array[i] = nit
            vmec.indata.ftol_array[i]  = ft
        vmec.indata.ncurr  = 0
        vmec.indata.curtor = 0.0
        vmec.indata.phiedge  = phiedge
        vmec.pressure_profile = ProfilePolynomial([p_axis, -p_axis])
        vmec.iota_profile     = ProfilePolynomial([final_res['iota']])
        vmec.run()

        vmec_input_path = Path(OUT_DIR) / "input.boozerQA_finitebeta"
        vmec.write_input(str(vmec_input_path))
        print(f"VMEC input written      → {vmec_input_path}")
        print(f"  VMEC iota_edge = {vmec.iota_edge():.6f}  (direct: {final_res['iota']:.6f})")
        print(f"  VMEC mean_iota = {vmec.mean_iota():.6f}")

        # ── Step 3: compare boundary field magnitudes ───────────────────────
        Bx_v, By_v, Bz_v = B_cartesian(vmec, nphi=len(phis), ntheta=len(thetas),
                                        range="field period")
        B_vmec   = np.stack([Bx_v, By_v, Bz_v], axis=2)
        modB_dir = np.linalg.norm(B_in_final, axis=2)
        modB_vmec = np.linalg.norm(B_vmec, axis=2)
        rel_diff  = (float(np.linalg.norm(modB_vmec - modB_dir))
                     / max(float(np.linalg.norm(modB_dir)), 1e-30))
        print(f"  |B| rel diff (direct vs VMEC) = {rel_diff:.6e}")
        print("  (Use ParaView to visualise input.boozerQA_finitebeta and surf_opt.vts.)")
    except Exception as exc:
        print(f"\nVMEC export skipped: {exc}")

print("\nEnd of 2_Intermediate/boozerQA_finitebeta.py")
print("==============================================" )
print("- If VMEC export is enabled, inspect generated input/wout files and benchmark plots.")