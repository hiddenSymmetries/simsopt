#!/usr/bin/env python3
r"""
In this example we solve both stage-I and stage-II optimization problems
using the finite-beta single-stage approach of R. Jorge et al in
https://arxiv.org/abs/2302.10622.  The objective function is
J = J_stage1 + coils_objective_weight*J_stage2.

This script follows single_stage_optimization_finite_beta.py, but uses
VmecJax, QuasisymmetryRatioResidualJax, and VirtualCasingJax.
Coils and their derivatives remain the native SIMSOPT coil objects.
"""

import os
from math import isnan
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import vmec_jax as vj
from vmec_jax._compat import enable_x64

from simsopt._core.util import ObjectiveFailure
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    ArclengthVariation,
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
    MeanSquaredCurvature,
    create_equally_spaced_curves,
    curves_to_vtk,
)
from simsopt.mhd import (
    B_cartesian_jax_tangent_columns,
    B_external_normal_jacobian_from_surface,
    VmecJax,
    QuasisymmetryRatioResidualJax,
    VirtualCasingJax,
    local_squared_flux_surface_gradient,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.util import MpiPartition, in_github_actions, proc0_print


class _SerialComm:
    rank = 0

    def Bcast(self, *args, **kwargs):
        return None


class _SerialMpiPartition:
    proc0_world = True
    comm_world = _SerialComm()


try:
    mpi = MpiPartition()
except RuntimeError as e:
    if "mpi4py is not installed" not in str(e):
        raise
    mpi = _SerialMpiPartition()
comm_world = mpi.comm_world
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ["1", "true", "yes", "on"]


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


quick_run = in_github_actions or _env_flag("SIMSOPT_JAX_QUICK")
MAXITER_stage_2 = _env_int("SIMSOPT_JAX_MAXITER_STAGE_2", 1 if quick_run else 10)
MAXITER_single_stage = _env_int("SIMSOPT_JAX_MAXITER_SINGLE_STAGE", 1 if quick_run else 10)
max_mode = _env_int("SIMSOPT_JAX_MAX_MODE", 1)
single_stage_check_only = (
    in_github_actions
    or _env_flag("SIMSOPT_JAX_SINGLE_STAGE_CHECK_ONLY")
)
vmec_input_filename = os.path.join(parent_path, 'inputs', 'input.QH_finitebeta')
ncoils = 3
aspect_ratio_target = 7.0
CC_THRESHOLD = 0.08
LENGTH_THRESHOLD = 3.3
CURVATURE_THRESHOLD = 7
MSC_THRESHOLD = 10
nphi_VMEC = _env_int("SIMSOPT_JAX_NPHI_VMEC", 12 if quick_run else 34)
ntheta_VMEC = _env_int("SIMSOPT_JAX_NTHETA_VMEC", 12 if quick_run else 34)
vc_src_nphi = _env_int("SIMSOPT_JAX_VC_SRC_NPHI", ntheta_VMEC)
nmodes_coils = 7
coil_numquadpoints = _env_int("SIMSOPT_JAX_COIL_NUMQUADPOINTS", 64 if quick_run else 128)
vmec_jax_inner_max_iter = os.environ.get("SIMSOPT_JAX_INNER_MAX_ITER")
vmec_jax_inner_max_iter = 3 if quick_run and vmec_jax_inner_max_iter is None else vmec_jax_inner_max_iter
vmec_jax_inner_max_iter = None if vmec_jax_inner_max_iter is None else int(vmec_jax_inner_max_iter)
vmec_jax_inner_ftol = os.environ.get("SIMSOPT_JAX_INNER_FTOL")
vmec_jax_inner_ftol = 1e-6 if quick_run and vmec_jax_inner_ftol is None else vmec_jax_inner_ftol
vmec_jax_inner_ftol = None if vmec_jax_inner_ftol is None else float(vmec_jax_inner_ftol)
coils_objective_weight = 1e+3
aspect_ratio_weight = 1
R0 = 1.0
R1 = 0.6
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
JACOBIAN_THRESHOLD = _env_float("SIMSOPT_JAX_JACOBIAN_THRESHOLD", 1e6 if quick_run else 100)
BFGS_MAX_SURFACE_STEP = _env_float("SIMSOPT_JAX_BFGS_MAX_SURFACE_STEP", 0.05 if quick_run else 0.1)
BFGS_MAX_COIL_STEP = _env_float("SIMSOPT_JAX_BFGS_MAX_COIL_STEP", 1.0 if quick_run else 2.0)
BFGS_GUARD_WEIGHT = _env_float("SIMSOPT_JAX_BFGS_GUARD_WEIGHT", JACOBIAN_THRESHOLD)
BFGS_INITIAL_INVERSE_HESSIAN_SCALE = _env_float(
    "SIMSOPT_JAX_BFGS_INITIAL_INVERSE_HESSIAN_SCALE",
    1e-4 if quick_run else 1e-3,
)
LENGTH_CON_WEIGHT = 0.1
LENGTH_WEIGHT = 1e-8
CC_WEIGHT = 1e+0
CURVATURE_WEIGHT = 1e-3
MSC_WEIGHT = 1e-3
ARCLENGTH_WEIGHT = 1e-9

directory = os.environ.get("SIMSOPT_JAX_OUTPUT_DIR", 'optimization_QH_finitebeta_jax')
vmec_verbose = False
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)

stage2_history = []
single_stage_history = []
single_stage_reference_dofs = None

proc0_print(f' Using vmec input file {vmec_input_filename}')
enable_x64(True)
vmec = VmecJax(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec.boundary

vc = VirtualCasingJax.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
total_current_vmec = vmec.external_current() / (2 * surf.nfp)

base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=coil_numquadpoints)
base_currents = [Current(total_current_vmec / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
total_current = Current(total_current_vmec)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]

coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
BdotN_init = np.copy(BdotN_surf)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)

Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum(QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves)))
JF = Jf + J_CC + J_LENGTH + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS

proc0_print('  Starting JAX-backed finite-beta optimization')


def fun_coils(dofss, info):
    info['Nfeval'] += 1
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target
        stage2_history.append((
            info['Nfeval'],
            J,
            jf,
            np.linalg.norm(grad),
            np.mean(np.abs(BdotN_surf)),
        ))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, mean(B.n)={np.mean(np.abs(BdotN_surf)):.1e}"
        outstr += f", |grad coils|={np.linalg.norm(grad):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        print(outstr)
    return J, grad


def _vmec_jax_spec_label_from_simsopt_dof(name):
    local_name = name.split(":")[-1]
    coeff, indices = local_name.split("(")
    m_str, n_str = indices.rstrip(")").split(",")
    return f"{coeff}{int(m_str)}{int(n_str)}"


def _build_exact_stage1_and_target_objectives():
    cfg, indata = vj.load_config(vmec_input_filename)
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes)
    specs = vj.boundary_param_specs(
        boundary,
        static.modes,
        max_mode=max_mode,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    residuals_fn = vj.make_qh_residuals_fn(
        static,
        indata,
        helicity_m=1,
        helicity_n=-1,
        target_aspect=aspect_ratio_target,
        surfaces=quasisymmetry_target_surfaces,
        aspect_weight=np.sqrt(aspect_ratio_weight),
    )
    exact_opt = vj.FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        specs,
        residuals_fn,
        inner_max_iter=vmec_jax_inner_max_iter,
        inner_ftol=vmec_jax_inner_ftol,
    )

    surf_label_to_index = {
        _vmec_jax_spec_label_from_simsopt_dof(name): i
        for i, name in enumerate(surf.dof_names)
    }
    missing = [spec.name for spec in specs if spec.name not in surf_label_to_index]
    if missing:
        raise ValueError(f"Exact VMEC-JAX specs are not active SIMSOPT surface dofs: {missing}")

    spec_to_surf = np.asarray([surf_label_to_index[spec.name] for spec in specs], dtype=int)
    surf_x0 = np.copy(surf.x)

    def surface_params(surface_x):
        surface_x = np.asarray(surface_x, dtype=float)
        return surface_x[spec_to_surf] - surf_x0[spec_to_surf]

    def stage1_objective_and_gradient(surface_x):
        params = surface_params(surface_x)
        cost, grad_params = exact_opt.objective_and_gradient_fun(params)
        grad_surface = np.zeros(number_vmec_dofs)
        grad_surface[spec_to_surf] = 2.0 * grad_params
        return 2.0 * cost, grad_surface

    def target_and_jacobian(surface_x):
        params = surface_params(surface_x)
        B_total, B_param_tangents = B_cartesian_jax_tangent_columns(
            exact_opt,
            params,
            quadpoints_phi=surf.quadpoints_phi,
            quadpoints_theta=surf.quadpoints_theta,
        )
        B_total_tangents = np.zeros(surf.gamma().shape + (number_vmec_dofs,))
        B_total_tangents[:, :, :, spec_to_surf] = B_param_tangents
        return B_external_normal_jacobian_from_surface(
            surf,
            B_total,
            B_total_tangents=B_total_tangents,
        )

    return stage1_objective_and_gradient, target_and_jacobian


def fun(dofss, stage1_objective_and_gradient, target_and_jacobian, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    if single_stage_reference_dofs is not None:
        guard = _single_stage_step_guard(dofss)
        if guard is not None:
            J, grad = guard
            proc0_print(f"fun#{info['Nfeval']}: Step guard active, returning J={J:.4e}")
            if mpi.proc0_world:
                single_stage_history.append((
                    info['Nfeval'],
                    J,
                    np.linalg.norm(grad),
                    np.linalg.norm(grad[:-number_vmec_dofs]),
                    np.linalg.norm(grad[-number_vmec_dofs:]),
                ))
            return J, grad

    os.chdir(vmec_results_path)
    vmec.x = dofss[-number_vmec_dofs:]
    coil_dofs = dofss[:-number_vmec_dofs]
    JF.full_unfix(free_coil_dofs)
    JF.x = coil_dofs
    bs.set_points(surf.gamma().reshape((-1, 3)))
    try:
        Jf.target, target_jacobian = target_and_jacobian(vmec.x)
        J_stage_1, prob_dJ = stage1_objective_and_gradient(vmec.x)
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
    except ObjectiveFailure:
        J = JACOBIAN_THRESHOLD
    if J > JACOBIAN_THRESHOLD or isnan(J):
        proc0_print(f"fun#{info['Nfeval']}: Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(coil_dofs)
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        coils_dJ = JF.dJ()
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        grad_with_respect_to_surface = (
            prob_dJ
            + coils_objective_weight * local_squared_flux_surface_gradient(
                surf,
                bs,
                Jf.target,
                target_jacobian,
            )
        )

    JF.fix_all()
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    if mpi.proc0_world:
        single_stage_history.append((
            info['Nfeval'],
            J,
            np.linalg.norm(grad),
            np.linalg.norm(grad_with_respect_to_coils),
            np.linalg.norm(grad_with_respect_to_surface),
        ))
    return J, grad


def _single_stage_step_guard(dofss):
    delta = np.asarray(dofss, dtype=float) - single_stage_reference_dofs
    coil_delta = delta[:-number_vmec_dofs]
    surface_delta = delta[-number_vmec_dofs:]
    surface_abs = np.abs(surface_delta)
    surface_max = np.max(surface_abs) if len(surface_abs) else 0.0
    coil_norm = np.linalg.norm(coil_delta)
    surface_excess = max(0.0, surface_max - BFGS_MAX_SURFACE_STEP)
    coil_excess = max(0.0, coil_norm - BFGS_MAX_COIL_STEP)
    if surface_excess == 0.0 and coil_excess == 0.0:
        return None

    surface_ratio = surface_excess / BFGS_MAX_SURFACE_STEP
    coil_ratio = coil_excess / BFGS_MAX_COIL_STEP
    J = JACOBIAN_THRESHOLD + BFGS_GUARD_WEIGHT * (surface_ratio**2 + coil_ratio**2)
    grad = np.zeros_like(delta)
    if surface_excess > 0.0:
        idx = int(np.argmax(surface_abs))
        grad[-number_vmec_dofs + idx] = (
            2.0 * BFGS_GUARD_WEIGHT * surface_ratio
            * np.sign(surface_delta[idx]) / BFGS_MAX_SURFACE_STEP
        )
    if coil_excess > 0.0 and coil_norm > 0.0:
        grad[:-number_vmec_dofs] = (
            2.0 * BFGS_GUARD_WEIGHT * coil_ratio
            * coil_delta / (BFGS_MAX_COIL_STEP * coil_norm)
        )
    return J, grad


def _save_history(filename, header, rows):
    if comm_world.rank == 0 and rows:
        np.savetxt(os.path.join(this_path, filename), np.asarray(rows), header=header)


def _save_bnormal_plot(filename, fields):
    if comm_world.rank != 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    vmax = max(float(np.max(np.abs(field))) for _, field in fields)
    vmax = max(vmax, 1e-16)
    fig, axes = plt.subplots(
        1, len(fields),
        figsize=(4.2 * len(fields), 3.4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, field) in zip(axes, fields):
        im = ax.imshow(field.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("phi index")
        ax.set_ylabel("theta index")
    fig.colorbar(im, ax=axes, shrink=0.85, label="B dot n")
    fig.savefig(os.path.join(this_path, filename), dpi=150)
    plt.close(fig)


def _save_history_plot(filename):
    if comm_world.rank != 0 or (not stage2_history and not single_stage_history):
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    if stage2_history:
        data = np.asarray(stage2_history)
        axes[0].semilogy(data[:, 0], data[:, 1], "o-", label="total")
        axes[0].semilogy(data[:, 0], data[:, 2], "s-", label="squared flux")
        axes[0].set_title("stage 2")
        axes[0].set_xlabel("evaluation")
        axes[0].set_ylabel("objective")
        axes[0].legend()
    if single_stage_history:
        data = np.asarray(single_stage_history)
        axes[1].semilogy(data[:, 0], data[:, 1], "o-", label="objective")
        axes[1].semilogy(data[:, 0], data[:, 2], "s-", label="gradient norm")
        axes[1].set_title("single stage")
        axes[1].set_xlabel("evaluation")
        axes[1].legend()
    fig.savefig(os.path.join(this_path, filename), dpi=150)
    plt.close(fig)


surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(surf.x))
qs = QuasisymmetryRatioResidualJax(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
stage1_objective_and_gradient, target_and_jacobian = _build_exact_stage1_and_target_objectives()
dofs = np.concatenate((JF.x, vmec.x))
bs.set_points(surf.gamma().reshape((-1, 3)))
vc = VirtualCasingJax.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
proc0_print(f"Aspect ratio before optimization: {vmec.aspect()}")
proc0_print(f"Mean iota before optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective before optimization: {qs.total()}")
proc0_print(f"Magnetic well before optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux before optimization: {Jf.J()}")
proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
BdotN_stage2 = np.copy(BdotN_surf)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_after_stage2"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_after_stage2"), extra_data=pointData)
proc0_print(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
dofs[:-number_vmec_dofs] = res.x
JF.x = dofs[:-number_vmec_dofs]
free_coil_dofs = JF.dofs_free_status
JF.fix_all()
mpi.comm_world.Bcast(dofs, root=0)
single_stage_reference_dofs = np.copy(dofs)
if mpi.proc0_world:
    if single_stage_check_only:
        J_check, grad_check = fun(
            dofs,
            stage1_objective_and_gradient,
            target_and_jacobian,
            {'Nfeval': 0},
        )
        label = "CI" if in_github_actions else "Benchmark"
        proc0_print(
            f"{label} single-stage check: J={J_check:.4f}, "
            f"|grad|={np.linalg.norm(grad_check):.4e}"
        )
    else:
        bfgs_options = {'maxiter': MAXITER_single_stage}
        if BFGS_INITIAL_INVERSE_HESSIAN_SCALE > 0.0:
            bfgs_options['hess_inv0'] = (
                BFGS_INITIAL_INVERSE_HESSIAN_SCALE
                * np.eye(len(dofs))
            )
        res = minimize(
            fun,
            dofs,
            args=(stage1_objective_and_gradient, target_and_jacobian, {'Nfeval': 0}),
            jac=True,
            method='BFGS',
            options=bfgs_options,
            tol=1e-9,
        )
        dofs = res.x
mpi.comm_world.Bcast(dofs, root=0)
JF.full_unfix(free_coil_dofs)
JF.x = dofs[:-number_vmec_dofs]
vmec.x = dofs[-number_vmec_dofs:]
bs.set_points(surf.gamma().reshape((-1, 3)))
Jf.target, _ = target_and_jacobian(vmec.x)

Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_opt"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_opt"), extra_data=pointData)
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
vmec.write_input(os.path.join(this_path, 'input.final'))
_save_history("stage2_history.txt", "eval J Jf grad_norm mean_abs_BdotN", stage2_history)
_save_history("single_stage_history.txt", "eval J grad_norm coil_grad_norm surface_grad_norm", single_stage_history)
_save_bnormal_plot(
    "Bnormal_finite_beta_jax.png",
    [("initial", BdotN_init), ("after stage 2", BdotN_stage2), ("optimized", BdotN_surf)],
)
_save_history_plot("finite_beta_history_jax.png")
proc0_print(f"Aspect ratio after optimization: {vmec.aspect()}")
proc0_print(f"Mean iota after optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective after optimization: {qs.total()}")
proc0_print(f"Magnetic well after optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux after optimization: {Jf.J()}")
