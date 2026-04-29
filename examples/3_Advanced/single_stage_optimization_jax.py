#!/usr/bin/env python3
r"""
In this example we solve both stage-I and stage-II optimization problems
using the single-stage approach of R. Jorge et al in
https://arxiv.org/abs/2302.10622.  The objective function is
J = J_stage1 + coils_objective_weight*J_stage2.

This script follows single_stage_optimization.py, but uses the JAX VMEC
and quasisymmetry wrappers. Coils and their derivatives remain the native
SIMSOPT coil objects.
"""

import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import vmec_jax as vj
from vmec_jax._compat import enable_x64

from simsopt._core.derivative import Derivative
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
    AspectRatioJax,
    VmecJax,
    QuasisymmetryRatioResidualJax,
    VmecJaxLeastSquaresProblem,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.util import MpiPartition, comm_world, proc0_print


mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)

MAXITER_stage_2 = 10
MAXITER_single_stage = 10
max_mode = 1
vmec_input_filename = os.path.join(parent_path, 'inputs', 'input.nfp4_QH_warm_start')
ncoils = 3
aspect_ratio_target = 7.0
CC_THRESHOLD = 0.08
LENGTH_THRESHOLD = 3.3
CURVATURE_THRESHOLD = 7
MSC_THRESHOLD = 10
nphi_VMEC = 34
ntheta_VMEC = 34
nmodes_coils = 7
coils_objective_weight = 1e+3
aspect_ratio_weight = 1
R0 = 1.0
R1 = 0.6
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
JACOBIAN_THRESHOLD = 100
LENGTH_CON_WEIGHT = 0.1
LENGTH_WEIGHT = 1e-8
CC_WEIGHT = 1e+0
CURVATURE_WEIGHT = 1e-3
MSC_WEIGHT = 1e-3
ARCLENGTH_WEIGHT = 1e-9

directory = 'optimization_QH_jax'
vmec_verbose = False
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)

proc0_print(f' Using vmec input file {vmec_input_filename}')
enable_x64(True)
vmec = VmecJax(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec.boundary

base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
base_currents = [Current(1) * 1e5 for _ in range(ncoils)]
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)

Jf = SquaredFlux(surf, bs, definition="local")
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

proc0_print('  Starting JAX-backed optimization')


def fun_coils(dofss, info):
    info['Nfeval'] += 1
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, mean(B.n)={np.mean(np.abs(BdotN_surf)):.1e}"
        outstr += f", |grad coils|={np.linalg.norm(grad):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        print(outstr)
    return J, grad


def _vmec_jax_spec_label_from_simsopt_dof(name):
    local_name = name.split(":")[-1]
    coeff, indices = local_name.split("(")
    m_str, n_str = indices.rstrip(")").split(",")
    return f"{coeff}{int(m_str)}{int(n_str)}"


def _build_exact_stage1_objective():
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
    aspect = AspectRatioJax(vmec)
    quasisymmetry = QuasisymmetryRatioResidualJax(
        vmec,
        quasisymmetry_target_surfaces,
        helicity_m=1,
        helicity_n=-1,
    )
    # This is the JAX-state analog of LeastSquaresProblem.from_tuples.
    # Each tuple is (objective function, target, weight).
    objective_tuple = [
        (aspect.value_from_state(static), aspect_ratio_target, aspect_ratio_weight),
        (quasisymmetry.residuals_from_state(static, indata), 0.0, 1.0),
    ]
    stage1_objective = VmecJaxLeastSquaresProblem.from_tuples(objective_tuple)
    exact_opt = vj.FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        specs,
        stage1_objective.residuals_from_state,
    )

    surf_label_to_index = {
        _vmec_jax_spec_label_from_simsopt_dof(name): i
        for i, name in enumerate(surf.dof_names)
    }
    missing = [spec.name for spec in specs if spec.name not in surf_label_to_index]
    if missing:
        raise ValueError(f"Exact VMEC-JAX stage-I specs are not active SIMSOPT surface dofs: {missing}")

    spec_to_surf = np.asarray([surf_label_to_index[spec.name] for spec in specs], dtype=int)
    surf_x0 = np.copy(surf.x)

    def stage1_objective_and_gradient(surface_x):
        params = np.asarray(surface_x, dtype=float)[spec_to_surf] - surf_x0[spec_to_surf]
        cost, grad_params = exact_opt.objective_and_gradient_fun(params)
        grad_surface = np.zeros(number_vmec_dofs)
        grad_surface[spec_to_surf] = 2.0 * grad_params
        return 2.0 * cost, grad_surface

    return stage1_objective_and_gradient


def fun(dofs, stage1_objective_and_gradient=None, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    JF.x = dofs[:-number_vmec_dofs]
    vmec.x = dofs[-number_vmec_dofs:]
    bs.set_points(surf.gamma().reshape((-1, 3)))
    os.chdir(vmec_results_path)
    J_stage_1, prob_dJ = stage1_objective_and_gradient(vmec.x)
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    if J > JACOBIAN_THRESHOLD or np.isnan(J):
        proc0_print(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(JF.x)
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        coils_dJ = JF.dJ()
        n = surf.normal()
        absn = np.linalg.norm(n, axis=2)
        B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
        Bcoil = bs.B().reshape(n.shape)
        unitn = n * (1./absn)[:, :, None]
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
        B_N = np.sum(Bcoil * n, axis=2)
        assert Jf.definition == "local"
        dJdx = (Bcoil_n/mod_Bcoil**2)[:, :, None] * (
            np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3)
        )
        dJdN = (Bcoil_n/mod_Bcoil**2)[:, :, None] * Bcoil - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
        deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
        mixed_dJ = Derivative({surf: deriv})(surf)
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        grad_with_respect_to_surface = prob_dJ + coils_objective_weight * mixed_dJ
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    return J, grad


surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(surf.x))
qs = QuasisymmetryRatioResidualJax(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
stage1_objective_and_gradient = _build_exact_stage1_objective()
dofs = np.concatenate((JF.x, vmec.x))
bs.set_points(surf.gamma().reshape((-1, 3)))
Jf = SquaredFlux(surf, bs, definition="local")
proc0_print(f"Aspect ratio before optimization: {vmec.aspect()}")
proc0_print(f"Mean iota before optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective before optimization: {qs.total()}")
proc0_print(f"Magnetic well before optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux before optimization: {Jf.J()}")
proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_after_stage2"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_after_stage2"), extra_data=pointData)
proc0_print(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
dofs = np.concatenate((JF.x, vmec.x))
if mpi.proc0_world:
    res = minimize(fun, dofs, args=(stage1_objective_and_gradient, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-15)
    dofs = res.x
mpi.comm_world.Bcast(dofs, root=0)
JF.x = dofs[:-number_vmec_dofs]
vmec.x = dofs[-number_vmec_dofs:]
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_opt"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_opt"), extra_data=pointData)
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
vmec.write_input(os.path.join(this_path, 'input.final'))
proc0_print(f"Aspect ratio after optimization: {vmec.aspect()}")
proc0_print(f"Mean iota after optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective after optimization: {qs.total()}")
proc0_print(f"Magnetic well after optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux after optimization: {Jf.J()}")
