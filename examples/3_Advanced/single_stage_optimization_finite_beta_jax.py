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

from simsopt import make_optimizable
from simsopt._core.finite_difference import MPIFiniteDifference
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
from simsopt.mhd import VmecJax, QuasisymmetryRatioResidualJax, VirtualCasingJax
from simsopt.objectives import LeastSquaresProblem, QuadraticPenalty, SquaredFlux
from simsopt.util import MpiPartition, comm_world, proc0_print


mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)

MAXITER_stage_2 = 10
MAXITER_single_stage = 10
max_mode = 1
vmec_input_filename = os.path.join(parent_path, 'inputs', 'input.QH_finitebeta')
ncoils = 3
aspect_ratio_target = 7.0
CC_THRESHOLD = 0.08
LENGTH_THRESHOLD = 3.3
CURVATURE_THRESHOLD = 7
MSC_THRESHOLD = 10
nphi_VMEC = 34
ntheta_VMEC = 34
vc_src_nphi = ntheta_VMEC
nmodes_coils = 7
coils_objective_weight = 1e+3
aspect_ratio_weight = 1
diff_method = "forward"
R0 = 1.0
R1 = 0.6
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 0
JACOBIAN_THRESHOLD = 100
LENGTH_CON_WEIGHT = 0.1
LENGTH_WEIGHT = 1e-8
CC_WEIGHT = 1e+0
CURVATURE_WEIGHT = 1e-3
MSC_WEIGHT = 1e-3
ARCLENGTH_WEIGHT = 1e-9

directory = 'optimization_QH_finitebeta_jax'
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
vmec = VmecJax(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec.boundary

vc = VirtualCasingJax.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
total_current_vmec = vmec.external_current() / (2 * surf.nfp)

base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
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
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, mean(B.n)={np.mean(np.abs(BdotN_surf)):.1e}"
        outstr += f", |grad coils|={np.linalg.norm(grad):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        print(outstr)
    return J, grad


def fun_J(prob, coils_prob):
    global previous_surf_dofs
    J_stage_1 = prob.objective()
    if np.any(previous_surf_dofs != prob.x):
        previous_surf_dofs = prob.x
        try:
            vc = VirtualCasingJax.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
            Jf.target = vc.B_external_normal
        except ObjectiveFailure:
            pass

    bs.set_points(surf.gamma().reshape((-1, 3)))
    J_stage_2 = coils_objective_weight * coils_prob.J()
    return J_stage_1 + J_stage_2


def fun(dofss, prob_jacobian, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    os.chdir(vmec_results_path)
    prob.x = dofss[-number_vmec_dofs:]
    coil_dofs = dofss[:-number_vmec_dofs]
    JF.full_unfix(free_coil_dofs)
    JF.x = coil_dofs
    J = fun_J(prob, JF)
    if J > JACOBIAN_THRESHOLD or isnan(J):
        proc0_print(f"fun#{info['Nfeval']}: Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(coil_dofs)
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        coils_dJ = JF.dJ()
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        JF.fix_all()
        grad_with_respect_to_surface = prob_jacobian.jac(prob.x)[0]

    JF.fix_all()
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    return J, grad


surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(surf.x))
qs = QuasisymmetryRatioResidualJax(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1)]
prob = LeastSquaresProblem.from_tuples(objective_tuple)
previous_surf_dofs = prob.x
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
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_after_stage2"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_after_stage2"), extra_data=pointData)
proc0_print(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
dofs[:-number_vmec_dofs] = res.x
JF.x = dofs[:-number_vmec_dofs]
mpi.comm_world.Bcast(dofs, root=0)
opt = make_optimizable(fun_J, prob, JF)
free_coil_dofs = JF.dofs_free_status
JF.fix_all()

with MPIFiniteDifference(opt.J, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
    if mpi.proc0_world:
        res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)

Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
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
