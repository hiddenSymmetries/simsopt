#!/usr/bin/env python


import numpy as np
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import MpiPartition, proc0_print

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/stage_one_splines_constrained.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 4,
    "n_cs": 6,
    "nfp": 2,
    "M": 9,
    "N": 4,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": True,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": True,
    "cs_basis": "polar",
    "nurbs": False,
    "use_bishop_frame": True,
}

spline_surf = SurfaceBSpline(**spline_kwargs, default_r=0.2)
# spline_surf.axis.fix("r_axis_0")

proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")

# Disable box bounds on the spline dofs (+-inf instead of the per-dof
# defaults from CrossSectionFixedZeta/PseudoAxis's own construction) --
# the linear inequality constraints built below express the ordering
# relationships that actually matter (e.g. theta_k <= theta_{k+1}
# within a cross section) directly, which per-dof box bounds can't, so
# they replace box bounds here rather than supplementing them.
n_dofs = len(spline_surf.x)
spline_surf.upper_bounds = np.inf * np.ones(n_dofs)
spline_surf.lower_bounds = -np.inf * np.ones(n_dofs)

A_lc, lb_lc, ub_lc, lc_titles = spline_surf.write_inequality_constraints(
    cs_r_max=0.6,
    axis_r_max =1.0

)
proc0_print(f"n linear (spline dof) constraints: {A_lc.shape[0]}")

vmec = Vmec.vmec_from_surf(
    nfp=spline_surf.nfp, surf=spline_surf, mpi=mpi, ns=13, M=12, N=12, ftol=1e-8
)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(
    vmec,
    np.arange(0, 1.01, 0.1),  # Radii to target
    helicity_m=1,
    helicity_n=0,  # -1
)  # (M, N) you want in |B|

# Nonlinear constraints: same target values stage_one_splines.py's
# LeastSquaresProblem used as (goal, weight) penalty terms for
# aspect/mean_iota, but expressed here as genuine (lb, ub) bounds -- hard
# constraints rather than soft penalties -- matching the pattern in
# constrained_optimization.py.
tuples_nlc = [(vmec.aspect, -np.inf, 10), (vmec.mean_iota, 0.42, np.inf)]

# Define problem: minimize QS error subject to the aspect/iota nonlinear
# constraints and the spline's own linear (dof-ordering) constraints.
prob = ConstrainedProblem(
    qs.total, tuples_nlc=tuples_nlc, tuple_lc=(A_lc, lb_lc, ub_lc)
)

vmec.run()
proc0_print("Initial Quasisymmetry:", qs.total())
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial rotational transform:", vmec.mean_iota())

proc0_print("Beginning optimization")
proc0_print(f"ndofs: {len(prob.x)}")
proc0_print(f"dofs names: {prob.dof_names}")

# solver options
# initial_tr_radius caps the size of trust-constr's very first trial
# step directly (a genuine trust-region radius, unlike SLSQP which
# exposes no equivalent hook) -- scipy's own default is 1.0, which was
# letting the first step wander into VMEC-breaking territory.
options = {"maxiter": 300, "initial_tr_radius": 1e-3}
# solve the problem
constrained_mpi_solve(
    prob,
    mpi,
    grad=True,
    #rel_step=1e-8,
    abs_step=1e-4,
    opt_method="trust-constr",
    options=options,
)
xopt = prob.x

# Preserve the output file from the last iteration, so it is not
# deleted when vmec runs again:
vmec.files_to_delete = []


# evaluate the solution
spline_surf.x = xopt
vmec.run()
if MPI.COMM_WORLD.rank == 0:
    spline_surf.plot()
proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())


proc0_print("")
proc0_print("End of 2_Intermediate/stage_one_splines_constrained.py")
proc0_print("=================================================")
