#!/usr/bin/env python


import numpy as np
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import MpiPartition, proc0_print

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/constrained_optimization.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 6,
    "n_cs": 4,
    "nfp": 4,
    "M": 12,
    "N": 12,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": True,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": True,
    "cs_basis": "polar",
    "nurbs": False,
}

spline_surf = SurfaceBSpline(**spline_kwargs, default_r=0.2)

proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")

vmec = Vmec.vmec_from_surf(
    nfp=spline_surf.nfp, surf=spline_surf, mpi=mpi, ns=13, M=12, N=12, ftol=1e-8
)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(
    vmec,
    np.arange(0, 1.01, 0.1),  # Radii to target
    helicity_m=1,
    helicity_n=-1,
)  # (M, N) you want in |B|
# nonlinear constraints
tuples_nlc = [(vmec.aspect, -np.inf, 8), (vmec.mean_iota, -1.05, -1.0)]

# define problem
prob = ConstrainedProblem(qs.total, tuples_nlc=tuples_nlc)

vmec.run()
proc0_print("Initial Quasisymmetry:", qs.total())
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial rotational transform:", vmec.mean_iota())


# Fourier modes of the boundary with m <= max_mode and |n| <= max_mode
# will be varied in the optimization. A larger range of modes are
# included in the VMEC calculations.

proc0_print("Beginning optimization")
proc0_print(f"ndofs: {len(prob.x)}")
proc0_print(f"dofs names: {prob.dof_names}")

# solver options
options = {"disp": True, "ftol": 1e-7, "maxiter": 300}
# solve the problem
constrained_mpi_solve(
    prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, options=options
)
xopt = prob.x

# Preserve the output file from the last iteration, so it is not
# deleted when vmec runs again:
vmec.files_to_delete = []

# evaluate the solution
spline_surf.x = xopt
vmec.run()
spline_surf.plot()
proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())


proc0_print("")
proc0_print("End of 2_Intermediate/constrained_optimization.py")
proc0_print("=================================================")
