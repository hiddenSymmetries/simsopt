#!/usr/bin/env python
# Duplicate of stage_one_splines.py, but using continuation on the
# *objective* (ramping up the quasisymmetry weight over a few stages)
# rather than on the dof space, since the spline control net has no
# natural nested-subspace structure to grow into the way Fourier modes
# do via max_mode (that would require knot insertion, which isn't
# implemented -- see surfacespline.py roadmap). Each stage warm-starts
# from the previous stage's optimum.

import numpy as np
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/stage_one_splines_objective_continuation.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 4,
    "n_cs": 4,
    "nfp": 2,
    "M": 8,
    "N": 4,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": True,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": False,
    "cs_basis": "polar",
    "nurbs": False,
    "use_bishop_frame": True,
}

spline_surf = SurfaceBSpline(**spline_kwargs, default_r=0.2)
spline_surf.axis.fix("r_axis_0")

proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")

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

vmec.run()
proc0_print("Initial Quasisymmetry:", qs.total())
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial rotational transform:", vmec.mean_iota())

# Continuation schedule: ramp the QS weight from small to full strength.
# aspect/iota keep their full weight (10) throughout every stage, so
# each stage still lands near the target aspect ratio/iota; only the
# QS term's influence grows. All dofs are free in every stage (unlike
# the Fourier max_mode continuation) -- this is a homotopy on the cost
# landscape, not on the parameter space.
qs_weights = [1e-2, 1e-1, 1.0]

for stage, qs_weight in enumerate(qs_weights):
    proc0_print("")
    proc0_print(
        f"Beginning stage {stage} with QS weight = {qs_weight}."
        f" Previous vmec iteration = {vmec.iter}"
    )

    prob = LeastSquaresProblem.from_tuples(
        [
            (qs.residuals, 0, qs_weight),
            (vmec.aspect, 6, 10),
            (vmec.mean_iota, 0.42, 10),
        ]
    )

    proc0_print(f"ndofs: {len(prob.x)}")
    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        rel_step=1e-12,
        abs_step=5e-6,
        x_scale="jac",
    )
    xopt = prob.x

    # Preserve the output file from the last iteration, so it is not
    # deleted when vmec runs again:
    vmec.files_to_delete = []

    spline_surf.x = xopt
    vmec.run()

    proc0_print(f"Completed stage {stage} with QS weight = {qs_weight}.")
    proc0_print(f"Final vmec iteration = {vmec.iter}")
    proc0_print("Quasisymmetry:", qs.total())
    proc0_print("aspect ratio:", vmec.aspect())
    proc0_print("rotational transform:", vmec.mean_iota())

if MPI.COMM_WORLD.rank == 0:
    spline_surf.plot()
proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())


proc0_print("")
proc0_print("End of 2_Intermediate/stage_one_splines_objective_continuation.py")
proc0_print("=================================================")
