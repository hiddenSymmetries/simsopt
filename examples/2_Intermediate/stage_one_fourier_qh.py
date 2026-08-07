#!/usr/bin/env python


import numpy as np
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/stage_one_fourier.py")
proc0_print("==================================================")

vmec = Vmec(verbose=False, mpi=mpi)
vmec.indata.mpol = 12
vmec.indata.ntor = 12
vmec.indata.ntheta = 32
vmec.indata.nzeta = 32
vmec.indata.ftol_array = np.concatenate(([1e-8], np.zeros(99)))
surf = vmec.boundary
# surf.fix_all()
# surf.fixed_range(mmin=0, mmax=3, nmin=-3, nmax=3, fixed=False)
# surf.fix("rc(0,0)")  # Major radius
# # put bound constraints on the variables
# n_dofs = len(surf.x)
# surf.upper_bounds = 10 * np.ones(n_dofs)
# surf.lower_bounds = -5 * np.ones(n_dofs)
# surf.set_upper_bound("rc(1,0)", 1.0)

vmec.run()

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(
    vmec,
    np.arange(0, 1.01, 0.1),  # Radii to target
    helicity_m=1,
    helicity_n=0,  # -1
)  # (M, N) you want in |B|
# nonlinear constraints
# tuples_nlc = [(vmec.aspect, -np.inf, 8), (vmec.mean_iota, -1.05, -1.0)]

# define problem
prob = LeastSquaresProblem.from_tuples(
    [(qs.residuals, 0, 1), (vmec.aspect, 6, 10), (vmec.mean_iota, 0.42, 10)]
    # [(qs.residuals, 0, 1), (vmec.aspect, 8, 10), (vmec.mean_iota, -1.05, 10)]
)

proc0_print("Initial Quasisymmetry:", qs.total())
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial rotational transform:", vmec.mean_iota())

proc0_print("Beginning optimization")
# proc0_print(f"ndofs: {len(prob.x)}")
# proc0_print(f"dofs names: {prob.dof_names}")

# solver options
# options = {"disp": True, "ftol": 1e-7, "maxiter": 300}
# solve the problem
for step in range(4):
    max_mode = step + 1

    # # VMEC's mpol & ntor will be 3, 4, 5:
    # vmec.indata.mpol = 3 + step
    # vmec.indata.ntor = vmec.indata.mpol

    proc0_print(
        "Beginning optimization with max_mode =",
        max_mode,
        ", vmec mpol=ntor=",
        vmec.indata.mpol,
        ". Previous vmec iteration = ",
        vmec.iter,
    )

    # Define parameter space:
    surf.fix_all()
    surf.fixed_range(
        mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False
    )
    surf.fix("rc(0,0)")  # Major radius

    # put bound constraints on the variables
    n_dofs = len(surf.x)
    surf.upper_bounds = 10 * np.ones(n_dofs)
    surf.lower_bounds = -5 * np.ones(n_dofs)
    surf.set_upper_bound("rc(1,0)", 1.0)

    # solver options
    options = {"disp": True, "ftol": 1e-7, "maxiter": 1}
    # solve the problem
    proc0_print(f"ndofs: {len(prob.x)}")
    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        rel_step=1e-8,
        abs_step=1e-5,  # **options
        # x_scale="jac",
    )
    xopt = prob.x

    # Preserve the output file from the last iteration, so it is not
    # deleted when vmec runs again:
    vmec.files_to_delete = []

    # evaluate the solution
    surf.x = xopt
    vmec.run()
    proc0_print("")
    proc0_print(f"Completed optimization with max_mode ={max_mode}. ")
    proc0_print(f"Final vmec iteration = {vmec.iter}")
    proc0_print("Quasisymmetry:", qs.total())
    proc0_print("aspect ratio:", vmec.aspect())
    proc0_print("rotational transform:", vmec.mean_iota())

# Preserve the output file from the last iteration, so it is not
# deleted when vmec runs again:
vmec.files_to_delete = []


# evaluate the solution
vmec.x = xopt
vmec.run()

proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())


proc0_print("")
proc0_print("End of 2_Intermediate/stage_one_fourier.py")
proc0_print("=================================================")
