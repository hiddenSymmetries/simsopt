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
vmec.indata.nfp = 3
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
    helicity_n=-1,  # -1
)  # (M, N) you want in |B|
# nonlinear constraints
# tuples_nlc = [(vmec.aspect, -np.inf, 8), (vmec.mean_iota, -1.05, -1.0)]

# define problem
prob = LeastSquaresProblem.from_tuples(
    #[(qs.residuals, 0, 1), (vmec.aspect, 6, 10), (vmec.mean_iota, 0.42, 10)]
    [(qs.residuals, 0, 1), (vmec.aspect, 8, 10), (vmec.mean_iota, -1.05, 10)]
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



# Running 2_Intermediate/stage_one_fourier.py
# ==================================================
# Initial Quasisymmetry: 3.057098315145689e-25
# Initial aspect ratio: 10.000000000000128
# Initial rotational transform: -6.0183972941451826e-30
# Beginning optimization
# Beginning optimization with max_mode = 1 , vmec mpol=ntor= 12 . Previous vmec iteration =  0
# ndofs: 8
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         2.5513e+01                                    9.90e+03    
#        1              3         7.2078e+00      1.83e+01       7.36e-02       6.47e+02    
# Function evaluation failed for <bound method QuasisymmetryRatioResidual.residuals of <simsopt.mhd.vmec_diagnostics.QuasisymmetryRatioResidual object at 0x13f672ef0>>
#        2              5         6.0063e+00      1.20e+00       3.65e-02       3.84e+02    
#        3              6         5.2862e+00      7.20e-01       3.77e-02       3.46e+02    
#        4              7         3.7423e+00      1.54e+00       3.47e-02       3.42e+02    
#        5              8         2.9116e+00      8.31e-01       7.02e-02       1.79e+03    
#        6              9         6.6608e-01      2.25e+00       7.11e-02       6.25e+02    
#        7             10         7.5027e-02      5.91e-01       1.01e-01       1.35e+02    
#        8             12         4.9587e-02      2.54e-02       1.33e-02       2.20e+01    
#        9             13         4.6842e-02      2.74e-03       3.31e-02       1.94e+02    
#       10             14         3.1736e-02      1.51e-02       5.94e-03       1.12e+01    
#       11             15         3.0569e-02      1.17e-03       1.59e-02       5.51e+01    
#       12             17         2.9311e-02      1.26e-03       4.17e-03       2.31e+00    
#       13             23         2.9311e-02      5.44e-08       4.57e-06       1.26e+00    
#       14             24         2.9311e-02      3.40e-07       1.33e-06       1.02e+00    
#       15             25         2.9311e-02      5.52e-07       2.50e-06       1.71e+00    
#       16             26         2.9310e-02      2.71e-07       4.78e-06       9.58e-01    
#       17             27         2.9310e-02      2.71e-07       1.33e-06       7.04e-01    
#       18             28         2.9310e-02      4.57e-07       2.47e-06       1.78e+00    
#       19             29         2.9309e-02      2.31e-07       4.82e-06       8.52e-01    
#       20             30         2.9309e-02      2.52e-07       1.35e-06       1.64e+00    
#       21             31         2.9309e-02      9.78e-08       2.42e-06       8.58e-01    
#       22             32         2.9309e-02      1.30e-07       6.96e-07       7.66e-01    
#       23             33         2.9309e-02      2.36e-07       1.33e-06       1.70e+00    
#       24             34         2.9308e-02      9.84e-08       2.41e-06       8.03e-01    
#       25             35         2.9308e-02      1.24e-07       6.90e-07       1.62e+00    
#       26             36         2.9308e-02      4.47e-08       1.21e-06       8.11e-01    
#       27             37         2.9308e-02      6.31e-08       3.50e-07       1.59e+00    
#       28             38         2.9308e-02      2.12e-08       6.03e-07       8.13e-01    
#       29             39         2.9308e-02      3.18e-08       1.76e-07       8.00e-01    
#       30             40         2.9308e-02      6.20e-08       3.48e-07       1.61e+00    
#       31             41         2.9308e-02      2.13e-08       6.02e-07       8.03e-01    
#       32             42         2.9308e-02      3.13e-08       1.75e-07       1.59e+00    
#       33             43         2.9308e-02      1.04e-08       3.01e-07       8.04e-01    
#       34             44         2.9308e-02      1.57e-08       8.78e-08       7.97e-01    
#       35             45         2.9308e-02      3.10e-08       1.75e-07       1.60e+00    
#       36             46         2.9308e-02      1.04e-08       3.01e-07       7.99e-01    
#       37             47         2.9308e-02      1.56e-08       8.76e-08       1.59e+00    
#       38             48         2.9308e-02      5.12e-09       1.50e-07       7.99e-01    
#       39             49         2.9308e-02      7.81e-09       4.39e-08       1.58e+00    
#       40             50         2.9308e-02      2.54e-09       7.52e-08       1.58e+00    
#       41             51         2.9308e-02      6.31e-10       1.88e-08       8.00e-01    
#       42             52         2.9308e-02      9.80e-10       5.50e-09       8.00e-01    
#       43             53         2.9308e-02      1.96e-09       1.10e-08       1.58e+00    
#       44             54         2.9308e-02      6.32e-10       1.88e-08       8.00e-01    
#       45             55         2.9308e-02      9.79e-10       5.49e-09       1.58e+00    
#       46             56         2.9308e-02      3.16e-10       9.40e-09       8.00e-01    
#       47             57         2.9308e-02      4.89e-10       2.75e-09       8.00e-01    
# `xtol` termination condition is satisfied.
# Function evaluations 57, initial cost 2.5513e+01, final cost 2.9308e-02, first-order optimality 8.00e-01.

# Completed optimization with max_mode =1. 
# Final vmec iteration = 107
# Quasisymmetry: 0.056137850918854766
# aspect ratio: 8.002045473855416
# rotational transform: -1.0343915527451537
# Beginning optimization with max_mode = 2 , vmec mpol=ntor= 12 . Previous vmec iteration =  107
# ndofs: 24
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         2.9308e-02                                    1.16e+01    
#        1              3         1.5758e-02      1.35e-02       1.98e-02       4.88e+01    
#        2              5         1.0487e-02      5.27e-03       9.09e-03       2.63e+01    
#        3              7         7.9955e-03      2.49e-03       5.71e-03       1.96e+01    
#        4              9         7.0455e-03      9.50e-04       3.02e-03       6.20e+00    
#        5             10         5.4122e-03      1.63e-03       5.47e-03       2.17e+01    
#        6             11         3.2934e-03      2.12e-03       9.63e-03       3.05e+01    
#        7             13         1.9469e-03      1.35e-03       6.06e-03       8.38e+00    
#        8             14         1.2167e-03      7.30e-04       1.00e-02       2.74e+01    
#        9             15         5.9470e-04      6.22e-04       1.17e-02       1.24e+01    
#       10             17         4.7984e-04      1.15e-04       5.16e-03       2.18e+00    
#       11             19         4.4054e-04      3.93e-05       2.36e-03       1.33e+00    
#       12             20         4.0708e-04      3.35e-05       4.64e-03       8.11e+00    
#       13             22         3.7266e-04      3.44e-05       2.58e-03       2.70e+00    
#       14             23         3.6276e-04      9.90e-06       5.11e-03       3.11e+01    
#       15             25         3.3265e-04      3.01e-05       1.22e-03       6.71e-01    
#       16             27         3.2994e-04      2.71e-06       6.61e-04       1.18e-01    
#       17             29         3.2879e-04      1.15e-06       3.11e-04       2.96e-02    
#       18             32         3.2865e-04      1.43e-07       4.15e-05       4.65e+00    
#       19             34         3.2858e-04      6.66e-08       1.95e-05       4.64e+00    
#       20             35         3.2856e-04      1.65e-08       4.91e-06       4.64e+00    
#       21             39         3.2856e-04      1.20e-12       2.41e-08       4.64e+00    
#       22             41         3.2856e-04      0.00e+00       0.00e+00       4.64e+00    
# `xtol` termination condition is satisfied.
# Function evaluations 41, initial cost 2.9308e-02, final cost 3.2856e-04, first-order optimality 4.64e+00.

# Completed optimization with max_mode =2. 
# Final vmec iteration = 216
# Quasisymmetry: 0.0006568563046887342
# aspect ratio: 8.000025951739003
# rotational transform: -1.0498370778059911
# Beginning optimization with max_mode = 3 , vmec mpol=ntor= 12 . Previous vmec iteration =  216
# ndofs: 48
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         3.2856e-04                                    1.00e+01    
#        1              3         2.3106e-04      9.75e-05       7.59e-03       1.51e+01    
#        2              4         1.4263e-04      8.84e-05       7.72e-03       9.49e+00    
#        3              5         1.3800e-04      4.63e-06       6.91e-03       1.39e+01    
#        4              6         8.3387e-05      5.46e-05       2.10e-03       1.24e+00    
#        5              9         7.8185e-05      5.20e-06       2.58e-04       8.51e-02    
#        6             10         7.4998e-05      3.19e-06       4.50e-04       1.46e-01    
#        7             12         7.4153e-05      8.45e-07       2.47e-04       4.60e-02    
#        8             13         7.3080e-05      1.07e-06       4.71e-04       1.13e-01    
#        9             14         7.2362e-05      7.18e-07       9.46e-04       3.08e-01    
#       10             15         7.2134e-05      2.28e-07       1.00e-03       2.27e+00    
#       11             18         7.2076e-05      5.83e-08       1.70e-05       2.28e+00    
#       12             25         7.2076e-05      0.00e+00       0.00e+00       2.28e+00    
# `xtol` termination condition is satisfied.
# Function evaluations 25, initial cost 3.2856e-04, final cost 7.2076e-05, first-order optimality 2.28e+00.

# Completed optimization with max_mode =3. 
# Final vmec iteration = 303
# Quasisymmetry: 0.0001440985994315676
# aspect ratio: 8.000010561935127
# rotational transform: -1.0499278107285572
# Beginning optimization with max_mode = 4 , vmec mpol=ntor= 12 . Previous vmec iteration =  303
# ndofs: 80
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         7.2076e-05                                    2.53e+00    
#        1              3         6.0149e-05      1.19e-05       4.57e-03       3.14e+00    
#        2              4         5.6748e-05      3.40e-06       4.54e-03       6.24e+00    
#        3              8         4.7375e-05      9.37e-06       6.79e-05       8.13e-01    
#        4             11         4.7339e-05      3.55e-08       8.85e-06       8.25e-01    
#        5             12         4.7309e-05      3.07e-08       8.83e-06       8.43e-01    
#        6             14         4.7301e-05      7.38e-09       2.22e-06       8.45e-01    
#        7             15         4.7294e-05      7.21e-09       2.21e-06       8.49e-01    
#        8             16         4.7287e-05      7.04e-09       2.22e-06       8.56e-01    
#        9             19         4.7286e-05      3.90e-10       1.42e-07       8.53e-01    
#       10             22         4.7286e-05      0.00e+00       0.00e+00       8.53e-01    
# `xtol` termination condition is satisfied.
# Function evaluations 22, initial cost 7.2076e-05, final cost 4.7286e-05, first-order optimality 8.53e-01.

# Completed optimization with max_mode =4. 
# Final vmec iteration = 397
# Quasisymmetry: 9.456273097549776e-05
# aspect ratio: 8.0000044151876
# rotational transform: -1.049968325238857

# Final vmec iteration = 398
# Quasisymmetry: 9.456273097549776e-05
# aspect ratio: 8.0000044151876
# rotational transform: -1.049968325238857

# End of 2_Intermediate/stage_one_fourier.py
# =================================================