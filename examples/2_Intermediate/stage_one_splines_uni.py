#!/usr/bin/env python


import numpy as np
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/stage_one_splines.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 5,
    "n_cs": 5,
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
    "knot_parametrization": 'uniform'
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
# nonlinear constraints
# tuples_nlc = [(vmec.aspect, -np.inf, 8), (vmec.mean_iota, -1.05, -1.0)]

# define problem
prob = LeastSquaresProblem.from_tuples(
    [(qs.residuals, 0, 1), (vmec.aspect, 6, 10), (vmec.mean_iota, 0.42, 10)]
    # [(qs.residuals, 0, 1), (vmec.aspect, 8, 10), (vmec.mean_iota, -1.05, 10)]
)

vmec.run()
proc0_print("Initial Quasisymmetry:", qs.total())
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial rotational transform:", vmec.mean_iota())

proc0_print("Beginning optimization")
proc0_print(f"ndofs: {len(prob.x)}")
proc0_print(f"dofs names: {prob.dof_names}")

# solver options
# options = {"disp": True, "ftol": 1e-7, "maxiter": 300}
# solve the problem
least_squares_mpi_solve(
    prob,
    mpi,
    grad=True,
    rel_step=1e-12,
    abs_step=1e-5,  # **options
    x_scale="jac",
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
proc0_print("End of 2_Intermediate/stage_one_splines.py")
proc0_print("=================================================")


# spline_kwargs = {
#     "axis_points": 3,
#     "points_per_cs": 5,
#     "n_cs": 5,
#     "nfp": 2,
#     "M": 8,
#     "N": 4,
#     "p_u": 3,
#     "p_v": 3,
#     "cs_equispaced": False,
#     "rays_equispaced": False,
#     "cs_global_angle_free": False,
#     "axis_angles_fixed": False,
#     "cs_basis": "polar",
#     "nurbs": False,
#     "use_bishop_frame": True,
# }


# Running 2_Intermediate/stage_one_splines.py
# ==================================================
# spline_surf.dof_names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta1:theta_2', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:r_4', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta2:theta_4', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:r_4', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta3:theta_4', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:r_4', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta4:theta_4', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'CrossSectionFixedZeta5:theta_2', 'PseudoAxis1:r_axis_0', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
# Initial Quasisymmetry: 0.0001940297678687007
# Initial aspect ratio: 5.812632015268553
# Initial rotational transform: 5.08677125506282e-19
# Beginning optimization
# ndofs: 45
# dofs names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta1:theta_2', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:r_4', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta2:theta_4', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:r_4', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta3:theta_4', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:r_4', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta4:theta_4', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'CrossSectionFixedZeta5:theta_2', 'PseudoAxis1:r_axis_0', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         1.0576e+00                                    8.17e+00    
#        1              2         8.8208e-01      1.76e-01       4.70e-02       4.49e-02    
#        2              4         8.8200e-01      8.91e-05       2.43e-02       2.06e-02    
#        3              5         8.8096e-01      1.03e-03       4.78e-02       8.31e-02    
#        4              6         8.6783e-01      1.31e-02       1.01e-01       4.37e-01    
#        5              7         8.0075e-01      6.71e-02       1.46e-01       1.16e+00    
#        6              8         5.7907e-01      2.22e-01       2.96e-01       2.02e+00    
#        7              9         2.3059e-01      3.48e-01       4.54e-01       4.38e+00    
#        8             10         6.9036e-02      1.62e-01       5.57e-01       1.80e+00    
# WARNING:simsopt.objectives.least_squares:Function evaluation failed for <bound method QuasisymmetryRatioResidual.residuals of <simsopt.mhd.vmec_diagnostics.QuasisymmetryRatioResidual object at 0x30dc37610>>
#        9             12         3.0215e-02      3.88e-02       6.97e-02       1.02e-01    
#       10             13         2.0217e-02      1.00e-02       1.06e-01       4.50e+00    
#       11             14         1.6085e-02      4.13e-03       1.37e-01       1.85e-01    
#       12             15         1.1487e-02      4.60e-03       1.19e-01       2.81e-01    
#       13             16         7.4926e-03      3.99e-03       1.12e-01       9.36e-02    
#       14             17         3.3126e-03      4.18e-03       2.22e-01       1.50e-01    
#       15             18         1.7633e-03      1.55e-03       2.46e-01       3.49e-01    
#       16             19         7.9525e-04      9.68e-04       8.54e-02       9.99e-02    
#       17             21         5.4778e-04      2.47e-04       1.02e-01       4.87e-02    
#       18             22         4.9357e-04      5.42e-05       1.71e-02       4.22e-02    
#       19             23         2.9694e-04      1.97e-04       1.26e-01       3.19e-02    
#       20             24         2.8690e-04      1.00e-05       6.30e-03       2.26e-02    
#       21             25         2.0860e-04      7.83e-05       7.39e-02       2.16e-02    
#       22             26         2.0074e-04      7.86e-06       8.05e-03       1.57e-02    
#       23             27         1.7713e-04      2.36e-05       4.39e-02       1.39e-02    
#       24             28         1.4838e-04      2.88e-05       9.57e-02       3.07e-02    
#       25             29         1.4307e-04      5.31e-06       8.71e-04       6.97e-03    
#       26             30         1.4265e-04      4.20e-07       1.47e-04       1.03e-02    
#       27             31         1.4094e-04      1.71e-06       4.20e-03       1.17e-02    
#       28             32         1.1896e-04      2.20e-05       6.82e-02       9.67e-03    
#       29             33         1.1849e-04      4.77e-07       1.18e-04       6.27e-03    
#       30             34         1.1819e-04      3.01e-07       1.28e-04       3.60e-03    
#       31             35         1.1808e-04      1.04e-07       5.99e-05       4.30e-03    
#       32             36         1.1793e-04      1.53e-07       5.95e-04       5.65e-03    
#       33             38         1.1023e-04      7.70e-06       4.05e-02       2.35e-02    
#       34             39         1.0992e-04      3.10e-07       2.61e-05       1.70e-02    
#       35             40         1.0882e-04      1.10e-06       1.86e-04       3.71e-03    
#       36             41         1.0815e-04      6.65e-07       6.79e-02       4.74e-02    
#       37             42         9.7280e-05      1.09e-05       6.33e-04       1.68e-02    
#       38             43         9.6110e-05      1.17e-06       1.67e-04       7.98e-03    
#       39             44         9.2849e-05      3.26e-06       1.54e-02       1.66e-03    
#       40             45         9.2818e-05      3.12e-08       8.98e-05       6.08e-04    
#       41             46         8.9103e-05      3.72e-06       4.27e-02       1.59e-02    
#       42             47         8.7966e-05      1.14e-06       2.09e-04       2.44e-03    
#       43             48         8.3941e-05      4.03e-06       2.52e-02       1.91e-03    
#       44             49         8.3923e-05      1.81e-08       5.74e-05       7.80e-04    
#       45             50         8.3906e-05      1.65e-08       9.32e-05       1.29e-03    
#       46             51         8.0989e-05      2.92e-06       2.28e-02       1.10e-03    
#       47             52         8.0965e-05      2.38e-08       1.37e-04       6.20e-04    
#       48             53         7.9147e-05      1.82e-06       5.05e-02       6.31e-03    
#       49             54         7.8769e-05      3.78e-07       3.60e-04       7.15e-03    
#       50             55         7.5055e-05      3.71e-06       2.78e-02       2.73e-03    
#       51             56         7.4970e-05      8.47e-08       6.55e-05       4.01e-03    
#       52             58         7.4438e-05      5.32e-07       1.19e-02       7.53e-04    
#       53             59         7.4416e-05      2.22e-08       8.89e-05       2.16e-03    
#       54             60         7.3772e-05      6.43e-07       1.76e-02       2.26e-03    
#       55             61         7.3738e-05      3.44e-08       4.12e-05       5.35e-04    
#       56             63         7.3633e-05      1.05e-07       1.86e-02       3.30e-03    
#       57             64         7.3575e-05      5.74e-08       4.91e-05       4.96e-04    
#       58             65         7.3467e-05      1.09e-07       3.13e-03       6.08e-04    
#       59             66         7.3322e-05      1.45e-07       3.02e-03       7.27e-05    
#       60             67         7.3236e-05      8.53e-08       7.84e-03       2.95e-04    
#       61             68         7.3058e-05      1.79e-07       7.98e-03       3.74e-04    
#       62             69         7.2912e-05      1.46e-07       7.19e-03       2.83e-04    
#       63             71         7.2845e-05      6.64e-08       1.27e-03       3.15e-05    
#       64             73         7.2815e-05      3.01e-08       5.71e-04       3.63e-05    
#       65             75         7.2801e-05      1.40e-08       2.50e-04       3.50e-05    
#       66             76         7.2774e-05      2.73e-08       4.09e-04       3.44e-05    
#       67             79         7.2771e-05      3.39e-09       5.04e-05       3.46e-05    
#       68             80         7.2764e-05      6.75e-09       1.01e-04       3.52e-05    
#       69             83         7.2763e-05      8.42e-10       1.26e-05       1.16e-01    
#       70             84         7.2761e-05      1.65e-09       3.25e-05       7.08e-05    
#       71             85         7.2761e-05      7.83e-10       3.03e-05       3.45e-05    
#       72             87         7.2760e-05      2.65e-10       7.30e-06       1.18e-01    
#       73             88         7.2760e-05      3.75e-11       2.68e-05       5.46e-05    
#       74             89         7.2760e-05      2.24e-10       1.02e-05       1.88e-02    
#       75             92         7.2760e-05      0.00e+00       0.00e+00       1.88e-02    
# `xtol` termination condition is satisfied.
# Function evaluations 92, initial cost 1.0576e+00, final cost 7.2760e-05, first-order optimality 1.88e-02.
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: divide by zero encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: overflow encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))

# Final vmec iteration = 394
# Quasisymmetry: 0.0001454742473170254
# aspect ratio: 6.000005685115758
# rotational transform: 0.41993244084565823

# End of 2_Intermediate/stage_one_splines.py
# =================================================
