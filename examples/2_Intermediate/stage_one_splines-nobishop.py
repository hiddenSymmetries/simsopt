#!/usr/bin/env python


import matplotlib.pyplot as plt
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
    "points_per_cs": 6,
    "n_cs": 5,
    "nfp": 2,
    "M": 8,
    "N": 4,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": False,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": False,
    "cs_basis": "polar",
    "nurbs": False,
    "use_bishop_frame": False,
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
    [(qs.residuals, 0, 1), (vmec.aspect, 6, 1), (vmec.mean_iota, 0.42, 1)]
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
    abs_step=5e-6,  # **options
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
    plt.show()
proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())
proc0_print("spline_surf.x:", repr(xopt))

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

###

# Running 2_Intermediate/stage_one_splines.py
# ==================================================
# spline_surf.dof_names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
# Initial Quasisymmetry: 0.00017188472681977897
# Initial aspect ratio: 6.776908509959019
# Initial rotational transform: -5.420544061956038e-19
# Beginning optimization
# ndofs: 36
# dofs names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
#        0              1         3.9008e-01                                    1.84e+00
#        1              2         9.1784e-02      2.98e-01       1.13e-01       1.77e-01
#        2              3         8.8200e-02      3.58e-03       2.19e-02       3.57e-03
#        3              5         8.7704e-02      4.96e-04       1.39e-01       2.31e-01
#        4              7         8.2445e-02      5.26e-03       1.72e-01       1.29e-01
#        5              8         7.2522e-02      9.92e-03       2.86e-01       2.89e-01
#        6              9         5.1838e-02      2.07e-02       4.83e-01       7.57e-01
#        7             10         1.8816e-02      3.30e-02       3.69e-01       1.77e-01
#        8             12         1.0805e-02      8.01e-03       9.58e-02       9.55e-02
#        9             13         5.2590e-03      5.55e-03       1.58e-01       4.00e-01
#       10             14         2.2429e-03      3.02e-03       1.73e-01       5.67e-01
#       11             15         9.5174e-04      1.29e-03       1.32e-01       1.58e-01
#       12             16         5.8761e-04      3.64e-04       1.01e-01       6.22e-02
#       13             17         4.6940e-04      1.18e-04       6.70e-02       3.69e-02
#       14             18         4.5895e-04      1.04e-05       3.02e-04       4.55e-03
#       15             19         4.4861e-04      1.03e-05       1.17e-02       4.30e-03
#       16             21         3.8238e-04      6.62e-05       7.30e-02       2.08e-02
#       17             22         3.4446e-04      3.79e-05       1.11e-01       6.37e-02
#       18             23         2.7683e-04      6.76e-05       6.19e-02       2.30e-02
#       19             24         2.6766e-04      9.16e-06       9.02e-02       6.81e-02
#       20             25         2.1190e-04      5.58e-05       3.66e-02       9.15e-03
#       21             26         1.9288e-04      1.90e-05       6.05e-02       2.19e-02
#       22             27         1.8712e-04      5.76e-06       3.04e-04       5.67e-03
#       23             28         1.8661e-04      5.02e-07       1.04e-04       3.64e-03
#       24             30         1.7049e-04      1.61e-05       3.69e-02       6.39e-03
#       25             31         1.5158e-04      1.89e-05       6.55e-02       1.89e-02
#       26             32         1.5089e-04      6.83e-07       8.67e-05       4.35e-03
#       27             33         1.5060e-04      2.91e-07       1.15e-03       5.40e-03
#       28             35         1.4093e-04      9.67e-06       4.05e-02       1.68e-02
#       29             36         1.2971e-04      1.12e-05       6.28e-02       2.24e-02
#       30             38         1.2122e-04      8.48e-06       1.59e-02       1.64e-03
#       31             39         1.1622e-04      5.01e-06       2.24e-02       5.78e-03
#       32             41         1.1499e-04      1.22e-06       1.60e-02       4.16e-03
#       33             42         1.1162e-04      3.37e-06       1.35e-02       3.32e-03
#       34             43         1.1158e-04      3.36e-08       2.06e-04       3.27e-03
#       35             44         1.0799e-04      3.59e-06       2.83e-02       1.59e-02
#       36             45         1.0258e-04      5.40e-06       5.19e-02       5.35e-02
#       37             46         9.8871e-05      3.71e-06       1.25e-04       1.91e-03
#       38             48         9.7332e-05      1.54e-06       1.15e-02       3.78e-04
#       39             49         9.5908e-05      1.42e-06       9.54e-03       1.98e-02
#       40             53         9.5691e-05      2.17e-07       2.10e-04       3.07e-03
#       41             54         9.5485e-05      2.06e-07       7.72e-05       8.00e-04
#       42             55         9.5390e-05      9.50e-08       1.65e-04       1.21e-04
#       43             56         9.5361e-05      2.93e-08       2.39e-04       1.96e-02
#       44             58         9.5304e-05      5.70e-08       4.12e-05       1.32e-03
#       45             59         9.5298e-05      6.45e-09       8.36e-06       9.33e-04
#       46             60         9.5296e-05      2.02e-09       5.76e-06       9.31e-04
#       47             61         9.5296e-05      2.57e-10       1.12e-06       9.28e-04
#       48             62         9.5295e-05      5.47e-11       2.66e-07       9.27e-04
#       49             63         9.5295e-05      1.31e-11       6.56e-08       9.27e-04
# `xtol` termination condition is satisfied.
# Function evaluations 63, initial cost 3.9008e-01, final cost 9.5295e-05, first-order optimality 9.27e-04.
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: divide by zero encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: overflow encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))

# Final vmec iteration = 265
# Quasisymmetry: 0.00019010865750514314
# aspect ratio: 6.0000534668329735
# rotational transform: 0.4193076400157887
# spline_surf.x: array([3.49151814e-05, 4.08858863e-01, 1.56187918e-01, 1.35325832e+00,
#        3.03304348e-02, 3.29309487e-01, 1.48040786e-01, 4.07487352e-01,
#        1.12945811e+00, 3.45660993e+00, 4.70892314e+00, 1.15714960e-01,
#        2.23743388e-01, 1.71422619e-01, 3.38760347e-01, 7.89176573e-01,
#        3.72275569e+00, 4.44881220e+00, 1.68678313e-01, 1.41421904e-01,
#        2.44246896e-01, 2.07282078e-01, 7.96088440e-01, 3.44958011e+00,
#        4.29717483e+00, 1.91544620e-01, 1.15674640e-01, 2.65035209e-01,
#        1.61074550e+00, 7.37716609e-01, 5.62288940e-01, 2.69764752e-01,
#        7.31087514e-01, 3.67273919e-01, 7.79087610e-01, 1.20291494e+00])

# End of 2_Intermediate/stage_one_splines.py
# =================================================
# # =================================================
