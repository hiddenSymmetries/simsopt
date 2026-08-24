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
vmec.indata.nfp = 2
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
max_mode = 3

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
    rel_step=1e-12,
    abs_step=1e-8,  # **options
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


# Best result: 
# Running 2_Intermediate/stage_one_fourier.py
# ==================================================
# Initial Quasisymmetry: 2.2483418916219907e-25
# Initial aspect ratio: 10.000000000000128
# Initial rotational transform: 5.289382185831063e-30
# Beginning optimization
# Beginning optimization with max_mode = 1 , vmec mpol=ntor= 12 . Previous vmec iteration =  0
# ndofs: 8
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         8.0882e+01                                    1.98e+04    
#        1              2         5.6903e+01      2.40e+01       2.87e-01       2.91e+03    
#        2              3         4.3924e+00      5.25e+01       1.98e-01       7.47e+02    
#        3              4         2.1572e+00      2.24e+00       1.54e-01       9.47e+02    
#        4              5         1.4885e-01      2.01e+00       1.28e-01       1.92e+02    
#        5              6         9.6794e-03      1.39e-01       2.88e-02       1.55e+01    
#        6              8         7.0055e-03      2.67e-03       2.66e-02       7.10e+00    
#        7             10         6.1285e-03      8.77e-04       1.37e-02       4.38e+00    
#        8             12         5.8399e-03      2.89e-04       6.80e-03       1.37e+00    
#        9             13         5.5448e-03      2.95e-04       1.40e-02       6.25e+00    
#       10             14         5.2842e-03      2.61e-04       1.34e-02       6.78e+00    
#       11             15         5.1015e-03      1.83e-04       1.28e-02       7.06e+00    
#       12             16         4.9938e-03      1.08e-04       1.22e-02       7.05e+00    
#       13             17         4.9212e-03      7.25e-05       1.04e-02       5.73e+00    
#       14             18         4.8600e-03      6.12e-05       7.36e-04       8.55e-03    
#       15             19         4.8595e-03      5.57e-07       6.70e-04       2.95e-02    
#       16             21         4.8594e-03      4.54e-08       2.21e-04       5.64e-03    
#       17             23         4.8594e-03      7.81e-09       9.85e-05       4.52e-04    
#       18             25         4.8594e-03      1.22e-09       2.99e-05       2.80e-04    
#       19             27         4.8594e-03      6.25e-10       2.24e-05       1.40e-04    
#       20             30         4.8594e-03      6.31e-11       2.76e-06       2.95e+00    
#       21             37         4.8594e-03      0.00e+00       0.00e+00       2.95e+00    
# `xtol` termination condition is satisfied.
# Function evaluations 37, initial cost 8.0882e+01, final cost 4.8594e-03, first-order optimality 2.95e+00.

# Completed optimization with max_mode =1. 
# Final vmec iteration = 60
# Quasisymmetry: 0.009692749940395672
# aspect ratio: 6.000261679049851
# rotational transform: 0.418405838869365
# Beginning optimization with max_mode = 2 , vmec mpol=ntor= 12 . Previous vmec iteration =  60
# ndofs: 24
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         4.8594e-03                                    6.72e+00    
#        1              4         3.6977e-03      1.16e-03       5.50e-03       1.89e+00    
#        2              5         2.4662e-03      1.23e-03       9.37e-03       4.24e+00    
#        3              6         1.4198e-03      1.05e-03       1.94e-02       1.43e+01    
#        4              7         7.6947e-04      6.50e-04       2.43e-02       1.44e+01    
#        5              8         2.5042e-04      5.19e-04       2.40e-02       3.82e+00    
#        6             10         1.2678e-04      1.24e-04       6.60e-03       1.29e+00    
#        7             11         1.2372e-04      3.06e-06       1.00e-02       2.20e+00    
#        8             12         8.7461e-05      3.63e-05       3.04e-03       4.28e-01    
#        9             13         8.1608e-05      5.85e-06       5.11e-03       4.03e-01    
#       10             15         7.9265e-05      2.34e-06       2.49e-03       1.65e-01    
#       11             18         7.9071e-05      1.93e-07       3.24e-04       2.67e-03    
#       12             19         7.8804e-05      2.68e-07       6.48e-04       1.01e-02    
#       13             21         7.8434e-05      3.69e-07       2.97e-04       4.56e-03    
#       14             22         7.8324e-05      1.10e-07       6.31e-04       1.06e-02    
#       15             25         7.8309e-05      1.50e-08       4.44e-05       1.33e-03    
#       16             28         7.8307e-05      1.96e-09       5.72e-06       1.48e-03    
#       17             30         7.8306e-05      9.65e-10       2.85e-06       1.45e-03    
#       18             32         7.8306e-05      4.81e-10       1.43e-06       1.19e-01    
#       19             33         7.8305e-05      9.40e-10       2.81e-06       1.22e-01    
#       20             34         7.8304e-05      5.53e-10       2.63e-06       1.18e-01    
#       21             35         7.8304e-05      5.07e-10       2.65e-06       1.13e-01    
#       22             36         7.8303e-05      4.80e-10       2.66e-06       1.09e-01    
#       23             37         7.8303e-05      4.61e-10       2.67e-06       1.04e-01    
#       24             38         7.8302e-05      4.48e-10       2.68e-06       1.00e-01    
#       25             39         7.8302e-05      4.42e-10       2.69e-06       9.61e-02    
#       26             40         7.8301e-05      4.39e-10       2.70e-06       9.21e-02    
#       27             41         7.8301e-05      4.40e-10       2.70e-06       8.82e-02    
#       28             42         7.8301e-05      4.44e-10       2.71e-06       8.44e-02    
#       29             43         7.8300e-05      4.49e-10       2.71e-06       8.07e-02    
#       30             44         7.8300e-05      4.55e-10       2.72e-06       7.71e-02    
#       31             45         7.8299e-05      4.62e-10       2.72e-06       7.36e-02    
#       32             46         7.8299e-05      4.69e-10       2.72e-06       7.02e-02    
#       33             47         7.8298e-05      4.77e-10       2.73e-06       6.68e-02    
#       34             49         7.8298e-05      1.67e-10       7.00e-07       6.68e-02    
#       35             50         7.8298e-05      1.16e-10       6.77e-07       6.58e-02    
#       36             51         7.8298e-05      1.15e-10       6.79e-07       6.50e-02    
#       37             52         7.8298e-05      1.16e-10       6.81e-07       6.41e-02    
#       38             53         7.8298e-05      1.17e-10       6.81e-07       6.33e-02    
#       39             54         7.8298e-05      1.17e-10       6.82e-07       6.24e-02    
#       40             55         7.8297e-05      1.28e-10       6.85e-07       6.17e-02    
#       41             56         7.8297e-05      1.56e-10       6.88e-07       6.13e-02    
#       42             57         7.8297e-05      1.76e-10       6.89e-07       6.10e-02    
#       43             58         7.8297e-05      2.28e-10       7.11e-07       6.13e-02    
#       44             59         7.8297e-05      1.93e-10       6.84e-07       6.11e-02    
#       45             60         7.8296e-05      2.02e-10       6.86e-07       6.09e-02    
#       46             61         7.8296e-05      2.07e-10       6.86e-07       6.07e-02    
#       47             62         7.8296e-05      2.41e-10       7.08e-07       6.11e-02    
#       48             63         7.8296e-05      2.10e-10       6.85e-07       6.10e-02    
#       49             64         7.8296e-05      2.13e-10       6.86e-07       6.08e-02    
#       50             65         7.8295e-05      2.14e-10       6.87e-07       6.06e-02    
#       51             66         7.8295e-05      2.14e-10       6.87e-07       6.05e-02    
#       52             67         7.8295e-05      2.43e-10       7.08e-07       6.09e-02    
#       53             68         7.8295e-05      2.13e-10       6.86e-07       6.07e-02    
#       54             69         7.8295e-05      2.11e-10       6.88e-07       6.05e-02    
#       55             70         7.8294e-05      2.11e-10       6.88e-07       6.04e-02    
#       56             71         7.8294e-05      2.11e-10       6.88e-07       6.02e-02    
#       57             72         7.8294e-05      2.10e-10       6.88e-07       6.00e-02    
#       58             73         7.8294e-05      2.38e-10       7.09e-07       6.04e-02    
#       59             74         7.8293e-05      2.10e-10       6.88e-07       6.02e-02    
#       60             75         7.8293e-05      2.08e-10       6.89e-07       6.00e-02    
#       61             76         7.8293e-05      2.08e-10       6.89e-07       5.98e-02    
#       62             77         7.8293e-05      2.08e-10       6.89e-07       5.96e-02    
#       63             78         7.8293e-05      2.35e-10       7.09e-07       6.00e-02    
#       64             79         7.8292e-05      2.08e-10       6.89e-07       5.98e-02    
#       65             80         7.8292e-05      2.07e-10       6.89e-07       5.96e-02    
#       66             81         7.8292e-05      2.07e-10       6.90e-07       5.94e-02    
#       67             82         7.8292e-05      2.07e-10       6.90e-07       5.92e-02    
#       68             83         7.8292e-05      2.07e-10       6.90e-07       5.90e-02    
#       69             84         7.8291e-05      2.33e-10       7.09e-07       5.93e-02    
#       70             85         7.8291e-05      2.07e-10       6.89e-07       5.91e-02    
#       71             86         7.8291e-05      2.06e-10       6.90e-07       5.89e-02    
#       72             87         7.8291e-05      2.06e-10       6.90e-07       5.87e-02    
#       73             88         7.8290e-05      2.06e-10       6.90e-07       5.85e-02    
#       74             89         7.8290e-05      2.06e-10       6.90e-07       5.83e-02    
#       75             90         7.8290e-05      2.32e-10       7.09e-07       5.86e-02    
#       76             91         7.8290e-05      2.06e-10       6.90e-07       5.84e-02    
#       77             92         7.8290e-05      2.05e-10       6.90e-07       5.81e-02    
#       78             93         7.8289e-05      2.05e-10       6.90e-07       5.79e-02    
#       79             94         7.8289e-05      2.05e-10       6.90e-07       5.77e-02    
#       80             95         7.8289e-05      2.05e-10       6.90e-07       5.75e-02    
#       81             96         7.8289e-05      2.31e-10       7.09e-07       5.78e-02    
#       82             97         7.8289e-05      2.05e-10       6.90e-07       5.75e-02    
#       83             98         7.8288e-05      2.04e-10       6.91e-07       5.73e-02    
#       84             99         7.8288e-05      2.04e-10       6.91e-07       5.71e-02    
#       85             100        7.8288e-05      2.04e-10       6.91e-07       5.69e-02    
#       86             101        7.8288e-05      2.04e-10       6.91e-07       5.66e-02    
#       87             102        7.8288e-05      2.30e-10       7.09e-07       5.70e-02    
#       88             103        7.8287e-05      2.05e-10       6.90e-07       5.67e-02    
#       89             104        7.8287e-05      2.04e-10       6.91e-07       5.65e-02    
#       90             105        7.8287e-05      2.04e-10       6.91e-07       5.62e-02    
#       91             106        7.8287e-05      2.04e-10       6.91e-07       5.60e-02    
#       92             107        7.8287e-05      2.04e-10       6.91e-07       5.58e-02    
#       93             108        7.8286e-05      2.04e-10       6.91e-07       5.55e-02    
#       94             109        7.8286e-05      2.30e-10       7.10e-07       5.58e-02    
#       95             110        7.8286e-05      2.04e-10       6.90e-07       5.56e-02    
#       96             111        7.8286e-05      2.03e-10       6.91e-07       5.53e-02    
#       97             112        7.8285e-05      2.04e-10       6.91e-07       5.51e-02    
#       98             113        7.8285e-05      2.04e-10       6.91e-07       5.49e-02    
#       99             114        7.8285e-05      2.04e-10       6.91e-07       5.46e-02    
#       100            115        7.8285e-05      2.29e-10       7.10e-07       5.49e-02    
#       101            116        7.8285e-05      2.04e-10       6.90e-07       5.47e-02    
#       102            117        7.8284e-05      2.03e-10       6.91e-07       5.44e-02    
#       103            118        7.8284e-05      2.03e-10       6.91e-07       5.42e-02    
#       104            119        7.8284e-05      2.03e-10       6.91e-07       5.39e-02    
#       105            120        7.8284e-05      2.04e-10       6.91e-07       5.37e-02    
#       106            121        7.8284e-05      2.04e-10       6.91e-07       5.34e-02    
#       107            122        7.8283e-05      2.29e-10       7.10e-07       5.37e-02    
#       108            123        7.8283e-05      2.04e-10       6.90e-07       5.35e-02    
#       109            124        7.8283e-05      2.03e-10       6.91e-07       5.32e-02    
#       110            125        7.8283e-05      2.03e-10       6.91e-07       5.30e-02    
#       111            126        7.8283e-05      2.03e-10       6.91e-07       5.27e-02    
#       112            127        7.8282e-05      2.03e-10       6.91e-07       5.25e-02    
#       113            128        7.8282e-05      2.04e-10       6.91e-07       5.22e-02    
#       114            129        7.8282e-05      2.29e-10       7.10e-07       5.25e-02    
#       115            130        7.8282e-05      2.04e-10       6.90e-07       5.23e-02    
#       116            131        7.8282e-05      2.03e-10       6.91e-07       5.20e-02    
#       117            132        7.8281e-05      2.03e-10       6.91e-07       5.17e-02    
#       118            133        7.8281e-05      2.03e-10       6.91e-07       5.15e-02    
#       119            134        7.8281e-05      2.03e-10       6.91e-07       5.12e-02    
#       120            135        7.8281e-05      2.03e-10       6.91e-07       5.10e-02    
#       121            136        7.8280e-05      2.29e-10       7.10e-07       5.13e-02    
#       122            137        7.8280e-05      2.04e-10       6.90e-07       5.10e-02    
#       123            138        7.8280e-05      2.03e-10       6.91e-07       5.07e-02    
#       124            139        7.8280e-05      2.03e-10       6.91e-07       5.05e-02    
#       125            140        7.8280e-05      2.03e-10       6.91e-07       5.02e-02    
#       126            141        7.8279e-05      2.03e-10       6.91e-07       5.00e-02    
#       127            142        7.8279e-05      2.03e-10       6.91e-07       4.97e-02    
#       128            143        7.8279e-05      2.03e-10       6.91e-07       4.95e-02    
#       129            144        7.8279e-05      2.29e-10       7.10e-07       4.98e-02    
#       130            145        7.8279e-05      2.04e-10       6.90e-07       4.95e-02    
#       131            146        7.8278e-05      2.03e-10       6.91e-07       4.92e-02    
#       132            147        7.8278e-05      2.03e-10       6.91e-07       4.90e-02    
#       133            148        7.8278e-05      2.03e-10       6.91e-07       4.87e-02    
#       134            149        7.8278e-05      2.03e-10       6.91e-07       4.85e-02    
#       135            150        7.8278e-05      2.03e-10       6.91e-07       4.82e-02    
#       136            151        7.8277e-05      2.29e-10       7.10e-07       4.85e-02    
#       137            152        7.8277e-05      2.03e-10       6.90e-07       4.82e-02    
#       138            153        7.8277e-05      2.03e-10       6.91e-07       4.79e-02    
#       139            154        7.8277e-05      2.03e-10       6.91e-07       4.77e-02    
#       140            155        7.8277e-05      2.03e-10       6.91e-07       4.74e-02    
#       141            156        7.8276e-05      2.03e-10       6.91e-07       4.72e-02    
#       142            157        7.8276e-05      2.03e-10       6.91e-07       4.69e-02    
#       143            158        7.8276e-05      2.03e-10       6.91e-07       4.66e-02    
#       144            159        7.8276e-05      2.29e-10       7.10e-07       4.69e-02    
#       145            160        7.8276e-05      2.03e-10       6.90e-07       4.67e-02    
#       146            161        7.8275e-05      2.02e-10       6.91e-07       4.64e-02    
#       147            162        7.8275e-05      2.03e-10       6.91e-07       4.61e-02    
#       148            163        7.8275e-05      2.03e-10       6.91e-07       4.59e-02    
#       149            164        7.8275e-05      2.03e-10       6.91e-07       4.56e-02    
#       150            165        7.8275e-05      2.03e-10       6.91e-07       4.53e-02    
#       151            166        7.8274e-05      2.03e-10       6.91e-07       4.51e-02    
#       152            167        7.8274e-05      2.29e-10       7.10e-07       4.54e-02    
#       153            168        7.8274e-05      2.03e-10       6.90e-07       4.51e-02    
#       154            169        7.8274e-05      2.02e-10       6.91e-07       4.48e-02    
#       155            170        7.8273e-05      2.03e-10       6.91e-07       4.45e-02    
#       156            171        7.8273e-05      2.03e-10       6.91e-07       4.43e-02    
#       157            172        7.8273e-05      2.03e-10       6.91e-07       4.40e-02    
#       158            173        7.8273e-05      2.03e-10       6.91e-07       4.37e-02    
#       159            174        7.8273e-05      2.03e-10       6.91e-07       4.35e-02    
#       160            175        7.8272e-05      2.29e-10       7.10e-07       4.38e-02    
#       161            176        7.8272e-05      2.03e-10       6.90e-07       4.35e-02    
#       162            177        7.8272e-05      2.02e-10       6.91e-07       4.32e-02    
#       163            178        7.8272e-05      2.02e-10       6.91e-07       4.29e-02    
#       164            179        7.8272e-05      2.03e-10       6.91e-07       4.27e-02    
#       165            180        7.8271e-05      2.03e-10       6.91e-07       4.24e-02    
#       166            181        7.8271e-05      2.03e-10       6.91e-07       4.21e-02    
#       167            182        7.8271e-05      2.03e-10       6.91e-07       4.19e-02    
#       168            183        7.8271e-05      2.29e-10       7.10e-07       4.22e-02    
#       169            184        7.8270e-05      4.08e-10       1.38e-06       4.16e-02    
#       170            185        7.8270e-05      8.17e-10       2.76e-06       4.06e-02    
#       171            186        7.8268e-05      1.64e-09       5.53e-06       8.45e-01    
#       172            193        7.8268e-05      2.58e-15       2.33e-09       8.45e-01    
# `xtol` termination condition is satisfied.
# Function evaluations 193, initial cost 4.8594e-03, final cost 7.8268e-05, first-order optimality 8.45e-01.

# Completed optimization with max_mode =2. 
# Final vmec iteration = 774
# Quasisymmetry: 0.00015650664691089537
# aspect ratio: 6.000001941672289
# rotational transform: 0.4199460065138643
# Beginning optimization with max_mode = 3 , vmec mpol=ntor= 12 . Previous vmec iteration =  774
# ndofs: 48
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
#        0              1         7.8268e-05                                    1.03e+00    
#        1              4         3.7964e-05      4.03e-05       3.05e-03       9.00e-01    
#        2              5         2.1532e-05      1.64e-05       6.02e-03       1.41e+00    
#        3              6         1.6147e-05      5.39e-06       6.19e-03       1.24e+00    
#        4              8         7.7368e-06      8.41e-06       1.56e-03       2.17e-01    
#        5              9         6.9972e-06      7.40e-07       2.97e-03       7.16e-01    
#        6             10         5.8826e-06      1.11e-06       2.96e-03       6.65e-01    
#        7             13         5.4366e-06      4.46e-07       1.85e-04       2.86e-03    
#        8             14         5.3628e-06      7.38e-08       4.00e-04       2.72e-02    
#        9             16         5.3394e-06      2.34e-08       1.98e-04       2.21e-03    
#       10             19         5.3365e-06      2.87e-09       2.54e-05       4.84e-04    
#       11             20         5.3309e-06      5.68e-09       5.17e-05       3.29e-02    
#       12             22         5.3281e-06      2.76e-09       2.57e-05       3.51e-02    
#       13             24         5.3268e-06      1.27e-09       1.18e-05       3.50e-02    
#       14             25         5.3255e-06      1.28e-09       1.22e-05       3.47e-02    
#       15             27         5.3252e-06      3.28e-10       3.08e-06       3.50e-02    
#       16             28         5.3249e-06      3.15e-10       2.95e-06       3.48e-02    
#       17             29         5.3246e-06      3.13e-10       2.99e-06       3.48e-02    
#       18             30         5.3243e-06      3.29e-10       3.08e-06       3.49e-02    
#       19             31         5.3239e-06      3.17e-10       3.03e-06       3.48e-02    
#       20             32         5.3236e-06      3.27e-10       3.09e-06       3.49e-02    
#       21             33         5.3233e-06      3.13e-10       2.95e-06       3.49e-02    
#       22             34         5.3230e-06      3.17e-10       3.04e-06       3.47e-02    
#       23             35         5.3227e-06      3.22e-10       3.04e-06       3.49e-02    
#       24             36         5.3223e-06      3.16e-10       2.96e-06       3.48e-02    
#       25             37         5.3220e-06      3.14e-10       3.02e-06       3.47e-02    
#       26             38         5.3217e-06      3.22e-10       3.04e-06       3.48e-02    
#       27             39         5.3214e-06      3.14e-10       2.95e-06       3.48e-02    
#       28             40         5.3211e-06      3.13e-10       3.02e-06       3.46e-02    
#       29             41         5.3208e-06      3.22e-10       3.03e-06       3.48e-02    
#       30             43         5.3207e-06      7.95e-11       7.36e-07       3.48e-02    
#       31             44         5.3207e-06      1.68e-11       1.79e-07       3.47e-02    
#       32             47         5.3207e-06      0.00e+00       0.00e+00       3.47e-02    
# `xtol` termination condition is satisfied.
# Function evaluations 47, initial cost 7.8268e-05, final cost 5.3207e-06, first-order optimality 3.47e-02.

# Completed optimization with max_mode =3. 
# Final vmec iteration = 983
# Quasisymmetry: 1.0640980510983138e-05
# aspect ratio: 6.000000319023268
# rotational transform: 0.4199940495074746

# Final vmec iteration = 984
# Quasisymmetry: 1.0640980510983138e-05
# aspect ratio: 6.000000319023268
# rotational transform: 0.4199940495074746

# End of 2_Intermediate/stage_one_fourier.py