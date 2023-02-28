import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.objectives import ConstrainedProblem
from simsopt.solve.mpi import constrained_mpi_solve
import os

"""
Optimize a VMEC equilibrium for quasi-axis symmetry (M=1, N=0)
throughout the volume.

Solve as a constrained opt problem
min quasi-axis symmetry error
s.t. 
  aspect ratio <= 6
  0.41 <= iota <= 0.44

Run with 
  mpiexec -n 9 constrained_optimization.py
"""

# This problem has 8 degrees of freedom, so we can use 8 + 1 = 9
# concurrent function evaluations for 1-sided finite difference
# gradients.
mpi = MpiPartition()

if mpi.proc0_world:
    print("Running 2_Intermediate/constrained_optimization.py")
    print("=============================================")


vmec_input = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA')
vmec = Vmec(vmec_input, mpi=mpi,verbose=False)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 1
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)") # Major radius

x0 = surf.x

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
# nonlinear constraints
tuples_nlc = [(vmec.aspect,-np.inf,6),(vmec.mean_iota,0.41,0.44)]

# define problem
prob = ConstrainedProblem(qs.total,tuples_nlc=tuples_nlc)

# solve the problem
constrained_mpi_solve(prob,mpi,grad=True, rel_step=1e-5, abs_step=1e-7)
xopt = prob.x

# evaluate the solution
surf.x = xopt
vmec.run()

if mpi.proc0_world:
    print("Quasisymmetry objective after optimization:", qs.total())
    print("Final aspect ratio:", vmec.aspect())
    print("Final rotational transform:", vmec.mean_iota())
    
    print("End of 2_Intermediate/constrained_optimization.py")
    print("============================================")

