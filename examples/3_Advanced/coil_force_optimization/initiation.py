#!/usr/bin/env python
"""
initiation.py
-------------

This script initiates a large set of coil optimizations
using SIMSOPT. The force objective can be specified 
and the optimization will randomly sample the parameter space
within some range. This range should be set in optimization_tools.py. 
The outputs are stored in the ./output/QA/B2Energy/ directory. 

Here is a slurm script that can be used to run the script in batch mode:
"
#SBATCH --time=500
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --array=0-20

# Submit an array of 20 jobs. Each core is running 
# a different coil optimization problem with varying weights. 
export OMP_NUM_THREADS=1  # number of threads for OpenMP
export MKL_NUM_THREADS=1  # number of threads for Intel MKL
# IMPORTANT: Activate your conda environment 
# (e.g., 'conda activate simsopt_env') or any other required environment before running this script.
srun python initiation.py B2Energy  # specify the force-related objective function to use
"

This pareto scan script was used to generate the results in the papers:

    Hurwitz, S., Landreman, M., Huslage, P. and Kaptanoglu, A., 2025. 
    Electromagnetic coil optimization for reduced Lorentz forces.
    Nuclear Fusion, 65(5), p.056044.
    https://iopscience.iop.org/article/10.1088/1741-4326/adc9bf/meta

    Kaptanoglu, A.A., Wiedman, A., Halpern, J., Hurwitz, S., Paul, E.J. and Landreman, M., 2025. 
    Reactor-scale stellarators with force and torque minimized dipole coils. 
    Nuclear Fusion, 65(4), p.046029.
    https://iopscience.iop.org/article/10.1088/1741-4326/adc318/meta
"""
from simsopt.util.coil_optimization_helper_functions import initial_vacuum_stage_II_optimizations
from simsopt.field import LpCurveForce, B2Energy, SquaredMeanForce, LpCurveTorque, SquaredMeanTorque
import sys
from pathlib import Path

force_dict = {"LpCurveForce": LpCurveForce, "B2Energy": B2Energy, 
              "SquaredMeanForce": SquaredMeanForce, 
              "LpCurveTorque": LpCurveTorque, 
              "SquaredMeanTorque": SquaredMeanTorque}
print(len(sys.argv), sys.argv)
if len(sys.argv) != 2:
    print("Usage: python initiation.py <function_name>"
          "\n If you want to run the optimizations with a force/torque/energy objective, "
          " otherwise, the optimization will be run with no such objective.")
    func = None
    func_name = ""
else: 
    func_name = sys.argv[1]
    if func_name in force_dict.keys():
        func = force_dict[func_name]
    else:
        print(f"The command line argument, function '{func_name}', is not a valid force objective.")
        sys.exit(1)

OUTPUT_DIR = "./output/QA/" + func_name + "/"
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
INPUT_FILE = TEST_DIR / "input.LandremanPaul2021_QA"
N = 5  # number of optimizations to run
MAXITER = 5  # maximum number of iterations for each optimization
initial_vacuum_stage_II_optimizations(OUTPUT_DIR=OUTPUT_DIR, INPUT_FILE=INPUT_FILE, 
                                      FORCE_OBJ=func, N=N, MAXITER=MAXITER)

