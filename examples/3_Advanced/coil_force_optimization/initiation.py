#!/usr/bin/env python

"""
initiation.py
-------------

This script initiates a large set of coil optimizations
using SIMSOPT. The force objective can be specified 
and the optimization will randomly sample the parameter space
within some range. This range should be set in optimization_tools.py. 
The outputs are stored in the ./output/QA/B2_Energy/ directory.

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
from optimization_tools import *
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python initiation.py <function_name>")
    sys.exit(1)

func_name = sys.argv[1]
OUTPUT_DIR = "./output/QA/" + func_name + "/"
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
INPUT_FILE = TEST_DIR / "input.LandremanPaul2021_QA"
N = 10  # number of optimizations to run
MAXITER = 10  # maximum number of iterations for each optimization

if func_name in globals() and callable(globals()[func_name]):
    func = globals()[func_name]
    initial_optimizations(OUTPUT_DIR=OUTPUT_DIR, INPUT_FILE=INPUT_FILE, 
                          with_force=True, FORCE_OBJ=func, N=N, MAXITER=MAXITER)
else:
    print(f"Function '{func_name}' not found in optimization_tools.py.")
    sys.exit(1)
