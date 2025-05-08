"""
initiation.py
-------------

This script initiates a large set of coil optimizations
using SIMSOPT. The force objective can be specified 
and the optimization will randomly sample the parameter space
within some range. This range should be set in optimization_tools.py

"""
from optimization_tools import *

OUTPUT_DIR = "./output/QA/TVE/"
INPUT_FILE = "../../../tests/test_files/input.LandremanPaul2021_QA"
initial_optimizations(OUTPUT_DIR=OUTPUT_DIR, INPUT_FILE=INPUT_FILE, with_force=True, FORCE_OBJ=TVE)
